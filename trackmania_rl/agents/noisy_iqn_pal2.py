import math
import random

import numpy as np
import torch

from .. import nn_utilities, noisy_linear
from ..experience_replay.experience_replay_interface import ExperienceReplayInterface


class Agent(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim,
        float_hidden_dim,
        conv_head_output_dim,
        dense_hidden_dimension,
        iqn_embedding_dimension,
        n_actions,
        float_inputs_mean,
        float_inputs_std,
    ):
        super().__init__()
        self.img_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(16, 16), stride=8),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(8, 8), stride=4),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Flatten(),
        )
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
        )

        dense_input_dimension = conv_head_output_dim + float_hidden_dim

        self.A_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, n_actions),
        )
        self.V_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, 1),
        )

        self.iqn_fc = torch.nn.Linear(
            iqn_embedding_dimension, dense_input_dimension
        )  # There is no word in the paper on how to init this layer?
        self.lrelu = torch.nn.LeakyReLU()
        self.initialize_weights()

        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.n_actions = n_actions

        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        for m in self.img_head:
            if isinstance(m, torch.nn.Conv2d):
                nn_utilities.init_kaiming(m)
        for m in self.float_feature_extractor:
            if isinstance(m, torch.nn.Linear):
                nn_utilities.init_kaiming(m)
        # This was uninitialized in Agade's code
        nn_utilities.init_kaiming(self.iqn_fc)
        # A_head and V_head are NoisyLinear, already initialized

    def forward(self, img, float_inputs, num_quantiles, tau):
        batch_size = img.shape[0]
        img_outputs = self.img_head((img.to(torch.float16) - 128) / 128)  # PERF
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        # (batch_size, dense_input_dimension) OK
        concat = torch.cat((img_outputs, float_outputs), 1)
        if tau is None:
            tau = torch.rand(
                size=(batch_size * num_quantiles, 1), device="cuda", dtype=torch.float32
            )  # (batch_size * num_quantiles, 1) (random numbers)
        quantile_net = tau.expand(
            [-1, self.iqn_embedding_dimension]
        )  # (batch_size*num_quantiles, iqn_embedding_dimension) (still random numbers)
        quantile_net = torch.cos(
            torch.arange(1, self.iqn_embedding_dimension + 1, 1, device="cuda") * math.pi * quantile_net
        )  # (batch_size*num_quantiles, iqn_embedding_dimension)
        # (8 or 32 initial random numbers, expanded with cos to iqn_embedding_dimension)
        # (batch_size*num_quantiles, dense_input_dimension)
        quantile_net = self.iqn_fc(quantile_net)
        # (batch_size*num_quantiles, dense_input_dimension)
        quantile_net = self.lrelu(quantile_net)
        # (batch_size*num_quantiles, dense_input_dimension)
        concat = concat.repeat(num_quantiles, 1)
        # (batch_size*num_quantiles, dense_input_dimension)
        concat = concat * quantile_net

        A = self.A_head(concat)  # (batch_size*num_quantiles, n_actions)
        V = self.V_head(concat)  # (batch_size*num_quantiles, 1) #need to check this

        Q = V + A - A.mean(dim=-1, keepdim=True)

        return Q, tau

    def reset_noise(self):
        pass
        # self.A_head[0].reset_noise()
        # self.A_head[2].reset_noise()
        # self.V_head[0].reset_noise()
        # self.V_head[2].reset_noise()


# ==========================================================================================================================


class Trainer:
    __slots__ = (
        "model",
        "model2",
        "optimizer",
        "scaler",
        "batch_size",
        "iqn_k",
        "iqn_n",
        "iqn_kappa",
        "epsilon",
        "gamma",
        "AL_alpha",
        "q_values_randomization_amplitude",
    )

    def __init__(
        self,
        model: Agent,
        model2: Agent,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.grad_scaler.GradScaler,
        batch_size: int,
        iqn_k: int,
        iqn_n: int,
        iqn_kappa: float,
        epsilon: float,
        gamma: float,
        AL_alpha: float,
        q_values_randomization_amplitude: float,
    ):
        self.model = model
        self.model2 = model2
        self.optimizer = optimizer
        self.scaler = scaler
        self.batch_size = batch_size
        self.iqn_k = iqn_k
        self.iqn_n = iqn_n
        self.iqn_kappa = iqn_kappa
        self.epsilon = epsilon
        self.gamma = gamma
        self.AL_alpha = AL_alpha
        self.q_values_randomization_amplitude = q_values_randomization_amplitude

    def train_on_batch(self, buffer: ExperienceReplayInterface):
        batch, idxs, is_weights = buffer.sample(self.batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            state_img_tensor = torch.as_tensor(np.array([memory.state_img for memory in batch])).to(
                memory_format=torch.channels_last, non_blocking=True, device="cuda"
            )
            state_float_tensor = torch.as_tensor(np.array([memory.state_float for memory in batch])).to(non_blocking=True, device="cuda")
            actions = torch.as_tensor(np.array([memory.action for memory in batch]), dtype=torch.int64).to(non_blocking=True, device="cuda")
            rewards = torch.as_tensor(np.array([memory.reward for memory in batch])).to(non_blocking=True, device="cuda")
            gammas_pow_nsteps = torch.as_tensor(np.array([memory.gamma_pow_nsteps for memory in batch])).to(
                non_blocking=True, device="cuda"
            )
            done = torch.as_tensor(np.array([memory.done for memory in batch])).to(non_blocking=True, device="cuda")
            next_state_img_tensor = torch.as_tensor(np.array([memory.next_state_img for memory in batch])).to(
                memory_format=torch.channels_last, non_blocking=True, device="cuda"
            )
            next_state_float_tensor = torch.as_tensor(np.array([memory.next_state_float for memory in batch])).to(
                non_blocking=True, device="cuda"
            )
            is_weights = torch.as_tensor(is_weights).to(non_blocking=True, device="cuda")

            with torch.no_grad():
                rewards = rewards.reshape(-1, 1).repeat(
                    [self.iqn_n, 1]
                )  # (batch_size*iqn_n, 1)     a,b,c,d devient a,b,c,d,a,b,c,d,a,b,c,d,...
                # (batch_size*iqn_n, 1)
                gammas_pow_nsteps = gammas_pow_nsteps.reshape(-1, 1).repeat([self.iqn_n, 1])
                done = done.reshape(-1, 1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                actions = actions[:, None]  # (batch_size, 1)
                actions_n = actions.repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)

                #
                #   Use model to choose an action for next state.
                #   This action is chosen AFTER reduction to the mean, and repeated to all quantiles
                #
                self.model.reset_noise()
                a__tpo__model__reduced_repeated = (
                    self.model(
                        next_state_img_tensor,
                        next_state_float_tensor,
                        self.iqn_n,
                        tau=None,
                    )[0]
                    .reshape([self.iqn_n, self.batch_size, self.model.n_actions])
                    .mean(dim=0)
                    .argmax(dim=1, keepdim=True)
                    .repeat([self.iqn_n, 1])
                )  # (iqn_n * batch_size, 1)

                #
                #   Use model2 to evaluate the action chosen, per quantile.
                #
                self.model2.reset_noise()
                q__stpo__model2__quantiles_tau2, tau2 = self.model2(
                    next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=None
                )  # (batch_size*iqn_n,n_actions)

                #
                #   Build IQN target on tau2 quantiles
                #
                outputs_target_tau2 = torch.where(
                    done,
                    rewards,
                    rewards + gammas_pow_nsteps * q__stpo__model2__quantiles_tau2.gather(1, a__tpo__model__reduced_repeated),
                )  # (batch_size*iqn_n, 1)

                # # =============== BEG PAL ==============
                # # V(x') dans PAL devient une distribution en IQN, on veux les quantiles de cette distribution. Il faut choisir une action, la meilleure action, et prendre les quantiles de cette action: notre V
                # # Parceque si tu max par quantile, tu melange des quantiles de distributions differentes
                # #
                # #   PAL Term
                # #
                # pal_term_tau2 = q__stpo__model2__quantiles_tau2.gather(
                #     1,
                #     q__stpo__model2__quantiles_tau2.reshape(
                #         [self.iqn_n, self.batch_size, self.model.n_actions]
                #     )
                #     .mean(dim=0)
                #     .argmax(dim=1, keepdim=True)
                #     .repeat([self.iqn_n, 1]),
                # ) - q__stpo__model2__quantiles_tau2.gather(1, actions_n)
                # # (batch_size*iqn_n, 1)
                #
                # #
                # #   AL Term
                # #
                # q__st__model2__quantiles_tau2, tau2 = self.model2(
                #     state_img_tensor, state_float_tensor, self.iqn_n, tau=tau2
                # )  # (batch_size*iqn_n,n_actions)
                # al_term_tau2 = q__st__model2__quantiles_tau2.gather(
                #     1,
                #     q__st__model2__quantiles_tau2.reshape(
                #         [self.iqn_n, self.batch_size, self.model.n_actions]
                #     )
                #     .mean(dim=0)
                #     .argmax(dim=1, keepdim=True)
                #     .repeat([self.iqn_n, 1]),
                # ) - q__st__model2__quantiles_tau2.gather(1, actions_n)
                # # (batch_size*iqn_n, 1)
                #
                # outputs_target_tau2 -= self.AL_alpha * torch.minimum(
                #     al_term_tau2, pal_term_tau2
                # )
                # # =============== END PAL ==============

                #
                #   This is our target
                #
                outputs_target_tau2 = outputs_target_tau2.reshape([self.iqn_n, self.batch_size, 1]).transpose(
                    0, 1
                )  # (batch_size, iqn_n, 1)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            self.model.reset_noise()
            q__st__model__quantiles_tau3, tau3 = self.model(
                state_img_tensor, state_float_tensor, self.iqn_n, tau=None
            )  # (batch_size*iqn_n,n_actions)

            mean_q_value = torch.mean(q__st__model__quantiles_tau3, dim=0).detach().cpu()

            outputs_tau3 = (
                q__st__model__quantiles_tau3.gather(1, actions_n).reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)
            )  # (batch_size, iqn_n, 1)

            TD_Error = outputs_target_tau2[:, :, None, :] - outputs_tau3[:, None, :, :]
            # (batch_size, iqn_n, iqn_n, 1)    WTF ????????
            # Huber loss, my alternative
            loss = torch.where(
                torch.abs(TD_Error) <= self.iqn_kappa,
                0.5 * TD_Error**2,
                self.iqn_kappa * (torch.abs(TD_Error) - 0.5 * self.iqn_kappa),
            )
            tau3 = tau3.reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)  # (batch_size, iqn_n, 1)
            tau3 = tau3[:, None, :, :].expand([-1, self.iqn_n, -1, -1])  # (batch_size, iqn_n, iqn_n, 1)
            loss = (
                (torch.where(TD_Error < 0, 1 - tau3, tau3) * loss / self.iqn_kappa).sum(dim=2).mean(dim=1)[:, 0]
            )  # pinball loss # (batch_size, )

            total_loss = torch.sum(is_weights * loss)  # total_loss.shape=torch.Size([])

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        buffer.update(idxs, loss.detach().cpu().numpy().astype(np.float32))
        return mean_q_value, total_loss.detach().cpu()

    def get_exploration_action(self, img_inputs, float_inputs):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                state_img_tensor = torch.as_tensor(np.expand_dims(img_inputs, axis=0)).to(
                    "cuda", memory_format=torch.channels_last, non_blocking=True
                )
                state_float_tensor = torch.as_tensor(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
                if self.epsilon > 0:
                    # We are not evaluating
                    self.model.reset_noise()
                q_values = (
                    self.model(state_img_tensor, state_float_tensor, self.iqn_k, tau=None)[0].cpu().numpy().astype(np.float32).mean(axis=0)
                )

        if random.random() < self.epsilon:
            # Choose a random action
            return (
                random.randrange(0, self.model.n_actions),
                False,
                np.max(q_values),
                q_values,
            )
        else:
            return (
                np.argmax(q_values + (self.epsilon > 0) * self.q_values_randomization_amplitude * np.random.randn(*q_values.shape)),
                True,
                np.max(q_values),
                q_values,
            )
