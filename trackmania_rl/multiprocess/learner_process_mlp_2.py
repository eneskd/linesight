"""
This file implements the learner process for MLP agents.
It handles training the MLP agent using experiences collected by collector processes.
"""

import importlib
import time
import joblib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from config_files import config_copy
from config_files import mlp_config_copy

from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents.mlp import make_untrained_mlp_agent


def dqn_loss(q_values, actions, targets):
    """
    Standard DQN loss: MSE between Q(s,a) and target.
    q_values: (batch, n_actions)
    actions: (batch,) int64
    targets: (batch,) float32
    """
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    return torch.nn.functional.mse_loss(q_selected, targets)


class MLPInferer:
    """
    Inference helper for MLP agents.
    """

    def __init__(self, network, n_actions):
        self.network = network
        self.n_actions = n_actions

    def infer_network(self, img_inputs_uint8, float_inputs):
        """
        Perform inference through the MLP network.

        Args:
            img_inputs_uint8: Image input as uint8 numpy array (1, H, W)
            float_inputs: Float features as numpy array (float_input_dim,)

        Returns:
            q_values: Q-values for each action
        """
        with torch.no_grad():
            # Normalize image to [-1, 1]
            state_img_tensor = (
                                       torch.from_numpy(img_inputs_uint8)
                                       .unsqueeze(0)
                                       .to("cuda", non_blocking=True, dtype=torch.float32)
                                       - 128
                               ) / 128

            state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            q_values = self.network(state_img_tensor, state_float_tensor).cpu().numpy()
            return q_values


def learner_process_fn(
        rollout_queues,
        uncompiled_shared_network,
        shared_network_lock,
        shared_steps: mp.Value,
        base_dir: Path,
        save_dir: Path,
        tensorboard_base_dir: Path,
):
    """
    Main learner process function for MLP agents.

    Args:
        rollout_queues: Queues containing rollout data from collector processes
        uncompiled_shared_network: Shared network model to be updated
        shared_network_lock: Lock for synchronizing network updates
        shared_steps: Shared counter for tracking training progress
        base_dir: Base directory of the project
        save_dir: Directory to save model checkpoints
        tensorboard_base_dir: Directory for tensorboard logs
    """
    # Create online and target networks
    online_network, uncompiled_online_network = make_untrained_mlp_agent(
        jit=config_copy.use_jit,
        is_inference=False,
    )
    target_network, _ = make_untrained_mlp_agent(
        jit=config_copy.use_jit,
        is_inference=False,
    )

    # Create optimizer
    optimizer = torch.optim.RAdam(
        online_network.parameters(),
        lr=utilities.from_exponential_schedule(config_copy.lr_schedule, 0),
        eps=config_copy.adam_epsilon,
        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
    )

    # Create gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler("cuda")

    # Create tensorboard writer
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + "_mlp")))

    # Create replay buffer
    buffer_size = utilities.from_exponential_schedule(config_copy.memory_size_schedule, 0)[0]
    buffer = buffer_management.create_buffer(
        buffer_size=buffer_size,
        prio_alpha=config_copy.prio_alpha,
        prio_epsilon=config_copy.prio_epsilon,
        prio_beta=config_copy.prio_beta,
    )

    # Create inferer for evaluation
    inferer = MLPInferer(online_network, n_actions=len(config_copy.inputs))

    # =========================
    # Load existing weights/stats if available
    # =========================
    accumulated_stats = defaultdict(int)
    try:
        online_network.load_state_dict(torch.load(save_dir / "weights1.torch"))
        target_network.load_state_dict(torch.load(save_dir / "weights2.torch"))
        print(" =====================     Learner MLP weights loaded !     ============================")
    except Exception as e:
        print(" Learner MLP could not load weights:", e)

    try:
        optimizer.load_state_dict(torch.load(save_dir / "optimizer1.torch"))
        print(" =====================     Learner MLP optimizer loaded !     ============================")
    except Exception as e:
        print(" Learner MLP could not load optimizer state:", e)

    try:
        scaler.load_state_dict(torch.load(save_dir / "scaler.torch"))
        print(" =====================     Learner MLP scaler loaded !     ============================")
    except Exception as e:
        print(" Learner MLP could not load scaler state:", e)

    try:
        accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
        shared_steps.value = accumulated_stats.get("cumul_number_frames_played", 0)
        print(" =====================      Learner MLP stats loaded !      ============================")
    except Exception as e:
        print(" Learner MLP could not load stats:", e)

    # Update shared network with loaded weights
    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    # Initialize stats tracking
    if "rolling_mean_ms" not in accumulated_stats:
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats.get(
        "cumul_number_single_memories_used", 0)
    transitions_learned_last_save = accumulated_stats.get("cumul_number_single_memories_used", 0)
    neural_net_reset_counter = 0
    single_reset_flag = config_copy.single_reset_flag

    # Stats tracking
    stats = defaultdict(list)
    last_target_update = 0
    last_save = 0
    last_shared_update = 0
    step = shared_steps.value if hasattr(shared_steps, "value") else 0
    wait_time_total = 0  # Total time spent waiting for rollouts
    cumul_number_batches_done = 0  # Track number of batches processed

    print("Learner process started.")
    print(f"Replay buffer capacity: {buffer.capacity}")
    print(f"Batch size: {config_copy.batch_size}")
    print(f"Initial shared step value: {shared_steps.value}")

    # Main training loop
    while True:
        # Wait for rollout data
        wait_start = time.perf_counter()
        while True:
            rollout_found = False
            for idx, queue in enumerate(rollout_queues):
                if not queue.empty():
                    (
                        rollout_results,
                        end_race_stats,
                        fill_buffer,
                        is_explo,
                        map_name,
                        map_status,
                        rollout_duration,
                        loop_number,
                    ) = queue.get()
                    rollout_found = True
                    break
            if rollout_found:
                break
            else:
                time.sleep(0.1)  # Sleep a little to avoid busy-waiting

        wait_time = time.perf_counter() - wait_start
        wait_time_total += wait_time
        if wait_time > 1.0:
            print(
                f"  [Timer] Waited {wait_time:.2f} seconds for workers to provide a rollout (total waited {wait_time_total:.1f}s).")

        # Process rollout data
        n_frames = len(rollout_results["frames"])

        # Compute rewards
        rewards = np.zeros(n_frames)
        for i in range(1, n_frames):
            prev_state = rollout_results["state_float"][i - 1]
            curr_state = rollout_results["state_float"][i]
            action = rollout_results["actions"][i]
            meters_advanced = rollout_results["meters_advanced_along_centerline"][i] - \
                              rollout_results["meters_advanced_along_centerline"][i - 1]
            ms_elapsed = config_copy.ms_per_action if (i < n_frames - 1 or ("race_time" not in rollout_results)) else \
            rollout_results["race_time"] - (n_frames - 2) * config_copy.ms_per_action

            # Calculate reward
            rewards[i] = (
                    config_copy.constant_reward_per_ms * ms_elapsed
                    + config_copy.reward_per_m_advanced_along_centerline * meters_advanced
            )

        # Add transitions to buffer
        if fill_buffer:
            # Prepare transitions for buffer
            transitions = []
            for i in range(n_frames - 1):
                transition = {
                    "state_img": rollout_results["frames"][i],
                    "state_float": rollout_results["state_float"][i],
                    "action": rollout_results["actions"][i],
                    "reward": rewards[i],
                    "next_state_img": rollout_results["frames"][i + 1],
                    "next_state_float": rollout_results["state_float"][i + 1],
                    "gamma": utilities.from_linear_schedule(config_copy.gamma_schedule,
                                                            accumulated_stats["cumul_number_frames_played"]),
                }
                transitions.append(transition)

            # Add transitions to buffer
            buffer.add_batch(transitions)
            print(
                f"  Added {len(transitions)} transitions to replay buffer. Buffer size: {len(buffer)} / {buffer.capacity}")

        # Only train if we have enough data
        if len(buffer) < config_copy.batch_size:
            print(f"  Not enough data in buffer. Waiting for buffer size >= batch size ({config_copy.batch_size})")
            continue

        # Sample batch and train
        batch, batch_info = buffer.sample(config_copy.batch_size, return_info=True)

        # Unpack batch
        state_img_tensor, state_float_tensor, actions, rewards, next_state_img_tensor, next_state_float_tensor, gammas = batch

        # Forward pass
        with torch.cuda.amp.autocast():
            # Get Q-values for current states
            q_values = online_network(state_img_tensor, state_float_tensor)

            # Get target Q-values
            with torch.no_grad():
                # Double DQN: use online network to select actions, target network to evaluate them
                if getattr(config_copy, "use_ddqn", False):
                    next_q_values = online_network(next_state_img_tensor, next_state_float_tensor)
                    next_actions = next_q_values.argmax(dim=1)
                    next_q_target = target_network(next_state_img_tensor, next_state_float_tensor)
                    next_q_target = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    # Standard DQN
                    next_q_target = target_network(next_state_img_tensor, next_state_float_tensor).max(dim=1)[0]

                # Compute targets
                targets = rewards + gammas * next_q_target

            # Compute loss
            loss = dqn_loss(q_values, actions, targets)

            # Apply importance sampling weights for prioritized replay
            if config_copy.prio_alpha > 0:
                IS_weights = torch.from_numpy(batch_info["_weight"]).to("cuda", non_blocking=True)
                loss = torch.mean(IS_weights * loss)

        # Print training progress and stats
        print(f"  Training step {step}")
        print(f"    Loss: {loss.item():.6f}")
        print(
            f"    Q value mean/max/min: {q_values.mean().item():.3f} / {q_values.max().item():.3f} / {q_values.min().item():.3f}")
        print(f"    Target mean: {targets.mean().item():.3f}")
        print(f"    Buffer size: {len(buffer)}")

        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(online_network.parameters(), config_copy.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Update step counter
        step += 1
        cumul_number_batches_done += 1
        with shared_steps.get_lock():
            shared_steps.value += 1
        print(f"    Shared steps (global): {shared_steps.value}")

        # Update target network periodically
        if step - last_target_update >= config_copy.number_memories_trained_on_between_target_network_updates // config_copy.batch_size:
            # Hard update
            target_network.load_state_dict(online_network.state_dict())
            last_target_update = step
            print(f"    Target network updated at step {step}")

        # Update shared network periodically
        if cumul_number_batches_done % config_copy.send_shared_network_every_n_batches == 0:
            with shared_network_lock:
                uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())
            last_shared_update = step
            print(f"    Shared uncompiled network updated at step {step}")

        # Save model and stats periodically
        if step - last_save >= 1000:  # Save every 1000 steps
            # Save weights
            utilities.save_checkpoint(
                save_dir,
                online_network,
                target_network,
                optimizer,
                scaler,
            )

            # Also save a versioned checkpoint for reference
            save_path = save_dir / f"mlp_model_step_{step}.pt"
            torch.save({
                'online_network': online_network.state_dict(),
                'target_network': target_network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'step': step,
            }, save_path)

            # Save stats
            accumulated_stats["cumul_number_frames_played"] = shared_steps.value
            accumulated_stats["cumul_number_batches_done"] = cumul_number_batches_done
            accumulated_stats["last_step"] = step
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")

            last_save = step
            print(f"    Model checkpoint and stats saved at step {step}")

        # Log stats to tensorboard
        tensorboard_writer.add_scalar('train/loss', loss.item(), step)
        tensorboard_writer.add_scalar('train/q_values_mean', q_values.mean().item(), step)
        tensorboard_writer.add_scalar('train/q_values_max', q_values.max().item(), step)
        tensorboard_writer.add_scalar('train/q_values_min', q_values.min().item(), step)
        tensorboard_writer.add_scalar('train/buffer_size', len(buffer), step)
        tensorboard_writer.add_scalar('train/target_mean', targets.mean().item(), step)

        # Log network parameters and gradients
        total_params = 0
        for name, param in online_network.named_parameters():
            num_param = param.numel()
            total_params += num_param
            tensorboard_writer.add_scalar(f"agent_params/{name.replace('.', '_')}_mean", param.data.mean().item(), step)
            tensorboard_writer.add_scalar(f"agent_params/{name.replace('.', '_')}_std", param.data.std().item(), step)
            if param.grad is not None:
                tensorboard_writer.add_scalar(f"agent_grads/{name.replace('.', '_')}_grad_mean",
                                              param.grad.mean().item(), step)
                tensorboard_writer.add_scalar(f"agent_grads/{name.replace('.', '_')}_grad_std", param.grad.std().item(),
                                              step)
        tensorboard_writer.add_scalar("agent_params/total_param_count", total_params, step)

        # Log optimizer state
        for i, group in enumerate(optimizer.param_groups):
            tensorboard_writer.add_scalar(f"optimizer/group_{i}_lr", group['lr'], step)

        # Log end race stats if available
        if end_race_stats:
            for k, v in end_race_stats.items():
                if isinstance(v, (int, float)):
                    tensorboard_writer.add_scalar(f'race/{k}', v, step)
                    print(f"    End race stat: {k} = {v}")

        # Update buffer size if needed
        current_buffer_size, current_buffer_test_size = utilities.from_exponential_schedule(
            config_copy.memory_size_schedule, accumulated_stats["cumul_number_frames_played"]
        )
        if current_buffer_size != buffer.capacity:
            print(f"  Updating buffer size from {buffer.capacity} to {current_buffer_size}")
            buffer.resize(current_buffer_size)

        # Update learning rate
        current_lr = utilities.from_exponential_schedule(config_copy.lr_schedule,
                                                         accumulated_stats["cumul_number_frames_played"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
