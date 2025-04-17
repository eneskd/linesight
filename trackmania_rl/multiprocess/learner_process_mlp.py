import copy
import importlib
import math
import random
import sys
import time
import typing
from collections import defaultdict
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import PrioritizedSampler

from config_files import config_copy
from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents.mlp import make_untrained_mlp_agent
from trackmania_rl.buffer_utilities import make_buffers, resize_buffers




def dqn_loss(q_values, actions, targets):
    """
    Standard DQN loss: MSE between Q(s,a) and target.
    q_values: (batch_size, n_actions)
    actions: (batch_size,) int64
    targets: (batch_size,) float32
    """

    q_selected = q_values.gather(1, actions.unsqueeze(1).long()).squeeze(1)
    return torch.nn.functional.mse_loss(q_selected, targets)

def learner_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
):
    layout_version = "lay_mono"
    SummaryWriter(log_dir=str(tensorboard_base_dir / layout_version)).add_custom_scalars(
        {
            layout_version: {
                "loss": ["Multiline", ["loss$", "loss_test$"]],
                "avg_Q": ["Multiline", ["avg_Q"]],
                "grad_norm_history": ["Multiline", ["grad_norm_history_d9", "grad_norm_history_d98"]],
                "priorities": ["Multiline", ["priorities"]],
            },
        }
    )

    # ========================================================
    # Create new stuff
    # ========================================================

    online_network, uncompiled_online_network = make_untrained_mlp_agent(config_copy.use_jit, is_inference=False)
    target_network, _ = make_untrained_mlp_agent(config_copy.use_jit, is_inference=False)

    print(online_network)
    utilities.count_parameters(online_network)

    accumulated_stats: defaultdict[str, typing.Any] = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    previous_alltime_min = None
    time_last_save = time.perf_counter()
    queue_check_order = list(range(len(rollout_queues)))
    rollout_queue_readers = [q._reader for q in rollout_queues]
    time_waited_for_workers_since_last_tensorboard_write = 0
    time_training_since_last_tensorboard_write = 0
    time_testing_since_last_tensorboard_write = 0

    # ========================================================
    # Load existing stuff
    # ========================================================
    try:
        online_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
        target_network.load_state_dict(torch.load(f=save_dir / "weights2.torch", weights_only=False))
        print(" =====================     Learner weights loaded !     ============================")
    except:
        print(" Learner could not load weights")

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    try:
        accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
        shared_steps.value = accumulated_stats["cumul_number_frames_played"]
        print(" =====================      Learner stats loaded !      ============================")
    except:
        print(" Learner could not load stats")

    if "rolling_mean_ms" not in accumulated_stats.keys():
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats["cumul_number_single_memories_used"]
    transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]
    neural_net_reset_counter = 0
    single_reset_flag = config_copy.single_reset_flag

    optimizer1 = torch.optim.RAdam(
        online_network.parameters(),
        lr=utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"]),
        eps=config_copy.adam_epsilon,
        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
    )

    scaler = torch.amp.GradScaler("cuda")
    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        config_copy.memory_size_schedule, accumulated_stats["cumul_number_frames_played"]
    )
    buffer, buffer_test = make_buffers(memory_size)
    offset_cumul_number_single_memories_used = memory_size_start_learn * config_copy.number_times_single_memory_is_used_before_discard

    try:
        optimizer1.load_state_dict(torch.load(f=save_dir / "optimizer1.torch", weights_only=False))
        scaler.load_state_dict(torch.load(f=save_dir / "scaler.torch", weights_only=False))
        print(" =========================     Optimizer loaded !     ================================")
    except:
        print(" Could not load optimizer")

    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule,
        accumulated_stats["cumul_number_frames_played"],
    )
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name  + '_' + config_copy.agent_type + tensorboard_suffix)))

    loss_history = []
    loss_test_history = []
    train_on_batch_duration_history = []
    grad_norm_history = []
    layer_grad_norm_history = defaultdict(list)

    # ========================================================
    # Training loop
    # ========================================================
    while True:
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time
        if time_waited > 1:
            print(f"Warning: learner waited {time_waited:.2f} seconds for workers to provide memories")
        time_waited_for_workers_since_last_tensorboard_write += time_waited
        for idx in queue_check_order:
            if not rollout_queues[idx].empty():
                (
                    rollout_results,
                    end_race_stats,
                    fill_buffer,
                    is_explo,
                    map_name,
                    map_status,
                    rollout_duration,
                    loop_number,
                ) = rollout_queues[idx].get()
                queue_check_order.append(queue_check_order.pop(queue_check_order.index(idx)))
                break

        importlib.reload(config_copy)

        new_tensorboard_suffix = utilities.from_staircase_schedule(
            config_copy.tensorboard_suffix_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_tensorboard_suffix != tensorboard_suffix:
            tensorboard_suffix = new_tensorboard_suffix
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name  + '_' + config_copy.agent_type + tensorboard_suffix)))

        (
            new_memory_size,
            new_memory_size_start_learn,
        ) = utilities.from_staircase_schedule(
            config_copy.memory_size_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_memory_size != memory_size:
            buffer, buffer_test = resize_buffers(buffer, buffer_test, new_memory_size)
            offset_cumul_number_single_memories_used += (
                new_memory_size_start_learn - memory_size_start_learn
            ) * config_copy.number_times_single_memory_is_used_before_discard
            memory_size_start_learn = new_memory_size_start_learn
            memory_size = new_memory_size

        learning_rate = utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"])
        weight_decay = config_copy.weight_decay_lr_ratio * learning_rate
        gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, accumulated_stats["cumul_number_frames_played"])

        for param_group in optimizer1.param_groups:
            param_group["lr"] = learning_rate
            param_group["epsilon"] = config_copy.adam_epsilon
            param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)

        if isinstance(buffer._sampler, PrioritizedSampler):
            buffer._sampler._alpha = config_copy.prio_alpha
            buffer._sampler._beta = config_copy.prio_beta
            buffer._sampler._eps = config_copy.prio_epsilon

        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

        # ===============================================
        #   FILL BUFFER WITH (S, A, R, S') transitions
        # ===============================================
        if fill_buffer:
            current_step = accumulated_stats["cumul_number_frames_played"]

            engineered_speedslide_reward = utilities.from_staircase_schedule(config_copy.engineered_speedslide_reward_schedule, current_step)
            engineered_neoslide_reward = utilities.from_staircase_schedule(config_copy.engineered_neoslide_reward_schedule, current_step)
            engineered_kamikaze_reward = utilities.from_staircase_schedule(config_copy.engineered_kamikaze_reward_schedule, current_step)
            engineered_close_to_vcp_reward = utilities.from_staircase_schedule(config_copy.engineered_close_to_vcp_reward_schedule, current_step)

            (
                buffer,
                buffer_test,
                number_memories_added_train,
                number_memories_added_test,
            ) = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
                buffer,
                buffer_test,
                rollout_results,
                config_copy.n_steps,
                gamma,
                config_copy.discard_non_greedy_actions_in_nsteps,
                engineered_speedslide_reward,
                engineered_neoslide_reward,
                engineered_kamikaze_reward,
                engineered_close_to_vcp_reward,
            )

        # ===============================================
        #   LEARN ON BATCH
        # ===============================================

        if not online_network.training:
            online_network.train()

        while (
            len(buffer) >= memory_size_start_learn
            and accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used
            <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
        ):
            train_start_time = time.perf_counter()
            optimizer1.zero_grad(set_to_none=True)
            batch, batch_info = buffer.sample(config_copy.batch_size, return_info=True)
            (
                state_img_tensor,         # (batch_size, 1, H, W)
                state_float_tensor,       # (batch_size, float_input_dim)
                actions,
                rewards,
                next_state_img_tensor,    # (batch_size, 1, H, W)
                next_state_float_tensor,  # (batch_size, float_input_dim)
                gammas_terminal,
            ) = batch

            # Ensure image tensors are (batch, 1, H, W)
            if state_img_tensor.ndim == 3:
                state_img_tensor = state_img_tensor.unsqueeze(1)
            if next_state_img_tensor.ndim == 3:
                next_state_img_tensor = next_state_img_tensor.unsqueeze(1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Compute Q(s,a)
                q_values = online_network(state_img_tensor, state_float_tensor)
                # Compute Q(s',a') for target
                with torch.no_grad():
                    q_next = target_network(next_state_img_tensor, next_state_float_tensor)
                    max_next_q = q_next.max(dim=1)[0]
                    targets = rewards + gammas_terminal * max_next_q

                loss = dqn_loss(q_values, actions, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer1)
            grad_norm = torch.nn.utils.clip_grad_norm_(online_network.parameters(), config_copy.clip_grad_norm).detach().cpu().item()
            torch.nn.utils.clip_grad_value_(online_network.parameters(), config_copy.clip_grad_value)
            scaler.step(optimizer1)
            scaler.update()

            train_on_batch_duration_history.append(time.perf_counter() - train_start_time)
            accumulated_stats["cumul_number_single_memories_used"] += config_copy.batch_size
            loss_history.append(loss.detach().cpu())
            if not math.isinf(grad_norm):
                grad_norm_history.append(grad_norm)

            accumulated_stats["cumul_number_batches_done"] += 1
            print(f"B    {loss=:<8.2e} {grad_norm=:<8.2e} {train_on_batch_duration_history[-1]*1000:<8.1f}")

            utilities.custom_weight_decay(online_network, 1 - weight_decay)
            if accumulated_stats["cumul_number_batches_done"] % config_copy.send_shared_network_every_n_batches == 0:
                with shared_network_lock:
                    uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

            # ===============================================
            #   UPDATE TARGET NETWORK
            # ===============================================
            if (
                accumulated_stats["cumul_number_single_memories_used"]
                >= accumulated_stats["cumul_number_single_memories_used_next_target_network_update"]
            ):
                accumulated_stats["cumul_number_target_network_updates"] += 1
                accumulated_stats["cumul_number_single_memories_used_next_target_network_update"] += (
                    config_copy.number_memories_trained_on_between_target_network_updates
                )
                utilities.soft_copy_param(target_network, online_network, config_copy.soft_update_tau)
            sys.stdout.flush()

        # ===============================================
        #   WRITE AGGREGATED STATISTICS TO TENSORBOARD EVERY 5 MINUTES
        # ===============================================
        save_frequency_s = 5 * 60
        if time.perf_counter() - time_last_save >= save_frequency_s:
            accumulated_stats["cumul_training_hours"] += (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save
            waited_percentage = time_waited_for_workers_since_last_tensorboard_write / time_since_last_save
            trained_percentage = time_training_since_last_tensorboard_write / time_since_last_save
            tested_percentage = time_testing_since_last_tensorboard_write / time_since_last_save
            time_waited_for_workers_since_last_tensorboard_write = 0
            time_training_since_last_tensorboard_write = 0
            time_testing_since_last_tensorboard_write = 0
            transitions_learned_per_second = (
                accumulated_stats["cumul_number_single_memories_used"] - transitions_learned_last_save
            ) / time_since_last_save
            time_last_save = time.perf_counter()
            transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]

            # ===============================================
            #   COLLECT VARIOUS STATISTICS
            # ===============================================
            step_stats = {
                "gamma": gamma,
                "n_steps": config_copy.n_steps,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "memory_size": len(buffer),
                "number_times_single_memory_is_used_before_discard": config_copy.number_times_single_memory_is_used_before_discard,
                "learner_percentage_waiting_for_workers": waited_percentage,
                "learner_percentage_training": trained_percentage,
                "learner_percentage_testing": tested_percentage,
                "transitions_learned_per_second": transitions_learned_per_second,
            }
            if len(loss_history) > 0:
                step_stats.update(
                    {
                        "loss": np.mean(loss_history),
                        "train_on_batch_duration": np.median(train_on_batch_duration_history),
                        "grad_norm_history_q1": np.quantile(grad_norm_history, 0.25),
                        "grad_norm_history_median": np.quantile(grad_norm_history, 0.5),
                        "grad_norm_history_q3": np.quantile(grad_norm_history, 0.75),
                        "grad_norm_history_d9": np.quantile(grad_norm_history, 0.9),
                        "grad_norm_history_d98": np.quantile(grad_norm_history, 0.98),
                        "grad_norm_history_max": np.max(grad_norm_history),
                    }
                )
                for key, val in layer_grad_norm_history.items():
                    step_stats.update(
                        {
                            f"{key}_median": np.quantile(val, 0.5),
                            f"{key}_q3": np.quantile(val, 0.75),
                            f"{key}_d9": np.quantile(val, 0.9),
                            f"{key}_c98": np.quantile(val, 0.98),
                            f"{key}_max": np.max(val),
                        }
                    )
            if isinstance(buffer._sampler, PrioritizedSampler):
                all_priorities = np.array([buffer._sampler._sum_tree.at(i) for i in range(len(buffer))])
                step_stats.update(
                    {
                        "priorities_min": np.min(all_priorities),
                        "priorities_q1": np.quantile(all_priorities, 0.1),
                        "priorities_mean": np.mean(all_priorities),
                        "priorities_median": np.quantile(all_priorities, 0.5),
                        "priorities_q3": np.quantile(all_priorities, 0.75),
                        "priorities_d9": np.quantile(all_priorities, 0.9),
                        "priorities_c98": np.quantile(all_priorities, 0.98),
                        "priorities_max": np.max(all_priorities),
                    }
                )
            for key, value in accumulated_stats.items():
                if key not in ["alltime_min_ms", "rolling_mean_ms"]:
                    step_stats[key] = value
            for key, value in accumulated_stats["alltime_min_ms"].items():
                step_stats[f"alltime_min_ms_{key}"] = value

            loss_history = []
            train_on_batch_duration_history = []
            grad_norm_history = []
            layer_grad_norm_history = defaultdict(list)

            # ===============================================
            #   WRITE TO TENSORBOARD
            # ===============================================

            walltime_tb = time.time()
            for name, param in online_network.named_parameters():
                tensorboard_writer.add_scalar(
                    tag=f"layer_{name}_L2",
                    scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
            assert len(optimizer1.param_groups) == 1
            try:
                for p, (name, _) in zip(
                    optimizer1.param_groups[0]["params"],
                    online_network.named_parameters(),
                ):
                    state = optimizer1.state[p]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    mod_lr = 1 / (exp_avg_sq.sqrt() + 1e-4)
                    tensorboard_writer.add_scalar(
                        tag=f"lr_ratio_{name}_L2",
                        scalar_value=np.sqrt((mod_lr**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"exp_avg_{name}_L2",
                        scalar_value=np.sqrt((exp_avg**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
                    tensorboard_writer.add_scalar(
                        tag=f"exp_avg_sq_{name}_L2",
                        scalar_value=np.sqrt((exp_avg_sq**2).mean().detach().cpu().item()),
                        global_step=accumulated_stats["cumul_number_frames_played"],
                        walltime=walltime_tb,
                    )
            except:
                pass

            for k, v in step_stats.items():
                tensorboard_writer.add_scalar(
                    tag=k,
                    scalar_value=v,
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )

            previous_alltime_min = previous_alltime_min or copy.deepcopy(accumulated_stats["alltime_min_ms"])

            tensorboard_writer.add_text(
                "times_summary",
                f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} "
                + " ".join(
                    [
                        f"{'**' if v < previous_alltime_min.get(k, 99999999) else ''}{k}: {v / 1000:.2f}{'**' if v < previous_alltime_min.get(k, 99999999) else ''}"
                        for k, v in accumulated_stats["alltime_min_ms"].items()
                    ]
                ),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

            previous_alltime_min = copy.deepcopy(accumulated_stats["alltime_min_ms"])

            # ===============================================
            #   SAVE
            # ===============================================
            utilities.save_checkpoint(save_dir, online_network, target_network, optimizer1, scaler)
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
