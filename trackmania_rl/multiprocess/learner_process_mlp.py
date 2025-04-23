import copy
import importlib
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import multiprocessing as mp
from multiprocessing.connection import wait
from torch.utils.tensorboard import SummaryWriter

from config_files import config_copy
from trackmania_rl import utilities
from trackmania_rl.agents.mlp import make_untrained_mlp_agent
from trackmania_rl.buffer_utilities import make_buffers
from trackmania_rl.multiprocess.learner_process_utils import (
    load_checkpoint,
    get_rollout_from_queues,
    update_buffer_size,
    update_optimizer_params,
    process_rollout,
    sample_batch_from_buffer,
    save_checkpoint,
    update_target_network,
    save_good_runs,
    dqn_loss
)


from trackmania_rl.tensorboard_logging_utils import (
    setup_tensorboard,
    get_tensorboard_writer,
    log_training_step,
    log_training_step_to_tensorboard,
    log_race_stats_to_tensorboard,
    log_detailed_tensorboard_stats,
    collect_periodic_stats,
    collect_race_stats
)


def initialize_networks_and_optimizer(accumulated_stats):
    """Initialize models and optimizer."""
    online_network, uncompiled_online_network = make_untrained_mlp_agent(config_copy.use_jit, is_inference=False)
    target_network, _ = make_untrained_mlp_agent(config_copy.use_jit, is_inference=False)

    print("Learner process started (MLP agent).")
    print(online_network)
    utilities.count_parameters(online_network)

    # Initialize optimizer with proper learning rate from schedule
    optimizer = torch.optim.RAdam(
        online_network.parameters(),
        lr=utilities.from_exponential_schedule(config_copy.lr_schedule,
                                               accumulated_stats["cumul_number_frames_played"]),
        eps=config_copy.adam_epsilon,
        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
    )

    scaler = torch.amp.GradScaler("cuda")

    return online_network, uncompiled_online_network, target_network, optimizer, scaler


def update_config_parameters(accumulated_stats, memory_size, memory_size_start_learn, tensorboard_suffix):
    """Update parameters based on current config and training progress."""
    current_step = accumulated_stats["cumul_number_frames_played"]

    # Get current memory size from schedule
    new_memory_size, new_memory_size_start_learn = utilities.from_staircase_schedule(
        config_copy.memory_size_schedule, current_step
    )

    # Get current tensorboard suffix from schedule
    new_tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule, current_step
    )

    # Get current learning rate and gamma from schedules
    learning_rate = utilities.from_exponential_schedule(config_copy.lr_schedule, current_step)
    weight_decay = config_copy.weight_decay_lr_ratio * learning_rate
    gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, current_step)

    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "tensorboard_suffix": new_tensorboard_suffix,
        "memory_size": new_memory_size,
        "memory_size_start_learn": new_memory_size_start_learn,
    }


def training_step(online_network, target_network, optimizer, scaler, batch_data, buffer=None):
    """Perform a single training step."""
    # Unpack batch data
    state_img_tensor = batch_data["state_img_tensor"]
    state_float_tensor = batch_data["state_float_tensor"]
    actions = batch_data["actions"]
    rewards = batch_data["rewards"]
    next_state_img_tensor = batch_data["next_state_img_tensor"]
    next_state_float_tensor = batch_data["next_state_float_tensor"]
    gammas_terminal = batch_data["gammas_terminal"]
    weights = batch_data["weights"]
    batch_info = batch_data["batch_info"]

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        # Compute Q(s,a)
        q_values = online_network(state_img_tensor, state_float_tensor)

        # Compute Q(s',a') for target
        with torch.no_grad():
            q_next = target_network(next_state_img_tensor, next_state_float_tensor)
            max_next_q = q_next.max(dim=1)[0]
            targets = rewards + gammas_terminal * max_next_q

        # Calculate loss with proper weight handling
        loss, td_errors = dqn_loss(q_values, actions, targets, weights)

        # Update priorities if using prioritized replay
        if "indices" in batch_info and buffer is not None and hasattr(buffer._sampler, "update_priorities"):
            buffer._sampler.update_priorities(batch_info["indices"], td_errors.cpu().numpy())

    # Backpropagation with gradient scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        online_network.parameters(),
        config_copy.clip_grad_norm
    ).detach().cpu().item()

    torch.nn.utils.clip_grad_value_(
        online_network.parameters(),
        config_copy.clip_grad_value
    )

    # Optimizer step and scaler update
    scaler.step(optimizer)
    scaler.update()

    # Compute statistics from this step
    stats = {
        "loss": loss.detach().cpu().item(),
        "mean_q": q_values.mean().detach().cpu().item(),
        "max_q": q_values.max().detach().cpu().item(),
        "min_q": q_values.min().detach().cpu().item(),
        "grad_norm": grad_norm,
        "max_next_q": max_next_q.mean().item(),
        "reward_mean": rewards.mean().item(),
        "reward_min": rewards.min().item(),
        "reward_max": rewards.max().item(),
        "reward_std": rewards.std().item(),
        "rewards": rewards,
        "td_errors": td_errors,
    }

    return stats, q_values


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
    Main learner process function for MLP agent training.

    Args:
        rollout_queues: Queues containing rollout data from collector processes
        uncompiled_shared_network: Network shared with collector processes
        shared_network_lock: Lock for synchronizing shared network updates
        shared_steps: Shared counter for tracking total frames
        base_dir: Base directory for saving models and data
        save_dir: Directory to save checkpoints
        tensorboard_base_dir: Directory for tensorboard logs
    """
    # Setup TensorBoard
    setup_tensorboard(tensorboard_base_dir)

    # Initialize stats tracking
    accumulated_stats = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}

    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models and optimizer
    online_network, uncompiled_online_network, target_network, optimizer, scaler = initialize_networks_and_optimizer(
        accumulated_stats)

    # Load checkpoint if available
    load_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats, shared_steps)

    # Sync shared network with loaded weights
    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    # Initialize tracking variables
    previous_alltime_min = None
    time_last_save = time.perf_counter()
    save_frequency_s = config_copy.save_frequency_s if hasattr(config_copy, 'save_frequency_s') else 5 * 60

    queue_check_order = list(range(len(rollout_queues)))
    rollout_queue_readers = [q._reader for q in rollout_queues]

    time_waited_for_workers_since_last_tensorboard_write = 0
    time_training_since_last_tensorboard_write = 0
    time_testing_since_last_tensorboard_write = 0

    # Ensure required keys exist in accumulated_stats
    if "rolling_mean_ms" not in accumulated_stats:
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats.get(
        "cumul_number_single_memories_used", 0)
    transitions_learned_last_save = accumulated_stats.get("cumul_number_single_memories_used", 0)
    neural_net_reset_counter = 0
    single_reset_flag = config_copy.single_reset_flag

    # Setup replay buffer
    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        config_copy.memory_size_schedule, accumulated_stats.get("cumul_number_frames_played", 0)
    )
    buffer, buffer_test = make_buffers(memory_size)
    offset_cumul_number_single_memories_used = memory_size_start_learn * config_copy.number_times_single_memory_is_used_before_discard

    # Setup TensorBoard writer
    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule,
        accumulated_stats.get("cumul_number_frames_played", 0),
    )
    tensorboard_writer = get_tensorboard_writer(
        tensorboard_base_dir,
        config_copy,
        tensorboard_suffix
    )

    # Initialize tracking histories
    loss_history = []
    loss_test_history = []
    train_on_batch_duration_history = []
    grad_norm_history = []
    layer_grad_norm_history = defaultdict(list)

    print(f"Replay buffer memory size: {memory_size}, memory_size_start_learn: {memory_size_start_learn}")
    print(f"Batch size: {config_copy.batch_size}")
    print(f"Initial shared step value: {shared_steps.value}")

    # ========================================================
    # Training loop
    # ========================================================
    # Load step from accumulated_stats if available, otherwise start from 0
    step = accumulated_stats.get("training_step", 0)
    print(f"Starting training from step {step}")

    while True:
        # Wait for data from collector processes
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time

        if time_waited > 1:
            print(f"Warning: learner waited {time_waited:.2f} seconds for workers to provide memories")

        time_waited_for_workers_since_last_tensorboard_write += time_waited

        # Get rollout data from worker queues
        rollout_data, idx = get_rollout_from_queues(rollout_queues, queue_check_order, step)

        if rollout_data:
            (
                rollout_results,
                end_race_stats,
                fill_buffer,
                is_explo,
                map_name,
                map_status,
                rollout_duration,
                loop_number,
            ) = rollout_data

        # Reload config and update parameters
        importlib.reload(config_copy)

        config_values = update_config_parameters(
            accumulated_stats, memory_size, memory_size_start_learn, tensorboard_suffix
        )

        # Update tensorboard writer if suffix changed
        # New refresh when suffix changes:
        if config_values["tensorboard_suffix"] != tensorboard_suffix:
            tensorboard_suffix = config_values["tensorboard_suffix"]
            tensorboard_writer = get_tensorboard_writer(
                tensorboard_base_dir,
                config_copy,
                tensorboard_suffix
            )

        # Update buffer size if needed
        buffer, buffer_test, memory_size, memory_size_start_learn, offset_cumul_number_single_memories_used = update_buffer_size(
            buffer, buffer_test, memory_size, config_values["memory_size"],
            memory_size_start_learn, config_values["memory_size_start_learn"],
            offset_cumul_number_single_memories_used
        )

        # Update optimizer parameters
        update_optimizer_params(optimizer, config_values["learning_rate"], buffer)

        # Extract current parameters for use in training
        learning_rate = config_values["learning_rate"]
        weight_decay = config_values["weight_decay"]
        gamma = config_values["gamma"]

        # Update frame count if we received new rollout data
        if rollout_data:
            accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])
            shared_steps.value = accumulated_stats["cumul_number_frames_played"]

            # Collect and log race statistics
            race_stats = collect_race_stats(rollout_results, end_race_stats, is_explo, map_name, map_status,
                                            rollout_duration, accumulated_stats)
            log_race_stats_to_tensorboard(tensorboard_writer, race_stats, accumulated_stats)

            # Save good runs
            save_good_runs(base_dir, save_dir, rollout_results, end_race_stats, map_name, is_explo,
                           accumulated_stats)

            # Process rollout data and fill buffer
            if fill_buffer:
                buffer, buffer_test, memories_added = process_rollout(
                    rollout_results, buffer, buffer_test, gamma, accumulated_stats
                )
                neural_net_reset_counter += memories_added

        # ===============================================
        #   Training loop
        # ===============================================

        # Ensure model is in training mode
        if not online_network.training:
            online_network.train()

        # Check if we should train (buffer size and memory usage conditions)
        should_train = (
                len(buffer) >= memory_size_start_learn and
                accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used <=
                accumulated_stats["cumul_number_single_memories_should_have_been_used"]
        )

        if not should_train:
            if len(buffer) < memory_size_start_learn:
                print(
                    f"[Not training] Buffer too small: len(buffer)={len(buffer)} < memory_size_start_learn={memory_size_start_learn}"
                )
            else:
                print(
                    "[Not training] Used memories condition not met: "
                    f"cumul_number_single_memories_used ({accumulated_stats['cumul_number_single_memories_used']}) + "
                    f"offset_cumul_number_single_memories_used ({offset_cumul_number_single_memories_used}) > "
                    f"cumul_number_single_memories_should_have_been_used ({accumulated_stats['cumul_number_single_memories_should_have_been_used']})"
                )

        # Training loop - continue until conditions no longer met
        while should_train:
            train_start_time = time.perf_counter()

            # Debug info
            print(accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used)
            print(accumulated_stats["cumul_number_single_memories_should_have_been_used"])

            # Sample batch from buffer
            batch_data = sample_batch_from_buffer(buffer, config_copy.batch_size)

            # Training step
            stats, q_values = training_step(online_network, target_network, optimizer, scaler, batch_data, buffer)

            # Track batch sizes for learning rate adjustment
            accumulated_stats["cumul_number_single_memories_used"] += (
                4 * config_copy.batch_size
                if (len(buffer) < buffer._storage.max_size and buffer._storage.max_size > 200_000)
                else config_copy.batch_size
            )  # do fewer batches while memory is not full

            # Log training step info
            log_training_step(step, stats, len(buffer))

            # Update stats
            train_duration_ms = (time.perf_counter() - train_start_time) * 1000
            train_on_batch_duration_history.append(train_duration_ms / 1000)  # Convert back to seconds for history
            time_training_since_last_tensorboard_write += time.perf_counter() - train_start_time

            # Update cumulative counters
            accumulated_stats["cumul_number_batches_done"] += 1

            # Update rolling mean for training step time
            key = "train_step"
            prev_mean = accumulated_stats["rolling_mean_ms"].get(key, 0)
            alpha = 0.01  # smoothing factor
            accumulated_stats["rolling_mean_ms"][key] = (1 - alpha) * prev_mean + alpha * train_duration_ms

            # Update all-time minimum if applicable
            if key not in accumulated_stats["alltime_min_ms"] or train_duration_ms < \
                    accumulated_stats["alltime_min_ms"][key]:
                accumulated_stats["alltime_min_ms"][key] = train_duration_ms

            # Add to history for later analysis
            loss_history.append(stats["loss"])
            if not math.isinf(stats["grad_norm"]):
                grad_norm_history.append(stats["grad_norm"])

            # Apply weight decay
            utilities.custom_weight_decay(online_network, 1 - weight_decay)

            # Update shared network periodically
            if accumulated_stats["cumul_number_batches_done"] % config_copy.send_shared_network_every_n_batches == 0:
                with shared_network_lock:
                    uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

            # Update target network if needed
            update_target_network(target_network, online_network, accumulated_stats)

            # Log to TensorBoard
            log_training_step_to_tensorboard(
                tensorboard_writer,
                step,
                stats,
                len(buffer),
                stats["max_next_q"],
                train_duration_ms * 1000,
                accumulated_stats["rolling_mean_ms"]
            )

            # Output to console
            sys.stdout.flush()
            step += 1

            # Check if we should continue training
            should_train = (
                    len(buffer) >= memory_size_start_learn and
                    accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used <=
                    accumulated_stats["cumul_number_single_memories_should_have_been_used"]
            )

        # Save to files - now include the step counter
        accumulated_stats["training_step"] = step
        utilities.save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats)

        # ===============================================
        #   Periodic checkpoint and stat logging
        # ===============================================
        if time.perf_counter() - time_last_save >= save_frequency_s:
            accumulated_stats["cumul_training_hours"] += (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save

            # Collect statistics
            training_params = {
                "gamma": gamma,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay
            }

            step_stats = collect_periodic_stats(
                accumulated_stats,
                loss_history,
                train_on_batch_duration_history,
                grad_norm_history,
                layer_grad_norm_history,
                buffer,
                training_params,
                time_waited_for_workers_since_last_tensorboard_write,
                time_training_since_last_tensorboard_write,
                time_testing_since_last_tensorboard_write,
                time_since_last_save,
                transitions_learned_last_save
            )

            # Reset timing counters
            time_waited_for_workers_since_last_tensorboard_write = 0
            time_training_since_last_tensorboard_write = 0
            time_testing_since_last_tensorboard_write = 0
            time_last_save = time.perf_counter()
            transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]

            # Log detailed stats to TensorBoard
            previous_alltime_min = log_detailed_tensorboard_stats(
                tensorboard_writer,
                online_network,
                optimizer,
                accumulated_stats,
                step_stats,
                previous_alltime_min,
                accumulated_stats["cumul_number_frames_played"]
            )

            # Reset history arrays
            loss_history = []
            train_on_batch_duration_history = []
            grad_norm_history = []
            layer_grad_norm_history = defaultdict(list)

            # Save checkpoint - now include the step counter
            accumulated_stats["training_step"] = step
            save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats)
