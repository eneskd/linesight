"""
This file implements the main training loop, tensorboard statistics tracking, etc...
"""

import copy
import importlib
import math
import random
import sys
import time
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
from config_files import iqn_config_copy

from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents import iqn as iqn
from trackmania_rl.agents.iqn import make_untrained_iqn_network
from trackmania_rl.analysis_metrics import (
    distribution_curves,
    highest_prio_transitions,
    loss_distribution,
    race_time_left_curves,
    tau_curves,
)
from trackmania_rl.buffer_utilities import make_buffers, resize_buffers
from trackmania_rl.map_reference_times import reference_times
from trackmania_rl.multiprocess.learner_process_utils import (
    setup_tensorboard,
    load_checkpoint,
    get_rollout_from_queues,
    update_buffer_size,
    log_training_step,
    log_to_tensorboard,
    collect_race_stats,
    log_race_stats_to_tensorboard,
    log_detailed_tensorboard_stats,
    save_checkpoint,
    update_target_network,
    save_good_runs,
)


def update_optimizer_params(optimizer, params, buffer):
    """Update optimizer parameters and buffer settings."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = params["learning_rate"]
        param_group["epsilon"] = config_copy.adam_epsilon
        param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)

    # Update prioritized sampler parameters if applicable
    if isinstance(buffer._sampler, PrioritizedSampler):
        buffer._sampler._alpha = config_copy.prio_alpha
        buffer._sampler._beta = config_copy.prio_beta
        buffer._sampler._eps = config_copy.prio_epsilon

def initialize_networks_and_optimizer(accumulated_stats):
    """Initialize models and optimizer."""
    online_network, uncompiled_online_network = make_untrained_iqn_network(config_copy.use_jit, is_inference=False)
    target_network, _ = make_untrained_iqn_network(config_copy.use_jit, is_inference=False)
    print("Learner process started (IQN agent).")
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

def process_rollout(rollout_results, buffer, buffer_test, params, accumulated_stats):
    """Process rollout data and fill buffer with transitions."""
    # Fill buffer with transitions
    buffer, buffer_test, number_memories_added_train, number_memories_added_test = (
        buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
            buffer,
            buffer_test,
            rollout_results,
            config_copy.n_steps,
            params["gamma"],
            config_copy.discard_non_greedy_actions_in_nsteps,
            params["engineered_speedslide_reward"],
            params["engineered_neoslide_reward"],
            params["engineered_kamikaze_reward"],
            params["engineered_close_to_vcp_reward"],
        )
    )

    # Update accumulated stats
    accumulated_stats["cumul_number_memories_generated"] += number_memories_added_train + number_memories_added_test
    accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
            config_copy.number_times_single_memory_is_used_before_discard * number_memories_added_train
    )

    return buffer, buffer_test, number_memories_added_train


def update_config_parameters(accumulated_stats):
    """Update parameters based on current config and training progress."""
    current_step = accumulated_stats["cumul_number_frames_played"]

    # Get current memory size from schedule
    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        config_copy.memory_size_schedule, current_step
    )

    # Get current tensorboard suffix from schedule
    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule, current_step
    )

    # Get current learning rate and gamma from schedules
    learning_rate = utilities.from_exponential_schedule(config_copy.lr_schedule, current_step)
    weight_decay = config_copy.weight_decay_lr_ratio * learning_rate
    gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, current_step)

    # Get reward coefficients from schedules
    engineered_speedslide_reward = utilities.from_linear_schedule(
        config_copy.engineered_speedslide_reward_schedule, current_step
    )
    engineered_neoslide_reward = utilities.from_linear_schedule(
        config_copy.engineered_neoslide_reward_schedule, current_step
    )
    engineered_kamikaze_reward = utilities.from_linear_schedule(
        config_copy.engineered_kamikaze_reward_schedule, current_step
    )
    engineered_close_to_vcp_reward = utilities.from_linear_schedule(
        config_copy.engineered_close_to_vcp_reward_schedule, current_step
    )

    # Get exploration parameters
    epsilon = utilities.from_exponential_schedule(config_copy.epsilon_schedule, current_step)
    epsilon_boltzmann = utilities.from_exponential_schedule(config_copy.epsilon_boltzmann_schedule, current_step)

    return {
        "memory_size": memory_size,
        "memory_size_start_learn": memory_size_start_learn,
        "tensorboard_suffix": tensorboard_suffix,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "engineered_speedslide_reward": engineered_speedslide_reward,
        "engineered_neoslide_reward": engineered_neoslide_reward,
        "engineered_kamikaze_reward": engineered_kamikaze_reward,
        "engineered_close_to_vcp_reward": engineered_close_to_vcp_reward,
        "epsilon": epsilon,
        "epsilon_boltzmann": epsilon_boltzmann,
    }



def collect_periodic_stats_iqn(accumulated_stats, loss_history, train_on_batch_duration_history, grad_norm_history,
                               layer_grad_norm_history, buffer, gamma, learning_rate, weight_decay,
                               time_waited_for_workers, time_training, time_testing, time_since_last_save,
                               transitions_learned_last_save, epsilon, epsilon_boltzmann):
    """Collect statistics for periodic reporting and saving."""
    transitions_learned_per_second = (
                                             accumulated_stats[
                                                 "cumul_number_single_memories_used"] - transitions_learned_last_save
                                     ) / time_since_last_save

    waited_percentage = time_waited_for_workers / time_since_last_save
    trained_percentage = time_training / time_since_last_save
    tested_percentage = time_testing / time_since_last_save

    # Basic statistics
    step_stats = {
        "gamma": gamma,
        "n_steps": config_copy.n_steps,
        "epsilon": epsilon,
        "epsilon_boltzmann": epsilon_boltzmann,
        "tau_epsilon_boltzmann": config_copy.tau_epsilon_boltzmann,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "discard_non_greedy_actions_in_nsteps": config_copy.discard_non_greedy_actions_in_nsteps,
        "memory_size": len(buffer),
        "number_times_single_memory_is_used_before_discard": config_copy.number_times_single_memory_is_used_before_discard,
        "learner_percentage_waiting_for_workers": waited_percentage,
        "learner_percentage_training": trained_percentage,
        "learner_percentage_testing": tested_percentage,
        "transitions_learned_per_second": transitions_learned_per_second,
    }

    # Loss and gradient statistics
    if len(loss_history) > 0:
        step_stats.update({
            "loss": np.mean(loss_history),
            "train_on_batch_duration": np.median(train_on_batch_duration_history),
            "grad_norm_history_q1": np.quantile(grad_norm_history, 0.25),
            "grad_norm_history_median": np.quantile(grad_norm_history, 0.5),
            "grad_norm_history_q3": np.quantile(grad_norm_history, 0.75),
            "grad_norm_history_d9": np.quantile(grad_norm_history, 0.9),
            "grad_norm_history_d98": np.quantile(grad_norm_history, 0.98),
            "grad_norm_history_max": np.max(grad_norm_history),
        })

        # Layer-specific gradient statistics
        for key, val in layer_grad_norm_history.items():
            if len(val) > 0:
                step_stats.update({
                    f"{key}_median": np.quantile(val, 0.5),
                    f"{key}_q3": np.quantile(val, 0.75),
                    f"{key}_d9": np.quantile(val, 0.9),
                    f"{key}_c98": np.quantile(val, 0.98),
                    f"{key}_max": np.max(val),
                })

    # Prioritized replay statistics
    if isinstance(buffer._sampler, PrioritizedSampler):
        try:
            all_priorities = np.array([buffer._sampler._sum_tree.at(i) for i in range(len(buffer))])
            step_stats.update({
                "priorities_min": np.min(all_priorities),
                "priorities_q1": np.quantile(all_priorities, 0.1),
                "priorities_mean": np.mean(all_priorities),
                "priorities_median": np.quantile(all_priorities, 0.5),
                "priorities_q3": np.quantile(all_priorities, 0.75),
                "priorities_d9": np.quantile(all_priorities, 0.9),
                "priorities_c98": np.quantile(all_priorities, 0.98),
                "priorities_max": np.max(all_priorities),
            })
        except Exception as e:
            print(f"Error collecting priority statistics: {e}")

    # Add accumulated stats except for complex nested objects
    for key, value in accumulated_stats.items():
        if key not in ["alltime_min_ms", "rolling_mean_ms"]:
            step_stats[key] = value

    # Add minimum times
    for key, value in accumulated_stats["alltime_min_ms"].items():
        step_stats[f"alltime_min_ms_{key}"] = value

    return step_stats





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
    Main learner process function for IQN agent training.

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

    # Initialize tracking variables
    accumulated_stats = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    previous_alltime_min = None
    time_last_save = time.perf_counter()
    queue_check_order = list(range(len(rollout_queues)))
    rollout_queue_readers = [q._reader for q in rollout_queues]
    time_waited_for_workers_since_last_tensorboard_write = 0
    time_training_since_last_tensorboard_write = 0
    time_testing_since_last_tensorboard_write = 0

    # Initialize networks and optimizer
    online_network, uncompiled_online_network, target_network, optimizer, scaler = initialize_networks_and_optimizer(
        accumulated_stats)

    # Load checkpoint if available
    load_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats, shared_steps)

    # Sync shared network with loaded weights
    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    # Initialize buffer and training state
    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        config_copy.memory_size_schedule, accumulated_stats["cumul_number_frames_played"]
    )
    buffer, buffer_test = make_buffers(memory_size)
    offset_cumul_number_single_memories_used = memory_size_start_learn * config_copy.number_times_single_memory_is_used_before_discard

    # Initialize training tracking
    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats.get(
        "cumul_number_single_memories_used", 0)
    transitions_learned_last_save = accumulated_stats.get("cumul_number_single_memories_used", 0)
    neural_net_reset_counter = 0
    single_reset_flag = config_copy.single_reset_flag

    # Setup tensorboard
    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule, accumulated_stats["cumul_number_frames_played"]
    )
    tensorboard_writer = SummaryWriter(
        log_dir=str(tensorboard_base_dir / (config_copy.run_name + '_' + config_copy.agent_type + tensorboard_suffix))
    )

    # Initialize history trackers
    loss_history = []
    loss_test_history = []
    train_on_batch_duration_history = []
    grad_norm_history = []
    layer_grad_norm_history = defaultdict(list)

    # Create trainer and inferer
    trainer = iqn.Trainer(
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        scaler=scaler,
        batch_size=config_copy.batch_size,
        iqn_n=iqn_config_copy.iqn_n,
    )

    inferer = iqn.Inferer(
        inference_network=online_network,
        iqn_k=iqn_config_copy.iqn_k,
        tau_epsilon_boltzmann=config_copy.tau_epsilon_boltzmann,
    )

    # Main training loop
    while True:
        # Wait for rollout data from workers
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time
        if time_waited > 1:
            print(f"Warning: learner waited {time_waited:.2f} seconds for workers to provide memories")
        time_waited_for_workers_since_last_tensorboard_write += time_waited


        rollout_data, idx = get_rollout_from_queues(rollout_queues, queue_check_order)
        # Get rollout data from queue

        if rollout_data is None:
            continue

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

        # Reload config in case it changed
        importlib.reload(config_copy)

        # Update tensorboard writer if suffix changed
        new_tensorboard_suffix = utilities.from_staircase_schedule(
            config_copy.tensorboard_suffix_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_tensorboard_suffix != tensorboard_suffix:
            tensorboard_suffix = new_tensorboard_suffix
            tensorboard_writer = SummaryWriter(
                log_dir=str(
                    tensorboard_base_dir / (config_copy.run_name + '_' + config_copy.agent_type + tensorboard_suffix))
            )

        # Update buffer size if needed
        new_memory_size, new_memory_size_start_learn = utilities.from_staircase_schedule(
            config_copy.memory_size_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )

        buffer, buffer_test, memory_size, memory_size_start_learn, offset_cumul_number_single_memories_used = (
            update_buffer_size(buffer, buffer_test, memory_size, new_memory_size, memory_size_start_learn, new_memory_size_start_learn, offset_cumul_number_single_memories_used))


        # Get current parameters from schedules
        params = update_config_parameters(accumulated_stats)

        # Update optimizer parameters
        update_optimizer_params(optimizer, params, buffer)

        # Generate analysis plots if enabled
        if config_copy.plot_race_time_left_curves and not is_explo and (loop_number // 5) % 17 == 0:
            race_time_left_curves(rollout_results, inferer, save_dir, map_name)
            tau_curves(rollout_results, inferer, save_dir, map_name)
            distribution_curves(buffer, save_dir, online_network, target_network)
            loss_distribution(buffer, save_dir, online_network, target_network)

        # Update frame count
        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

        # Collect and log race statistics
        race_stats = collect_race_stats(rollout_results, end_race_stats, is_explo, map_name, map_status,
                                        rollout_duration, accumulated_stats)
        log_race_stats_to_tensorboard(tensorboard_writer, race_stats, accumulated_stats)

        # Save good runs
        save_good_runs(base_dir, save_dir, rollout_results, end_race_stats, map_name, is_explo,
                       accumulated_stats)

        # Process rollout and fill buffer
        if fill_buffer:
            buffer, buffer_test, number_memories_added_train = process_rollout(
                rollout_results, buffer, buffer_test, params, accumulated_stats
            )

            shared_steps.value = accumulated_stats["cumul_number_frames_played"]
            neural_net_reset_counter += number_memories_added_train

            # Handle periodic network reset
            if neural_net_reset_counter >= config_copy.reset_every_n_frames_generated or single_reset_flag != config_copy.single_reset_flag:
                neural_net_reset_counter = 0
                single_reset_flag = config_copy.single_reset_flag
                accumulated_stats[
                    "cumul_number_single_memories_should_have_been_used"] += config_copy.additional_transition_after_reset

                # Reset network weights
                _, untrained_iqn_network = make_untrained_iqn_network(config_copy.use_jit, False)
                utilities.soft_copy_param(online_network, untrained_iqn_network, config_copy.overall_reset_mul_factor)

                # Reset output layer weights
                with torch.no_grad():
                    online_network.A_head[2].weight = utilities.linear_combination(
                        online_network.A_head[2].weight,
                        untrained_iqn_network.A_head[2].weight,
                        config_copy.last_layer_reset_factor,
                    )
                    online_network.A_head[2].bias = utilities.linear_combination(
                        online_network.A_head[2].bias,
                        untrained_iqn_network.A_head[2].bias,
                        config_copy.last_layer_reset_factor,
                    )
                    online_network.V_head[2].weight = utilities.linear_combination(
                        online_network.V_head[2].weight,
                        untrained_iqn_network.V_head[2].weight,
                        config_copy.last_layer_reset_factor,
                    )
                    online_network.V_head[2].bias = utilities.linear_combination(
                        online_network.V_head[2].bias,
                        untrained_iqn_network.V_head[2].bias,
                        config_copy.last_layer_reset_factor,
                    )

            # Ensure network is in training mode
            if not online_network.training:
                online_network.train()

            # Check if we should train
            can_train = (
                    len(buffer) >= memory_size_start_learn and
                    accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used <=
                    accumulated_stats["cumul_number_single_memories_should_have_been_used"]
            )

            if not can_train:
                if len(buffer) < memory_size_start_learn:
                    print(
                        f"[Not training] Buffer too small: len(buffer)={len(buffer)} < memory_size_start_learn={memory_size_start_learn}")
                else:
                    print(
                        "[Not training] Used memories condition not met: "
                        f"cumul_number_single_memories_used ({accumulated_stats['cumul_number_single_memories_used']}) + "
                        f"offset_cumul_number_single_memories_used ({offset_cumul_number_single_memories_used}) > "
                        f"cumul_number_single_memories_should_have_been_used ({accumulated_stats['cumul_number_single_memories_should_have_been_used']})"
                    )

            # Training loop
            while can_train:
                # Decide whether to train on test buffer or main buffer
                if (random.random() < config_copy.buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                    # Test on test buffer (no learning)
                    test_start_time = time.perf_counter()
                    loss, _ = trainer.train_on_batch(buffer_test, do_learn=False)
                    time_testing_since_last_tensorboard_write += time.perf_counter() - test_start_time
                    loss_test_history.append(loss)
                    print(f"BT   {loss=:<8.2e}")
                else:
                    # Train on main buffer
                    train_start_time = time.perf_counter()
                    loss, grad_norm = trainer.train_on_batch(buffer, do_learn=True)
                    train_duration = time.perf_counter() - train_start_time
                    train_on_batch_duration_history.append(train_duration)
                    time_training_since_last_tensorboard_write += train_duration

                    # Update memory usage counters
                    batch_size_multiplier = 4 if (
                                len(buffer) < buffer._storage.max_size and buffer._storage.max_size > 200_000) else 1
                    accumulated_stats[
                        "cumul_number_single_memories_used"] += batch_size_multiplier * config_copy.batch_size

                    # Record training stats
                    loss_history.append(loss)
                    if not math.isinf(grad_norm):
                        grad_norm_history.append(grad_norm)

                    accumulated_stats["cumul_number_batches_done"] += 1
                    print(f"B    {loss=:<8.2e} {grad_norm=:<8.2e} {train_duration * 1000:<8.1f}")

                    # Apply weight decay
                    utilities.custom_weight_decay(online_network, 1 - params["weight_decay"])

                    # Update shared network periodically
                    if accumulated_stats[
                        "cumul_number_batches_done"] % config_copy.send_shared_network_every_n_batches == 0:
                        with shared_network_lock:
                            uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

                    # Update target network if needed
                    update_target_network(target_network, online_network, accumulated_stats)

                # Check if we should continue training
                can_train = (
                        accumulated_stats[
                            "cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used <=
                        accumulated_stats["cumul_number_single_memories_should_have_been_used"]
                )

            sys.stdout.flush()

        # Periodically save checkpoint and log statistics
        utilities.save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats)

        save_frequency_s = config_copy.save_frequency_s
        if time.perf_counter() - time_last_save >= save_frequency_s:
            # Update training time tracking
            accumulated_stats["cumul_training_hours"] += (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save


            # Collect statistics
            step_stats = collect_periodic_stats_iqn(
                accumulated_stats,
                loss_history,
                train_on_batch_duration_history,
                grad_norm_history,
                layer_grad_norm_history,
                buffer,
                params["gamma"],
                params["learning_rate"],
                params["weight_decay"],
                time_waited_for_workers_since_last_tensorboard_write,
                time_training_since_last_tensorboard_write,
                time_testing_since_last_tensorboard_write,
                time_since_last_save,
                transitions_learned_last_save,
                params["epsilon"],
                params["epsilon_boltzmann"]
            )

            # Reset timing counters
            time_waited_for_workers_since_last_tensorboard_write = 0
            time_training_since_last_tensorboard_write = 0
            time_testing_since_last_tensorboard_write = 0
            time_last_save = time.perf_counter()
            transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]

            # Reset history trackers
            loss_history = []
            loss_test_history = []
            train_on_batch_duration_history = []
            grad_norm_history = []
            layer_grad_norm_history = defaultdict(list)

            # Collect IQN spread statistics
            if online_network.training:
                online_network.eval()
            tau = torch.linspace(0.05, 0.95, iqn_config_copy.iqn_k)[:, None].to("cuda")
            per_quantile_output = inferer.infer_network(rollout_results["frames"][0], rollout_results["state_float"][0],
                                                        tau)
            for i, std in enumerate(list(per_quantile_output.std(axis=0))):
                step_stats[f"std_within_iqn_quantiles_for_action{i}"] = std

            # Log detailed statistics to tensorboard
            previous_alltime_min = log_detailed_tensorboard_stats(
                tensorboard_writer,
                online_network,
                optimizer,
                accumulated_stats,
                step_stats,
                previous_alltime_min,
                accumulated_stats["cumul_number_frames_played"]
            )

            # Log buffer statistics
            if len(buffer) > 0:
                state_floats = np.array([experience.state_float for experience in buffer._storage])
                mean_in_buffer = state_floats.mean(axis=0)
                std_in_buffer = state_floats.std(axis=0)

                print("Raw mean in buffer  :", mean_in_buffer.round(1))
                print("Raw std in buffer   :", std_in_buffer.round(1))
                print("")
                print(
                    "Corr mean in buffer :",
                    ((mean_in_buffer - config_copy.float_inputs_mean) / config_copy.float_inputs_std).round(1),
                )
                print(
                    "Corr std in buffer  :",
                    (std_in_buffer / config_copy.float_inputs_std).round(1),
                )
                print("")

            # Generate high priority transition visualizations
            if config_copy.make_highest_prio_figures and isinstance(buffer._sampler, PrioritizedSampler):
                highest_prio_transitions(buffer, save_dir)

            # Save checkpoint
            save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats)
