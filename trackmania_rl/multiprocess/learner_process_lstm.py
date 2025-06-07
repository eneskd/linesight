"""
LSTM Learner Process for Trackmania RL

This file implements the main training loop for LSTM-based agents, handling:
- Sequence-based training with temporal dependencies
- LSTM hidden state management
- Sequence sampling from replay buffer
- Tensorboard statistics tracking
"""

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
import torch.nn as nn
from torch import multiprocessing as mp
from multiprocessing.connection import wait
from torch.utils.tensorboard import SummaryWriter

from config_files import config_copy
from config_files import lstm_config_copy
from trackmania_rl import utilities
from trackmania_rl.agents.lstm import make_untrained_lstm_agent
from trackmania_rl.buffer_lstm import LSTMReplayBuffer
from trackmania_rl.multiprocess.learner_process_utils import (
    load_checkpoint,
    get_rollout_from_queues,
    save_checkpoint,
    update_target_network,
    save_good_runs,
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
    """Initialize LSTM models and optimizer."""
    online_network, uncompiled_online_network = make_untrained_lstm_agent(config_copy.use_jit, is_inference=False)
    target_network, _ = make_untrained_lstm_agent(config_copy.use_jit, is_inference=False)

    print("Learner process started (LSTM agent).")
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


def update_config_parameters(accumulated_stats):
    """Update parameters based on current config and training progress."""
    current_step = accumulated_stats["cumul_number_frames_played"]

    # Get current learning rate and gamma from schedules
    learning_rate = utilities.from_exponential_schedule(config_copy.lr_schedule, current_step)
    weight_decay = config_copy.weight_decay_lr_ratio * learning_rate
    gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, current_step)

    # Get current tensorboard suffix from schedule
    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule, current_step
    )

    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "tensorboard_suffix": tensorboard_suffix,
    }


def update_optimizer_params(optimizer, learning_rate):
    """Update optimizer parameters."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["epsilon"] = config_copy.adam_epsilon
        param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)


def lstm_dqn_loss(q_values, actions, targets, weights=None, mask=None):
    """
    Compute DQN loss for LSTM sequences.
    
    Args:
        q_values: (batch, seq_len, n_actions) - Q-values from network
        actions: (batch, seq_len) - Actions taken
        targets: (batch, seq_len) - Target Q-values
        weights: (batch,) - Importance sampling weights (optional)
        mask: (batch, seq_len) - Mask for valid timesteps (optional)
    
    Returns:
        loss: Scalar loss value
        td_errors: (batch, seq_len) - TD errors for priority updates
    """
    batch_size, seq_len, n_actions = q_values.shape
    
    # Gather Q-values for taken actions
    q_taken = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
    
    # Compute TD errors
    td_errors = targets - q_taken  # (batch, seq_len)
    
    # Apply mask if provided (to handle variable length sequences)
    if mask is not None:
        td_errors = td_errors * mask
        
    # Compute squared TD errors
    squared_errors = td_errors ** 2
    
    # Average over sequence length (considering mask)
    if mask is not None:
        # Only average over valid timesteps
        valid_steps = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        sequence_loss = squared_errors.sum(dim=1) / valid_steps.squeeze()  # (batch,)
    else:
        sequence_loss = squared_errors.mean(dim=1)  # (batch,)
    
    # Apply importance sampling weights if provided
    if weights is not None:
        sequence_loss = sequence_loss * weights
    
    # Final loss is mean over batch
    loss = sequence_loss.mean()
    
    return loss, td_errors


def process_rollout_lstm(rollout_results, buffer, gamma, accumulated_stats):
    """Process rollout data and add episodes to LSTM buffer."""
    # Import reward calculation function
    from trackmania_rl.reward_calculation import calculate_frame_rewards
    
    # Get current parameters for reward calculation
    current_step = accumulated_stats["cumul_number_frames_played"]
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
    
    # Calculate rewards for the episode
    frame_rewards = calculate_frame_rewards(
        rollout_results,
        engineered_speedslide_reward,
        engineered_neoslide_reward,
        engineered_kamikaze_reward,
        engineered_close_to_vcp_reward,
    )
    
    # Create terminal flags - True for the last frame if race finished, False otherwise
    n_frames = len(rollout_results["frames"])
    terminals = [False] * n_frames
    if "race_time" in rollout_results:  # Race finished
        terminals[-1] = True
    
    # Convert rollout to episode format for LSTM buffer
    episode_data = {
        "frames": rollout_results["frames"],
        "state_float": rollout_results["state_float"],
        "actions": rollout_results["actions"],
        "rewards": frame_rewards,
        "terminals": terminals,
    }
    
    # Add episode to buffer
    buffer.add_episode(episode_data)
    
    # Update accumulated stats
    episode_length = len(rollout_results["frames"])
    accumulated_stats["cumul_number_memories_generated"] += episode_length
    accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
        config_copy.number_times_single_memory_is_used_before_discard * episode_length
    )
    
    return buffer, episode_length


def training_step_lstm(online_network, target_network, optimizer, scaler, batch_data):
    """Perform a single LSTM training step."""
    # Unpack batch data
    img_seq = batch_data["frames"]  # (batch, seq_len, 1, H, W)
    float_seq = batch_data["state_float"]  # (batch, seq_len, float_dim)
    actions = batch_data["actions"]  # (batch, seq_len)
    rewards = batch_data["rewards"]  # (batch, seq_len)
    terminals = batch_data["terminals"]  # (batch, seq_len)
    weights = batch_data.get("weights", None)  # (batch,) - importance sampling weights
    
    batch_size, seq_len = actions.shape
    
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        # Forward pass through online network
        q_values, _ = online_network(img_seq, float_seq)  # (batch, seq_len, n_actions)
        
        # Compute targets using target network
        with torch.no_grad():
            # For LSTM, we need to compute targets for each timestep
            # Create next state sequences (shift by 1 timestep)
            next_img_seq = torch.zeros_like(img_seq)
            next_float_seq = torch.zeros_like(float_seq)
            
            # Shift sequences to get next states
            next_img_seq[:, :-1] = img_seq[:, 1:]
            next_float_seq[:, :-1] = float_seq[:, 1:]
            # Last timestep uses same state (will be masked out by terminals)
            next_img_seq[:, -1] = img_seq[:, -1]
            next_float_seq[:, -1] = float_seq[:, -1]
            
            # Get Q-values for next states
            q_next, _ = target_network(next_img_seq, next_float_seq)  # (batch, seq_len, n_actions)
            max_next_q = q_next.max(dim=2)[0]  # (batch, seq_len)
            
            # Compute targets: r + gamma * max_q_next * (1 - terminal)
            gamma_tensor = torch.full_like(rewards, lstm_config_copy.gamma)
            targets = rewards + gamma_tensor * max_next_q * (1 - terminals.float())

        # Compute loss
        loss, td_errors = lstm_dqn_loss(q_values, actions, targets, weights)

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

    # Compute statistics
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
        "td_errors": td_errors,
    }

    return stats


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
    Main learner process function for LSTM agent training.

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

    # Setup LSTM replay buffer
    buffer = LSTMReplayBuffer(
        capacity=lstm_config_copy.replay_buffer_capacity,
        seq_len=lstm_config_copy.lstm_seq_len,
        use_prioritized=False,  # Can be made configurable
        config=lstm_config_copy
    )

    # Setup TensorBoard writer
    config_values = update_config_parameters(accumulated_stats)
    tensorboard_suffix = config_values["tensorboard_suffix"]
    tensorboard_writer = get_tensorboard_writer(
        tensorboard_base_dir,
        config_copy,
        tensorboard_suffix
    )

    # Initialize tracking histories
    loss_history = []
    train_on_batch_duration_history = []
    grad_norm_history = []
    layer_grad_norm_history = defaultdict(list)

    print(f"LSTM replay buffer capacity: {lstm_config_copy.replay_buffer_capacity}")
    print(f"LSTM sequence length: {lstm_config_copy.lstm_seq_len}")
    print(f"Batch size: {lstm_config_copy.lstm_batch_size}")
    print(f"Initial shared step value: {shared_steps.value}")

    # Load step from accumulated_stats if available, otherwise start from 0
    step = accumulated_stats.get("training_step", 0)
    print(f"Starting LSTM training from step {step}")

    # ========================================================
    # Main Training Loop
    # ========================================================
    while True:
        # Wait for data from collector processes
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time

        if time_waited > 1:
            print(f"Warning: learner waited {time_waited:.2f} seconds for workers to provide memories")

        time_waited_for_workers_since_last_tensorboard_write += time_waited

        # Get rollout data from worker queues
        rollout_data, idx = get_rollout_from_queues(rollout_queues, queue_check_order)

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
        importlib.reload(lstm_config_copy)

        config_values = update_config_parameters(accumulated_stats)

        # Update tensorboard writer if suffix changed
        if config_values["tensorboard_suffix"] != tensorboard_suffix:
            tensorboard_suffix = config_values["tensorboard_suffix"]
            tensorboard_writer = get_tensorboard_writer(
                tensorboard_base_dir,
                config_copy,
                tensorboard_suffix
            )

        # Update optimizer parameters
        update_optimizer_params(optimizer, config_values["learning_rate"])

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
                buffer, episode_length = process_rollout_lstm(
                    rollout_results, buffer, gamma, accumulated_stats
                )

        # ===============================================
        #   LSTM Training Loop
        # ===============================================

        # Ensure model is in training mode
        if not online_network.training:
            online_network.train()

        # Check if we should train (buffer has enough episodes)
        min_episodes_to_start = max(1, lstm_config_copy.lstm_batch_size // 4)  # Start training with some episodes
        should_train = (
                len(buffer) >= min_episodes_to_start and
                accumulated_stats["cumul_number_single_memories_used"] <=
                accumulated_stats["cumul_number_single_memories_should_have_been_used"]
        )

        if not should_train:
            if len(buffer) < min_episodes_to_start:
                print(
                    f"[Not training] Buffer too small: episodes={len(buffer)} < min_episodes={min_episodes_to_start}"
                )
            else:
                print(
                    "[Not training] Used memories condition not met: "
                    f"cumul_number_single_memories_used ({accumulated_stats['cumul_number_single_memories_used']}) > "
                    f"cumul_number_single_memories_should_have_been_used ({accumulated_stats['cumul_number_single_memories_should_have_been_used']})"
                )

        # Training loop - continue until conditions no longer met
        while should_train:
            train_start_time = time.perf_counter()

            # Sample batch from LSTM buffer
            try:
                batch_data = buffer.sample_batch(lstm_config_copy.lstm_batch_size)
                
                # Move batch to GPU
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].to("cuda")

                # Training step
                stats = training_step_lstm(online_network, target_network, optimizer, scaler, batch_data)

                # Track memory usage for learning rate adjustment
                sequences_used = lstm_config_copy.lstm_batch_size * lstm_config_copy.lstm_seq_len
                accumulated_stats["cumul_number_single_memories_used"] += sequences_used

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
                if not math.isinf(stats["loss"]):
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
                if accumulated_stats["cumul_number_batches_done"] % lstm_config_copy.target_update_frequency == 0:
                    target_network.load_state_dict(online_network.state_dict())
                    print(f"Updated target network at step {step}")

                # Log to TensorBoard
                log_training_step_to_tensorboard(
                    tensorboard_writer,
                    step,
                    stats,
                    len(buffer),
                    stats["max_next_q"],
                    train_duration_ms,
                    accumulated_stats["rolling_mean_ms"]
                )

                # Output to console
                sys.stdout.flush()
                step += 1

            except Exception as e:
                print(f"Error during training step: {e}")
                # Continue to next iteration
                break

            # Check if we should continue training
            should_train = (
                    len(buffer) >= min_episodes_to_start and
                    accumulated_stats["cumul_number_single_memories_used"] <=
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
                buffer,  # Pass LSTM buffer
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
                step
            )

            # Reset history arrays
            loss_history = []
            train_on_batch_duration_history = []
            grad_norm_history = []
            layer_grad_norm_history = defaultdict(list)

            # Save checkpoint - now include the step counter
            accumulated_stats["training_step"] = step
            save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats)

            print(f"LSTM Buffer stats: {len(buffer)} episodes stored")