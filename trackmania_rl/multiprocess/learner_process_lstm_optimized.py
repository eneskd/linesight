"""
Optimized LSTM Learner Process for Trackmania RL

This file implements an optimized training loop for LSTM-based agents with:
- Advanced memory management and caching
- Optimized sequence processing
- Improved batch sampling strategies
- Enhanced computational efficiency
- Better gradient handling and loss computation
"""

import copy
import importlib
import math
import random
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
from multiprocessing.connection import wait
from torch.utils.tensorboard import SummaryWriter

from config_files import config_copy
from config_files import lstm_config_optimized_copy as lstm_config_copy
from trackmania_rl import utilities
from trackmania_rl.agents.lstm_optimized import make_optimized_lstm_agent as make_untrained_lstm_agent
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


class OptimizedLSTMTrainer:
    """Optimized trainer class for LSTM agents with advanced features."""
    
    def __init__(self, online_network, target_network, optimizer, scaler, device="cuda"):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        
        # Optimization features
        self.use_gradient_accumulation = getattr(lstm_config_copy, 'gradient_accumulation_steps', 1)
        self.use_mixed_precision = False  # Temporarily disable to debug gradient issues
        self.use_sequence_masking = True
        self.use_temporal_weighting = getattr(lstm_config_copy, 'temporal_weighting', True)
        
        # Memory optimization
        self.max_sequence_cache_size = 1000
        self.sequence_cache = deque(maxlen=self.max_sequence_cache_size)
        self.cleanup_frequency = 1000
        
        # Performance tracking
        self.forward_times = deque(maxlen=100)
        self.backward_times = deque(maxlen=100)
        
        
        # Memory monitoring
        import gc
        self.gc = gc
    
    def cleanup_memory(self):
        """Cleanup GPU memory and caches to prevent memory leaks."""
        # Clear sequence cache
        self.sequence_cache.clear()
        
        # Force garbage collection
        self.gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"Memory cleanup performed. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        
    def compute_sequence_mask(self, terminals, seq_len):
        """Create mask for valid timesteps in sequences."""
        batch_size = terminals.shape[0]
        mask = torch.ones_like(terminals, dtype=torch.float32)
        
        # For each sequence, mask out timesteps after terminal
        for b in range(batch_size):
            terminal_indices = torch.where(terminals[b])[0]
            if len(terminal_indices) > 0:
                first_terminal = terminal_indices[0]
                if first_terminal < seq_len - 1:
                    mask[b, first_terminal + 1:] = 0.0
        
        return mask
    
    def compute_temporal_weights(self, seq_len, decay_factor=0.95):
        """Compute temporal importance weights for sequence timesteps."""
        weights = torch.pow(decay_factor, torch.arange(seq_len - 1, -1, -1, dtype=torch.float32))
        return weights.to(self.device)
    
    def optimized_lstm_loss(self, q_values, actions, targets, weights=None, mask=None, temporal_weights=None):
        """
        Optimized DQN loss computation for LSTM sequences with advanced features.
        """
        batch_size, seq_len, n_actions = q_values.shape
        
        # Gather Q-values for taken actions - optimized indexing
        # Ensure actions are of type Long for gather operation
        q_taken = torch.gather(q_values, 2, actions.long().unsqueeze(-1)).squeeze(-1)
        
        # Compute TD errors
        td_errors = targets - q_taken
        
        # Apply sequence mask if provided
        if mask is not None:
            td_errors = td_errors * mask
            
        # Apply temporal weighting
        if temporal_weights is not None and self.use_temporal_weighting:
            td_errors = td_errors * temporal_weights.unsqueeze(0)
        
        # Compute loss with Huber loss for better stability
        if getattr(lstm_config_copy, 'use_huber_loss', True):
            huber_delta = getattr(lstm_config_copy, 'huber_delta', 1.0)
            # Cast to float32 to avoid float16 underflow
            q_taken_f32 = q_taken.float()
            targets_f32 = targets.float()
            
            # Ensure beta parameter is float
            loss_per_timestep = F.smooth_l1_loss(q_taken_f32, targets_f32, reduction='none', beta=float(huber_delta))
        else:
            loss_per_timestep = (td_errors ** 2)
        
        # Apply mask to loss
        if mask is not None:
            loss_per_timestep = loss_per_timestep * mask
            valid_steps = mask.sum(dim=1, keepdim=True).clamp(min=1)
            sequence_loss = loss_per_timestep.sum(dim=1) / valid_steps.squeeze()
        else:
            sequence_loss = loss_per_timestep.mean(dim=1)
        
        # Apply importance sampling weights
        if weights is not None:
            sequence_loss = sequence_loss * weights

        # Final loss
        loss = sequence_loss.mean()

        return loss, td_errors.abs().detach()
    
    def train_step(self, batch_data, step):
        """Optimized training step with advanced features."""
        start_time = time.perf_counter()
        
        # Unpack batch data
        img_seq = batch_data["frames"]
        float_seq = batch_data["state_float"]
        actions = batch_data["actions"]
        rewards = batch_data["rewards"]
        terminals = batch_data["terminals"]
        weights = batch_data.get("weights", None)
        
        batch_size, seq_len = actions.shape
        
        # Create sequence mask
        mask = self.compute_sequence_mask(terminals, seq_len) if self.use_sequence_masking else None
        
        # Create temporal weights
        temporal_weights = self.compute_temporal_weights(seq_len) if self.use_temporal_weighting else None
        
        # Zero gradients only when not accumulating
        if step % self.use_gradient_accumulation == 0:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision (only for network and target computation)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_mixed_precision):
            forward_start = time.perf_counter()
            
            # Forward pass through online network
            q_values, _, _ = self.online_network(img_seq, float_seq)
            
            # Compute targets using target network
            with torch.no_grad():
                # Optimized next state computation
                next_img_seq = torch.cat([img_seq[:, 1:], img_seq[:, -1:]], dim=1)
                next_float_seq = torch.cat([float_seq[:, 1:], float_seq[:, -1:]], dim=1)
                
                # Get Q-values for next states
                q_next, _, _ = self.target_network(next_img_seq, next_float_seq)
                max_next_q = q_next.max(dim=2)[0]
                
                # Compute targets with proper gamma handling
                gamma_value = getattr(lstm_config_copy, 'gamma', 0.99)  # Use fallback if not found
                gamma_tensor = torch.full_like(rewards, gamma_value)
                targets = rewards + gamma_tensor * max_next_q * (1 - terminals.float())
            
            forward_time = time.perf_counter() - forward_start
            self.forward_times.append(forward_time)
        # Compute loss OUTSIDE autocast
        loss, td_errors = self.optimized_lstm_loss(
            q_values, actions, targets, weights, mask, temporal_weights
        )
        
        # Scale loss for gradient accumulation
        loss = loss / self.use_gradient_accumulation
        
        # Backward pass
        backward_start = time.perf_counter()
        self.scaler.scale(loss).backward()
        
        # Update weights only after accumulating gradients
        if (step + 1) % self.use_gradient_accumulation == 0:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            
            # Advanced gradient clipping
            grad_norm = self.advanced_gradient_clipping()
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # For gradient accumulation steps, compute norm on scaled gradients
            # and then divide by the scale factor to get the true norm
            current_scale = self.scaler.get_scale()
            
            # Compute norm on scaled gradients
            total_norm = 0.0
            for param in self.online_network.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            scaled_grad_norm = total_norm ** 0.5
            
            # Unscale the norm to get the true gradient norm
            grad_norm = scaled_grad_norm / current_scale if current_scale > 0 else 0.0
        
        backward_time = time.perf_counter() - backward_start
        self.backward_times.append(backward_time)
        
        # Compute statistics
        stats = {
            "loss": loss.detach().cpu().item() * self.use_gradient_accumulation,  # Unscale for logging
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
            "forward_time": forward_time,
            "backward_time": backward_time,
            "total_time": time.perf_counter() - start_time,
        }
        
        # Add mask statistics if used
        if mask is not None:
            stats["avg_sequence_length"] = mask.sum().item() / batch_size
            stats["mask_efficiency"] = mask.mean().item()
        
        return stats
    
    def advanced_gradient_clipping(self):
        """Advanced gradient clipping with per-layer analysis."""
        # Check if gradients exist
        grad_count = 0
        for param in self.online_network.parameters():
            if param.grad is not None:
                grad_count += 1

        if grad_count == 0:
            return 0.0

        # Compute gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=getattr(lstm_config_copy, 'max_grad_norm', 0.5)
        )
        
        # Additional gradient value clipping
        torch.nn.utils.clip_grad_value_(
            self.online_network.parameters(),
            clip_value=getattr(lstm_config_copy, 'max_grad_value', 0.1)
        )

        return total_norm.detach().cpu().item()


def initialize_networks_and_optimizer(accumulated_stats):
    """Initialize LSTM models and optimizer with optimizations."""
    online_network, uncompiled_online_network = make_untrained_lstm_agent(config_copy.use_jit, is_inference=False)
    target_network, _ = make_untrained_lstm_agent(config_copy.use_jit, is_inference=False)

    print("Optimized LSTM Learner process started.")
    print(online_network)
    utilities.count_parameters(online_network)

    # Use AdamW optimizer for better weight decay handling
    optimizer_class = getattr(lstm_config_copy, 'optimizer', 'RAdam')
    if optimizer_class == 'AdamW':
        optimizer = torch.optim.AdamW(
            online_network.parameters(),
            lr=utilities.from_exponential_schedule(config_copy.lr_schedule,
                                                   accumulated_stats["cumul_number_frames_played"]),
            eps=config_copy.adam_epsilon,
            betas=(config_copy.adam_beta1, config_copy.adam_beta2),
            weight_decay=getattr(lstm_config_copy, 'weight_decay', 0.01)
        )
    else:
        optimizer = torch.optim.RAdam(
            online_network.parameters(),
            lr=utilities.from_exponential_schedule(config_copy.lr_schedule,
                                                   accumulated_stats["cumul_number_frames_played"]),
            eps=config_copy.adam_epsilon,
            betas=(config_copy.adam_beta1, config_copy.adam_beta2),
        )

    scaler = torch.amp.GradScaler("cuda")
    
    # Enable optimizations
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = True
    
    return online_network, uncompiled_online_network, target_network, optimizer, scaler


def optimized_process_rollout_lstm(rollout_results, buffer, gamma, accumulated_stats):
    """Optimized rollout processing with caching and vectorization."""
    # Cache reward calculation parameters to avoid repeated schedule lookups
    if not hasattr(optimized_process_rollout_lstm, '_cached_params'):
        optimized_process_rollout_lstm._cached_params = {}
        optimized_process_rollout_lstm._last_update_step = -1
    
    current_step = accumulated_stats["cumul_number_frames_played"]

    # Update cached parameters periodically OR if cache is empty (first call)
    if (current_step - optimized_process_rollout_lstm._last_update_step > 1000 or
            not optimized_process_rollout_lstm._cached_params):
        
        optimized_process_rollout_lstm._cached_params = {
            'engineered_speedslide_reward': utilities.from_linear_schedule(
                config_copy.engineered_speedslide_reward_schedule, current_step
            ),
            'engineered_neoslide_reward': utilities.from_linear_schedule(
                config_copy.engineered_neoslide_reward_schedule, current_step
            ),
            'engineered_kamikaze_reward': utilities.from_linear_schedule(
                config_copy.engineered_kamikaze_reward_schedule, current_step
            ),
            'engineered_close_to_vcp_reward': utilities.from_linear_schedule(
                config_copy.engineered_close_to_vcp_reward_schedule, current_step
            ),
            'wall_collision_penalty' : utilities.from_exponential_schedule(
            config_copy.wall_collision_penalty_schedule,current_step
            ),
        }
        optimized_process_rollout_lstm._last_update_step = current_step


    # Calculate rewards using cached parameters
    from trackmania_rl.reward_calculation import calculate_frame_rewards
    frame_rewards = calculate_frame_rewards(
        rollout_results,
        optimized_process_rollout_lstm._cached_params['engineered_speedslide_reward'],
        optimized_process_rollout_lstm._cached_params['engineered_neoslide_reward'],
        optimized_process_rollout_lstm._cached_params['engineered_kamikaze_reward'],
        optimized_process_rollout_lstm._cached_params['engineered_close_to_vcp_reward'],
        optimized_process_rollout_lstm._cached_params['wall_collision_penalty'],
    )
    
    # Vectorized terminal flag creation
    n_frames = len(rollout_results["frames"])
    terminals = np.zeros(n_frames, dtype=bool)
    if "race_time" in rollout_results:
        terminals[-1] = True
    
    # Pre-process data for better memory layout
    # Handle NaN values in actions (which occur when race finishes successfully)
    actions_raw = rollout_results["actions"]
    
    # Filter out NaN values from actions and corresponding data
    valid_indices = []
    valid_actions = []
    
    for i, action in enumerate(actions_raw):
        if not (isinstance(action, float) and np.isnan(action)):
            valid_indices.append(i)
            valid_actions.append(int(action))
    
    # If we have valid actions, process them
    if valid_actions:
        # Filter other arrays to match valid actions
        valid_frames = [rollout_results["frames"][i] for i in valid_indices]
        valid_state_float = [rollout_results["state_float"][i] for i in valid_indices]
        valid_rewards = [frame_rewards[i] for i in valid_indices]
        valid_terminals = terminals[valid_indices] if len(valid_indices) < len(terminals) else terminals[:len(valid_indices)]
        
        episode_data = {
            "frames": valid_frames,
            "state_float": np.array(valid_state_float, dtype=np.float32),
            "actions": np.array(valid_actions, dtype=np.int64),
            "rewards": np.array(valid_rewards, dtype=np.float32),
            "terminals": valid_terminals,
        }
    else:
        # If no valid actions, create empty episode data
        print("Warning: No valid actions found in rollout, skipping episode")
        episode_data = None
    
    # Add episode to buffer only if we have valid data
    if episode_data is not None:
        buffer.add_episode(episode_data)
        episode_length = len(episode_data["frames"])
    else:
        episode_length = 0
    
    # Update accumulated stats
    total_frames = len(rollout_results["frames"])
    accumulated_stats["cumul_number_memories_generated"] += total_frames
    accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
        config_copy.number_times_single_memory_is_used_before_discard * total_frames
    )
    
    return buffer, episode_length


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
    Optimized main learner process function for LSTM agent training.
    """
    # Setup TensorBoard
    setup_tensorboard(tensorboard_base_dir)

    # Initialize stats tracking
    accumulated_stats = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models and optimizer
    online_network, uncompiled_online_network, target_network, optimizer, scaler = initialize_networks_and_optimizer(
        accumulated_stats)

    # Load checkpoint if available
    load_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats, shared_steps)

    # Sync shared network
    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    # Initialize optimized trainer
    trainer = OptimizedLSTMTrainer(online_network, target_network, optimizer, scaler)

    # Initialize tracking variables
    previous_alltime_min = None
    time_last_save = time.perf_counter()
    save_frequency_s = getattr(config_copy, 'save_frequency_s', 5 * 60)

    queue_check_order = list(range(len(rollout_queues)))
    rollout_queue_readers = [q._reader for q in rollout_queues]

    # Timing tracking
    time_waited_for_workers_since_last_tensorboard_write = 0
    time_training_since_last_tensorboard_write = 0
    time_testing_since_last_tensorboard_write = 0

    # Initialize stats
    if "rolling_mean_ms" not in accumulated_stats:
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats.get(
        "cumul_number_single_memories_used", 0)
    transitions_learned_last_save = accumulated_stats.get("cumul_number_single_memories_used", 0)

    # Setup optimized LSTM replay buffer
    buffer = LSTMReplayBuffer(
        capacity=lstm_config_copy.replay_buffer_capacity,
        seq_len=lstm_config_copy.lstm_seq_len,
        use_prioritized=getattr(lstm_config_copy, 'use_prioritized_replay', False),
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

    print(f"Optimized LSTM replay buffer capacity: {lstm_config_copy.replay_buffer_capacity}")
    print(f"LSTM sequence length: {lstm_config_copy.lstm_seq_len}")
    print(f"Batch size: {lstm_config_copy.lstm_batch_size}")
    print(f"Gradient accumulation steps: {trainer.use_gradient_accumulation}")
    print(f"Mixed precision: {trainer.use_mixed_precision}")
    print(f"Sequence masking: {trainer.use_sequence_masking}")
    print(f"Temporal weighting: {trainer.use_temporal_weighting}")
    print(f"Boltzmann exploration enabled: epsilon={config_values['epsilon']:.4f}, epsilon_boltzmann={config_values['epsilon_boltzmann']:.4f}, tau={config_values['tau_epsilon_boltzmann']:.4f}")

    step = accumulated_stats.get("training_step", 0)
    print(f"Starting optimized LSTM training from step {step}")

    # ========================================================
    # Optimized Main Training Loop
    # ========================================================
    while True:
        # Wait for data from collector processes
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time

        if time_waited > 1:
            print(f"Warning: learner waited {time_waited:.2f} seconds for workers")

        time_waited_for_workers_since_last_tensorboard_write += time_waited

        # Get rollout data
        rollout_data, idx = get_rollout_from_queues(rollout_queues, queue_check_order)

        if rollout_data:
            (rollout_results, end_race_stats, fill_buffer, is_explo, 
             map_name, map_status, rollout_duration, loop_number) = rollout_data

        # Reload config periodically
        if step % 100 == 0:
            importlib.reload(config_copy)
            importlib.reload(lstm_config_copy)

        config_values = update_config_parameters(accumulated_stats)

        # Update tensorboard writer if needed
        if config_values["tensorboard_suffix"] != tensorboard_suffix:
            tensorboard_suffix = config_values["tensorboard_suffix"]
            tensorboard_writer = get_tensorboard_writer(
                tensorboard_base_dir, config_copy, tensorboard_suffix
            )

        # Update optimizer parameters
        update_optimizer_params(optimizer, config_values["learning_rate"])

        # Process rollout data
        if rollout_data:
            accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])
            shared_steps.value = accumulated_stats["cumul_number_frames_played"]

            # Log race statistics
            race_stats = collect_race_stats(rollout_results, end_race_stats, is_explo, 
                                          map_name, map_status, rollout_duration, accumulated_stats)
            log_race_stats_to_tensorboard(tensorboard_writer, race_stats, accumulated_stats)

            # Save good runs
            save_good_runs(base_dir, save_dir, rollout_results, end_race_stats, 
                          map_name, is_explo, accumulated_stats)

            # Process rollout with optimizations
            if fill_buffer:
                buffer, episode_length = optimized_process_rollout_lstm(
                    rollout_results, buffer, config_values["gamma"], accumulated_stats
                )

        # ===============================================
        #   Optimized LSTM Training Loop
        # ===============================================

        # Ensure model is in training mode
        if not online_network.training:
            online_network.train()

        # Dynamic training condition
        min_episodes_to_start = max(1, lstm_config_copy.lstm_batch_size // 2)
        should_train = (
            len(buffer) >= min_episodes_to_start and
            accumulated_stats["cumul_number_single_memories_used"] <=
            accumulated_stats["cumul_number_single_memories_should_have_been_used"]
        )

        if not should_train and len(buffer) > 0:
            if len(buffer) < min_episodes_to_start:
                print(f"[Not training] Buffer too small: {len(buffer)} < {min_episodes_to_start}")

        # Optimized training loop
        training_steps_this_iteration = 0
        max_training_steps_per_iteration = getattr(lstm_config_copy, 'max_training_steps_per_iteration', 10)
        
        while should_train and training_steps_this_iteration < max_training_steps_per_iteration:

            train_start_time = time.perf_counter()

            try:
                # Sample batch with optimized strategy
                batch_data = buffer.sample_batch(lstm_config_copy.lstm_batch_size)
                
                # Move batch to GPU efficiently
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].to("cuda", non_blocking=True)

                # Optimized training step
                stats = trainer.train_step(batch_data, step)

                # Periodic memory cleanup
                if step % trainer.cleanup_frequency == 0:
                    trainer.cleanup_memory()

                # Update memory usage tracking
                sequences_used = lstm_config_copy.lstm_batch_size * lstm_config_copy.lstm_seq_len
                accumulated_stats["cumul_number_single_memories_used"] += sequences_used

                # Log training step
                if step % lstm_config_copy.log_frequency == 0:  # Reduce logging frequency
                    log_training_step(step, stats, len(buffer))

                # Update timing stats
                train_duration_ms = (time.perf_counter() - train_start_time) * 1000
                train_on_batch_duration_history.append(train_duration_ms / 1000)
                time_training_since_last_tensorboard_write += time.perf_counter() - train_start_time

                # Update counters
                accumulated_stats["cumul_number_batches_done"] += 1

                # Update rolling statistics
                key = "train_step"
                prev_mean = accumulated_stats["rolling_mean_ms"].get(key, 0)
                alpha = 0.01
                accumulated_stats["rolling_mean_ms"][key] = (1 - alpha) * prev_mean + alpha * train_duration_ms

                # Track minimum times
                if key not in accumulated_stats["alltime_min_ms"] or train_duration_ms < accumulated_stats["alltime_min_ms"][key]:
                    accumulated_stats["alltime_min_ms"][key] = train_duration_ms

                # Add to histories
                if not math.isinf(stats["loss"]):
                    loss_history.append(stats["loss"])
                if not math.isinf(stats["grad_norm"]):
                    grad_norm_history.append(stats["grad_norm"])

                # Apply weight decay
                utilities.custom_weight_decay(online_network, 1 - config_values["weight_decay"])

                # Update shared network periodically
                if accumulated_stats["cumul_number_batches_done"] % config_copy.send_shared_network_every_n_batches == 0:
                    with shared_network_lock:
                        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

                # Update target network
                target_update_freq = getattr(lstm_config_copy, 'target_update_frequency', 1000)
                if accumulated_stats["cumul_number_batches_done"] % target_update_freq == 0:
                    target_network.load_state_dict(online_network.state_dict())
                    print(f"Updated target network at step {step}")

                # Log to TensorBoard
                if step % lstm_config_copy.tensorboard_log_frequency == 0:  # Reduce TensorBoard logging frequency
                    log_training_step_to_tensorboard(
                        tensorboard_writer, step, stats, len(buffer),
                        stats["max_next_q"], train_duration_ms,
                        accumulated_stats["rolling_mean_ms"]
                    )

                step += 1
                training_steps_this_iteration += 1

            except Exception as e:
                print(f"Error during training step: {e}")
                break

            # Check if we should continue training
            should_train = (
                len(buffer) >= min_episodes_to_start and
                accumulated_stats["cumul_number_single_memories_used"] <=
                accumulated_stats["cumul_number_single_memories_should_have_been_used"]
            )

        # Save checkpoint
        accumulated_stats["training_step"] = step
        utilities.save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats)

        # ===============================================
        #   Periodic Statistics and Checkpointing
        # ===============================================
        if time.perf_counter() - time_last_save >= save_frequency_s:
            accumulated_stats["cumul_training_hours"] += (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save

            # Collect enhanced statistics
            training_params = {
                "gamma": config_values["gamma"],
                "learning_rate": config_values["learning_rate"],
                "weight_decay": config_values["weight_decay"],
                "epsilon": config_values["epsilon"],
                "epsilon_boltzmann": config_values["epsilon_boltzmann"],
                "tau_epsilon_boltzmann": config_values["tau_epsilon_boltzmann"]
            }

            step_stats = collect_periodic_stats(
                accumulated_stats, loss_history, train_on_batch_duration_history,
                grad_norm_history, layer_grad_norm_history, buffer, training_params,
                time_waited_for_workers_since_last_tensorboard_write,
                time_training_since_last_tensorboard_write,
                time_testing_since_last_tensorboard_write,
                time_since_last_save, transitions_learned_last_save
            )

            # Add optimization-specific stats
            if trainer.forward_times:
                step_stats["avg_forward_time"] = np.mean(trainer.forward_times)
                step_stats["avg_backward_time"] = np.mean(trainer.backward_times)

            # Reset timing counters
            time_waited_for_workers_since_last_tensorboard_write = 0
            time_training_since_last_tensorboard_write = 0
            time_testing_since_last_tensorboard_write = 0
            time_last_save = time.perf_counter()
            transitions_learned_last_save = accumulated_stats["cumul_number_single_memories_used"]

            # Log detailed stats
            previous_alltime_min = log_detailed_tensorboard_stats(
                tensorboard_writer, online_network, optimizer, accumulated_stats,
                step_stats, previous_alltime_min, step
            )

            # Reset histories
            loss_history = []
            train_on_batch_duration_history = []
            grad_norm_history = []
            layer_grad_norm_history = defaultdict(list)

            # Save checkpoint
            accumulated_stats["training_step"] = step
            save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats)

            print(f"Optimized LSTM Buffer: {len(buffer)} episodes, "
                  f"Avg forward time: {np.mean(trainer.forward_times):.4f}s, "
                  f"Avg backward time: {np.mean(trainer.backward_times):.4f}s")


def update_config_parameters(accumulated_stats):
    """Update parameters based on current config and training progress."""
    current_step = accumulated_stats["cumul_number_frames_played"]

    learning_rate = utilities.from_exponential_schedule(config_copy.lr_schedule, current_step)
    weight_decay = config_copy.weight_decay_lr_ratio * learning_rate
    gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, current_step)
    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule, current_step
    )
    
    # Add Boltzmann exploration parameters
    epsilon = utilities.from_exponential_schedule(config_copy.epsilon_schedule, current_step)
    epsilon_boltzmann = utilities.from_exponential_schedule(config_copy.epsilon_boltzmann_schedule, current_step)

    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "tensorboard_suffix": tensorboard_suffix,
        "epsilon": epsilon,
        "epsilon_boltzmann": epsilon_boltzmann,
        "tau_epsilon_boltzmann": config_copy.tau_epsilon_boltzmann,
    }


def update_optimizer_params(optimizer, learning_rate):
    """Update optimizer parameters."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["epsilon"] = config_copy.adam_epsilon
        param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)
