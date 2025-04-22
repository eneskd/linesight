import copy
import importlib
import math
import random
import sys
import time
import typing
from collections import defaultdict, deque
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchrl.data.replay_buffers import PrioritizedSampler

from config_files import config_copy
from config_files import lstm_config_copy
from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents.lstm import make_untrained_lstm_agent
from trackmania_rl.multiprocess.buffer_lstm import LSTMReplayBuffer


# Helper function to check tensors for numerical issues
def check_tensor_for_issues(tensor, name):
    """Check if tensor contains NaN or Inf values."""
    if tensor is None:
        return False
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"WARNING: Inf detected in {name}")
        return True
    return False


def dqn_loss(q_values, actions, targets, weights=None):
    """
    Standard DQN loss: MSE between Q(s,a) and target, with optional importance sampling weights.

    Args:
        q_values: (batch, seq_len, n_actions) Q-values from the network
        actions: (batch, seq_len) int64 actions that were taken
        targets: (batch, seq_len) float32 target values
        weights: (batch,) importance sampling weights for prioritized replay

    Returns:
        Scalar loss value and TD errors for priority updates
    """
    q_selected = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

    # Calculate TD error
    td_error = q_selected - targets

    # If weights are provided, apply them to the squared error
    if weights is not None:
        # Element-wise multiplication with weights before mean
        return (weights.unsqueeze(1) * td_error.pow(2)).mean(), td_error.detach().abs()
    else:
        # Standard MSE loss
        return td_error.pow(2).mean(), td_error.detach().abs()


def setup_tensorboard(tensorboard_base_dir, layout_version="lay_lstm"):
    """Setup TensorBoard with custom layout for LSTM agent."""
    SummaryWriter(log_dir=str(tensorboard_base_dir / layout_version)).add_custom_scalars(
        {
            layout_version: {
                "loss": ["Multiline", ["loss$", "loss_test$"]],
                "avg_Q": ["Multiline", ["avg_Q"]],
                "grad_norm_history": ["Multiline", ["grad_norm_history_d9", "grad_norm_history_d98"]],
                "priorities": ["Multiline", ["priorities"]],
                "sequence": ["Multiline", ["sequence_length", "sequence_overlap"]],
            },
        }
    )


def initialize_networks_and_optimizer(accumulated_stats):
    """Initialize models and optimizer."""
    online_network, uncompiled_online_network = make_untrained_lstm_agent(
        jit=config_copy.use_jit,
        is_inference=False,
    )
    target_network, _ = make_untrained_lstm_agent(
        jit=config_copy.use_jit,
        is_inference=False,
    )

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

    # Initialize learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Since we want to maximize reward
        factor=getattr(lstm_config_copy, "lr_reduction_factor", 0.5),
        patience=getattr(lstm_config_copy, "lr_patience", 5000),
        verbose=True,
        threshold=getattr(lstm_config_copy, "lr_threshold", 0.01),
        min_lr=getattr(lstm_config_copy, "min_lr", 1e-6)
    )

    scaler = torch.amp.GradScaler("cuda")

    return online_network, uncompiled_online_network, target_network, optimizer, lr_scheduler, scaler


def load_checkpoint(save_dir, online_network, target_network, optimizer, lr_scheduler, scaler, accumulated_stats,
                    shared_steps):
    """Load checkpoint if available."""
    try:
        online_network.load_state_dict(torch.load(f=save_dir / "lstm_weights1.torch", weights_only=False))
        target_network.load_state_dict(torch.load(f=save_dir / "lstm_weights2.torch", weights_only=False))
        print(" =====================     Learner LSTM weights loaded !     ============================")

        try:
            optimizer.load_state_dict(torch.load(f=save_dir / "lstm_optimizer.torch", weights_only=False))

            # Try to load scheduler state if it exists
            scheduler_state = torch.load(save_dir / "lstm_scheduler.torch")
            lr_scheduler.load_state_dict(scheduler_state)
            print(" =====================     Learner LSTM scheduler loaded !     ============================")

            scaler.load_state_dict(torch.load(f=save_dir / "lstm_scaler.torch", weights_only=False))
            print(
                " =====================     Learner LSTM optimizer and scaler loaded !     ============================")
        except Exception as e:
            print(f" Could not load optimizer, scheduler, or scaler: {e}")

        try:
            stats = joblib.load(save_dir / "lstm_accumulated_stats.joblib")
            accumulated_stats.update(stats)
            shared_steps.value = accumulated_stats["cumul_number_frames_played"]
            print(" =====================      Learner LSTM stats loaded !      ============================")
        except Exception as e:
            print(f" Learner LSTM could not load stats: {e}")

        return True
    except Exception as e:
        print(f" Learner LSTM could not load weights: {e}")

    # Try to load consolidated checkpoint
    checkpoint_files = list(save_dir.glob("lstm_model_step_*.pt"))

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
        try:
            checkpoint = torch.load(latest_checkpoint)
            online_network.load_state_dict(checkpoint['online_network'])
            target_network.load_state_dict(checkpoint['target_network'])
            accumulated_stats.update(checkpoint['stats'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])

            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            shared_steps.value = accumulated_stats["cumul_number_frames_played"]
            print(f" ===================== Loaded checkpoint from {latest_checkpoint} ============================")
            print(
                f" ===================== Resumed from step {accumulated_stats['cumul_number_frames_played']} ============================")
            return True
        except Exception as e:
            print(f" Error loading checkpoint from {latest_checkpoint}: {e}")
            return False

    return False


def get_rollout_from_queues(rollout_queues, queue_check_order, step):
    """Get rollout data from worker queues."""
    for idx in queue_check_order:
        if not rollout_queues[idx].empty():
            rollout_data = rollout_queues[idx].get()
            # Move this queue to the end for round-robin queue checking
            queue_check_order.append(queue_check_order.pop(queue_check_order.index(idx)))

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

            print(f"\n[Step {step}] Received rollout from queue {idx}:")
            print(
                f"  Map: {map_name}, Explo: {is_explo}, Loop: {loop_number}, Rollout duration: {rollout_duration:.2f}s")
            print(
                f"  Rollout stats: status={map_status}, end_race_stats_keys={list(end_race_stats.keys()) if end_race_stats and end_race_stats.get('race_finished', False) else None}")
            print(f"  fill_buffer stats: status={fill_buffer}")

            return rollout_data, idx

    return None, -1


def update_config_parameters(accumulated_stats):
    """Update parameters based on current config and training progress."""
    current_step = accumulated_stats["cumul_number_frames_played"]

    # Get current learning rate and gamma from schedules
    learning_rate = utilities.from_exponential_schedule(config_copy.lr_schedule, current_step)
    weight_decay = config_copy.weight_decay_lr_ratio * learning_rate
    gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, current_step)

    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gamma": gamma,
    }


def update_buffer_parameters(buffer):
    """Update buffer settings."""
    if hasattr(buffer, "use_prioritized") and buffer.use_prioritized:
        buffer.alpha = getattr(lstm_config_copy, "prio_alpha", 0.6)
        buffer.beta = getattr(lstm_config_copy, "prio_beta", 0.4)
        buffer.beta_annealing = getattr(lstm_config_copy, "prio_beta_annealing", 0.001)


def update_optimizer_params(optimizer, learning_rate):
    """Update optimizer parameters."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["epsilon"] = config_copy.adam_epsilon
        param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)


def process_rollout(rollout_results, buffer, gamma):
    """Process rollout data and fill buffer with transitions."""
    # Compute rewards
    n_frames = len(rollout_results["frames"])
    rewards = np.zeros(n_frames)
    for i in range(1, n_frames):
        meters_advanced = rollout_results["meters_advanced_along_centerline"][i] - \
                          rollout_results["meters_advanced_along_centerline"][i - 1]
        ms_elapsed = config_copy.ms_per_action if (i < n_frames - 1 or ("race_time" not in rollout_results)) else \
        rollout_results["race_time"] - (n_frames - 2) * config_copy.ms_per_action

        # Calculate reward
        rewards[i] = (
                config_copy.constant_reward_per_ms * ms_elapsed
                + config_copy.reward_per_m_advanced_along_centerline * meters_advanced
        )

    # Check if we have explicit "done" flags, otherwise assume only the last state is terminal
    dones = rollout_results.get("dones", np.zeros(n_frames, dtype=bool))
    if not np.any(dones) and n_frames > 0:
        dones[-1] = True  # Mark last state as terminal if no explicit dones

    # Create gammas with 0 for terminal states
    gammas_tensor = np.ones(n_frames) * gamma
    gammas_tensor[dones] = 0.0  # Zero out gamma for terminal states

    # Prepare episode data for LSTM buffer
    episode = {
        "state_imgs": rollout_results["frames"],
        "state_floats": rollout_results["state_float"],
        "actions": rollout_results["actions"],
        "rewards": rewards,
        "next_state_imgs": rollout_results.get("next_frames",
                                               rollout_results["frames"][1:] + [rollout_results["frames"][-1]]),
        "next_state_floats": rollout_results.get("next_state_float", rollout_results["state_float"][1:] + [
            rollout_results["state_float"][-1]]),
        "gammas": gammas_tensor,
        "dones": dones.astype(int)
    }

    # Check if episode is too large for buffer
    if n_frames > buffer.capacity:
        print(f"WARNING: Episode length ({n_frames}) exceeds buffer capacity ({buffer.capacity}). Trimming episode.")
        # Trim the episode to fit in buffer
        max_frames = buffer.capacity - 1  # Leave room for buffer overhead

        # Trim all episode components
        for key in episode:
            if isinstance(episode[key], list):
                episode[key] = episode[key][:max_frames]
            elif isinstance(episode[key], np.ndarray):
                episode[key] = episode[key][:max_frames]

        print(f"  Episode trimmed to {max_frames} frames")
        n_frames = max_frames

    # Add episode to buffer
    buffer.add_episode(episode)
    print(f"  Added episode to replay buffer. Buffer size: {len(buffer)} / {buffer.capacity}")

    return buffer, n_frames


def sample_batch_from_buffer(buffer, batch_size):
    """Sample a batch from the replay buffer."""
    batch = buffer.sample_batch(batch_size)

    # Move batch data to GPU
    state_img_seq = batch["state_imgs"].to("cuda")  # (batch, seq_len, 1, H, W)
    state_float_seq = batch["state_floats"].to("cuda")  # (batch, seq_len, float_input_dim)
    actions_seq = batch["actions"].to("cuda").long()  # (batch, seq_len)
    rewards_seq = batch["rewards"].to("cuda")  # (batch, seq_len)
    next_state_img_seq = batch["next_state_imgs"].to("cuda")  # (batch, seq_len, 1, H, W)
    next_state_float_seq = batch["next_state_floats"].to("cuda")  # (batch, seq_len, float_input_dim)
    gammas_seq = batch["gammas"].to("cuda")  # (batch, seq_len)

    # Extract prioritized replay specific components if present
    weights = batch.get("weights", None)
    indices = batch.get("indices", None)

    if weights is not None:
        weights = weights.to("cuda")

    return {
        "state_img_seq": state_img_seq,
        "state_float_seq": state_float_seq,
        "actions_seq": actions_seq,
        "rewards_seq": rewards_seq,
        "next_state_img_seq": next_state_img_seq,
        "next_state_float_seq": next_state_float_seq,
        "gammas_seq": gammas_seq,
        "weights": weights,
        "indices": indices,
        "dones_seq": batch.get("dones", torch.zeros_like(rewards_seq)).to("cuda")
    }


def training_step(online_network, target_network, optimizer, scaler, batch_data, use_double_q=True, buffer=None):
    """Perform a single training step."""
    # Unpack batch data
    state_img_seq = batch_data["state_img_seq"]
    state_float_seq = batch_data["state_float_seq"]
    actions_seq = batch_data["actions_seq"]
    rewards_seq = batch_data["rewards_seq"]
    next_state_img_seq = batch_data["next_state_img_seq"]
    next_state_float_seq = batch_data["next_state_float_seq"]
    gammas_seq = batch_data["gammas_seq"]
    weights = batch_data["weights"]
    indices = batch_data["indices"]
    dones_seq = batch_data["dones_seq"]

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        # Initialize hidden states for both networks
        batch_size = state_img_seq.size(0)
        h_0 = torch.zeros(lstm_config_copy.lstm_num_layers, batch_size, lstm_config_copy.lstm_hidden_dim).to("cuda")
        c_0 = torch.zeros(lstm_config_copy.lstm_num_layers, batch_size, lstm_config_copy.lstm_hidden_dim).to("cuda")
        hidden_online = (h_0, c_0)
        hidden_target = (h_0.clone(), c_0.clone())

        # Process sequences through online network, maintaining hidden state properly
        q_values_list = []

        # Process each timestep in the sequence while maintaining hidden states
        for t in range(state_img_seq.size(1)):  # iterate through sequence length
            # Get current timestep data
            curr_state_img = state_img_seq[:, t]  # [batch_size, 1, H, W]
            curr_state_float = state_float_seq[:, t]  # [batch_size, float_dim]

            # Forward pass for this timestep
            q_t, hidden_online = online_network(curr_state_img, curr_state_float, hidden_online)
            q_values_list.append(q_t)

        # Stack the outputs to get the full sequence output
        q_values = torch.stack(q_values_list, dim=1)  # [batch_size, seq_len, n_actions]

        with torch.no_grad():
            if use_double_q:
                # Double Q-learning implementation

                # First, get next state Q-values from online network for action selection
                q_next_online_list = []
                hidden_online_next = (h_0.clone(), c_0.clone())  # Fresh hidden state for next states

                for t in range(next_state_img_seq.size(1)):
                    curr_next_state_img = next_state_img_seq[:, t]
                    curr_next_state_float = next_state_float_seq[:, t]

                    q_next_online_t, hidden_online_next = online_network(
                        curr_next_state_img, curr_next_state_float, hidden_online_next)
                    q_next_online_list.append(q_next_online_t)

                # Stack to get full sequence output from online network for next states
                q_next_online = torch.stack(q_next_online_list, dim=1)

                # Select actions using online network (argmax)
                next_actions = q_next_online.max(dim=2)[1].unsqueeze(2)  # [batch_size, seq_len, 1]

                # Process sequences through target network for evaluation
                q_next_list = []

                for t in range(next_state_img_seq.size(1)):
                    curr_next_state_img = next_state_img_seq[:, t]
                    curr_next_state_float = next_state_float_seq[:, t]

                    q_next_t, hidden_target = target_network(
                        curr_next_state_img, curr_next_state_float, hidden_target)
                    q_next_list.append(q_next_t)

                # Stack to get full sequence output from target network
                q_next = torch.stack(q_next_list, dim=1)

                # Gather Q-values for the actions selected by online network
                q_next_selected = q_next.gather(2, next_actions).squeeze(2)

                # Apply gamma only to non-terminal states
                future_rewards = gammas_seq * q_next_selected * (1.0 - dones_seq)
                targets = rewards_seq + future_rewards
                max_next_q = q_next_selected  # For logging
            else:
                # Standard Q-learning implementation
                q_next_list = []

                for t in range(next_state_img_seq.size(1)):
                    curr_next_state_img = next_state_img_seq[:, t]
                    curr_next_state_float = next_state_float_seq[:, t]

                    q_next_t, hidden_target = target_network(
                        curr_next_state_img, curr_next_state_float, hidden_target)
                    q_next_list.append(q_next_t)

                q_next = torch.stack(q_next_list, dim=1)
                max_next_q = q_next.max(dim=2)[0]

                # Apply gamma only to non-terminal states
                future_rewards = gammas_seq * max_next_q * (1.0 - dones_seq)
                targets = rewards_seq + future_rewards

        # Calculate loss with proper weight handling
        loss, td_errors = dqn_loss(q_values, actions_seq, targets, weights)

        # Update priorities if using prioritized replay
        if indices is not None and buffer is not None and hasattr(buffer, "update_priorities"):
            with torch.no_grad():
                # Use absolute TD errors as priorities
                new_priorities = td_errors.mean(dim=1).cpu().numpy()  # Mean across sequence dimension
                buffer.update_priorities(indices, new_priorities)

    # Check for numerical issues in tensors
    has_numerical_issues = False
    has_numerical_issues |= check_tensor_for_issues(q_values, "q_values")
    has_numerical_issues |= check_tensor_for_issues(q_next, "q_next")

    if use_double_q:
        has_numerical_issues |= check_tensor_for_issues(q_next_online, "q_next_online")
        has_numerical_issues |= check_tensor_for_issues(q_next_selected, "q_next_selected")
    else:
        has_numerical_issues |= check_tensor_for_issues(max_next_q, "max_next_q")

    has_numerical_issues |= check_tensor_for_issues(targets, "targets")
    has_numerical_issues |= check_tensor_for_issues(loss, "loss")
    has_numerical_issues |= check_tensor_for_issues(td_errors, "td_errors")

    # Skip optimization if numerical issues detected
    if has_numerical_issues:
        print("  WARNING: Numerical issues detected. Skipping optimization for this batch.")
        grad_norm = 0
        grad_has_issues = True
    else:
        # Backpropagation with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Check for gradient issues after unscaling
        grad_has_issues = False
        for name, param in online_network.named_parameters():
            if param.grad is not None:
                if check_tensor_for_issues(param.grad, f"{name}.grad"):
                    grad_has_issues = True
                    break

        if grad_has_issues:
            print("  WARNING: Numerical issues in gradients. Skipping parameter update.")
            grad_norm = 0
        else:
            # Apply gradient clipping
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
        "grad_norm": grad_norm if not grad_has_issues else 0,
        "max_next_q": max_next_q.mean().detach().cpu().item(),
        "reward_mean": rewards_seq.mean().detach().cpu().item(),
        "reward_min": rewards_seq.min().detach().cpu().item(),
        "reward_max": rewards_seq.max().detach().cpu().item(),
        "reward_std": rewards_seq.std().detach().cpu().item(),
        "has_numerical_issues": has_numerical_issues,
        "grad_has_issues": grad_has_issues,
    }

    return stats, q_values


def log_training_step(step, stats, buffer_size, use_double_q=True):
    """Log training step information to console."""
    print(f"  Training step {step}")
    print(f"    Loss: {stats['loss']:.6f}")
    print(f"    Q value mean/max/min: {stats['mean_q']:.3f} / {stats['max_q']:.3f} / {stats['min_q']:.3f}")
    print(f"    Grad norm: {stats['grad_norm']:.3f}")
    print(f"    Buffer size: {buffer_size}")
    print(f"    Rewards - mean: {stats['reward_mean']:.6f}, min: {stats['reward_min']:.6f}, "
          f"max: {stats['reward_max']:.6f}, std: {stats['reward_std']:.6f}")

    if use_double_q:
        print(f"    Double Q-learning: Target net selected Q mean: {stats['max_next_q']:.3f}")
    else:
        print(f"    Standard Q-learning: Target net max_next_q mean: {stats['max_next_q']:.3f}")


def log_to_tensorboard(tensorboard_writer, step, stats, buffer_size, train_duration_ms,
                       end_race_stats=None, rolling_means=None):
    """Log training metrics to TensorBoard."""
    # Basic training metrics
    tensorboard_writer.add_scalar('train/loss', stats['loss'], step)
    tensorboard_writer.add_scalar('train/q_values_mean', stats['mean_q'], step)
    tensorboard_writer.add_scalar('train/q_values_max', stats['max_q'], step)
    tensorboard_writer.add_scalar('train/q_values_min', stats['min_q'], step)
    tensorboard_writer.add_scalar('train/buffer_size', buffer_size, step)
    tensorboard_writer.add_scalar('train/grad_norm', stats['grad_norm'], step)
    tensorboard_writer.add_scalar('train/max_next_q', stats['max_next_q'], step)
    tensorboard_writer.add_scalar('train/train_step_ms', train_duration_ms, step)

    # Log numerical issues
    if stats.get('has_numerical_issues', False):
        tensorboard_writer.add_scalar('errors/numerical_issues', 1, step)
    if stats.get('grad_has_issues', False):
        tensorboard_writer.add_scalar('errors/gradient_issues', 1, step)

    # Log rolling means if provided
    if rolling_means:
        for key, value in rolling_means.items():
            tensorboard_writer.add_scalar(f'rolling_mean_ms/{key}', value, step)

    # Log end race stats if available
    if end_race_stats and end_race_stats.get('race_finished', False):
        for k, v in end_race_stats.items():
            if isinstance(v, (int, float)):
                tensorboard_writer.add_scalar(f'race/{k}', v, step)
                print(f"    End race stat: {k} = {v}")


def save_checkpoint(save_dir, online_network, target_network, optimizer, lr_scheduler, scaler, accumulated_stats, step,
                    current_avg_reward=None):
    """Save checkpoint with model weights, optimizer state, and statistics."""
    # Save weights
    torch.save(online_network.state_dict(), save_dir / "lstm_weights1.torch")
    torch.save(target_network.state_dict(), save_dir / "lstm_weights2.torch")
    torch.save(optimizer.state_dict(), save_dir / "lstm_optimizer.torch")
    torch.save(lr_scheduler.state_dict(), save_dir / "lstm_scheduler.torch")
    torch.save(scaler.state_dict(), save_dir / "lstm_scaler.torch")

    # Also save a versioned checkpoint for reference
    save_path = save_dir / f"lstm_model_step_{step}.pt"
    torch.save({
        'online_network': online_network.state_dict(),
        'target_network': target_network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step,
        'stats': accumulated_stats,
        'avg_reward': current_avg_reward
    }, save_path)

    # Save stats
    accumulated_stats["cumul_number_frames_played"] = step
    accumulated_stats["last_step"] = step

    joblib.dump(accumulated_stats, save_dir / "lstm_accumulated_stats.joblib")

    print(f"    Model checkpoint and stats saved at step {step}")
    return True


def update_target_network(target_network, online_network):
    """Update target network."""
    target_network.load_state_dict(online_network.state_dict())
    return True
