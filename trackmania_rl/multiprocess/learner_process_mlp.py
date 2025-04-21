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


def dqn_loss(q_values, actions, targets, weights=None):
    """
    Standard DQN loss: MSE between Q(s,a) and target, with optional importance sampling weights.

    Args:
        q_values: (batch_size, n_actions) Q-values from the network
        actions: (batch_size,) int64 actions that were taken
        targets: (batch_size,) float32 target values
        weights: (batch_size,) importance sampling weights for prioritized replay

    Returns:
        Scalar loss value and TD errors for priority updates
    """
    q_selected = q_values.gather(1, actions.unsqueeze(1).long()).squeeze(1)

    # Calculate TD error
    td_error = q_selected - targets

    # If weights are provided, apply them to the squared error
    if weights is not None:
        # Element-wise multiplication with weights before mean
        return (weights * td_error.pow(2)).mean(), td_error.detach().abs()
    else:
        # Standard MSE loss
        return td_error.pow(2).mean(), td_error.detach().abs()


def setup_tensorboard(tensorboard_base_dir, layout_version="lay_mono"):
    """Setup TensorBoard with custom layout."""
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


def load_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats, shared_steps):
    """Load checkpoint if available."""

    try:
        online_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
        target_network.load_state_dict(torch.load(f=save_dir / "weights2.torch", weights_only=False))
        print(" =====================     Learner weights loaded !     ============================")

        try:
            optimizer.load_state_dict(torch.load(f=save_dir / "optimizer1.torch", weights_only=False))
            scaler.load_state_dict(torch.load(f=save_dir / "scaler.torch", weights_only=False))
            print(" =========================     Optimizer loaded !     ================================")
        except Exception as e:
            print(f" Could not load optimizer: {e}")

        try:
            stats = joblib.load(save_dir / "accumulated_stats.joblib")
            accumulated_stats.update(stats)
            shared_steps.value = accumulated_stats["cumul_number_frames_played"]
            print(" =====================      Learner stats loaded !      ============================")
        except Exception as e:
            print(f" Learner could not load stats: {e}")

        return True
    except Exception as e:
        print(f" Learner could not load weights: {e}")


    checkpoint_files = list(save_dir.glob("checkpoint_step_*.pt"))

    # Try to load consolidated checkpoint
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
        try:
            checkpoint = torch.load(latest_checkpoint)
            online_network.load_state_dict(checkpoint['online_network'])
            target_network.load_state_dict(checkpoint['target_network'])
            accumulated_stats.update(checkpoint['stats'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            shared_steps.value = accumulated_stats["cumul_number_frames_played"]
            print(f" ===================== Loaded checkpoint from {latest_checkpoint} ============================")
            print(
                f" ===================== Resumed from step {accumulated_stats['cumul_number_frames_played']} ============================")
        except Exception as e:
            print(f" Error loading checkpoint from {latest_checkpoint}: {e}")
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


def update_buffer_size(buffer, buffer_test, memory_size, new_memory_size, memory_size_start_learn,
                       new_memory_size_start_learn, offset_cumul_number_single_memories_used):
    """Update buffer size if needed."""
    if new_memory_size != memory_size:
        buffer, buffer_test = resize_buffers(buffer, buffer_test, new_memory_size)
        offset_adjustment = (
                                        new_memory_size_start_learn - memory_size_start_learn) * config_copy.number_times_single_memory_is_used_before_discard
        offset_cumul_number_single_memories_used += offset_adjustment
        return buffer, buffer_test, new_memory_size, new_memory_size_start_learn, offset_cumul_number_single_memories_used
    return buffer, buffer_test, memory_size, memory_size_start_learn, offset_cumul_number_single_memories_used


def update_optimizer_params(optimizer, learning_rate, buffer):
    """Update optimizer parameters and buffer settings."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["epsilon"] = config_copy.adam_epsilon
        param_group["betas"] = (config_copy.adam_beta1, config_copy.adam_beta2)

    # Update prioritized sampler parameters if applicable
    if isinstance(buffer._sampler, PrioritizedSampler):
        buffer._sampler._alpha = config_copy.prio_alpha
        buffer._sampler._beta = config_copy.prio_beta
        buffer._sampler._eps = config_copy.prio_epsilon


def process_rollout(rollout_results, buffer, buffer_test, gamma, accumulated_stats):
    """Process rollout data and fill buffer with transitions."""
    current_step = accumulated_stats["cumul_number_frames_played"]

    # Get reward coefficients from config
    engineered_speedslide_reward = utilities.from_staircase_schedule(
        config_copy.engineered_speedslide_reward_schedule, current_step)
    engineered_neoslide_reward = utilities.from_staircase_schedule(
        config_copy.engineered_neoslide_reward_schedule, current_step)
    engineered_kamikaze_reward = utilities.from_staircase_schedule(
        config_copy.engineered_kamikaze_reward_schedule, current_step)
    engineered_close_to_vcp_reward = utilities.from_staircase_schedule(
        config_copy.engineered_close_to_vcp_reward_schedule, current_step)

    # Fill buffer with transitions
    buffer, buffer_test, number_memories_added_train, number_memories_added_test = (
        buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
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
    )

    # Update accumulated stats
    accumulated_stats["cumul_number_memories_generated"] += number_memories_added_train + number_memories_added_test
    accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
            config_copy.number_times_single_memory_is_used_before_discard * number_memories_added_train
    )

    return buffer, buffer_test, number_memories_added_train


def sample_batch_from_buffer(buffer, batch_size):
    """Sample a batch from the replay buffer with error handling."""
    try:
        # Try improved dictionary-style sampling if available
        if hasattr(buffer, "sample_batch_with_improved_strategy"):
            batch = buffer.sample_batch_with_improved_strategy(batch_size)
            batch_info = {"indices": batch.pop("indices")} if "indices" in batch else {}

            # Extract tensors from batch dictionary
            state_img_tensor = batch["state_img_tensor"]
            state_float_tensor = batch["state_float_tensor"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_state_img_tensor = batch["next_state_img_tensor"]
            next_state_float_tensor = batch["next_state_float_tensor"]
            gammas_terminal = batch["gammas_terminal"]

            # Get weights for prioritized replay if available
            weights = batch.get("weights", torch.ones_like(rewards))
        else:
            # Standard tuple-style batch
            batch, batch_info = buffer.sample(batch_size, return_info=True)

            # Standard 7-tuple unpacking
            (
                state_img_tensor,
                state_float_tensor,
                actions,
                rewards,
                next_state_img_tensor,
                next_state_float_tensor,
                gammas_terminal,
            ) = batch

            # Default weights (no prioritization)
            weights = torch.ones_like(rewards)
    except Exception as e:
        # Fallback to standard sampling
        print(f"Warning: Advanced batch sampling failed with {e}, falling back to standard sampling")
        batch, batch_info = buffer.sample(batch_size, return_info=True)

        # Standard 7-tuple unpacking
        (
            state_img_tensor,
            state_float_tensor,
            actions,
            rewards,
            next_state_img_tensor,
            next_state_float_tensor,
            gammas_terminal,
        ) = batch

        # Default weights (no prioritization)
        weights = torch.ones_like(rewards)

    # Ensure image tensors are (batch, 1, H, W)
    if state_img_tensor.ndim == 3:
        state_img_tensor = state_img_tensor.unsqueeze(1)
    if next_state_img_tensor.ndim == 3:
        next_state_img_tensor = next_state_img_tensor.unsqueeze(1)

    return {
        "state_img_tensor": state_img_tensor,
        "state_float_tensor": state_float_tensor,
        "actions": actions,
        "rewards": rewards,
        "next_state_img_tensor": next_state_img_tensor,
        "next_state_float_tensor": next_state_float_tensor,
        "gammas_terminal": gammas_terminal,
        "weights": weights,
        "batch_info": batch_info
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


def log_training_step(step, stats, buffer_size):
    """Log training step information to console."""
    print(f"  Training step {step}")
    print(f"    Loss: {stats['loss']:.6f}")
    print(f"    Q value mean/max/min: {stats['mean_q']:.3f} / {stats['max_q']:.3f} / {stats['min_q']:.3f}")
    print(f"    Grad norm: {stats['grad_norm']:.3f}")
    print(f"    Buffer size: {buffer_size}")
    print(f"    Rewards - mean: {stats['reward_mean']:.6f}, min: {stats['reward_min']:.6f}, "
          f"max: {stats['reward_max']:.6f}, std: {stats['reward_std']:.6f}")
    print(f"    First 5 rewards: {stats['rewards'][:5].cpu().numpy().tolist() if 'rewards' in stats else None}")


def log_to_tensorboard(tensorboard_writer, step, stats, buffer_size, max_next_q_mean, train_duration_ms,
                       end_race_stats=None, rolling_means=None):
    """Log training metrics to TensorBoard."""
    # Basic training metrics
    tensorboard_writer.add_scalar('train/loss', stats['loss'], step)
    tensorboard_writer.add_scalar('train/q_values_mean', stats['mean_q'], step)
    tensorboard_writer.add_scalar('train/q_values_max', stats['max_q'], step)
    tensorboard_writer.add_scalar('train/q_values_min', stats['min_q'], step)
    tensorboard_writer.add_scalar('train/buffer_size', buffer_size, step)
    tensorboard_writer.add_scalar('train/grad_norm', stats['grad_norm'], step)
    tensorboard_writer.add_scalar('train/max_next_q', max_next_q_mean, step)
    tensorboard_writer.add_scalar('train/train_step_ms', train_duration_ms, step)

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


def log_detailed_tensorboard_stats(tensorboard_writer, online_network, optimizer, accumulated_stats, step_stats,
                                   previous_alltime_min, frame_count):
    """Log detailed statistics to TensorBoard periodically."""
    walltime_tb = time.time()

    # Log layer L2 norms
    for name, param in online_network.named_parameters():
        tensorboard_writer.add_scalar(
            tag=f"layer_{name}_L2",
            scalar_value=np.sqrt((param ** 2).mean().detach().cpu().item()),
            global_step=frame_count,
            walltime=walltime_tb,
        )

    # Log optimizer state
    try:
        assert len(optimizer.param_groups) == 1
        for p, (name, _) in zip(
                optimizer.param_groups[0]["params"],
                online_network.named_parameters(),
        ):
            state = optimizer.state[p]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            mod_lr = 1 / (exp_avg_sq.sqrt() + 1e-4)
            tensorboard_writer.add_scalar(
                tag=f"lr_ratio_{name}_L2",
                scalar_value=np.sqrt((mod_lr ** 2).mean().detach().cpu().item()),
                global_step=frame_count,
                walltime=walltime_tb,
            )
            tensorboard_writer.add_scalar(
                tag=f"exp_avg_{name}_L2",
                scalar_value=np.sqrt((exp_avg ** 2).mean().detach().cpu().item()),
                global_step=frame_count,
                walltime=walltime_tb,
            )
            tensorboard_writer.add_scalar(
                tag=f"exp_avg_sq_{name}_L2",
                scalar_value=np.sqrt((exp_avg_sq ** 2).mean().detach().cpu().item()),
                global_step=frame_count,
                walltime=walltime_tb,
            )
    except Exception:
        pass

    # Log all stats
    for k, v in step_stats.items():
        tensorboard_writer.add_scalar(
            tag=k,
            scalar_value=v,
            global_step=frame_count,
            walltime=walltime_tb,
        )

    # Log time summary comparison with previous mins
    previous_alltime_min = previous_alltime_min or copy.deepcopy(accumulated_stats["alltime_min_ms"])

    times_summary = f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} " + " ".join(
        [
            f"{'**' if v < previous_alltime_min.get(k, 99999999) else ''}{k}: {v / 1000:.2f}{'**' if v < previous_alltime_min.get(k, 99999999) else ''}"
            for k, v in accumulated_stats["alltime_min_ms"].items()
        ]
    )

    tensorboard_writer.add_text(
        "times_summary",
        times_summary,
        global_step=frame_count,
        walltime=walltime_tb,
    )

    return copy.deepcopy(accumulated_stats["alltime_min_ms"])


def save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats):
    """Save checkpoint with model weights, optimizer state, and statistics."""
    checkpoint_path = save_dir / f"checkpoint_step_{accumulated_stats['cumul_number_frames_played']}.pt"
    try:
        # Save consolidated checkpoint
        torch.save({
            'online_network': online_network.state_dict(),
            'target_network': target_network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'step': accumulated_stats['cumul_number_frames_played'],
            'stats': accumulated_stats
        }, checkpoint_path)

        # Also save standard checkpoint files for backward compatibility
        utilities.save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats)
        print(f"Checkpoint saved at step {accumulated_stats['cumul_number_frames_played']}")
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False




def collect_periodic_stats(accumulated_stats, loss_history, train_on_batch_duration_history, grad_norm_history,
                           layer_grad_norm_history, buffer, gamma, learning_rate, weight_decay,
                           time_waited_for_workers, time_training, time_testing, time_since_last_save,
                           transitions_learned_last_save):
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
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
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

    return step_stats, transitions_learned_per_second


def update_target_network(target_network, online_network, accumulated_stats):
    """Update target network if conditions are met."""
    if (accumulated_stats["cumul_number_single_memories_used"]
            >= accumulated_stats.get("cumul_number_single_memories_used_next_target_network_update", 0)):
        accumulated_stats["cumul_number_target_network_updates"] = accumulated_stats.get(
            "cumul_number_target_network_updates", 0) + 1
        accumulated_stats["cumul_number_single_memories_used_next_target_network_update"] = (
                accumulated_stats.get("cumul_number_single_memories_used_next_target_network_update", 0) +
                config_copy.number_memories_trained_on_between_target_network_updates
        )
        utilities.soft_copy_param(target_network, online_network, config_copy.soft_update_tau)
        return True
    return False


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
    tensorboard_writer = SummaryWriter(
        log_dir=str(tensorboard_base_dir / (config_copy.run_name + '_' + config_copy.agent_type + tensorboard_suffix))
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
    step = 0

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
        if config_values["tensorboard_suffix"] != tensorboard_suffix:
            tensorboard_suffix = config_values["tensorboard_suffix"]
            tensorboard_writer = SummaryWriter(
                log_dir=str(
                    tensorboard_base_dir / (config_copy.run_name + '_' + config_copy.agent_type + tensorboard_suffix))
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
            log_to_tensorboard(
                tensorboard_writer,
                step,
                stats,
                len(buffer),
                stats["max_next_q"],
                train_duration_ms,
                end_race_stats if rollout_data else None,
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

        # Save to files
        utilities.save_checkpoint(save_dir, online_network, target_network, optimizer, scaler)

        # ===============================================
        #   Periodic checkpoint and stat logging
        # ===============================================
        if time.perf_counter() - time_last_save >= save_frequency_s:
            accumulated_stats["cumul_training_hours"] += (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save

            # Collect statistics
            step_stats, transitions_learned_per_second = collect_periodic_stats(
                accumulated_stats,
                loss_history,
                train_on_batch_duration_history,
                grad_norm_history,
                layer_grad_norm_history,
                buffer,
                gamma,
                learning_rate,
                weight_decay,
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

            # Save checkpoint
            save_checkpoint(save_dir, online_network, target_network, optimizer, scaler, accumulated_stats)
