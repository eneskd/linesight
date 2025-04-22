"""
Common utility functions for learner processes.
This file contains shared functions used by both MLP and IQN learner processes.
"""

import copy
import math
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import PrioritizedSampler

from config_files import config_copy
from trackmania_rl import buffer_management, utilities
from trackmania_rl.map_reference_times import reference_times
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

    # Try to load consolidated checkpoint
    checkpoint_files = list(save_dir.glob("checkpoint_step_*.pt"))
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
            return True
        except Exception as e:
            print(f" Error loading checkpoint from {latest_checkpoint}: {e}")
            return False

    return False




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



def get_rollout_from_queues(rollout_queues, queue_check_order, step=None):
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

            step_info = f"\n[Step {step}] " if step is not None else ""
            print(f"{step_info}Received rollout from queue {idx}:")
            print(
                f"  Map: {map_name}, Explo: {is_explo}, Loop: {loop_number}, Rollout duration: {rollout_duration:.2f}s")
            print(
                f"  Rollout stats: status={map_status}, end_race_stats_keys={list(end_race_stats.keys()) if end_race_stats and end_race_stats.get('race_finished', False) else None}")
            print(f"  fill_buffer stats: status={fill_buffer}")

            return rollout_data, idx

    return None, -1


def process_rollout(rollout_results, buffer, buffer_test, gamma, accumulated_stats, engineered_rewards=None):
    """Process rollout data and fill buffer with transitions."""
    # Set default engineered rewards if not provided
    if engineered_rewards is None:
        current_step = accumulated_stats["cumul_number_frames_played"]
        engineered_rewards = {
            "engineered_speedslide_reward": utilities.from_staircase_schedule(
                config_copy.engineered_speedslide_reward_schedule, current_step),
            "engineered_neoslide_reward": utilities.from_staircase_schedule(
                config_copy.engineered_neoslide_reward_schedule, current_step),
            "engineered_kamikaze_reward": utilities.from_staircase_schedule(
                config_copy.engineered_kamikaze_reward_schedule, current_step),
            "engineered_close_to_vcp_reward": utilities.from_staircase_schedule(
                config_copy.engineered_close_to_vcp_reward_schedule, current_step),
        }

    # Fill buffer with transitions
    buffer, buffer_test, number_memories_added_train, number_memories_added_test = (
        buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
            buffer,
            buffer_test,
            rollout_results,
            config_copy.n_steps,
            gamma,
            config_copy.discard_non_greedy_actions_in_nsteps,
            engineered_rewards["engineered_speedslide_reward"],
            engineered_rewards["engineered_neoslide_reward"],
            engineered_rewards["engineered_kamikaze_reward"],
            engineered_rewards["engineered_close_to_vcp_reward"],
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






def save_good_runs(base_dir, save_dir, rollout_results, end_race_stats, map_name, is_explo, accumulated_stats):
    """Save good runs based on race time."""
    if end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"].get(map_name, 99999999999):
        # This is a new alltime_minimum
        accumulated_stats["alltime_min_ms"][map_name] = end_race_stats["race_time"]
        if accumulated_stats["cumul_number_frames_played"] > config_copy.frames_before_save_best_runs:
            name = f"{map_name}_{end_race_stats['race_time']}"
            utilities.save_run(
                base_dir,
                save_dir / "best_runs" / name,
                rollout_results,
                f"{name}.inputs",
                inputs_only=False,
            )

    if end_race_stats["race_time"] < config_copy.threshold_to_save_all_runs_ms:
        name = f"{map_name}_{end_race_stats['race_time']}_{datetime.now().strftime('%m%d_%H%M%S')}_{accumulated_stats['cumul_number_frames_played']}_{'explo' if is_explo else 'eval'}"
        utilities.save_run(
            base_dir,
            save_dir / "good_runs",
            rollout_results,
            f"{name}.inputs",
            inputs_only=True,
        )


def setup_tensorboard(tensorboard_base_dir, layout_version="lay_mono", custom_layout=None):
    """Setup TensorBoard with custom layout."""
    default_layout = {
        layout_version: {
            "loss": ["Multiline", ["loss$", "loss_test$"]],
            "avg_Q": ["Multiline", ["avg_Q"]],
            "grad_norm_history": ["Multiline", ["grad_norm_history_d9", "grad_norm_history_d98"]],
            "priorities": ["Multiline", ["priorities"]],
        },
    }

    layout = custom_layout if custom_layout else default_layout
    SummaryWriter(log_dir=str(tensorboard_base_dir / layout_version)).add_custom_scalars(layout)


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


def collect_periodic_stats(accumulated_stats, loss_history, train_on_batch_duration_history, grad_norm_history,
                           layer_grad_norm_history, buffer, training_params,
                           time_waited_for_workers, time_training, time_testing, time_since_last_save,
                           transitions_learned_last_save):
    """Collect statistics for periodic reporting and saving."""
    gamma = training_params.get("gamma", 0.99)
    learning_rate = training_params.get("learning_rate", 0.001)
    weight_decay = training_params.get("weight_decay", 0.0)
    epsilon = training_params.get("epsilon", 0.0)
    epsilon_boltzmann = training_params.get("epsilon_boltzmann", 0.0)

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

    # Add exploration parameters if available
    if epsilon is not None:
        step_stats["epsilon"] = epsilon
    if epsilon_boltzmann is not None:
        step_stats["epsilon_boltzmann"] = epsilon_boltzmann
        if hasattr(config_copy, "tau_epsilon_boltzmann"):
            step_stats["tau_epsilon_boltzmann"] = config_copy.tau_epsilon_boltzmann

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


def collect_race_stats(rollout_results, end_race_stats, is_explo, map_name, map_status, rollout_duration,
                       accumulated_stats):
    """Collect statistics from a completed race."""
    race_stats_to_write = {
        f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / (rollout_duration * 1000),
        f"explo_race_time_{map_status}_{map_name}" if is_explo else f"eval_race_time_{map_status}_{map_name}":
            end_race_stats[
                "race_time"
            ]
            / 1000,
        f"explo_race_finished_{map_status}_{map_name}" if is_explo else f"eval_race_finished_{map_status}_{map_name}":
            end_race_stats[
                "race_finished"
            ],
        f"mean_action_gap_{map_name}": -(
                np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1,
                                                                                                  initial=None).reshape(
            -1, 1)
        ).mean(),
        f"single_zone_reached_{map_status}_{map_name}": rollout_results["furthest_zone_idx"],
        "instrumentation__answer_normal_step": end_race_stats["instrumentation__answer_normal_step"],
        "instrumentation__answer_action_step": end_race_stats["instrumentation__answer_action_step"],
        "instrumentation__between_run_steps": end_race_stats["instrumentation__between_run_steps"],
        "instrumentation__grab_frame": end_race_stats["instrumentation__grab_frame"],
        "instrumentation__convert_frame": end_race_stats["instrumentation__convert_frame"],
        "instrumentation__grab_floats": end_race_stats["instrumentation__grab_floats"],
        "instrumentation__exploration_policy": end_race_stats["instrumentation__exploration_policy"],
        "instrumentation__request_inputs_and_speed": end_race_stats["instrumentation__request_inputs_and_speed"],
        "tmi_protection_cutoff": end_race_stats["tmi_protection_cutoff"],
        "worker_time_in_rollout_percentage": rollout_results["worker_time_in_rollout_percentage"],
    }

    if not is_explo:
        race_stats_to_write[f"avg_Q_{map_status}_{map_name}"] = np.mean(rollout_results["q_values"])

    if end_race_stats["race_finished"]:
        race_stats_to_write[f"{'explo' if is_explo else 'eval'}_race_time_finished_{map_status}_{map_name}"] = (
                end_race_stats["race_time"] / 1000
        )
        if not is_explo:
            accumulated_stats["rolling_mean_ms"][map_name] = (
                    accumulated_stats["rolling_mean_ms"].get(map_name,
                                                             config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms)
                    * 0.9
                    + end_race_stats["race_time"] * 0.1
            )

    if (
            (not is_explo)
            and end_race_stats["race_finished"]
            and end_race_stats["race_time"] < 1.02 * accumulated_stats["rolling_mean_ms"][map_name]
    ):
        race_stats_to_write[f"eval_race_time_robust_{map_status}_{map_name}"] = end_race_stats["race_time"] / 1000
        if map_name in reference_times:
            for reference_time_name in ["author", "gold"]:
                if reference_time_name in reference_times[map_name]:
                    reference_time = reference_times[map_name][reference_time_name]
                    race_stats_to_write[f"eval_ratio_{map_status}_{reference_time_name}_{map_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                    )
                    race_stats_to_write[f"eval_agg_ratio_{map_status}_{reference_time_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                    )

    for i in [0]:
        race_stats_to_write[f"q_value_{i}_starting_frame_{map_name}"] = end_race_stats[f"q_value_{i}_starting_frame"]

    if not is_explo:
        for i, split_time in enumerate(
                [
                    (e - s) / 1000
                    for s, e in zip(
                    end_race_stats["cp_time_ms"][:-1],
                    end_race_stats["cp_time_ms"][1:],
                )
                ]
        ):
            race_stats_to_write[f"split_{map_name}_{i}"] = split_time

    return race_stats_to_write


def log_race_stats_to_tensorboard(tensorboard_writer, race_stats_to_write, accumulated_stats):
    """Log race statistics to TensorBoard."""
    walltime_tb = time.time()
    for tag, value in race_stats_to_write.items():
        tensorboard_writer.add_scalar(
            tag=tag,
            scalar_value=value,
            global_step=accumulated_stats["cumul_number_frames_played"],
            walltime=walltime_tb,
        )

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


