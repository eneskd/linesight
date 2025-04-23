
"""
Utilities for TensorBoard logging in Trackmania AI training processes.
This module centralizes all TensorBoard-related functionality for different agent types.
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import PrioritizedSampler
from config_files import config_copy

from trackmania_rl.map_reference_times import reference_times


def setup_tensorboard(tensorboard_base_dir: Path) -> None:
    """
    Setup TensorBoard base directory.
    
    Args:
        tensorboard_base_dir: Base directory for TensorBoard logs
    """
    tensorboard_base_dir.mkdir(parents=True, exist_ok=True)


def get_tensorboard_writer(tensorboard_base_dir: Path, config, suffix: str) -> SummaryWriter:
    """
    Get a TensorBoard writer with the appropriate log directory.
    
    Args:
        tensorboard_base_dir: Base directory for TensorBoard logs
        config: Configuration object containing run_name and agent_type
        suffix: Suffix to append to the TensorBoard log directory
        
    Returns:
        A TensorBoard SummaryWriter instance
    """
    log_dir = str(tensorboard_base_dir / (config.run_name + '_' + config.agent_type + suffix))
    return SummaryWriter(log_dir=log_dir)


def log_training_step(step: int, stats: Dict[str, Any], buffer_size: int) -> None:
    """
    Log basic training step information to console.
    
    Args:
        step: Current training step
        stats: Dictionary of statistics from training step
        buffer_size: Current size of the replay buffer
    """
    loss = stats.get("loss", 0)
    grad_norm = stats.get("grad_norm", 0)
    print(f"Step {step}, Loss: {loss:.6f}, Grad norm: {grad_norm:.6f}, Buffer size: {buffer_size}")


def log_training_step_to_tensorboard(
    writer: SummaryWriter,
    step: int,
    stats: Dict[str, Any],
    buffer_size: int,
    max_next_q: Optional[float] = None,
    train_duration_ms: Optional[float] = None,
    rolling_means: Optional[Dict[str, float]] = None
) -> None:
    """
    Log training step metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        step: Current training step
        stats: Dictionary of statistics from training step
        buffer_size: Current size of the replay buffer
        max_next_q: Maximum Q-value for next state (optional)
        train_duration_ms: Duration of the training step in milliseconds (optional)
        rolling_means: Dictionary of rolling mean values for various metrics (optional)
    """
    # Basic training metrics
    writer.add_scalar("loss/loss", stats["loss"], step)
    
    # Q-values if available
    if "mean_q" in stats:
        writer.add_scalar("q_values/mean_q", stats["mean_q"], step)
    if "max_q" in stats:
        writer.add_scalar("q_values/max_q", stats["max_q"], step)
    if "min_q" in stats:
        writer.add_scalar("q_values/min_q", stats["min_q"], step)
    if max_next_q is not None:
        writer.add_scalar("q_values/max_next_q", max_next_q, step)
    
    # Reward metrics if available
    if "reward_mean" in stats:
        writer.add_scalar("rewards/mean", stats["reward_mean"], step)
    if "reward_min" in stats:
        writer.add_scalar("rewards/min", stats["reward_min"], step)
    if "reward_max" in stats:
        writer.add_scalar("rewards/max", stats["reward_max"], step)
    if "reward_std" in stats:
        writer.add_scalar("rewards/std", stats["reward_std"], step)


    if "quantile_huber_loss" in stats:
        writer.add_scalar("iqn/quantile_huber_loss", stats["quantile_huber_loss"], step)
    if "tau_mean" in stats:
        writer.add_scalar("iqn/tau_mean", stats["tau_mean"], step)
    if "tau_std" in stats:
        writer.add_scalar("iqn/tau_std", stats["tau_std"], step)
    if "tau_min" in stats:
        writer.add_scalar("iqn/tau_min", stats["tau_min"], step)
    if "tau_max" in stats:
        writer.add_scalar("iqn/tau_max", stats["tau_max"], step)


    # Training performance metrics
    if "grad_norm" in stats:
        writer.add_scalar("metrics/grad_norm", stats["grad_norm"], step)
    writer.add_scalar("metrics/buffer_size", buffer_size, step)
    
    # Log training duration if provided
    if train_duration_ms is not None:
        writer.add_scalar("metrics/train_step_ms", train_duration_ms, step)
    
    # Add rolling mean if available
    if rolling_means is not None and "train_step" in rolling_means:
        writer.add_scalar("metrics/train_step_ms_rolling", rolling_means["train_step"], step)


def log_to_tensorboard(writer: SummaryWriter, metrics: Dict[str, Any], step: int) -> None:
    """
    Log various metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        metrics: Dictionary of metrics to log
        step: Current training step
    """
    # Log all metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Group metrics by category based on key prefix
            if key.startswith("loss"):
                writer.add_scalar(f"loss/{key}", value, step)
            elif key.startswith("q_values"):
                writer.add_scalar(f"q_values/{key}", value, step)
            elif key.startswith("reward"):
                writer.add_scalar(f"rewards/{key}", value, step)
            elif key.startswith("grad_"):
                writer.add_scalar(f"gradients/{key}", value, step)
            elif key.startswith("priority"):
                writer.add_scalar(f"priorities/{key}", value, step)
            else:
                writer.add_scalar(f"metrics/{key}", value, step)



def log_race_stats_to_tensorboard(
        writer: SummaryWriter,
        race_stats: Dict[str, Any],
        accumulated_stats: Dict[str, Any]
) -> None:
    """
    Log race statistics to TensorBoard with organized categories.

    Args:
        writer: TensorBoard SummaryWriter instance
        race_stats: Dictionary of race statistics from collect_race_stats
        accumulated_stats: Dictionary of accumulated statistics
    """
    step = accumulated_stats["cumul_number_frames_played"]
    walltime = time.time()

    # Process all key-value pairs in race_stats
    for key, value in race_stats.items():
        # Skip non-numeric values and special keys
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue

        # Categorize metrics based on key prefixes and patterns
        tag = key

        # Determine the appropriate category based on key patterns
        if "race_time" in key:
            category = "race_times"
        elif "ratio" in key:
            category = "completion_ratios"
        elif key.startswith("explo_"):
            category = "exploration"
        elif key.startswith("eval_"):
            category = "evaluation"
        elif "q_value" in key:
            category = "q_values"
        elif "split_" in key:
            category = "splits"
        elif key.startswith("instrumentation__"):
            category = "instrumentation"
            tag = key.replace("instrumentation__", "")
        elif "worker_time" in key:
            category = "performance"
        else:
            category = "metrics"

        # Extract map name if present in the key
        map_specific = False
        map_name = None
        for part in key.split('_'):
            if part in reference_times:
                map_name = part
                map_specific = True
                break

        # Construct the tag with appropriate categorization
        if map_specific and map_name:
            # For map-specific metrics
            tag_path = f"{category}/by_map/{map_name}/{tag}"
        else:
            # For general metrics
            tag_path = f"{category}/{tag}"

        # Log to TensorBoard
        writer.add_scalar(tag_path, value, step, walltime=walltime)


    # Log aggregated metrics by map status if available
    for status in ["normal", "reverse", "mirror", "reversed_mirror"]:
        status_prefix = f"{status}_"
        status_metrics = {k.replace(status_prefix, ''): v for k, v in race_stats.items() if status_prefix in k}

        for metric_key, metric_value in status_metrics.items():
            if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                writer.add_scalar(f"by_status/{status}/{metric_key}", metric_value, step)

def log_detailed_tensorboard_stats(
    writer: SummaryWriter,
    online_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accumulated_stats: Dict[str, Any],
    step_stats: Dict[str, Any],
    previous_alltime_min: Optional[Dict[str, float]],
    current_step: int
) -> Dict[str, float]:
    """
    Log detailed statistics to TensorBoard for periodic reporting.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        online_network: The online neural network model
        optimizer: The optimizer used for training
        accumulated_stats: Dictionary of accumulated statistics
        step_stats: Dictionary of statistics for the current step
        previous_alltime_min: Previous all-time minimum values
        current_step: Current training step
        
    Returns:
        Updated all-time minimum values
    """
    # If previous_alltime_min is None, initialize it
    if previous_alltime_min is None:
        previous_alltime_min = {}
    
    # Log hyperparameters
    for key, value in step_stats.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            writer.add_scalar(f"hyperparams/{key}", value, current_step)
    
    # Log model parameters
    for name, param in online_network.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"model/{name}", param.data, current_step)
            if param.grad is not None:
                writer.add_histogram(f"grad/{name}", param.grad, current_step)
    
    # Log optimizer state
    for i, param_group in enumerate(optimizer.param_groups):
        for key, value in param_group.items():
            if key != 'params' and isinstance(value, (int, float)):
                writer.add_scalar(f"optimizer/group{i}_{key}", value, current_step)
    
    # Log accumulated statistics
    for key, value in accumulated_stats.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "alltime_min_ms" and key != "rolling_mean_ms":
            writer.add_scalar(f"accumulated/{key}", value, current_step)
    
    # Log all-time minimum values
    for key, value in accumulated_stats["alltime_min_ms"].items():
        writer.add_scalar(f"performance/alltime_min_ms_{key}", value, current_step)
        # Check if this is a new minimum
        if key not in previous_alltime_min or value < previous_alltime_min[key]:
            previous_alltime_min[key] = value
            writer.add_scalar(f"performance/new_min_ms_{key}", value, current_step)
    
    # Return updated all-time minimum values
    return previous_alltime_min


def log_iqn_specific_stats(
    writer: SummaryWriter,
    per_quantile_output: npt.NDArray,
    step: int
) -> None:
    """
    Log IQN-specific statistics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        per_quantile_output: Output from IQN network's quantiles
        step: Current training step
    """
    # Log standard deviation within IQN quantiles for each action
    for i, std in enumerate(list(per_quantile_output.std(axis=0))):
        writer.add_scalar(f"iqn/std_within_quantiles_action{i}", std, step)

    # If per_quantile_output has mean and min/max values
    if hasattr(per_quantile_output, 'mean') and hasattr(per_quantile_output, 'min'):
        writer.add_scalar("iqn/mean_quantile_value", per_quantile_output.mean().item(), step)
        writer.add_scalar("iqn/min_quantile_value", per_quantile_output.min().item(), step)
        writer.add_scalar("iqn/max_quantile_value", per_quantile_output.max().item(), step)


def collect_race_stats(
    rollout_results: Dict[str, Any],
    end_race_stats: Dict[str, Any],
    is_explo: bool,
    map_name: str,
    map_status: str,
    rollout_duration: float,
    accumulated_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Collect race statistics from rollout results.
    
    Args:
        rollout_results: Results from the rollout
        end_race_stats: Statistics from the end of the race
        is_explo: Whether this was an exploration run
        map_name: Name of the map
        map_status: Status of the map
        rollout_duration: Duration of the rollout
        accumulated_stats: Accumulated statistics dictionary
        
    Returns:
        Dictionary of race statistics
    """
    # Increment race counter
    accumulated_stats["cumul_number_races_done"] += 1
    
    # Build race stats dictionary
    race_stats = {
        f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / (rollout_duration * 1000),
        f"explo_race_time_{map_status}_{map_name}" if is_explo else f"eval_race_time_{map_status}_{map_name}" :
            end_race_stats["race_time"]/ 1000,
        f"explo_race_finished_{map_status}_{map_name}" if is_explo else f"eval_race_finished_{map_status}_{map_name}":
            end_race_stats["race_finished"],
        f"mean_action_gap_{map_name}": -(np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1,initial=None)
                                         .reshape(-1, 1)).mean(),
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

        "race_finished": end_race_stats.get("race_finished"),
        "race_time": end_race_stats.get("race_time"),
        "ratio_completion": end_race_stats.get("ratio_completion"),
        "max_speed": end_race_stats.get("max_speed"),
        "avg_speed": end_race_stats.get("avg_speed"),
        "is_exploration": is_explo,
        "map_name": map_name,
        "map_status": map_status,
        "episode": accumulated_stats["cumul_number_races_done"],
        "rollout_duration": rollout_duration
    }

    if not is_explo:
        race_stats[f"avg_Q_{map_status}_{map_name}"] = np.mean(rollout_results["q_values"])

    if end_race_stats["race_finished"]:
        race_stats[f"{'explo' if is_explo else 'eval'}_race_time_finished_{map_status}_{map_name}"] = (
                end_race_stats["race_time"] / 1000
        )
        if not is_explo:
            accumulated_stats["rolling_mean_ms"][map_name] = (
                    accumulated_stats["rolling_mean_ms"].get(map_name, config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms)
                    * 0.9
                    + end_race_stats["race_time"] * 0.1
            )


    if ((not is_explo) and end_race_stats["race_finished"] and end_race_stats["race_time"] < 1.02 * accumulated_stats["rolling_mean_ms"][map_name]):
        race_stats[f"eval_race_time_robust_{map_status}_{map_name}"] = end_race_stats["race_time"] / 1000

        if map_name in reference_times:
            for reference_time_name in ["author", "gold"]:
                if reference_time_name in reference_times[map_name]:
                    reference_time = reference_times[map_name][reference_time_name]
                    race_stats[f"eval_ratio_{map_status}_{reference_time_name}_{map_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                    )
                    race_stats[f"eval_agg_ratio_{map_status}_{reference_time_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                    )

    for i in [0]:
        race_stats[f"q_value_{i}_starting_frame_{map_name}"] = end_race_stats[f"q_value_{i}_starting_frame"]
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
            race_stats[f"split_{map_name}_{i}"] = split_time
    
    return race_stats


def collect_periodic_stats(
    accumulated_stats: Dict[str, Any],
    loss_history: List[float],
    train_on_batch_duration_history: List[float],
    grad_norm_history: List[float],
    layer_grad_norm_history: Dict[str, List[float]],
    buffer,
    training_params: Dict[str, float],
    time_waited_for_workers: float,
    time_training: float,
    time_testing: float,
    time_since_last_save: float,
    transitions_learned_last_save: int
) -> Dict[str, Any]:
    """
    Collect statistics for periodic reporting and saving.
    
    Args:
        accumulated_stats: Dictionary of accumulated statistics
        loss_history: List of recent loss values
        train_on_batch_duration_history: List of recent training batch durations
        grad_norm_history: List of recent gradient norm values
        layer_grad_norm_history: Dictionary mapping layer names to gradient norm histories
        buffer: Replay buffer
        training_params: Dictionary of training parameters
        time_waited_for_workers: Time spent waiting for worker processes
        time_training: Time spent on training
        time_testing: Time spent on testing
        time_since_last_save: Time since the last checkpoint save
        transitions_learned_last_save: Number of transitions learned since last save
        
    Returns:
        Dictionary of collected statistics
    """
    # Calculate transitions learned per second
    transitions_learned_per_second = (
        (accumulated_stats["cumul_number_single_memories_used"] - transitions_learned_last_save) / 
        time_since_last_save if time_since_last_save > 0 else 0
    )

    # Calculate time percentages
    waited_percentage = time_waited_for_workers / time_since_last_save if time_since_last_save > 0 else 0
    trained_percentage = time_training / time_since_last_save if time_since_last_save > 0 else 0
    tested_percentage = time_testing / time_since_last_save if time_since_last_save > 0 else 0

    # Create stats dictionary with training parameters
    step_stats = {
        "gamma": training_params.get("gamma", 0),
        "learning_rate": training_params.get("learning_rate", 0),
        "weight_decay": training_params.get("weight_decay", 0),
        "memory_size": len(buffer),
        "learner_percentage_waiting_for_workers": waited_percentage,
        "learner_percentage_training": trained_percentage,
        "learner_percentage_testing": tested_percentage,
        "transitions_learned_per_second": transitions_learned_per_second,
    }
    
    # Add additional parameters if they exist
    if hasattr(buffer, "config_copy"):
        step_stats["n_steps"] = buffer.config_copy.n_steps if hasattr(buffer.config_copy, "n_steps") else None
    
    # Add epsilon values if provided
    if "epsilon" in training_params:
        step_stats["epsilon"] = training_params["epsilon"]
    if "epsilon_boltzmann" in training_params:
        step_stats["epsilon_boltzmann"] = training_params["epsilon_boltzmann"]

    # Loss and gradient statistics
    if len(loss_history) > 0:
        step_stats.update({
            "loss": np.mean(loss_history),
            "train_on_batch_duration": np.median(train_on_batch_duration_history),
            "grad_norm_history_q1": np.quantile(grad_norm_history, 0.25) if len(grad_norm_history) > 0 else 0,
            "grad_norm_history_median": np.quantile(grad_norm_history, 0.5) if len(grad_norm_history) > 0 else 0,
            "grad_norm_history_q3": np.quantile(grad_norm_history, 0.75) if len(grad_norm_history) > 0 else 0,
            "grad_norm_history_d9": np.quantile(grad_norm_history, 0.9) if len(grad_norm_history) > 0 else 0,
            "grad_norm_history_d98": np.quantile(grad_norm_history, 0.98) if len(grad_norm_history) > 0 else 0,
            "grad_norm_history_max": np.max(grad_norm_history) if len(grad_norm_history) > 0 else 0,
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
    if hasattr(buffer, "_sampler") and isinstance(buffer._sampler, PrioritizedSampler):
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
        if key not in ["alltime_min_ms", "rolling_mean_ms"] and isinstance(value, (int, float)):
            step_stats[key] = value

    # Add minimum times
    for key, value in accumulated_stats["alltime_min_ms"].items():
        step_stats[f"alltime_min_ms_{key}"] = value

    return step_stats


def collect_periodic_stats_iqn(
    accumulated_stats: Dict[str, Any],
    loss_history: List[float],
    train_on_batch_duration_history: List[float],
    grad_norm_history: List[float],
    layer_grad_norm_history: Dict[str, List[float]],
    buffer,
    gamma: float,
    learning_rate: float,
    weight_decay: float,
    time_waited_for_workers: float,
    time_training: float,
    time_testing: float,
    time_since_last_save: float,
    transitions_learned_last_save: int,
    epsilon: float,
    epsilon_boltzmann: float
) -> Dict[str, Any]:
    """
    Collect statistics for periodic reporting and saving for IQN agent.
    
    Args:
        accumulated_stats: Dictionary of accumulated statistics
        loss_history: List of recent loss values
        train_on_batch_duration_history: List of recent training batch durations
        grad_norm_history: List of recent gradient norm values
        layer_grad_norm_history: Dictionary mapping layer names to gradient norm histories
        buffer: Replay buffer
        gamma: Discount factor
        learning_rate: Current learning rate
        weight_decay: Current weight decay value
        time_waited_for_workers: Time spent waiting for worker processes
        time_training: Time spent on training
        time_testing: Time spent on testing
        time_since_last_save: Time since the last checkpoint save
        transitions_learned_last_save: Number of transitions learned since last save
        epsilon: Current epsilon value
        epsilon_boltzmann: Current epsilon boltzmann value
        
    Returns:
        Dictionary of collected statistics
    """
    # Use the unified function by converting parameters
    training_params = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epsilon": epsilon,
        "epsilon_boltzmann": epsilon_boltzmann
    }
    
    return collect_periodic_stats(
        accumulated_stats,
        loss_history,
        train_on_batch_duration_history,
        grad_norm_history,
        layer_grad_norm_history,
        buffer,
        training_params,
        time_waited_for_workers,
        time_training,
        time_testing,
        time_since_last_save,
        transitions_learned_last_save
    )
