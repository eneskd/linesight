"""
Utilities for TensorBoard logging in Trackmania AI training processes.
This module centralizes all TensorBoard-related functionality.
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


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


def log_training_step_to_tensorboard(
    writer: SummaryWriter,
    step: int,
    stats: Dict[str, Any],
    buffer_size: int,
    max_next_q: float,
    train_duration_ms: float,
    rolling_means: Dict[str, float]
) -> None:
    """
    Log training step metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        step: Current training step
        stats: Dictionary of statistics from training step
        buffer_size: Current size of the replay buffer
        max_next_q: Maximum Q-value for next state
        train_duration_ms: Duration of the training step in milliseconds
        rolling_means: Dictionary of rolling mean values for various metrics
    """
    # Basic training metrics
    writer.add_scalar("loss/loss", stats["loss"], step)
    writer.add_scalar("q_values/mean_q", stats["mean_q"], step)
    writer.add_scalar("q_values/max_q", stats["max_q"], step)
    writer.add_scalar("q_values/min_q", stats["min_q"], step)
    writer.add_scalar("q_values/max_next_q", max_next_q, step)
    
    # Reward metrics
    writer.add_scalar("rewards/mean", stats["reward_mean"], step)
    writer.add_scalar("rewards/min", stats["reward_min"], step)
    writer.add_scalar("rewards/max", stats["reward_max"], step)
    writer.add_scalar("rewards/std", stats["reward_std"], step)
    
    # Training performance metrics
    writer.add_scalar("metrics/grad_norm", stats["grad_norm"], step)
    writer.add_scalar("metrics/buffer_size", buffer_size, step)
    writer.add_scalar("metrics/train_step_ms", train_duration_ms, step)
    
    # Add rolling mean if available
    if "train_step" in rolling_means:
        writer.add_scalar("metrics/train_step_ms_rolling", rolling_means["train_step"], step)


def log_race_stats_to_tensorboard(
    writer: SummaryWriter,
    race_stats: Dict[str, Any],
    accumulated_stats: Dict[str, Any]
) -> None:
    """
    Log race statistics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        race_stats: Dictionary of race statistics
        accumulated_stats: Dictionary of accumulated statistics
    """
    step = accumulated_stats["cumul_number_frames_played"]
    
    # Race metrics
    if "race_time" in race_stats and race_stats["race_time"] is not None:
        writer.add_scalar("race/race_time", race_stats["race_time"] / 1000, step)
    
    if "ratio_completion" in race_stats:
        writer.add_scalar("race/ratio_completion", race_stats["ratio_completion"], step)
    
    if "max_speed" in race_stats:
        writer.add_scalar("race/max_speed", race_stats["max_speed"], step)
        
    if "avg_speed" in race_stats:
        writer.add_scalar("race/avg_speed", race_stats["avg_speed"], step)
        
    # Map specific metrics
    if "map_name" in race_stats and "race_time" in race_stats and race_stats["race_time"] is not None:
        map_name = race_stats["map_name"]
        writer.add_scalar(f"race_by_map/{map_name}_time", race_stats["race_time"] / 1000, step)
        
    # Exploration metrics
    if "is_exploration" in race_stats:
        writer.add_scalar("race/is_exploration", int(race_stats["is_exploration"]), step)
        
    # Episode metrics
    if "episode" in race_stats:
        writer.add_scalar("race/episode", race_stats["episode"], step)
        
    # Training metrics
    if "rollout_duration" in race_stats:
        writer.add_scalar("metrics/rollout_duration", race_stats["rollout_duration"], step)


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
    stats: Dict[str, Any],
    step: int
) -> None:
    """
    Log IQN-specific statistics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        stats: Dictionary of IQN-specific statistics
        step: Current training step
    """
    if "quantile_huber_loss" in stats:
        writer.add_scalar("loss/quantile_huber_loss", stats["quantile_huber_loss"], step)
    
    if "tau_mean" in stats:
        writer.add_scalar("tau/mean", stats["tau_mean"], step)
    
    if "tau_std" in stats:
        writer.add_scalar("tau/std", stats["tau_std"], step)
    
    if "tau_min" in stats:
        writer.add_scalar("tau/min", stats["tau_min"], step)
    
    if "tau_max" in stats:
        writer.add_scalar("tau/max", stats["tau_max"], step)