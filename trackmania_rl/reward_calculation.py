"""
Reward calculation utilities for Trackmania RL.

This module centralizes reward computation logic that was previously embedded in buffer_management.py.
"""

import numpy as np
from numba import jit

from config_files import config_copy
from trackmania_rl.reward_shaping import speedslide_quality_tarmac

# Define indices as named constants for better maintainability
WHEELS_TOUCH_GROUND_START_IDX = 25
WHEELS_TOUCH_GROUND_END_IDX = 29
LATERAL_SPEED_IDX = 56
FORWARD_SPEED_IDX = 58
VELOCITY_START_IDX = 56
VELOCITY_END_IDX = 59
CURRENT_VCP_START_IDX = 62
CURRENT_VCP_END_IDX = 65
VCP_AHEAD_START_IDX = 65
VCP_AHEAD_END_IDX = 68


@jit(nopython=True)
def get_potential(state_float):
    """
    Calculate potential-based shaping component as per Andrew Ng's paper.
    Used for reward shaping.

    Args:
        state_float: The state float features

    Returns:
        The potential value for this state
    """
    vector_vcp_to_vcp_further_ahead = state_float[VCP_AHEAD_START_IDX:VCP_AHEAD_END_IDX] - state_float[CURRENT_VCP_START_IDX:CURRENT_VCP_END_IDX]
    vector_vcp_to_vcp_further_ahead_normalized = vector_vcp_to_vcp_further_ahead / np.linalg.norm(vector_vcp_to_vcp_further_ahead)

    return (
        config_copy.shaped_reward_dist_to_cur_vcp
        * max(
            config_copy.shaped_reward_min_dist_to_cur_vcp,
            min(config_copy.shaped_reward_max_dist_to_cur_vcp, np.linalg.norm(state_float[CURRENT_VCP_START_IDX:CURRENT_VCP_END_IDX])),
        )
    ) + (config_copy.shaped_reward_point_to_vcp_ahead * (vector_vcp_to_vcp_further_ahead_normalized[2] - 1))


def calculate_frame_rewards(
    rollout_results,
    engineered_speedslide_reward,
    engineered_neoslide_reward,
    engineered_kamikaze_reward,
    engineered_close_to_vcp_reward,
    wall_collision_penalty=0.0,
):
    """
    Calculate per-frame rewards for each step in a rollout.

    Args:
        rollout_results: Dictionary containing rollout data (frames, states, etc.)
        engineered_speedslide_reward: Reward coefficient for speedslides
        engineered_neoslide_reward: Reward coefficient for neoslides
        engineered_kamikaze_reward: Reward coefficient for kamikaze moves
        engineered_close_to_vcp_reward: Reward coefficient for checkpoint proximity
        wall_collision_penalty: Negative reward for hitting walls

    Returns:
        Array of rewards for each frame in the rollout
    """
    n_frames = len(rollout_results["frames"])
    reward_into = np.zeros(n_frames)

    for i in range(1, n_frames):
        # Time penalty (constant reward per ms)
        reward_into[i] += config_copy.constant_reward_per_ms * (
            config_copy.ms_per_action
            if (i < n_frames - 1 or ("race_time" not in rollout_results))
            else rollout_results["race_time"] - (n_frames - 2) * config_copy.ms_per_action
        )

        # Progress reward (advancing along centerline)
        reward_into[i] += (
            rollout_results["meters_advanced_along_centerline"][i] - rollout_results["meters_advanced_along_centerline"][i - 1]
        ) * config_copy.reward_per_m_advanced_along_centerline

        if i < n_frames - 1:
            # Final speed reward
            if config_copy.final_speed_reward_per_m_per_s != 0 and rollout_results["state_float"][i][FORWARD_SPEED_IDX] > 0:
                # car has velocity *forward*
                reward_into[i] += config_copy.final_speed_reward_per_m_per_s * (
                    np.linalg.norm(rollout_results["state_float"][i][VELOCITY_START_IDX:VELOCITY_END_IDX]) -
                    np.linalg.norm(rollout_results["state_float"][i - 1][VELOCITY_START_IDX:VELOCITY_END_IDX])
                )

            # Speedslide reward
            if engineered_speedslide_reward != 0 and np.all(rollout_results["state_float"][i][WHEELS_TOUCH_GROUND_START_IDX:WHEELS_TOUCH_GROUND_END_IDX]):
                # all wheels touch the ground
                reward_into[i] += engineered_speedslide_reward * max(
                    0.0,
                    1 - abs(speedslide_quality_tarmac(
                        rollout_results["state_float"][i][LATERAL_SPEED_IDX],
                        rollout_results["state_float"][i][FORWARD_SPEED_IDX]) - 1),
                )

            # Neoslide reward
            reward_into[i] += (
                engineered_neoslide_reward if abs(rollout_results["state_float"][i][LATERAL_SPEED_IDX]) >= 2.0 else 0
            )

            # Kamikaze reward
            if (
                engineered_kamikaze_reward != 0
                and rollout_results["actions"][i] <= 2
                or np.sum(rollout_results["state_float"][i][WHEELS_TOUCH_GROUND_START_IDX:WHEELS_TOUCH_GROUND_END_IDX]) <= 1
            ):
                reward_into[i] += engineered_kamikaze_reward

            # Checkpoint proximity reward
            if engineered_close_to_vcp_reward != 0:
                reward_into[i] += engineered_close_to_vcp_reward * max(
                    config_copy.engineered_reward_min_dist_to_cur_vcp,
                    min(config_copy.engineered_reward_max_dist_to_cur_vcp,
                        np.linalg.norm(rollout_results["state_float"][i][CURRENT_VCP_START_IDX:CURRENT_VCP_END_IDX])),
                )

            # Wall collision penalty
            if wall_collision_penalty != 0.0 and "has_lateral_contact" in rollout_results:
                if rollout_results["has_lateral_contact"][i]:
                    reward_into[i] += wall_collision_penalty  # This should be negative

    return reward_into


def calculate_n_step_returns(reward_into, n_steps, gamma):
    """
    Calculate n-step returns from single-step rewards.

    Args:
        reward_into: Array of rewards for each step
        n_steps: Maximum number of steps to look ahead
        gamma: Discount factor

    Returns:
        Array of n-step returns for each starting state
    """
    n_frames = len(reward_into)
    n_step_returns = []

    for i in range(n_frames - 1):
        actual_n_steps = min(n_steps, n_frames - 1 - i)
        rewards = np.empty(n_steps).astype(np.float32)

        for j in range(actual_n_steps):
            rewards[j] = (gamma**j) * reward_into[i + j + 1] + (rewards[j - 1] if j >= 1 else 0)

        n_step_returns.append(rewards)

    return n_step_returns