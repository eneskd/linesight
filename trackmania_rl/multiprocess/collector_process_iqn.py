"""
Collector process for IQN agent: handles Trackmania game instance and provides rollout results.
"""
import importlib
import time
from itertools import chain, count, cycle
from pathlib import Path

import numpy as np
import torch
from torch import multiprocessing as mp

from config_files import config_copy
from config_files import iqn_config_copy
from trackmania_rl import utilities
from trackmania_rl.agents import iqn


def collector_process_fn(
        rollout_queue,
        uncompiled_shared_network,
        shared_network_lock,
        game_spawning_lock,
        shared_steps: mp.Value,
        base_dir: Path,
        save_dir: Path,
        tmi_port: int,
):
    """Main collector process function for IQN agent"""
    # Import here to avoid circular imports
    from trackmania_rl.map_loader import analyze_map_cycle, load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager

    # Initialize game instance manager
    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=tmi_port,
    )

    # Load IQN agent and create inferer
    inference_network, uncompiled_inference_network = iqn.make_untrained_iqn_network(config_copy.use_jit,
                                                                                     is_inference=True)
    try:
        inference_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
        print("IQN collector: weights loaded successfully")
    except Exception as e:
        print(f"IQN collector: could not load weights, exception: {e}")

    # Use the Inferer from the iqn module instead of the local IQNInferer class
    inferer = iqn.Inferer(inference_network, iqn_config_copy.iqn_k, config_copy.tau_epsilon_boltzmann)

    def update_network():
        """Update weights of the inference network from shared network"""
        with shared_network_lock:
            uncompiled_inference_network.load_state_dict(uncompiled_shared_network.state_dict())

    # Initialize map cycle
    map_cycle_str = str(config_copy.map_cycle)
    set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
    map_cycle_iter = cycle(chain(*config_copy.map_cycle))
    zone_centers_filename = None

    # Warmup pytorch
    print("IQN collector: warming up model...")
    for _ in range(5):
        dummy_img = np.random.randint(low=0, high=255, size=(1, config_copy.H_downsized, config_copy.W_downsized),
                                      dtype=np.uint8)
        dummy_float = np.random.rand(config_copy.float_input_dim).astype(np.float32)
        inferer.infer_network(dummy_img, dummy_float)
    print("IQN collector: warmup complete")

    # Main collection loop
    time_since_last_queue_push = time.perf_counter()
    for loop_number in count(1):
        # Reload config
        importlib.reload(config_copy)
        tmi.max_minirace_duration_ms = config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms

        # Check if map cycle changed
        if str(config_copy.map_cycle) != map_cycle_str:
            map_cycle_str = str(config_copy.map_cycle)
            set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
            map_cycle_iter = cycle(chain(*config_copy.map_cycle))

        # Get next map from cycle
        next_map_tuple = next(map_cycle_iter)
        if next_map_tuple[2] != zone_centers_filename:
            zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
            zone_centers_filename = next_map_tuple[2]

        map_name, map_path, _, is_explo, fill_buffer = next_map_tuple
        map_status = "trained" if map_name in set_maps_trained else "blind"

        # Update exploration parameters
        inferer.epsilon = utilities.from_exponential_schedule(config_copy.epsilon_schedule, shared_steps.value)
        inferer.epsilon_boltzmann = utilities.from_exponential_schedule(config_copy.epsilon_boltzmann_schedule,
                                                                        shared_steps.value)
        inferer.tau_epsilon_boltzmann = config_copy.tau_epsilon_boltzmann
        inferer.is_explo = is_explo

        # Set network mode (train for exploration, eval for evaluation)
        if inference_network.training and not is_explo:
            inference_network.eval()
        elif is_explo and not inference_network.training:
            inference_network.train()

        # Update network weights
        update_network()

        # Run one episode
        rollout_start_time = time.perf_counter()
        rollout_results, end_race_stats = tmi.rollout(
            exploration_policy=inferer.get_exploration_action,
            map_path=map_path,
            zone_centers=zone_centers,
            update_network=update_network,
        )
        rollout_end_time = time.perf_counter()

        # Track timing
        rollout_duration = rollout_end_time - rollout_start_time
        rollout_results["worker_time_in_rollout_percentage"] = rollout_duration / (
                time.perf_counter() - time_since_last_queue_push)
        time_since_last_queue_push = time.perf_counter()

        # Log progress
        print(f"IQN collector: completed loop {loop_number}, map {map_name}, duration {rollout_duration:.1f}s")

        # Send results to learner process if rollout was successful
        if not tmi.last_rollout_crashed:
            rollout_queue.put((
                rollout_results,
                end_race_stats,
                fill_buffer,
                is_explo,
                map_name,
                map_status,
                rollout_duration,
                loop_number,
            ))
