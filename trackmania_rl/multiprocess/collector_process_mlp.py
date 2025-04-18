"""
Collector process for MLP agent: handles a Trackmania game instance and provides rollout results to the learner process.
Ensures all images are single-channel with shape (batch_size, 1, H, W).
"""

import importlib
import time
from itertools import chain, count, cycle
from pathlib import Path

import numpy as np
import torch
from torch import multiprocessing as mp

from config_files import config_copy
from config_files import mlp_config_copy

from trackmania_rl import utilities
from trackmania_rl.agents.mlp import make_untrained_mlp_agent

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
    from trackmania_rl.map_loader import analyze_map_cycle, load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=tmi_port,
    )

    # Load MLP agent for inference
    inference_network, uncompiled_inference_network = make_untrained_mlp_agent(config_copy.use_jit, is_inference=True)
    try:
        inference_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
    except Exception as e:
        print("Worker could not load weights, exception:", e)

    # Epsilon-greedy inferer for MLP agent with frame and float inputs
    class MLPInferer:
        def __init__(self, network, n_actions):
            self.network = network
            self.n_actions = n_actions
            self.epsilon = 0.1
            self.is_explo = True
            
        def preprocess_img(self, img):

            if img.shape == (1, config_copy.H_downsized, config_copy.W_downsized):
                return img

            # If (H, W), expand to (1, H, W)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            # If (H, W, 1), transpose to (1, H, W)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.transpose(img, (2, 0, 1))
            # If (H, W, C) with C > 1, convert to grayscale by averaging channels
            elif img.ndim == 3 and img.shape[2] > 1:
                img = np.mean(img, axis=2, keepdims=True)  # (H, W, 1)
                img = np.transpose(img, (2, 0, 1))  # (1, H, W)
            # If (C, H, W) with C > 1, convert to grayscale by averaging channels
            elif img.ndim == 3 and img.shape[0] > 1:
                img = np.mean(img, axis=0, keepdims=True)  # (1, H, W)

            return img.astype(np.float32)

        def get_exploration_action(self, img, float_inputs):
            # Preprocess image to ensure single-channel (1, H, W) format
            img = self.preprocess_img(img)

            if np.random.rand() < self.epsilon and self.is_explo:
                return np.random.randint(self.n_actions), False, 0.0, np.zeros(self.n_actions, dtype=np.float32)
            with torch.no_grad():
                # Convert image to torch tensor and move to cuda
                img_tensor = torch.from_numpy(img).unsqueeze(0).to("cuda", dtype=torch.float32)  # (1, 1, H, W)
                float_tensor = torch.from_numpy(float_inputs).unsqueeze(0).to("cuda", dtype=torch.float32)
                q_values = self.network(img_tensor, float_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
                return action, True, float(q_values[0, action].item()), q_values.cpu().numpy().squeeze()

    inferer = MLPInferer(inference_network, n_actions=len(config_copy.inputs))

    def update_network():
        # Update weights of the inference network
        with shared_network_lock:
            uncompiled_inference_network.load_state_dict(uncompiled_shared_network.state_dict())

    # ========================================================
    # Training loop
    # ========================================================
    inference_network.train()

    map_cycle_str = str(config_copy.map_cycle)
    set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
    map_cycle_iter = cycle(chain(*config_copy.map_cycle))

    zone_centers_filename = None

    # ========================================================
    # Warmup pytorch
    # ========================================================
    for _ in range(5):
        dummy_img = np.random.rand(config_copy.H_downsized, config_copy.W_downsized).astype(np.float32)
        # dummy_img = inferer.preprocess_img(dummy_img)
        dummy_float = np.random.rand(config_copy.float_input_dim).astype(np.float32)
        inferer.get_exploration_action(dummy_img, dummy_float)

    time_since_last_queue_push = time.perf_counter()
    for loop_number in count(1):
        importlib.reload(config_copy)

        tmi.max_minirace_duration_ms = config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms

        # ===============================================
        #   DID THE CYCLE CHANGE ?
        # ===============================================
        if str(config_copy.map_cycle) != map_cycle_str:
            map_cycle_str = str(config_copy.map_cycle)
            set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
            map_cycle_iter = cycle(chain(*config_copy.map_cycle))

        # ===============================================
        #   GET NEXT MAP FROM CYCLE
        # ===============================================
        next_map_tuple = next(map_cycle_iter)
        if next_map_tuple[2] != zone_centers_filename:
            zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
        map_name, map_path, zone_centers_filename, is_explo, fill_buffer = next_map_tuple
        map_status = "trained" if map_name in set_maps_trained else "blind"

        # Epsilon schedule
        inferer.epsilon = utilities.from_exponential_schedule(config_copy.epsilon_schedule, shared_steps.value)
        inferer.is_explo = is_explo

        # ===============================================
        #   PLAY ONE ROUND
        # ===============================================

        rollout_start_time = time.perf_counter()

        if inference_network.training and not is_explo:
            inference_network.eval()
        elif is_explo and not inference_network.training:
            inference_network.train()

        update_network()

        rollout_start_time = time.perf_counter()
        rollout_results, end_race_stats = tmi.rollout(
            exploration_policy=inferer.get_exploration_action,
            map_path=map_path,
            zone_centers=zone_centers,
            update_network=update_network,
        )
        rollout_end_time = time.perf_counter()
        rollout_duration = rollout_end_time - rollout_start_time
        rollout_results["worker_time_in_rollout_percentage"] = rollout_duration / (time.perf_counter() - time_since_last_queue_push)
        time_since_last_queue_push = time.perf_counter()
        print("", flush=True)

        if not tmi.last_rollout_crashed:
            rollout_queue.put(
                (
                    rollout_results,
                    end_race_stats,
                    fill_buffer,
                    is_explo,
                    map_name,
                    map_status,
                    rollout_duration,
                    loop_number,
                )
            )
