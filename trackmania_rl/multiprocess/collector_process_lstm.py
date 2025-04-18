import importlib
import time
from itertools import chain, count, cycle
from pathlib import Path

import numpy as np
import torch
from torch import multiprocessing as mp

from config_files import config_copy
from trackmania_rl import utilities
from trackmania_rl.agents.lstm import make_untrained_lstm_agent

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

    # Load LSTM agent for inference
    inference_network, uncompiled_inference_network = make_untrained_lstm_agent(
        jit=config_copy.use_jit,
        is_inference=True,
    )
    try:
        inference_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
    except Exception as e:
        print("Worker could not load weights, exception:", e)

    class LSTMInferer:
        def __init__(self, network, n_actions):
            self.network = network
            self.n_actions = n_actions
            self.epsilon = 0.1
            self.is_explo = True
            self.hidden = None

        def preprocess_img(self, img):
            # (H, W) -> (1, H, W)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.transpose(img, (2, 0, 1))
            elif img.ndim == 3 and img.shape[2] > 1:
                img = np.mean(img, axis=2, keepdims=True)
                img = np.transpose(img, (2, 0, 1))
            return img.astype(np.float32)

        def get_exploration_action(self, img_seq, float_inputs_seq):
            # img_seq: list of np.ndarray (H, W) or (1, H, W)
            # float_inputs_seq: list of np.ndarray (float_input_dim,)
            seq_len = len(img_seq)
            imgs = np.stack([self.preprocess_img(img) for img in img_seq])  # (seq_len, 1, H, W)
            floats = np.stack(float_inputs_seq)  # (seq_len, float_input_dim)
            imgs_tensor = torch.from_numpy(imgs).unsqueeze(0).to("cuda", dtype=torch.float32)  # (1, seq_len, 1, H, W)
            floats_tensor = torch.from_numpy(floats).unsqueeze(0).to("cuda", dtype=torch.float32)  # (1, seq_len, float_input_dim)
            with torch.no_grad():
                q_values, self.hidden = self.network(imgs_tensor, floats_tensor, self.hidden)
                q_values = q_values[0, -1]  # (n_actions) for last step
                if np.random.rand() < self.epsilon and self.is_explo:
                    action = np.random.randint(self.n_actions)
                    return action, False, 0.0, np.zeros(self.n_actions, dtype=np.float32)
                action = int(torch.argmax(q_values).item())
                return action, True, float(q_values[action].item()), q_values.cpu().numpy()

    inferer = LSTMInferer(inference_network, n_actions=len(config_copy.inputs))

    def update_network():
        with shared_network_lock:
            uncompiled_inference_network.load_state_dict(uncompiled_shared_network.state_dict())

    inference_network.train()

    map_cycle_str = str(config_copy.map_cycle)
    set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
    map_cycle_iter = cycle(chain(*config_copy.map_cycle))

    zone_centers_filename = None

    # Warmup
    for _ in range(5):
        dummy_img_seq = [np.random.rand(config_copy.H_downsized, config_copy.W_downsized).astype(np.float32) for _ in range(4)]
        dummy_float_seq = [np.random.rand(config_copy.float_input_dim).astype(np.float32) for _ in range(4)]
        inferer.get_exploration_action(dummy_img_seq, dummy_float_seq)

    time_since_last_queue_push = time.perf_counter()
    for loop_number in count(1):
        importlib.reload(config_copy)
        tmi.max_minirace_duration_ms = config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms

        if str(config_copy.map_cycle) != map_cycle_str:
            map_cycle_str = str(config_copy.map_cycle)
            set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
            map_cycle_iter = cycle(chain(*config_copy.map_cycle))

        next_map_tuple = next(map_cycle_iter)
        if next_map_tuple[2] != zone_centers_filename:
            zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
        map_name, map_path, zone_centers_filename, is_explo, fill_buffer = next_map_tuple
        map_status = "trained" if map_name in set_maps_trained else "blind"

        inferer.epsilon = utilities.from_exponential_schedule(config_copy.epsilon_schedule, shared_steps.value)
        inferer.is_explo = is_explo

        rollout_start_time = time.perf_counter()

        if inference_network.training and not is_explo:
            inference_network.eval()
        elif is_explo and not inference_network.training:
            inference_network.train()

        update_network()

        # The rollout function should be adapted to support sequence-based agents
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