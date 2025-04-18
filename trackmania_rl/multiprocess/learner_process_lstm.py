import importlib
import time
import joblib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from config_files import config_copy
from config_files import lstm_config_copy

from trackmania_rl import buffer_management, utilities
from trackmania_rl.agents.lstm import make_untrained_lstm_agent

from trackmania_rl.multiprocess.buffer_lstm import LSTMReplayBuffer


def dqn_loss(q_values, actions, targets):
    """
    Standard DQN loss: MSE between Q(s,a) and target.
    q_values: (batch, seq_len, n_actions)
    actions: (batch, seq_len) int64
    targets: (batch, seq_len) float32
    """
    q_selected = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
    return torch.nn.functional.mse_loss(q_selected, targets)

def learner_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
):
    online_network, uncompiled_online_network = make_untrained_lstm_agent(
        jit=config_copy.use_jit,
        is_inference=False,
    )
    target_network, _ = make_untrained_lstm_agent(
        jit=config_copy.use_jit,
        is_inference=False,
    )

    try:
        accumulated_stats = joblib.load(save_dir / "lstm_accumulated_stats.joblib")
        shared_steps.value = accumulated_stats.get("cumul_number_frames_played", 0)
        print(" =====================      Learner LSTM stats loaded !      ============================")
    except Exception as e:
        print(" Learner LSTM could not load stats:", e)
        accumulated_stats = defaultdict(int)

    optimizer = torch.optim.RAdam(
        online_network.parameters(),
        lr=utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"]),
        eps=config_copy.adam_epsilon,
        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
    )
    scaler = torch.amp.GradScaler("cuda")
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + "_lstm")))

    # Create LSTM buffer
    buffer = LSTMReplayBuffer(capacity=lstm_config_copy.replay_buffer_capacity, seq_len=lstm_config_copy.lstm_seq_len)
    
    # =========================
    # Load existing weights/stats if available
    # =========================
    accumulated_stats = defaultdict(int)
    try:
        online_network.load_state_dict(torch.load(save_dir / "lstm_weights1.torch"))
        target_network.load_state_dict(torch.load(save_dir / "lstm_weights2.torch"))
        print(" =====================     Learner LSTM weights loaded !     ============================")
    except Exception as e:
        print(" Learner LSTM could not load weights:", e)

    try:
        optimizer.load_state_dict(torch.load(save_dir / "lstm_optimizer.torch"))
        print(" =====================     Learner LSTM optimizer loaded !     ============================")
    except Exception as e:
        print(" Learner LSTM could not load optimizer state:", e)

    try:
        scaler.load_state_dict(torch.load(save_dir / "lstm_scaler.torch"))
        print(" =====================     Learner LSTM scaler loaded !     ============================")
    except Exception as e:
        print(" Learner LSTM could not load scaler state:", e)

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())



    if "rolling_mean_ms" not in accumulated_stats:
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats.get("cumul_number_single_memories_used", 0)
    transitions_learned_last_save = accumulated_stats.get("cumul_number_single_memories_used", 0)
    neural_net_reset_counter = 0
    single_reset_flag = config_copy.single_reset_flag

    # Stats tracking
    stats = defaultdict(list)
    last_target_update = 0
    last_save = 0
    last_shared_update = 0
    step = shared_steps.value if hasattr(shared_steps, "value") else 0
    wait_time_total = 0  # Total time spent waiting for rollouts
    cumul_number_batches_done = 0  # Track number of batches processed
    
    print("Learner process started.")
    print(f"Replay buffer capacity: {buffer.capacity}, LSTM sequence length: {buffer.seq_len}")
    print(f"Batch size: {lstm_config_copy.lstm_batch_size}")
    print(f"Initial shared step value: {shared_steps.value}")


    while True:
        wait_start = time.perf_counter()
        # Always block until rollout_results is available to avoid UnboundLocalError
        while True:
            rollout_found = False
            for idx, queue in enumerate(rollout_queues):
                if not queue.empty():
                    (
                        rollout_results,
                        end_race_stats,
                        fill_buffer,
                        is_explo,
                        map_name,
                        map_status,
                        rollout_duration,
                        loop_number,
                    ) = queue.get()
                    rollout_found = True
                    break
            if rollout_found:
                break
            else:
                time.sleep(0.1)  # Sleep a little to avoid busy-waiting
        wait_time = time.perf_counter() - wait_start
        wait_time_total += wait_time
        if wait_time > 1.0:
            print(f"  [Timer] Waited {wait_time:.2f} seconds for workers to provide a rollout (total waited {wait_time_total:.1f}s).")

        # Compute rewards
        n_frames = len(rollout_results["frames"])
        rewards = np.zeros(n_frames)
        for i in range(1, n_frames):
            prev_state = rollout_results["state_float"][i - 1]
            curr_state = rollout_results["state_float"][i]
            action = rollout_results["actions"][i]
            meters_advanced = rollout_results["meters_advanced_along_centerline"][i] - rollout_results["meters_advanced_along_centerline"][i - 1]
            ms_elapsed = config_copy.ms_per_action if (i < n_frames - 1 or ("race_time" not in rollout_results)) else rollout_results["race_time"] - (n_frames - 2) * config_copy.ms_per_action

            # Calculate reward
            rewards[i] = (
                config_copy.constant_reward_per_ms * ms_elapsed
                + config_copy.reward_per_m_advanced_along_centerline * meters_advanced
            )

        # Prepare episode data for LSTM buffer
        episode = {
            "state_imgs": rollout_results["frames"],         # list of (1, H, W)
            "state_floats": rollout_results["state_float"],  # list of (float_input_dim,)
            "actions": rollout_results["actions"],           # list of ints
            "rewards": rewards,                              # list of floats
            "next_state_imgs": rollout_results.get("next_frames", rollout_results["frames"][1:] + [rollout_results["frames"][-1]]),
            "next_state_floats": rollout_results.get("next_state_float", rollout_results["state_float"][1:] + [rollout_results["state_float"][-1]]),
            "gammas": np.ones(n_frames) * utilities.from_linear_schedule(config_copy.gamma_schedule, accumulated_stats["cumul_number_frames_played"]) # list of floats
        }
        
        # Add episode to buffer
        buffer.add_episode(episode)
        print(f"  Added episode to replay buffer. Buffer size: {len(buffer)} / {buffer.capacity}")
        
        # Only train if we have enough data
        if len(buffer) < lstm_config_copy.lstm_batch_size:
            print(f"  Not enough data in buffer. Waiting for buffer size >= batch size ({config_copy.batch_size})")
            continue
            
        # Sample a batch of sequences from the buffer
        batch = buffer.sample_batch(lstm_config_copy.lstm_batch_size)
        print(f"  Sampled batch of {lstm_config_copy.lstm_batch_size} sequences from buffer for training")
        
        # Move batch to device
        state_img_seq = batch["state_imgs"].to("cuda")       # (batch, seq_len, 1, H, W)
        state_float_seq = batch["state_floats"].to("cuda")   # (batch, seq_len, float_input_dim)
        actions_seq = batch["actions"].to("cuda").long()     # (batch, seq_len)
        rewards_seq = batch["rewards"].to("cuda")            # (batch, seq_len)
        next_state_img_seq = batch["next_state_imgs"].to("cuda")  # (batch, seq_len, 1, H, W)
        next_state_float_seq = batch["next_state_floats"].to("cuda")  # (batch, seq_len, float_input_dim)
        gammas_seq = batch["gammas"].to("cuda")              # (batch, seq_len)

        # Forward pass
        with torch.cuda.amp.autocast():
            q_values, _ = online_network(state_img_seq, state_float_seq)
            with torch.no_grad():
                q_next, _ = target_network(next_state_img_seq, next_state_float_seq)
                max_next_q = q_next.max(dim=2)[0]
                targets = rewards_seq + gammas_seq * max_next_q

            loss = dqn_loss(q_values, actions_seq, targets)

        # Print training progress and stats
        print(f"  Training step {step}")
        print(f"    Loss: {loss.item():.6f}")
        print(f"    Q value mean/max/min: {q_values.mean().item():.3f} / {q_values.max().item():.3f} / {q_values.min().item():.3f}")
        print(f"    Target net max_next_q mean: {max_next_q.mean().item():.3f}")
        print(f"    Buffer size: {len(buffer)}")
        if step % 50 == 0:
            # Show some weights stats for debugging if desired
            first_layer = next(online_network.parameters())
            print(f"    First param tensor (mean/std/min/max): {first_layer.mean().item():.3f} / {first_layer.std().item():.3f} / {first_layer.min().item():.3f} / {first_layer.max().item():.3f}")

        # Optimization step
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(online_network.parameters(), config_copy.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Update step counter
        step += 1
        cumul_number_batches_done += 1  # Increment batch counter
        with shared_steps.get_lock():
            shared_steps.value += 1
        print(f"    Shared steps (global): {shared_steps.value}")
        
        # Periodically update target network
        target_network.load_state_dict(online_network.state_dict())
        last_target_update = step
        print(f"    Target network updated at step {step}")
            
        # Periodically update shared network (using send_shared_network_every_n_batches)
        if cumul_number_batches_done % config_copy.send_shared_network_every_n_batches == 0:
            with shared_network_lock:
                uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())
            last_shared_update = step
            print(f"    Shared uncompiled network updated at step {step}")
            
        # Periodically save model and stats
        # Save weights
        torch.save(online_network.state_dict(), save_dir / "lstm_weights1.torch")
        torch.save(target_network.state_dict(), save_dir / "lstm_weights2.torch")
        torch.save(optimizer.state_dict(), save_dir / "lstm_optimizer.torch")
        torch.save(scaler.state_dict(), save_dir / "lstm_scaler.torch")

        # Also save a versioned checkpoint for reference
        save_path = save_dir / f"lstm_model_step_{step}.pt"
        torch.save({
            'online_network': online_network.state_dict(),
            'target_network': target_network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'step': step,
        }, save_path)

        # Save stats
        accumulated_stats["cumul_number_frames_played"] = shared_steps.value
        accumulated_stats["cumul_number_batches_done"] = cumul_number_batches_done
        accumulated_stats["last_step"] = step
        joblib.dump(accumulated_stats, save_dir / "lstm_accumulated_stats.joblib")

        last_save = step
        print(f"    Model checkpoint and stats saved at step {step}")
            
        # Log stats to tensorboard
        tensorboard_writer.add_scalar('train/loss', loss.item(), step)
        tensorboard_writer.add_scalar('train/q_values_mean', q_values.mean().item(), step)
        tensorboard_writer.add_scalar('train/q_values_max', q_values.max().item(), step)
        tensorboard_writer.add_scalar('train/q_values_min', q_values.min().item(), step)
        tensorboard_writer.add_scalar('train/buffer_size', len(buffer), step)
        tensorboard_writer.add_scalar('train/max_next_q', max_next_q.mean().item(), step)

        # Log network parameters and gradients
        total_params = 0
        for name, param in online_network.named_parameters():
            num_param = param.numel()
            total_params += num_param
            tensorboard_writer.add_scalar(f"agent_params/{name.replace('.', '_')}_mean", param.data.mean().item(), step)
            tensorboard_writer.add_scalar(f"agent_params/{name.replace('.', '_')}_std", param.data.std().item(), step)
            tensorboard_writer.add_scalar(f"agent_params/{name.replace('.', '_')}_min", param.data.min().item(), step)
            tensorboard_writer.add_scalar(f"agent_params/{name.replace('.', '_')}_max", param.data.max().item(), step)
            if param.grad is not None:
                tensorboard_writer.add_scalar(f"agent_grads/{name.replace('.', '_')}_grad_mean", param.grad.mean().item(), step)
                tensorboard_writer.add_scalar(f"agent_grads/{name.replace('.', '_')}_grad_std", param.grad.std().item(), step)
        tensorboard_writer.add_scalar("agent_params/total_param_count", total_params, step)
        # Optimizer state log (LR etc.)
        for i, group in enumerate(optimizer.param_groups):
            tensorboard_writer.add_scalar(f"optimizer/group_{i}_lr", group['lr'], step)

        # Log end race stats if available
        if end_race_stats:
            for k, v in end_race_stats.items():
                if isinstance(v, (int, float)):
                    tensorboard_writer.add_scalar(f'race/{k}', v, step)
                    print(f"    End race stat: {k} = {v}")

        # Log wait time stats
        tensorboard_writer.add_scalar('timing/wait_time_total', wait_time_total, step)
