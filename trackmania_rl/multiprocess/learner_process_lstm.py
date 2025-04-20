import time
import joblib
import os
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    Standard DQN loss: MSE between Q(s,a) and target.
    q_values: (batch, seq_len, n_actions)
    actions: (batch, seq_len) int64
    targets: (batch, seq_len) float32
    weights: (batch,) float32 - importance sampling weights for prioritized replay
    """
    q_selected = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
    td_errors = targets - q_selected
    
    if weights is not None:
        # Apply importance sampling weights
        return (td_errors.pow(2) * weights.unsqueeze(1)).mean()
    else:
        return td_errors.pow(2).mean()

def validate_on_held_out_maps(
    online_network,
    step,
    validation_maps,
    tensorboard_writer,
    save_dir,
    accumulated_stats,
    best_validation_reward
):
    """
    Evaluate the model on held-out validation maps to check for generalization performance
    
    Args:
        online_network: The current LSTM network to evaluate
        step: Current training step
        validation_maps: List of paths to validation maps
        tensorboard_writer: TensorBoard writer for logging
        save_dir: Directory to save best validation model
        accumulated_stats: Dictionary of accumulated statistics
        best_validation_reward: Current best validation reward
    
    Returns:
        The updated best validation reward
    """
    print(f"\nRunning validation at step {step}...")
    
    if not validation_maps:
        print("No validation maps defined. Skipping validation.")
        return best_validation_reward
    
    # Import necessary components for validation
    from trackmania_rl.tmi_interaction.environment import TrackmaniaEnvironment
    from trackmania_rl.multiprocess.collector_process_lstm import LSTMInferer
    
    # Configure inferer for validation
    inferer = LSTMInferer(online_network, "cuda", lstm_config_copy)
    
    results = {}
    
    for map_path in validation_maps:
        map_name = os.path.basename(map_path)
        print(f"Validating on map: {map_name}")
        
        # Create environment with this map
        try:
            env = TrackmaniaEnvironment(
                map_path=map_path,
                game_config=config_copy,
                headless=True
            )
        except Exception as e:
            print(f"Error creating environment for map {map_name}: {e}")
            continue
        
        # Run several episodes on this map
        map_rewards = []
        map_lengths = []
        num_validation_episodes = getattr(lstm_config_copy, "num_validation_episodes", 5)
        
        for episode in range(num_validation_episodes):
            episode_reward = 0
            episode_length = 0
            done = False
            
            try:
                obs = env.reset()
                
                # Reset hidden state at the start of each episode
                inferer.reset_hidden_state()
                
                while not done:
                    # Process observation
                    processed_image = obs.get('image')  # Adjust based on your preprocessing
                    float_inputs = torch.tensor(obs.get('inputs', []), dtype=torch.float32)
                    
                    # Get action from the network, passing and updating hidden state
                    q_values = inferer(processed_image, float_inputs, obs.get('info', {}))
                    action = q_values.argmax().item()
                    
                    # Execute action
                    next_obs, reward, done, info = env.step(action)
                    
                    # Update tracking variables
                    episode_reward += reward
                    episode_length += 1
                    obs = next_obs
                    
                    # Optional early stopping for very long episodes
                    max_episode_length = getattr(lstm_config_copy, "max_validation_episode_length", 2000)
                    if episode_length >= max_episode_length:
                        done = True
                
                map_rewards.append(episode_reward)
                map_lengths.append(episode_length)
                
                print(f"  Episode {episode+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
                
            except Exception as e:
                print(f"Error during validation episode {episode+1} on map {map_name}: {e}")
                continue
        
        # Store results for this map if we have any valid episodes
        if map_rewards:
            results[map_name] = {
                'mean_reward': np.mean(map_rewards),
                'std_reward': np.std(map_rewards),
                'mean_length': np.mean(map_lengths),
                'std_length': np.std(map_lengths),
            }
        
        # Close the environment
        try:
            env.close()
        except:
            pass
    
    # Log the validation results
    print("\nValidation Results:")
    
    if not results:
        print("  No valid validation results collected.")
        return best_validation_reward
    
    for map_name, stats in results.items():
        print(f"  {map_name}: Mean Reward = {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        
        # Log to TensorBoard
        tensorboard_writer.add_scalar(f'validation/{map_name}/mean_reward', stats['mean_reward'], step)
        tensorboard_writer.add_scalar(f'validation/{map_name}/mean_length', stats['mean_length'], step)
    
    # Calculate and log the average across all validation maps
    mean_validation_reward = np.mean([stats['mean_reward'] for stats in results.values()])
    tensorboard_writer.add_scalar('validation/overall_mean_reward', mean_validation_reward, step)
    
    # Store best validation performance
    new_best = False
    if mean_validation_reward > best_validation_reward:
        new_best = True
        best_validation_reward = mean_validation_reward
        
        # Save a special validation checkpoint
        validation_checkpoint_path = save_dir / 'lstm_best_validation_model.pt'
        torch.save({
            'online_network': online_network.state_dict(),
            'train_step': step,
            'accumulated_stats': accumulated_stats,
            'validation_reward': mean_validation_reward,
            'validation_results': results
        }, validation_checkpoint_path)
        print(f"New best validation performance! Saved model with reward {mean_validation_reward:.2f}")
    
    print("Validation complete\n")
    return best_validation_reward

def learner_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
):
    # Add this configuration parameter
    use_double_q = getattr(lstm_config_copy, "use_double_q", True)
    online_network, uncompiled_online_network = make_untrained_lstm_agent(
        jit=config_copy.use_jit,
        is_inference=False,
    )
    target_network, _ = make_untrained_lstm_agent(
        jit=config_copy.use_jit,
        is_inference=False,
    )

    # Initialize accumulated_stats only once
    accumulated_stats = defaultdict(int)
    try:
        loaded_stats = joblib.load(save_dir / "lstm_accumulated_stats.joblib")
        accumulated_stats.update(loaded_stats)
        shared_steps.value = accumulated_stats.get("cumul_number_frames_played", 0)
        print(" =====================      Learner LSTM stats loaded !      ============================")
    except Exception as e:
        print(" Learner LSTM could not load stats:", e)

    optimizer = torch.optim.RAdam(
        online_network.parameters(),
        lr=utilities.from_exponential_schedule(config_copy.lr_schedule, accumulated_stats["cumul_number_frames_played"]),
        eps=config_copy.adam_epsilon,
        betas=(config_copy.adam_beta1, config_copy.adam_beta2),
    )
    scaler = torch.amp.GradScaler("cuda")
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + "_lstm")))
    
    # Print Double Q-learning status
    print(f"Using Double Q-learning: {use_double_q}")
    
    # Initialize learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Since we want to maximize reward
        factor=getattr(lstm_config_copy, "lr_reduction_factor", 0.5),  # Multiply LR by this factor on plateau
        patience=getattr(lstm_config_copy, "lr_patience", 5000),  # Number of steps with no improvement before reducing LR
        verbose=True,
        threshold=getattr(lstm_config_copy, "lr_threshold", 0.01),  # Minimum significant improvement
        min_lr=getattr(lstm_config_copy, "min_lr", 1e-6)  # Minimum learning rate
    )
    
    # Track performance for LR scheduling
    performance_metric_history = deque(maxlen=getattr(lstm_config_copy, "performance_history_size", 100))

    # Create LSTM buffer with prioritized experience replay if enabled
    # Note: The buffer should maintain sequence continuity for proper LSTM training
    if getattr(lstm_config_copy, "use_prioritized_replay", False):
        buffer = LSTMReplayBuffer(
            capacity=lstm_config_copy.replay_buffer_capacity, 
            seq_len=lstm_config_copy.lstm_seq_len,
            use_prioritized=True,
            alpha=getattr(lstm_config_copy, "prio_alpha", 0.6),
            beta=getattr(lstm_config_copy, "prio_beta", 0.4),
            beta_annealing=getattr(lstm_config_copy, "prio_beta_annealing", 0.001)
        )
    else:
        buffer = LSTMReplayBuffer(
            capacity=lstm_config_copy.replay_buffer_capacity, 
            seq_len=lstm_config_copy.lstm_seq_len
        )
        
    # Define max number of episodes to track
    max_episodes_to_track = 100
    
    # Load episode rewards history if available
    if "episode_rewards" in accumulated_stats:
        # Convert existing rewards to deque with max length
        episode_rewards = deque(accumulated_stats["episode_rewards"], maxlen=max_episodes_to_track)
        print(f" Loaded {len(episode_rewards)} previous episode rewards")
    else:
        episode_rewards = deque(maxlen=max_episodes_to_track)
        
    # Load best average reward if available
    if "best_avg_reward" in accumulated_stats:
        best_avg_reward = accumulated_stats["best_avg_reward"]
        print(f" Loaded previous best average reward: {best_avg_reward:.2f}")
        
    # Load performance metric history if available
    if "performance_metric_history" in accumulated_stats:
        performance_metric_history = deque(
            accumulated_stats["performance_metric_history"], 
            maxlen=getattr(lstm_config_copy, "performance_history_size", 100)
        )
        print(f" Loaded {len(performance_metric_history)} previous performance metrics")
    else:
        performance_metric_history = deque(maxlen=getattr(lstm_config_copy, "performance_history_size", 100))
    
    # =========================
    # Load existing weights/stats if available
    # =========================
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
        
    # Try to load scheduler state if it exists
    try:
        scheduler_state = torch.load(save_dir / "lstm_scheduler.torch")
        lr_scheduler.load_state_dict(scheduler_state)
        print(" =====================     Learner LSTM scheduler loaded !     ============================")
    except Exception as e:
        print(" Learner LSTM could not load scheduler state:", e)

    try:
        scaler.load_state_dict(torch.load(save_dir / "lstm_scaler.torch"))
        print(" =====================     Learner LSTM scaler loaded !     ============================")
    except Exception as e:
        print(" Learner LSTM could not load scaler state:", e)

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())



    # Ensure required keys exist in accumulated_stats
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
    
    # Define update frequencies
    target_update_frequency = getattr(lstm_config_copy, "target_update_frequency", 10)
    network_update_frequency = getattr(lstm_config_copy, "network_update_frequency", 
                                      config_copy.send_shared_network_every_n_batches)
    save_frequency = getattr(lstm_config_copy, "save_frequency", 1000)
    validation_frequency = getattr(lstm_config_copy, "validation_frequency", 10000)
    
    # Performance tracking for conditional saving
    best_avg_reward = float('-inf')
    best_validation_reward = float('-inf')
    episode_rewards = []
    reward_improvement_threshold = getattr(lstm_config_copy, "reward_improvement_threshold", 1.05)  # 5% improvement
    
    print("Learner process started.")
    print(f"Replay buffer capacity: {buffer.capacity}, LSTM sequence length: {buffer.seq_len}")
    print(f"Batch size: {lstm_config_copy.lstm_batch_size}")
    print(f"Initial shared step value: {shared_steps.value}")
    print(f"Target update frequency: {target_update_frequency} steps")
    print(f"Network update frequency: {network_update_frequency} steps")
    print(f"Save frequency: {save_frequency} steps")
    print(f"Validation frequency: {validation_frequency} steps")
    
    # Get validation maps if defined
    validation_maps = getattr(lstm_config_copy, "validation_maps", [])
    if validation_maps:
        print(f"Found {len(validation_maps)} validation maps for periodic evaluation")
    else:
        print("No validation maps defined. Validation will be skipped.")

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

        # Check if we have explicit "done" flags, otherwise assume only the last state is terminal
        dones = rollout_results.get("dones", np.zeros(n_frames, dtype=bool))
        if not np.any(dones) and n_frames > 0:
            dones[-1] = True  # Mark last state as terminal if no explicit dones
            
        # Create gammas with 0 for terminal states
        current_gamma = utilities.from_linear_schedule(config_copy.gamma_schedule, accumulated_stats["cumul_number_frames_played"])
        gammas = np.ones(n_frames) * current_gamma
        gammas[dones] = 0.0  # Zero out gamma for terminal states
        
        # Prepare episode data for LSTM buffer
        episode = {
            "state_imgs": rollout_results["frames"],         # list of (1, H, W)
            "state_floats": rollout_results["state_float"],  # list of (float_input_dim,)
            "actions": rollout_results["actions"],           # list of ints
            "rewards": rewards,                              # list of floats
            "next_state_imgs": rollout_results.get("next_frames", rollout_results["frames"][1:] + [rollout_results["frames"][-1]]),
            "next_state_floats": rollout_results.get("next_state_float", rollout_results["state_float"][1:] + [rollout_results["state_float"][-1]]),
            "gammas": gammas,                                # list of floats with 0 for terminal states
            "dones": dones.astype(int)                       # Store done flags for use by replay buffer/sampler
        }
        
        # Check if episode is too large for buffer
        if n_frames > buffer.capacity:
            print(f"WARNING: Episode length ({n_frames}) exceeds buffer capacity ({buffer.capacity}). Trimming episode.")
            # Trim the episode to fit in buffer if needed
            max_frames = buffer.capacity - 1  # Leave room for buffer overhead
            
            # Trim all episode components
            for key in episode:
                if isinstance(episode[key], list):
                    episode[key] = episode[key][:max_frames]
                elif isinstance(episode[key], np.ndarray):
                    episode[key] = episode[key][:max_frames]
                    
            print(f"  Episode trimmed to {max_frames} frames")
            n_frames = max_frames  # Update n_frames to reflect the new size
        
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
        
        # Extract prioritized replay specific components if present
        weights = batch.get("weights", None)
        indices = batch.get("indices", None)
        
        if weights is not None:
            weights = weights.to("cuda")

        # Forward pass
        with torch.cuda.amp.autocast():
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
                    dones_seq = batch.get("dones", torch.zeros_like(rewards_seq)).to("cuda")
                    future_rewards = gammas_seq * q_next_selected * (1.0 - dones_seq)
                    targets = rewards_seq + future_rewards
                else:
                    # Standard Q-learning implementation (original code)
                    # Process sequences through target network, maintaining hidden state
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
                    dones_seq = batch.get("dones", torch.zeros_like(rewards_seq)).to("cuda")
                    future_rewards = gammas_seq * max_next_q * (1.0 - dones_seq)
                    targets = rewards_seq + future_rewards

            # Get done flags if available
            dones_seq = batch.get("dones", torch.zeros_like(rewards_seq)).to("cuda")
            
            # Calculate TD errors for loss and priority updates
            q_selected = q_values.gather(2, actions_seq.unsqueeze(2)).squeeze(2)
            td_errors = targets - q_selected
            
            # Standard DQN loss calculation
            if weights is not None:
                # Apply importance sampling weights for prioritized replay
                loss = (td_errors.pow(2) * weights.unsqueeze(1)).mean()
            else:
                loss = td_errors.pow(2).mean()
        
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
        has_numerical_issues |= check_tensor_for_issues(q_selected, "q_selected")
        has_numerical_issues |= check_tensor_for_issues(td_errors, "td_errors")
        has_numerical_issues |= check_tensor_for_issues(loss, "loss")

        # Print training progress and stats
        print(f"  Training step {step}")
        print(f"    Loss: {loss.item():.6f}")
        print(f"    Q value mean/max/min: {q_values.mean().item():.3f} / {q_values.max().item():.3f} / {q_values.min().item():.3f}")
        
        if use_double_q:
            print(f"    Double Q-learning: Target net selected Q mean: {q_next_selected.mean().item():.3f}")
        else:
            print(f"    Standard Q-learning: Target net max_next_q mean: {max_next_q.mean().item():.3f}")
            
        print(f"    Buffer size: {len(buffer)}")
        if step % 50 == 0:
            # Show some weights stats for debugging if desired
            first_layer = next(online_network.parameters())
            print(f"    First param tensor (mean/std/min/max): {first_layer.mean().item():.3f} / {first_layer.std().item():.3f} / {first_layer.min().item():.3f} / {first_layer.max().item():.3f}")

        # Skip optimization if numerical issues detected
        if has_numerical_issues:
            print("  WARNING: Numerical issues detected. Skipping optimization for this batch.")
            tensorboard_writer.add_scalar('errors/numerical_issues', 1, step)
        else:
            # Optimization step
            optimizer.zero_grad(set_to_none=True)
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
                tensorboard_writer.add_scalar('errors/gradient_issues', 1, step)
            else:
                # Apply gradient clipping and update parameters
                torch.nn.utils.clip_grad_norm_(online_network.parameters(), config_copy.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        
        # Update priorities in buffer if using prioritized replay
        if indices is not None:
            with torch.no_grad():
                # Use absolute TD errors as priorities
                new_priorities = td_errors.abs().mean(dim=1).cpu().numpy()  # Mean across sequence dimension
                buffer.update_priorities(indices, new_priorities)
        
        # Update step counter
        step += 1
        cumul_number_batches_done += 1  # Increment batch counter
        with shared_steps.get_lock():
            shared_steps.value += 1
        print(f"    Shared steps (global): {shared_steps.value}")
        
        # Periodically update target network
        if step - last_target_update >= target_update_frequency:
            target_network.load_state_dict(online_network.state_dict())
            last_target_update = step
            print(f"    Target network updated at step {step}")
            
        # Periodically update shared network
        if step - last_shared_update >= network_update_frequency:
            with shared_network_lock:
                uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())
            last_shared_update = step
            print(f"    Shared uncompiled network updated at step {step}")
            
        # Track episode rewards for performance-based saving and LR scheduling
        if end_race_stats and "total_reward" in end_race_stats:
            episode_rewards.append(end_race_stats["total_reward"])
            # No need to manually pop since deque handles max length
            
            # Update performance metrics for learning rate scheduling
            if len(episode_rewards) >= 10:
                current_avg_reward = np.mean(episode_rewards)
                performance_metric_history.append(current_avg_reward)
                
                # Calculate a stable performance metric by averaging recent performance
                if len(performance_metric_history) >= 10:
                    avg_performance = np.mean(performance_metric_history)
                    
                    # Update learning rate based on performance
                    old_lr = optimizer.param_groups[0]['lr']
                    lr_scheduler.step(avg_performance)
                    new_lr = optimizer.param_groups[0]['lr']
                    
                    # Log if learning rate changed
                    if old_lr != new_lr:
                        print(f"    Learning rate adjusted: {old_lr:.6f} -> {new_lr:.6f}")
                    
                    # Log current learning rate
                    tensorboard_writer.add_scalar('train/learning_rate', new_lr, step)
        
        # Conditional model saving logic
        should_save = False
        current_avg_reward = None
        save_reason = ""
        
        # Option 1: Save periodically based on frequency
        if step - last_save >= save_frequency:
            should_save = True
            save_reason = f"periodic checkpoint at step {step}"
        
        # Option 2: Save if performance has improved significantly
        if len(episode_rewards) >= 10:  # Need at least 10 episodes to calculate meaningful average
            current_avg_reward = np.mean(episode_rewards)
            
            # Save if we have a new best performance (improvement of at least threshold %)
            if current_avg_reward > best_avg_reward * reward_improvement_threshold:
                best_avg_reward = current_avg_reward
                should_save = True
                save_reason = f"performance improved! New best avg reward: {current_avg_reward:.2f}"
                
                # Also save a special "best" checkpoint
                best_checkpoint_path = save_dir / "lstm_best_model.pt"
                torch.save({
                    'online_network': online_network.state_dict(),
                    'target_network': target_network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'step': step,
                    'accumulated_stats': accumulated_stats,
                    'avg_reward': current_avg_reward
                }, best_checkpoint_path)
                print(f"    Saved new best model with avg reward {current_avg_reward:.2f}")
        
        # Save if conditions are met
        if should_save:
            # Save weights
            torch.save(online_network.state_dict(), save_dir / "lstm_weights1.torch")
            torch.save(target_network.state_dict(), save_dir / "lstm_weights2.torch")
            torch.save(optimizer.state_dict(), save_dir / "lstm_optimizer.torch")
            torch.save(scaler.state_dict(), save_dir / "lstm_scaler.torch")
            torch.save(lr_scheduler.state_dict(), save_dir / "lstm_scheduler.torch")
            
            # Also save a versioned checkpoint for reference
            save_path = save_dir / f"lstm_model_step_{step}.pt"
            torch.save({
                'online_network': online_network.state_dict(),
                'target_network': target_network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'step': step,
                'accumulated_stats': accumulated_stats,
                'avg_reward': current_avg_reward if current_avg_reward is not None else None
            }, save_path)
            
            # Save stats
            accumulated_stats["cumul_number_frames_played"] = shared_steps.value
            accumulated_stats["cumul_number_batches_done"] = cumul_number_batches_done
            accumulated_stats["last_step"] = step
            accumulated_stats["best_avg_reward"] = best_avg_reward
            accumulated_stats["best_validation_reward"] = best_validation_reward
            accumulated_stats["episode_rewards"] = list(episode_rewards)
            accumulated_stats["performance_metric_history"] = list(performance_metric_history)
            accumulated_stats["current_learning_rate"] = optimizer.param_groups[0]['lr']
            
            # Save both individual files and the combined checkpoint
            joblib.dump(accumulated_stats, save_dir / "lstm_accumulated_stats.joblib")
            
            last_save = step
            print(f"    Model checkpoint and stats saved at step {step} - Reason: {save_reason}")
            
        # Log stats to tensorboard
        tensorboard_writer.add_scalar('train/loss', loss.item(), step)
        tensorboard_writer.add_scalar('train/q_values_mean', q_values.mean().item(), step)
        tensorboard_writer.add_scalar('train/q_values_max', q_values.max().item(), step)
        tensorboard_writer.add_scalar('train/q_values_min', q_values.min().item(), step)
        tensorboard_writer.add_scalar('train/buffer_size', len(buffer), step)
        
        # Log different metrics based on whether we're using Double Q or not
        if use_double_q:
            tensorboard_writer.add_scalar('train/next_q_selected', q_next_selected.mean().item(), step)
            tensorboard_writer.add_scalar('train/double_q_action_diff', 
                                         (q_next.max(dim=2)[1] != next_actions.squeeze(2)).float().mean().item(), step)
        else:
            tensorboard_writer.add_scalar('train/max_next_q', max_next_q.mean().item(), step)
            
        tensorboard_writer.add_scalar('train/steps_since_target_update', step - last_target_update, step)
        
        # Log numerical stability metrics
        if not hasattr(accumulated_stats, "numerical_issues_count"):
            accumulated_stats["numerical_issues_count"] = 0
        if not hasattr(accumulated_stats, "gradient_issues_count"):
            accumulated_stats["gradient_issues_count"] = 0
            
        if has_numerical_issues:
            accumulated_stats["numerical_issues_count"] += 1
            tensorboard_writer.add_scalar('errors/numerical_issues_cumulative', 
                                         accumulated_stats["numerical_issues_count"], step)
        
        if 'grad_has_issues' in locals() and grad_has_issues:
            accumulated_stats["gradient_issues_count"] += 1
            tensorboard_writer.add_scalar('errors/gradient_issues_cumulative', 
                                         accumulated_stats["gradient_issues_count"], step)
        
        # Log prioritized replay specific metrics if enabled
        if hasattr(buffer, "use_prioritized") and buffer.use_prioritized:
            if hasattr(buffer, "beta"):
                tensorboard_writer.add_scalar('train/priority_beta', buffer.beta, step)
            if weights is not None:
                tensorboard_writer.add_scalar('train/priority_weights_mean', weights.mean().item(), step)
                tensorboard_writer.add_scalar('train/priority_weights_max', weights.max().item(), step)

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
            
        # Log performance metrics used for LR scheduling
        if len(performance_metric_history) > 0:
            tensorboard_writer.add_scalar('performance/lr_scheduling_metric', 
                                         np.mean(performance_metric_history), step)

        # Log end race stats if available
        if end_race_stats:
            for k, v in end_race_stats.items():
                if isinstance(v, (int, float)):
                    tensorboard_writer.add_scalar(f'race/{k}', v, step)
                    print(f"    End race stat: {k} = {v}")
            
            # Log average reward over last episodes
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                tensorboard_writer.add_scalar('performance/avg_reward_last_episodes', avg_reward, step)
                tensorboard_writer.add_scalar('performance/best_avg_reward', best_avg_reward, step)

        # Log wait time stats
        tensorboard_writer.add_scalar('timing/wait_time_total', wait_time_total, step)
        
        # Run validation on held-out maps if it's time
        if validation_maps and step % validation_frequency == 0:
            validate_on_held_out_maps(
                online_network=online_network,
                step=step,
                validation_maps=validation_maps,
                tensorboard_writer=tensorboard_writer,
                save_dir=save_dir,
                accumulated_stats=accumulated_stats,
                best_validation_reward=best_validation_reward
            )
