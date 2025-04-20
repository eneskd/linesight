def sample_batch_with_improved_strategy(self, batch_size):
    """
    Enhanced sampling that prioritizes important transitions and recent episodes.
    
    This implementation offers three sampling strategies:
    1. Uniform: Standard random sampling
    2. Recency-biased: Favors more recent episodes
    3. Prioritized: Uses TD errors for importance sampling
    
    Returns: batch dict of tensors, each of shape (batch, seq_len, ...)
    """
    # Get sampling strategy from config
    strategy = getattr(self.config, "sequence_sampling_strategy", "uniform")
    print(f"Using {strategy} sequence sampling strategy")
    
    batch = {}
    indices = []
    weights = None
    
    if strategy == "prioritized" and self.use_prioritized:
        # Increase beta over time to reduce importance sampling bias
        self.beta = min(1.0, self.beta + self.beta_annealing)
        
        # Calculate weights for importance sampling
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Sample from priority tree
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, episode = self.tree.get(s)
            episode_idx = idx - self.capacity + 1
            
            # Sample a random starting point in the episode
            ep_len = len(episode["actions"])
            if ep_len <= self.seq_len:
                start = 0
            else:
                start = random.randint(0, ep_len - self.seq_len)
            
            # Store the episode index and start position for priority updates
            indices.append((episode_idx, start))
            
            # Calculate importance sampling weight
            # p(i) = priority^alpha / sum(priority^alpha)
            # weight = (1/N * 1/p(i))^beta = (N * p(i))^-beta
            sampling_prob = priority / self.tree.total()
            weights[i] = (batch_size * sampling_prob) ** -self.beta
            
            # Extract sequence from episode
            if i == 0:
                # Initialize batch dictionary with keys from the first episode
                batch = {k: [] for k in episode.keys()}
            
            # Add sequence to batch
            end = start + self.seq_len
            for k in batch:
                batch[k].append(np.array(episode[k][start:end]))
        
    elif strategy == "recency_biased":
        # Recency-biased sampling: favor more recent episodes
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate episode weights based on recency
        if self.use_prioritized:
            num_episodes = self.tree.n_entries
        else:
            num_episodes = len(self.episodes)
            
        # Linearly decreasing weights from newest to oldest
        episode_weights = np.linspace(1.0, 0.1, num_episodes)
        
        for i in range(batch_size):
            # Sample episode based on recency weight
            if self.use_prioritized:
                # For prioritized buffer, adjust the sampling
                episode_idx = np.random.choice(num_episodes, p=episode_weights/episode_weights.sum())
                episode = self.tree.data[episode_idx]
            else:
                # For standard buffer, more recent episodes are at the end
                # We need to convert linear weights to sample from the end
                episode_idx = np.random.choice(num_episodes, p=episode_weights/episode_weights.sum())
                # Map this to actual index in the buffer
                actual_idx = (self.position - 1 - episode_idx) % num_episodes
                episode = self.episodes[actual_idx]
            
            # Sample sequence from this episode
            ep_len = len(episode["actions"])
            if ep_len <= self.seq_len:
                start = 0
            else:
                start = np.random.randint(0, ep_len - self.seq_len)
            
            # Store the episode index and start position
            indices.append((episode_idx, start))
            
            # Extract sequence
            if i == 0:
                batch = {k: [] for k in episode.keys()}
            
            # Add sequence to batch
            end = start + self.seq_len
            for k in batch:
                batch[k].append(np.array(episode[k][start:end]))
    
    else:  # "uniform" sampling
        # Standard uniform sampling
        for i in range(batch_size):
            if self.use_prioritized:
                # For prioritized buffer, sample randomly from all episodes
                episode_idx = random.randint(0, self.tree.n_entries - 1)
                episode = self.tree.data[episode_idx]
            else:
                # Sample a random episode
                episode_idx = random.randint(0, len(self.episodes) - 1)
                episode = self.episodes[episode_idx]
            
            # Sample a random starting point in the episode
            ep_len = len(episode["actions"])
            if ep_len <= self.seq_len:
                start = 0
            else:
                start = random.randint(0, ep_len - self.seq_len)
            
            # Store the episode index and start position for potential priority updates
            indices.append((episode_idx, start))
            
            # Extract sequence from episode
            if i == 0:
                # Initialize batch dictionary with keys from the first episode
                batch = {k: [] for k in episode.keys()}
            
            # Add sequence to batch
            end = start + self.seq_len
            for k in batch:
                batch[k].append(np.array(episode[k][start:min(end, len(episode[k]))]))
                
                # Pad if necessary
                if end > len(episode[k]):
                    padding_needed = end - len(episode[k])
                    pad_shape = list(batch[k][-1].shape)
                    pad_shape[0] = padding_needed
                    padding = np.zeros(pad_shape)
                    batch[k][-1] = np.concatenate([batch[k][-1], padding], axis=0)
        
    # Stack and convert to torch tensors
    for k in batch:
        if len(batch[k]) > 0:  # Safety check
            batch[k] = torch.from_numpy(np.stack(batch[k])).float()
    
    # Cast actions to long
    if "actions" in batch:
        batch["actions"] = batch["actions"].long()
    
    # Add weights for importance sampling
    if weights is not None:
        if np.max(weights) > 0:  # Safety check
            weights = weights / np.max(weights)
        batch["weights"] = torch.from_numpy(weights).float()
    
    # Add indices for priority updates
    if indices:
        batch["indices"] = indices
    
    return batch