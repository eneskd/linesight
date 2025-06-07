"""
LSTM Replay Buffer for episode-based storage and sequence sampling.
Stores complete episodes and samples sequences for LSTM training.
"""

import random
import numpy as np
import torch
from collections import deque
from typing import Dict, List, Any, Optional, Tuple


class SumTree:
    """Sum tree data structure for prioritized sampling."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class LSTMReplayBuffer:
    """
    Replay buffer for LSTM agents that stores complete episodes
    and samples sequences for training.
    """
    
    def __init__(
        self,
        capacity: int,
        seq_len: int,
        use_prioritized: bool = False,
        config: Any = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 0.001,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.use_prioritized = use_prioritized
        self.config = config
        
        # Episode storage
        self.episodes = deque(maxlen=capacity)
        self.position = 0
        
        # Prioritized replay parameters
        if use_prioritized:
            self.tree = SumTree(capacity)
            self.alpha = alpha
            self.beta = beta
            self.beta_annealing = beta_annealing
            self.epsilon = epsilon
            self.max_priority = 1.0
        else:
            self.tree = None
    
    def add_episode(self, episode_data: Dict[str, List[Any]]):
        """
        Add a complete episode to the buffer.
        
        Args:
            episode_data: Dictionary containing episode data with keys:
                - 'frames': List of image observations
                - 'state_float': List of float state vectors
                - 'actions': List of actions taken
                - 'rewards': List of rewards received
                - 'terminals': List of terminal flags
        """
        # Ensure all lists have the same length
        episode_length = len(episode_data['frames'])
        for key, data in episode_data.items():
            if len(data) != episode_length:
                raise ValueError(f"Episode data length mismatch: {key} has {len(data)} items, expected {episode_length}")
        
        # Convert to numpy arrays for efficient storage
        episode = {}
        for key, data in episode_data.items():
            if key == 'frames':
                # Handle image data - ensure proper shape
                episode[key] = np.array(data)
            elif key == 'actions':
                episode[key] = np.array(data, dtype=np.int64)
            elif key == 'terminals':
                episode[key] = np.array(data, dtype=np.bool_)
            else:
                episode[key] = np.array(data, dtype=np.float32)
        
        if self.use_prioritized:
            # Add with maximum priority for new episodes
            self.tree.add(self.max_priority ** self.alpha, episode)
        else:
            # Simple circular buffer
            if len(self.episodes) >= self.capacity:
                self.episodes.popleft()
            self.episodes.append(episode)
            self.position = (self.position + 1) % self.capacity
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of sequences from the buffer.
        
        Args:
            batch_size: Number of sequences to sample
            
        Returns:
            Dictionary of tensors with shape (batch_size, seq_len, ...)
        """
        return self.sample_batch_with_improved_strategy(batch_size)
    
    def sample_batch_with_improved_strategy(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Enhanced sampling that prioritizes important transitions and recent episodes.
        
        This implementation offers three sampling strategies:
        1. Uniform: Standard random sampling
        2. Recency-biased: Favors more recent episodes
        3. Prioritized: Uses TD errors for importance sampling
        
        Returns: batch dict of tensors, each of shape (batch, seq_len, ...)
        """
        if len(self.episodes) == 0 and (not self.use_prioritized or self.tree.n_entries == 0):
            raise ValueError("Cannot sample from empty buffer")
        
        # Get sampling strategy from config
        strategy = getattr(self.config, "sequence_sampling_strategy", "uniform") if self.config else "uniform"
        
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
                
            if num_episodes == 0:
                raise ValueError("No episodes available for sampling")
            
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
                    if self.tree.n_entries == 0:
                        raise ValueError("No episodes in prioritized buffer")
                    episode_idx = random.randint(0, self.tree.n_entries - 1)
                    episode = self.tree.data[episode_idx]
                else:
                    # Sample a random episode
                    if len(self.episodes) == 0:
                        raise ValueError("No episodes in buffer")
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
                    sequence = episode[k][start:min(end, len(episode[k]))]
                    
                    # Pad if necessary
                    if end > len(episode[k]):
                        padding_needed = end - len(episode[k])
                        if k == 'frames':
                            # Pad images with zeros
                            pad_shape = list(sequence.shape)
                            pad_shape[0] = padding_needed
                            padding = np.zeros(pad_shape, dtype=sequence.dtype)
                        elif k == 'actions':
                            # Pad actions with last action
                            padding = np.full(padding_needed, sequence[-1] if len(sequence) > 0 else 0, dtype=sequence.dtype)
                        elif k == 'terminals':
                            # Pad terminals with True (episode ended)
                            padding = np.full(padding_needed, True, dtype=sequence.dtype)
                        else:
                            # Pad other data with zeros
                            pad_shape = list(sequence.shape)
                            pad_shape[0] = padding_needed
                            padding = np.zeros(pad_shape, dtype=sequence.dtype)
                        
                        sequence = np.concatenate([sequence, padding], axis=0)
                    
                    batch[k].append(sequence)
            
        # Stack and convert to torch tensors
        for k in batch:
            if len(batch[k]) > 0:  # Safety check
                batch[k] = torch.from_numpy(np.stack(batch[k])).float()
        
        # Cast actions to long
        if "actions" in batch:
            batch["actions"] = batch["actions"].long()
        
        # Cast terminals to float for loss computation
        if "terminals" in batch:
            batch["terminals"] = batch["terminals"].float()
        
        # Add weights for importance sampling
        if weights is not None:
            if np.max(weights) > 0:  # Safety check
                weights = weights / np.max(weights)
            batch["weights"] = torch.from_numpy(weights).float()
        
        # Add indices for priority updates
        if indices:
            batch["indices"] = indices
        
        return batch
    
    def update_priorities(self, indices: List[Tuple[int, int]], td_errors: np.ndarray):
        """
        Update priorities for prioritized replay.
        
        Args:
            indices: List of (episode_idx, start_pos) tuples
            td_errors: TD errors for priority computation
        """
        if not self.use_prioritized:
            return
        
        for i, (episode_idx, start_pos) in enumerate(indices):
            # Compute priority from TD error
            priority = (abs(td_errors[i].mean()) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            
            # Update priority in tree
            tree_idx = episode_idx + self.capacity - 1
            if 0 <= tree_idx < len(self.tree.tree):
                self.tree.update(tree_idx, priority)
    
    def __len__(self):
        """Return number of episodes stored."""
        if self.use_prioritized:
            return self.tree.n_entries
        else:
            return len(self.episodes)
    
    def is_empty(self):
        """Check if buffer is empty."""
        return len(self) == 0