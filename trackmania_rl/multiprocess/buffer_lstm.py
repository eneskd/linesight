import random
import numpy as np
import torch
import math
from collections import deque

class SumTree:
    """
    A sum tree data structure for efficient sampling based on priorities.
    Used for prioritized experience replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
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
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class LSTMReplayBuffer:
    """
    Stores full episodes (as dicts of arrays/lists) and samples random sub-sequences for LSTM training.
    Optionally supports prioritized experience replay.
    """
    def __init__(self, capacity, seq_len, use_prioritized=False, alpha=0.6, beta=0.4, beta_annealing=0.001):
        self.capacity = capacity
        self.seq_len = seq_len
        self.use_prioritized = use_prioritized
        
        # Parameters for prioritized experience replay
        self.alpha = alpha  # How much prioritization to use (0 = no prioritization, 1 = full prioritization)
        self.beta = beta    # Importance sampling correction factor
        self.beta_annealing = beta_annealing  # How quickly to increase beta to 1
        self.max_priority = 1.0
        
        # Storage for episodes
        if use_prioritized:
            self.tree = SumTree(capacity)
            self.position = 0
            self.episodes_info = {}  # Store episode metadata
        else:
            self.episodes = []
            self.position = 0

    def add_episode(self, episode):
        """
        Add a full episode (dict of lists/arrays) to the buffer.
        """
        if self.use_prioritized:
            # For prioritized replay, add with max priority to ensure it gets sampled at least once
            self.tree.add(self.max_priority ** self.alpha, episode)
            # Store episode info for later reference
            self.episodes_info[self.position] = {
                'length': len(episode['actions']),
                'td_errors': np.ones(len(episode['actions'])) * self.max_priority
            }
            self.position = (self.position + 1) % self.capacity
        else:
            # Standard uniform replay
            if len(self.episodes) < self.capacity:
                self.episodes.append(episode)
            else:
                self.episodes[self.position] = episode
                self.position = (self.position + 1) % self.capacity

    def update_priorities(self, indices, td_errors):
        """
        Update priorities for prioritized experience replay.
        
        Args:
            indices: List of (episode_idx, start_idx) tuples
            td_errors: List of TD errors for each sequence
        """
        if not self.use_prioritized:
            return
            
        for (episode_idx, start_idx), td_error in zip(indices, td_errors):
            # Update the priority in the sum tree
            priority = (abs(td_error) + 1e-5) ** self.alpha  # Add small constant to avoid zero priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update the episode's TD error at the specific sequence
            for i in range(self.seq_len):
                if start_idx + i < self.episodes_info[episode_idx]['length']:
                    self.episodes_info[episode_idx]['td_errors'][start_idx + i] = priority
            
            # Update the tree with the average priority for the episode
            tree_idx = episode_idx + self.capacity - 1
            avg_priority = np.mean(self.episodes_info[episode_idx]['td_errors'])
            self.tree.update(tree_idx, avg_priority)

    def __len__(self):
        """Return the number of episodes in the buffer."""
        if self.use_prioritized:
            return self.tree.n_entries
        else:
            return len(self.episodes)

    def sample_batch(self, batch_size):
        """
        Sample a batch of random sub-sequences from random episodes.
        Returns: batch dict of tensors, each of shape (batch, seq_len, ...)
        
        For prioritized replay, also returns indices and weights for priority updates.
        """
        batch = {}
        indices = []
        weights = None
        
        if self.use_prioritized:
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
        else:
            # Standard uniform sampling
            for i in range(batch_size):
                ep = random.choice(self.episodes)
                ep_len = len(ep["actions"])
                
                if ep_len <= self.seq_len:
                    start = 0
                else:
                    start = random.randint(0, ep_len - self.seq_len)
                
                end = start + self.seq_len
                
                if i == 0:
                    # Initialize batch dictionary with keys from the first episode
                    batch = {k: [] for k in ep.keys()}
                
                # Add sequence to batch
                for k in batch:
                    batch[k].append(np.array(ep[k][start:end]))
        
        # Stack and convert to torch tensors
        for k in batch:
            batch[k] = torch.from_numpy(np.stack(batch[k])).float()
        
        # Cast actions to long
        if "actions" in batch:
            batch["actions"] = batch["actions"].long()
        
        # Add weights for prioritized replay
        if self.use_prioritized:
            # Normalize weights to [0, 1]
            weights = weights / weights.max()
            batch["weights"] = torch.from_numpy(weights).float()
            batch["indices"] = indices
        
        return batch