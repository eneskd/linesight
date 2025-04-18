import random
import numpy as np
import torch

class LSTMReplayBuffer:
    """
    Stores full episodes (as dicts of arrays/lists) and samples random sub-sequences for LSTM training.
    """
    def __init__(self, capacity, seq_len):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = []
        self.position = 0

    def add_episode(self, episode):
        """
        Add a full episode (dict of lists/arrays) to the buffer.
        """
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
        else:
            self.episodes[self.position] = episode
            self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.episodes)

    def sample_batch(self, batch_size):
        """
        Sample a batch of random sub-sequences from random episodes.
        Returns: batch dict of tensors, each of shape (batch, seq_len, ...)
        """
        batch = {k: [] for k in self.episodes[0].keys()}
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            ep_len = len(ep["actions"])
            if ep_len <= self.seq_len:
                start = 0
            else:
                start = random.randint(0, ep_len - self.seq_len)
            end = start + self.seq_len
            for k in batch:
                batch[k].append(np.array(ep[k][start:end]))
        # Stack and convert to torch tensors
        for k in batch:
            batch[k] = torch.from_numpy(np.stack(batch[k])).float()
        # Cast actions to long
        if "actions" in batch:
            batch["actions"] = batch["actions"].long()
        return batch

# Example usage:
# buffer = LSTMReplayBuffer(capacity=1000, seq_len=16)
# buffer.add_episode(episode_dict)
# batch = buffer.sample_batch(batch_size=32)