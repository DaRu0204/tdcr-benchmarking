import random
from collections import deque, namedtuple
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'truncated'))

class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)
        self.iterations = 0

    def add(self, state, action, next_state, reward, done, truncated):
        self.buffer.append(Transition(state, action, next_state, reward, done, truncated))
        self.iterations += 1

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones, truncateds = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
            torch.tensor(truncateds, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
