import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer


class DQN:
    lr = 3e-4
    gamma = 0.99
    tau = 0.005
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_episodes = 100

    def __init__(self, state_dim: int, n_actions: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q = QNetwork(state_dim, n_actions).to(self.device)
        self.q_target = QNetwork(state_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=self.lr)
        self.n_actions = n_actions

        self.epsilon = self.eps_start
        self.eps_decay = (self.eps_end / self.eps_start) ** (
            1.0 / max(1, self.eps_decay_episodes)
        )

        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state_np: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            s = torch.tensor(
                state_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            qvals = self.q(s)
            return int(torch.argmax(qvals, dim=1).item())

    def train_step(self, replay: ReplayBuffer, batch_size: int = 256) -> None:
        if len(replay) < batch_size:
            return

        states, actions, next_states, rewards, dones, truncateds = replay.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        truncateds = truncateds.to(self.device)

        # Time-limit truncations are bootstrapped, not treated as true terminals.
        not_done = 1.0 - (dones * (1.0 - truncateds))

        q_pred_all = self.q(states)
        q_pred = q_pred_all.gather(1, actions)

        with torch.no_grad():
            q_next = self.q_target(next_states).max(dim=1, keepdim=True).values
            q_target = rewards + not_done * self.gamma * q_next

        loss = self.criterion(q_pred, q_target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optim.step()

        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                tp.copy_(self.tau * p + (1 - self.tau) * tp)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def _get_model_dir(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        directory = os.path.join(project_root, "DQNLearnedModel")
        os.makedirs(directory, exist_ok=True)
        return directory

    def save(self, filename: str) -> None:
        directory = self._get_model_dir()

        torch.save(self.q.state_dict(), os.path.join(directory, filename + "_q.pth"))
        torch.save(self.q_target.state_dict(), os.path.join(directory, filename + "_q_target.pth"))

    def load(self, filename: str) -> None:
        directory = self._get_model_dir()

        q_state = torch.load(
            os.path.join(directory, filename + "_q.pth"),
            map_location=self.device,
        )
        q_target_state = torch.load(
            os.path.join(directory, filename + "_q_target.pth"),
            map_location=self.device,
        )

        self.q.load_state_dict(q_state)
        self.q_target.load_state_dict(q_target_state)
