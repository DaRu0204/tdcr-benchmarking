import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 128)

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

        self.max_action = float(max_action)

    def forward(self, state: torch.Tensor):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    @torch.no_grad()
    def act(self, state, deterministic: bool = False):
        dev = next(self.parameters()).device
        s = torch.as_tensor(state, dtype=torch.float32, device=dev).reshape(1, -1)
        if deterministic:
            mean, _ = self.forward(s)
            a = torch.tanh(mean)
        else:
            a, _, _, _ = self.sample(s)
        a = self.max_action * a
        return a.squeeze(0).cpu().numpy()

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        u = dist.rsample()
        a = torch.tanh(u)

        # Stable: log(1 - tanh(u)^2) = 2*(log2 - u - softplus(-2u))
        gaussian_logp = dist.log_prob(u).sum(dim=1, keepdim=True)
        tanh_correction = (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=1, keepdim=True)
        logp = gaussian_logp - tanh_correction

        action = self.max_action * a
        return action, logp, u, mean


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        in_dim = state_dim + action_dim

        self.q1_l1 = nn.Linear(in_dim, 128)
        self.q1_l2 = nn.Linear(128, 256)
        self.q1_l3 = nn.Linear(256, 256)
        self.q1_l4 = nn.Linear(256, 128)
        self.q1_out = nn.Linear(128, 1)

        self.q2_l1 = nn.Linear(in_dim, 128)
        self.q2_l2 = nn.Linear(128, 256)
        self.q2_l3 = nn.Linear(256, 256)
        self.q2_l4 = nn.Linear(256, 128)
        self.q2_out = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=1)

        q1 = torch.relu(self.q1_l1(x))
        q1 = torch.relu(self.q1_l2(q1))
        q1 = torch.relu(self.q1_l3(q1))
        q1 = torch.relu(self.q1_l4(q1))
        q1 = self.q1_out(q1)

        q2 = torch.relu(self.q2_l1(x))
        q2 = torch.relu(self.q2_l2(q2))
        q2 = torch.relu(self.q2_l3(q2))
        q2 = torch.relu(self.q2_l4(q2))
        q2 = self.q2_out(q2)

        return q1, q2
