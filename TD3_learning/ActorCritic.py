import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = float(max_action)

        self.l1  = nn.Linear(state_dim, 128)
        self.l2  = nn.Linear(128, 256)
        self.l3  = nn.Linear(256, 256)
        self.l4  = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.max_action * torch.tanh(self.out(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1  = nn.Linear(state_dim + action_dim, 128)
        self.l2  = nn.Linear(128, 256)
        self.l3  = nn.Linear(256, 256)
        self.l4  = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.out(x)
