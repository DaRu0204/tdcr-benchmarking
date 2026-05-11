import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(layer):
    fan_in = layer.weight.data.size(0)
    lim = 1.0 / (fan_in ** 0.5)
    layer.weight.data.uniform_(-lim, lim)
    if layer.bias is not None:
        layer.bias.data.uniform_(-lim, lim)

def small_out_init(layer, scale=3e-3):
    nn.init.uniform_(layer.weight, -scale, scale)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -scale, scale)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = float(max_action)

        self.l1  = nn.Linear(state_dim, 128)
        self.l2  = nn.Linear(128, 256)
        self.l3  = nn.Linear(256, 256)
        self.l4  = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_dim)

        self.reset_parameters()

    def reset_parameters(self):
        fanin_init(self.l1)
        fanin_init(self.l2)
        fanin_init(self.l3)
        fanin_init(self.l4)
        small_out_init(self.out, 3e-3)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.max_action * torch.tanh(self.out(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.s1 = nn.Linear(state_dim, 128)

        # DDPG concatenates the action after the first state-only layer.
        self.l2 = nn.Linear(128 + action_dim, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 128)

        self.out = nn.Linear(128, 1)

        self.reset_parameters()

    def reset_parameters(self):
        fanin_init(self.s1)
        fanin_init(self.l2)
        fanin_init(self.l3)
        fanin_init(self.l4)
        small_out_init(self.out, 3e-3)

    def forward(self, state, action):
        h1 = F.relu(self.s1(state))
        x  = torch.cat([h1, action], dim=1)
        x  = F.relu(self.l2(x))
        x  = F.relu(self.l3(x))
        x  = F.relu(self.l4(x))
        return self.out(x)
