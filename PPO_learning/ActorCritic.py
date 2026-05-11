import math
from typing import Tuple, Iterable, Union

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

Tensorable = Union[torch.Tensor, "np.ndarray"]


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Iterable[int] = (128, 256, 256, 128),
        activation: nn.Module = nn.Tanh,
        out_activation: nn.Module = None,
        ortho_init: bool = True,
        last_layer_gain: float = 1.0,
    ):
        super().__init__()
        layers = []
        linears = []
        prev = in_dim
        for hs in hidden_sizes:
            lin = nn.Linear(prev, hs)
            layers += [lin, activation()]
            linears.append(lin)
            prev = hs
        last_lin = nn.Linear(prev, out_dim)
        layers.append(last_lin)
        linears.append(last_lin)
        if out_activation is not None:
            layers.append(out_activation())
        self.net = nn.Sequential(*layers)

        if ortho_init:
            for lin in linears[:-1]:
                nn.init.orthogonal_(lin.weight, gain=math.sqrt(2))
                nn.init.constant_(lin.bias, 0.0)
            nn.init.orthogonal_(linears[-1].weight, gain=last_layer_gain)
            nn.init.constant_(linears[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_action: float,
        hidden_sizes: Iterable[int] = (128, 256, 256, 128),
        activation: nn.Module = nn.Tanh,
        ortho_init: bool = True,
        init_log_std: float = -0.5,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        device: str = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.register_buffer("action_scale", torch.tensor(float(max_action)))
        self.register_buffer("eps", torch.tensor(1e-6))

        self.actor = MLP(
            obs_dim, act_dim, hidden_sizes, activation,
            ortho_init=ortho_init, last_layer_gain=0.01
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * float(init_log_std))
        self.log_std_bounds = log_std_bounds

        self.critic = MLP(
            obs_dim, 1, hidden_sizes, activation,
            ortho_init=ortho_init, last_layer_gain=1.0
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(torch.device(device))

    def _clamped_log_std(self) -> torch.Tensor:
        lo, hi = self.log_std_bounds
        return torch.clamp(self.log_std, lo, hi)

    def _distribution(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mean = self.actor(obs)
        std = self._clamped_log_std().exp()
        return Normal(mean, std), mean

    @staticmethod
    def _tanh_squash(u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u)

    def _scale_action(self, a_tanh: torch.Tensor) -> torch.Tensor:
        return a_tanh * self.action_scale

    def _log_prob_squashed(self, base_dist: Normal, u: torch.Tensor, a_tanh: torch.Tensor) -> torch.Tensor:
        log_prob_u = base_dist.log_prob(u)
        corr = torch.log(1.0 - a_tanh.pow(2) + self.eps)
        return (log_prob_u - corr).sum(-1)

    def _unsquash_action(self, a_scaled: torch.Tensor) -> torch.Tensor:
        a = (a_scaled / (self.action_scale + self.eps)).clamp(-1 + 1e-6, 1 - 1e-6)
        return 0.5 * (torch.log1p(a) - torch.log1p(-a))

    @torch.no_grad()
    def act(self, obs: Tensorable, deterministic: bool = False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device, dtype=torch.float32)

        dist, _ = self._distribution(obs)
        u = dist.mean if deterministic else dist.rsample()
        a_tanh = self._tanh_squash(u)
        a = self._scale_action(a_tanh)
        logp = self._log_prob_squashed(dist, u, a_tanh)
        v = self.critic(obs).squeeze(-1)
        return a, logp, v

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        obs = obs.to(self.device, dtype=torch.float32)
        actions = actions.to(self.device, dtype=torch.float32)

        dist, _ = self._distribution(obs)
        u = self._unsquash_action(actions)
        a_tanh = torch.tanh(u)

        logp = self._log_prob_squashed(dist, u, a_tanh)
        entropy = dist.entropy().sum(-1)
        v = self.critic(obs).squeeze(-1)
        return logp, entropy, v

    @property
    def device(self):
        return next(self.parameters()).device
