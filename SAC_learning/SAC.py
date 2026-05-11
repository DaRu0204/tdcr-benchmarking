from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ActorCritic import GaussianPolicy, QNetwork


class SAC:
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    batch_size: int = 128
    target_entropy_coef: float = 1.0  # target_entropy = -coef * action_dim

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        device: str | None = None,
        auto_alpha: bool = True,
        init_alpha: float = 0.2,
        lr_actor: float | None = None,
        lr_critic: float | None = None,
        lr_alpha: float | None = None,
        gamma: float | None = None,
        tau: float | None = None,
        batch_size: int | None = None,
        target_entropy_coef: float | None = None,
    ):
        if lr_actor is not None: self.lr_actor = lr_actor
        if lr_critic is not None: self.lr_critic = lr_critic
        if lr_alpha is not None: self.lr_alpha = lr_alpha
        if gamma is not None: self.gamma = gamma
        if tau is not None: self.tau = tau
        if batch_size is not None: self.batch_size = batch_size
        if target_entropy_coef is not None: self.target_entropy_coef = target_entropy_coef

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = float(max_action)
        self.action_dim = int(action_dim)

        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        self.critic = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.auto_alpha = auto_alpha
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(init_alpha), dtype=torch.float32, requires_grad=True, device=self.device
            )
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr_alpha)
            self.target_entropy = - self.target_entropy_coef * action_dim
        else:
            self._alpha = init_alpha

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item() if self.auto_alpha else self._alpha

    def select_action(self, state, eval: bool = False):
        return self.actor.act(state, deterministic=eval)

    def get_action(self, state, eval: bool = False):
        return self.select_action(state, eval=eval)

    def train_step(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return None

        state, action, next_state, reward, done, truncated = replay_buffer.sample(self.batch_size)
        device = self.device
        state = state.to(device); action = action.to(device); next_state = next_state.to(device)
        reward = reward.to(device); done = done.to(device); truncated = truncated.to(device)

        # Treat time-limit truncation as non-terminal
        not_done = 1.0 - (done * (1.0 - truncated))

        with torch.no_grad():
            next_action, next_logp, _, _ = self.actor.sample(next_state)
            q1_t, q2_t = self.critic_target(next_state, next_action)
            q_t_min = torch.min(q1_t, q2_t)
            alpha_t = self.log_alpha.exp() if self.auto_alpha else torch.tensor(self.alpha, device=device)
            target_q = reward + not_done * self.gamma * (q_t_min - alpha_t * next_logp)

        q1, q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()

        pi, logp_pi, _, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha_now = self.log_alpha.exp() if self.auto_alpha else torch.tensor(self.alpha, device=device)
        actor_loss = (alpha_now * logp_pi - q_pi).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss_value = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha_loss_value = float(alpha_loss.item())

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha),
            "alpha_loss": alpha_loss_value,
            "q1_pi_mean": float(q1_pi.mean().item()),
            "q2_pi_mean": float(q2_pi.mean().item()),
            "logp_mean": float(logp_pi.mean().item()),
        }

    def _ckpt_dir(self) -> Path:
        here = Path(__file__).resolve()
        proj_root = here.parents[1]
        d = proj_root / "SACLearnedModel" 
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, filename: str):
        d = self._ckpt_dir()
        torch.save(self.actor.state_dict(), d / f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), d / f"{filename}_critic.pth")
        if self.auto_alpha:
            torch.save({"log_alpha": self.log_alpha.detach().cpu()}, d / f"{filename}_alpha.pth")

    def load(self, filename: str):
        d = self._ckpt_dir()
        self.actor.load_state_dict(torch.load(d / f"{filename}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(d / f"{filename}_critic.pth", map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict())
        alpha_path = d / f"{filename}_alpha.pth"
        if self.auto_alpha and alpha_path.exists():
            pkg = torch.load(alpha_path, map_location="cpu")
            self.log_alpha = torch.tensor(pkg["log_alpha"], dtype=torch.float32, requires_grad=True, device=self.device)
