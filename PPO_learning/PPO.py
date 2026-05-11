from dataclasses import dataclass
import os
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from ActorCritic import ActorCritic
from RolloutBuffer import RolloutBuffer


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.01

    learning_rate: float = 3e-4
    epochs: int = 10
    minibatch_size: int = 64
    anneal_lr: bool = True

    rollout_steps: int = 2048

    device: Optional[str] = None
    adv_norm_eps: float = 1e-8
    seed: Optional[int] = None


class PPO:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_action: float,
        config: PPOConfig = PPOConfig(),
        actor_critic_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if actor_critic_kwargs is None:
            actor_critic_kwargs = {}

        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        if config.seed is not None:
            import random, numpy as np
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)

        self.ac = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            max_action=max_action,
            device=str(self.device),
            **actor_critic_kwargs,
        )

        self.optimizer = optim.Adam(self.ac.parameters(), lr=config.learning_rate)

        self.buf = RolloutBuffer(
            num_steps=config.rollout_steps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=self.device,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        self.cfg = config
        self.update_idx = 0

    @torch.no_grad()
    def select_action(self, obs, deterministic: bool = False):
        a, logp, v = self.ac.act(obs, deterministic=deterministic)
        return a, logp, v
   
    def collect_rollout(self, env) -> Tuple[torch.Tensor, bool, bool]:
        """
        Collect exactly cfg.rollout_steps transitions from a single env.
        Returns (last_obs, last_done, last_truncated) where last_obs is the
        observation AFTER the final stored transition (used for bootstrapping).
        """
        self.buf.reset()

        obs = env.state if hasattr(env, "state") else env.reset()

        bootstrap_obs = obs
        last_done = False
        last_truncated = False

        for _ in range(self.cfg.rollout_steps):
            with torch.no_grad():
                action_t, logp_t, value_t = self.ac.act(obs, deterministic=False)

            action_np = action_t.detach().cpu().numpy()

            next_obs, reward, done, truncated = env.step(action_np)

            self.buf.add(
                obs=obs,
                action=action_np,
                reward=reward,
                done=done,
                truncated=truncated,
                value=value_t.item(),
                logprob=logp_t.item(),
            )

            bootstrap_obs = next_obs
            last_done = bool(done)
            last_truncated = bool(truncated)

            obs = next_obs
            if done:
                obs = env.reset()

        # Bootstrap from the observation after the final stored transition.
        with torch.no_grad():
            last_obs_t = torch.as_tensor(bootstrap_obs, dtype=torch.float32, device=self.device)
            last_value = self.ac.critic(last_obs_t).squeeze(-1).item()

        self.buf.compute_returns_and_advantages(
            last_value=last_value,
            last_done=last_done,
            last_truncated=last_truncated,
        )

        return torch.as_tensor(bootstrap_obs, dtype=torch.float32, device=self.device), last_done, last_truncated

    def _anneal_lr(self, frac_remaining: float):
        if not self.cfg.anneal_lr:
            return
        lrnow = self.cfg.learning_rate * frac_remaining
        for pg in self.optimizer.param_groups:
            pg["lr"] = lrnow

    def update(self) -> Dict[str, float]:
        adv = self.buf.advantages
        adv = (adv - adv.mean()) / (adv.std() + self.cfg.adv_norm_eps)
        self.buf.advantages.copy_(adv)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }

        n_mb = 0
        for epoch in range(self.cfg.epochs):
            for batch in self.buf.get(batch_size=self.cfg.minibatch_size, shuffle=True):
                new_logprob, entropy, new_value = self.ac.evaluate_actions(batch["obs"], batch["actions"])

                log_ratio = new_logprob - batch["logprobs"]
                ratio = torch.exp(log_ratio)

                unclipped = ratio * batch["advantages"]
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * batch["advantages"]
                policy_loss = -torch.min(unclipped, clipped).mean()

                if self.cfg.vf_clip_coef is not None and self.cfg.vf_clip_coef > 0.0:
                    v_pred_clipped = batch["values"] + (new_value - batch["values"]).clamp(
                        -self.cfg.vf_clip_coef, self.cfg.vf_clip_coef
                    )
                    v_losses = (new_value - batch["returns"]).pow(2)
                    v_losses_clipped = (v_pred_clipped - batch["returns"]).pow(2)
                    value_loss = 0.5 * torch.max(v_losses, v_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (new_value - batch["returns"]).pow(2).mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.cfg.vf_coef * value_loss + self.cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (0.5 * (log_ratio.pow(2))).mean()
                    clip_frac = (torch.abs(ratio - 1.0) > self.cfg.clip_coef).float().mean()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["approx_kl"] += approx_kl.item()
                stats["clip_frac"] += clip_frac.item()
                n_mb += 1

                if self.cfg.target_kl is not None and approx_kl.item() > self.cfg.target_kl:
                    break
            else:
                continue
            break

        for k in stats:
            stats[k] = stats[k] / max(1, n_mb)

        self.update_idx += 1
        return stats

    def learn(self, env, total_updates: int, progress_fn=None):
        for up in range(total_updates):
            frac_remaining = 1.0 - (up / max(1, total_updates))
            self._anneal_lr(frac_remaining)

            last_obs, last_done, last_truncated = self.collect_rollout(env)

            stats = self.update()

            if progress_fn is not None:
                progress_fn(self.update_idx, stats)

        return self

    def save(self, path: str):
        payload = {
            "actor_critic_state_dict": self.ac.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg.__dict__,
            "update_idx": self.update_idx,
        }
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True):
        payload = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(payload["actor_critic_state_dict"], strict=strict)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.update_idx = payload.get("update_idx", 0)
        return self
