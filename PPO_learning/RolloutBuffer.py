from typing import Iterator, Dict, Any, Optional

import torch


class RolloutBuffer:
    """
    Rollout storage for PPO with GAE(λ) and proper time-limit bootstrapping.

    A time-limit truncation bootstraps the value estimate; a true terminal does not.
    """

    def __init__(
        self,
        num_steps: int,
        obs_dim: int,
        act_dim: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.T = int(num_steps)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.gamma = float(gamma)
        self.lmbda = float(gae_lambda)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        self._allocate()
        self.reset()

    def _allocate(self) -> None:
        T, obs_dim, act_dim, dev, dt = self.T, self.obs_dim, self.act_dim, self.device, self.dtype
        self.obs = torch.zeros((T, obs_dim), dtype=dt, device=dev)
        self.actions = torch.zeros((T, act_dim), dtype=dt, device=dev)
        self.rewards = torch.zeros((T,), dtype=dt, device=dev)
        self.dones = torch.zeros((T,), dtype=torch.bool, device=dev)
        self.truncateds = torch.zeros((T,), dtype=torch.bool, device=dev)
        self.values = torch.zeros((T,), dtype=dt, device=dev)
        self.logprobs = torch.zeros((T,), dtype=dt, device=dev)

        self.returns = torch.zeros((T,), dtype=dt, device=dev)
        self.advantages = torch.zeros((T,), dtype=dt, device=dev)

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

    def add(
        self,
        obs,
        action,
        reward: float,
        done: bool,
        truncated: bool,
        value: float,
        logprob: float,
    ) -> None:
        if self.full:
            raise RuntimeError("RolloutBuffer is full. Call reset() or finish_path() first.")

        i = self.ptr
        self.obs[i] = torch.as_tensor(obs, dtype=self.dtype, device=self.device)
        self.actions[i] = torch.as_tensor(action, dtype=self.dtype, device=self.device)
        self.rewards[i] = torch.as_tensor(reward, dtype=self.dtype, device=self.device)
        self.dones[i] = bool(done)
        self.truncateds[i] = bool(truncated)
        self.values[i] = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        self.logprobs[i] = torch.as_tensor(logprob, dtype=self.dtype, device=self.device)

        self.ptr += 1
        if self.ptr >= self.T:
            self.full = True

    @torch.no_grad()
    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        last_truncated: bool,
    ) -> None:
        """
        `last_value` is V(s_T), the observation after the final stored transition.
        """
        if not self.full:
            raise RuntimeError("RolloutBuffer not full yet; cannot compute advantages.")

        dev, dt = self.device, self.dtype
        last_value = torch.as_tensor(last_value, dtype=dt, device=dev)
        last_nonterminal = torch.as_tensor(1.0 - float(last_done) + float(last_truncated), dtype=dt, device=dev)

        adv = torch.zeros(1, dtype=dt, device=dev)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]
            nonterminal = last_nonterminal if t == self.T - 1 else (1.0 - self.dones[t].float() + self.truncateds[t].float())

            delta = self.rewards[t] + self.gamma * next_value * nonterminal - self.values[t]
            adv = delta + self.gamma * self.lmbda * adv * nonterminal
            self.advantages[t] = adv

        self.returns = self.advantages + self.values

    def get(self, batch_size: int, shuffle: bool = True) -> Iterator[Dict[str, torch.Tensor]]:
        if not self.full:
            raise RuntimeError("Buffer not ready. Fill and compute advantages before calling get().")

        T = self.T
        idx = torch.randperm(T, device=self.device) if shuffle else torch.arange(T, device=self.device)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            mb_idx = idx[start:end]
            yield {
                "obs": self.obs[mb_idx],
                "actions": self.actions[mb_idx],
                "logprobs": self.logprobs[mb_idx],
                "advantages": self.advantages[mb_idx],
                "returns": self.returns[mb_idx],
                "values": self.values[mb_idx],
            }

    def __len__(self) -> int:
        return self.T

    def is_full(self) -> bool:
        return self.full
