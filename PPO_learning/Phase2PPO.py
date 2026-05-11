#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

from phase2_common import Phase2Spec, _is_stochastic, _noise_sigma, run_phase2_evaluation
from ContinuumRobot import ContinuumRobotEnv
from PPO import PPO, PPOConfig


def policy_action(agent, state_np: np.ndarray, cfg) -> np.ndarray:
    state = np.asarray(state_np, dtype=float)
    if _is_stochastic(cfg):
        state = state + np.random.normal(0.0, _noise_sigma(cfg), size=state.shape)
    with torch.no_grad():
        action_t, _logp, _value = agent.select_action(state, deterministic=True)
    action = np.asarray(action_t.detach().cpu().numpy(), dtype=float).reshape(-1)
    if _is_stochastic(cfg):
        action = action + np.random.normal(0.0, _noise_sigma(cfg), size=action.shape)
    return action


def load_agent_for_eval(model_name: str, env_kwargs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ContinuumRobotEnv(**({} if env_kwargs is None else dict(env_kwargs)))
    agent = PPO(
        obs_dim=env.state_dim,
        act_dim=env.action_dim,
        max_action=float(env.max_action),
        config=PPOConfig(device=str(device)),
    )
    ckpt_path = model_name if model_name.endswith(".pth") else f"{model_name}.pth"
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "PPOLearnedModel", ckpt_path)
    agent.save_dir = os.path.dirname(os.path.abspath(ckpt_path))
    agent.device = device
    print(f"[load] expecting files in {agent.save_dir} with base '{model_name}'")
    agent.load(ckpt_path)
    return agent, env


SPEC = Phase2Spec(
    algorithm="ppo",
    default_csv_log_path="phase2_ppo_log.csv",
    load_agent_for_eval=load_agent_for_eval,
    policy_action=policy_action,
    script_file=__file__,
)


if __name__ == "__main__":
    raise SystemExit(run_phase2_evaluation(SPEC))
