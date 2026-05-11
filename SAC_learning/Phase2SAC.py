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
from SAC import SAC


def policy_action(agent, state_np: np.ndarray, cfg) -> np.ndarray:
    state = np.asarray(state_np, dtype=float)
    if _is_stochastic(cfg):
        state = state + np.random.normal(0.0, _noise_sigma(cfg), size=state.shape)
    action = np.asarray(agent.select_action(state, eval=True), dtype=float)
    if _is_stochastic(cfg):
        action = action + np.random.normal(0.0, _noise_sigma(cfg), size=action.shape)
    return action


def load_agent_for_eval(model_name: str, env_kwargs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ContinuumRobotEnv(**({} if env_kwargs is None else dict(env_kwargs)))
    agent = SAC(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=env.max_action,
        device=str(device),
    )
    ckpt_dir = agent._ckpt_dir() if hasattr(agent, "_ckpt_dir") else getattr(agent, "save_dir", "")
    print(f"[load] expecting files in {ckpt_dir} with base '{model_name}'")
    agent.load(model_name)
    agent.actor.eval()
    return agent, env


SPEC = Phase2Spec(
    algorithm="sac",
    default_csv_log_path="phase2_sac_log.csv",
    load_agent_for_eval=load_agent_for_eval,
    policy_action=policy_action,
    script_file=__file__,
)


if __name__ == "__main__":
    raise SystemExit(run_phase2_evaluation(SPEC))
