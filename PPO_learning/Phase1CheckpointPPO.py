#!/usr/bin/env python3
"""PPO adapter for the shared Phase 1 checkpoint selector."""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

from phase1_checkpoint_common import CheckpointSpec, run_phase1_checkpoint_selection
from ContinuumRobot import ContinuumRobotEnv
from PPO import PPO, PPOConfig


def create_agent(env, device):
    cfg = PPOConfig(device=str(device))
    return PPO(
        obs_dim=env.state_dim,
        act_dim=env.action_dim,
        max_action=env.max_action,
        config=cfg,
    )


def load_checkpoint(agent, model_name: str, _episode: int) -> None:
    """PPO saves each checkpoint as a single payload file."""
    agent.load(os.path.join(PROJECT_ROOT, "PPOLearnedModel", f"{model_name}.pth"))


def select_action(agent, state: np.ndarray) -> np.ndarray:
    action, _logp, _value = agent.select_action(state, deterministic=True)
    return np.asarray(action.detach().cpu().numpy(), dtype=float)


SPEC = CheckpointSpec(
    algorithm="ppo",
    model_dir="PPOLearnedModel",
    checkpoint_pattern=re.compile(
        r"^ppo_continuum_robot_s(?P<S>\d+)u(?P<U>[01])r(?P<R>\d+)b(?P<B>[01])_e(?P<E>\d+)\.pth$"
    ),
    create_agent=create_agent,
    load_checkpoint=load_checkpoint,
    select_action=select_action,
)


if __name__ == "__main__":
    raise SystemExit(run_phase1_checkpoint_selection(SPEC, ContinuumRobotEnv, __file__))
