#!/usr/bin/env python3
"""DDPG adapter for the shared Phase 1 checkpoint selector."""
from __future__ import annotations

import os
import re
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

from phase1_checkpoint_common import CheckpointSpec, run_phase1_checkpoint_selection
from ContinuumRobot import ContinuumRobotEnv
from DDPG import DDPG


def create_agent(env, device):
    return DDPG(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=env.max_action,
        device=device,
    )


def load_checkpoint(agent, model_name: str, _episode: int) -> None:
    """DDPG stores actor/critic files under a shared checkpoint base name."""
    agent.load(model_name)


def select_action(agent, state: np.ndarray) -> np.ndarray:
    return np.asarray(agent.select_action(state), dtype=float)


SPEC = CheckpointSpec(
    algorithm="ddpg",
    model_dir="DDPGLearnedModel",
    checkpoint_pattern=re.compile(
        r"^ddpg_continuum_robot_s(?P<S>\d+)u(?P<U>[01])r(?P<R>\d+)b(?P<B>[01])_e(?P<E>\d+)_actor\.pth$"
    ),
    create_agent=create_agent,
    load_checkpoint=load_checkpoint,
    select_action=select_action,
)


if __name__ == "__main__":
    raise SystemExit(run_phase1_checkpoint_selection(SPEC, ContinuumRobotEnv, __file__))
