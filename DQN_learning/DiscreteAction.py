import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ContinuumRobot import ContinuumRobotEnv


class DiscreteActionEnv:
    """DQN wrapper with the fixed 22-action catalogue from the benchmark."""

    def __init__(self, base_env: ContinuumRobotEnv | None = None):
        self.base = base_env if base_env is not None else ContinuumRobotEnv()
        self.state_dim = self.base.state_dim

        # Catalogue entries are encoded tendon increments; the base environment applies clipping.
        a01 = np.array([+0.1, +0.1, -0.1])
        a02 = np.array([+0.1, -0.1, +0.1])
        a03 = np.array([-0.1, +0.1, +0.1])
        a04 = np.array([-0.1, -0.1, +0.1])
        a05 = np.array([-0.1, +0.1, -0.1])
        a06 = np.array([+0.1, -0.1, -0.1])

        b01 = np.array([+0.2, +0.2, -0.2])
        b02 = np.array([+0.2, -0.2, +0.2])
        b03 = np.array([-0.2, +0.2, +0.2])
        b04 = np.array([-0.2, -0.2, +0.2])
        b05 = np.array([-0.2, +0.2, -0.2])
        b06 = np.array([+0.2, -0.2, -0.2])

        lift_01 = np.array([+0.1, 0.0, 0.0])
        lift_02 = np.array([0.0, +0.1, 0.0])
        lift_03 = np.array([0.0, 0.0, +0.1])

        lift_21 = np.array([+0.2, 0.0, 0.0])
        lift_22 = np.array([0.0, +0.2, 0.0])
        lift_23 = np.array([0.0, 0.0, +0.2])

        pos_2a = np.array([+0.1, +0.1, 0.0])
        pos_2b = np.array([+0.1, 0.0, +0.1])
        pos_2c = np.array([0.0, +0.1, +0.1])

        noop = np.array([0.0, 0.0, 0.0])

        actions = [
            a01, a02, a03, a04, a05, a06,
            b01, b02, b03, b04, b05, b06,
            lift_01, lift_02, lift_03,
            lift_21, lift_22, lift_23,
            pos_2a, pos_2b, pos_2c,
            noop,
        ]

        actions_arr = np.stack(actions, axis=0).astype(np.float32)

        action_limit = float(self.base.max_action)
        # Keep DQN under the same incremental actuation bounds as the continuous agents.
        self.actions = np.clip(actions_arr, -action_limit, action_limit)

        self.n_actions = int(self.actions.shape[0])

    def reset(self):
        return self.base.reset()

    def step(self, action_idx: int):
        delta_u = self.actions[int(action_idx)]
        return self.base.step(delta_u)

    @property
    def max_step(self) -> int:
        return self.base.max_step

    @property
    def u(self):
        return self.base.u

    def distance(self) -> float:
        return self.base.distance()
