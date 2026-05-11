import os
import random
import numpy as np
import torch
import joblib
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = current_dir
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
from SL_learning.SurrogateLearning import NeuralNetwork


class ContinuumRobotEnv:
    """Static DDKS environment used for the paper's formulation benchmark.

    The transition model is the learned forward kinematic surrogate from the
    motion-capture dataset. It is intentionally quasi-static: tendon history,
    friction, hysteresis, and sensing latency are outside the benchmark scope.
    """

    max_step = 1000

    def __init__(
        self,
        max_action=1.0,
        reward_type: int = 3,
        use_success_bonus: bool = False,
        success_bonus: float = 0.0,
        use_step_penalty: bool = False,
        step_penalty: float = 0.0,
        state_type: int = 1,
        include_u_abs: bool = False
    ):
        model_dir = os.path.join(proj_root, "SurrogateModel")
        model_path = os.path.join(model_dir, "trained_model_sl_1.pth")
        scaler_X_path = os.path.join(model_dir, "scaler_X_sl_1.pkl")
        scaler_y_path = os.path.join(model_dir, "scaler_y_sl_1.pkl")
        dataset_path = os.path.join(proj_root, "dataset", "RL_training.txt")

        if not os.path.isdir(model_dir):
            print(f"Error: Directory '{model_dir}' does not exist.")
            exit(1)
        if not (os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path)):
            print(f"Error: Model or scaler files are missing in the '{model_dir}' directory.")
            exit(1)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file '{dataset_path}' does not exist.")
            exit(1)

        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.dataset = pd.read_csv(dataset_path, header=None, sep=",").values

        # Paper convention: encoded l_i in [0, 100], with 1 unit = 0.1 mm tendon shortening.
        self.u_max = 100.0
        self.max_action = float(max_action)
        self.action_dim = 3
        
        self.state_type = int(state_type)
        self.include_u_abs = bool(include_u_abs)
        
        # Keep the scaling tied to the recorded workspace used by the surrogate.
        positions_m = self.dataset[:, -3:] / 1000.0
        self.pos_scale = float(max(np.max(np.abs(positions_m)), 1e-6))
        self.dist_scale = float(max(np.max(np.linalg.norm(positions_m, axis=1)), 1e-6))
        
        if self.state_type == 1:
            base_dim = 3
        elif self.state_type == 2:
            base_dim = 6
        else:
            raise ValueError("state_type must be 1 or 2")
        
        self.state_dim = base_dim + (3 if self.include_u_abs else 0)
        
        self.goal_eps = 1e-3
        self.prev_distance = None
        self.reward_type = int(reward_type)
        self.use_success_bonus = bool(use_success_bonus)
        self.success_bonus = float(success_bonus)
        self.use_step_penalty = bool(use_step_penalty)
        self.step_penalty = float(step_penalty)

    def _random_target_row(self, l_min=0, l_max=100):
        """Sample a recorded pose inside the encoded tendon limits."""
        ds = np.asarray(self.dataset)
        actions = ds[:, :3].astype(int)
        mask = (actions >= l_min).all(axis=1) & (actions <= l_max).all(axis=1)
        eligible = ds[mask]
        row = eligible[random.randrange(eligible.shape[0])]
        l1, l2, l3 = row[:3].astype(int)
        x, y, z = row[-3:]
        return np.array([x, y, z], dtype=float), np.array([l1, l2, l3], dtype=float)

    def reset(self):
        """Start a training episode from recorded actuation and target samples."""
        self.current_step = 0
        target_pos_mm, _ = self._random_target_row()
        init_pos_mm, self.u = self._random_target_row()

        self.target_position = target_pos_mm / 1000.0
        init_pos = init_pos_mm / 1000.0

        dist = float(np.linalg.norm(init_pos - self.target_position))
        self.initial_distance = max(dist, 1e-8)
        self.last_distance = dist
        self.prev_distance = dist

        error = self.target_position - init_pos
        self.state = self._build_state(pos_m=init_pos, target_m=self.target_position, error_m=error, distance_m=dist, u_abs=self.u)
        return self.state.copy()

    def step(self, delta_u):
        self.current_step += 1
        
        # Actions are incremental tendon commands; absolute actuation stays inside the dataset range.
        du = np.clip(np.asarray(delta_u, dtype=float), -self.max_action, self.max_action)
        self.u = np.clip(self.u + du, 0.0, self.u_max)

        next_pos = self._simulate(self.u)

        distance_to_target = float(np.linalg.norm(next_pos - self.target_position))
        self.prev_distance = getattr(self, "last_distance", None)
        self.last_distance = distance_to_target

        error = self.target_position - next_pos
        self.state = self._build_state(pos_m=next_pos, target_m=self.target_position, error_m=error, distance_m=distance_to_target, u_abs=self.u)

        reached = distance_to_target <= self.goal_eps
        done = reached or (self.current_step >= self.max_step)
        truncated = (self.current_step >= self.max_step) and not reached

        reward = self._compute_reward(distance_to_target, reached)
        
        return self.state.copy(), float(reward), bool(done), bool(truncated)


    @torch.no_grad()
    def _simulate(self, u_abs):
        """Predict tip position in meters from encoded tendon values."""
        x_in = self.scaler_X.transform([u_abs])
        x_tensor = torch.tensor(x_in, dtype=torch.float32)
        pred = self.model(x_tensor).numpy()
        pos_mm = self.scaler_y.inverse_transform(pred)[0]
        return np.array(pos_mm, dtype=float) / 1000.0

    def _compute_reward(self, distance_to_target: float, reached: bool) -> float:
        if self.reward_type == 1:
            # Reward Type 1 in the paper: dense distance penalty in meters.
            reward = -float(distance_to_target)

        elif self.reward_type == 2:
            # Reward Type 2: ternary progress signal with the paper's 1e-9 m equality band.
            prev = getattr(self, "prev_distance", None)
            if prev is None or np.isnan(prev):
                prev = distance_to_target
            tol = 1e-9
            if distance_to_target < (prev - tol):
                reward = 1.0
            elif abs(distance_to_target - prev) <= tol:
                reward = -0.5
            else:
                reward = -1.0

        else:
            # Reward Type 3: shaped radial reward with radius max(2*d0, 6 mm).
            alpha = 2.0
            alpha_min = 0.006
            eps = 1e-8
            Dg = getattr(self, "initial_distance", None)
            if Dg is None or np.isnan(Dg):
                Dg = max(getattr(self, "last_distance", distance_to_target), eps)

            radius = max(alpha * Dg, alpha_min)

            if distance_to_target <= radius:
                x = max(distance_to_target / max(Dg, eps), 1e-8)
                z = 1.0 - np.sqrt(2.0 * x)
                yd = 2.0 * z if z >= 0.0 else z
                reward = float(yd)
            else:
                reward = -1.0

        if self.use_step_penalty and self.step_penalty != 0.0:
            reward -= self.step_penalty

        if reached and self.use_success_bonus and self.success_bonus != 0.0:
            # Composite reward variant from Table 1 adds the terminal bonus at the 1 mm goal.
            reward += self.success_bonus

        return reward

    def distance(self):
        return float(self.last_distance)
    
    def _build_state(
        self,
        pos_m: np.ndarray,
        target_m: np.ndarray,
        error_m: np.ndarray,
        distance_m: float,
        u_abs: np.ndarray,
    ) -> np.ndarray:
        """Build State Type 1 or 2, optionally augmented by absolute tendon state."""
        # These constants are the normalization ranges used by the benchmark environment.
        A_ERROR    = np.array([0.13, 0.05, 0.14])
        POS_CENTER = np.array([-0.05, -0.11, 0.27])
        A_POS      = np.array([0.06, 0.03, 0.07])
        A_DIST     = 0.2

        if self.state_type == 1:
            # Error-based state: translation-invariant target-tip error.
            base = np.concatenate([np.clip(error_m / A_ERROR, -1.0, 1.0)])

        elif self.state_type == 2:
            # Pose-target state: absolute current and target positions retain workspace context.
            pos_scaled = np.clip((pos_m - POS_CENTER) / A_POS, -1.0, 1.0)
            tgt_scaled = np.clip((target_m - POS_CENTER) / A_POS, -1.0, 1.0)
            base = np.concatenate([pos_scaled, tgt_scaled])

        else:
            raise ValueError("state_type must be 1 or 2")

        if self.include_u_abs:
            # Actuator augmentation tests whether absolute tendon configuration helps the policy.
            u_scaled = np.clip((u_abs - 50.0) / 50.0, -1.0, 1.0)
            state = np.concatenate([base, u_scaled])
        else:
            state = base

        return state
