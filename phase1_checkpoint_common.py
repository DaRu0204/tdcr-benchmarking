from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


BEST_RESULTS_HEADER = [
    "algorithm",
    "state_type",
    "include_u_abs",
    "reward_type",
    "bonus_flag",
    "best_checkpoint",
    "validation_mae_mm",
]

DEFAULT_DATASET_DIR = "dataset"
DEFAULT_STEP_CAP = 500
CONVERGENCE_FILENAMES = ("{algorithm}_convergence.csv",)


@dataclass(frozen=True)
class CheckpointSpec:
    algorithm: str
    model_dir: str
    checkpoint_pattern: re.Pattern[str]
    create_agent: Callable[[Any, torch.device], Any]
    load_checkpoint: Callable[[Any, str, int], None]
    select_action: Callable[[Any, np.ndarray], np.ndarray]


def project_root_from_script(script_file: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(script_file))
    if os.path.isdir(os.path.join(script_dir, DEFAULT_DATASET_DIR)):
        return script_dir
    return os.path.abspath(os.path.join(script_dir, ".."))


def resolve_project_path(project_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_root, path)


def resolve_script_path(script_file: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(script_file)), path)


def get_reward_params(reward_type: int) -> Tuple[float, float]:
    """Return the Table 1 step penalty and success bonus for augmented rewards."""
    if reward_type == 1:
        return 0.05, 5.0
    if reward_type == 2:
        return 0.6, 60.0
    if reward_type == 3:
        return 1.1, 100.0
    raise ValueError(f"Unsupported reward_type: {reward_type}")


def load_phase1_targets(dataset_dir: str) -> np.ndarray:
    """Load the Phase I validation targets; region files store millimeter coordinates."""
    chunks = []
    for idx in (1, 2, 3):
        path = os.path.join(dataset_dir, f"phase1_region{idx}.txt")
        points = np.loadtxt(path)
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != 3:
            raise ValueError(f"{path} must contain exactly 3 coordinate columns, got {points.shape}")
        chunks.append(points)
    return np.vstack(chunks) / 1000.0


def discover_checkpoints(spec: CheckpointSpec, model_dir: str) -> Dict[Tuple[int, int, int, int], List[int]]:
    groups: Dict[Tuple[int, int, int, int], set[int]] = defaultdict(set)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    for filename in os.listdir(model_dir):
        match = spec.checkpoint_pattern.match(filename)
        if not match:
            continue
        key = (
            int(match.group("S")),
            int(match.group("U")),
            int(match.group("R")),
            int(match.group("B")),
        )
        groups[key].add(int(match.group("E")))

    return {key: sorted(values) for key, values in sorted(groups.items())}


def autodetect_convergence_path(spec: CheckpointSpec, script_file: str, explicit_path: Optional[str]) -> Optional[str]:
    """Look beside the adapter first; generated convergence CSVs usually live there."""
    if explicit_path:
        return explicit_path

    script_dir = os.path.dirname(os.path.abspath(script_file))
    project_root = project_root_from_script(script_file)
    cwd = os.getcwd()
    candidates = []
    for template in CONVERGENCE_FILENAMES:
        filename = template.format(algorithm=spec.algorithm)
        candidates.append(os.path.join(script_dir, filename))
        candidates.append(os.path.join(project_root, filename))
        candidates.append(os.path.join(cwd, filename))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def parse_base_config(base_name: str, algorithm: str) -> Optional[Tuple[int, int, int, int]]:
    pattern = re.compile(
        rf"^{re.escape(algorithm)}_continuum_robot_s(?P<S>\d+)u(?P<U>[01])r(?P<R>\d+)b(?P<B>[01])$",
        re.IGNORECASE,
    )
    match = pattern.match(base_name.strip())
    if not match:
        return None
    return (
        int(match.group("S")),
        int(match.group("U")),
        int(match.group("R")),
        int(match.group("B")),
    )


def read_converged_configs(path: str, algorithm: str) -> set[Tuple[int, int, int, int]]:
    converged = set()
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            flag = row[1].strip().lower()
            if flag not in ("yes", "y", "true", "1"):
                continue
            name = row[0].strip()
            base_name = name[:-4] if name.lower().endswith(".csv") else name
            key = parse_base_config(base_name, algorithm)
            if key is not None:
                converged.add(key)
    return converged


def start_fixed_episode(env: Any, target_m: np.ndarray) -> np.ndarray:
    """Reset to the zero-tendon start used for checkpoint validation."""
    env.current_step = 0
    env.u = np.array([0.0, 0.0, 0.0], dtype=float)
    env.target_position = np.asarray(target_m, dtype=float).copy()

    pos_m = env._simulate(env.u)
    dist = float(np.linalg.norm(pos_m - env.target_position))

    env.initial_distance = max(dist, 1e-8)
    env.last_distance = dist
    env.prev_distance = dist

    error_m = env.target_position - pos_m
    env.state = env._build_state(
        pos_m=pos_m,
        target_m=env.target_position,
        error_m=error_m,
        distance_m=dist,
        u_abs=env.u,
    )
    return env.state.copy()


def evaluate_checkpoint_mae(
    agent: Any,
    env: Any,
    targets_m: np.ndarray,
    select_action: Callable[[Any, np.ndarray], np.ndarray],
    step_cap: int,
) -> float:
    """Select checkpoints by reward-independent MAE over Phase I validation targets."""
    errors_mm = []
    max_steps = min(int(getattr(env, "max_step", 10**9)), int(step_cap))

    for target_m in targets_m:
        state = start_fixed_episode(env, target_m)
        steps = 0
        while True:
            action = select_action(agent, state)
            state, _reward, done, truncated = env.step(action)
            steps += 1
            if done or truncated or steps >= max_steps:
                break
        errors_mm.append(float(env.last_distance) * 1000.0)

    return float(np.mean(errors_mm))


def make_env(env_class: Any, state_type: int, include_u_abs: bool, reward_type: int, bonus_flag: int) -> Any:
    step_penalty, success_bonus = get_reward_params(reward_type)
    use_composite_augmentation = bool(bonus_flag)
    # Recreate the exact state/reward formulation encoded in the checkpoint filename.
    return env_class(
        max_action=1.0,
        reward_type=reward_type,
        use_success_bonus=use_composite_augmentation,
        success_bonus=success_bonus if use_composite_augmentation else 0.0,
        use_step_penalty=use_composite_augmentation,
        step_penalty=step_penalty if use_composite_augmentation else 0.0,
        state_type=state_type,
        include_u_abs=include_u_abs,
    )


def write_best_results(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEST_RESULTS_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]], spec: CheckpointSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Select best {spec.algorithm.upper()} Phase 1 checkpoints for Phase 2 evaluation."
    )
    parser.add_argument("--model-dir", default=spec.model_dir)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--convergence", default=None)
    parser.add_argument("--best-csv", default="best_results.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--step-cap", type=int, default=DEFAULT_STEP_CAP)
    return parser.parse_args(argv)


def run_phase1_checkpoint_selection(
    spec: CheckpointSpec,
    env_class: Any,
    script_file: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    args = parse_args(argv, spec)
    project_root = project_root_from_script(script_file)
    model_dir = resolve_project_path(project_root, args.model_dir)
    dataset_dir = resolve_project_path(project_root, args.dataset_dir)
    best_csv_path = resolve_script_path(script_file, args.best_csv)
    convergence_path = autodetect_convergence_path(spec, script_file, args.convergence)

    if convergence_path is None or not os.path.exists(convergence_path):
        names = " or ".join(name.format(algorithm=spec.algorithm) for name in CONVERGENCE_FILENAMES)
        print(f"[error] Convergence CSV not found. Provide --convergence or create {names}.")
        return 2

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    targets_m = load_phase1_targets(dataset_dir)
    checkpoints = discover_checkpoints(spec, model_dir)
    converged = read_converged_configs(convergence_path, spec.algorithm)

    print(f"[setup] algorithm={spec.algorithm.upper()}")
    print(f"[setup] model_dir={model_dir}")
    print(f"[setup] convergence={os.path.abspath(convergence_path)}")
    print(f"[setup] validation_targets={len(targets_m)}")
    print(f"[setup] device={device}, step_cap={args.step_cap}")
    print(f"[setup] discovered_configs={len(checkpoints)}, converged_configs={len(converged)}")

    results: List[Dict[str, Any]] = []
    evaluated_configs = 0
    skipped_configs = 0

    for key, episodes in sorted(checkpoints.items()):
        state_type, include_u_abs_flag, reward_type, bonus_flag = key
        base_name = f"{spec.algorithm}_continuum_robot_s{state_type}u{include_u_abs_flag}r{reward_type}b{bonus_flag}"

        if key not in converged:
            skipped_configs += 1
            print(f"[skip] {base_name}: not converged")
            continue
        if state_type not in (1, 2):
            skipped_configs += 1
            print(f"[skip] {base_name}: unsupported state_type={state_type}")
            continue

        print(f"[config] {base_name}: checkpoints={len(episodes)}")
        env = make_env(env_class, state_type, bool(include_u_abs_flag), reward_type, bonus_flag)
        agent = spec.create_agent(env, device)

        best_checkpoint: Optional[int] = None
        best_mae = float("inf")

        for episode in episodes:
            model_name = f"{base_name}_e{episode}"
            try:
                spec.load_checkpoint(agent, model_name, episode)
                validation_mae = evaluate_checkpoint_mae(
                    agent=agent,
                    env=env,
                    targets_m=targets_m,
                    select_action=spec.select_action,
                    step_cap=args.step_cap,
                )
            except Exception as exc:
                print(f"[checkpoint] {model_name}: failed ({exc})")
                continue

            print(f"[checkpoint] {model_name}: validation_mae_mm={validation_mae:.6f}")
            if validation_mae < best_mae:
                best_mae = validation_mae
                best_checkpoint = episode

        if best_checkpoint is None:
            print(f"[result] {base_name}: no valid checkpoints")
            continue

        evaluated_configs += 1
        results.append(
            {
                "algorithm": spec.algorithm,
                "state_type": state_type,
                "include_u_abs": include_u_abs_flag,
                "reward_type": reward_type,
                "bonus_flag": bonus_flag,
                "best_checkpoint": best_checkpoint,
                "validation_mae_mm": f"{best_mae:.6f}",
            }
        )
        print(f"[best] {base_name}: e{best_checkpoint}, validation_mae_mm={best_mae:.6f}")

    write_best_results(best_csv_path, results)
    print(f"[done] evaluated_configs={evaluated_configs}, skipped_configs={skipped_configs}")
    print(f"[done] wrote {len(results)} rows to {best_csv_path}")
    return 0
