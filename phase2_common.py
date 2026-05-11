from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Phase2Spec:
    algorithm: str
    default_csv_log_path: str
    load_agent_for_eval: Callable[[str, Optional[Dict[str, Any]]], Tuple[Any, Any]]
    policy_action: Callable[[Any, np.ndarray, Dict[str, Any]], np.ndarray]
    script_file: Optional[str] = None


# ---------------- CSV schema ----------------

CSV_HEADER: List[str] = [
    "model_name",
    "eval_mode",
    "noise_sigma",
    "seed",
    "device",
    "state_type",
    "include_u_abs",
    "test_family",
    "aggregation",
    "region",
    "zone",
    "slice",
    "traj_kind",
    "traj_waypoints_requested",
    "traj_waypoints_realized",
    "traj_dense_points",
    "n_tests",
    "mean_error_mm",
    "std_error_mm",
    "std_max_error",
    "max_error",
    "std_mean_steps",
    "mean_steps",
    "std_success_rate_pct",
    "success_rate_pct",
    "global_rmse_mm",
    "std_global_rmse_mm",
    "global_max_tracking_error_mm",
    "std_global_max_tracking_error_mm",
    "std_steps",
    "mean_waypoint_error_mm",
    "std_waypoint_error_mm",
    "executed_path",
    "ideal_path",
    "dense_reference_path",
    "per_wp_error_mm",
]


def csv_log_append(path: str, row_dict: Dict[str, Any]) -> None:
    """Append one Phase II metric row and reject stale log schemas."""
    first = not os.path.exists(path)
    if not first:
        with open(path, "r", newline="") as rf:
            existing = (rf.readline() or "").strip("\r\n")
        expected = ",".join(CSV_HEADER)
        if existing and existing != expected:
            raise RuntimeError(
                "CSV header mismatch for log file.\n"
                f"  file: {path}\n"
                "  action: either delete/rename this file or change --csv-log-path\n"
                "  reason: the tester schema changed."
            )

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        if first:
            w.writeheader()
        row = {k: "" for k in CSV_HEADER}
        row.update(row_dict)
        w.writerow(row)


def infer_env_cfg_from_model_name(model_name: str) -> Tuple[int, bool]:
    """Recover the state family and actuator-augmentation flag from benchmark filenames."""
    s = str(model_name)
    m = re.search(r"(?:^|_)s([123])u([01])(?:_|$)", s)
    if m is None:
        m = re.search(r"s([123])u([01])", s)
    if m is None:
        raise ValueError(
            f"Could not infer env configuration from model_name='{model_name}'. "
            "Expected substring like 's1u0', 's2u1', or 's3u0'."
        )
    state_type = int(m.group(1))
    u_flag = int(m.group(2))
    include_u_abs = bool(u_flag == 1)
    return state_type, include_u_abs


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random as pyr

    pyr.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_seed_everywhere(seed: Optional[int], env: Optional[Any] = None) -> None:
    set_seed(seed)
    if env is not None:
        if hasattr(env, "seed") and callable(getattr(env, "seed")):
            try:
                env.seed(seed)
            except TypeError:
                try:
                    env.seed()
                except Exception:
                    pass
            except Exception:
                pass
        if hasattr(env, "reset") and callable(getattr(env, "reset")):
            try:
                env.reset(seed=seed)
            except TypeError:
                pass


def _is_stochastic(cfg: Dict[str, Any]) -> bool:
    return str(cfg.get("eval_mode", "deterministic")).strip().lower().startswith("stoch")


def _noise_sigma(cfg: Dict[str, Any]) -> float:
    return float(cfg.get("noise_sigma", 0.005))


def policy_action(agent: Any, state_np: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    return cfg["_policy_action"](agent, state_np, cfg)


def load_zone_slice_points(zone: int, slice_idx: int) -> np.ndarray:
    """Load held-out Task A targets; Phase II region files store millimeter coordinates."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_root, "dataset")
    fname = f"phase2_region{slice_idx}.txt"
    fpath = os.path.join(dataset_dir, fname)
    if not os.path.exists(fpath):
        region = f"phase2_region{slice_idx}"
        raise FileNotFoundError(
            f"Missing region dataset for {region}: expected file '{fpath}'. "
            f"Create '{fname}' under '{dataset_dir}' (relative to this tester) and re-run."
        )
    arr = np.loadtxt(fpath)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 3:
        raise ValueError(f"Expected dataset with 3 columns (x,y,z in mm), got shape {arr.shape}")
    return arr / 1000.0


class OscillationCutoff:
    """Stop rollouts that are only dithering inside a narrow distance band."""

    def __init__(self, band: float, patience: int):
        self.band = float(band)
        self.patience = int(patience)
        self.buf = deque(maxlen=patience)

    def reset(self):
        self.buf.clear()

    def update(self, dist_m: float) -> bool:
        self.buf.append(float(dist_m))
        if len(self.buf) < self.buf.maxlen:
            return False
        return (max(self.buf) - min(self.buf)) <= self.band


def _single_roll(agent, env, cfg: Dict[str, Any], state, initial_pos=None) -> Dict[str, Any]:
    cutoff = OscillationCutoff(cfg["cutoff_band"], cfg["cutoff_patience"])
    traj = []
    steps = 0
    with torch.no_grad():
        while True:
            action = policy_action(agent, state, cfg)
            state, reward, done, truncated = env.step(action)
            steps += 1

            cur_pos = env._simulate(env.u)
            traj.append(cur_pos.copy())

            done_flag = done or truncated
            if cutoff.update(env.distance()):
                # Prevent long failed approaches from dominating wall-clock time.
                done_flag = True
            if done_flag:
                break

    final_dist = env.distance()
    success = bool(final_dist <= env.goal_eps)
    init = initial_pos if initial_pos is not None else (traj[0].copy() if len(traj) > 0 else None)
    traj_arr = np.vstack(traj) if len(traj) > 0 else np.zeros((0, 3))
    out = {
        "final_distance_m": float(final_dist),
        "final_distance_mm": float(final_dist * 1000.0),
        "success": success,
        "steps": int(steps),
        "target": env.target_position.copy(),
        "initial": init,
        "trajectory": traj_arr,
    }
    return out


def _run_zero_start_to_target(agent, env, cfg: Dict[str, Any], target_pos_m, initial_pos=None) -> Dict[str, Any]:
    """Run one Task A fixed-target rollout from the zero-tendon state."""
    env.u = np.zeros(3, dtype=float)
    if initial_pos is None:
        start_pos = env._simulate(env.u)
    else:
        start_pos = np.asarray(initial_pos, dtype=float)

    env.target_position = np.asarray(target_pos_m, dtype=float).copy()
    env.current_step = 0

    dist0 = float(np.linalg.norm(start_pos - env.target_position))
    env.last_distance = dist0
    error0 = env.target_position - start_pos
    env.state = env._build_state(
        pos_m=start_pos,
        target_m=env.target_position,
        error_m=error0,
        distance_m=dist0,
        u_abs=env.u,
    )
    state = env.state.copy()
    return _single_roll(agent, env, cfg, state, initial_pos=start_pos)


# ---------------- Task A ----------------


def _batch_summary_from_arrays(final_d_mm, steps_list, successes, spatial_std=True) -> Dict[str, Any]:
    """Summarize Task A errors as MAE, Emax, mean successful steps, and SR."""
    final_d_mm = np.asarray(final_d_mm, dtype=float)
    steps_arr = np.asarray(steps_list, dtype=float)
    succ = np.asarray(successes, dtype=bool)

    std_error_mm = 0.0
    if spatial_std:
        std_error_mm = float(final_d_mm.std(ddof=1)) if final_d_mm.size > 1 else 0.0

    succ_steps = steps_arr[succ]
    mean_steps_success = float(succ_steps.mean()) if succ_steps.size > 0 else 0.0

    rep = {
        "mean_error_mm": float(final_d_mm.mean()) if final_d_mm.size > 0 else 0.0,
        "max_error": float(final_d_mm.max()) if final_d_mm.size > 0 else 0.0,
        "std_error_mm": float(std_error_mm),
        "mean_steps": float(mean_steps_success),
        "success_rate_pct": float(100.0 * succ.mean()) if succ.size > 0 else 0.0,
        "n_tests": int(final_d_mm.size),
    }
    return rep


def run_task_a_point_reaching_region(agent, env, cfg: Dict[str, Any], zone: int, slice_idx: int):
    """Evaluate one held-out workspace region from the Task A protocol."""
    pts_m = load_zone_slice_points(zone, slice_idx)
    final_d_mm = []
    steps_list = []
    successes = []
    for target_pos_m in pts_m:
        out = _run_zero_start_to_target(agent, env, cfg, target_pos_m)
        final_d_mm.append(out["final_distance_mm"])
        steps_list.append(out["steps"])
        successes.append(out["success"])
    return final_d_mm, steps_list, successes


run_batch_zero_start_zone = run_task_a_point_reaching_region


def _print_batch_line(prefix: str, rep: Dict[str, Any], std_label: str) -> None:
    if "std_max_error" in rep or "std_mean_steps" in rep or "std_success_rate_pct" in rep:
        print(
            f"{prefix} n={rep.get('n_tests', 0)} | "
            f"MAE={rep['mean_error_mm']:.2f} ± {rep['std_error_mm']:.2f} mm {std_label} | "
            f"max_error={rep['max_error']:.2f} ± {rep.get('std_max_error', 0.0):.2f} mm {std_label} | "
            f"mean_steps={rep['mean_steps']:.1f} ± {rep.get('std_mean_steps', 0.0):.1f} {std_label} | "
            f"success={rep['success_rate_pct']:.1f} ± {rep.get('std_success_rate_pct', 0.0):.1f}% {std_label}"
        )
    else:
        print(
            f"{prefix} n={rep.get('n_tests', 0)} | "
            f"MAE={rep['mean_error_mm']:.2f} ± {rep['std_error_mm']:.2f} mm {std_label} | "
            f"max_error={rep['max_error']:.2f} mm | "
            f"mean_steps={rep['mean_steps']:.1f} | "
            f"success={rep['success_rate_pct']:.1f}%"
        )


def run_batch_all_regions(agent, env, cfg: Dict[str, Any]) -> None:
    regions: List[Tuple[int, int]] = [
        (1, 1),
        (2, 1),
        (3, 1),
    ]
    stochastic = _is_stochastic(cfg)
    std_label = "(robustness)" if stochastic else "(spatial)"

    repeats = 5 if stochastic else 1
    seeds = list(cfg.get("seeds", []))
    if stochastic:
        if len(seeds) < repeats:
            raise ValueError(f"stochastic batch requires at least {repeats} seeds; got {len(seeds)}")
        seeds = seeds[:repeats]
    else:
        seeds = [int(cfg.get("seed", 0))]

    per_region_runs: Dict[str, List[Dict[str, Any]]] = {f"phase2_region{sl}": [] for sl, _zn in regions}
    global_runs: List[Dict[str, Any]] = []

    print("\n=== BATCH: zero-start over Phase 2 regions ===")
    if stochastic:
        print(f"[batch] stochastic repeats={repeats} | seeds={seeds}")

    for seed in seeds:
        set_seed_everywhere(int(seed), env=env)

        global_final: List[float] = []
        global_steps: List[int] = []
        global_succ: List[bool] = []

        for slice_idx, zone in regions:
            region_tag = f"phase2_region{slice_idx}"
            final_d_mm, steps_list, successes = run_task_a_point_reaching_region(
                agent, env, cfg, zone=zone, slice_idx=slice_idx
            )
            rep_region = _batch_summary_from_arrays(final_d_mm, steps_list, successes, spatial_std=(not stochastic))
            per_region_runs[region_tag].append(rep_region)
            global_final.extend(final_d_mm)
            global_steps.extend(steps_list)
            global_succ.extend(successes)

        rep_global = _batch_summary_from_arrays(global_final, global_steps, global_succ, spatial_std=(not stochastic))
        global_runs.append(rep_global)

    for slice_idx, zone in regions:
        region_tag = f"phase2_region{slice_idx}"
        runs = per_region_runs[region_tag]
        if not stochastic:
            rep = runs[0]
        else:
            mae = np.asarray([r["mean_error_mm"] for r in runs], dtype=float)
            mx = np.asarray([r["max_error"] for r in runs], dtype=float)
            ms = np.asarray([r["mean_steps"] for r in runs], dtype=float)
            sr = np.asarray([r["success_rate_pct"] for r in runs], dtype=float)
            rep = {
                "mean_error_mm": float(mae.mean()) if mae.size > 0 else 0.0,
                "std_error_mm": float(mae.std(ddof=1)) if mae.size > 1 else 0.0,
                "max_error": float(mx.mean()) if mx.size > 0 else 0.0,
                "std_max_error": float(mx.std(ddof=1)) if mx.size > 1 else 0.0,
                "mean_steps": float(ms.mean()) if ms.size > 0 else 0.0,
                "std_mean_steps": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
                "success_rate_pct": float(sr.mean()) if sr.size > 0 else 0.0,
                "std_success_rate_pct": float(sr.std(ddof=1)) if sr.size > 1 else 0.0,
                "n_tests": int(runs[0].get("n_tests", 0)) if runs else 0,
            }
        _print_batch_line(prefix=f"[batch:{region_tag}]", rep=rep, std_label=std_label)

        csv_log_append(cfg["csv_log_path"], {
            "model_name": cfg["model_name"],
            "eval_mode": cfg.get("eval_mode", "deterministic"),
            "noise_sigma": cfg.get("noise_sigma", 0.005) if stochastic else 0.0,
            "seed": "|".join(str(x) for x in seeds),
            "device": str(agent.device),
            "state_type": int(cfg.get("env_state_type", 1)),
            "include_u_abs": bool(cfg.get("env_include_u_abs", False)),
            "test_family": "batch",
            "aggregation": "per-region",
            "region": region_tag,
            "zone": int(zone),
            "slice": int(slice_idx),
            **rep,
        })

    if not stochastic:
        rep_g = global_runs[0]
    else:
        mae = np.asarray([r["mean_error_mm"] for r in global_runs], dtype=float)
        mx = np.asarray([r["max_error"] for r in global_runs], dtype=float)
        ms = np.asarray([r["mean_steps"] for r in global_runs], dtype=float)
        sr = np.asarray([r["success_rate_pct"] for r in global_runs], dtype=float)
        rep_g = {
            "mean_error_mm": float(mae.mean()) if mae.size > 0 else 0.0,
            "std_error_mm": float(mae.std(ddof=1)) if mae.size > 1 else 0.0,
            "max_error": float(mx.mean()) if mx.size > 0 else 0.0,
            "std_max_error": float(mx.std(ddof=1)) if mx.size > 1 else 0.0,
            "mean_steps": float(ms.mean()) if ms.size > 0 else 0.0,
            "std_mean_steps": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
            "success_rate_pct": float(sr.mean()) if sr.size > 0 else 0.0,
            "std_success_rate_pct": float(sr.std(ddof=1)) if sr.size > 1 else 0.0,
            "n_tests": int(global_runs[0].get("n_tests", 0)) if global_runs else 0,
        }

    _print_batch_line(prefix="[batch:GLOBAL]", rep=rep_g, std_label=std_label)
    csv_log_append(cfg["csv_log_path"], {
        "model_name": cfg["model_name"],
        "eval_mode": cfg.get("eval_mode", "deterministic"),
        "noise_sigma": cfg.get("noise_sigma", 0.005) if stochastic else 0.0,
        "seed": "|".join(str(x) for x in seeds),
        "device": str(agent.device),
        "state_type": int(cfg.get("env_state_type", 1)),
        "include_u_abs": bool(cfg.get("env_include_u_abs", False)),
        "test_family": "batch",
        "aggregation": "global",
        "region": "GLOBAL",
        **rep_g,
    })


# ---------------- Task B ----------------


def circle_waypoints(center_mm, diameter_mm, n_pts) -> np.ndarray:
    """Build the 50-waypoint circular Task B path, then append the first point to close it."""
    center_mm = np.asarray(center_mm, dtype=float)
    r = float(diameter_mm) / 2.0
    ang = np.linspace(0, 2 * np.pi, int(n_pts), endpoint=False)
    pts = []
    for a in ang:
        x = center_mm[0] + r * math.cos(a)
        y = center_mm[1]
        z = center_mm[2] + r * math.sin(a)
        pts.append([x, y, z])
    pts.append(pts[0])  # Paper Task B closes the circle by revisiting waypoint 0.
    return (np.asarray(pts) / 1000.0)


def spiral_waypoints_from_dataset(env, tip_mm, y_end_mm, n_rev, n_pts, slice_window_mm=1.0) -> np.ndarray:
    """Generate the Task B spiral along the empirical workspace envelope."""
    assert hasattr(env, "dataset") and env.dataset is not None, "env.dataset is required for spiral mode"
    ds = np.asarray(env.dataset)
    pos_mm = ds[:, -3:]
    x_mm = pos_mm[:, 0]
    y_mm = pos_mm[:, 1]
    z_mm = pos_mm[:, 2]

    cx = float(x_mm.mean())
    cz = float(z_mm.mean())

    tip_mm = np.asarray(tip_mm, dtype=float)
    y_start = float(tip_mm[1])
    y_end = float(y_end_mm)

    tip_dx = tip_mm[0] - cx
    tip_dz = tip_mm[2] - cz
    theta_tip = math.atan2(tip_dz, tip_dx)

    t_vals = np.linspace(0.0, 1.0, int(n_pts))
    slice_window = float(slice_window_mm)

    pts_mm = []
    for t in t_vals:
        y = y_start + (y_end - y_start) * t
        theta = theta_tip + 2.0 * math.pi * float(n_rev) * t

        # Estimate the local x-z boundary from recorded samples near the current y level.
        mask = (y_mm >= y - slice_window) & (y_mm <= y + slice_window)
        if np.any(mask):
            sx = x_mm[mask]
            sz = z_mm[mask]
        else:
            sx = x_mm
            sz = z_mm

        u = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        v = np.stack([sx - cx, sz - cz], axis=1)
        proj = v @ u
        r = float(proj.max()) if proj.size > 0 else 0.0
        r = max(r, 0.0)

        x = cx + r * math.cos(theta)
        z = cz + r * math.sin(theta)
        pts_mm.append([x, y, z])

    pts_mm = np.asarray(pts_mm, dtype=float)
    pts_mm[0, :] = tip_mm
    return pts_mm / 1000.0


def _build_dense_segment_references(waypoints_m: np.ndarray, dense_points_total: int = 1500):
    """Allocate dense reference samples by segment length for Task B RMSE/Emax."""
    wps = np.asarray(waypoints_m, dtype=float)
    n_seg = max(0, len(wps) - 1)
    if n_seg == 0:
        return np.zeros((0, 3), dtype=float), []

    dense_points_total = max(int(dense_points_total), n_seg * 2)
    seg_vecs = wps[1:] - wps[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    total_len = float(seg_lens.sum())
    if total_len > 0.0:
        raw = seg_lens / total_len * dense_points_total
    else:
        raw = np.full(n_seg, dense_points_total / n_seg, dtype=float)

    counts = np.floor(raw).astype(int)
    counts = np.maximum(counts, 2)

    cur = int(counts.sum())
    if cur < dense_points_total:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        k = 0
        while cur < dense_points_total:
            counts[order[k % len(order)]] += 1
            cur += 1
            k += 1
    elif cur > dense_points_total:
        order = np.argsort(-counts)
        k = 0
        guard = 0
        while cur > dense_points_total and guard < 10 * len(order):
            j = order[k % len(order)]
            if counts[j] > 2:
                counts[j] -= 1
                cur -= 1
            k += 1
            guard += 1

    dense_segments = []
    dense_path = []
    for i in range(n_seg):
        n_i = int(counts[i])
        t = np.linspace(0.0, 1.0, n_i, endpoint=True)
        seg = (1.0 - t)[:, None] * wps[i] + t[:, None] * wps[i + 1]
        dense_segments.append(seg)
        dense_path.extend(seg.tolist())

    return np.asarray(dense_path, dtype=float), dense_segments


def _attempt_reach_first_waypoint(agent, env, first_wp_m, cfg: Dict[str, Any], step_cap=None, tol=None):
    """Reach the first waypoint when the path has no prescribed starting actuation."""
    tol = env.goal_eps if tol is None else float(tol)
    step_cap = env.max_step if step_cap is None else int(step_cap)
    cutoff = OscillationCutoff(cfg["cutoff_band"], cfg["cutoff_patience"])

    if not hasattr(env, "u") or env.u is None:
        env.u = np.zeros(3, dtype=float)
    if not hasattr(env, "current_step"):
        env.current_step = 0

    env.target_position = np.asarray(first_wp_m, dtype=float).copy()
    cur_pos = env._simulate(env.u)
    dist0 = float(np.linalg.norm(cur_pos - env.target_position))
    env.last_distance = dist0

    env.state = env._build_state(
        pos_m=cur_pos,
        target_m=env.target_position,
        error_m=(env.target_position - cur_pos),
        distance_m=dist0,
        u_abs=env.u,
    )
    state = env.state.copy()

    steps = 0
    with torch.no_grad():
        while True:
            action = policy_action(agent, state, cfg)
            state, reward, done, truncated = env.step(action)
            steps += 1
            if cutoff.update(env.distance()):
                return (env.distance() <= tol), steps
            if env.distance() <= tol:
                return True, steps
            if steps >= step_cap:
                return False, steps
            if done or truncated:
                return (env.distance() <= tol), steps


def _eval_waypoint_sequence(agent, env, cfg: Dict[str, Any], waypoints_m: np.ndarray, start_actions_abs=None) -> Dict[str, Any]:
    """Evaluate Task B with state carried forward between consecutive waypoints."""
    tol = env.goal_eps if cfg["traj_tol"] is None else float(cfg["traj_tol"])
    cap_cfg = env.max_step if cfg["traj_seg_cap"] is None else int(cfg["traj_seg_cap"])
    cap = int(min(500, cap_cfg))
    goal_tolerance_m = 1e-3

    dense_path, dense_segments = _build_dense_segment_references(waypoints_m, cfg.get("traj_dense_points", 1500))

    executed: List[np.ndarray] = []
    per_wp_error_mm: List[float] = []
    per_wp_steps: List[int] = []

    tracking_sse_m2 = 0.0
    tracking_max_m = 0.0
    tracking_k = 0
    reached_waypoints = 0

    if start_actions_abs is not None:
        env.u = np.clip(np.asarray(start_actions_abs, float), 0.0, env.u_max)
        cur_pos = env._simulate(env.u)
        if np.linalg.norm(cur_pos - waypoints_m[0]) <= goal_tolerance_m:
            reached_waypoints += 1
    else:
        _ok, _steps0 = _attempt_reach_first_waypoint(agent, env, waypoints_m[0], cfg, step_cap=cap, tol=tol)
        cur_pos = env._simulate(env.u)
        if np.linalg.norm(cur_pos - waypoints_m[0]) <= goal_tolerance_m:
            reached_waypoints += 1

    n_segments = max(0, len(waypoints_m) - 1)
    for i in range(n_segments):
        wp_target = waypoints_m[i + 1]
        env.target_position = wp_target.copy()
        env.current_step = 0
        if not hasattr(env, "u") or env.u is None:
            env.u = np.zeros(3, dtype=float)

        cur_pos = env._simulate(env.u)
        dist0 = float(np.linalg.norm(cur_pos - env.target_position))
        env.last_distance = dist0
        env.state = env._build_state(
            pos_m=cur_pos,
            target_m=env.target_position,
            error_m=(env.target_position - cur_pos),
            distance_m=dist0,
            u_abs=env.u,
        )

        cutoff = OscillationCutoff(cfg["cutoff_band"], cfg["cutoff_patience"])
        ref_seg = dense_segments[i] if i < len(dense_segments) else np.vstack([waypoints_m[i], wp_target])

        steps = 0
        traj_seg = [cur_pos.copy()]
        with torch.no_grad():
            while True:
                action = policy_action(agent, env.state, cfg)
                state, reward, done, truncated = env.step(action)
                steps += 1

                cur_pos = env._simulate(env.u)
                traj_seg.append(cur_pos.copy())

                diff = ref_seg - cur_pos
                d2 = np.einsum("ij,ij->i", diff, diff)
                min_d2 = float(d2.min())
                tracking_sse_m2 += min_d2
                tracking_k += 1
                err_track_m = float(math.sqrt(min_d2))
                if err_track_m > tracking_max_m:
                    tracking_max_m = err_track_m

                if env.distance() <= tol:
                    break
                if cutoff.update(env.distance()):
                    break
                if steps >= cap:
                    break
                if done or truncated:
                    break

        err_m = env.distance()
        per_wp_error_mm.append(float(err_m * 1000.0))
        per_wp_steps.append(int(steps))
        if err_m <= goal_tolerance_m:
            reached_waypoints += 1
        executed.extend(traj_seg)

    executed_arr = np.vstack(executed) if len(executed) > 0 else np.zeros((0, 3))
    per_wp_err_arr = np.asarray(per_wp_error_mm, dtype=float)
    per_wp_steps_arr = np.asarray(per_wp_steps, dtype=float)

    std_wp_err = float(per_wp_err_arr.std(ddof=1)) if per_wp_err_arr.size > 1 else 0.0
    std_steps = float(per_wp_steps_arr.std(ddof=1)) if per_wp_steps_arr.size > 1 else 0.0

    path_rmse_mm = float(math.sqrt(tracking_sse_m2 / tracking_k) * 1000.0) if tracking_k > 0 else 0.0
    max_track_mm = float(tracking_max_m * 1000.0)
    n_waypoints = int(len(waypoints_m))
    success_rate_pct = float(100.0 * (reached_waypoints / n_waypoints)) if n_waypoints > 0 else 0.0

    rep: Dict[str, Any] = {
        "mean_error_mm": float(per_wp_err_arr.mean()) if per_wp_err_arr.size > 0 else 0.0,
        "std_error_mm": std_wp_err,
        "mean_steps": float(per_wp_steps_arr.mean()) if per_wp_steps_arr.size > 0 else 0.0,
        "std_steps": std_steps,
        "success_rate_pct": success_rate_pct,
        "global_rmse_mm": path_rmse_mm,
        "global_max_tracking_error_mm": max_track_mm,
        # Older analysis scripts read these aliases.
        "mean_waypoint_error_mm": float(per_wp_err_arr.mean()) if per_wp_err_arr.size > 0 else 0.0,
        "std_waypoint_error_mm": std_wp_err,
        "executed_path": executed_arr,
        "ideal_path": waypoints_m,
        "dense_reference_path": dense_path,
        "per_wp_error_mm": per_wp_error_mm,
    }

    return rep


def eval_trajectory_circle(agent, env, cfg: Dict[str, Any]) -> Dict[str, Any]:
    waypoints_m = circle_waypoints(cfg["traj_center_mm"], cfg["traj_diameter_mm"], cfg["traj_circle_points"])
    cfg_local = {**cfg, "traj_dense_points": int(cfg["traj_circle_dense_points"])}
    return _eval_waypoint_sequence(agent, env, cfg_local, waypoints_m, start_actions_abs=cfg["traj_start_actions_abs"])


def eval_trajectory_spiral(agent, env, cfg: Dict[str, Any]) -> Dict[str, Any]:
    waypoints_m = spiral_waypoints_from_dataset(
        env,
        cfg["traj_spiral_tip_mm"],
        cfg["traj_spiral_y_end_mm"],
        cfg["traj_spiral_revs"],
        cfg["traj_spiral_points"],
        slice_window_mm=cfg["traj_spiral_slice_window_mm"],
    )
    cfg_local = {**cfg, "traj_dense_points": int(cfg["traj_spiral_dense_points"])}
    if cfg_local.get("traj_zero_start", False):
        env.u = np.zeros(3, dtype=float)
    return _eval_waypoint_sequence(agent, env, cfg_local, waypoints_m, start_actions_abs=None)


def _print_traj_line(prefix: str, rep: Dict[str, Any], std_label: str) -> None:
    has_robust = ("std_global_rmse_mm" in rep) or ("std_global_max_tracking_error_mm" in rep) or ("std_success_rate_pct" in rep)
    if has_robust:
        print(
            f"{prefix} "
            f"wp_error(MAE)={rep['mean_error_mm']:.2f} ± {rep['std_error_mm']:.2f} mm {std_label} | "
            f"mean_steps={rep['mean_steps']:.1f} ± {rep['std_steps']:.1f} {std_label} | "
            f"RMSE={rep['global_rmse_mm']:.2f} ± {rep.get('std_global_rmse_mm', 0.0):.2f} mm {std_label} | "
            f"max_error={rep['global_max_tracking_error_mm']:.2f} ± {rep.get('std_global_max_tracking_error_mm', 0.0):.2f} mm {std_label} | "
            f"success={rep['success_rate_pct']:.1f} ± {rep.get('std_success_rate_pct', 0.0):.1f}% {std_label}"
        )
    else:
        print(
            f"{prefix} "
            f"wp_error(MAE)={rep['mean_error_mm']:.2f} ± {rep['std_error_mm']:.2f} mm {std_label} | "
            f"mean_steps={rep['mean_steps']:.1f} ± {rep['std_steps']:.1f} {std_label} | "
            f"RMSE={rep['global_rmse_mm']:.2f} mm | "
            f"max_error={rep['global_max_tracking_error_mm']:.2f} mm | "
            f"success={rep['success_rate_pct']:.1f}%"
        )


def run_trajectories(agent, env, cfg: Dict[str, Any]) -> None:
    stochastic = _is_stochastic(cfg)
    std_label = "(robustness)" if stochastic else "(spatial)"
    repeats = 10 if stochastic else 1
    seeds = list(cfg.get("seeds", []))
    if stochastic:
        if len(seeds) < repeats:
            raise ValueError(f"stochastic trajectory requires at least {repeats} seeds; got {len(seeds)}")
        seeds = seeds[:repeats]
    else:
        seeds = [int(cfg.get("seed", 0))]

    print("\n=== TRAJECTORY: circle + spiral ===")
    if stochastic:
        print(f"[trajectory] stochastic repeats={repeats} | seeds={seeds}")

    cfg_base = dict(cfg)

    circle_runs: List[Dict[str, Any]] = []
    spiral_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        set_seed_everywhere(int(seed), env=env)
        # Circle and spiral are run in the same seed pass used for stochastic repeats.
        rep_c = eval_trajectory_circle(agent, env, cfg_base)
        rep_s = eval_trajectory_spiral(agent, env, cfg_base)
        circle_runs.append(rep_c)
        spiral_runs.append(rep_s)

    def _aggregate(runs: List[Dict[str, Any]], keep_paths: bool, kind: str) -> Dict[str, Any]:
        if not stochastic:
            return runs[0]

        if len(runs) != repeats:
            raise RuntimeError(
                f"Trajectory aggregation expected {repeats} runs for '{kind}', got {len(runs)}. "
                f"Ensure seeds has at least {repeats} entries and per-run reports were collected."
            )

        def _req(r: Dict[str, Any], key: str):
            if key not in r:
                raise RuntimeError(
                    f"Missing key '{key}' in per-run trajectory report for '{kind}'. "
                    "This indicates an unexpected change in the per-run metric dict."
                )
            return r[key]

        mae = np.asarray([float(_req(r, "mean_error_mm")) for r in runs], dtype=float)
        steps = np.asarray([float(_req(r, "mean_steps")) for r in runs], dtype=float)
        succ = np.asarray([float(_req(r, "success_rate_pct")) for r in runs], dtype=float)
        rmse = np.asarray([float(_req(r, "global_rmse_mm")) for r in runs], dtype=float)
        mxtr = np.asarray([float(_req(r, "global_max_tracking_error_mm")) for r in runs], dtype=float)

        rep: Dict[str, Any] = {
            "mean_error_mm": float(mae.mean()) if mae.size > 0 else 0.0,
            "std_error_mm": float(mae.std(ddof=1)) if mae.size > 1 else 0.0,
            "mean_steps": float(steps.mean()) if steps.size > 0 else 0.0,
            "std_steps": float(steps.std(ddof=1)) if steps.size > 1 else 0.0,
            "success_rate_pct": float(succ.mean()) if succ.size > 0 else 0.0,
            "std_success_rate_pct": float(succ.std(ddof=1)) if succ.size > 1 else 0.0,
            "global_rmse_mm": float(rmse.mean()) if rmse.size > 0 else 0.0,
            "std_global_rmse_mm": float(rmse.std(ddof=1)) if rmse.size > 1 else 0.0,
            "global_max_tracking_error_mm": float(mxtr.mean()) if mxtr.size > 0 else 0.0,
            "std_global_max_tracking_error_mm": float(mxtr.std(ddof=1)) if mxtr.size > 1 else 0.0,
            # Older analysis scripts read these aliases.
            "mean_waypoint_error_mm": float(mae.mean()) if mae.size > 0 else 0.0,
            "std_waypoint_error_mm": float(mae.std(ddof=1)) if mae.size > 1 else 0.0,
        }

        if keep_paths and runs:
            for k in ("executed_path", "ideal_path", "dense_reference_path", "per_wp_error_mm"):
                if k in runs[0]:
                    rep[k] = runs[0][k]
        return rep

    rep_circle = _aggregate(circle_runs, keep_paths=(not stochastic), kind="circle")
    rep_spiral = _aggregate(spiral_runs, keep_paths=(not stochastic), kind="spiral")

    circle_req = int(cfg["traj_circle_points"])
    circle_real = int(circle_req + 1)  # closed loop
    spiral_req = int(cfg["traj_spiral_points"])

    _print_traj_line(
        prefix=f"[trajectory:circle] waypoints={circle_req} (closed={circle_real}) dense={int(cfg['traj_circle_dense_points'])} |",
        rep=rep_circle,
        std_label=std_label,
    )
    _print_traj_line(
        prefix=f"[trajectory:spiral] waypoints={spiral_req} dense={int(cfg['traj_spiral_dense_points'])} |",
        rep=rep_spiral,
        std_label=std_label,
    )

    for kind, rep, wp_req, wp_real, dense_pts in [
        ("circle", rep_circle, circle_req, circle_real, int(cfg["traj_circle_dense_points"])),
        ("spiral", rep_spiral, spiral_req, spiral_req, int(cfg["traj_spiral_dense_points"])),
    ]:
        csv_log_append(cfg["csv_log_path"], {
            "model_name": cfg["model_name"],
            "eval_mode": cfg.get("eval_mode", "deterministic"),
            "noise_sigma": cfg.get("noise_sigma", 0.005) if stochastic else 0.0,
            "seed": "|".join(str(x) for x in seeds),
            "device": str(agent.device),
            "state_type": int(cfg.get("env_state_type", 1)),
            "include_u_abs": bool(cfg.get("env_include_u_abs", False)),
            "test_family": "trajectory",
            "aggregation": "per-trajectory",
            "traj_kind": kind,
            "traj_waypoints_requested": int(wp_req),
            "traj_waypoints_realized": int(wp_real),
            "traj_dense_points": int(dense_pts),
            **rep,
        })


# ---------------- model selection (convergence + best_results) ----------------


def _candidate_base_dirs(spec: Phase2Spec) -> List[str]:
    dirs = []
    if spec.script_file:
        script_dir = os.path.dirname(os.path.abspath(spec.script_file))
        dirs.append(script_dir)
        dirs.append(os.path.abspath(os.path.join(script_dir, "..")))
    dirs.append(os.getcwd())

    unique_dirs = []
    seen = set()
    for path in dirs:
        norm = os.path.abspath(path)
        if norm not in seen:
            seen.add(norm)
            unique_dirs.append(norm)
    return unique_dirs


def _resolve_existing_input_path(path: str, spec: Phase2Spec) -> Optional[str]:
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    for base_dir in _candidate_base_dirs(spec):
        candidate = os.path.join(base_dir, path)
        if os.path.exists(candidate):
            return candidate
    return None


def _autodetect_convergence_path(spec: Phase2Spec) -> Optional[str]:
    return _resolve_existing_input_path(f"{spec.algorithm}_convergence.csv", spec)


def read_converged_base_configs(path: str) -> Tuple[set, Dict[str, str]]:
    converged: set = set()
    status: Dict[str, str] = {}
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            name = (row[0] or "").strip()
            if not name:
                continue
            flag = (row[1] if len(row) > 1 else "").strip().lower()
            base = name[:-4] if name.lower().endswith(".csv") else name
            status[base] = flag
            if flag in ("yes", "y", "true", "1"):
                converged.add(base)
    return converged, status


def _parse_int_field(row: Dict[str, str], key: str) -> Optional[int]:
    if key not in row:
        return None
    v = str(row.get(key, "")).strip()
    if v == "":
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def iter_best_results_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            yield {k: (str(v).strip() if v is not None else "") for k, v in raw.items()}


def build_model_name_from_best_row(row: Dict[str, str], algorithm: str) -> Optional[str]:
    st = _parse_int_field(row, "state_type")
    iu = _parse_int_field(row, "include_u_abs")
    rt = _parse_int_field(row, "reward_type")
    bf = _parse_int_field(row, "bonus_flag")
    ck = _parse_int_field(row, "best_checkpoint")
    if None in (st, iu, rt, bf, ck):
        return None
    base = f"{algorithm}_continuum_robot_s{st}u{iu}r{rt}b{bf}"
    return f"{base}_e{ck}"


def base_config_from_model_name(model_name: str) -> str:
    m = re.match(r"^(.*)_e\d+$", model_name.strip())
    return m.group(1) if m else model_name.strip()


# ---------------- defaults ----------------


def default_eval_config(spec: Phase2Spec) -> Dict[str, Any]:
    return {
        "env_state_type": 1,
        "env_include_u_abs": False,
        "mode": "all",
        "eval_mode": "stochastic",
        "noise_sigma": 0.01,
        "seeds": [151, 12, 103, 504, 555, 201, 222, 303, 704, 255],
        "model_name": "",
        "seed": 123,
        "csv_log_path": spec.default_csv_log_path,
        "cutoff_patience": 300,
        "cutoff_band": 1e-4,
        "traj_center_mm": [-43.19, -125.0, 276.85],
        "traj_diameter_mm": 90.0,
        "traj_circle_points": 50,
        "traj_circle_dense_points": 1500,
        "traj_start_actions_abs": [0.03, 45.24, 67.99],
        "traj_spiral_tip_mm": [-43.19, -144.24, 276.85],
        "traj_spiral_y_end_mm": -115.0,
        "traj_spiral_revs": 3,
        "traj_spiral_slice_window_mm": 1.0,
        "traj_zero_start": True,
        "traj_spiral_points": 150,
        "traj_spiral_dense_points": 4500,
        "traj_seg_cap": 500,
        "traj_tol": None,
        "_policy_action": spec.policy_action,
    }


# ---------------- main harness ----------------


def parse_args(spec: Phase2Spec, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Automated {spec.algorithm.upper()} Phase 2 evaluator.")
    p.add_argument("--best-results", default="best_results.csv", help="Path to best_results.csv")
    p.add_argument(
        "--convergence",
        default=None,
        help=f"Path to {spec.algorithm}_convergence.csv (if omitted, auto-detect near the adapter and CWD).",
    )
    p.add_argument("--csv-log-path", default=spec.default_csv_log_path, help="Output CSV log path (single file for all models).")
    p.add_argument("--mode", default="batch", choices=["batch", "trajectory", "all"], help="Which test family to run.")
    p.add_argument("--eval-mode", default="stochastic", choices=["deterministic", "stochastic"], help="Deterministic or stochastic evaluation.")
    p.add_argument("--noise-sigma", type=float, default=0.01, help="Sigma for N(0, sigma) noise in stochastic mode.")
    p.add_argument(
        "--seeds",
        default="151,12,103,504,555,201,222,303,704,255",
        help="Comma-separated seeds list (stochastic uses first 5 for batch, first 10 for trajectory).",
    )
    p.add_argument("--seed", type=int, default=123, help="Single seed used for deterministic mode.")
    return p.parse_args(argv)


def run_phase2_evaluation(spec: Phase2Spec, argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(spec, argv)

    conv_path = (
        _resolve_existing_input_path(args.convergence, spec)
        if args.convergence
        else _autodetect_convergence_path(spec)
    )
    if conv_path is None or not os.path.exists(conv_path):
        searched = ", ".join(_candidate_base_dirs(spec))
        print(
            f"[error] Convergence CSV not found. Provide --convergence or place "
            f"'{spec.algorithm}_convergence.csv' in one of: {searched}"
        )
        return 2
    best_results_path = _resolve_existing_input_path(args.best_results, spec)
    if best_results_path is None:
        searched = ", ".join(_candidate_base_dirs(spec))
        print(f"[error] best_results.csv not found. Searched '{args.best_results}' in: {searched}")
        return 2

    converged, status_map = read_converged_base_configs(conv_path)
    print(f"[select] convergence file: {os.path.abspath(conv_path)}")
    print(f"[select] converged base configs: {len(converged)} / {len(status_map)}")

    selected_models: List[str] = []
    skipped_rows = 0
    for row in iter_best_results_rows(best_results_path):
        model_name = build_model_name_from_best_row(row, spec.algorithm)
        if not model_name:
            skipped_rows += 1
            continue
        base = base_config_from_model_name(model_name)
        if base in converged:
            selected_models.append(model_name)

    seen = set()
    selected_models_unique: List[str] = []
    for model_name in selected_models:
        if model_name not in seen:
            seen.add(model_name)
            selected_models_unique.append(model_name)

    print(f"[select] best_results file: {os.path.abspath(best_results_path)}")
    print(f"[select] rows skipped (missing/invalid fields): {skipped_rows}")
    print(f"[select] models selected for eval: {len(selected_models_unique)}")
    if not selected_models_unique:
        print("[select] No converged models matched best_results.csv. Nothing to do.")
        return 0

    log_dir = os.path.dirname(os.path.abspath(args.csv_log_path))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    failures = 0
    for idx, model_name in enumerate(selected_models_unique, start=1):
        print("\n" + "=" * 90)
        print(f"[MODEL {idx}/{len(selected_models_unique)}] {model_name}")
        print("=" * 90)

        cfg = default_eval_config(spec)
        cfg["model_name"] = model_name
        cfg["mode"] = str(args.mode).strip().lower()
        cfg["eval_mode"] = str(args.eval_mode).strip().lower()
        cfg["noise_sigma"] = float(args.noise_sigma)
        cfg["csv_log_path"] = args.csv_log_path
        cfg["seed"] = int(args.seed)

        try:
            seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip() != ""]
        except Exception:
            seeds = default_eval_config(spec)["seeds"]
        cfg["seeds"] = seeds

        try:
            st, iu = infer_env_cfg_from_model_name(model_name)
        except Exception as e:
            print(f"[error] {e}")
            failures += 1
            continue
        cfg["env_state_type"] = int(st)
        cfg["env_include_u_abs"] = bool(iu)

        env_kwargs = {
            "state_type": cfg.get("env_state_type", 1),
            "include_u_abs": cfg.get("env_include_u_abs", False),
        }

        t0 = time.perf_counter()
        try:
            agent, env = spec.load_agent_for_eval(model_name, env_kwargs=env_kwargs)
        except Exception as e:
            print(f"[error] Failed to load model '{model_name}': {e}")
            failures += 1
            continue

        device = getattr(agent, "device", "unknown")
        print(
            f"Loaded '{model_name}' on device={device}, "
            f"state_dim={env.state_dim}, action_dim={env.action_dim}, "
            f"cuda={torch.cuda.is_available()} in {time.perf_counter()-t0:.2f}s"
        )
        print(
            f"[env] inferred from model_name: state_type={cfg['env_state_type']} | "
            f"include_u_abs={cfg['env_include_u_abs']} | eval_mode={cfg.get('eval_mode','deterministic')} | "
            f"sigma={cfg.get('noise_sigma', 0.005) if _is_stochastic(cfg) else 0.0}"
        )

        try:
            mode = str(cfg.get("mode", "all")).strip().lower()
            if mode not in ("batch", "trajectory", "all"):
                raise ValueError(f"Unsupported mode '{cfg.get('mode')}'. Use 'batch', 'trajectory', or 'all'.")
            if mode in ("batch", "all"):
                run_batch_all_regions(agent, env, cfg)
            if mode in ("trajectory", "all"):
                run_trajectories(agent, env, cfg)
        except Exception as e:
            print(f"[error] Evaluation failed for '{model_name}': {e}")
            failures += 1
            continue

    print("\n" + "=" * 90)
    print(f"Done. models={len(selected_models_unique)} | failures={failures} | csv_log={os.path.abspath(args.csv_log_path)}")
    print("=" * 90)
    return 1 if failures else 0
