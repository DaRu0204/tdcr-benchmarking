from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Pattern, Sequence, Tuple


THRESHOLD = 0.001
REQUIRED_STREAK = 100


@dataclass(frozen=True)
class ConvergenceSpec:
    algorithm: str
    output_file: str
    log_filename_pattern: Pattern[str]


def discover_csv_files(root_dir: str, output_filename: str) -> List[str]:
    csv_files: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith(".csv"):
                continue

            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_dir)
            rel_posix = rel_path.replace(os.sep, "/")

            if filename.lower() == output_filename.lower() or rel_posix.lower() == output_filename.lower():
                continue

            csv_files.append(rel_posix)

    csv_files.sort()
    return csv_files


def normalize_header_cell(cell: str) -> str:
    return (cell or "").strip().lower()


def safe_parse_float(value: str) -> Optional[float]:
    try:
        parsed = float((value or "").strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def safe_parse_episode(value: str) -> Optional[int]:
    parsed = safe_parse_float(value)
    if parsed is None:
        return None
    episode = int(round(parsed))
    if abs(parsed - episode) > 1e-9:
        return None
    return episode


def detect_convergence(
    data_rows: List[List[str]],
    avg_idx: int,
    episode_idx: Optional[int],
) -> Tuple[bool, Optional[int]]:
    """Find the Phase I screening point: 100 consecutive averages at or below 1 mm."""
    streak = 0
    prev_episode: Optional[int] = None
    row_index = 0  # Fallback when logs do not include an episode column.

    for row in data_rows:
        if not row or all((cell or "").strip() == "" for cell in row):
            continue

        row_index += 1

        if episode_idx is not None:
            episode_raw = row[episode_idx] if episode_idx < len(row) else ""
            episode = safe_parse_episode(episode_raw)
        else:
            episode = row_index

        avg_raw = row[avg_idx] if avg_idx < len(row) else ""
        avg_distance = safe_parse_float(avg_raw)

        # Missing or malformed rows break the sustained-threshold evidence.
        qualifies = (
            episode is not None
            and avg_distance is not None
            and avg_distance <= THRESHOLD
        )

        is_consecutive = (
            prev_episode is not None
            and episode is not None
            and episode == prev_episode + 1
        )

        if qualifies:
            if streak > 0 and is_consecutive:
                streak += 1
            else:
                streak = 1

            if streak >= REQUIRED_STREAK:
                return True, episode
        else:
            streak = 0

        prev_episode = episode if episode is not None else None

    return False, None


def evaluate_csv_file(path: str, spec: ConvergenceSpec) -> Tuple[bool, bool, Optional[int]]:
    """Classify one training log for the Phase I convergence table."""
    basename = os.path.basename(path)
    probable_log = bool(spec.log_filename_pattern.match(basename))

    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        # Expected logs that cannot be read still count as screened-but-failed runs.
        if probable_log:
            return True, False, None
        return False, False, None

    header_idx: Optional[int] = None
    avg_idx: Optional[int] = None
    episode_idx: Optional[int] = None

    for idx, row in enumerate(rows):
        normalized = [normalize_header_cell(cell) for cell in row]
        if "avg_distance" in normalized:
            header_idx = idx
            avg_idx = normalized.index("avg_distance")
            episode_idx = normalized.index("episode") if "episode" in normalized else None
            break

    if header_idx is None or avg_idx is None:
        # Keep malformed expected logs so the screening denominator is auditable.
        if probable_log:
            return True, False, None
        return False, False, None

    data_rows = rows[header_idx + 1 :]
    converged, convergence_episode = detect_convergence(data_rows, avg_idx, episode_idx)
    return True, converged, convergence_episode


def write_results(output_path: str, results: List[Tuple[str, bool, Optional[int]]]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for filename, converged, convergence_episode in results:
            status = "yes" if converged else "no"
            episode = str(convergence_episode) if (converged and convergence_episode is not None) else "NaN"
            writer.writerow([filename, status, episode])


def parse_args(argv: Optional[Sequence[str]], spec: ConvergenceSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Detect Phase 1 convergence for {spec.algorithm.upper()} training logs."
    )
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-file", default=spec.output_file)
    return parser.parse_args(argv)


def run_phase1_convergence(spec: ConvergenceSpec, argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv, spec)
    csv_files = discover_csv_files(args.root_dir, args.output_file)

    results: List[Tuple[str, bool, Optional[int]]] = []
    for rel_path in csv_files:
        full_path = os.path.join(args.root_dir, rel_path)
        is_relevant, converged, convergence_episode = evaluate_csv_file(full_path, spec)
        if is_relevant:
            results.append((rel_path, converged, convergence_episode))

    results.sort(key=lambda item: item[0])
    write_results(os.path.join(args.root_dir, args.output_file), results)
    print(f"[done] {spec.algorithm.upper()}: wrote {len(results)} rows to {args.output_file}")
    return 0


def make_spec(algorithm: str) -> ConvergenceSpec:
    algorithm = algorithm.lower()
    return ConvergenceSpec(
        algorithm=algorithm,
        output_file=f"{algorithm}_convergence.csv",
        log_filename_pattern=re.compile(
            rf"^{re.escape(algorithm)}_continuum_robot_.*\.csv$",
            re.IGNORECASE,
        ),
    )
