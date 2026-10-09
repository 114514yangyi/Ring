"""Health check for a (re-)collected lab 200 Hz drop.

For every trial it answers three questions without touching any model:

1. **Is the pen stroke fully inside the ring recording?**  ``coverage`` = fraction of the
   tablet pen time that falls inside the ring window, and ``start_ratio``/``end_ratio`` =
   where the pen window sits inside the window (the collectors' rule of thumb is 19%-73%).
2. **Do the two alignments agree?**  Ours (``estimate_speed_lag``: correlation between the ring
   motion envelope and the pen speed profile) versus the collector's own ``mask*`` columns.
   Agreement within ~0.1 s means both are right.
3. **Which trials are still suspicious?**  Anything with coverage < 0.95, a pen window outside
   19%-73%, or an alignment disagreement > 0.15 s is listed for review.

Works with or without a ``_meta/manifest.csv``: when the manifest is absent the coarse clock
offset is taken from the collector's mask column (or from the first-event/first-frame pairing),
then refined with the same spike alignment used by ``build_lab_dataset``.

Usage::

    python -m writing_state.verify_lab_alignment --root /path/to/raw/new \
        --out outputs/lab_eval/new_data_check
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .build_lab_dataset import (
    box_size_mm,
    estimate_device_scales,
    estimate_speed_lag,
    gt_raw_bbox_height_px,
    remove_gravity,
)
from .evaluate_real_data import (
    DEFAULT_FS,
    Trial,
    motion_profile,
    pen_runs,
    read_imu,
    read_manifest,
    uniform_grid,
)

MASK_COLUMNS = ("mask_v3", "mask_v2", "mask")
POSITION_BAND = (0.19, 0.73)


def find_imu_masks(path: Path) -> dict[str, np.ndarray]:
    """Read the collector's per-frame mask columns from a raw IMU csv (any subset)."""

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = [name for name in (reader.fieldnames or []) if name.strip().lower() in MASK_COLUMNS]
        values: dict[str, list[float]] = {name.strip().lower(): [] for name in names}
        for row in reader:
            for name in names:
                raw = (row.get(name) or "").strip()
                values[name.strip().lower()].append(float(raw) if raw else 0.0)
    return {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}


def discover_trials(root: Path) -> list[Trial]:
    """Same rows as ``read_manifest`` but scanned from disk when no manifest exists."""

    manifest = root / "_meta" / "manifest.csv"
    if manifest.exists():
        return read_manifest(root)
    trials: list[Trial] = []
    for imu_path in sorted(root.glob("*/*/*_imu.csv")):
        trial = imu_path.name[: -len("_imu.csv")]
        gt_raw = imu_path.with_name(f"{trial}_gt_raw.csv")
        if not gt_raw.exists():
            continue
        parts = trial.split("_")
        trials.append(
            Trial(
                condition=imu_path.parent.parent.name,
                letter=imu_path.parent.name.lower(),
                letter_idx=max(ord(imu_path.parent.name.lower()[:1]) - ord("a"), 0),
                trial=trial,
                source=parts[-2] if len(parts) >= 3 else "unknown",
                imu_path=imu_path,
                gt_path=imu_path.with_name(f"{trial}_gt_100hz.csv"),
                gt_raw_path=gt_raw,
                delta=float("nan"),
            )
        )
    return trials


def mask_run(mask: np.ndarray, grid: np.ndarray) -> tuple[float, float] | None:
    """First and last grid timestamp flagged as pen contact."""

    active = np.where(mask > 0.5)[0]
    if len(active) < 2:
        return None
    return float(grid[active[0]]), float(grid[active[-1]])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/lab_eval/new_data_check"))
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--gravity-cutoff", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--search", nargs=3, type=float, metavar=("LOW", "HIGH", "STEP"),
                        default=[-1.0, 2.5, 0.01],
                        help="spike-alignment lag search range, same default as build_lab_dataset")
    parser.add_argument("--scale-json", type=Path, default=None,
                        help="per-source mm-per-pixel table; defaults to <root>/_meta/device_scale.json")
    parser.add_argument("--progress", type=int, default=0)
    args = parser.parse_args(argv)

    trials = discover_trials(args.root)
    if args.limit:
        trials = trials[: args.limit]
    if not trials:
        raise SystemExit(f"no trials found under {args.root}")

    has_manifest = any(trial.delta == trial.delta for trial in trials)
    scale_json = args.scale_json or (args.root / "_meta" / "device_scale.json")
    if scale_json.exists():
        scales = {str(key): float(value) for key, value in json.loads(scale_json.read_text()).items()}
        print(f"[check] 复用 {scale_json}: {scales}", flush=True)
    else:
        scales = estimate_device_scales(trials, progress=args.progress)
    print(f"[check] {len(trials)} trials | manifest={'yes' if has_manifest else 'no'} "
          f"| mm/px={ {key: round(v, 5) for key, v in scales.items()} }", flush=True)

    rows: list[dict[str, Any]] = []
    for index, trial in enumerate(trials, start=1):
        imu = read_imu(trial.imu_path)
        grid = uniform_grid(imu.t, args.fs)
        acc = np.stack([np.interp(grid, imu.t, imu.acc[:, axis]) for axis in range(3)], axis=1)
        gyro = np.stack([np.interp(grid, imu.t, imu.gyro[:, axis]) for axis in range(3)], axis=1)
        x = np.concatenate([remove_gravity(acc, args.fs, args.gravity_cutoff), gyro], axis=1)
        envelope = motion_profile(x[:, :3], x[:, 3:])
        runs = [run for run in pen_runs(trial.gt_raw_path) if len(run) > 1]
        if not runs:
            continue
        masks = find_imu_masks(trial.imu_path)
        epoch = float(np.median(imu.received_at - imu.t))
        pen_start, pen_end = runs[0][0][0], runs[-1][-1][0]
        pen_seconds = sum(run[-1][0] - run[0][0] for run in runs)

        if has_manifest:
            coarse = epoch + trial.delta
        else:
            coarse = None
            for name in MASK_COLUMNS:  # prefer the collector's own mask for the coarse offset
                if name in masks:
                    span = mask_run(masks[name], grid)
                    if span:
                        coarse = pen_start - span[0]
                        break
            if coarse is None:
                coarse = pen_start - float(imu.received_at[0])
        lag, lag_score = estimate_speed_lag(
            runs, grid, envelope, coarse, args.fs, scales[trial.source],
            search=tuple(args.search),
        )
        offset = coarse - lag

        start = pen_start - offset - grid[0]
        end = pen_end - offset - grid[0]
        span = float(grid[-1] - grid[0])
        inside = sum(
            max(0.0, min(run[-1][0] - offset, float(grid[-1])) - max(run[0][0] - offset, float(grid[0])))
            for run in runs
        )
        row: dict[str, Any] = {
            "condition": trial.condition,
            "trial": trial.trial,
            "letter": trial.letter.upper(),
            "source": trial.source,
            "window_s": round(span, 3),
            "pen_s": round(pen_seconds, 3),
            "coverage": round(inside / pen_seconds, 4) if pen_seconds > 1e-6 else 1.0,
            "start_ratio": round(start / span, 4) if span > 0 else float("nan"),
            "end_ratio": round(end / span, 4) if span > 0 else float("nan"),
            "lag_s": round(float(lag), 4),
            "lag_score": round(float(lag_score), 3) if lag_score == lag_score else "",
        }
        for name in MASK_COLUMNS:
            if name not in masks:
                continue
            span_m = mask_run(masks[name], grid)
            if span_m is None:
                continue
            row[f"{name}_start_ratio"] = round((span_m[0] - grid[0]) / span, 4)
            row[f"{name}_end_ratio"] = round((span_m[1] - grid[0]) / span, 4)
            row[f"{name}_delta_s"] = round((span_m[0] - grid[0]) - start, 4)
        low, high, step = args.search
        if abs(lag - low) <= step or abs(lag - high) <= step:
            row["boundary_hit"] = 1
        rows.append(row)
        if args.progress and index % args.progress == 0:
            print(f"[check] {index}/{len(trials)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with (args.out / "per_trial.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    boundary = sum(row.get("boundary_hit", 0) for row in rows)
    if boundary:
        print(f"[warn] {boundary} 个 trial 的最优 lag 撞到搜索边界，建议放宽 --search")
    coverage = np.asarray([row["coverage"] for row in rows])
    start_ratio = np.asarray([row["start_ratio"] for row in rows])
    end_ratio = np.asarray([row["end_ratio"] for row in rows])
    print(f"\n=== {len(rows)} trials ===")
    print("覆盖率: 中位 %.3f | >=0.95 %.1f%% | >=0.90 %.1f%% | <0.60 %.1f%%" % (
        np.median(coverage), 100 * (coverage >= 0.95).mean(), 100 * (coverage >= 0.90).mean(),
        100 * (coverage < 0.60).mean()))
    print("接触段位置: start %.2f [%.2f, %.2f]  end %.2f [%.2f, %.2f]  (期望落在 %.0f%%~%.0f%%)" % (
        np.median(start_ratio), np.percentile(start_ratio, 5), np.percentile(start_ratio, 95),
        np.median(end_ratio), np.percentile(end_ratio, 5), np.percentile(end_ratio, 95),
        100 * POSITION_BAND[0], 100 * POSITION_BAND[1]))
    for name in MASK_COLUMNS:
        key = f"{name}_delta_s"
        if rows[0].get(key) is None:
            continue
        delta = np.asarray([row[key] for row in rows], dtype=np.float64)
        print("%s vs 我方尖峰对齐: 中位 %+.3f s | |差| 中位 %.3f | 差>0.15 s 占 %.1f%%" % (
            name, np.median(delta), np.median(np.abs(delta)), 100 * (np.abs(delta) > 0.15).mean()))

    review = [
        row for row in rows
        if row["coverage"] < 0.95
        or not (POSITION_BAND[0] <= row["start_ratio"] <= POSITION_BAND[1])
        or any(abs(row.get(f"{name}_delta_s", 0.0)) > 0.15 for name in MASK_COLUMNS if row.get(f"{name}_delta_s") is not None)
    ]
    print(f"需要复核: {len(review)} 个 -> {args.out / 'per_trial.csv'}")
    for row in review[:10]:
        print("   %-22s %s cov=%.2f start=%.2f end=%.2f" % (
            row["trial"], row["letter"], row["coverage"], row["start_ratio"], row["end_ratio"]))


if __name__ == "__main__":  # pragma: no cover
    main()
