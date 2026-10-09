"""Turn the lab's own 200 Hz ring recordings into a WritingRing-style training set.

The output mirrors the official ``clean_data_delete_g`` layout so that the existing
training code (``train_paper_trajectory.py`` / ``train_character_classifier.py``) can be
reused unchanged::

    <out>/user_<k>/<letter_idx>/<idx>_<trial>_{x,board,mask,y,timestamp}.npy
    <out>/meta.csv

``user_<k>`` groups trials by (source, condition); ``letter_idx`` is A=0 .. Z=25 and each
trial is one sample, so one ``action`` directory holds all repetitions of a letter.

Per frame (200 Hz grid over the whole ring recording):

* ``x``      : 6 channels, ``[acc_highpass(3, m/s^2), gyro(3, rad/s)]`` -- the exact input
  convention of the paper pipeline (1 Hz zero-phase high-pass removes gravity);
* ``board``  : pen position in **physical** units (``pixel * mm_per_px / 240``), interpolated
  only inside pen-down runs (pen-up travels are *not* interpolated across).  The per-device
  ``mm_per_px`` is estimated from the raw touch bounding boxes against the known 3x3/5x5 cm
  writing boxes, which removes the canvas-size dependent scaling of raw pixel coordinates
  (the phone and the tablet recordings otherwise disagree by a factor ~2.8);
* ``mask``   : 1 inside pen-down runs;
* ``y``      : ``board[t+1] - board[t]`` where both frames are pen-down, else 0;
* ``timestamp``: ring device seconds for each grid frame.

Per-trial alignment defaults to ``--align correlation``: the ring motion envelope is
correlated against the pen-down speed profile on the tablet clock and the lag with the
highest correlation wins.  The profile keeps the velocity spikes at the pen-down/pen-up
instants (the paper's "spike alignment"), which is what makes the estimate sharp: swapping
it for a stroke-interior speed profile or a box-shaped pen window collapses the achievable
target fit (linear-probe test R^2 0.56 -> ~0.0).

Usage (repo parent directory, ``writing_state -> Ring`` symlink)::

    python -m writing_state.build_lab_dataset --output-dir /data/huyang/datasets/RingLab/lab200_v1
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .evaluate_real_data import (
    DEFAULT_FS,
    DEFAULT_RAW_ROOT,
    estimate_alignment,
    estimate_contrast_lag,
    motion_profile,
    pen_runs,
    read_gt,
    read_imu,
    read_imu_masks,
    read_manifest,
    uniform_grid,
)

USER_GROUPS = [
    ("fk", "3x3-wrist-lifted"),
    ("fk", "3x3-wrist-resting"),
    ("fk", "5x5-wrist-lifted"),
    ("fk", "5x5-wrist-resting"),
    ("yjx", "3x3-wrist-lifted"),
    ("yjx", "3x3-wrist-resting"),
    ("fk", "new"),
    ("yjx", "new"),
]
GROUP_INDEX = {key: index for index, key in enumerate(USER_GROUPS)}  # (source, condition) -> user_k
BOARD_MM_REFERENCE = 240.0  # 1.0 board unit == 240 mm, matching the paper's Sensel Morph width


def box_size_mm(condition: str) -> float:
    """``3x3-wrist-lifted`` -> 30 mm: the writing box the letter must fill."""

    head = condition.split("x", 1)[0].split("-", 1)[0].strip()
    value = float(head)
    return value * 10.0 if value < 20.0 else value


def gt_raw_bbox_height_px(path: Path) -> float:
    """Height of the raw touch bounding box, in tablet pixels."""

    ys: list[float] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ys.append(float(row["y"]))
    if len(ys) < 2:
        return 0.0
    return float(max(ys) - min(ys))


def estimate_device_scales(trials: Sequence[Any], progress: int = 0) -> dict[str, float]:
    """Millimetres per tablet pixel for every recording source.

    Letters are written inside a known physical box (3x3 or 5x5 cm) and in practice fill it
    vertically, so ``bbox_height_px ~ px_per_mm * box_mm``.  A least-squares slope through the
    origin over both box sizes recovers ``px_per_mm`` per device; the inverse converts raw
    touch coordinates to millimetres.  Without this step each device lives in its own canvas
    scale (phone 894-956 px vs tablet 1080 px wide) and a single model cannot fit both.
    """

    numerator: dict[str, float] = {}
    denominator: dict[str, float] = {}
    for index, trial in enumerate(trials, start=1):
        try:
            size = box_size_mm(trial.condition)
        except ValueError:  # conditions without a guide box (e.g. the "new" re-collection)
            continue
        height = gt_raw_bbox_height_px(trial.gt_raw_path)
        if height <= 0:
            continue
        numerator[trial.source] = numerator.get(trial.source, 0.0) + size * height
        denominator[trial.source] = denominator.get(trial.source, 0.0) + size * size
        if progress and index % progress == 0:
            print(f"[lab-scale] {index}/{len(trials)}", flush=True)
    scales: dict[str, float] = {}
    for source, total in numerator.items():
        px_per_mm = total / max(denominator[source], 1e-9)
        if px_per_mm > 0:
            scales[source] = 1.0 / px_per_mm
    return scales


def remove_gravity(acc: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    b, a = butter(2, cutoff_hz / (fs / 2.0), btype="high")
    return filtfilt(b, a, acc, axis=0)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def pen_speed_profile(runs: Sequence[Sequence[tuple[float, float, float]]], grid: np.ndarray,
                      offset: float, shift: float, fs: float, mm_per_px: float) -> np.ndarray:
    """Pen-tip speed (mm/s) on the ring grid, zero outside the pen-down runs.

    Positions are interpolated inside each run only, differentiated on the *whole* grid and
    then masked back to the runs.  The zero position outside the runs leaves the sharp
    velocity spikes at the pen-down/pen-up instants intact; those edges are what the paper's
    "spike alignment" matches against the motion burst, and they make the per-trial lag
    estimate far sharper than the stroke-interior speed structure.
    """

    board = np.zeros((len(grid), 2), dtype=np.float64)
    mask = np.zeros(len(grid), dtype=bool)
    for run in runs:
        if len(run) < 2:
            continue
        stamps = np.asarray([point[0] for point in run], dtype=np.float64) - offset + shift
        local = np.asarray([point[1] for point in run], dtype=np.float64) * mm_per_px
        vertical = np.asarray([point[2] for point in run], dtype=np.float64) * mm_per_px
        inside = (grid >= stamps[0]) & (grid <= stamps[-1])
        if inside.sum() < 2:
            continue
        board[inside, 0] = np.interp(grid[inside], stamps, local)
        board[inside, 1] = np.interp(grid[inside], stamps, vertical)
        mask[inside] = True
    speed = np.hypot(np.gradient(board[:, 0]), np.gradient(board[:, 1])) * fs
    return speed * mask


def estimate_speed_lag(runs: Sequence[Sequence[tuple[float, float, float]]], grid: np.ndarray,
                       envelope: np.ndarray, offset: float, fs: float, mm_per_px: float,
                       search: tuple[float, float, float] = (-1.0, 2.5, 0.01),
                       smooth: int = 9) -> tuple[float, float]:
    """Lag (seconds) that best matches the pen-down speed profile to the ring motion envelope.

    Positive lag means the tablet window is late relative to the ring envelope: applying
    ``offset -= lag`` re-synchronises the pen trace.  The candidate profile is the *masked*
    pen speed, so the criterion reacts to the stroke-level speed structure instead of the
    box-shaped pen window, which makes the optimum far sharper than the ratio criterion.
    """

    reference = moving_average(envelope, smooth)
    reference = (reference - reference.mean()) / max(reference.std(), 1e-9)

    def score(shift: float) -> float:
        speed = moving_average(pen_speed_profile(runs, grid, offset, shift, fs, mm_per_px), smooth)
        if speed.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(reference, speed)[0, 1])

    coarse = search[2] if search[2] > 0 else 0.02
    best_shift, best_score = 0.0, float("-inf")
    for shift in np.arange(search[0], search[1] + 1e-9, coarse):
        value = score(shift)
        if np.isfinite(value) and value > best_score:
            best_shift, best_score = float(shift), value
    for shift in np.arange(best_shift - coarse, best_shift + coarse + 1e-9, 0.005):
        value = score(shift)
        if np.isfinite(value) and value > best_score:
            best_shift, best_score = float(shift), value
    return best_shift, best_score


def build_trial(trial: Any, fs: float, cutoff_hz: float, mm_per_px: float = 0.0,
                align: str = "correlation",
                search: tuple[float, float, float] = (-1.0, 2.5, 0.01)) -> dict[str, Any]:
    imu = read_imu(trial.imu_path)
    gt = read_gt(trial.gt_path)
    grid = uniform_grid(imu.t, fs)
    acc = np.stack([np.interp(grid, imu.t, imu.acc[:, axis]) for axis in range(3)], axis=1)
    gyro = np.stack([np.interp(grid, imu.t, imu.gyro[:, axis]) for axis in range(3)], axis=1)
    x = np.concatenate([remove_gravity(acc, fs, cutoff_hz), gyro], axis=1).astype(np.float32)

    if mm_per_px <= 0.0:
        raise ValueError("mm_per_px must be positive; pass a per-device scale from estimate_device_scales")
    runs = pen_runs(trial.gt_raw_path)
    epoch = float(np.median(imu.received_at - imu.t))
    offset = epoch + trial.delta
    lag, lag_score = 0.0, float("nan")
    if align == "received":
        offset = epoch + (float(np.median(gt.t_unix)) - float(np.median(imu.received_at)))
    elif align == "end":
        offset = float(gt.t_unix[-1]) - float(imu.t[-1])
    elif align == "contrast":
        profile = motion_profile(x[:, :3], x[:, 3:])
        lag, lag_score = estimate_contrast_lag(profile, grid, runs, offset, search)
        offset = offset - lag
    elif align == "correlation":
        envelope = motion_profile(x[:, :3], x[:, 3:])
        lag, lag_score = estimate_speed_lag(
            runs, grid, envelope, offset, fs, mm_per_px, search
        )
        offset = offset - lag
    elif align == "tablet_mask":
        # Re-collected drops ship per-frame contact columns inside the IMU csv; the tablet's
        # own down/up events remain the contact ground truth, so the mask is only used to
        # place the window, then a narrow spike search polishes it.
        masks = read_imu_masks(trial.imu_path)
        column = next((name for name in ("mask_v3", "mask_v2", "mask") if name in masks), None)
        if column is None:
            raise ValueError(f"no contact mask column in {trial.imu_path.name}")
        active = np.where(masks[column] > 0.5)[0]
        if len(active) < 2:
            raise ValueError(f"empty contact mask in {trial.imu_path.name}")
        first = max(int(active[0]) - 1, 0)  # the collector marks the impact frame as 0
        offset = runs[0][0][0] - float(imu.t[first])
        envelope = motion_profile(x[:, :3], x[:, 3:])
        lag, lag_score = estimate_speed_lag(
            runs, grid, envelope, offset, fs, mm_per_px, search=(-0.2, 0.2, 0.01)
        )
        offset = offset - lag
    elif align != "manifest":
        raise ValueError(f"unknown align mode: {align}")

    count = len(grid)
    board = np.zeros((count, 2), dtype=np.float64)
    mask = np.zeros(count, dtype=np.float32)
    for run in runs:
        times = np.asarray([point[0] for point in run], dtype=np.float64) - offset
        xs = np.asarray([point[1] for point in run], dtype=np.float64) * mm_per_px / BOARD_MM_REFERENCE
        ys = np.asarray([point[2] for point in run], dtype=np.float64) * mm_per_px / BOARD_MM_REFERENCE
        if times[-1] < grid[0] or times[0] > grid[-1]:
            continue
        order = np.argsort(times)
        times, xs, ys = times[order], xs[order], ys[order]
        inside = (grid >= times[0]) & (grid <= times[-1])
        if inside.sum() < 2:
            continue
        board[inside, 0] = np.interp(grid[inside], times, xs)
        board[inside, 1] = np.interp(grid[inside], times, ys)
        mask[inside] = 1.0

    pen_seconds = 0.0
    covered_seconds = 0.0
    for run in runs:
        if len(run) < 2:
            continue
        start = run[0][0] - offset
        stop = run[-1][0] - offset
        pen_seconds += max(0.0, stop - start)
        covered_seconds += max(0.0, min(stop, float(grid[-1])) - max(start, float(grid[0])))

    y = np.zeros_like(board, dtype=np.float64)
    valid = (mask[1:] > 0) & (mask[:-1] > 0)
    y[:-1][valid] = board[1:][valid] - board[:-1][valid]
    return {
        "arrays": {
            "x": x,
            "board": board.astype(np.float32),
            "mask": mask,
            "y": y.astype(np.float32),
            "timestamp": grid.astype(np.float64),
        },
        "offset": offset,
        "lag": lag,
        "lag_score": lag_score,
        "pen_seconds": pen_seconds,
        "coverage": covered_seconds / pen_seconds if pen_seconds > 1e-6 else 1.0,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trajectory dataset from the lab 200 Hz recordings")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--gravity-cutoff", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--align", choices=("manifest", "received", "end", "contrast", "correlation",
                                            "tablet_mask"),
                        default="correlation",
                        help="per-trial ring<->tablet alignment; 'correlation' maximises the "
                             "correlation between the ring motion envelope and the pen speed "
                             "(recommended), 'contrast' maximises the motion burst inside the pen "
                             "window, 'tablet_mask' uses the collector's mask column as the anchor")
    parser.add_argument("--device-scale", type=float, default=None,
                        help="override the auto-estimated mm-per-pixel scale (single source only)")
    parser.add_argument("--scale-json", type=Path, default=None,
                        help="reuse a previously written per-source mm-per-pixel calibration")
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="drop trials whose pen-down window is only partially inside the ring "
                             "recording (coverage = captured pen time / total pen time)")
    parser.add_argument("--progress", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    trials = read_manifest(args.raw_root, args.conditions, args.sources)
    if args.limit:
        trials = trials[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.scale_json and args.scale_json.exists():
        payload = json.loads(args.scale_json.read_text(encoding="utf-8"))
        table = payload.get("mm_per_px") or payload.get("device_mm_per_px") or {}
        device_scales = {key: float(value) for key, value in table.items()}
        if not device_scales:
            raise SystemExit(f"no per-device mm_per_px table found in {args.scale_json}")
        print(f"[lab-scale] reused {args.scale_json}: {device_scales}", flush=True)
    else:
        device_scales = estimate_device_scales(trials, progress=args.progress)
    if args.device_scale is not None:
        device_scales = {source: float(args.device_scale) for source in device_scales}
    print(f"[lab-scale] mm per pixel: "
          f"{ {key: round(value, 5) for key, value in device_scales.items()} }", flush=True)
    missing = sorted({trial.source for trial in trials} - set(device_scales))
    if missing:
        raise SystemExit(f"no device scale for sources: {missing}")

    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for index, trial in enumerate(trials, start=1):
        key = (trial.source, trial.condition)
        if key not in GROUP_INDEX:
            skipped["unknown_group"] += 1
            continue
        try:
            built = build_trial(trial, args.fs, args.gravity_cutoff,
                                mm_per_px=device_scales[trial.source], align=args.align)
        except Exception as error:  # pragma: no cover - per-trial robustness
            skipped[f"error:{type(error).__name__}"] += 1
            continue
        arrays = built["arrays"]
        if arrays["mask"].sum() < 16:
            skipped["short_pen_window"] += 1
            continue
        if built["coverage"] < args.min_coverage:
            skipped["low_coverage"] += 1
            continue
        user = f"user_{GROUP_INDEX[key]}"
        action = args.output_dir / user / str(trial.letter_idx)
        action.mkdir(parents=True, exist_ok=True)
        sample = f"{index:05d}_{trial.trial}"
        for name, array in arrays.items():
            np.save(action / f"{sample}_{name}.npy", array)
        rows.append({
            "user": user,
            "action": trial.letter_idx,
            "sample": sample,
            "trial": trial.trial,
            "source": trial.source,
            "condition": trial.condition,
            "letter": trial.letter.upper(),
            "frames": int(len(arrays["mask"])),
            "pen_frames": int(arrays["mask"].sum()),
            "duration_s": round(float(arrays["timestamp"][-1] - arrays["timestamp"][0]), 3),
            "pen_duration_s": round(float(arrays["mask"].sum() / args.fs), 3),
            "delta": trial.delta,
            "device_mm_per_px": round(float(device_scales[trial.source]), 6),
            "align_lag_s": round(float(built["lag"]), 4),
            "coverage": round(float(built["coverage"]), 4),
            "pen_seconds": round(float(built["pen_seconds"]), 4),
            "align_score": round(float(built["lag_score"]), 3) if built["lag_score"] == built["lag_score"] else "",
            "offset": built["offset"],
        })
        if args.progress and index % args.progress == 0:
            print(f"[lab-dataset] {index}/{len(trials)}", flush=True)

    with (args.output_dir / "meta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "raw_root": str(args.raw_root),
        "output_dir": str(args.output_dir),
        "samples": len(rows),
        "skipped": dict(skipped),
        "users": {user: sum(1 for row in rows if row["user"] == user) for user in sorted({row["user"] for row in rows})},
        "frames_total": int(sum(row["frames"] for row in rows)),
        "pen_frames_total": int(sum(row["pen_frames"] for row in rows)),
        "fs": args.fs,
        "gravity_cutoff_hz": args.gravity_cutoff,
        "align": args.align,
        "min_coverage": args.min_coverage,
        "align_note": "manifest/received/end use the per-trial clock estimates; correlation maximises "
                      "the pen-speed/motion-envelope correlation; contrast maximises the motion burst "
                      "inside the pen-down window (lag stored per trial in meta.csv)",
        "device_mm_per_px": {source: round(float(value), 6) for source, value in device_scales.items()},
        "board_unit": f"pixel * mm_per_px / {BOARD_MM_REFERENCE:g} (1.0 == 240 mm)",
        "input_channels": ["acc_hp_x", "acc_hp_y", "acc_hp_z", "gyro_x", "gyro_y", "gyro_z"],
        "target": "y[t] = board[t+1] - board[t], valid where mask[t] & mask[t+1]",
    }
    with (args.output_dir / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
