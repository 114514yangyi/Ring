"""Evaluate the WritingRing trajectory and character models on the lab's own ring dataset.

Data layout (``data/raw/200hz``)::

    <condition>/<LETTER>/<letter>_<source>_<n>_{imu,gt_raw,gt_100hz}.csv
    _meta/manifest.csv     # per-trial rows with `delta` = tablet clock - ring clock

The ring logs 23 columns (acc/gyro in g and deg/s, device `timestamp` in seconds since
power-on, `received_at` in PC Unix seconds, plus the device's own `lin_acc`).  The tablet
provides the pen trace in CSS pixels with its own Unix clock, so each trial needs the
per-trial constant offset from the manifest (``delta``).

Pipeline mirrors the paper evaluation:

1. resample the IMU onto a uniform 200 Hz grid, convert units (g -> m/s2, deg/s -> rad/s),
   remove gravity with the same 1 Hz zero-phase high-pass used for the official data;
2. feed the 6-channel stream to the trajectory TCN+LSTM, integrate the predicted per-frame
   displacements inside the pen-down interval given by the touch ground truth;
3. compare the reconstructed path with the tablet trace (paper normalised metric, plus a
   scale-aligned variant that removes the unknown device gain);
4. classify both the ground-truth and the reconstructed path with the 26-class letter CNN.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .character_dataset import normalize_trajectory, resample_trajectory
from .character_model import load_character_model
from .paper_trajectory import (
    integrate_contact_segments,
    load_trajectory_predictor,
)

GRAVITY = 9.80665
DEFAULT_FS = 200.0
DEFAULT_RAW_ROOT = Path("/home/huyang/data/trae_projects/raw/200hz")
REPO_ROOT = Path(__file__).resolve().parent
LABELS = [chr(ord("A") + index) for index in range(26)]


@dataclass(frozen=True)
class Trial:
    condition: str
    letter: str
    letter_idx: int
    trial: str
    source: str
    imu_path: Path
    gt_path: Path
    gt_raw_path: Path
    delta: float


def _read_csv(path: Path, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with open(path, newline="", encoding=encoding) as handle:
        return list(csv.DictReader(handle))


def read_manifest(root: Path, conditions: Sequence[str] | None = None,
                  sources: Sequence[str] | None = None) -> list[Trial]:
    rows = _read_csv(root / "_meta" / "manifest.csv", encoding="utf-8-sig")
    trials: list[Trial] = []
    for row in rows:
        condition = row["condition"].strip()
        if conditions and condition not in conditions:
            continue
        source = row["trial"].rsplit("_", 2)[1]
        if sources and source not in sources:
            continue
        trials.append(
            Trial(
                condition=condition,
                letter=row["letter"].strip().lower(),
                letter_idx=int(row["letter_idx"]),
                trial=row["trial"].strip(),
                source=source,
                imu_path=root / row["imu_path"].strip(),
                gt_path=root / row["gt_100hz_path"].strip(),
                gt_raw_path=root / row["gt_raw_path"].strip(),
                delta=float(row["delta"]),
            )
        )
    return trials


@dataclass
class ImuStream:
    t: np.ndarray            # device seconds
    acc: np.ndarray          # (n, 3) m/s^2, gravity included
    lin_device: np.ndarray   # (n, 3) m/s^2, device fusion output
    gyro: np.ndarray         # (n, 3) rad/s
    quat: np.ndarray         # (n, 4) w, x, y, z (device fusion)
    received_at: np.ndarray  # (n,) PC unix seconds


def read_imu(path: Path) -> ImuStream:
    rows = _read_csv(path)
    t = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
    acc = np.asarray([[float(r[f"acc_{a}"]) for a in "xyz"] for r in rows], dtype=np.float64) * GRAVITY
    lin = np.asarray([[float(r[f"lin_acc_{a}"]) for a in "xyz"] for r in rows], dtype=np.float64) * GRAVITY
    gyro = np.asarray([[float(r[f"gyro_{a}"]) for a in "xyz"] for r in rows], dtype=np.float64) * np.pi / 180.0
    quat = np.asarray([[float(r[f"quat_{a}"]) for a in ("w", "x", "y", "z")] for r in rows], dtype=np.float64)
    received = np.asarray([float(r["received_at"]) for r in rows], dtype=np.float64)
    return ImuStream(t=t, acc=acc, lin_device=lin, gyro=gyro, quat=quat, received_at=received)


def read_imu_masks(path: Path) -> dict[str, np.ndarray]:
    """Per-frame contact columns shipped with the re-collected drop (``mask*``).

    The collector labels the pen-contact window straight inside the IMU csv with a 0/1
    column, one value per frame.  Three flavours coexist: ``mask`` (largest lin-acc spike),
    ``mask_v2`` (whole-shape cross-correlation against the pen speed) and ``mask_v3`` (the
    more plausible of the two under a position prior).  Missing columns are simply absent.
    """

    wanted = ("mask", "mask_v2", "mask_v3")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = [name.strip().lower() for name in (reader.fieldnames or []) if name.strip().lower() in wanted]
        columns: dict[str, list[float]] = {name: [] for name in names}
        for row in reader:
            for original, name in zip(reader.fieldnames or [], [n.strip().lower() for n in (reader.fieldnames or [])]):
                if name in columns:
                    raw = (row.get(original) or "").strip()
                    columns[name].append(float(raw) if raw else 0.0)
    return {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


@dataclass
class GtTrack:
    t_unix: np.ndarray
    xy_px: np.ndarray
    width: float
    height: float


def read_gt(path: Path) -> GtTrack:
    rows = _read_csv(path)
    t = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
    xy = np.asarray([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)
    raw = _read_csv(path.with_name(path.name.replace("_gt_100hz.csv", "_gt_raw.csv")))
    width = float(raw[0]["width"])
    height = float(raw[0]["height"])
    return GtTrack(t_unix=t, xy_px=xy, width=width, height=height)


def pen_runs(path: Path) -> list[list[tuple[float, float, float]]]:
    """Split the raw touch stream into pen-down runs ``[(t, x, y), ...]``."""

    runs: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] | None = None
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            point = (float(row["timestamp"]), float(row["x"]), float(row["y"]))
            event = row["event_type"].strip().lower()
            if event == "down":
                current = [point]
            elif event == "up":
                if current is not None:
                    current.append(point)
                    runs.append(current)
                    current = None
            elif current is not None:
                current.append(point)
    if current is not None and len(current) > 1:
        runs.append(current)
    return runs


def estimate_contrast_lag(profile: np.ndarray, grid: np.ndarray,
                          runs: Sequence[Sequence[tuple[float, float, float]]],
                          base_offset: float,
                          search: tuple[float, float, float] = (-0.4, 1.8, 0.02),
                          min_fraction: float = 0.15) -> tuple[float, float]:
    """Lag that maximises ``mean(motion inside pen window) / mean(motion outside)``.

    A positive lag places the ground truth later in the ring recording.  Both regions must keep
    at least ``min_fraction`` of the frames so the contrast cannot be inflated by shrinking the
    outside region.  This is far more robust than correlating the envelope with the pen speed,
    because the ring is never really still outside the pen window.
    """

    low, high, step = search
    best_lag, best_score = 0.0, -1.0
    for lag in np.arange(low, high + 1e-9, step):
        mask = np.zeros(len(grid), dtype=bool)
        for run in runs:
            times = np.asarray([point[0] for point in run], dtype=np.float64) - base_offset + lag
            mask |= (grid >= times[0]) & (grid <= times[-1])
        inside, outside = int(mask.sum()), int((~mask).sum())
        if inside < 20 or outside < 20:
            continue
        if inside < min_fraction * len(grid) or outside < min_fraction * len(grid):
            continue
        score = float(profile[mask].mean() / max(profile[~mask].mean(), 1e-9))
        if score > best_score:
            best_lag, best_score = float(lag), score
    return best_lag, best_score


def highpass(values: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:  # pragma: no cover - scipy is available in this project
        window = max(1, int(round(fs / max(cutoff_hz, 1e-3))))
        kernel = np.ones(window) / window
        padded = np.pad(values, ((window, window), (0, 0)), mode="edge")
        smooth = np.stack(
            [np.convolve(padded[:, axis], kernel, "same")[window:window + len(values)] for axis in range(values.shape[1])],
            axis=1,
        )
        return values - smooth
    b, a = butter(2, cutoff_hz / (fs / 2.0), btype="high")
    return filtfilt(b, a, values, axis=0)


def uniform_grid(t: np.ndarray, fs: float) -> np.ndarray:
    count = int(np.floor((t[-1] - t[0]) * fs)) + 1
    return t[0] + np.arange(count, dtype=np.float64) / fs


def build_input(imu: ImuStream, grid: np.ndarray, source: str, cutoff_hz: float,
                gain: float) -> np.ndarray:
    def resample(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(grid, imu.t, values[:, axis]) for axis in range(values.shape[1])], axis=1
        )

    if source == "device":
        linear = resample(imu.lin_device)
    else:
        linear = highpass(resample(imu.acc), DEFAULT_FS, cutoff_hz)
    gyro = resample(imu.gyro)
    return (np.concatenate([linear, gyro], axis=1) * gain).astype(np.float32)


def align_offset(imu: ImuStream, delta: float) -> float:
    """Tablet unix seconds -> ring device seconds uses a constant per-trial offset."""

    epoch = float(np.median(imu.received_at - imu.t))
    return epoch + delta


def motion_profile(linear: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """Scale-free motion envelope used for spike alignment against the pen trace."""

    parts = []
    for values in (gyro, linear):
        norm = np.linalg.norm(values, axis=1)
        std = float(norm.std())
        parts.append(norm / std if std > 1e-9 else norm * 0.0)
    return np.sum(parts, axis=0)


def estimate_alignment(grid: np.ndarray, profile: np.ndarray, gt_t: np.ndarray,
                       gt_speed: np.ndarray, search: tuple[float, float],
                       coarse: float = 0.01, fine: float = 0.001) -> tuple[float, float]:
    """Find the lag (seconds) that best matches the pen speed to the IMU motion."""

    def score(lag: float) -> float:
        shifted = np.interp(grid, gt_t + lag, gt_speed, left=0.0, right=0.0)
        if shifted.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(profile, shifted)[0, 1])

    best_lag, best_score = 0.0, float("-inf")
    for lag in np.arange(search[0], search[1] + 1e-9, coarse):
        value = score(lag)
        if np.isfinite(value) and value > best_score:
            best_lag, best_score = float(lag), value
    for lag in np.arange(best_lag - coarse, best_lag + coarse + 1e-9, fine):
        value = score(lag)
        if np.isfinite(value) and value > best_score:
            best_lag, best_score = float(lag), value
    return best_lag, best_score


def scale_aligned(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares signed scalar that maps ``pred`` onto ``gt`` (no offset, no rotation)."""

    denom = float(np.sum(pred * pred))
    if denom < 1e-12:
        return pred, float("nan")
    scale = float(np.sum(pred * gt) / denom)
    return pred * scale, scale


def path_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    span = gt.max(axis=0) - gt.min(axis=0)
    diagonal = float(np.linalg.norm(span))
    if len(gt) < 2 or diagonal < 1e-9:
        return {"n_points": int(len(gt)), "diagonal": diagonal, "skipped": 1}
    raw = np.linalg.norm(pred - gt, axis=1) / diagonal
    aligned_xy, scale = scale_aligned(pred - pred.mean(axis=0), gt - gt.mean(axis=0))
    aligned = np.linalg.norm(aligned_xy + gt.mean(axis=0) - gt, axis=1) / diagonal
    pred_len = float(np.linalg.norm(np.diff(pred, axis=0), axis=1).sum())
    gt_len = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    return {
        "n_points": int(len(gt)),
        "diagonal": diagonal,
        "skipped": 0,
        "normalized_raw": float(raw.mean()),
        "normalized_raw_p50": float(np.percentile(raw, 50)),
        "normalized_aligned": float(aligned.mean()),
        "scale": scale,
        "pred_length": pred_len,
        "gt_length": gt_len,
        "path_ratio": pred_len / gt_len if gt_len > 1e-9 else float("nan"),
        "frac_raw_gt_0.1": float((raw > 0.1).mean()),
    }


def evaluate_trial(trial: Trial, predictor: Any, fs: float, source: str,
                   cutoff_hz: float, gain: float, align: str = "motion",
                   search: tuple[float, float] = (-0.5, 2.5),
                   trim_to_gt: bool = True) -> dict[str, Any]:
    imu = read_imu(trial.imu_path)
    gt = read_gt(trial.gt_path)
    grid = uniform_grid(imu.t, fs)
    offset = align_offset(imu, trial.delta)
    gt_t_device = gt.t_unix - offset
    x = build_input(imu, grid, source, cutoff_hz, gain)
    lag = 0.0
    align_score = float("nan")
    if align == "contrast":
        linear = highpass(
            np.stack([np.interp(grid, imu.t, imu.acc[:, axis]) for axis in range(3)], axis=1),
            fs, cutoff_hz,
        )
        gyro_grid = np.stack([np.interp(grid, imu.t, imu.gyro[:, axis]) for axis in range(3)], axis=1)
        lag, align_score = estimate_contrast_lag(
            motion_profile(linear, gyro_grid), grid, pen_runs(trial.gt_raw_path), offset
        )
        offset = offset - lag
        gt_t_device = gt.t_unix - offset
    elif align == "motion":
        gt_norm_full = gt.xy_px / np.asarray([gt.width, gt.height], dtype=np.float64)
        gt_speed = np.linalg.norm(np.diff(gt_norm_full, axis=0), axis=1) * fs
        linear = highpass(
            np.stack([np.interp(grid, imu.t, imu.acc[:, axis]) for axis in range(3)], axis=1),
            fs, cutoff_hz,
        )
        gyro = np.stack([np.interp(grid, imu.t, imu.gyro[:, axis]) for axis in range(3)], axis=1)
        lag, align_score = estimate_alignment(
            grid, motion_profile(linear, gyro), gt_t_device[1:], gt_speed, search
        )
        gt_t_device = gt_t_device + lag
    in_range = (gt_t_device >= grid[0]) & (gt_t_device <= grid[-1])
    if in_range.sum() < 8:
        return {"trial": trial.trial, "condition": trial.condition, "letter": trial.letter,
                "letter_idx": trial.letter_idx, "source": trial.source,
                "align_lag": lag, "align_score": align_score, "skipped": "gt_out_of_range"}

    gt_t = gt_t_device[in_range]
    gt_norm = gt.xy_px[in_range] / np.asarray([gt.width, gt.height], dtype=np.float64)
    mask = ((grid >= gt_t[0]) & (grid <= gt_t[-1])).astype(np.int64)
    if trim_to_gt:
        keep = (grid >= min(gt_t[0], grid[-1])) & (grid <= max(gt_t[-1], grid[0]))
        x = x[keep]
        mask = mask[keep]
        grid = grid[keep]
        if len(grid) < 16:
            return {"trial": trial.trial, "condition": trial.condition, "letter": trial.letter,
                    "letter_idx": trial.letter_idx, "source": trial.source,
                    "align_lag": lag, "align_score": align_score, "skipped": "short_window"}
    deltas = predictor.predict_deltas(x)
    segments = integrate_contact_segments(
        deltas, mask, window_offset=predictor.config.window_offset, min_frames=5
    )
    if not segments:
        return {"trial": trial.trial, "condition": trial.condition, "letter": trial.letter,
                "letter_idx": trial.letter_idx, "source": trial.source,
                "align_lag": lag, "align_score": align_score, "skipped": "no_segment"}

    pred_xy = np.concatenate([segment.xy for segment in segments], axis=0)
    gt_grid = np.stack(
        [np.interp(grid, gt_t, gt_norm[:, axis]) for axis in range(2)], axis=1
    )
    contact = mask.astype(bool)
    gt_xy = gt_grid[contact][: len(pred_xy)]
    pred_xy = pred_xy[: len(gt_xy)]
    if len(gt_xy) < 8:
        return {"trial": trial.trial, "condition": trial.condition, "letter": trial.letter,
                "letter_idx": trial.letter_idx, "source": trial.source,
                "align_lag": lag, "align_score": align_score, "skipped": "short_overlap"}
    pred_xy = pred_xy - pred_xy.mean(axis=0) + gt_xy.mean(axis=0)
    metrics = path_metrics(pred_xy, gt_xy)
    return {
        "trial": trial.trial,
        "condition": trial.condition,
        "letter": trial.letter,
        "letter_idx": trial.letter_idx,
        "source": trial.source,
        "align_lag": lag,
        "align_score": align_score,
        "skipped": "",
        "gt_traj": gt_xy,
        "pred_traj": pred_xy,
        **metrics,
    }


def classify(model: Any, trajectories: Sequence[np.ndarray], batch_size: int = 512) -> np.ndarray:
    import torch

    if not trajectories:
        return np.zeros((0, 26), dtype=np.float32)
    inputs = np.stack(
        [
            normalize_trajectory(resample_trajectory(points, 64)).astype(np.float32)
            for points in trajectories
        ]
    )
    logits: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start:start + batch_size])
            logits.append(model(batch).cpu().numpy())
    return np.concatenate(logits, axis=0)


def accuracy_block(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    if len(labels) == 0:
        return {"n": 0, "top1": float("nan"), "top3": float("nan")}
    top1 = probs.argmax(axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    return {
        "n": int(len(labels)),
        "top1": float((top1 == labels).mean()),
        "top3": float(np.mean([label in row for label, row in zip(labels, top3)])),
    }


def per_letter_accuracy(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    top1 = probs.argmax(axis=1)
    out: dict[str, Any] = {}
    for index, letter in enumerate(LABELS):
        selected = labels == index
        if not selected.any():
            continue
        out[letter] = {
            "n": int(selected.sum()),
            "top1": float((top1[selected] == index).mean()),
            "top3": float(np.mean([index in row for row in np.argsort(-probs[selected], axis=1)[:, :3]])),
        }
    return out


def confusion_matrix(labels: np.ndarray, probs: np.ndarray) -> list[list[int]]:
    top1 = probs.argmax(axis=1)
    matrix = np.zeros((26, 26), dtype=int)
    for label, prediction in zip(labels, top1):
        matrix[int(label), int(prediction)] += 1
    return matrix.tolist()


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("skipped")]
    weights = np.asarray([row["n_points"] for row in valid], dtype=np.float64)

    def weighted(key: str) -> float:
        values = np.asarray([row.get(key, np.nan) for row in valid], dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            return float("nan")
        return float((values[finite] * weights[finite]).sum() / weights[finite].sum())

    return {
        "trials": len(rows),
        "evaluated": len(valid),
        "skipped": {
            str(key): int(sum(1 for row in rows if row.get("skipped") == key))
            for key in {row.get("skipped") for row in rows if row.get("skipped")}
        },
        "normalized_raw": weighted("normalized_raw"),
        "normalized_aligned": weighted("normalized_aligned"),
        "normalized_raw_p50": weighted("normalized_raw_p50"),
        "path_ratio": weighted("path_ratio"),
        "frac_raw_gt_0.1": weighted("frac_raw_gt_0.1"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.raw_root)
    trials = read_manifest(root, conditions=args.conditions, sources=args.sources)
    if args.stride and args.stride > 1:
        trials = trials[:: args.stride]
    if args.limit:
        trials = trials[: args.limit]
    if not trials:
        raise SystemExit("no trials selected")
    predictor = load_trajectory_predictor(args.trajectory_checkpoint, device=args.device)
    rows: list[dict[str, Any]] = []
    for index, trial in enumerate(trials, 1):
        try:
            rows.append(
                evaluate_trial(trial, predictor, args.fs, args.imu_source,
                               args.gravity_cutoff, args.input_gain,
                               align=args.align, search=tuple(args.align_search))
            )
        except Exception as error:  # pragma: no cover - defensive per-trial isolation
            rows.append({"trial": trial.trial, "condition": trial.condition, "letter": trial.letter,
                         "letter_idx": trial.letter_idx, "source": trial.source,
                         "skipped": f"error:{type(error).__name__}"})
        if args.progress and index % args.progress == 0:
            print(f"[real-eval] {index}/{len(trials)}", flush=True)

    valid = [row for row in rows if not row.get("skipped")]
    labels = np.asarray([row["letter_idx"] for row in valid], dtype=np.int64)
    model = load_character_model(args.character_checkpoint, device=args.device)
    gt_probs = classify(model, [row["gt_traj"] for row in valid])
    pred_probs = classify(model, [row["pred_traj"] for row in valid])

    report: dict[str, Any] = {
        "raw_root": str(root),
        "trajectory_checkpoint": str(args.trajectory_checkpoint),
        "character_checkpoint": str(args.character_checkpoint),
        "imu_source": args.imu_source,
        "input_gain": args.input_gain,
        "fs": args.fs,
        "gravity_cutoff_hz": args.gravity_cutoff,
        "trials_total": len(trials),
        "trajectory": summarize(rows),
        "character": {
            "ground_truth": accuracy_block(labels, gt_probs),
            "reconstructed": accuracy_block(labels, pred_probs),
            "per_letter_gt": per_letter_accuracy(labels, gt_probs),
            "per_letter_recon": per_letter_accuracy(labels, pred_probs),
            "confusion_gt": confusion_matrix(labels, gt_probs),
            "confusion_recon": confusion_matrix(labels, pred_probs),
        },
    }
    if args.per_condition:
        per_condition: dict[str, Any] = {}
        for condition in sorted({row["condition"] for row in valid}):
            index = [i for i, row in enumerate(valid) if row["condition"] == condition]
            per_condition[condition] = {
                "trials": len(index),
                "trajectory": summarize([valid[i] for i in index]),
                "gt_top1": accuracy_block(labels[index], gt_probs[index]),
                "recon_top1": accuracy_block(labels[index], pred_probs[index]),
            }
        report["per_condition"] = per_condition
    return report, rows, labels, gt_probs, pred_probs


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WritingRing models on the lab 200 Hz dataset")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--trajectory-checkpoint", type=Path,
                        default=REPO_ROOT / "models_trajectory_topology/b2u/paper_trajectory.pt")
    parser.add_argument("--character-checkpoint", type=Path,
                        default=REPO_ROOT / "models_character/user_both_ud/character_cnn.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_data_eval"))
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--sources", nargs="*", default=None, help="fk / yjx")
    parser.add_argument("--imu-source", choices=("highpass", "device"), default="highpass")
    parser.add_argument("--align", choices=("contrast", "motion", "manifest"), default="contrast",
                        help="contrast = maximises the motion burst inside the pen window (recommended); "
                             "motion = correlate against the pen speed; manifest = trust the per-trial delta")
    parser.add_argument("--align-search", nargs=2, type=float, default=(-0.5, 2.5),
                        metavar=("LO", "HI"), help="lag search window in seconds")
    parser.add_argument("--input-gain", type=float, default=1.0)
    parser.add_argument("--gravity-cutoff", type=float, default=1.0)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="take every n-th manifest row")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--per-condition", action="store_true")
    parser.add_argument("--progress", type=int, default=200)
    parser.add_argument("--save-trajectories", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report, rows, labels, gt_probs, pred_probs = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    columns = ["trial", "condition", "letter", "source", "skipped", "n_points", "normalized_raw",
               "normalized_aligned", "path_ratio", "frac_raw_gt_0.1"]
    with (args.output_dir / "per_trial.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    if args.save_trajectories:
        np.savez_compressed(
            args.output_dir / "trajectories.npz",
            **{f"gt_{i}": row["gt_traj"] for i, row in enumerate(rows) if not row.get("skipped")},
            **{f"pred_{i}": row["pred_traj"] for i, row in enumerate(rows) if not row.get("skipped")},
            trial_names=np.asarray([row["trial"] for row in rows if not row.get("skipped")]),
            labels=labels,
            gt_probs=gt_probs,
            pred_probs=pred_probs,
        )
    print(json.dumps({key: value for key, value in report.items()
                      if key in ("trials_total", "imu_source", "input_gain")}, ensure_ascii=False))
    print(json.dumps(report["trajectory"], ensure_ascii=False, indent=2))
    print(json.dumps(report["character"]["ground_truth"], ensure_ascii=False))
    print(json.dumps(report["character"]["reconstructed"], ensure_ascii=False))


if __name__ == "__main__":
    main()
