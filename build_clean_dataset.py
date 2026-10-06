"""Rebuild the WritingRing ``clean`` dataset from the raw recordings.

The official cleaned release (``clean_data_delete_g``) is a resampled, cropped
and gravity-removed version of the raw ``data/`` recordings.  The mapping below
was reverse-engineered sample by sample (evidence in ``.trae/documents/NOTES.md``):

1. crop window = raw ring rows whose timestamp lies in
   ``[board_start + 0.6 s, board_end - 1.0 s]`` (``--head-trim`` / ``--tail-trim``);
   this reproduces the official sample length ``T`` to within 0-7 frames;
2. ``timestamp`` = uniform grid ``t0 + k * dt`` with ``dt = 4977.75 us``
   (~200.894 Hz; the official grid spacing is exactly this constant);
3. ``x[:, 3:6]`` (gyro) = the cropped raw ring rows verbatim -- bit-identical to
   the official clean gyro at the aligned row offset;
4. ``x[:, 0:3]`` (lin_acc) = raw acceleration with gravity removed.  The exact
   official operator is *not* recoverable (the paper claims Madgwick
   beta=0.041, but the published clean acc does not match Madgwick); the closest
   simple model is a ~1 Hz high-pass (``--gravity butter``), per-axis
   correlation 0.89-0.97 with the official release;
5. ``board`` = raw Sensel contact #0 position, linearly resampled onto the grid
   (official match ~5e-5 in normalised units);
6. ``mask`` = 1 where the resampled point is bracketed by two contact frames
   (official agreement 98.8%, IoU 0.975);
7. ``y`` = raw frame-rate contact displacement, resampled onto the grid and
   scaled by the grid step (official correlation 0.98-0.99).

Residual uncertainty: the official grid's absolute anchor sits 0-0.15 s away
from the raw ring timestamps (the ring and the Sensel are separate devices and
their clocks are not in the released files).  The default anchor here is the
first cropped ring row; ``--anchor-shift-us`` shifts the whole grid, and
``--validate-against`` reports the per-sample shift that best matches the
official release together with the fidelity metrics at that shift.

Examples
--------
Quick check with validation on a few samples::

    python build_clean_dataset.py \
        --output-root /tmp/clean_rebuilt \
        --users user_0 --actions 0 --samples 1,2,3 \
        --validate-against /data/huyang/datasets/WritingRing/clean_data_delete_g/data

Full rebuild (566 samples, ~12 GB of raw board files -> several minutes)::

    python build_clean_dataset.py \
        --output-root /data/huyang/datasets/WritingRing/clean_data_rebuilt/data \
        --workers 4
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_RAW_ROOT = Path("/data/huyang/datasets/WritingRing/data")
DEFAULT_CLEAN_ROOT = Path("/data/huyang/datasets/WritingRing/clean_data_delete_g/data")

#: uniform grid step of the official clean release (microseconds)
DEFAULT_DT_US = 4977.75
#: crop trims relative to the board recording (seconds)
DEFAULT_HEAD_TRIM_S = 0.6
DEFAULT_TAIL_TRIM_S = 1.0

ACTIONS = (0, 1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# raw readers
# ---------------------------------------------------------------------------


class _CompatObject:
    """Permissive stand-in for the authors' pickle classes."""

    def __new__(cls, *args, **kwargs):  # noqa: D102 - pickle stub
        return object.__new__(cls)

    def __init__(self, *args, **kwargs) -> None:  # noqa: D102 - pickle stub
        pass

    def __setstate__(self, state) -> None:  # noqa: D102 - pickle stub
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:  # pragma: no cover - defensive
            self.__dict__["_state"] = state


class _CompatUnpickler(pickle.Unpickler):
    _CLASSES = {
        ("core.sensel_lib.frame_data", "FrameData"): _CompatObject,
        ("core.sensel_lib.frame_data", "ContactData"): _CompatObject,
        ("frame_data", "FrameData"): _CompatObject,
        ("frame_data", "ContactData"): _CompatObject,
    }

    def find_class(self, module, name):  # noqa: D102
        target = self._CLASSES.get((module, name))
        if target is None:
            return super().find_class(module, name)
        return target


def read_ring(path: Path) -> np.ndarray:
    """Return the raw ring stream as ``(frames, 7)`` float64 (acc3, gyro3, t_us)."""

    data = np.fromfile(path, dtype=np.float64)
    if data.size % 7:
        raise ValueError(f"{path}: size {data.size} is not divisible by 7")
    return data.reshape(-1, 7)


def read_board(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load Sensel board frames.

    Returns ``(timestamps_us, xy)`` sorted by time, where ``xy`` holds contact
    #0 (NaN when a frame has no contact).
    """

    stamps: list[float] = []
    points: list[tuple[float, float]] = []
    for path in paths:
        with gzip.open(path, "rb") as handle:
            frames = _CompatUnpickler(handle).load()
        for frame in frames:
            contacts = getattr(frame, "contacts", None) or []
            stamps.append(float(frame.timestamp))
            if contacts:
                points.append((float(contacts[0].x), float(contacts[0].y)))
            else:
                points.append((float("nan"), float("nan")))
    stamps_arr = np.asarray(stamps, dtype=np.float64)
    xy = np.asarray(points, dtype=np.float64)
    order = np.argsort(stamps_arr, kind="stable")
    return stamps_arr[order], xy[order]


def discover_raw_samples(raw_root: Path, users: list[str] | None, actions: tuple[int, ...]):
    """Yield ``(user, action, sample_id, ring_path, board_paths)`` for raw samples."""

    for user_dir in sorted(raw_root.iterdir()):
        if not user_dir.is_dir() or (users and user_dir.name not in users):
            continue
        for action in actions:
            action_dir = user_dir / str(action)
            if not action_dir.is_dir():
                continue
            for ring_path in sorted(action_dir.glob("*_ring_0.bin")):
                sample_id = ring_path.name.split("_", 1)[0]
                boards = sorted(
                    action_dir.glob(f"{sample_id}_board_*.gz"),
                    key=lambda p: int(p.name.rsplit("_", 1)[1].split(".")[0]),
                )
                if boards:
                    yield user_dir.name, action, sample_id, ring_path, boards


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, window)
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(values)
    for axis in range(values.shape[1]):
        out[:, axis] = np.convolve(padded[:, axis], kernel, "valid")[: len(values)]
    return out


def remove_gravity(
    acc: np.ndarray,
    method: str = "butter",
    cutoff_hz: float = 1.0,
    window_s: float = 0.8,
    sample_rate: float = 1e6 / DEFAULT_DT_US,
) -> np.ndarray:
    """Approximate the official gravity removal on the acceleration channels."""

    acc = np.asarray(acc, dtype=np.float64)
    if method == "none":
        return acc.copy()
    if method == "global":
        return acc - acc.mean(axis=0, keepdims=True)
    if method == "movavg":
        return acc - _moving_average(acc, int(round(window_s * sample_rate)))
    if method == "butter":
        try:
            from scipy.signal import butter, filtfilt
        except ImportError:  # pragma: no cover - scipy is optional
            return acc - _moving_average(acc, int(round(window_s * sample_rate)))
        b, a = butter(2, cutoff_hz / (sample_rate / 2.0), btype="high")
        return filtfilt(b, a, acc, axis=0)
    raise ValueError(f"unknown gravity method: {method}")


@dataclass
class BoardSample:
    board: np.ndarray
    mask: np.ndarray
    y: np.ndarray


def resample_board(
    board_ts: np.ndarray,
    board_xy: np.ndarray,
    grid: np.ndarray,
    dt_us: float = DEFAULT_DT_US,
    mask_mode: str = "and",
) -> BoardSample:
    """Resample the raw touchpad contact path onto a uniform time grid."""

    upper = np.clip(np.searchsorted(board_ts, grid), 1, len(board_ts) - 1)
    lower = upper - 1
    span = np.maximum(board_ts[upper] - board_ts[lower], 1e-9)
    weight = np.clip((grid - board_ts[lower]) / span, 0.0, 1.0)

    p_lo, p_hi = board_xy[lower], board_xy[upper]
    lo_ok = ~np.isnan(p_lo[:, 0])
    hi_ok = ~np.isnan(p_hi[:, 0])

    if mask_mode == "and":
        valid = lo_ok & hi_ok
    elif mask_mode == "or":
        valid = lo_ok | hi_ok
    elif mask_mode == "next":
        valid = hi_ok
    elif mask_mode == "prev":
        valid = lo_ok
    else:
        raise ValueError(f"unknown mask mode: {mask_mode}")

    interp = p_lo * (1.0 - weight[:, None]) + p_hi * weight[:, None]
    board = np.nan_to_num(interp, nan=0.0).astype(np.float32)
    board[~valid] = 0.0

    filled = board_xy.copy()
    frame_index = np.arange(len(filled), dtype=np.float64)
    for axis in range(2):
        column = filled[:, axis]
        good = ~np.isnan(column)
        if good.sum() < 2:
            filled[:, axis] = 0.0
        else:
            filled[:, axis] = np.interp(frame_index, frame_index[good], column[good])
    span_us = np.maximum(np.diff(board_ts), 1e-9)
    velocity = np.empty_like(filled)
    velocity[1:] = (filled[1:] - filled[:-1]) / (span_us[:, None] / 1e6)
    velocity[0] = velocity[1] if len(velocity) > 1 else 0.0
    y = np.empty((len(grid), 2), dtype=np.float64)
    for axis in range(2):
        y[:, axis] = np.interp(grid, board_ts, velocity[:, axis]) * (dt_us / 1e6)
    y[~valid] = 0.0
    y = y.astype(np.float32)

    return BoardSample(board=board, mask=valid.astype(np.int64), y=y)


@dataclass
class SampleArrays:
    x: np.ndarray
    board: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    timestamp: np.ndarray


def build_sample(
    ring: np.ndarray,
    board_ts: np.ndarray,
    board_xy: np.ndarray,
    dt_us: float = DEFAULT_DT_US,
    head_trim_s: float = DEFAULT_HEAD_TRIM_S,
    tail_trim_s: float = DEFAULT_TAIL_TRIM_S,
    gravity: str = "butter",
    gravity_cutoff_hz: float = 1.0,
    gravity_window_s: float = 0.8,
    mask_mode: str = "and",
    anchor: str = "ring",
    anchor_shift_us: float = 0.0,
) -> SampleArrays:
    """Build one clean sample from one raw sample pair."""

    if len(board_ts) < 3:
        raise ValueError("board recording is empty")

    lo = board_ts[0] + head_trim_s * 1e6
    hi = board_ts[-1] - tail_trim_s * 1e6
    ring_ts = ring[:, 6]
    keep = (ring_ts >= lo) & (ring_ts <= hi)
    track = ring[keep]
    if len(track) < 8:
        raise ValueError("ring crop is too short")

    count = len(track)
    if anchor == "ring":
        anchor_us = track[0, 6]
    elif anchor == "board":
        anchor_us = lo
    else:
        raise ValueError(f"unknown anchor: {anchor}")
    clean_ts = (anchor_us + anchor_shift_us) + np.arange(count, dtype=np.float64) * dt_us

    resampled = resample_board(board_ts, board_xy, clean_ts, dt_us=dt_us, mask_mode=mask_mode)
    acc = remove_gravity(
        track[:, :3],
        method=gravity,
        cutoff_hz=gravity_cutoff_hz,
        window_s=gravity_window_s,
    )
    x = np.concatenate([acc, track[:, 3:6]], axis=1).astype(np.float32)

    return SampleArrays(
        x=x,
        board=resampled.board,
        mask=resampled.mask,
        y=resampled.y,
        timestamp=clean_ts.astype(np.float64),
    )


# ---------------------------------------------------------------------------
# validation against the official release
# ---------------------------------------------------------------------------


def _align_offset(built_gyro: np.ndarray, ref_gyro: np.ndarray, search: int = 300) -> int:
    """Row offset ``o`` such that ``built[i] ~= ref[i + o]`` (best gyro match)."""

    best = (np.inf, 0)
    for offset in range(-search, search + 1):
        lo = max(0, -offset)
        hi = min(len(built_gyro), len(ref_gyro) - offset)
        if hi - lo < 64:
            continue
        err = np.abs(built_gyro[lo:hi] - ref_gyro[lo + offset : hi + offset]).mean()
        if err < best[0]:
            best = (err, offset)
    return best[1]


def _board_distance(board_built, board_ref, both) -> float:
    if not both.any():
        return float("nan")
    return float(
        np.linalg.norm(board_built[both] - board_ref[both], axis=1).mean()
    )


def fit_anchor_shift(
    built: SampleArrays,
    ref: dict,
    board_ts: np.ndarray,
    board_xy: np.ndarray,
    dt_us: float,
    mask_mode: str,
    search_s: float = 0.35,
    coarse_s: float = 0.01,
    fine_s: float = 0.0005,
) -> dict:
    """Fit the grid anchor shift (us) that best matches the official board/gyro."""

    offset = _align_offset(built.x[:, 3:6], ref["x"][:, 3:6].astype(np.float64))
    lo = max(0, -offset)
    hi = min(len(built.x), len(ref["x"]) - offset)
    if hi - lo < 64:
        return {"offset_rows": int(offset), "status": "no-overlap"}

    def score(shift_s: float) -> tuple[float, float, float]:
        grid = built.timestamp + shift_s * 1e6
        resampled = resample_board(board_ts, board_xy, grid, dt_us=dt_us, mask_mode=mask_mode)
        mask_built = resampled.mask[lo:hi] == 1
        mask_ref = ref["mask"][lo + offset : hi + offset] == 1
        both = mask_built & mask_ref
        distance = _board_distance(resampled.board[lo:hi], ref["board"][lo + offset : hi + offset], both)
        if both.any():
            y_corr = float(
                np.mean(
                    [
                        np.corrcoef(
                            resampled.y[lo:hi][both, axis].astype(np.float64),
                            ref["y"][lo + offset : hi + offset][both, axis].astype(np.float64),
                        )[0, 1]
                        for axis in range(2)
                    ]
                )
            )
        else:
            y_corr = float("nan")
        return distance, y_corr, float(both.mean())

    candidates = np.arange(-search_s, search_s + 1e-9, coarse_s)
    best = None
    for shift in candidates:
        distance, y_corr, _ = score(float(shift))
        if np.isnan(distance):
            continue
        if best is None or distance < best[0]:
            best = (distance, float(shift), y_corr)
    if best is None:
        return {"offset_rows": int(offset), "status": "no-overlap"}

    centre = best[1]
    for shift in np.arange(centre - coarse_s, centre + coarse_s + 1e-9, fine_s):
        distance, y_corr, _ = score(float(shift))
        if np.isnan(distance):
            continue
        if distance < best[0]:
            best = (distance, float(shift), y_corr)

    distance, shift, _ = best
    grid = built.timestamp + shift * 1e6
    resampled = resample_board(board_ts, board_xy, grid, dt_us=dt_us, mask_mode=mask_mode)
    mask_built = resampled.mask[lo:hi] == 1
    mask_ref = ref["mask"][lo + offset : hi + offset] == 1
    both = mask_built & mask_ref
    tp = int((mask_built & mask_ref).sum())
    fp = int((mask_built & ~mask_ref).sum())
    fn = int((~mask_built & mask_ref).sum())
    y_corr = y_rel = float("nan")
    if both.any():
        y_corr = float(
            np.mean(
                [
                    np.corrcoef(
                        resampled.y[lo:hi][both, axis].astype(np.float64),
                        ref["y"][lo + offset : hi + offset][both, axis].astype(np.float64),
                    )[0, 1]
                    for axis in range(2)
                ]
            )
        )
        y_rel = float(
            np.mean(np.abs(resampled.y[lo:hi][both] - ref["y"][lo + offset : hi + offset][both]))
            / max(np.mean(np.abs(ref["y"][lo + offset : hi + offset][both])), 1e-12)
        )
    acc_corr = float(
        np.mean(
            [
                np.corrcoef(
                    built.x[lo:hi, axis].astype(np.float64),
                    ref["x"][lo + offset : hi + offset, axis].astype(np.float64),
                )[0, 1]
                for axis in range(3)
            ]
        )
    )
    return {
        "status": "ok",
        "offset_rows": int(offset),
        "n_built": int(len(built.x)),
        "n_official": int(len(ref["x"])),
        "anchor_shift_us": float(shift * 1e6),
        "gyro_max_abs_diff": float(
            np.abs(built.x[lo:hi, 3:6] - ref["x"][lo + offset : hi + offset, 3:6].astype(np.float64)).max()
        ),
        "acc_corr": acc_corr,
        "board_mean_dist": distance,
        "mask_agreement": float((mask_built == mask_ref).mean()),
        "mask_iou": float(tp / max(tp + fp + fn, 1)),
        "y_corr": y_corr,
        "y_rel_err": y_rel,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


@dataclass
class BuildConfig:
    raw_root: str
    output_root: str
    dt_us: float
    head_trim_s: float
    tail_trim_s: float
    gravity: str
    gravity_cutoff_hz: float
    gravity_window_s: float
    mask_mode: str
    anchor: str
    anchor_shift_us: float


def _worker(job):
    config, user, action, sample_id, ring_path, board_paths, ref_root = job
    try:
        ring = read_ring(Path(ring_path))
        board_ts, board_xy = read_board([Path(p) for p in board_paths])
        arrays = build_sample(
            ring,
            board_ts,
            board_xy,
            dt_us=config.dt_us,
            head_trim_s=config.head_trim_s,
            tail_trim_s=config.tail_trim_s,
            gravity=config.gravity,
            gravity_cutoff_hz=config.gravity_cutoff_hz,
            gravity_window_s=config.gravity_window_s,
            mask_mode=config.mask_mode,
            anchor=config.anchor,
            anchor_shift_us=config.anchor_shift_us,
        )
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return user, action, sample_id, {"built": False, "error": str(exc)}

    out_dir = Path(config.output_root) / user / str(action)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("x", "board", "mask", "y", "timestamp"):
        np.save(out_dir / f"{sample_id}_{name}.npy", getattr(arrays, name))

    report = {"built": True, "n": int(len(arrays.x))}
    if ref_root is not None:
        ref_dir = Path(ref_root) / user / str(action)
        ref_path = ref_dir / f"{sample_id}_x.npy"
        if ref_path.exists():
            ref = {
                name: np.load(ref_dir / f"{sample_id}_{name}.npy")
                for name in ("x", "board", "mask", "y", "timestamp")
            }
            report.update(
                fit_anchor_shift(
                    arrays,
                    ref,
                    board_ts,
                    board_xy,
                    config.dt_us,
                    config.mask_mode,
                )
            )
    return user, action, sample_id, report


def _fmt(report: dict) -> str:
    if not report.get("built"):
        return f"FAILED: {report.get('error', '')}"
    text = f"n={report['n']}"
    if "anchor_shift_us" in report:
        text += (
            f" shift={report['anchor_shift_us']/1e3:+.1f}ms"
            f" gyroΔ={report['gyro_max_abs_diff']:.1e}"
            f" acc_r={report['acc_corr']:.3f}"
            f" board_d={report['board_mean_dist']:.2e}"
            f" mask={report['mask_agreement']:.3f}"
            f" y_r={report['y_corr']:.3f}"
        )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the WritingRing clean dataset from raw recordings"
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--users", type=str, default=None, help="comma separated, e.g. user_0")
    parser.add_argument("--actions", type=str, default=None, help="comma separated, e.g. 0,1")
    parser.add_argument("--samples", type=str, default=None, help="comma separated sample ids")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--dt-us", type=float, default=DEFAULT_DT_US)
    parser.add_argument("--head-trim", type=float, default=DEFAULT_HEAD_TRIM_S)
    parser.add_argument("--tail-trim", type=float, default=DEFAULT_TAIL_TRIM_S)
    parser.add_argument(
        "--gravity",
        choices=("butter", "movavg", "global", "none"),
        default="butter",
        help="gravity-removal method for the acc channels (default: butter)",
    )
    parser.add_argument("--gravity-cutoff", type=float, default=1.0, help="Hz, for --gravity butter")
    parser.add_argument("--gravity-window", type=float, default=0.8, help="s, for --gravity movavg")
    parser.add_argument("--mask-mode", choices=("and", "or", "prev", "next"), default="and")
    parser.add_argument("--anchor", choices=("ring", "board"), default="ring")
    parser.add_argument("--anchor-shift-us", type=float, default=0.0)
    parser.add_argument("--validate-against", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    users = [u.strip() for u in args.users.split(",")] if args.users else None
    actions = tuple(int(a) for a in args.actions.split(",")) if args.actions else ACTIONS
    sample_filter = set(s.strip() for s in args.samples.split(",")) if args.samples else None

    config = BuildConfig(
        raw_root=str(args.raw_root),
        output_root=str(args.output_root),
        dt_us=args.dt_us,
        head_trim_s=args.head_trim,
        tail_trim_s=args.tail_trim,
        gravity=args.gravity,
        gravity_cutoff_hz=args.gravity_cutoff,
        gravity_window_s=args.gravity_window,
        mask_mode=args.mask_mode,
        anchor=args.anchor,
        anchor_shift_us=args.anchor_shift_us,
    )

    samples = []
    for user, action, sample_id, ring_path, boards in discover_raw_samples(
        args.raw_root, users, actions
    ):
        if sample_filter is not None and sample_id not in sample_filter:
            continue
        samples.append((user, action, sample_id, ring_path, boards))
    if args.limit_samples is not None:
        samples = samples[: args.limit_samples]

    print(f"[build-clean] {len(samples)} raw samples -> {args.output_root}")
    if args.dry_run:
        for user, action, sample_id, _, _ in samples[:20]:
            print(f"  {user}/{action}/{sample_id}")
        return

    results = []
    ref_root = str(args.validate_against) if args.validate_against else None
    jobs = [
        (config, user, action, sample_id, str(ring_path), [str(p) for p in boards], ref_root)
        for user, action, sample_id, ring_path, boards in samples
    ]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                user, action, sample_id, report = future.result()
                results.append((user, action, sample_id, report))
                print(f"  [{done}/{len(jobs)}] {user}/{action}/{sample_id} {_fmt(report)}")
    else:
        for done, job in enumerate(jobs, 1):
            user, action, sample_id, report = _worker(job)
            results.append((user, action, sample_id, report))
            print(f"  [{done}/{len(jobs)}] {user}/{action}/{sample_id} {_fmt(report)}")

    summary = _summarise(results, config)
    report_path = Path(args.output_root) / "clean_build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))
    print(f"[build-clean] report -> {report_path}")


def _summarise(results: list, config: BuildConfig) -> dict:
    built = [r for *_, r in results if r.get("built")]
    validated = [r for r in built if r.get("status") == "ok"]
    summary: dict = {
        "config": asdict(config),
        "samples_total": len(results),
        "samples_built": len(built),
        "samples_failed": len(results) - len(built),
        "samples_validated": len(validated),
        "samples": [{"user": u, "action": a, "sample": s, **r} for u, a, s, r in results],
    }
    if validated:
        keys = (
            "anchor_shift_us",
            "gyro_max_abs_diff",
            "acc_corr",
            "board_mean_dist",
            "mask_agreement",
            "mask_iou",
            "y_corr",
            "y_rel_err",
        )
        summary["validation_mean"] = {
            key: float(np.nanmean([r[key] for r in validated])) for key in keys
        }
        summary["n_abs_diff_mean"] = float(
            np.nanmean([r["n_built"] - r["n_official"] for r in validated])
        )
    return summary


if __name__ == "__main__":
    main()
