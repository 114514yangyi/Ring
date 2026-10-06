from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .madgwick import MadgwickConfig, cross_correlation_offset, madgwick_linear_acc
from .paper_trajectory_dataset import (
    discover_trajectory_samples,
    load_sample_arrays,
    sample_is_valid,
)


DEFAULT_CLEAN_ROOT = Path(
    "/data/huyang/datasets/WritingRing/clean_data_delete_g/data"
)
DEFAULT_RAW_ROOT = Path("/data/huyang/datasets/WritingRing/data")


def _mean_axis_correlation(left: np.ndarray, right: np.ndarray) -> float:
    values = []
    for axis in range(3):
        if np.std(left[:, axis]) < 1e-9 or np.std(right[:, axis]) < 1e-9:
            continue
        values.append(float(np.corrcoef(left[:, axis], right[:, axis])[0, 1]))
    return float(np.mean(values)) if values else 0.0


def align_raw_offset(
    clean_gyro: np.ndarray, raw_gyro: np.ndarray, search: int = 4
) -> tuple[int, float]:
    clean_gyro = np.asarray(clean_gyro, dtype=np.float64)
    raw_gyro = np.asarray(raw_gyro, dtype=np.float64)
    offset = cross_correlation_offset(
        np.linalg.norm(clean_gyro, axis=1), np.linalg.norm(raw_gyro, axis=1)
    )
    count = len(clean_gyro)
    best: tuple[int, float] | None = None
    for candidate in range(offset - search, offset + search + 1):
        if candidate < 0 or candidate + count > len(raw_gyro):
            continue
        correlation = _mean_axis_correlation(
            clean_gyro, raw_gyro[candidate : candidate + count]
        )
        if best is None or correlation > best[1]:
            best = (candidate, correlation)
    if best is None:
        raise ValueError("cannot align raw block")
    return best


def build_sample_arrays(
    clean_x: np.ndarray,
    raw: np.ndarray,
    config: MadgwickConfig | None = None,
    min_correlation: float = 0.9,
) -> np.ndarray | None:
    config = config or MadgwickConfig()
    clean_x = np.asarray(clean_x)
    raw = np.asarray(raw, dtype=np.float64)
    try:
        offset, correlation = align_raw_offset(clean_x[:, 3:6], raw[:, 3:6])
    except ValueError:
        return None
    if correlation < min_correlation:
        return None
    count = len(clean_x)
    linear = madgwick_linear_acc(raw[: offset + count, :3], raw[: offset + count, 3:6], config)
    built = np.empty((count, 6), dtype=np.float32)
    built[:, :3] = linear[offset : offset + count]
    built[:, 3:] = clean_x[:, 3:6]
    return built


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the clean dataset with Madgwick gravity removal"
    )
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--beta", type=float, default=0.041)
    parser.add_argument("--min-correlation", type=float, default=0.9)
    args = parser.parse_args()

    samples = [
        sample
        for sample in discover_trajectory_samples(args.clean_root)
        if sample_is_valid(sample)
    ]
    if args.limit_samples is not None and args.limit_samples < len(samples):
        step = max(1, len(samples) // args.limit_samples)
        samples = samples[::step][: args.limit_samples]

    config = MadgwickConfig(beta=args.beta)
    built = 0
    missing_raw = 0
    failed_align = 0
    for sample in samples:
        raw_path = (
            args.raw_root / sample.user / str(sample.action) / f"{sample.sample_id}_ring_0.bin"
        )
        if not raw_path.exists():
            missing_raw += 1
            continue
        raw = np.fromfile(raw_path, dtype=np.float64).reshape(-1, 7)
        x, board, mask, y, timestamp = load_sample_arrays(sample, mmap=False)
        rebuilt = build_sample_arrays(
            np.asarray(x), raw, config, min_correlation=args.min_correlation
        )
        if rebuilt is None:
            failed_align += 1
            continue
        output_dir = args.output_root / sample.user / str(sample.action)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"{sample.sample_id}_x.npy", rebuilt)
        np.save(output_dir / f"{sample.sample_id}_board.npy", np.asarray(board))
        np.save(output_dir / f"{sample.sample_id}_mask.npy", np.asarray(mask))
        np.save(output_dir / f"{sample.sample_id}_y.npy", np.asarray(y))
        np.save(output_dir / f"{sample.sample_id}_timestamp.npy", np.asarray(timestamp))
        built += 1

    summary = {
        "output_root": str(args.output_root),
        "candidates": len(samples),
        "built": built,
        "missing_raw": missing_raw,
        "failed_align": failed_align,
        "beta": args.beta,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
