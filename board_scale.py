from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _percentile_span(values: np.ndarray, q: float) -> float:
    low = float(np.percentile(values, q))
    high = float(np.percentile(values, 100.0 - q))
    return high - low


def fit_scale_from_ranges(
    raw_xy: np.ndarray, clean_xy: np.ndarray, q: float = 1.0
) -> tuple[float, float]:
    raw = np.asarray(raw_xy, dtype=np.float64)
    clean = np.asarray(clean_xy, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("raw_xy must have shape [points, 2]")
    if clean.ndim != 2 or clean.shape[1] != 2:
        raise ValueError("clean_xy must have shape [points, 2]")
    scales: list[float] = []
    for axis in range(2):
        raw_span = _percentile_span(raw[:, axis], q)
        clean_span = _percentile_span(clean[:, axis], q)
        scales.append(clean_span / raw_span if raw_span > 1e-9 else float("nan"))
    return scales[0], scales[1]


def fetch_raw_contacts(
    chunk_paths: Sequence[Path], core_path: str | Path | None = None
) -> np.ndarray:
    import sys

    if core_path is not None:
        resolved = str(Path(core_path))
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    import compress_pickle

    points: list[tuple[float, float]] = []
    for path in chunk_paths:
        frames = compress_pickle.load(str(path))
        for frame in frames:
            for contact in frame.contacts:
                points.append((float(contact.x), float(contact.y)))
    return np.asarray(points, dtype=np.float64)


DEFAULT_CORE_PATH = Path("/data/huyang/datasets/WritingRing/hf_repo")


def calibrate_sample(
    raw_block_dir: str | Path,
    clean_sample: str | Path,
    raw_chunk_glob: str = "*_board_*.gz",
    q: float = 1.0,
    core_path: str | Path | None = DEFAULT_CORE_PATH,
) -> dict[str, Any]:
    clean_prefix = Path(clean_sample)
    try:
        raw_xy = fetch_raw_contacts(
            sorted(Path(raw_block_dir).glob(raw_chunk_glob)), core_path=core_path
        )
    except ImportError as error:
        return {
            "ok": False,
            "reason": f"compress_pickle is not installed: {error}",
            "hint": "pip install compress_pickle, or keep the default board_mm_scale",
        }
    if len(raw_xy) == 0:
        return {"ok": False, "reason": "no raw contacts found"}

    board = np.load(clean_prefix.parent / f"{clean_prefix.name}_board.npy")
    mask = np.load(clean_prefix.parent / f"{clean_prefix.name}_mask.npy")
    clean_xy = np.asarray(board[mask > 0], dtype=np.float64)
    if len(clean_xy) == 0:
        return {"ok": False, "reason": "no clean contact points found"}

    scale_x, scale_y = fit_scale_from_ranges(raw_xy, clean_xy, q=q)
    result: dict[str, Any] = {
        "ok": True,
        "raw_points": int(len(raw_xy)),
        "clean_points": int(len(clean_xy)),
        "raw_min": raw_xy.min(axis=0).tolist(),
        "raw_max": raw_xy.max(axis=0).tolist(),
        "clean_min": clean_xy.min(axis=0).tolist(),
        "clean_max": clean_xy.max(axis=0).tolist(),
        "clean_units_per_raw_unit": [scale_x, scale_y],
    }
    if max(raw_xy.max(axis=0).tolist()) > 1.5:
        mm_scale = [
            1.0 / scale_x if np.isfinite(scale_x) and scale_x > 0 else float("nan"),
            1.0 / scale_y if np.isfinite(scale_y) and scale_y > 0 else float("nan"),
        ]
        result["suggested_board_mm_scale"] = mm_scale
        result["interpretation"] = (
            "raw contacts look like millimetres; suggested_board_mm_scale converts "
            "clean board units to mm"
        )
    else:
        result["suggested_board_mm_scale"] = None
        result["interpretation"] = (
            "raw contacts look normalised; keep the Sensel Morph size assumption "
            f"(default {[240.0, 169.5]})"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate clean board coordinates against raw Sensel contacts"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/data/huyang/datasets/WritingRing/data"),
    )
    parser.add_argument(
        "--clean-sample",
        type=Path,
        required=True,
        help="clean sample prefix, e.g. .../clean_data_delete_g/data/user_0/1/0",
    )
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument(
        "--core-path",
        type=Path,
        default=DEFAULT_CORE_PATH,
        help="directory that provides the 'core' package used to unpickle raw board chunks",
    )
    args = parser.parse_args()

    clean_prefix = args.clean_sample
    action = clean_prefix.parent.name
    user = clean_prefix.parent.parent.name
    raw_block_dir = args.raw_root / user / action
    result = calibrate_sample(
        raw_block_dir, clean_prefix, q=args.q, core_path=args.core_path
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
