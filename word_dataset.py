from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .character_dataset import (
    _read_labels,
    contact_runs,
    normalize_trajectory,
    resample_trajectory,
)
from .paper_trajectory_dataset import (
    discover_trajectory_samples,
    load_sample_arrays,
    sample_is_valid,
)


WORD_ACTIONS = (2, 3, 4, 5)


@dataclass(frozen=True)
class WordSample:
    user: str
    action: int
    sample_id: str
    word: str
    group: str
    runs: tuple[tuple[int, int], ...]


def word_group(action: int) -> str:
    return "connected" if action in (2, 3) else "unconnected"


def discover_word_samples(
    clean_root: str | Path,
    raw_root: str | Path,
    actions: Sequence[int] = WORD_ACTIONS,
    min_contact_frames: int = 20,
) -> list[WordSample]:
    clean_root = Path(clean_root)
    raw_root = Path(raw_root)
    blocks = [
        block
        for block in discover_trajectory_samples(clean_root)
        if block.action in actions and sample_is_valid(block)
    ]
    found: list[WordSample] = []
    for block in blocks:
        label_path = (
            raw_root
            / block.user
            / str(block.action)
            / f"{block.sample_id}_timestamp.txt"
        )
        if not label_path.exists():
            continue
        labels = _read_labels(label_path)
        if not labels:
            continue
        _, _, mask, _, timestamp = load_sample_arrays(block, mmap=False)
        mask = np.asarray(mask)
        times = np.asarray(timestamp, dtype=np.float64)
        runs = contact_runs(mask, min_frames=8)
        if not runs:
            continue
        frames = np.clip(
            np.searchsorted(times, np.asarray([ts for ts, _ in labels], dtype=np.float64)),
            0,
            len(times) - 1,
        ).astype(int)
        for index, (label_time, raw_word) in enumerate(labels):
            if raw_word.lower() == "wrong":
                continue
            if label_time < times[0] or label_time > times[-1]:
                continue
            low = int(frames[index])
            high = int(frames[index + 1]) if index + 1 < len(labels) else len(times)
            selected = tuple(
                (start, stop) for start, stop in runs if low <= start < high
            )
            if not selected:
                continue
            contact_frames = sum(stop - start for start, stop in selected)
            if contact_frames < min_contact_frames:
                continue
            word = raw_word.lower()
            if not word:
                continue
            found.append(
                WordSample(
                    user=block.user,
                    action=block.action,
                    sample_id=block.sample_id,
                    word=word,
                    group=word_group(block.action),
                    runs=selected,
                )
            )
    return found


def deskew_trajectory(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    direction = points[-1] - points[0]
    if float(np.linalg.norm(direction)) < 1e-12:
        return points
    angle = float(np.arctan2(direction[1], direction[0]))
    cosine, sine = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return points @ rotation.T


def _runs_trajectory(
    runs: Sequence[tuple[int, int]],
    board: np.ndarray,
    resample_frames: int,
    deskew: bool = True,
) -> np.ndarray:
    board = np.asarray(board, dtype=np.float64)
    pieces = []
    for start, stop in runs:
        run = board[start:stop]
        if len(run):
            pieces.append(run - run[0])
    if not pieces:
        points = np.zeros((1, 2), dtype=np.float64)
    else:
        points = np.concatenate(pieces, axis=0)
    points = resample_trajectory(points, resample_frames)
    if deskew:
        points = deskew_trajectory(points)
    return normalize_trajectory(points)


def build_word_examples(
    samples: Sequence[WordSample],
    clean_root: str | Path,
    resample_frames: int = 128,
    deskew: bool = True,
) -> list[dict]:
    clean_root = Path(clean_root)
    cache: dict[tuple[str, int, str], np.ndarray] = {}
    examples: list[dict] = []
    for sample in samples:
        key = (sample.user, sample.action, sample.sample_id)
        if key not in cache:
            cache[key] = np.load(
                clean_root
                / sample.user
                / str(sample.action)
                / f"{sample.sample_id}_board.npy"
            )
        examples.append(
            {
                "trajectory": _runs_trajectory(
                    sample.runs, cache[key], resample_frames, deskew=deskew
                ),
                "word": sample.word,
                "group": sample.group,
                "user": sample.user,
                "action": sample.action,
                "sample_id": sample.sample_id,
                "runs": sample.runs,
                "source": "gt",
            }
        )
    return examples


def build_word_recon_examples(
    samples: Sequence[WordSample],
    run_points_by_block: dict[tuple[str, int, str], dict[tuple[int, int], np.ndarray]],
    resample_frames: int = 128,
    deskew: bool = True,
) -> list[dict]:
    examples: list[dict] = []
    for sample in samples:
        block_points = run_points_by_block.get(
            (sample.user, sample.action, sample.sample_id)
        )
        if not block_points:
            continue
        pieces = []
        for run in sample.runs:
            points = block_points.get(run)
            if points is not None and len(points):
                pieces.append(points - points[0])
        if not pieces:
            continue
        trajectory = resample_trajectory(
            np.concatenate(pieces, axis=0), resample_frames
        )
        if deskew:
            trajectory = deskew_trajectory(trajectory)
        trajectory = normalize_trajectory(trajectory)
        examples.append(
            {
                "trajectory": trajectory,
                "word": sample.word,
                "group": sample.group,
                "user": sample.user,
                "action": sample.action,
                "sample_id": sample.sample_id,
                "runs": sample.runs,
                "source": "recon",
            }
        )
    return examples
