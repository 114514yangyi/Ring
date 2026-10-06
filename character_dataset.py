from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .paper_trajectory_dataset import (
    discover_trajectory_samples,
    load_sample_arrays,
    sample_is_valid,
)


@dataclass(frozen=True)
class CharacterSample:
    user: str
    action: int
    sample_id: str
    label: int
    char: str
    runs: tuple[tuple[int, int], ...]


def contact_runs(mask: np.ndarray, min_frames: int = 8) -> list[tuple[int, int]]:
    values = (np.asarray(mask) > 0).astype(np.int8)
    padded = np.concatenate([[0], values, [0]])
    changes = np.flatnonzero(np.diff(padded))
    return [
        (int(start), int(stop))
        for start, stop in zip(changes[::2], changes[1::2])
        if stop - start >= min_frames
    ]


def _read_labels(path: Path) -> list[tuple[int, str]]:
    tokens = path.read_text().split()
    return [
        (int(float(tokens[index])), tokens[index + 1])
        for index in range(0, len(tokens), 2)
    ]


def discover_character_samples(
    clean_root: str | Path,
    raw_root: str | Path,
    actions: Sequence[int] = (0, 1),
    min_contact_frames: int = 8,
) -> list[CharacterSample]:
    clean_root = Path(clean_root)
    raw_root = Path(raw_root)
    blocks = [
        sample
        for sample in discover_trajectory_samples(clean_root)
        if sample.action in actions and sample_is_valid(sample)
    ]
    found: list[CharacterSample] = []
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
        runs = contact_runs(mask, min_frames=min_contact_frames)
        if not runs:
            continue
        frames = np.clip(
            np.searchsorted(times, np.asarray([ts for ts, _ in labels], dtype=np.float64)),
            0,
            len(times) - 1,
        ).astype(int)
        for index, (label_time, char) in enumerate(labels):
            if char == "wrong":
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
            letter = char.upper()
            if len(letter) != 1 or not ("A" <= letter <= "Z"):
                continue
            found.append(
                CharacterSample(
                    user=block.user,
                    action=block.action,
                    sample_id=block.sample_id,
                    label=ord(letter) - ord("A"),
                    char=letter,
                    runs=selected,
                )
            )
    return found


def resample_trajectory(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if count < 1:
        raise ValueError("count must be positive")
    if len(points) == 0:
        return np.zeros((count, 2), dtype=np.float64)
    if len(points) == 1:
        return np.tile(points[0], (count, 1))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    if cumulative[-1] < 1e-12:
        return np.tile(points[0], (count, 1))
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.stack(
        [
            np.interp(targets, cumulative, points[:, 0]),
            np.interp(targets, cumulative, points[:, 1]),
        ],
        axis=1,
    )


def normalize_trajectory(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points
    centered = points - points[0]
    span = centered.max(axis=0) - centered.min(axis=0)
    scale = float(np.linalg.norm(span))
    if scale < 1e-12:
        return centered
    return centered / scale


def extract_gt_trajectory(
    sample: CharacterSample, board: np.ndarray, resample_frames: int = 64
) -> np.ndarray:
    board = np.asarray(board, dtype=np.float64)
    pieces = []
    for start, stop in sample.runs:
        run = board[start:stop]
        if len(run):
            pieces.append(run - run[0])
    if not pieces:
        points = np.zeros((1, 2), dtype=np.float64)
    else:
        points = np.concatenate(pieces, axis=0)
    return normalize_trajectory(resample_trajectory(points, resample_frames))


def build_gt_examples(
    samples: Sequence[CharacterSample],
    clean_root: str | Path,
    resample_frames: int = 64,
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
        trajectory = extract_gt_trajectory(
            sample, cache[key], resample_frames=resample_frames
        )
        examples.append(
            {
                "trajectory": trajectory,
                "label": sample.label,
                "char": sample.char,
                "user": sample.user,
                "action": sample.action,
                "sample_id": sample.sample_id,
                "runs": sample.runs,
            }
        )
    return examples
