from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .paper_dataset import user_number


@dataclass(frozen=True)
class TrajectorySample:
    user: str
    action: int
    sample_id: str
    x_path: Path
    board_path: Path
    mask_path: Path
    y_path: Path
    timestamp_path: Path


@dataclass(frozen=True)
class SegmentRef:
    sample_index: int
    start: int
    length: int


def _action_sort_key(path: Path) -> int:
    return int(path.name) if path.name.isdigit() else 999


def _sample_sort_key(path: Path) -> int:
    return int(path.stem.split("_")[0])


def discover_trajectory_samples(root: str | Path) -> list[TrajectorySample]:
    root = Path(root)
    samples: list[TrajectorySample] = []
    for user_dir in sorted(root.glob("user_*"), key=lambda path: user_number(path.name)):
        if not user_dir.is_dir():
            continue
        for action_dir in sorted(user_dir.iterdir(), key=_action_sort_key):
            if not action_dir.is_dir() or not action_dir.name.isdigit():
                continue
            action = int(action_dir.name)
            for x_path in sorted(action_dir.glob("*_x.npy"), key=_sample_sort_key):
                sample_id = x_path.stem[:-2]
                board_path = action_dir / f"{sample_id}_board.npy"
                mask_path = action_dir / f"{sample_id}_mask.npy"
                y_path = action_dir / f"{sample_id}_y.npy"
                timestamp_path = action_dir / f"{sample_id}_timestamp.npy"
                if not all(
                    path.exists()
                    for path in (board_path, mask_path, y_path, timestamp_path)
                ):
                    continue
                samples.append(
                    TrajectorySample(
                        user=user_dir.name,
                        action=action,
                        sample_id=sample_id,
                        x_path=x_path,
                        board_path=board_path,
                        mask_path=mask_path,
                        y_path=y_path,
                        timestamp_path=timestamp_path,
                    )
                )
    return samples


def sample_length(sample: TrajectorySample) -> int:
    return int(np.load(sample.x_path, mmap_mode="r").shape[0])


def load_sample_arrays(
    sample: TrajectorySample, mmap: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = "r" if mmap else None
    x = np.load(sample.x_path, mmap_mode=mode)
    board = np.load(sample.board_path, mmap_mode=mode)
    mask = np.load(sample.mask_path, mmap_mode=mode)
    y = np.load(sample.y_path, mmap_mode=mode)
    timestamp = np.load(sample.timestamp_path, mmap_mode=mode)
    lengths = {len(x), len(board), len(mask), len(y)}
    if len(lengths) != 1:
        raise ValueError(
            f"inconsistent sample lengths for {sample.user}/{sample.action}/"
            f"{sample.sample_id}: x={len(x)} board={len(board)} mask={len(mask)} "
            f"y={len(y)}"
        )
    return x, board, mask, y, timestamp


def sample_is_valid(sample: TrajectorySample) -> bool:
    try:
        arrays = load_sample_arrays(sample)
    except (OSError, ValueError):
        return False
    return len({len(array) for array in arrays}) == 1


def build_segment_refs(
    samples: Sequence[TrajectorySample],
    segment_frames: int,
    strategy: str = "pad_trim",
    min_chunk: int = 1000,
) -> list[SegmentRef]:
    if segment_frames < 1:
        raise ValueError("segment_frames must be positive")
    refs: list[SegmentRef] = []
    for sample_index, sample in enumerate(samples):
        if not sample_is_valid(sample):
            continue
        length = sample_length(sample)
        if strategy == "pad_trim":
            if length <= segment_frames:
                refs.append(SegmentRef(sample_index, 0, length))
            else:
                start = (length - segment_frames) // 2
                refs.append(SegmentRef(sample_index, start, segment_frames))
        elif strategy == "chunks":
            for start in range(0, length, segment_frames):
                chunk = min(segment_frames, length - start)
                if chunk < min_chunk:
                    break
                refs.append(SegmentRef(sample_index, start, chunk))
        else:
            raise ValueError(f"unknown segment strategy: {strategy}")
    return refs


def materialize_segment(
    sample: TrajectorySample,
    ref: SegmentRef,
    segment_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, board, mask, y, _ = load_sample_arrays(sample)
    stop = ref.start + ref.length
    out_x = np.zeros((segment_frames, 6), dtype=np.float32)
    out_y = np.zeros((segment_frames, 2), dtype=np.float32)
    out_mask = np.zeros(segment_frames, dtype=np.float32)
    out_board = np.zeros((segment_frames, 2), dtype=np.float32)
    pad = segment_frames - ref.length
    out_x[pad:] = np.asarray(x[ref.start:stop], dtype=np.float32)
    out_y[pad:] = np.asarray(y[ref.start:stop], dtype=np.float32)
    out_mask[pad:] = np.asarray(mask[ref.start:stop], dtype=np.float32)
    out_board[pad:] = np.asarray(board[ref.start:stop], dtype=np.float32)
    return out_x, out_y, out_mask, out_board


def compute_normalization(
    samples: Sequence[TrajectorySample],
    refs: Sequence[SegmentRef] | None = None,
    segment_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    sum_values = np.zeros(6, dtype=np.float64)
    sum_squares = np.zeros(6, dtype=np.float64)

    def accumulate(array: np.ndarray) -> None:
        nonlocal count
        values = np.asarray(array, dtype=np.float64)
        count += len(values)
        sum_values[:] += values.sum(axis=0)
        sum_squares[:] += np.square(values).sum(axis=0)

    if refs is None:
        for sample in samples:
            accumulate(np.load(sample.x_path, mmap_mode="r"))
    else:
        if segment_frames is None:
            raise ValueError("segment_frames is required when refs are supplied")
        for ref in refs:
            x = np.load(samples[ref.sample_index].x_path, mmap_mode="r")
            accumulate(x[ref.start : ref.start + ref.length])
    if count == 0:
        raise ValueError("no frames found for normalization")
    mean = sum_values / count
    variance = np.maximum(sum_squares / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class PaperTrajectoryDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[TrajectorySample],
        refs: Sequence[SegmentRef],
        mean: np.ndarray,
        std: np.ndarray,
        segment_frames: int,
        input_norm: str = "global",
    ):
        if input_norm not in {"global", "per_sample"}:
            raise ValueError(f"unknown input_norm: {input_norm}")
        self.samples = list(samples)
        self.refs = list(refs)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (6,) or self.std.shape != (6,):
            raise ValueError("mean and std must each contain six values")
        if np.any(self.std <= 0):
            raise ValueError("std values must be positive")
        self.segment_frames = segment_frames
        self.input_norm = input_norm

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.refs[index]
        x, y, mask, board = materialize_segment(
            self.samples[ref.sample_index], ref, self.segment_frames
        )
        if self.input_norm == "per_sample":
            mean = x.mean(axis=0, keepdims=True)
            std = np.maximum(x.std(axis=0, keepdims=True), 1e-3)
            normalized = (x - mean) / std
        else:
            normalized = (x - self.mean) / self.std
        return (
            torch.from_numpy(normalized),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            torch.from_numpy(board),
        )
