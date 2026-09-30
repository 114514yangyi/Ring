from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .paper_touch import TOUCH_LABELS, PaperTouchConfig


LABEL_TO_INDEX = {label: index for index, label in enumerate(TOUCH_LABELS)}
INDEX_TO_LABEL = dict(enumerate(TOUCH_LABELS))


@dataclass(frozen=True)
class CleanedSample:
    user: str
    action: int
    sample_id: str
    x_path: Path
    mask_path: Path


@dataclass(frozen=True)
class WindowRef:
    sample_index: int
    start: int
    label: int


def user_number(user: str) -> int:
    return int(str(user).split("_")[-1])


def discover_cleaned_samples(root: str | Path) -> list[CleanedSample]:
    root = Path(root)
    samples: list[CleanedSample] = []
    for user_dir in sorted(root.glob("user_*"), key=lambda path: user_number(path.name)):
        if not user_dir.is_dir():
            continue
        for action_dir in sorted(user_dir.iterdir(), key=lambda path: int(path.name) if path.name.isdigit() else 999):
            if not action_dir.is_dir() or not action_dir.name.isdigit():
                continue
            action = int(action_dir.name)
            for x_path in sorted(action_dir.glob("*_x.npy"), key=lambda path: int(path.stem.split("_")[0])):
                sample_id = x_path.stem[:-2]
                mask_path = action_dir / f"{sample_id}_mask.npy"
                if mask_path.exists():
                    samples.append(
                        CleanedSample(
                            user=user_dir.name,
                            action=action,
                            sample_id=sample_id,
                            x_path=x_path,
                            mask_path=mask_path,
                        )
                    )
    return samples


def default_user_split(users: Sequence[str]) -> tuple[set[str], set[str], set[str]]:
    ordered = sorted(set(users), key=user_number)
    if len(ordered) < 5:
        raise ValueError("need at least five users for a user-disjoint train/val/test split")
    test_count = max(1, round(len(ordered) * 0.2))
    val_count = max(1, round(len(ordered) * 0.1))
    test_users = set(ordered[-test_count:])
    val_users = set(ordered[-test_count - val_count : -test_count])
    train_users = set(ordered[: -test_count - val_count])
    return train_users, val_users, test_users


def window_labels(mask: np.ndarray, window_frames: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int8)
    if len(mask) < window_frames:
        return np.zeros(0, dtype=np.int64)
    start = mask[: len(mask) - window_frames + 1]
    end = mask[window_frames - 1 :]
    labels = np.empty_like(start, dtype=np.int64)
    labels[(start == 1) & (end == 1)] = LABEL_TO_INDEX["contact"]
    labels[(start == 0) & (end == 0)] = LABEL_TO_INDEX["air"]
    labels[(start == 1) & (end == 0)] = LABEL_TO_INDEX["lift"]
    labels[(start == 0) & (end == 1)] = LABEL_TO_INDEX["press"]
    return labels


def build_window_refs(
    samples: Sequence[CleanedSample],
    config: PaperTouchConfig,
    max_per_class: int | None = None,
    seed: int = 42,
) -> list[WindowRef]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    by_class: dict[int, list[tuple[int, np.ndarray]]] = {
        index: [] for index in range(len(TOUCH_LABELS))
    }
    for sample_index, sample in enumerate(samples):
        mask = np.load(sample.mask_path, mmap_mode="r")
        labels = window_labels(mask, config.window_frames)
        for label in range(len(TOUCH_LABELS)):
            starts = np.flatnonzero(labels == label)
            if len(starts) == 0:
                continue
            by_class[label].append((sample_index, starts.astype(np.int64, copy=False)))

    refs: list[WindowRef] = []
    for label, groups in by_class.items():
        total = int(sum(len(starts) for _, starts in groups))
        if total == 0:
            continue
        if max_per_class is None or total <= max_per_class:
            for sample_index, starts in groups:
                refs.extend(WindowRef(sample_index, int(start), label) for start in starts)
            continue

        offsets = np.cumsum([0] + [len(starts) for _, starts in groups])
        selected = np.sort(np_rng.choice(total, size=max_per_class, replace=False))
        group_index = np.searchsorted(offsets[1:], selected, side="right")
        for group_id in np.unique(group_index):
            selected_positions = selected[group_index == group_id] - offsets[group_id]
            sample_index, starts = groups[int(group_id)]
            refs.extend(
                WindowRef(sample_index, int(starts[int(position)]), label)
                for position in selected_positions
            )
    rng.shuffle(refs)
    return refs


def class_counts(refs: Iterable[WindowRef]) -> dict[str, int]:
    counts = {label: 0 for label in TOUCH_LABELS}
    for ref in refs:
        counts[INDEX_TO_LABEL[ref.label]] += 1
    return counts


def compute_normalization(samples: Sequence[CleanedSample]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    sum_values = np.zeros(6, dtype=np.float64)
    sum_squares = np.zeros(6, dtype=np.float64)
    for sample in samples:
        x = np.load(sample.x_path, mmap_mode="r")
        arr = np.asarray(x, dtype=np.float64)
        count += len(arr)
        sum_values += arr.sum(axis=0)
        sum_squares += np.square(arr).sum(axis=0)
    if count == 0:
        raise ValueError("no frames found for normalization")
    mean = sum_values / count
    var = np.maximum(sum_squares / count - np.square(mean), 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


class PaperTouchWindowDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[CleanedSample],
        refs: Sequence[WindowRef],
        mean: np.ndarray,
        std: np.ndarray,
        config: PaperTouchConfig | None = None,
    ):
        self.samples = list(samples)
        self.refs = list(refs)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.config = config or PaperTouchConfig()
        self._arrays = [np.load(sample.x_path, mmap_mode="r") for sample in self.samples]

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ref = self.refs[index]
        arr = self._arrays[ref.sample_index]
        window = np.asarray(
            arr[ref.start : ref.start + self.config.window_frames],
            dtype=np.float32,
        )
        window = (window - self.mean) / self.std
        return torch.from_numpy(window), torch.tensor(ref.label, dtype=torch.long)
