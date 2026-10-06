from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


BLANK_INDEX = 0
NUM_CLASSES = 27


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def greedy_ctc_decode(logits: np.ndarray) -> str:
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim == 3:
        array = array[:, 0, :]
    indices = np.argmax(array, axis=-1)
    letters: list[str] = []
    previous: int | None = None
    for index in indices:
        index = int(index)
        if index != previous and index != BLANK_INDEX:
            letters.append(chr(ord("a") + index - 1))
        previous = index
    return "".join(letters)


def snap_to_vocabulary(
    word: str,
    vocabulary: Sequence[str] | set[str],
    max_distance: int | None = None,
) -> str | None:
    if not vocabulary:
        return None
    if word in vocabulary:
        return word
    best_word: str | None = None
    best_distance: int | None = None
    for candidate in vocabulary:
        distance = levenshtein_distance(word, candidate)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_word = candidate
    if max_distance is not None and (best_distance is None or best_distance > max_distance):
        return None
    return best_word


class WordCTC(nn.Module):
    """Convolutional-recurrent letter predictor for CTC word decoding."""

    def __init__(self, dropout: float = 0.2, hidden_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.sequence = nn.LSTM(
            128,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"expected [batch, frames, 2], got {tuple(x.shape)}")
        features = self.features(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence).permute(1, 0, 2)


def build_word_ctc_summary() -> dict[str, Any]:
    model = WordCTC()
    return {
        "num_classes": NUM_CLASSES,
        "blank_index": BLANK_INDEX,
        "resample_frames": 128,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_word_ctc(checkpoint: str | Path, device: str = "cpu") -> WordCTC:
    payload = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a state dict payload")
    model = WordCTC()
    model.load_state_dict(payload.get("state_dict", payload))
    model.to(device)
    model.eval()
    return model
