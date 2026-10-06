from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


RESAMPLE_FRAMES = 64
NUM_CLASSES = 26


class CharacterCNN(nn.Module):
    """Small 1-D CNN over a resampled handwriting trajectory."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"expected [batch, frames, 2], got {tuple(x.shape)}")
        features = self.features(x.transpose(1, 2))
        pooled = self.pool(features).squeeze(-1)
        return self.classifier(self.dropout(pooled))


def build_character_summary() -> dict[str, Any]:
    model = CharacterCNN()
    return {
        "num_classes": NUM_CLASSES,
        "labels": [chr(ord("A") + index) for index in range(NUM_CLASSES)],
        "resample_frames": RESAMPLE_FRAMES,
        "input_shape": [RESAMPLE_FRAMES, 2],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_character_model(
    checkpoint: str | Path, device: str = "cpu"
) -> CharacterCNN:
    payload = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a state dict payload")
    model = CharacterCNN()
    model.load_state_dict(payload.get("state_dict", payload))
    model.to(device)
    model.eval()
    return model
