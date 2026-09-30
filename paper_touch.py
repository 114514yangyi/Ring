"""WritingRing-style touch-state detection pipeline.

This module reproduces the touch detector described in WritingRing:

* 0.1 s windows containing 20 frames of six-axis IMU data;
* a small 1-D ResNet with residual-block widths 8, 16, and 32;
* four window labels: contact, air, lift, and press;
* press/lift events decoded into a causal writing-validity gate.

The paper does not publish trained weights in this repository. Inference is
therefore disabled until a checkpoint is supplied; constructing the model is
still useful for validating data shape and the eventual training interface.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


class TouchState(str, Enum):
    CONTACT = "contact"
    AIR = "air"
    LIFT = "lift"
    PRESS = "press"


TOUCH_LABELS = tuple(item.value for item in TouchState)


@dataclass(frozen=True)
class PaperTouchConfig:
    """Paper-compatible defaults, with explicit project adaptation knobs."""

    sample_rate: float = 200.0
    window_seconds: float = 0.1
    window_frames: int = 20
    stride_frames: int = 1
    input_channels: int = 6
    dropout: float = 0.2
    resample_window: bool = True

    def validate(self) -> None:
        expected = int(round(self.sample_rate * self.window_seconds))
        if self.window_frames != expected and not self.resample_window:
            raise ValueError(
                "paper window configuration is inconsistent: "
                f"{self.sample_rate:g} Hz * {self.window_seconds:g} s "
                f"requires {expected} frames, got {self.window_frames}"
            )
        if self.input_channels != 6:
            raise ValueError("WritingRing touch detector expects six IMU channels")
        if self.stride_frames < 1:
            raise ValueError("stride_frames must be positive")


def _scalar(source: Any, key: str, default: float = 0.0) -> float:
    if isinstance(source, Mapping):
        value = source.get(key, default)
    else:
        value = getattr(source, key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _triplet(source: Any, object_name: str, flat_names: Sequence[str]) -> np.ndarray:
    value = (
        source.get(object_name)
        if isinstance(source, Mapping)
        else getattr(source, object_name, None)
    )
    if value is not None:
        return np.asarray(
            [
                _scalar(value, "x"),
                _scalar(value, "y"),
                _scalar(value, "z"),
            ],
            dtype=np.float32,
        )
    return np.asarray(
        [_scalar(source, name) for name in flat_names],
        dtype=np.float32,
    )


def _has_vector(source: Any, object_name: str, flat_names: Sequence[str]) -> bool:
    if isinstance(source, Mapping):
        return object_name in source or any(name in source for name in flat_names)
    return hasattr(source, object_name) or any(hasattr(source, name) for name in flat_names)


def sample_to_six_axes(sample: Any) -> np.ndarray:
    """Convert a SmartRing frame or normalized mapping to [lin_acc, gyro]."""

    lin_acc = _triplet(
        sample,
        "lin_acc",
        ("lin_acc_x", "lin_acc_y", "lin_acc_z"),
    )
    if not _has_vector(
        sample,
        "lin_acc",
        ("lin_acc_x", "lin_acc_y", "lin_acc_z"),
    ):
        lin_acc = _triplet(sample, "acc", ("acc_x", "acc_y", "acc_z"))
    gyro = _triplet(sample, "gyro", ("gyro_x", "gyro_y", "gyro_z"))
    return np.concatenate([lin_acc, gyro]).astype(np.float32, copy=False)


def sample_timestamp(sample: Any, fallback: float) -> float:
    value = (
        sample.get("timestamp", sample.get("time", fallback))
        if isinstance(sample, Mapping)
        else getattr(sample, "timestamp", fallback)
    )
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if np.isfinite(value) else fallback


@dataclass(frozen=True)
class TouchWindow:
    values: np.ndarray
    start_timestamp: float
    end_timestamp: float
    center_timestamp: float


class TouchWindowizer:
    """Causal time-windowing used by the touch classifier.

    The model input stays at 20 frames. If the source stream is not 200 Hz,
    the 0.1 s time window is linearly resampled to 20 frames before inference.
    """

    def __init__(self, config: PaperTouchConfig | None = None):
        self.config = config or PaperTouchConfig()
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        maxlen = max(
            self.config.window_frames * 4,
            int(round(self.config.sample_rate * self.config.window_seconds * 4)),
        )
        self._samples: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._timestamps: Deque[float] = deque(maxlen=maxlen)
        self._seen = 0

    def push(self, sample: Any) -> TouchWindow | None:
        fallback = (
            self._timestamps[-1] + 1.0 / self.config.sample_rate
            if self._timestamps
            else 0.0
        )
        self._samples.append(sample_to_six_axes(sample))
        self._timestamps.append(sample_timestamp(sample, fallback))
        self._seen += 1

        if len(self._samples) < 2:
            return None
        if (self._seen - self.config.window_frames) % self.config.stride_frames:
            return None

        values_all = np.stack(tuple(self._samples), axis=0)
        timestamps = np.asarray(tuple(self._timestamps), dtype=np.float64)
        if float(timestamps[-1] - timestamps[0]) < self.config.window_seconds:
            return None
        end = float(timestamps[-1])
        start = end - self.config.window_seconds
        mask = timestamps >= start
        if int(np.count_nonzero(mask)) < 2:
            return None
        timestamps_window = timestamps[mask]
        values_window = values_all[mask]
        if self.config.resample_window:
            target_times = np.linspace(start, end, self.config.window_frames)
            values = np.stack(
                [
                    np.interp(target_times, timestamps_window, values_window[:, channel])
                    for channel in range(self.config.input_channels)
                ],
                axis=1,
            ).astype(np.float32)
        else:
            if len(values_window) < self.config.window_frames:
                return None
            values = values_window[-self.config.window_frames :]
            target_times = timestamps_window[-self.config.window_frames :]

        return TouchWindow(
            values=values,
            start_timestamp=float(target_times[0]),
            end_timestamp=float(target_times[-1]),
            center_timestamp=float(target_times[len(target_times) // 2]),
        )


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        return self.activation(x + residual)


class WritingRingTouchResNet(nn.Module):
    """Small six-axis, four-class 1-D ResNet from the paper description."""

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if input_channels != 6:
            raise ValueError("WritingRingTouchResNet expects six input channels")
        if num_classes != 4:
            raise ValueError("WritingRingTouchResNet expects four output classes")

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(8, 8, dropout),
            ResidualBlock1D(8, 16, dropout),
            ResidualBlock1D(16, 32, dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, frames, channels], got {tuple(x.shape)}")
        if x.shape[-1] != 6:
            raise ValueError(f"expected six channels, got {x.shape[-1]}")
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def load_touch_model(
    checkpoint: str | Path,
    config: PaperTouchConfig | None = None,
    device: str = "cpu",
) -> WritingRingTouchResNet:
    config = config or PaperTouchConfig()
    model = WritingRingTouchResNet(dropout=config.dropout)
    payload = torch.load(str(checkpoint), map_location=device, weights_only=True)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint must contain a PyTorch state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_touch_classifier(
    checkpoint: str | Path,
    config: PaperTouchConfig | None = None,
    device: str = "cpu",
) -> PaperTouchClassifier:
    """Load a trained touch classifier with normalization metadata."""

    payload = torch.load(str(checkpoint), map_location=device, weights_only=True)
    config = config or PaperTouchConfig()
    model = load_touch_model(checkpoint, config=config, device=device)
    mean = payload.get("mean")
    std = payload.get("std")
    if mean is None or std is None:
        raise ValueError("checkpoint is missing mean/std normalization metadata")
    return PaperTouchClassifier(model, config=config, device=device, mean=mean, std=std)


@dataclass(frozen=True)
class TouchPrediction:
    label: TouchState
    probabilities: tuple[float, ...]
    start_timestamp: float
    end_timestamp: float
    center_timestamp: float

    @property
    def confidence(self) -> float:
        return max(self.probabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "confidence": self.confidence,
            "probabilities": dict(zip(TOUCH_LABELS, self.probabilities)),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "center_timestamp": self.center_timestamp,
        }


class PaperTouchClassifier:
    def __init__(
        self,
        model: WritingRingTouchResNet,
        config: PaperTouchConfig | None = None,
        device: str = "cpu",
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ):
        self.config = config or PaperTouchConfig()
        self.config.validate()
        self.model = model.to(device).eval()
        self.device = device
        self.mean = np.asarray(mean if mean is not None else np.zeros(6), dtype=np.float32)
        self.std = np.asarray(std if std is not None else np.ones(6), dtype=np.float32)
        if self.mean.shape != (6,) or self.std.shape != (6,):
            raise ValueError("mean and std must each contain six values")
        if np.any(self.std <= 0):
            raise ValueError("std values must be positive")

    def predict(self, window: TouchWindow) -> TouchPrediction:
        values = (window.values - self.mean) / self.std
        tensor = torch.from_numpy(values[None]).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=-1)[0].cpu().numpy()
        index = int(np.argmax(probabilities))
        return TouchPrediction(
            label=TouchState(TOUCH_LABELS[index]),
            probabilities=tuple(float(value) for value in probabilities),
            start_timestamp=window.start_timestamp,
            end_timestamp=window.end_timestamp,
            center_timestamp=window.center_timestamp,
        )


@dataclass(frozen=True)
class TouchEvent:
    type: str
    timestamp: float
    available_at: float
    state: TouchState
    valid_operation: bool
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "available_at": self.available_at,
            "state": self.state.value,
            "valid_operation": self.valid_operation,
            "confidence": self.confidence,
        }


class PaperTouchDecoder:
    """Decode four-class window predictions into press/lift events."""

    def __init__(self):
        self.contact_active = False

    @property
    def valid_operation(self) -> bool:
        return self.contact_active

    def reset(self) -> None:
        self.contact_active = False

    def update(self, prediction: TouchPrediction) -> list[TouchEvent]:
        event_type: str | None = None
        if prediction.label == TouchState.PRESS and not self.contact_active:
            self.contact_active = True
            event_type = "writing_start"
        elif prediction.label == TouchState.LIFT and self.contact_active:
            self.contact_active = False
            event_type = "pen_up"
        elif prediction.label == TouchState.CONTACT:
            self.contact_active = True

        if event_type is None:
            return []
        return [
            TouchEvent(
                type=event_type,
                timestamp=prediction.center_timestamp,
                available_at=prediction.end_timestamp,
                state=prediction.label,
                valid_operation=self.contact_active,
                confidence=prediction.confidence,
            )
        ]


class PaperTouchDetector:
    """Streaming paper-style detector with no threshold-based decisions."""

    def __init__(
        self,
        classifier: PaperTouchClassifier,
        config: PaperTouchConfig | None = None,
    ):
        self.config = config or classifier.config
        self.windowizer = TouchWindowizer(self.config)
        self.classifier = classifier
        self.decoder = PaperTouchDecoder()

    @property
    def valid_operation(self) -> bool:
        return self.decoder.valid_operation

    def reset(self) -> None:
        self.windowizer.reset()
        self.decoder.reset()

    def push(self, sample: Any) -> tuple[TouchPrediction | None, list[TouchEvent]]:
        window = self.windowizer.push(sample)
        if window is None:
            return None, []
        prediction = self.classifier.predict(window)
        return prediction, self.decoder.update(prediction)

    def process(
        self, samples: Iterable[Any]
    ) -> tuple[list[TouchPrediction], list[TouchEvent]]:
        predictions: list[TouchPrediction] = []
        events: list[TouchEvent] = []
        for sample in samples:
            prediction, new_events = self.push(sample)
            if prediction is not None:
                predictions.append(prediction)
                events.extend(new_events)
        return predictions, events


def build_model_summary(config: PaperTouchConfig | None = None) -> dict[str, Any]:
    config = config or PaperTouchConfig()
    config.validate()
    model = WritingRingTouchResNet(dropout=config.dropout)
    return {
        "window_seconds": config.window_seconds,
        "window_frames": config.window_frames,
        "input_shape": [config.window_frames, config.input_channels],
        "labels": list(TOUCH_LABELS),
        "residual_block_channels": [8, 16, 32],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inference_ready": False,
        "reason": "no trained checkpoint supplied",
    }


def export_windows(
    samples: Iterable[Any],
    out_npz: str | Path,
    manifest_csv: str | Path | None = None,
    config: PaperTouchConfig | None = None,
) -> dict[str, Any]:
    config = config or PaperTouchConfig()
    windowizer = TouchWindowizer(config)
    values: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for sample in samples:
        window = windowizer.push(sample)
        if window is None:
            continue
        index = len(values)
        values.append(window.values.astype(np.float32, copy=False))
        rows.append(
            {
                "window_index": index,
                "start_timestamp": window.start_timestamp,
                "end_timestamp": window.end_timestamp,
                "center_timestamp": window.center_timestamp,
            }
        )

    array = np.stack(values, axis=0) if values else np.zeros((0, config.window_frames, 6), dtype=np.float32)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        windows=array,
        labels=np.asarray(TOUCH_LABELS),
        window_seconds=np.asarray([config.window_seconds], dtype=np.float32),
    )

    if manifest_csv is not None:
        manifest_csv = Path(manifest_csv)
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        with manifest_csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "window_index",
                    "start_timestamp",
                    "end_timestamp",
                    "center_timestamp",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    return {
        "window_count": int(array.shape[0]),
        "shape": list(array.shape),
        "out_npz": str(out_npz),
        "manifest_csv": str(manifest_csv) if manifest_csv is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the WritingRing touch detector reproduction")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--csv", type=Path, help="export paper-shaped windows from a SmartRing/OpenZen CSV")
    parser.add_argument("--out-npz", type=Path, default=Path("outputs/writing_touch_windows.npz"))
    parser.add_argument("--manifest-out", type=Path, default=Path("outputs/writing_touch_windows.csv"))
    parser.add_argument("--sample-rate", type=float, default=200.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = PaperTouchConfig(sample_rate=args.sample_rate)
    if args.csv is not None:
        from .io import iter_samples, read_sensor_csv

        df = read_sensor_csv(args.csv)
        summary = export_windows(
            iter_samples(df),
            out_npz=args.out_npz,
            manifest_csv=args.manifest_out,
            config=config,
        )
        print(json.dumps(summary, indent=2) if args.json else summary)
        return

    summary = build_model_summary(config)
    if args.checkpoint is not None:
        load_touch_model(args.checkpoint)
        summary["inference_ready"] = True
        summary["checkpoint"] = str(args.checkpoint)
        summary.pop("reason", None)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
