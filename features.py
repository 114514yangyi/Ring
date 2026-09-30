from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _value(source: Any, name: str, default: float = 0.0) -> float:
    if isinstance(source, Mapping):
        value = source.get(name, default)
    else:
        value = getattr(source, name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _vector(source: Any, name: str, fields: Sequence[str]) -> np.ndarray:
    value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
    if value is not None:
        if isinstance(value, Mapping):
            values = [_value(value, field) for field in fields]
        else:
            values = [_value(value, field) for field in fields]
        return np.asarray(values, dtype=float)

    prefix = {
        "lin_acc": ("lin_acc_x", "lin_acc_y", "lin_acc_z"),
        "acc": ("acc_x", "acc_y", "acc_z"),
        "gyro": ("gyro_x", "gyro_y", "gyro_z"),
    }.get(name)
    if prefix is None:
        return np.zeros(3, dtype=float)
    return np.asarray([_value(source, field) for field in prefix], dtype=float)


def timestamp_of(sample: Any, fallback: float) -> float:
    if isinstance(sample, Mapping):
        value = sample.get("timestamp", sample.get("time", fallback))
    else:
        value = getattr(sample, "timestamp", fallback)
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return timestamp if np.isfinite(timestamp) else float(fallback)


@dataclass(frozen=True)
class FrameFeatures:
    """Small, model-independent feature set used by the state machine."""

    timestamp: float
    lin_acc_norm: float
    acc_norm: float
    gyro_norm: float
    jerk_norm: float
    gyro_delta_norm: float
    motion_score: float
    impact_score: float


def extract_features(
    sample: Any,
    previous: FrameFeatures | None = None,
    fallback_timestamp: float = 0.0,
) -> FrameFeatures:
    lin_acc = _vector(sample, "lin_acc", ("x", "y", "z"))
    acc = _vector(sample, "acc", ("x", "y", "z"))
    gyro = _vector(sample, "gyro", ("x", "y", "z"))

    lin_acc_norm = float(np.linalg.norm(lin_acc))
    acc_norm = float(np.linalg.norm(acc))
    gyro_norm = float(np.linalg.norm(gyro))

    if previous is None:
        jerk_norm = 0.0
        gyro_delta_norm = 0.0
    else:
        jerk_norm = abs(lin_acc_norm - previous.lin_acc_norm)
        gyro_delta_norm = abs(gyro_norm - previous.gyro_norm)

    # The input data currently uses g for linear acceleration and the native
    # LPMS gyro unit. Keeping these ratios explicit makes calibration easy.
    motion_score = 0.65 * lin_acc_norm / 0.08 + 0.35 * gyro_norm / 20.0
    impact_score = max(
        lin_acc_norm / 0.35,
        jerk_norm / 0.12,
        gyro_delta_norm / 35.0,
    )

    return FrameFeatures(
        timestamp=timestamp_of(sample, fallback_timestamp),
        lin_acc_norm=lin_acc_norm,
        acc_norm=acc_norm,
        gyro_norm=gyro_norm,
        jerk_norm=jerk_norm,
        gyro_delta_norm=gyro_delta_norm,
        motion_score=float(motion_score),
        impact_score=float(impact_score),
    )
