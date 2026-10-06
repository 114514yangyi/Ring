from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PaperTrajectoryConfig:
    """WritingRing trajectory-reconstruction defaults with explicit knobs."""

    sample_rate: float = 200.0
    window_frames: int = 13
    tcn_kernel: int = 3
    tcn_channels: tuple[int, ...] = (16, 16, 32, 32, 64, 128)
    tcn_style: str = "valid"
    feature_dim: int = 128
    lstm_hidden: int = 128
    lstm_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.2
    board_mm_scale: tuple[float, float] | None = (240.0, 169.5)

    @property
    def window_offset(self) -> int:
        return self.window_frames // 2

    @property
    def segment_frames(self) -> int:
        return int(round(self.sample_rate * 75.0))

    def validate(self) -> None:
        if self.window_frames < 3 or self.window_frames % 2 == 0:
            raise ValueError("window_frames must be an odd number >= 3")
        if self.tcn_kernel < 1 or self.tcn_kernel % 2 == 0:
            raise ValueError("tcn_kernel must be a positive odd number")
        if not self.tcn_channels:
            raise ValueError("tcn_channels must not be empty")
        if self.tcn_style not in {"valid", "dilated"}:
            raise ValueError(f"unknown tcn_style: {self.tcn_style}")
        if self.tcn_style == "valid":
            reduction = len(self.tcn_channels) * (self.tcn_kernel - 1)
            if self.window_frames - reduction != 1:
                raise ValueError(
                    "TCN must reduce the window to a single frame: "
                    f"{self.window_frames} - {len(self.tcn_channels)}*"
                    f"({self.tcn_kernel}-1) != 1"
                )
            if self.tcn_channels[-1] != self.feature_dim:
                raise ValueError("feature_dim must match the last TCN channel count")
        if self.lstm_layers < 1:
            raise ValueError("lstm_layers must be positive")
        if self.segment_frames < self.window_frames:
            raise ValueError("segment_frames must cover at least one window")


def stack_windows(x: Tensor, window_frames: int = 13) -> Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected [frames, channels], got {tuple(x.shape)}")
    if x.shape[-1] != 6:
        raise ValueError("expected six input channels")
    if x.shape[0] < window_frames:
        return x.new_zeros((0, 6, window_frames))
    return x.unfold(0, window_frames, 1)


def chunk_slices(
    total: int, chunk_len: int, offset: int
) -> list[tuple[int, int, int, int]]:
    if chunk_len < 1:
        raise ValueError("chunk_len must be positive")
    ranges: list[tuple[int, int, int, int]] = []
    for start in range(0, total, chunk_len):
        stop = min(total, start + chunk_len)
        out_lo = max(offset, start)
        out_hi = min(total - offset, stop)
        if out_lo >= out_hi:
            continue
        x_lo = max(0, out_lo - offset)
        x_hi = min(total, out_hi + offset)
        ranges.append((x_lo, x_hi, out_lo, out_hi))
    return ranges


class _TCNLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel: int, dropout: float
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel)
        self.bn = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity() if in_channels == out_channels else None
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.drop(self.bn(self.conv(x)))
        if self.skip is not None:
            residual = self.skip(x)
            output = output + residual[:, :, : output.shape[-1]]
        return self.activation(output)


class _DilatedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel, padding=dilation, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(
            self.drop(self.bn(self.conv(x))) + self.skip(x)
        )


class TrajectoryTCN(nn.Module):
    def __init__(self, config: PaperTrajectoryConfig | None = None):
        super().__init__()
        self.config = config or PaperTrajectoryConfig()
        self.config.validate()
        layers = []
        in_channels = 6
        for out_channels in self.config.tcn_channels:
            layers.append(
                _TCNLayer(in_channels, out_channels, self.config.tcn_kernel, self.config.dropout)
            )
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, windows: Tensor) -> Tensor:
        return self.layers(windows).squeeze(-1)


class WritingRingTrajectoryNet(nn.Module):
    """TCN feature extractor plus streaming LSTM velocity head."""

    def __init__(self, config: PaperTrajectoryConfig | None = None):
        super().__init__()
        self.config = config or PaperTrajectoryConfig()
        self.config.validate()
        if self.config.tcn_style == "valid":
            self.tcn = TrajectoryTCN(self.config)
            self.tcn_projection = None
        else:
            blocks = []
            in_channels = 6
            for index, out_channels in enumerate(self.config.tcn_channels):
                blocks.append(
                    _DilatedBlock(
                        in_channels,
                        out_channels,
                        self.config.tcn_kernel,
                        dilation=2**index,
                        dropout=self.config.dropout,
                    )
                )
                in_channels = out_channels
            self.tcn = nn.Sequential(*blocks)
            self.tcn_projection = nn.Linear(
                self.config.tcn_channels[-1] * self.config.window_frames,
                self.config.feature_dim,
            )
        self.lstm = nn.LSTM(
            self.config.feature_dim,
            self.config.lstm_hidden,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            bidirectional=self.config.bidirectional,
        )
        head_input = self.config.lstm_hidden * (2 if self.config.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(head_input, self.config.lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.lstm_hidden, 2),
        )

    def window_features(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, frames, channels], got {tuple(x.shape)}")
        if x.shape[-1] != 6:
            raise ValueError("expected six input channels")
        batch, total, _ = x.shape
        window = self.config.window_frames
        if total < window:
            return x.new_zeros((batch, 0, self.config.feature_dim))
        windows = x.unfold(1, window, 1).reshape(
            batch * (total - window + 1), 6, window
        )
        extracted = self.tcn(windows)
        if self.tcn_projection is not None:
            extracted = self.tcn_projection(extracted.flatten(start_dim=1))
        else:
            extracted = extracted.squeeze(-1)
        return extracted.reshape(batch, total - window + 1, self.config.feature_dim)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch, total, _ = x.shape
        output = x.new_full((batch, total, 2), float("nan"))
        if total < self.config.window_frames:
            return output, state
        features = self.window_features(x)
        hidden, state = self.lstm(features, state)
        predictions = self.head(hidden)
        offset = self.config.window_offset
        output[:, offset : offset + predictions.shape[1]] = predictions
        return output, state


def predict_sequence(
    model: WritingRingTrajectoryNet,
    x: np.ndarray | Tensor,
    chunk_len: int | None = None,
) -> np.ndarray:
    array = (
        x.detach().cpu().numpy()
        if isinstance(x, Tensor)
        else np.asarray(x, dtype=np.float32)
    )
    total = int(array.shape[0])
    output = np.full((total, 2), np.nan, dtype=np.float32)
    if model.config.bidirectional and chunk_len is not None and chunk_len < total:
        raise ValueError(
            "a bidirectional model needs the full sequence; chunking would leak future "
            "context across chunks"
        )
    if total < model.config.window_frames:
        return output

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    state: tuple[Tensor, Tensor] | None = None
    with torch.no_grad():
        for x_lo, x_hi, out_lo, out_hi in chunk_slices(
            total, chunk_len or total, model.config.window_offset
        ):
            tensor = torch.from_numpy(
                np.ascontiguousarray(array[x_lo:x_hi], dtype=np.float32)
            ).to(device)[None]
            chunk_output, state = model(tensor, state)
            if state is not None:
                state = (state[0].detach(), state[1].detach())
            offset = model.config.window_offset
            local_lo = out_lo - x_lo - offset
            local_hi = out_hi - x_lo - offset
            output[out_lo:out_hi] = (
                chunk_output[
                    0, offset + local_lo : offset + local_hi
                ]
                .cpu()
                .numpy()
            )
    if was_training:
        model.train()
    return output


class TrajectoryPredictor:
    def __init__(
        self,
        model: WritingRingTrajectoryNet,
        config: PaperTrajectoryConfig,
        mean: Sequence[float],
        std: Sequence[float],
        board_mm_scale: Sequence[float] | None = None,
    ):
        self.model = model
        self.config = config
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (6,) or self.std.shape != (6,):
            raise ValueError("mean and std must each contain six values")
        if np.any(self.std <= 0):
            raise ValueError("std values must be positive")
        self.board_mm_scale = (
            tuple(float(value) for value in board_mm_scale)
            if board_mm_scale is not None
            else None
        )

    def predict_deltas(self, x: np.ndarray) -> np.ndarray:
        values = (np.asarray(x, dtype=np.float32) - self.mean) / self.std
        return predict_sequence(self.model, values)


def _payload_config(payload: Mapping[str, Any]) -> PaperTrajectoryConfig:
    raw = payload.get("config")
    if not raw:
        return PaperTrajectoryConfig()
    data = dict(raw)
    if "tcn_channels" in data:
        data["tcn_channels"] = tuple(data["tcn_channels"])
    if data.get("board_mm_scale") is not None:
        data["board_mm_scale"] = tuple(data["board_mm_scale"])
    known = {field.name for field in fields(PaperTrajectoryConfig)}
    return PaperTrajectoryConfig(**{key: value for key, value in data.items() if key in known})


def load_trajectory_model(
    checkpoint: str | Path,
    config: PaperTrajectoryConfig | None = None,
    device: str = "cpu",
) -> WritingRingTrajectoryNet:
    payload = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a state dict payload")
    config = config or _payload_config(payload)
    model = WritingRingTrajectoryNet(config)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_trajectory_predictor(
    checkpoint: str | Path, device: str = "cpu"
) -> TrajectoryPredictor:
    payload = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a state dict payload")
    mean = payload.get("mean")
    std = payload.get("std")
    if mean is None or std is None:
        raise ValueError("checkpoint is missing mean/std normalization metadata")
    config = _payload_config(payload)
    model = load_trajectory_model(checkpoint, config=config, device=device)
    return TrajectoryPredictor(
        model, config, mean, std, payload.get("board_mm_scale")
    )


def build_trajectory_summary(
    config: PaperTrajectoryConfig | None = None,
) -> dict[str, Any]:
    config = config or PaperTrajectoryConfig()
    config.validate()
    model = WritingRingTrajectoryNet(config)
    return {
        "sample_rate": config.sample_rate,
        "window_frames": config.window_frames,
        "window_offset": config.window_offset,
        "segment_frames": config.segment_frames,
        "tcn_kernel": config.tcn_kernel,
        "tcn_channels": list(config.tcn_channels),
        "tcn_style": config.tcn_style,
        "feature_dim": config.feature_dim,
        "lstm_hidden": config.lstm_hidden,
        "lstm_layers": config.lstm_layers,
        "bidirectional": config.bidirectional,
        "input_shape": [config.window_frames, 6],
        "output_dim": 2,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "board_mm_scale": list(config.board_mm_scale)
        if config.board_mm_scale is not None
        else None,
    }


@dataclass(frozen=True)
class SegmentTrajectory:
    start: int
    end: int
    xy: np.ndarray
    lost_frames: int = 0


def contact_segments(mask: np.ndarray, min_frames: int = 5) -> list[tuple[int, int]]:
    values = (np.asarray(mask) > 0).astype(np.int8)
    padded = np.concatenate([[0], values, [0]])
    changes = np.flatnonzero(np.diff(padded))
    return [
        (int(start), int(stop))
        for start, stop in zip(changes[::2], changes[1::2])
        if stop - start >= min_frames
    ]


def integrate_contact_segments(
    deltas: np.ndarray,
    mask: np.ndarray,
    window_offset: int = 6,
    min_frames: int = 5,
) -> list[SegmentTrajectory]:
    """Integrate predicted per-frame deltas inside contact runs.

    ``deltas[t]`` estimates the displacement from frame ``t`` to ``t + 1``
    (dataset truth: ``y[t] = board[t+1] - board[t]``). A segment covering
    times ``[start, end)`` has ``xy[0] = 0`` at ``start`` and
    ``xy[k] = sum(deltas[start:start+k])`` at ``start + k``.
    ``window_offset`` is accepted for interface parity with the streaming
    model; non-finite predictions (block edges) are detected directly and
    split out of the segment.
    """

    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] != 2:
        raise ValueError("deltas must have shape [frames, 2]")
    mask = np.asarray(mask)
    if len(mask) != len(deltas):
        raise ValueError("deltas and mask must have the same length")

    segments: list[SegmentTrajectory] = []
    for run_start, run_stop in contact_segments(mask, min_frames=1):
        delta_stop = run_stop - 1
        if delta_stop <= run_start:
            continue
        finite = np.isfinite(deltas[run_start:delta_stop]).all(axis=1)
        padded = np.concatenate([[False], finite, [False]]).astype(np.int8)
        changes = np.flatnonzero(np.diff(padded))
        for first, last in zip(changes[::2], changes[1::2]):
            d0 = run_start + int(first)
            d1 = run_start + int(last)
            if d1 <= d0:
                continue
            n_points = d1 - d0 + 1
            if n_points < min_frames:
                continue
            xy = np.zeros((n_points, 2), dtype=np.float32)
            xy[1:] = np.cumsum(deltas[d0:d1], axis=0)
            segments.append(
                SegmentTrajectory(
                    start=d0,
                    end=d1 + 1,
                    xy=xy,
                    lost_frames=(run_stop - run_start) - n_points,
                )
            )
    return segments


def trajectory_metrics(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    mm_scale: Sequence[float] | None = None,
) -> dict[str, Any]:
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(
            f"prediction and ground truth shapes differ: {pred.shape} vs {gt.shape}"
        )
    count = len(gt)
    if count:
        span = gt.max(axis=0) - gt.min(axis=0)
        diagonal = float(np.linalg.norm(span))
    else:
        diagonal = 0.0

    if count == 0 or diagonal < 1e-6:
        return {
            "n_points": count,
            "normalized_mean": float("nan"),
            "normalized_p50": float("nan"),
            "normalized_p90": float("nan"),
            "frac_gt_0.1": float("nan"),
            "mm_mean": float("nan"),
            "skipped_degenerate": 1,
        }

    distances = np.linalg.norm(pred - gt, axis=1)
    normalized = distances / diagonal
    mm_mean = float("nan")
    if mm_scale is not None:
        scale = np.asarray(mm_scale, dtype=np.float64)
        if scale.shape != (2,):
            raise ValueError("mm_scale must contain two values")
        mm_distances = np.linalg.norm((pred - gt) * scale, axis=1)
        mm_mean = float(mm_distances.mean())
    return {
        "n_points": count,
        "normalized_mean": float(normalized.mean()),
        "normalized_p50": float(np.percentile(normalized, 50)),
        "normalized_p90": float(np.percentile(normalized, 90)),
        "frac_gt_0.1": float((normalized > 0.1).mean()),
        "mm_mean": mm_mean,
        "skipped_degenerate": 0,
    }


def summarize_segment_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("skipped_degenerate", 0) == 0 and row.get("n_points", 0) > 0
    ]
    skipped = len(rows) - len(valid)
    if not valid:
        return {
            "segments": 0,
            "skipped_degenerate": skipped,
            "n_points": 0,
            "normalized_mean": float("nan"),
            "normalized_p50": float("nan"),
            "normalized_p90": float("nan"),
            "frac_gt_0.1": float("nan"),
            "mm_mean": float("nan"),
        }
    weights = np.asarray([row["n_points"] for row in valid], dtype=np.float64)

    def weighted_mean(key: str) -> float:
        values = np.asarray([row.get(key, float("nan")) for row in valid], dtype=np.float64)
        finite = ~np.isnan(values)
        if not finite.any():
            return float("nan")
        return float((values[finite] * weights[finite]).sum() / weights[finite].sum())

    return {
        "segments": len(valid),
        "skipped_degenerate": skipped,
        "n_points": int(weights.sum()),
        "normalized_mean": weighted_mean("normalized_mean"),
        "normalized_p50": weighted_mean("normalized_p50"),
        "normalized_p90": weighted_mean("normalized_p90"),
        "frac_gt_0.1": weighted_mean("frac_gt_0.1"),
        "mm_mean": weighted_mean("mm_mean"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the WritingRing trajectory reconstruction model"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = build_trajectory_summary()
    if args.checkpoint is not None:
        load_trajectory_model(args.checkpoint)
        summary["checkpoint"] = str(args.checkpoint)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
