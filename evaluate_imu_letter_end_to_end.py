"""Evaluate 200 Hz letter recognition with the current touch decoder in front.

Unlike ``train_imu_letter_classifier.py``, this script does not use the
touchscreen-derived contact mask to crop an IMU recording.  It applies the
current four-state touch model on each causal 20-frame window and mirrors
``PaperTouchDecoder``: contact/press opens the gate and lift closes it.  All
frames accepted by that gate are then resampled for the already-trained letter
classifier.  The result is therefore a zero-shot, end-to-end test of the
existing two models, rather than a retrained joint model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .paper_touch import TOUCH_LABELS, PaperTouchConfig, load_touch_model
from .train_imu_letter_classifier import (
    IMULetterCNN,
    LABELS,
    load_examples,
    normalization,
    resample_imu,
    split_examples,
)


def _metrics(labels: np.ndarray, guesses: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for label, guess in zip(labels, guesses):
        confusion[int(label), int(guess)] += 1
    return {
        "n": int(len(labels)),
        "top1": float((guesses == labels).mean()) if len(labels) else 0.0,
        "top3": float((np.argsort(-probabilities, axis=1)[:, :3] == labels[:, None]).any(axis=1).mean()) if len(labels) else 0.0,
        "confusion": confusion.tolist(),
        "per_class": {
            LABELS[index]: {
                "support": int(confusion[index].sum()),
                "accuracy": float(confusion[index, index] / confusion[index].sum()) if confusion[index].sum() else None,
            }
            for index in range(len(LABELS))
        },
    }


def decoder_mask(values: np.ndarray, model: torch.nn.Module, mean: np.ndarray, std: np.ndarray, device: str, frames: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Return the causal decoder gate and the window labels at their end frame."""
    values = np.asarray(values, dtype=np.float32)
    gate = np.zeros(len(values), dtype=bool)
    window_labels = np.full(len(values), -1, dtype=np.int64)
    if len(values) < frames:
        return gate, window_labels
    windows = np.stack([values[index - frames + 1 : index + 1] for index in range(frames - 1, len(values))])
    with torch.no_grad():
        logits = model(torch.from_numpy((windows - mean) / std).to(device))
        predicted = logits.argmax(dim=1).cpu().numpy()
    active = False
    for end_index, label in zip(range(frames - 1, len(values)), predicted):
        window_labels[end_index] = int(label)
        # This is PaperTouchDecoder.update expressed at the available frame.
        if TOUCH_LABELS[int(label)] == "press" and not active:
            active = True
        elif TOUCH_LABELS[int(label)] == "lift" and active:
            active = False
        elif TOUCH_LABELS[int(label)] == "contact":
            active = True
        gate[end_index] = active
    return gate, window_labels


def run(args: argparse.Namespace) -> dict[str, Any]:
    letter_payload = torch.load(args.letter_checkpoint, map_location=args.device, weights_only=True)
    touch_payload = torch.load(args.touch_checkpoint, map_location="cpu", weights_only=True)
    touch_config = PaperTouchConfig(
        sample_rate=float(touch_payload["config"]["sample_rate"]),
        window_frames=int(touch_payload["config"]["window_frames"]),
        dropout=float(touch_payload["config"]["dropout"]),
    )
    touch_model = load_touch_model(args.touch_checkpoint, config=touch_config, device=args.device)
    letter_model = IMULetterCNN().to(args.device).eval()
    letter_model.load_state_dict(letter_payload["state_dict"])
    frames = int(letter_payload["frames"])

    examples = load_examples(args.data_root, frames)
    train, _, _, split = split_examples(examples)
    test = [row for row in examples if row["user"] in split["test_users"]]
    conversion = json.loads((args.data_root / "conversion_manifest.json").read_text(encoding="utf-8"))
    records_by_source = {int(row["source_row"]): row for row in conversion["records"]}
    letter_mean, letter_std = normalization(train)
    touch_mean = np.asarray(touch_payload["mean"], dtype=np.float32)
    touch_std = np.asarray(touch_payload["std"], dtype=np.float32)

    accepted: list[np.ndarray] = []
    labels: list[int] = []
    gate_ious: list[float] = []
    gate_lengths: list[dict[str, int]] = []
    state_counts = np.zeros(len(TOUCH_LABELS), dtype=np.int64)
    abstentions = 0
    for row in test:
        record = records_by_source[int(row["source_row"])]
        x_path = args.data_root / record["pseudo_user"] / "0" / f"{record['sample_id']}_x.npy"
        mask_path = args.data_root / record["pseudo_user"] / "0" / f"{record['sample_id']}_mask.npy"
        raw = np.load(x_path).astype(np.float32)
        truth = np.load(mask_path).astype(bool)
        gate, window_labels = decoder_mask(raw, touch_model, touch_mean, touch_std, args.device, touch_config.window_frames)
        valid_labels = window_labels[window_labels >= 0]
        state_counts += np.bincount(valid_labels, minlength=len(TOUCH_LABELS))
        union = int(np.logical_or(gate, truth).sum())
        gate_ious.append(float(np.logical_and(gate, truth).sum() / union) if union else 1.0)
        gate_lengths.append({"gt_contact_frames": int(truth.sum()), "decoder_gate_frames": int(gate.sum())})
        if not gate.any():
            abstentions += 1
            continue
        accepted.append(resample_imu(raw[gate], frames))
        labels.append(int(row["label"]))

    if accepted:
        inputs = (np.stack(accepted) - letter_mean) / letter_std
        with torch.no_grad():
            logits = letter_model(torch.from_numpy(inputs.astype(np.float32)).to(args.device)).cpu().numpy()
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        metrics = _metrics(np.asarray(labels), probabilities.argmax(axis=1), probabilities)
    else:
        metrics = _metrics(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, len(LABELS)), dtype=np.float32))

    result = {
        "protocol": "zero-shot end-to-end: current PaperTouchDecoder gate -> fixed GT-crop-trained IMU letter CNN; trial-disjoint pseudo-user test, not writer-disjoint",
        "data_root": str(args.data_root),
        "touch_checkpoint": str(args.touch_checkpoint),
        "letter_checkpoint": str(args.letter_checkpoint),
        "split": split,
        "test_trials": len(test),
        "abstentions": abstentions,
        "coverage": float((len(test) - abstentions) / len(test)) if test else 0.0,
        "decoder_gate": {
            "mean_iou_vs_gt_contact": float(np.mean(gate_ious)) if gate_ious else 0.0,
            "median_iou_vs_gt_contact": float(np.median(gate_ious)) if gate_ious else 0.0,
            "mean_gt_contact_frames": float(np.mean([row["gt_contact_frames"] for row in gate_lengths])) if gate_lengths else 0.0,
            "mean_decoder_gate_frames": float(np.mean([row["decoder_gate_frames"] for row in gate_lengths])) if gate_lengths else 0.0,
            "window_prediction_counts": dict(zip(TOUCH_LABELS, (int(value) for value in state_counts))),
        },
        "letter_metrics_on_covered_trials": metrics,
        "end_to_end_top1_all_trials": float(metrics["top1"] * len(labels) / len(test)) if test else 0.0,
        "end_to_end_top3_all_trials": float(metrics["top3"] * len(labels) / len(test)) if test else 0.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "end_to_end_touch_letter_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved report: {report_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 200 Hz IMU letters with the current touch decoder")
    parser.add_argument("--data-root", type=Path, default=Path("outputs/touch_200hz_v3/data"))
    parser.add_argument("--touch-checkpoint", type=Path, default=Path("outputs/touch_200hz_v3/model/paper_touch_resnet.pt"))
    parser.add_argument("--letter-checkpoint", type=Path, default=Path("outputs/imu_letter_200hz_v1/imu_letter_cnn.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/imu_letter_200hz_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
