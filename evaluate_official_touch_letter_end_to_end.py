"""Run the official WritingRing touch gate through trajectory-to-letter inference.

This evaluates the missing link in the existing official-data character result:
the trajectory and character models normally receive the touchscreen-derived
``*_mask.npy``.  Here that mask is replaced by the causal gate from the
trained four-state touch detector, without using raw touch boundaries to
repair it.  The clean release supplies IMU/board arrays; the raw release is
used only for the author-provided character timestamps and labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .character_dataset import CharacterSample, contact_runs, discover_character_samples
from .character_model import load_character_model
from .paper_dataset import default_user_split, user_number
from .paper_touch import TOUCH_LABELS, PaperTouchConfig, load_touch_model
from .paper_trajectory import load_trajectory_predictor
from .paper_trajectory_dataset import discover_trajectory_samples, load_sample_arrays, sample_is_valid
from .train_character_classifier import build_recon_examples, evaluate_examples, predict_run_points


def _read_labels(path: Path) -> list[tuple[int, str]]:
    tokens = path.read_text().split()
    return [(int(float(tokens[index])), tokens[index + 1]) for index in range(0, len(tokens), 2)]


def _predict_gate(
    values: np.ndarray,
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    frames: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact state transitions of ``PaperTouchDecoder`` causally."""
    values = np.asarray(values, dtype=np.float32)
    gate = np.zeros(len(values), dtype=bool)
    labels = np.full(len(values), -1, dtype=np.int64)
    if len(values) < frames:
        return gate, labels
    windows = np.stack([values[end - frames + 1 : end + 1] for end in range(frames - 1, len(values))])
    predicted = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy((windows[start : start + batch_size] - mean) / std).to(device)
            predicted.append(model(batch).argmax(dim=1).cpu().numpy())
    active = False
    for end, label in zip(range(frames - 1, len(values)), np.concatenate(predicted)):
        label = int(label)
        labels[end] = label
        if TOUCH_LABELS[label] == "press" and not active:
            active = True
        elif TOUCH_LABELS[label] == "lift" and active:
            active = False
        elif TOUCH_LABELS[label] == "contact":
            active = True
        gate[end] = active
    return gate, labels


def _samples_from_gate(
    block: Any, raw_root: Path, gate: np.ndarray, timestamp: np.ndarray
) -> tuple[list[CharacterSample], list[str]]:
    label_path = raw_root / block.user / str(block.action) / f"{block.sample_id}_timestamp.txt"
    if not label_path.exists():
        return [], []
    labels = _read_labels(label_path)
    runs = contact_runs(gate, min_frames=8)
    if not runs:
        return [], []
    times = np.asarray(timestamp, dtype=np.float64)
    frames = np.clip(np.searchsorted(times, np.asarray([time for time, _ in labels], dtype=np.float64)), 0, len(times) - 1).astype(int)
    output: list[CharacterSample] = []
    identities: list[str] = []
    for index, (label_time, char) in enumerate(labels):
        letter = char.upper()
        if char == "wrong" or label_time < times[0] or label_time > times[-1] or len(letter) != 1 or not ("A" <= letter <= "Z"):
            continue
        low = int(frames[index])
        high = int(frames[index + 1]) if index + 1 < len(labels) else len(times)
        selected = tuple((start, stop) for start, stop in runs if low <= start < high)
        if sum(stop - start for start, stop in selected) < 8:
            continue
        output.append(CharacterSample(block.user, block.action, block.sample_id, ord(letter) - ord("A"), letter, selected))
        identities.append(f"{block.user}/{block.action}/{block.sample_id}/{index}")
    return output, identities


def _test_users(samples: Sequence[CharacterSample]) -> set[str]:
    _, _, test_users = default_user_split([sample.user for sample in samples])
    return test_users


def run(args: argparse.Namespace) -> dict[str, Any]:
    clean_root, raw_root = Path(args.clean_root), Path(args.raw_root)
    touch_payload = torch.load(args.touch_checkpoint, map_location="cpu", weights_only=True)
    touch_config = PaperTouchConfig(
        sample_rate=float(touch_payload["config"]["sample_rate"]),
        window_frames=int(touch_payload["config"]["window_frames"]),
        dropout=float(touch_payload["config"]["dropout"]),
    )
    touch_model = load_touch_model(args.touch_checkpoint, touch_config, args.device)
    touch_mean = np.asarray(touch_payload["mean"], dtype=np.float32)
    touch_std = np.asarray(touch_payload["std"], dtype=np.float32)
    trajectory = load_trajectory_predictor(args.trajectory_checkpoint, device=args.device)
    character = load_character_model(args.character_checkpoint, device=args.device)

    gt_characters = discover_character_samples(clean_root, raw_root)
    test_users = _test_users(gt_characters)
    gt_test = [sample for sample in gt_characters if sample.user in test_users]
    blocks = {
        (block.user, block.action, block.sample_id): block
        for block in discover_trajectory_samples(clean_root)
        if block.action in (0, 1) and sample_is_valid(block) and block.user in test_users
    }
    predicted_samples: list[CharacterSample] = []
    points_by_block: dict[tuple[str, int, str], dict[tuple[int, int], np.ndarray]] = {}
    ious: list[float] = []
    state_counts = np.zeros(len(TOUCH_LABELS), dtype=np.int64)
    gate_frames: list[int] = []
    truth_frames: list[int] = []
    gt_identities: set[str] = set()
    predicted_identities: set[str] = set()
    for key, block in blocks.items():
        x, _, truth_mask, _, timestamp = load_sample_arrays(block, mmap=False)
        gate, labels = _predict_gate(np.asarray(x), touch_model, touch_mean, touch_std, args.device, touch_config.window_frames, args.touch_batch_size)
        valid = labels[labels >= 0]
        state_counts += np.bincount(valid, minlength=len(TOUCH_LABELS))
        truth = np.asarray(truth_mask) > 0
        union = int(np.logical_or(gate, truth).sum())
        ious.append(float(np.logical_and(gate, truth).sum() / union) if union else 1.0)
        gate_frames.append(int(gate.sum()))
        truth_frames.append(int(truth.sum()))
        gt_samples_for_block, gt_ids_for_block = _samples_from_gate(block, raw_root, truth, np.asarray(timestamp))
        gt_identities.update(gt_ids_for_block)
        predicted_for_block, predicted_ids_for_block = _samples_from_gate(block, raw_root, gate, np.asarray(timestamp))
        for sample, identity in zip(predicted_for_block, predicted_ids_for_block):
            if identity in gt_ids_for_block:
                predicted_samples.append(sample)
                predicted_identities.add(identity)
        points_by_block[key] = predict_run_points(trajectory, np.asarray(x), gate, args.trajectory_chunk_len)

    predicted_examples = build_recon_examples(predicted_samples, points_by_block)
    metrics = evaluate_examples(character, predicted_examples, args.device)
    result = {
        "protocol": "official clean IMU -> current causal PaperTouchDecoder gate -> fixed b2u trajectory model -> fixed user_both_ud character CNN; no GT mask used downstream; user-disjoint test",
        "clean_root": str(clean_root),
        "raw_root": str(raw_root),
        "checkpoints": {"touch": str(args.touch_checkpoint), "trajectory": str(args.trajectory_checkpoint), "character": str(args.character_checkpoint)},
        "test_users": sorted(test_users, key=user_number),
        "test_blocks": len(blocks),
        "gt_character_samples": len(gt_identities),
        "predicted_gate_character_samples": len(predicted_samples),
        "character_coverage_vs_gt": float(len(predicted_identities) / len(gt_identities)) if gt_identities else 0.0,
        "touch_gate": {
            "mean_iou_vs_gt_mask": float(np.mean(ious)) if ious else 0.0,
            "median_iou_vs_gt_mask": float(np.median(ious)) if ious else 0.0,
            "mean_gt_contact_frames": float(np.mean(truth_frames)) if truth_frames else 0.0,
            "mean_decoder_gate_frames": float(np.mean(gate_frames)) if gate_frames else 0.0,
            "window_prediction_counts": dict(zip(TOUCH_LABELS, (int(value) for value in state_counts))),
        },
        "letter_metrics_on_predicted_segments": metrics,
        "end_to_end_top1_with_missing_as_wrong": float(metrics["top1"] * len(predicted_identities) / len(gt_identities)) if gt_identities and np.isfinite(metrics["top1"]) else 0.0,
        "end_to_end_top3_with_missing_as_wrong": float(metrics["top3"] * len(predicted_identities) / len(gt_identities)) if gt_identities and np.isfinite(metrics["top3"]) else 0.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "official_touch_to_letter_report.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved report: {report}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official data: current touch gate through trajectory-to-letter recognition")
    parser.add_argument("--clean-root", type=Path, default=Path("/data/fan/writingRing/upstream_writingring_author_release/clean_data_delete_g/data"))
    parser.add_argument("--raw-root", type=Path, default=Path("/data/fan/writingRing/upstream_writingring_author_release/data"))
    parser.add_argument("--touch-checkpoint", type=Path, default=Path("models/exp_50/paper_touch_resnet.pt"))
    parser.add_argument("--trajectory-checkpoint", type=Path, default=Path("models_trajectory_topology/b2u/paper_trajectory.pt"))
    parser.add_argument("--character-checkpoint", type=Path, default=Path("models_character/user_both_ud/character_cnn.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/official_touch_to_letter"))
    parser.add_argument("--touch-batch-size", type=int, default=4096)
    parser.add_argument("--trajectory-chunk-len", type=int, default=15000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
