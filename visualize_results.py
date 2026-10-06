"""Render result figures for the WritingRing reproduction.

Produces (into ``--output-dir``):

- ``fig_trajectory_examples.png``   reconstructed vs ground-truth test segments
- ``fig_trajectory_quality.png``    per-segment error statistics vs paper / noise floor
- ``fig_character_recognition.png`` confusion matrix + accuracy summary
- ``fig_character_examples.png``    letter trajectories with prediction
- ``fig_word_recognition.png``      word-level accuracy vs baselines and paper
- ``fig_word_examples.png``         word trajectories with the decoded text

Run from the repository parent as ``python -m Ring.visualize_results`` (or via
the documented ``writing_state`` symlink).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .character_dataset import build_gt_examples, discover_character_samples
from .character_model import load_character_model
from .paper_trajectory import (
    integrate_contact_segments,
    load_trajectory_predictor,
    predict_sequence,
    trajectory_metrics,
)
from .paper_trajectory_dataset import (
    discover_trajectory_samples,
    load_sample_arrays,
    sample_is_valid,
)
from .train_character_classifier import (
    build_recon_examples,
    character_split,
    predict_run_points,
)
from .train_paper_trajectory import random_split_samples, set_seed
from .word_ctc import greedy_ctc_decode, load_word_ctc, snap_to_vocabulary
from .word_dataset import (
    WORD_ACTIONS,
    build_word_examples,
    build_word_recon_examples,
    discover_word_samples,
)


REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_CLEAN_ROOT = Path("/data/huyang/datasets/WritingRing/clean_data_delete_g/data")
DEFAULT_RAW_ROOT = Path("/data/huyang/datasets/WritingRing/data")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/figures"

TRAJECTORY_MODELS = {
    "v1": REPO_ROOT / "models_trajectory_topology/v1/paper_trajectory.pt",
    "v4": REPO_ROOT / "models_trajectory_topology/v4/paper_trajectory.pt",
    "b2": REPO_ROOT / "models_trajectory_topology/b2/paper_trajectory.pt",
    "x3": REPO_ROOT / "models_trajectory_topology/x3/paper_trajectory.pt",
}
ENSEMBLE_MEMBERS = ("v1", "x3", "b2", "v4")

CHARACTER_CHECKPOINTS = {
    "user_gt": REPO_ROOT / "models_character/user_gt/character_cnn.pt",
    "user_both": REPO_ROOT / "models_character/user_both/character_cnn.pt",
    "user_both_ud": REPO_ROOT / "models_character/user_both_ud/character_cnn.pt",
    "random_both": REPO_ROOT / "models_character/random_both/character_cnn.pt",
}
WORD_CHECKPOINTS = {
    "user_both": REPO_ROOT / "models_words/ctc_user_both/word_ctc.pt",
    "user_both_ud": REPO_ROOT / "models_words/ctc_user_both_ud/word_ctc.pt",
    "user_gt": REPO_ROOT / "models_words/ctc_user_gt/word_ctc.pt",
    "random_both": REPO_ROOT / "models_words/ctc_random_both/word_ctc.pt",
}
WORD_REPORTS = {
    "baseline_user": REPO_ROOT / "models_words/user/word_report.json",
    "ctc_user_gt": REPO_ROOT / "models_words/ctc_user_gt/word_ctc_report.json",
    "ctc_user_both": REPO_ROOT / "models_words/ctc_user_both/word_ctc_report.json",
    "ctc_user_both_ud": REPO_ROOT / "models_words/ctc_user_both_ud/word_ctc_report.json",
    "ctc_random_both": REPO_ROOT / "models_words/ctc_random_both/word_ctc_report.json",
}
RECONSTRUCTION_CHECKPOINT = REPO_ROOT / "models_trajectory_topology/x3/paper_trajectory.pt"
# Strict protocol: front-end never saw the downstream test users (16-20).
STRICT_RECONSTRUCTION_CHECKPOINT = REPO_ROOT / "models_trajectory_topology/b2u/paper_trajectory.pt"
STRICT_CHARACTER_CHECKPOINT = CHARACTER_CHECKPOINTS["user_both_ud"]
STRICT_WORD_CHECKPOINT = WORD_CHECKPOINTS["user_both_ud"]
STRICT_WORD_REPORT_KEY = "ctc_user_both_ud"

FRONT_END_NOTE = {
    "default": "recon front-end trained on a random split (upper bound)",
    "strict": "recon front-end trained user-disjoint (never saw these users)",
}



# Paper reference numbers (He et al., CHI 2025) as recorded in .trae/documents/NOTES.md.
PAPER = {
    "trajectory_normalized": 0.073,
    "trajectory_mm": 1.64,
    "trajectory_frac_under_0.1": 0.80,
    "noise_floor_normalized": 0.0377,
    "noise_floor_mm": 1.28,
    "letter_top1": 0.887,
    "word_unconnected": 0.682,
    "word_connected": 0.531,
    "word_3000_unconnected": 0.844,
    "word_3000_connected": 0.742,
}

GT_COLOR = "#8c8c8c"
OURS_COLOR = "#1f77b4"
HIT_COLOR = "#2ca02c"
MISS_COLOR = "#d62728"
PAPER_COLOR = "#ff7f0e"
FLOOR_COLOR = "#7f7f7f"


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _nan_mean(stack: np.ndarray) -> np.ndarray:
    """NaN-aware mean over the leading axis, without all-NaN warnings."""

    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[count == 0] = np.nan
    return mean


def _predict_deltas(predictor, x: np.ndarray) -> np.ndarray:
    values = (np.asarray(x, dtype=np.float32) - predictor.mean) / predictor.std
    return predict_sequence(predictor.model, values, chunk_len=None)


# ---------------------------------------------------------------------------
# trajectory reconstruction
# ---------------------------------------------------------------------------


@dataclass
class SegmentRecord:
    sample_key: str
    start: int
    end: int
    gt: np.ndarray
    pred: np.ndarray
    normalized_error: np.ndarray
    mm_error: np.ndarray


def _segment_records(
    deltas: np.ndarray,
    board: np.ndarray,
    mask: np.ndarray,
    mm_scale: Sequence[float] | None,
    sample_key: str,
    keep_geometry: bool,
) -> list[SegmentRecord]:
    records: list[SegmentRecord] = []
    for segment in integrate_contact_segments(deltas, mask, min_frames=5):
        ground_truth = (
            np.asarray(board[segment.start : segment.end], dtype=np.float64)
            - np.asarray(board[segment.start], dtype=np.float64)
        )
        metrics = trajectory_metrics(segment.xy, ground_truth, mm_scale=mm_scale)
        if metrics["skipped_degenerate"]:
            continue
        difference = np.asarray(segment.xy, dtype=np.float64) - ground_truth
        span = ground_truth.max(axis=0) - ground_truth.min(axis=0)
        diagonal = float(np.linalg.norm(span))
        normalized = np.linalg.norm(difference, axis=1) / diagonal
        mm = (
            np.linalg.norm(difference * np.asarray(mm_scale, dtype=np.float64), axis=1)
            if mm_scale
            else np.full(len(ground_truth), np.nan)
        )
        records.append(
            SegmentRecord(
                sample_key=sample_key,
                start=segment.start,
                end=segment.end,
                gt=ground_truth.astype(np.float32) if keep_geometry else np.zeros((0, 2), np.float32),
                pred=np.asarray(segment.xy, dtype=np.float32) if keep_geometry else np.zeros((0, 2), np.float32),
                normalized_error=normalized,
                mm_error=mm,
            )
        )
    return records


def _aggregate(records: Sequence[SegmentRecord]) -> dict[str, float]:
    if not records:
        return {}
    point_values = np.concatenate([record.normalized_error for record in records])
    mm_values = np.concatenate([record.mm_error for record in records])
    segment_means = np.asarray([record.normalized_error.mean() for record in records])
    return {
        "segments": len(records),
        "points": int(len(point_values)),
        "normalized_mean": float(point_values.mean()),
        "normalized_p50": float(np.percentile(point_values, 50)),
        "normalized_p90": float(np.percentile(point_values, 90)),
        "mm_mean": float(np.nanmean(mm_values)),
        "segment_mean": float(segment_means.mean()),
        "segment_median": float(np.median(segment_means)),
        "frac_points_under_0.1": float((point_values < 0.1).mean()),
        "frac_segments_under_0.1": float((segment_means < 0.1).mean()),
    }


def collect_trajectory_results(clean_root: Path, device: str, log=print) -> dict[str, Any]:
    samples = [
        sample
        for sample in discover_trajectory_samples(clean_root)
        if sample_is_valid(sample)
    ]
    _, _, test_samples, split_info = random_split_samples(samples, seed=42)
    log(f"[trajectory] test samples: {len(test_samples)} (split={split_info['strategy']})")

    predictors = {
        name: load_trajectory_predictor(path, device=device)
        for name, path in TRAJECTORY_MODELS.items()
    }
    mm_scale = predictors["x3"].board_mm_scale or (240.0, 169.5)
    for name, predictor in predictors.items():
        if not np.allclose(predictor.mean, predictors["x3"].mean, atol=1e-4):
            log(f"[trajectory] WARNING: {name} uses different normalization")

    model_names = list(predictors) + ["ensemble"]
    records_by_model: dict[str, list[SegmentRecord]] = {name: [] for name in model_names}
    segment_means: dict[str, list[float]] = {name: [] for name in model_names}
    segment_lengths: dict[str, list[tuple[float, float]]] = {name: [] for name in model_names}

    for count, sample in enumerate(test_samples, start=1):
        x, board, mask, _, _ = load_sample_arrays(sample, mmap=False)
        key = f"{sample.user}/{sample.action}/{sample.sample_id}"
        deltas = {
            name: _predict_deltas(predictor, x)
            for name, predictor in predictors.items()
        }
        deltas["ensemble"] = _nan_mean(
            np.stack([deltas[name] for name in ENSEMBLE_MEMBERS], axis=0)
        )
        for name in model_names:
            keep_geometry = name == "x3"
            records = _segment_records(
                deltas[name], board, mask, mm_scale, key, keep_geometry
            )
            records_by_model[name].extend(records)
            segment_means[name].extend(
                record.normalized_error.mean() for record in records
            )
            segment_lengths[name].extend(
                (
                    float(np.linalg.norm(np.diff(record.gt, axis=0), axis=1).sum()),
                    float(np.linalg.norm(np.diff(record.pred, axis=0), axis=1).sum()),
                )
                for record in records
                if len(record.gt)
            )
        if count % 20 == 0:
            log(f"[trajectory] {count}/{len(test_samples)} samples")

    result = {
        "split": split_info,
        "test_samples": len(test_samples),
        "models": {
            name: {
                "metrics": _aggregate(records_by_model[name]),
                "segment_errors": segment_means[name],
                "segment_lengths": segment_lengths[name],
            }
            for name in model_names
        },
        "x3_records": records_by_model["x3"],
    }
    for name in model_names:
        metrics = result["models"][name]["metrics"]
        log(
            f"[trajectory] {name:9s} norm={metrics['normalized_mean']:.4f} "
            f"mm={metrics['mm_mean']:.2f} segments={metrics['segments']} "
            f"seg<0.1={metrics['frac_segments_under_0.1']:.3f} "
            f"pt<0.1={metrics['frac_points_under_0.1']:.3f}"
        )
    return result


def draw_trajectory_examples(trajectory: Mapping[str, Any], output_dir: Path) -> Path:
    plt = _load_matplotlib()
    records = [record for record in trajectory["x3_records"] if len(record.gt) >= 8]
    errors = np.asarray([record.normalized_error.mean() for record in records])
    order = np.argsort(errors)
    picks = [order[int(round((len(order) - 1) * q))] for q in np.linspace(0.02, 0.98, 8)]

    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.4))
    for axis, index in zip(axes.ravel(), picks):
        record = records[index]
        axis.plot(record.gt[:, 0], record.gt[:, 1], color=GT_COLOR, linestyle="--", lw=2.4, label="Ground truth")
        axis.plot(record.pred[:, 0], record.pred[:, 1], color=OURS_COLOR, lw=1.7, label="Reconstructed")
        axis.invert_yaxis()
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"{record.sample_key}\nmean err={record.normalized_error.mean():.3f}"
            f"  p50={np.median(record.normalized_error):.3f}",
            fontsize=8,
        )
    axes[0, 0].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Trajectory reconstruction — random-split test segments (X3 model)\n"
        "dashed: touchpad ground truth; solid: integrated IMU prediction "
        "(error normalised by GT bounding-box diagonal)",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = output_dir / "fig_trajectory_examples.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def draw_trajectory_quality(trajectory: Mapping[str, Any], output_dir: Path) -> Path:
    plt = _load_matplotlib()
    models = trajectory["models"]
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))

    axis = axes[0, 0]
    errors = np.asarray(models["x3"]["segment_errors"])
    axis.hist(np.clip(errors, 0, 0.45), bins=np.linspace(0, 0.45, 60), color=OURS_COLOR, alpha=0.75)
    axis.axvline(np.median(errors), color=OURS_COLOR, lw=1.6, label=f"X3 median {np.median(errors):.3f}")
    axis.axvline(PAPER["noise_floor_normalized"], color=FLOOR_COLOR, lw=1.6, linestyle=":", label="protocol noise floor 0.038")
    axis.axvline(PAPER["trajectory_normalized"], color=PAPER_COLOR, lw=1.6, linestyle="--", label="paper mean 0.073")
    axis.set_xlabel("per-segment mean normalised error")
    axis.set_ylabel("segments")
    axis.set_title(f"Per-segment error (X3, {len(errors)} test segments, x-clipped at 0.45)")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    pairs = np.asarray(models["x3"]["segment_lengths"])
    lengths, predicted = pairs[:, 0], pairs[:, 1]
    axis.scatter(lengths, predicted, s=6, alpha=0.35, color=OURS_COLOR, edgecolors="none")
    limit = float(np.nanpercentile(np.concatenate([lengths, predicted]), 99))
    axis.plot([0, limit], [0, limit], color=GT_COLOR, linestyle="--", lw=1.3, label="y = x")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("ground-truth path length (normalised units)")
    axis.set_ylabel("reconstructed path length")
    ratio = float(np.median(predicted / np.maximum(lengths, 1e-9)))
    correlation = float(np.corrcoef(lengths, predicted)[0, 1])
    axis.set_title(f"Path-length fidelity (median ratio {ratio:.2f}, r={correlation:.3f})")
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    for name in ["v4", "b2", "x3", "ensemble"]:
        means = np.sort(np.asarray(models[name]["segment_errors"]))
        cumulative = np.arange(1, len(means) + 1) / len(means)
        axis.plot(np.clip(means, 0, 0.5), cumulative, lw=1.9, label=name)
    axis.axvline(0.1, color=FLOOR_COLOR, linestyle=":", lw=1.4)
    axis.text(0.103, 0.05, "0.1", color=FLOOR_COLOR, fontsize=8)
    axis.set_xlim(0, 0.5)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("per-segment mean normalised error")
    axis.set_ylabel("fraction of segments ≤ x")
    axis.set_title("Error CDF (paper: 80% of trajectories below 0.1)")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    order = ["v4", "b2", "x3", "ensemble"]
    labels = ["V4\nstreaming", "B2\nbi-LSTM", "X3\nbest single", "Ensemble\n4 models", "Paper\nCHI 2025", "Noise\nfloor"]
    values = [models[name]["metrics"]["normalized_mean"] for name in order]
    mm = [models[name]["metrics"]["mm_mean"] for name in order]
    values += [PAPER["trajectory_normalized"], PAPER["noise_floor_normalized"]]
    mm += [PAPER["trajectory_mm"], PAPER["noise_floor_mm"]]
    colors = [OURS_COLOR] * 4 + [PAPER_COLOR, FLOOR_COLOR]
    bars = axis.bar(labels, values, color=colors, alpha=0.9)
    for bar, value, mm_value in zip(bars, values, mm):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.3f}\n{mm_value:.2f} mm",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    axis.set_ylabel("mean normalised error (lower is better)")
    axis.set_ylim(0, max(values) * 1.35)
    axis.set_title("Normalised error vs paper (random-split test)")

    figure.suptitle(
        f"Trajectory reconstruction quality — {trajectory['test_samples']} random-split test samples, "
        f"{models['x3']['metrics']['segments']} contact segments",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = output_dir / "fig_trajectory_quality.png"
    figure.savefig(path)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# character recognition
# ---------------------------------------------------------------------------


def collect_character_results(
    clean_root: Path,
    raw_root: Path,
    device: str,
    log=print,
    recon_checkpoint: Path | None = None,
    character_checkpoint: Path | None = None,
) -> dict[str, Any]:
    recon_path = Path(recon_checkpoint) if recon_checkpoint else RECONSTRUCTION_CHECKPOINT
    model_path = (
        Path(character_checkpoint)
        if character_checkpoint
        else CHARACTER_CHECKPOINTS["user_both"]
    )
    samples = discover_character_samples(clean_root, raw_root)
    train_samples, _, test_samples, split_info = character_split(samples, "user")
    log(f"[character] letters: train={len(train_samples)} test={len(test_samples)}")

    predictor = load_trajectory_predictor(recon_path, device=device)
    blocks = {
        (block.user, block.action, block.sample_id): block
        for block in discover_trajectory_samples(clean_root)
        if block.action in (0, 1) and sample_is_valid(block)
    }
    needed = {
        (sample.user, sample.action, sample.sample_id)
        for sample in train_samples + test_samples
    }
    run_points: dict[tuple[str, int, str], dict[tuple[int, int], np.ndarray]] = {}
    for key in needed:
        x, _, mask, _, _ = load_sample_arrays(blocks[key], mmap=False)
        run_points[key] = predict_run_points(predictor, np.asarray(x), np.asarray(mask))
    log(f"[character] reconstructed {len(run_points)} blocks with {recon_path.parent.name}")

    recon_test = build_recon_examples(test_samples, run_points)
    for example in recon_test:
        example["source"] = "recon"
    log(f"[character] recon test examples: {len(recon_test)}")

    model = load_character_model(model_path, device=device)
    trajectories = torch.from_numpy(
        np.stack([example["trajectory"] for example in recon_test]).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        logits = model(trajectories).cpu().numpy()
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = np.asarray([example["label"] for example in recon_test])
    predictions = np.argmax(logits, axis=1)
    top3 = np.argsort(-logits, axis=1)[:, :3]
    confusion = np.zeros((26, 26), dtype=np.int64)
    for true_value, predicted in zip(labels, predictions):
        confusion[true_value, predicted] += 1
    metrics = {
        "n": int(len(labels)),
        "top1": float((predictions == labels).mean()),
        "top3": float((top3 == labels[:, None]).any(axis=1).mean()),
    }
    log(
        f"[character] recon top1={metrics['top1']:.4f} "
        f"top3={metrics['top3']:.4f} (n={metrics['n']})"
    )

    gt_test = build_gt_examples(test_samples, clean_root)
    for example in gt_test:
        example["source"] = "gt"
    gt_trajectories = torch.from_numpy(
        np.stack([example["trajectory"] for example in gt_test]).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        gt_logits = model(gt_trajectories).cpu().numpy()
    gt_labels = np.asarray([example["label"] for example in gt_test])
    gt_predictions = np.argmax(gt_logits, axis=1)
    gt_top3 = np.argsort(-gt_logits, axis=1)[:, :3]
    gt_metrics = {
        "n": int(len(gt_labels)),
        "top1": float((gt_predictions == gt_labels).mean()),
        "top3": float((gt_top3 == gt_labels[:, None]).any(axis=1).mean()),
    }
    log(
        f"[character] gt    top1={gt_metrics['top1']:.4f} "
        f"top3={gt_metrics['top3']:.4f} (n={gt_metrics['n']})"
    )

    per_class: dict[str, Any] = {}
    for index in range(26):
        letter = chr(ord("A") + index)
        row: dict[str, Any] = {}
        for name, labels_array, preds, topk in (
            ("gt", gt_labels, gt_predictions, gt_top3),
            ("recon", labels, predictions, top3),
        ):
            selected = labels_array == index
            count = int(selected.sum())
            row[name] = {
                "n": count,
                "top1": float((preds[selected] == index).mean()) if count else float("nan"),
                "top3": float((topk[selected] == index).any(axis=1).mean())
                if count
                else float("nan"),
            }
        per_class[letter] = row

    reports = {}
    for name, path in CHARACTER_CHECKPOINTS.items():
        report_path = path.parent / "character_report.json"
        if report_path.exists():
            reports[name] = json.loads(report_path.read_text())
    return {
        "split": split_info,
        "test_samples": len(test_samples),
        "metrics": metrics,
        "gt_metrics": gt_metrics,
        "per_class": per_class,
        "confusion": confusion,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "examples": recon_test,
        "reports": reports,
    }


def draw_character_recognition(
    character: Mapping[str, Any], output_dir: Path, protocol: str = "default"
) -> Path:
    plt = _load_matplotlib()
    confusion = character["confusion"]
    row_sums = np.maximum(confusion.sum(axis=1), 1)
    normalized = confusion / row_sums[:, None]
    labels = [chr(ord("A") + index) for index in range(26)]

    figure = plt.figure(figsize=(13.4, 8.6))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.85])

    axis = figure.add_subplot(grid[0, 0])
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(26), labels, fontsize=6)
    axis.set_yticks(range(26), labels, fontsize=6)
    axis.set_xlabel("prediction")
    axis.set_ylabel("ground truth")
    axis.set_title(
        f"Confusion matrix — reconstructed letters "
        f"(n={character['metrics']['n']}, row-normalised)"
    )
    for i in range(26):
        for j in range(26):
            if i != j and normalized[i, j] >= 0.06:
                axis.text(
                    j,
                    i,
                    f"{normalized[i, j] * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#333333",
                )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)

    axis = figure.add_subplot(grid[0, 1])
    accuracy = np.diag(normalized)
    colors = [
        HIT_COLOR if value >= 0.9 else (PAPER_COLOR if value >= 0.8 else MISS_COLOR)
        for value in accuracy
    ]
    axis.bar(labels, accuracy * 100, color=colors)
    axis.set_ylim(0, 108)
    axis.set_ylabel("per-class top-1 (%)")
    axis.set_title("Per-letter accuracy (reconstructed trajectories)")
    for index, value in enumerate(accuracy):
        if value < 0.9:
            axis.text(index, value * 100 + 2, f"{value * 100:.0f}", ha="center", fontsize=6)

    axis = figure.add_subplot(grid[1, :])
    reports = character["reports"]
    values = []
    names = []
    if "user_gt" in reports:
        values.append(reports["user_gt"]["test"]["gt"]["top1"] * 100)
        names.append("GT-trained\n(GT eval)")
    if "user_both" in reports:
        values.append(reports["user_both"]["test"]["gt"]["top1"] * 100)
        names.append("GT+recon trained\n(GT eval)")
    values.append(character["metrics"]["top1"] * 100)
    names.append("This run\n(recon eval)")
    values.append(PAPER["letter_top1"] * 100)
    names.append("Paper\n(Google IME)")
    colors = [OURS_COLOR, OURS_COLOR, HIT_COLOR, PAPER_COLOR]
    bars = axis.bar(names, values, color=colors[: len(values)], alpha=0.92)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}%", ha="center", fontsize=9
        )
    axis.set_ylim(0, 112)
    axis.set_ylabel("letter top-1 accuracy (%)")
    axis.set_title(
        "Letter accuracy — user-disjoint split (unseen users 16/18/19/20); "
        f"{FRONT_END_NOTE.get(protocol, FRONT_END_NOTE['default'])}.\n"
        "The paper decodes with an external IME, so protocols are not directly comparable."
    )

    figure.suptitle(
        "Character recognition — 26-class letter classifier on IMU-reconstructed trajectories",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = output_dir / "fig_character_recognition.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def draw_character_examples(character: Mapping[str, Any], output_dir: Path) -> Path:
    plt = _load_matplotlib()
    examples = character["examples"]
    labels = character["labels"]
    predictions = character["predictions"]
    probabilities = character["probabilities"]
    rng = np.random.default_rng(42)

    wrong = np.where(predictions != labels)[0]
    correct = np.where(predictions == labels)[0]
    wrong = wrong[np.argsort(-probabilities[wrong, predictions[wrong]])][:12]
    if len(correct) > 12:
        correct = rng.choice(correct, size=12, replace=False)
    elif len(correct) < 12:
        correct = np.concatenate(
            [correct, rng.choice(correct, size=12 - len(correct), replace=False)]
        )
    picks = list(wrong) + list(correct)
    if len(picks) < 24:
        extra = [index for index in range(len(examples)) if index not in picks]
        picks += list(rng.choice(extra, size=min(24 - len(picks), len(extra)), replace=False))
    picks = picks[:24]
    rng.shuffle(picks)

    figure, axes = plt.subplots(4, 6, figsize=(13.0, 8.4))
    for axis, index in zip(axes.ravel(), picks):
        trajectory = examples[index]["trajectory"]
        hit = predictions[index] == labels[index]
        color = HIT_COLOR if hit else MISS_COLOR
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, lw=1.8)
        axis.scatter(trajectory[0, 0], trajectory[0, 1], s=14, color=color, zorder=3)
        axis.invert_yaxis()
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xticks([])
        axis.set_yticks([])
        true_char = chr(ord("A") + int(labels[index]))
        pred_char = chr(ord("A") + int(predictions[index]))
        axis.set_title(
            f"{true_char} → {pred_char}  {probabilities[index, predictions[index]]:.2f}",
            fontsize=8,
            color=color,
        )
    figure.suptitle(
        "Letter recognition examples — trajectories reconstructed end-to-end from IMU only\n"
        "green: correct (12 sampled)   red: highest-confidence misclassifications (12)   "
        "dot marks the pen-down point",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path = output_dir / "fig_character_examples.png"
    figure.savefig(path)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# word recognition
# ---------------------------------------------------------------------------


def collect_word_results(
    clean_root: Path,
    raw_root: Path,
    device: str,
    log=print,
    recon_checkpoint: Path | None = None,
    word_checkpoint: Path | None = None,
) -> dict[str, Any]:
    recon_path = Path(recon_checkpoint) if recon_checkpoint else RECONSTRUCTION_CHECKPOINT
    model_path = (
        Path(word_checkpoint) if word_checkpoint else WORD_CHECKPOINTS["user_both"]
    )
    samples = discover_word_samples(clean_root, raw_root)
    train_samples, val_samples, test_samples, split_info = character_split(samples, "user")
    log(f"[word] words: train={len(train_samples)} test={len(test_samples)}")

    predictor = load_trajectory_predictor(recon_path, device=device)
    blocks = {
        (block.user, block.action, block.sample_id): block
        for block in discover_trajectory_samples(clean_root)
        if block.action in WORD_ACTIONS and sample_is_valid(block)
    }
    run_points: dict[tuple[str, int, str], dict[tuple[int, int], np.ndarray]] = {}
    for key in {
        (sample.user, sample.action, sample.sample_id)
        for sample in train_samples + val_samples + test_samples
    }:
        x, _, mask, _, _ = load_sample_arrays(blocks[key], mmap=False)
        run_points[key] = predict_run_points(predictor, np.asarray(x), np.asarray(mask))
    recon_test = build_word_recon_examples(test_samples, run_points)
    gt_test = build_word_examples(test_samples, clean_root)
    log(f"[word] recon test examples: {len(recon_test)} / gt {len(gt_test)}")

    payload = torch.load(str(model_path), map_location="cpu", weights_only=True)
    vocabulary = payload["vocabulary"]
    model = load_word_ctc(model_path, device=device)

    trajectories = torch.from_numpy(
        np.stack([example["trajectory"] for example in recon_test]).astype(np.float32)
    ).to(device)
    decoded: list[str] = []
    with torch.no_grad():
        for start in range(0, len(recon_test), 256):
            logits = model(trajectories[start : start + 256]).cpu().numpy()
            for index in range(logits.shape[1]):
                decoded.append(greedy_ctc_decode(logits[:, index, :]))
    raw_hits = [word == example["word"] for word, example in zip(decoded, recon_test)]
    log(f"[word] raw top1 on recon test: {np.mean(raw_hits):.4f}")

    reports = {}
    for name, path in WORD_REPORTS.items():
        if path.exists():
            reports[name] = json.loads(path.read_text())
    return {
        "split": split_info,
        "test_samples": len(test_samples),
        "recon_test": recon_test,
        "gt_test": gt_test,
        "decoded": decoded,
        "raw_hits": raw_hits,
        "raw_top1": float(np.mean(raw_hits)) if raw_hits else float("nan"),
        "vocabulary": vocabulary,
        "reports": reports,
    }


def draw_word_recognition(
    word: Mapping[str, Any], output_dir: Path, report_key: str = "ctc_user_both",
    protocol: str = "default",
) -> Path:
    plt = _load_matplotlib()
    reports = word["reports"]
    ctc = reports[report_key]["test"]
    # Headline word numbers quote the reconstructed-trajectory sub-block
    # (GT+recon 合并口径见 ctc 顶层；报告 JSON 的 recon 子块与页面文案一致).
    ctc_recon = ctc.get("recon") or ctc
    base = reports["baseline_user"]["test"]
    random_ctc = reports["ctc_random_both"]["test"]

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), width_ratios=[1.25, 1.0])

    axis = axes[0]
    groups = ["all words", "connected", "unconnected"]
    baseline = [
        base["gt"]["closed"]["top1"],
        base["gt"]["connected"]["top1"],
        base["gt"]["unconnected"]["top1"],
    ]
    ctc_snapped = [
        ctc_recon["snapped_top1"],
        ctc_recon["connected"]["snapped_top1"],
        ctc_recon["unconnected"]["snapped_top1"],
    ]
    x = np.arange(len(groups))
    width = 0.34
    bars_a = axis.bar(x - width / 2, np.asarray(baseline) * 100, width, label="nearest instance (GT traj)")
    bars_b = axis.bar(x + width / 2, np.asarray(ctc_snapped) * 100, width, label="CTC + vocabulary (recon subset)")
    for bars in (bars_a, bars_b):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{bar.get_height():.1f}",
                ha="center",
                fontsize=7.5,
            )
    axis.set_xticks(x, groups)
    axis.set_ylabel("word top-1 accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title(
        f"Ours — user-disjoint label split, {ctc_recon['n']} test words; "
        f"{FRONT_END_NOTE.get(protocol, FRONT_END_NOTE['default'])}, "
        f"{ctc['vocabulary_size']}-word vocabulary\n"
        f"raw CTC before snapping (recon subset): {ctc_recon['raw_top1'] * 100:.1f}%",
        fontsize=9.0,
    )
    axis.set_xlabel(
        "nearest-instance 'all words' number is closed-set; open-vocabulary words score 0 in that protocol",
        fontsize=7,
    )
    axis.axhline(PAPER["word_unconnected"] * 100, color=PAPER_COLOR, linestyle="--", lw=1.2, label="paper unconnected 68.2")
    axis.axhline(PAPER["word_connected"] * 100, color=PAPER_COLOR, linestyle=":", lw=1.2, label="paper connected 53.1")
    axis.legend(fontsize=7.5, loc="upper left")

    axis = axes[1]
    labels = ["CTC random split\n(recon, snapped)", "Paper IME\nopen vocabulary", "Paper IME\n3000-word vocab"]
    connected = [
        random_ctc["connected"]["snapped_top1"] * 100,
        PAPER["word_connected"] * 100,
        PAPER["word_3000_connected"] * 100,
    ]
    unconnected = [
        random_ctc["unconnected"]["snapped_top1"] * 100,
        PAPER["word_unconnected"] * 100,
        PAPER["word_3000_unconnected"] * 100,
    ]
    x = np.arange(len(labels))
    bars_a = axis.bar(x - 0.18, connected, 0.34, label="connected", color=MISS_COLOR, alpha=0.85)
    bars_b = axis.bar(x + 0.18, unconnected, 0.34, label="unconnected", color=HIT_COLOR, alpha=0.85)
    for bars in (bars_a, bars_b):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{bar.get_height():.0f}",
                ha="center",
                fontsize=7.5,
            )
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylim(0, 100)
    axis.set_ylabel("word top-1 accuracy (%)")
    axis.set_title("Reference points — protocols differ\n(paper decoder uses an IME language model)")
    axis.legend(fontsize=8)

    figure.suptitle(
        "Word recognition — CTC letter-sequence decoder + edit-distance vocabulary snap",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = output_dir / "fig_word_recognition.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def draw_word_examples(word: Mapping[str, Any], output_dir: Path) -> Path:
    plt = _load_matplotlib()
    examples = word["recon_test"]
    decoded = word["decoded"]
    vocabulary = word["vocabulary"]

    def pick(indices, group, count):
        chosen = []
        for index in indices:
            if len(chosen) >= count:
                break
            if examples[index]["group"] == group:
                chosen.append(index)
        return chosen

    hits = [index for index, example in enumerate(examples) if decoded[index] == example["word"]]
    misses = [index for index, example in enumerate(examples) if decoded[index] != example["word"]]
    selection = (
        pick(hits, "connected", 2)
        + pick(hits, "unconnected", 2)
        + pick(misses, "connected", 3)
        + pick(misses, "unconnected", 1)
    )
    if len(selection) < 8:
        selection += [
            index
            for index in range(len(examples))
            if index not in selection
        ][: 8 - len(selection)]

    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.4))
    for axis, index in zip(axes.ravel(), selection):
        example = examples[index]
        trajectory = example["trajectory"]
        snapped = snap_to_vocabulary(decoded[index], vocabulary)
        hit = snapped == example["word"]
        color = HIT_COLOR if hit else MISS_COLOR
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, lw=2.0)
        axis.scatter(trajectory[0, 0], trajectory[0, 1], s=16, color=color, zorder=3)
        axis.invert_yaxis()
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xticks([])
        axis.set_yticks([])
        title = (
            f'GT: "{example["word"]}"  ({example["group"]})\n'
            f'CTC: "{snapped}"'
        )
        title += "  ✓" if hit else "  ✗"
        if snapped != decoded[index]:
            title += f'\nraw: "{decoded[index]}"'
        axis.set_title(title, fontsize=8.5, color=color)

    figure.suptitle(
        "Word recognition examples — trajectories reconstructed end-to-end from the ring IMU\n"
        "green: correct after vocabulary alignment   red: wrong   dot marks the pen-down point",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path = output_dir / "fig_word_examples.png"
    figure.savefig(path)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------



def draw_letter_accuracies(
    character: Mapping[str, Any], output_dir: Path, protocol: str = "default"
) -> tuple[Path, Path]:
    """Per-letter top-1/top-3 table (GT and reconstructed) with an annotated bar chart."""

    plt = _load_matplotlib()
    per_class = character["per_class"]
    letters = [chr(ord("A") + index) for index in range(26)]
    gt_rates = np.asarray([per_class[letter]["gt"]["top1"] for letter in letters]) * 100
    recon_rates = np.asarray([per_class[letter]["recon"]["top1"] for letter in letters]) * 100
    counts = [per_class[letter]["recon"]["n"] for letter in letters]

    figure, axis = plt.subplots(figsize=(13.5, 5.6))
    positions = np.arange(26)
    width = 0.38
    axis.bar(positions - width / 2, gt_rates, width, label="GT trajectory", color=OURS_COLOR, alpha=0.9)
    axis.bar(positions + width / 2, recon_rates, width, label="reconstructed trajectory", color=HIT_COLOR, alpha=0.9)
    for position, gt_rate, recon_rate, count in zip(positions, gt_rates, recon_rates, counts):
        axis.text(position - width / 2 - 0.02, gt_rate + 1.2, f"{gt_rate:.0f}", ha="center", fontsize=5.5)
        axis.text(position + width / 2 + 0.02, recon_rate + 1.2, f"{recon_rate:.0f}", ha="center", fontsize=5.5)
        axis.text(position, -7.5, f"n={count}", ha="center", fontsize=5.4, color="#555555")
    axis.set_xticks(positions, letters)
    axis.set_ylim(0, 116)
    axis.set_ylabel("top-1 accuracy per letter (%)")
    axis.set_title(
        "Per-letter recognition — user-disjoint classifier test, 26-class CNN "
        f"(n={character['metrics']['n']}; {FRONT_END_NOTE.get(protocol, FRONT_END_NOTE['default'])})"
    )
    axis.legend(fontsize=9, loc="lower right")
    figure.tight_layout()
    figure_path = output_dir / "fig_letter_accuracies.png"
    figure.savefig(figure_path)
    plt.close(figure)

    csv_path = output_dir / "letter_accuracy.csv"
    lines = ["letter,n_gt,gt_top1,gt_top3,n_recon,recon_top1,recon_top3"]
    for letter in letters:
        row = per_class[letter]
        lines.append(
            f"{letter},{row['gt']['n']},{row['gt']['top1']:.4f},{row['gt']['top3']:.4f},"
            f"{row['recon']['n']},{row['recon']['top1']:.4f},{row['recon']['top3']:.4f}"
        )
    csv_path.write_text("\n".join(lines) + "\n")
    return figure_path, csv_path




def draw_word_examples_gt_vs_pred(word: Mapping[str, Any], output_dir: Path) -> Path:
    """Word examples as GT trajectory | reconstructed trajectory cards."""

    plt = _load_matplotlib()
    examples = word["recon_test"]
    decoded = word["decoded"]
    vocabulary = word["vocabulary"]

    def key(example):
        return (
            example["user"],
            example["action"],
            example["sample_id"],
            tuple(tuple(run) for run in example["runs"]),
        )

    gt_by_key = {key(example): example for example in word["gt_test"]}

    def pick(indices, group, count):
        chosen = []
        for index in indices:
            if len(chosen) >= count:
                break
            if examples[index]["group"] == group:
                chosen.append(index)
        return chosen

    hits = [i for i, example in enumerate(examples) if decoded[i] == example["word"]]
    misses = [i for i, example in enumerate(examples) if decoded[i] != example["word"]]
    selection = (
        pick(hits, "unconnected", 2)
        + pick(hits, "connected", 2)
        + pick(misses, "unconnected", 2)
        + pick(misses, "connected", 2)
    )
    if len(selection) < 8:
        selection += [i for i in range(len(examples)) if i not in selection][: 8 - len(selection)]
    selection = selection[:8]
    while len(selection) < 8:
        selection.append(selection[-1])

    figure, axes = plt.subplots(4, 4, figsize=(12.4, 12.6))
    for card, index in enumerate(selection):
        row, column = divmod(card, 2)
        axis_gt = axes[row, column * 2]
        axis_recon = axes[row, column * 2 + 1]
        example = examples[index]
        gt_example = gt_by_key.get(key(example))
        snapped = snap_to_vocabulary(decoded[index], vocabulary)
        hit = snapped == example["word"]
        color = HIT_COLOR if hit else MISS_COLOR
        group = example["group"]

        for axis, trajectory, line_color, linestyle in (
            (axis_gt, np.asarray(gt_example["trajectory"]) if gt_example else None, GT_COLOR, "--"),
            (axis_recon, np.asarray(example["trajectory"]), color, "-"),
        ):
            if trajectory is not None:
                axis.plot(trajectory[:, 0], trajectory[:, 1], color=line_color, linestyle=linestyle, lw=2.0)
                axis.scatter(trajectory[0, 0], trajectory[0, 1], s=14, color=line_color, zorder=3)
            axis.invert_yaxis()
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xticks([])
            axis.set_yticks([])

        axis_gt.set_title(f'GT traj · "{example["word"]}" ({group})', fontsize=8.6, color="#444444")
        title = f'Recon traj · CTC: "{snapped}"'
        axis_recon.set_title(title + ("  ✓" if hit else "  ✗"), fontsize=8.6, color=color)
        if snapped != decoded[index]:
            axis_recon.set_xlabel(f'raw decode: "{decoded[index]}"', fontsize=7)

    figure.suptitle(
        "Word test examples — ground-truth trajectory | IMU-reconstructed trajectory | true word | recognised word\n"
        "left panel of each pair: touchpad ground truth (dashed grey); right panel: reconstructed trajectory, "
        "title shows the CTC prediction (green = correct, red = wrong)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = output_dir / "fig_word_examples_gt_vs_pred.png"
    figure.savefig(path)
    plt.close(figure)
    return path



def main() -> None:
    parser = argparse.ArgumentParser(description="Render WritingRing result figures")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--figures",
        default="trajectory,character,word",
        help="comma-separated subset of: trajectory,character,word",
    )
    parser.add_argument(
        "--protocol",
        choices=("default", "strict"),
        default="default",
        help=(
            "strict = user-disjoint trajectory front-end (b2u, never saw users 16-20) "
            "with the retrained character/CTC heads"
        ),
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    set_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {name.strip() for name in args.figures.split(",") if name.strip()}
    written: list[Path] = []
    summary: dict[str, Any] = {"paper": PAPER}
    strict = args.protocol == "strict"
    recon_checkpoint = STRICT_RECONSTRUCTION_CHECKPOINT if strict else None
    character_checkpoint = STRICT_CHARACTER_CHECKPOINT if strict else None
    word_checkpoint = STRICT_WORD_CHECKPOINT if strict else None
    word_report_key = STRICT_WORD_REPORT_KEY if strict else "ctc_user_both"
    print(f"protocol: {'strict (b2u front-end)' if strict else 'default (x3 front-end)'}")

    if "trajectory" in wanted:
        trajectory = collect_trajectory_results(args.data_root, device)
        written.append(draw_trajectory_examples(trajectory, args.output_dir))
        written.append(draw_trajectory_quality(trajectory, args.output_dir))
        summary["trajectory"] = {
            "split": trajectory["split"],
            "test_samples": trajectory["test_samples"],
            "models": {
                name: payload["metrics"] for name, payload in trajectory["models"].items()
            },
        }
    if "character" in wanted:
        character = collect_character_results(
            args.data_root,
            args.raw_root,
            device,
            recon_checkpoint=recon_checkpoint,
            character_checkpoint=character_checkpoint,
        )
        written.append(draw_character_recognition(character, args.output_dir, protocol=args.protocol))
        written.append(draw_character_examples(character, args.output_dir))
        accuracy_figure, accuracy_csv = draw_letter_accuracies(
            character, args.output_dir, protocol=args.protocol
        )
        written.append(accuracy_figure)
        written.append(accuracy_csv)
        summary["character"] = {
            "protocol": args.protocol,
            "split": character["split"],
            "test_samples": character["test_samples"],
            "metrics": character["metrics"],
            "gt_metrics": character["gt_metrics"],
            "per_class": character["per_class"],
            "reports": {
                name: report["test"] for name, report in character["reports"].items()
            },
        }
    if "word" in wanted:
        word = collect_word_results(
            args.data_root,
            args.raw_root,
            device,
            recon_checkpoint=recon_checkpoint,
            word_checkpoint=word_checkpoint,
        )
        written.append(
            draw_word_recognition(
                word, args.output_dir, report_key=word_report_key, protocol=args.protocol
            )
        )
        written.append(draw_word_examples(word, args.output_dir))
        written.append(draw_word_examples_gt_vs_pred(word, args.output_dir))
        summary["word"] = {
            "protocol": args.protocol,
            "split": word["split"],
            "test_samples": word["test_samples"],
            "raw_top1_recon_recomputed": word["raw_top1"],
            "reports": {
                name: report["test"] for name, report in word["reports"].items()
            },
        }

    summary_path = args.output_dir / "metrics_summary.json"
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text())
            for key, value in previous.items():
                summary.setdefault(key, value)
        except json.JSONDecodeError:
            pass
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"saved: {summary_path}")
    for path in written:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
