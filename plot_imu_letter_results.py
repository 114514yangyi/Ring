"""Visualize direct IMU-to-letter predictions on the 200 Hz test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .train_imu_letter_classifier import (
    IMULetterCNN,
    LABELS,
    load_examples,
    normalization,
    split_examples,
)


def _contact_path(gt: pd.DataFrame, start: float, end: float) -> np.ndarray:
    rows = gt[(gt["timestamp"] >= start) & (gt["timestamp"] <= end)]
    points = rows[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
    return points.astype(np.float32)


def _predict(model: IMULetterCNN, examples: list[dict], mean: np.ndarray, std: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([(row["x"] - mean) / std for row in examples]).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True), np.asarray([row["label"] for row in examples])


def _select_examples(predictions: np.ndarray, labels: np.ndarray) -> tuple[list[int], list[int]]:
    guesses = predictions.argmax(axis=1)
    confidence = predictions.max(axis=1)
    correct: list[int] = []
    seen: set[int] = set()
    for index in np.argsort(-confidence):
        if guesses[index] == labels[index] and int(labels[index]) not in seen:
            correct.append(int(index))
            seen.add(int(labels[index]))
        if len(correct) == 6:
            break
    incorrect: list[int] = []
    seen = set()
    for index in np.argsort(-confidence):
        if guesses[index] != labels[index] and int(labels[index]) not in seen:
            incorrect.append(int(index))
            seen.add(int(labels[index]))
        if len(incorrect) == 6:
            break
    return correct, incorrect


def _plot_path(axis, points: np.ndarray, title: str, correct: bool) -> None:
    colour = "#0072B2" if correct else "#D55E00"  # Okabe-Ito blue / vermillion
    if len(points):
        axis.plot(points[:, 0], points[:, 1], color=colour, linewidth=2.0)
        axis.scatter(points[0, 0], points[0, 1], color="#009E73", s=20, zorder=3, label="start")
        axis.scatter(points[-1, 0], points[-1, 1], color="#CC79A7", marker="x", s=25, zorder=3, label="end")
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(title, fontsize=9, loc="left", fontweight="bold")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#BBBBBB")


def make_figures(data_root: Path, source_root: Path, checkpoint: Path, output_dir: Path, device: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = json.loads((data_root / "conversion_manifest.json").read_text(encoding="utf-8"))
    examples = load_examples(data_root, 128)
    records = converted["records"]
    if len(examples) != len(records):
        raise ValueError("conversion records and loadable examples do not align")
    train, _, _, split = split_examples(examples)
    test_indices = [index for index, row in enumerate(examples) if row["user"] in split["test_users"]]
    test_examples = [examples[index] for index in test_indices]
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    mean, std = normalization(train)
    model = IMULetterCNN().to(device).eval()
    model.load_state_dict(payload["state_dict"])
    probabilities, labels = _predict(model, test_examples, mean, std, device)
    guesses = probabilities.argmax(axis=1)
    selected_correct, selected_wrong = _select_examples(probabilities, labels)
    raw_manifest = pd.read_csv(source_root / "_meta" / "manifest.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.3))
    for row_index, selected in enumerate(selected_correct + selected_wrong):
        axis = axes.flat[row_index]
        original_index = test_indices[selected]
        record = records[original_index]
        source = raw_manifest.iloc[int(record["source_row"])]
        gt = pd.read_csv(source_root / source["gt_raw_path"])
        path = _contact_path(gt, float(record["contact_start"]), float(record["contact_end"]))
        top3 = np.argsort(-probabilities[selected])[:3]
        true_label, guessed_label = LABELS[labels[selected]].upper(), LABELS[guesses[selected]].upper()
        top3_text = " ".join(f"{LABELS[item].upper()} {probabilities[selected, item]:.0%}" for item in top3)
        is_correct = labels[selected] == guesses[selected]
        prefix = "Correct" if is_correct else "Error"
        _plot_path(axis, path, f"{prefix}: GT {true_label} → {guessed_label}\n{top3_text}", bool(is_correct))
    fig.suptitle("Independent test examples — direct IMU classifier, GT contact crop", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.01, 0.01, "Top row: high-confidence correct examples. Bottom row: high-confidence incorrect examples. Green=start, pink ×=end.", fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.84, bottom=0.11, wspace=0.03, hspace=0.34)
    examples_png = output_dir / "imu_letter_examples.png"
    fig.savefig(examples_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "imu_letter_examples.pdf", bbox_inches="tight")
    plt.close(fig)

    confusion = np.zeros((26, 26), dtype=np.int64)
    for truth, guess in zip(labels, guesses):
        confusion[truth, guess] += 1
    support = confusion.sum(axis=1)
    per_class = np.divide(np.diag(confusion), support, out=np.zeros(26), where=support > 0)
    fig, (heatmap_axis, bar_axis) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]}, constrained_layout=True)
    image = heatmap_axis.imshow(np.divide(confusion, support[:, None], out=np.zeros_like(confusion, dtype=float), where=support[:, None] > 0), cmap="viridis", vmin=0, vmax=1)
    heatmap_axis.set_xticks(range(26), [label.upper() for label in LABELS], fontsize=8)
    heatmap_axis.set_yticks(range(26), [label.upper() for label in LABELS], fontsize=8)
    heatmap_axis.set_xlabel("Predicted letter")
    heatmap_axis.set_ylabel("True letter")
    heatmap_axis.set_title("Row-normalized confusion matrix")
    colourbar = fig.colorbar(image, ax=heatmap_axis, shrink=0.78)
    colourbar.set_label("Fraction of each true letter")
    bar_axis.bar(range(26), per_class * 100, color="#0072B2")
    bar_axis.axhline(per_class.mean() * 100, color="#D55E00", linestyle="--", linewidth=1.5, label=f"macro mean {per_class.mean():.1%}")
    bar_axis.set_xticks(range(26), [label.upper() for label in LABELS], fontsize=8)
    bar_axis.set_ylim(0, 105)
    bar_axis.set_ylabel("Top-1 accuracy (%)")
    bar_axis.set_title("Per-letter accuracy")
    bar_axis.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Direct IMU → letter: top-1 {(guesses == labels).mean():.2%}, top-3 {(np.argsort(-probabilities, axis=1)[:, :3] == labels[:, None]).any(axis=1).mean():.2%}; n={len(labels)}", fontsize=13, fontweight="bold")
    diagnostics_png = output_dir / "imu_letter_diagnostics.png"
    fig.savefig(diagnostics_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "imu_letter_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"examples": str(examples_png), "diagnostics": str(diagnostics_png), "top1": float((guesses == labels).mean()), "top3": float((np.argsort(-probabilities, axis=1)[:, :3] == labels[:, None]).any(axis=1).mean()), "n": int(len(labels))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 200 Hz IMU letter-classifier test examples")
    parser.add_argument("--data-root", type=Path, default=Path("outputs/touch_200hz_v3/data"))
    parser.add_argument("--source-root", type=Path, default=Path("/data/huyang/trae_projects/raw/200hz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/imu_letter_200hz_v1/imu_letter_cnn.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/imu_letter_200hz_v1/figures"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(make_figures(args.data_root, args.source_root, args.checkpoint, args.output_dir, args.device), indent=2))
