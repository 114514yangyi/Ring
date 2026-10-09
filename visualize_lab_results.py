"""Figures for the lab 200 Hz dataset: trajectory reconstruction, letter recognition, coverage.

Produces (into ``--output-dir``):

* ``trajectory_examples.png``  ground-truth pen trace vs trajectory rebuilt from the ring
* ``letter_accuracy.png``      per-letter top-1 accuracy (ground truth and reconstructed input)
* ``confusion_recon.png``      confusion matrix of the end-to-end (reconstructed) recogniser
* ``error_vs_coverage.png``    reconstruction error against how much of the letter was recorded
* ``recognition_examples.png`` a few letters with their predicted trajectory and prediction

Usage::

    python -m writing_state.visualize_lab_results --data-root /data/huyang/datasets/RingLab/lab200_v4 \
        --trajectory-checkpoint models_lab/traj_a_scratch/paper_trajectory.pt \
        --character-report models_character/lab_both_a/character_report.json \
        --output-dir outputs/lab_eval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .paper_trajectory import (
    integrate_contact_segments,
    load_trajectory_predictor,
    trajectory_metrics,
)
from .paper_trajectory_dataset import discover_trajectory_samples, load_sample_arrays
from .train_paper_trajectory import random_split_samples

LETTERS = [chr(ord("A") + index) for index in range(26)]


def configure_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def coverage_table(meta_path: Path, collection_csv: Path | None = None) -> dict[str, float]:
    """Sample id -> fraction of the pen-down time that lies inside the ring recording.

    ``collection_csv`` (``coverage_all.csv``) holds one row per raw trial in manifest order, so
    the row position recovers the ``<index>_<trial>`` prefix used by the dataset builder.  It is
    needed for datasets built before the coverage column existed in ``meta.csv``.
    """

    table: dict[str, float] = {}
    if collection_csv is not None and Path(collection_csv).exists():
        with open(collection_csv, newline="", encoding="utf-8") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                table[f"{index:05d}"] = float(row["coverage"])
    with open(meta_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("coverage")
            if value:
                table[row["sample"]] = float(value)
    return table


def reconstruct(predictor: Any, sample: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, board, mask, _, _ = load_sample_arrays(sample, mmap=False)
    x = np.asarray(x)
    board = np.asarray(board)
    mask = np.asarray(mask)
    deltas = np.asarray(predictor.predict_deltas(x.astype(np.float32)))
    segments = integrate_contact_segments(
        deltas, mask, window_offset=predictor.config.window_offset, min_frames=5
    )
    if not segments:
        return np.zeros((0, 2)), np.zeros((0, 2)), board
    predicted = np.concatenate([segment.xy for segment in segments], axis=0)
    truth = np.concatenate([board[segment.start:segment.end] - board[segment.start] for segment in segments], axis=0)
    return predicted, truth, board


def normalize_shape(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    centred = points - points.mean(axis=0, keepdims=True)
    scale = np.abs(centred).max()
    return centred / scale if scale > 1e-9 else centred


def figure_trajectory_examples(rows: Sequence[dict[str, Any]], path: Path, columns: int = 4, count: int = 16) -> None:
    chosen = list(rows)
    chosen.sort(key=lambda row: row["normalized"])
    picks = chosen[:4] + chosen[len(chosen) // 4 :: max(1, len(chosen) // 4)][: count - 8] + chosen[-4:]
    picks = picks[:count]
    rows_n = (len(picks) + columns - 1) // columns
    fig, axes = plt.subplots(rows_n, columns, figsize=(3.0 * columns, 2.7 * rows_n))
    axes = np.atleast_2d(axes)
    for axis, row in zip(axes.ravel(), picks):
        axis.plot(row["gt"][:, 0], row["gt"][:, 1], color="0.15", lw=2.4, label="真值轨迹")
        axis.plot(row["pred"][:, 0], row["pred"][:, 1], color="#d94801", lw=1.6, label="重建轨迹")
        axis.set_title(
            f"{LETTERS[row['label']]}  归一化误差 {row['normalized']:.2f}  覆盖率 {row['coverage']:.2f}",
            fontsize=9,
        )
        axis.set_aspect("equal")
        axis.invert_yaxis()  # 采集坐标 y 轴向下（屏幕/CSS 约定），翻转后字母方向才与书写一致
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(picks):]:
        axis.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    fig.suptitle("自采 200 Hz 数据：真值轨迹 vs 由戒指信号重建的轨迹", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def figure_letter_accuracy(report_path: Path, path: Path, gt_report: Path | None, title: str) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recon = report.get("test_recon", {})
    per_letter = recon.get("per_letter", {})
    gt_per_letter: dict[str, Any] = {}
    if gt_report is not None and Path(gt_report).exists():
        gt_per_letter = json.loads(Path(gt_report).read_text(encoding="utf-8")).get("test_gt", {}).get("per_letter", {})
    values_recon = [per_letter.get(letter, {}).get("top1", np.nan) for letter in LETTERS]
    values_gt = [gt_per_letter.get(letter, {}).get("top1", np.nan) for letter in LETTERS]
    counts = [per_letter.get(letter, {}).get("n", 0) for letter in LETTERS]
    positions = np.arange(len(LETTERS))
    fig, axis = plt.subplots(figsize=(12.4, 4.6))
    if not np.all(np.isnan(values_gt)):
        axis.bar(positions - 0.2, np.nan_to_num(values_gt) * 100, width=0.4, color="#9ecae1", label="真值轨迹输入")
    axis.bar(positions + 0.2, np.nan_to_num(values_recon) * 100, width=0.4, color="#08519c", label="重建轨迹输入")
    for position, count in zip(positions, counts):
        axis.text(position + 0.2, 102, f"n={count}", ha="center", fontsize=6.5, rotation=90,
                  color="0.35")
    axis.set_xticks(positions, LETTERS)
    axis.set_ylim(0, 126)
    axis.set_ylabel("Top-1 识别率 (%)")
    axis.set_xlabel("字母（按字母表顺序）")
    axis.axhline(recon.get("top1", np.nan) * 100, color="#08519c", ls="--", lw=1, alpha=0.7)
    axis.legend(frameon=False, ncol=2, loc="upper left", fontsize=9.5)
    axis.set_title(title, fontsize=13)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_confusion(report_path: Path, path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matrix = np.asarray(report["test_recon"]["confusion"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(7.6, 6.8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(26), LETTERS, fontsize=8)
    axis.set_yticks(range(26), LETTERS, fontsize=8)
    axis.set_xlabel("预测字母")
    axis.set_ylabel("真实字母")
    for i in range(26):
        for j in range(26):
            if normalized[i, j] >= 0.25 and i != j:
                axis.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=6.5, color="#a63603")
    fig.colorbar(image, ax=axis, fraction=0.046, label="行归一化比例")
    axis.set_title(f"端到端字母识别混淆矩阵（重建轨迹，全部 {int(matrix.sum())} 个测试样本）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_error_vs_coverage(rows: Sequence[dict[str, Any]], path: Path) -> None:
    coverage = np.array([row["coverage"] for row in rows])
    error = np.array([row["normalized"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    axes[0].scatter(coverage, np.clip(error, 1e-2, None), s=14, alpha=0.6, color="#08519c")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("覆盖率（被戒指录到的落笔时长比例）")
    axes[0].set_ylabel("归一化重建误差（对数轴）")
    axes[0].axhline(np.median(error), color="#d94801", ls="--", lw=1, label=f"中位数 {np.median(error):.2f}")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("误差 vs 采集覆盖率（每个点一个测试样本）", fontsize=11)

    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.001)]
    labels, means, medians, sizes = [], [], [], []
    for low, high in bins:
        selected = error[(coverage >= low) & (coverage < high)]
        labels.append(f"{low:.2f}-{high if high < 1 else 1.0:.2f}")
        means.append(selected.mean() if selected.size else np.nan)
        medians.append(np.median(selected) if selected.size else np.nan)
        sizes.append(int(selected.size))
    positions = np.arange(len(bins))
    axes[1].bar(positions - 0.18, means, width=0.36, color="#6baed6", label="均值")
    axes[1].bar(positions + 0.18, medians, width=0.36, color="#08519c", label="中位数")
    for position, size in zip(positions, sizes):
        axes[1].text(position - 0.18, 0.02, f"n={size}", ha="center", fontsize=7, rotation=90, color="w")
    axes[1].set_xticks(positions, labels)
    axes[1].set_xlabel("覆盖率区间")
    axes[1].set_ylabel("归一化重建误差")
    axes[1].set_title("按覆盖率分组的重建误差", fontsize=11)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_recognition_examples(rows: Sequence[dict[str, Any]], path: Path, count: int = 8) -> None:
    picks = sorted(rows, key=lambda row: row["normalized"])[:count]
    columns = 4
    rows_n = (len(picks) + columns - 1) // columns
    fig, axes = plt.subplots(rows_n, columns, figsize=(3.1 * columns, 2.9 * rows_n))
    axes = np.atleast_2d(axes)
    for axis, row in zip(axes.ravel(), picks):
        axis.plot(row["gt"][:, 0], row["gt"][:, 1], color="0.15", lw=2.4, label="真值")
        axis.plot(row["pred"][:, 0], row["pred"][:, 1], color="#d94801", lw=1.6, label="重建")
        hit = "√" if row.get("predicted") == row["label"] else "×"
        axis.set_title(
            f"{LETTERS[row['label']]}  →  {LETTERS[row['predicted']] if row.get('predicted') is not None else '?'}  {hit}"
            f"   误差 {row['normalized']:.2f}",
            fontsize=9.5,
        )
        axis.set_aspect("equal")
        axis.invert_yaxis()  # 同上：采集坐标 y 轴向下
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(picks):]:
        axis.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    fig.suptitle("端到端识别样例：戒指信号重建的字母轨迹与识别结果", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)



def figure_data_quality(collection_csv: Path, path: Path) -> None:
    """Coverage of the ring recording against the pen-down window, per trial and per letter."""

    rows = list(csv.DictReader(open(collection_csv, newline="", encoding="utf-8")))
    coverage = np.array([float(row["coverage"]) for row in rows])
    letters = [row["letter"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3))
    axes[0].hist(coverage, bins=np.arange(0, 1.05, 0.05), color="#6baed6", edgecolor="w")
    axes[0].axvline(0.95, color="#d94801", ls="--", lw=1.2)
    axes[0].set_xlabel("覆盖率（被戒指录到的落笔时长比例）")
    axes[0].set_ylabel("trial 数")
    axes[0].set_title(f"全部 {len(coverage)} 个 trial 的覆盖率分布；≥0.95 仅 {(coverage >= 0.95).mean() * 100:.0f}%", fontsize=11)
    axes[0].grid(axis="y", alpha=0.25)

    means = [np.mean([c for c, l in zip(coverage, letters) if l == letter]) for letter in LETTERS]
    positions = np.arange(len(LETTERS))
    axes[1].bar(positions, np.array(means) * 100, color="#08519c")
    axes[1].set_xticks(positions, LETTERS, fontsize=8)
    axes[1].set_ylim(0, 100)
    axes[1].axhline(coverage.mean() * 100, color="#d94801", ls="--", lw=1, label=f"整体 {coverage.mean() * 100:.0f}%")
    axes[1].set_xlabel("字母")
    axes[1].set_ylabel("平均覆盖率 (%)")
    axes[1].set_title("逐字母平均覆盖率（采集质量）", fontsize=11)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise lab-dataset trajectory and recognition results")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--trajectory-checkpoint", type=Path, required=True)
    parser.add_argument("--character-report", type=Path, required=True,
                        help="report of the letter CNN trained with reconstructed trajectories")
    parser.add_argument("--gt-report", type=Path, default=None)
    parser.add_argument("--character-checkpoint", type=Path, default=None,
                        help="optional letter CNN checkpoint used to label the sample trajectories")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lab_eval"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coverage-csv", type=Path,
                        default=Path("/data/huyang/datasets/RingLab/coverage_all.csv"),
                        help="per-trial coverage table (row order = manifest order)")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    configure_fonts()
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_trajectory_samples(args.data_root)
    train, val, test, _ = random_split_samples(samples, seed=42)
    subset = {"train": train, "val": val, "test": test}[args.split]
    if args.limit:
        subset = subset[: args.limit]
    coverage = coverage_table(args.data_root / "meta.csv", args.coverage_csv)
    predictor = load_trajectory_predictor(args.trajectory_checkpoint, device=args.device)

    rows: list[dict[str, Any]] = []
    for sample in subset:
        predicted, truth, _ = reconstruct(predictor, sample)
        if len(predicted) < 2:
            continue
        label = int(Path(sample.board_path).parent.name)
        metrics = trajectory_metrics(predicted, truth, predictor.board_mm_scale)
        rows.append({
            "sample": sample.sample_id,
            "label": label,
            "coverage": coverage.get(sample.sample_id) or coverage.get(sample.sample_id.split("_")[0], 1.0),
            "normalized": metrics["normalized_mean"],
            "mm": metrics["mm_mean"],
            "gt": truth,
            "pred": predicted,
        })

    predicted_labels = None
    if args.character_checkpoint is not None:
        from .character_dataset import normalize_trajectory, resample_trajectory
        from .character_model import RESAMPLE_FRAMES, load_character_model
        import torch

        model = load_character_model(args.character_checkpoint, device=args.device)
        target = next(model.parameters()).device
        batch = torch.tensor(
            np.stack([
                normalize_trajectory(resample_trajectory(row["pred"], RESAMPLE_FRAMES)) for row in rows
            ]),
            dtype=torch.float32,
            device=target,
        )
        with torch.no_grad():
            logits = model(batch)
        predicted_labels = logits.argmax(dim=1).cpu().numpy().tolist()
    if predicted_labels is not None:
        for row, label in zip(rows, predicted_labels):
            row["predicted"] = int(label)

    if args.coverage_csv is not None and Path(args.coverage_csv).exists():
        figure_data_quality(args.coverage_csv, args.output_dir / "data_quality.png")
    figure_trajectory_examples(rows, args.output_dir / "trajectory_examples.png")
    figure_error_vs_coverage(rows, args.output_dir / "error_vs_coverage.png")
    figure_letter_accuracy(
        args.character_report,
        args.output_dir / "letter_accuracy.png",
        args.gt_report,
        "26 个字母的识别率（随机划分测试集）",
    )
    figure_confusion(args.character_report, args.output_dir / "confusion_recon.png")
    if predicted_labels is not None:
        figure_recognition_examples(rows, args.output_dir / "recognition_examples.png")

    summary = {
        "samples": len(rows),
        "normalized_mean": float(np.mean([row["normalized"] for row in rows])),
        "normalized_median": float(np.median([row["normalized"] for row in rows])),
        "mm_mean": float(np.mean([row["mm"] for row in rows])),
        "mm_median": float(np.median([row["mm"] for row in rows])),
    }
    with (args.output_dir / "trajectory_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
