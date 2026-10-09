"""Overview of the ground-truth pen traces collected in the 200 Hz letter set.

Reads the raw collection through ``evaluate_real_data.read_manifest`` (so it works on any
future drop of the same format) and writes

* ``gt_letters_overview.png`` -- one panel per letter with up to 15 pen traces, each trace
  centred on its own bounding box and scaled by its bounding-box diagonal so that the *shape*
  (not the position on the tablet) is compared;
* ``gt_letters_gallery.png``  -- three rows: typical traces, the widest traces and the most
  partial ones, for a quick quality read;
* ``gt_letter_stats.csv``     -- per-letter counts, strokes, pen duration and size in mm.

Usage::

    python -m writing_state.visualize_gt_letters --output-dir outputs/lab_eval
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate_real_data import DEFAULT_RAW_ROOT, pen_runs, read_manifest
from .visualize_lab_results import LETTERS, configure_fonts

MM_PER_PX = {"fk": 0.176409, "yjx": 0.250143}


def load_traces(root: Path, conditions: Sequence[str] | None = None,
                sources: Sequence[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    trials = read_manifest(root, conditions, sources)
    table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        runs = [run for run in pen_runs(trial.gt_raw_path) if len(run) > 1]
        if not runs:
            continue
        scale = MM_PER_PX.get(trial.source, 0.25)
        segments = []
        for run in runs:
            points = np.asarray([[point[1], point[2]] for point in run], dtype=np.float64) * scale
            if len(points) > 1:
                segments.append(points)
        if not segments:
            continue
        points = np.concatenate(segments, axis=0)
        duration = sum((run[-1][0] - run[0][0]) for run in runs)
        table[trial.letter.upper()].append({
            "trial": trial.trial,
            "source": trial.source,
            "condition": trial.condition,
            "segments": segments,
            "points": points,
            "diagonal_mm": float(np.linalg.norm(points.max(0) - points.min(0))),
            "bytes_mm": float(np.abs(np.diff(points, axis=0)).sum()),
            "pen_seconds": float(duration),
            "strokes": len(segments),
        })
    return table


def shape_only(points: np.ndarray) -> np.ndarray:
    centre = (points.max(0) + points.min(0)) / 2.0
    span = float(np.linalg.norm(points.max(0) - points.min(0)))
    if span < 1e-9:
        return np.zeros_like(points)
    return (points - centre) / span


def figure_overview(table: dict[str, list[dict[str, Any]]], path: Path, per_letter: int = 15) -> None:
    columns, rows = 6, 5
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.25 * rows))
    axes = np.atleast_2d(axes)
    for axis, letter in zip(axes.ravel(), LETTERS):
        entries = table.get(letter, [])
        if not entries:
            axis.set_title(f"{letter} (无样本)", fontsize=9)
            axis.axis("off")
            continue
        chosen = entries[:per_letter]
        for entry in chosen:
            shape = shape_only(entry["points"])
            axis.plot(shape[:, 0], shape[:, 1], color="#4c78a8", lw=1.0, alpha=0.55)
        example = entries[0]
        for segment in example["segments"]:
            shape = shape_only(segment)
            axis.plot(shape[:, 0], shape[:, 1], color="#d94801", lw=1.6)
        diagonal = float(np.median([entry["diagonal_mm"] for entry in entries]))
        strokes = float(np.median([entry["strokes"] for entry in entries]))
        axis.set_title(f"{letter}   n={len(entries)}  {strokes:.0f}笔  尺寸中位 {diagonal:.0f} mm", fontsize=8.5)
        axis.set_xlim(-0.75, 0.75)
        axis.set_ylim(-0.75, 0.75)
        axis.set_aspect("equal")
        axis.invert_yaxis()  # 采集坐标 y 轴向下（屏幕/CSS 约定），翻转后字母方向才与书写一致
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(alpha=0.15)
    for axis in axes.ravel()[len(LETTERS):]:
        axis.axis("off")
    fig.suptitle("自采 200 Hz 数据集:26 个字母的真值轨迹(蓝=各样本,橙=示例;按各自外框归一化)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_gallery(table: dict[str, list[dict[str, Any]]], path: Path, per_letter: int = 3) -> None:
    pool: list[tuple[str, dict[str, Any]]] = []
    for letter in LETTERS:
        entries = table.get(letter, [])
        if not entries:
            continue
        ordered = sorted(entries, key=lambda entry: entry["diagonal_mm"])
        picks = ordered[-per_letter:] + ordered[:per_letter] + ordered[len(ordered) // 2:][:per_letter]
        pool.extend((letter, entry) for entry in picks)
    columns = 9
    rows = (len(pool) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(1.25 * columns, 1.45 * rows))
    axes = np.atleast_2d(axes)
    for axis, (letter, entry) in zip(axes.ravel(), pool):
        shape = shape_only(entry["points"])
        axis.plot(shape[:, 0], shape[:, 1], color="#08519c", lw=1.2)
        axis.set_title(f"{letter} {entry['diagonal_mm']:.0f}mm {entry['source']}", fontsize=6.5)
        axis.set_xlim(-0.7, 0.7)
        axis.set_ylim(-0.7, 0.7)
        axis.set_aspect("equal")
        axis.invert_yaxis()  # 同上：采集坐标 y 轴向下
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(pool):]:
        axis.axis("off")
    fig.suptitle("真值轨迹画廊:每个字母取最大 / 居中 / 最小各若干(可见部分只记录到一部分笔画的样本)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_stats(table: dict[str, list[dict[str, Any]]], path: Path) -> None:
    rows = []
    for letter in LETTERS:
        entries = table.get(letter, [])
        if not entries:
            continue
        diagonal = np.array([entry["diagonal_mm"] for entry in entries])
        pen = np.array([entry["pen_seconds"] for entry in entries])
        strokes = np.array([entry["strokes"] for entry in entries])
        rows.append({
            "letter": letter,
            "trials": len(entries),
            "fk": sum(1 for entry in entries if entry["source"] == "fk"),
            "yjx": sum(1 for entry in entries if entry["source"] == "yjx"),
            "strokes_median": float(np.median(strokes)),
            "strokes_max": int(strokes.max()),
            "pen_seconds_median": round(float(np.median(pen)), 3),
            "diagonal_mm_median": round(float(np.median(diagonal)), 1),
            "diagonal_mm_p10": round(float(np.percentile(diagonal, 10)), 1),
            "diagonal_mm_p90": round(float(np.percentile(diagonal, 90)), 1),
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise the ground-truth letters of a collection")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lab_eval"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    configure_fonts()
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = load_traces(args.raw_root, args.conditions, args.sources)
    figure_overview(table, args.output_dir / "gt_letters_overview.png")
    figure_gallery(table, args.output_dir / "gt_letters_gallery.png")
    write_stats(table, args.output_dir / "gt_letter_stats.csv")
    total = sum(len(entries) for entries in table.values())
    print(f"letters={len(table)} trials={total} -> {args.output_dir}")


if __name__ == "__main__":
    main()
