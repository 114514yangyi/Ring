"""Contact sheets of *every* trial in the lab 200 Hz collection, coloured by recording coverage.

Produces (into ``--output-dir``):

* ``all_samples_overview.png``  -- every trial as a mini panel, one row per letter, sorted by
  coverage; green = coverage >= 0.9, amber = 0.6-0.9, red = < 0.6;
* ``all_samples/<LETTER>.png``  -- one readable sheet per letter, each panel titled with the
  trial id, its coverage and how many pen rows the tablet recorded;
* ``coverage_bands_per_letter.png`` -- stacked bars: share of each letter that was recorded
  completely / partially / barely;
* ``all_samples_summary.csv``   -- the same numbers as a table.

Usage::

    python -m writing_state.visualize_all_samples --output-dir outputs/lab_eval
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
from .visualize_gt_letters import LETTERS, shape_only
from .visualize_lab_results import configure_fonts

BANDS = ((0.9, 1.01, "#1e8449", "录全了 (>=90%)"),
         (0.6, 0.9, "#e6a817", "缺一点 (60-90%)"),
         (0.0, 0.6, "#c0392b", "缺一半以上 (<60%)"))


def band_of(coverage: float) -> int:
    for index, (low, high, _, _) in enumerate(BANDS):
        if low <= coverage < high:
            return index
    return 0


def load_trials(root: Path, coverage_csv: Path = Path("/data/huyang/datasets/RingLab/coverage_all.csv")) -> list[dict[str, Any]]:
    trials = read_manifest(root)
    table: dict[tuple[str, str], float] = {}
    with open(coverage_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[(row["condition"].strip(), row["trial"].strip())] = float(row["coverage"])
    entries: list[dict[str, Any]] = []
    for trial in trials:
        runs = [np.asarray([[point[1], point[2]] for point in run], dtype=np.float64)
                for run in pen_runs(trial.gt_raw_path) if len(run) > 1]
        if not runs:
            continue
        points = np.concatenate(runs, axis=0)
        entries.append({
            "trial": trial.trial,
            "letter": trial.letter.upper(),
            "source": trial.source,
            "condition": trial.condition,
            "coverage": table.get((trial.condition, trial.trial), float("nan")),
            "rows": len(points),
            "points": points,
            "shape": shape_only(points),
        })
    return entries


def figure_overview(entries: Sequence[dict[str, Any]], path: Path) -> None:
    by_letter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_letter[entry["letter"]].append(entry)
    width = max(len(rows) for rows in by_letter.values())
    fig, axes = plt.subplots(len(LETTERS), width, figsize=(0.40 * width, 0.52 * len(LETTERS)))
    axes = np.atleast_2d(axes)
    for row, letter in enumerate(LETTERS):
        rows = sorted(by_letter.get(letter, []), key=lambda item: -item["coverage"])
        for column in range(width):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            axis.invert_yaxis()
            for side in axis.spines.values():
                side.set_visible(False)
            if column == 0:
                axis.set_ylabel(letter, fontsize=7, rotation=0, labelpad=9, va="center")
            if column >= len(rows):
                axis.axis("off")
                continue
            entry = rows[column]
            color = BANDS[band_of(entry["coverage"])][2]
            axis.plot(entry["shape"][:, 0], entry["shape"][:, 1], color=color, lw=0.8)
            axis.set_xlim(-0.75, 0.75)
            axis.set_ylim(-0.75, 0.75)
    handles = [plt.Line2D([], [], color=color, lw=3, label=label) for _, _, color, label in BANDS]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3, fontsize=9)
    fig.suptitle("全部 %d 个样本(每行一个字母,按覆盖率从左到右排序);颜色=录制窗覆盖了字母书写的多少"
                 % len(entries), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_letter_sheet(letter: str, rows: Sequence[dict[str, Any]], path: Path, columns: int = 9) -> None:
    rows = sorted(rows, key=lambda item: -item["coverage"])
    height = (len(rows) + columns - 1) // columns
    fig, axes = plt.subplots(height, columns, figsize=(1.55 * columns, 1.75 * height))
    axes = np.atleast_2d(axes)
    for axis, entry in zip(axes.ravel(), rows):
        color = BANDS[band_of(entry["coverage"])][2]
        axis.plot(entry["shape"][:, 0], entry["shape"][:, 1], color=color, lw=1.4)
        axis.set_xlim(-0.75, 0.75)
        axis.set_ylim(-0.75, 0.75)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title("%s  cov %.2f  n=%d" % (entry["trial"], entry["coverage"], entry["rows"]),
                       fontsize=5.6)
    for axis in axes.ravel()[len(rows):]:
        axis.axis("off")
    complete = sum(1 for entry in rows if entry["coverage"] >= 0.9)
    fig.suptitle("%s:共 %d 个样本,其中覆盖率 >=0.9 的 %d 个(%.0f%%)"
                 % (letter, len(rows), complete, 100 * complete / len(rows)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_bands(entries: Sequence[dict[str, Any]], path: Path) -> None:
    by_letter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_letter[entry["letter"]].append(entry)
    fig, axis = plt.subplots(figsize=(13, 4.2))
    bottom = np.zeros(len(LETTERS))
    for index, (_, _, color, label) in enumerate(BANDS):
        share = np.array([sum(1 for entry in by_letter[letter] if band_of(entry["coverage"]) == index)
                          / max(1, len(by_letter[letter])) for letter in LETTERS])
        axis.bar(LETTERS, share, bottom=bottom, color=color, label=label)
        for position, value in enumerate(share):
            if value > 0.08:
                axis.text(position, bottom[position] + value / 2, "%.0f" % (100 * value),
                          ha="center", va="center", fontsize=7, color="white")
        bottom += share
    axis.set_ylim(0, 1)
    axis.set_ylabel("样本占比")
    axis.legend(loc="upper center", ncol=3, frameon=False, fontsize=9)
    axis.set_title("逐字母:录制窗把字母录全的程度(绿=录全, 黄=缺一点, 红=缺一半以上)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(entries: Sequence[dict[str, Any]], path: Path) -> None:
    by_letter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_letter[entry["letter"]].append(entry)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["letter", "trials", "complete(>=0.9)", "partial(0.6-0.9)", "poor(<0.6)",
                         "complete_share", "smallest_rows"])
        for letter in LETTERS:
            rows = by_letter.get(letter, [])
            counts = [sum(1 for entry in rows if band_of(entry["coverage"]) == index) for index in range(3)]
            writer.writerow([letter, len(rows), counts[0], counts[1], counts[2],
                             round(counts[0] / max(1, len(rows)), 3),
                             min((entry["rows"] for entry in rows), default=0)])


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contact sheets of every trial of a collection")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--coverage-csv", type=Path,
                        default=Path("/data/huyang/datasets/RingLab/coverage_all.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lab_eval"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    configure_fonts()
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheets = args.output_dir / "all_samples"
    sheets.mkdir(parents=True, exist_ok=True)
    entries = load_trials(args.raw_root, args.coverage_csv)
    by_letter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_letter[entry["letter"]].append(entry)
    figure_overview(entries, args.output_dir / "all_samples_overview.png")
    for letter in LETTERS:
        if by_letter.get(letter):
            figure_letter_sheet(letter, by_letter[letter], sheets / ("%s.png" % letter))
    figure_bands(entries, args.output_dir / "coverage_bands_per_letter.png")
    write_summary(entries, args.output_dir / "all_samples_summary.csv")
    complete = sum(1 for entry in entries if entry["coverage"] >= 0.9)
    print("samples=%d  coverage>=0.9: %d (%.0f%%)  -> %s"
          % (len(entries), complete, 100 * complete / len(entries), args.output_dir))


if __name__ == "__main__":
    main()
