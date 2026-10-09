"""Diagnose the lab's own 200 Hz ring dataset and plot the evaluation results.

Two jobs, both aimed at the question "can our models be used on this data?":

1. per-trial quality metrics -> ``quality.json`` + ``fig_real_data_diagnostic.png``:
   - 6-channel input amplitude versus the training normalisation stored in the trajectory
     checkpoint (a mismatch here means the model sees out-of-distribution inputs);
   - gyro / accelerometer self-consistency (rigid-body check, R^2 of d(acc_dir)/dt vs -w x acc_dir);
   - quaternion-predicted gravity versus the measured accelerometer direction;
   - correlation between the IMU motion envelope and the tablet pen speed, at zero lag and at
     the best lag, together with how much slack the recording has for that lag.
2. plots from ``evaluate_real_data.py`` outputs (``report.json`` + ``trajectories.npz``)
   -> ``fig_real_data_trajectories.png`` + ``fig_real_data_recognition.png``.

Run from the repo parent directory (``writing_state -> Ring`` symlink)::

    python -m writing_state.analyze_real_data --output-dir outputs/real_data_eval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .evaluate_real_data import (
    DEFAULT_FS,
    DEFAULT_RAW_ROOT,
    REPO_ROOT,
    build_input,
    highpass,
    motion_profile,
    read_gt,
    read_imu,
    read_manifest,
    uniform_grid,
)

LABELS = [chr(ord("A") + index) for index in range(26)]
CHANNEL_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]
QUAT_CONVENTIONS = ("wxyz", "xyzw")


# ----------------------------------------------------------------------------- helpers


def quat_to_matrix(quat: np.ndarray, convention: str) -> np.ndarray:
    """Rotation matrices R(q) for a stack of quaternions, shape (n, 3, 3)."""

    if convention == "wxyz":
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    else:
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rows = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    )
    return rows.reshape(-1, 3, 3)


def gravity_cosines(acc: np.ndarray, quat: np.ndarray) -> dict[str, float]:
    """mean |cos| between the accelerometer direction and a quaternion-rotated world gravity."""

    norms = np.linalg.norm(acc, axis=1)
    keep = norms > 1e-6
    if keep.sum() < 32:
        return {f"{conv}/{mode}": float("nan") for conv in QUAT_CONVENTIONS for mode in ("R", "R^T")}
    acc_unit = acc[keep] / norms[keep, None]
    world = np.zeros((int(keep.sum()), 3), dtype=np.float64)
    world[:, 2] = 1.0
    out: dict[str, float] = {}
    for convention in QUAT_CONVENTIONS:
        matrix = quat_to_matrix(quat[keep], convention)
        for mode, rotated in (
            ("R", np.einsum("nij,nj->ni", matrix, world)),
            ("R^T", np.einsum("nji,nj->ni", matrix, world)),
        ):
            cos = np.einsum("ni,ni->n", rotated, acc_unit)
            out[f"{convention}/{mode}"] = float(np.mean(np.abs(cos)))
    return out


def gyro_acc_r2(acc: np.ndarray, gyro: np.ndarray, fs: float) -> float:
    """Rigid-body check on the gravity direction: d(acc/|acc|)/dt should be -w x (acc/|acc|)."""

    norms = np.linalg.norm(acc, axis=1)
    keep = norms > 1e-6
    if keep.sum() < 64:
        return float("nan")
    unit = acc[keep] / norms[keep, None]
    rate = np.gradient(unit, axis=0) * fs
    rate = highpass(rate, fs, 5.0)
    omega = gyro[keep]
    predicted = -np.cross(omega, unit)
    denom = float(np.sum(predicted * predicted))
    if denom < 1e-12:
        return float("nan")
    scale = float(np.sum(rate * predicted) / denom)
    residual = rate - scale * predicted
    total = float(np.sum((rate - rate.mean(axis=0)) ** 2))
    if total < 1e-12:
        return float("nan")
    return float(1.0 - np.sum(residual ** 2) / total)


def lag_curve(grid: np.ndarray, profile: np.ndarray, gt_t: np.ndarray,
              gt_speed: np.ndarray, lags: np.ndarray) -> np.ndarray:
    scores = np.full(len(lags), np.nan, dtype=np.float64)
    for index, lag in enumerate(lags):
        shifted = np.interp(grid, gt_t + lag, gt_speed, left=0.0, right=0.0)
        if shifted.std() < 1e-9:
            continue
        scores[index] = float(np.corrcoef(profile, shifted)[0, 1])
    return scores


def trial_quality(trial: Any, checkpoint_std: np.ndarray, fs: float, cutoff_hz: float,
                  lags: np.ndarray) -> dict[str, Any]:
    imu = read_imu(trial.imu_path)
    gt = read_gt(trial.gt_path)
    grid = uniform_grid(imu.t, fs)
    x = build_input(imu, grid, "highpass", cutoff_hz, 1.0)
    acc = np.stack([np.interp(grid, imu.t, imu.acc[:, axis]) for axis in range(3)], axis=1)
    gyro = np.stack([np.interp(grid, imu.t, imu.gyro[:, axis]) for axis in range(3)], axis=1)
    quat = np.stack([np.interp(grid, imu.t, imu.quat[:, axis]) for axis in range(4)], axis=1)

    offset = float(np.median(imu.received_at - imu.t)) + trial.delta
    gt_t_device = gt.t_unix - offset
    gt_norm = gt.xy_px / np.asarray([gt.width, gt.height], dtype=np.float64)
    dt_gt = float(np.median(np.diff(gt.t_unix))) if len(gt.t_unix) > 2 else 0.01
    gt_speed = np.linalg.norm(np.diff(gt_norm, axis=0), axis=1) / max(dt_gt, 1e-6)

    in_pen = (grid >= gt_t_device[0]) & (grid <= gt_t_device[-1])
    window = x[in_pen] if in_pen.sum() >= 16 else x
    profile = motion_profile(x[:, :3], x[:, 3:])
    curve = lag_curve(grid, profile, gt_t_device[1:], gt_speed, lags)
    finite = np.isfinite(curve)
    best = float(np.nanmax(curve)) if finite.any() else float("nan")
    best_lag = float(lags[int(np.nanargmax(curve))]) if finite.any() else float("nan")
    zero_index = int(np.argmin(np.abs(lags)))
    slack = float((imu.t[-1] - imu.t[0]) - (gt.t_unix[-1] - gt.t_unix[0]))

    return {
        "trial": trial.trial,
        "condition": trial.condition,
        "letter": trial.letter,
        "source": trial.source,
        "input_std": [float(value) for value in x.std(axis=0)],
        "input_std_in_pen": [float(value) for value in window.std(axis=0)],
        "input_std_ratio": [float(value) for value in (x.std(axis=0) / checkpoint_std)],
        "input_std_ratio_in_pen": [float(value) for value in (window.std(axis=0) / checkpoint_std)],
        "gyro_acc_r2": gyro_acc_r2(acc, gyro, fs),
        "gravity_cos": gravity_cosines(acc, quat),
        "corr_zero_lag": float(curve[zero_index]),
        "corr_best_lag": best,
        "best_lag_s": best_lag,
        "slack_s": slack,
        "lag_curve": [float(value) for value in curve],
    }


def summarize_quality(rows: Sequence[dict[str, Any]], checkpoint_std: np.ndarray,
                      lags: np.ndarray) -> dict[str, Any]:
    def med(key: str) -> float:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return float(np.nanmedian(values))

    def quant(key: str, q: float) -> float:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return float(np.nanpercentile(values, q))

    std_ratio = np.asarray([row["input_std_ratio"] for row in rows], dtype=np.float64)
    curves = np.asarray([row["lag_curve"] for row in rows], dtype=np.float64)
    gravity_keys = list(rows[0]["gravity_cos"].keys())
    gravity = {key: float(np.nanmean([row["gravity_cos"][key] for row in rows])) for key in gravity_keys}
    best_gravity = max(gravity, key=lambda key: gravity[key])
    r2 = np.asarray([row["gyro_acc_r2"] for row in rows], dtype=np.float64)
    corr_best = np.asarray([row["corr_best_lag"] for row in rows], dtype=np.float64)
    corr_zero = np.asarray([row["corr_zero_lag"] for row in rows], dtype=np.float64)
    best_lag = np.asarray([row["best_lag_s"] for row in rows], dtype=np.float64)
    slack = np.asarray([row["slack_s"] for row in rows], dtype=np.float64)
    return {
        "trials": len(rows),
        "checkpoint_std": [float(value) for value in checkpoint_std],
        "input_std_median": [
            float(np.nanmedian([row["input_std"][i] for row in rows])) for i in range(6)
        ],
        "input_std_ratio_median": [float(value) for value in np.nanmedian(std_ratio, axis=0)],
        "input_std_ratio_p10": [float(value) for value in np.nanpercentile(std_ratio, 10, axis=0)],
        "input_std_ratio_p90": [float(value) for value in np.nanpercentile(std_ratio, 90, axis=0)],
        "input_std_ratio_in_pen_median": [
            float(value) for value in np.nanmedian(
                np.asarray([row["input_std_ratio_in_pen"] for row in rows], dtype=np.float64), axis=0
            )
        ],
        "gyro_acc_r2_median": float(np.nanmedian(r2)),
        "gravity_cos_mean": gravity,
        "gravity_cos_best_convention": best_gravity,
        "gravity_cos_best": gravity[best_gravity],
        "corr_zero_lag_median": float(np.nanmedian(corr_zero)),
        "corr_best_lag_median": float(np.nanmedian(corr_best)),
        "corr_best_lag_p90": float(np.nanpercentile(corr_best, 90)),
        "frac_corr_best_below_0.3": float(np.nanmean(corr_best < 0.3)),
        "best_lag_median_s": float(np.nanmedian(best_lag)),
        "best_lag_p90_s": float(np.nanpercentile(best_lag, 90)),
        "slack_median_s": float(np.nanmedian(slack)),
        "frac_best_lag_beyond_slack": float(np.nanmean(best_lag > np.maximum(slack, 0.0))),
        "lag_curve_median": [float(value) for value in np.nanmedian(curves, axis=0)],
        "lag_curve_lags": [float(value) for value in lags],
        "r2_median": float(np.nanmedian(r2)),
        "r2_quantiles": [quant("gyro_acc_r2", 10), quant("gyro_acc_r2", 50), quant("gyro_acc_r2", 90)],
        "median_trial_r2": med("gyro_acc_r2"),
    }


# ----------------------------------------------------------------------------- figures


def _style() -> None:
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 130


def figure_diagnostic(quality: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    ratio = np.asarray(quality["input_std_ratio_median"])
    ratio_pen = np.asarray(quality.get("input_std_ratio_in_pen_median", ratio))
    colors = ["#2f6fdb"] * 3 + ["#d1542b"] * 3
    positions = np.arange(6)
    axes[0, 0].bar(positions - 0.2, ratio, width=0.4, color=colors, alpha=0.55, label="整段录制")
    axes[0, 0].bar(positions + 0.2, ratio_pen, width=0.4, color=colors, label="落笔时间窗")
    axes[0, 0].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    for index, value in enumerate(ratio_pen):
        axes[0, 0].text(index + 0.2, value + 0.012, f"{value:.2f}", ha="center", fontsize=8.5)
    axes[0, 0].set_xticks(positions, CHANNEL_NAMES)
    axes[0, 0].set_ylim(0, max(1.2, float(ratio_pen.max()) * 1.25))
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_title("真实数据 / 训练集 输入幅度比\n(checkpoint 正规化 std,中位数;1.0 = 与训练一致)")
    axes[0, 0].set_ylabel("std 比值")

    lags = np.asarray(quality["lag_curve_lags"])
    curve = np.asarray(quality["lag_curve_median"])
    axes[0, 1].plot(lags, curve, color="#2f6fdb")
    axes[0, 1].axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
    axes[0, 1].axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    axes[0, 1].set_xlabel("滞后 (s)")
    axes[0, 1].set_ylabel("相关系数 (中位数)")
    axes[0, 1].set_title("IMU 运动包络 vs 平板笔速:滞后扫描\n(峰值平坦 = 无法确定对齐)")

    gravity = quality["gravity_cos_mean"]
    names = list(gravity)
    values = [gravity[name] for name in names]
    axes[1, 0].bar(range(len(names)), values, color=["#2f6fdb", "#7aa7e8", "#d1542b", "#e8a07a"])
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_xticks(range(len(names)), names, fontsize=8, rotation=15)
    axes[1, 0].set_ylim(0, 1.15)
    axes[1, 0].set_ylabel("mean |cos|")
    axes[1, 0].set_title(f"四元数重力方向 vs 加速度计方向\n(理想=1.0,最优约定 {quality['gravity_cos_best_convention']}={quality['gravity_cos_best']:.2f})")

    r2 = quality["r2_quantiles"]
    axes[1, 1].bar([0], [quality["r2_median"]], color="#2f6fdb", width=0.5)
    axes[1, 1].errorbar([0], [quality["r2_median"]],
                        yerr=[[quality["r2_median"] - r2[0]], [r2[2] - quality["r2_median"]]],
                        fmt="none", ecolor="black", capsize=6)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xticks([0], ["d(acc)/dt vs -ω×acc"])
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("R²")
    axes[1, 1].set_title(f"陀螺-加速度计自洽性\n(中位数 {quality['r2_median']:.2f},理想≈1)")
    fig.suptitle("真实数据(200 Hz)IMU 质量诊断", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output)
    plt.close(fig)


def _load_eval(output_dir: Path) -> tuple[list[dict[str, str]], np.lib.npyio.NpzFile | None]:
    with (output_dir / "per_trial.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    path = output_dir / "trajectories.npz"
    return rows, (np.load(path, allow_pickle=True) if path.exists() else None)


def figure_trajectories(rows: list[dict[str, str]], data: np.lib.npyio.NpzFile,
                        output: Path, per_condition: int = 3) -> None:
    import matplotlib.pyplot as plt

    _style()
    letters = data["labels"]
    gt_probs = data["gt_probs"]
    pred_probs = data["pred_probs"]
    conditions = []
    for row in rows:
        if row["condition"] not in conditions:
            conditions.append(row["condition"])
    n_rows, n_cols = len(conditions), per_condition
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.9 * n_rows))
    axes = np.atleast_2d(axes)
    for row_index, condition in enumerate(conditions):
        picked = []
        for index, row in enumerate(rows):
            if row["condition"] != condition or row["skipped"] not in ("0", ""):
                continue
            if row["letter"] in picked:
                continue
            picked.append(row["letter"])
            axis = axes[row_index][len(picked) - 1]
            gt = data[f"gt_{index}"]
            pred = data[f"pred_{index}"]
            pred = pred - pred.mean(axis=0) + gt.mean(axis=0)
            axis.plot(gt[:, 0], gt[:, 1], color="#2f6fdb", linewidth=2.0, label="GT (平板)")
            axis.plot(pred[:, 0], pred[:, 1], color="#d1542b", linewidth=1.6, linestyle="--", label="重建 (IMU)")
            axis.invert_yaxis()
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xticks([])
            axis.set_yticks([])
            gt_hit = int(np.argmax(gt_probs[index])) == int(letters[index])
            pred_hit = int(np.argmax(pred_probs[index])) == int(letters[index])
            axis.set_title(
                f"{LABELS[int(letters[index])]}  GT识别:{'对' if gt_hit else '错'} / 重建识别:{'对' if pred_hit else '错'}",
                fontsize=10,
            )
            if len(picked) >= per_condition:
                break
        axes[row_index][0].set_ylabel(condition, fontsize=9)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("真实数据:平板 GT 轨迹 vs IMU 重建轨迹", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output)
    plt.close(fig)


def figure_recognition(report: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    per_gt = report["character"]["per_letter_gt"]
    per_recon = report["character"]["per_letter_recon"]
    letters = [label for label in LABELS if label in per_gt]
    gt_values = [per_gt[label]["top1"] * 100 for label in letters]
    recon_values = [per_recon[label]["top1"] * 100 for label in letters]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    positions = np.arange(len(letters))
    axes[0].bar(positions - 0.2, gt_values, width=0.4, color="#2f6fdb", label="GT 轨迹")
    axes[0].bar(positions + 0.2, recon_values, width=0.4, color="#d1542b", label="重建轨迹")
    axes[0].set_xticks(positions, letters, fontsize=8)
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("top-1 准确率 (%)")
    axes[0].set_title("26 字母识别率(真实数据,n=%d)" % report["character"]["ground_truth"]["n"])
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)
    for position, value in zip(positions, gt_values):
        axes[0].text(position - 0.2, value + 2, f"{value:.0f}", ha="center", fontsize=6.5, color="#2f6fdb")

    confusion = np.asarray(report["character"]["confusion_recon"], dtype=np.float64)
    row_sum = confusion.sum(axis=1, keepdims=True)
    normalised = np.divide(confusion, np.maximum(row_sum, 1.0))
    image = axes[1].imshow(normalised, cmap="magma", vmin=0, vmax=max(0.6, float(normalised.max())))
    axes[1].set_xticks(range(len(letters)), letters, fontsize=6.5)
    axes[1].set_yticks(range(len(letters)), letters, fontsize=6.5)
    axes[1].set_xlabel("预测")
    axes[1].set_ylabel("真实")
    axes[1].set_title("重建轨迹混淆矩阵(行归一化)")
    fig.colorbar(image, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def figure_writing_style(raw_root: Path, output: Path, letters: Sequence[str] = ("T", "X", "K", "E", "H", "L")) -> None:
    """Show that this dataset draws every letter in a single stroke (one down/up per trial)."""

    import matplotlib.pyplot as plt

    _style()
    trials = read_manifest(raw_root)
    fig, axes = plt.subplots(2, len(letters), figsize=(2.1 * len(letters), 4.6))
    for column, letter in enumerate(letters):
        picks = [trial for trial in trials if trial.letter == letter.lower()][:2]
        for row, trial in enumerate(picks):
            gt = read_gt(trial.gt_path)
            strokes = sum(1 for line in open(trial.gt_raw_path, encoding="utf-8")
                          if ",down," in line or line.rstrip().endswith(",down"))
            xy = gt.xy_px / np.asarray([gt.width, gt.height])
            axis = axes[row][column]
            axis.plot(xy[:, 0], xy[:, 1], "-", color="#2f6fdb", linewidth=1.6)
            axis.plot(xy[0, 0], xy[0, 1], "o", color="#2e9e4f", markersize=5)
            axis.plot(xy[-1, 0], xy[-1, 1], "s", color="#d1542b", markersize=4)
            axis.invert_yaxis()
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"{letter}\n{strokes} 笔", fontsize=11)
    axes[0][0].set_ylabel("起点● 终点■", fontsize=9)
    fig.suptitle("真实数据 GT 书写风格:所有字母均为一笔连写(每个 trial 仅 1 个 down 事件)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output)
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the lab 200 Hz dataset and plot evaluation results")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_data_eval"))
    parser.add_argument("--trajectory-checkpoint", type=Path,
                        default=REPO_ROOT / "models_trajectory_topology/b2u/paper_trajectory.pt")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--gravity-cutoff", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lag-lo", type=float, default=-1.0)
    parser.add_argument("--lag-hi", type=float, default=2.5)
    parser.add_argument("--lag-step", type=float, default=0.05)
    parser.add_argument("--figures", default="diagnostic,trajectories,recognition,style",
                        help="comma separated: diagnostic,trajectories,recognition,style")
    parser.add_argument("--skip-metrics", action="store_true", help="only regenerate figures")
    parser.add_argument("--progress", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    import torch

    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {name.strip() for name in args.figures.split(",") if name.strip()}

    quality: dict[str, Any] | None = None
    quality_path = args.output_dir / "quality.json"
    if not args.skip_metrics:
        checkpoint = torch.load(args.trajectory_checkpoint, map_location="cpu")
        checkpoint_std = np.asarray(checkpoint["std"], dtype=np.float64)
        trials = read_manifest(args.raw_root, args.conditions, args.sources)
        if args.stride > 1:
            trials = trials[:: args.stride]
        if args.limit:
            trials = trials[: args.limit]
        lags = np.arange(args.lag_lo, args.lag_hi + 1e-9, args.lag_step)
        rows = []
        for index, trial in enumerate(trials, start=1):
            try:
                rows.append(trial_quality(trial, checkpoint_std, args.fs, args.gravity_cutoff, lags))
            except Exception as error:  # pragma: no cover - per-trial robustness
                print(f"[quality] skip {trial.trial}: {error}")
            if args.progress and index % args.progress == 0:
                print(f"[quality] {index}/{len(trials)}")
        quality = summarize_quality(rows, checkpoint_std, lags)
        quality["worst_trials"] = sorted(
            rows, key=lambda row: (row["corr_best_lag"] if np.isfinite(row["corr_best_lag"]) else 9.9)
        )[:20]
        with quality_path.open("w", encoding="utf-8") as handle:
            json.dump(quality, handle, ensure_ascii=False, indent=2)
        print(json.dumps({key: quality[key] for key in (
            "trials", "input_std_ratio_median", "gyro_acc_r2_median", "gravity_cos_best",
            "corr_zero_lag_median", "corr_best_lag_median", "best_lag_median_s", "slack_median_s")},
            ensure_ascii=False, indent=1))
    elif quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))

    if "diagnostic" in wanted:
        if quality is None:
            raise SystemExit("--skip-metrics needs an existing quality.json")
        figure_diagnostic(quality, args.output_dir / "fig_real_data_diagnostic.png")
        print(f"[figure] {args.output_dir / 'fig_real_data_diagnostic.png'}")
    if "style" in wanted:
        figure_writing_style(args.raw_root, args.output_dir / "fig_real_data_writing_style.png")
        print(f"[figure] {args.output_dir / 'fig_real_data_writing_style.png'}")
    if {"trajectories", "recognition"} & wanted:
        rows, data = _load_eval(args.output_dir)
        if data is not None and "trajectories" in wanted:
            figure_trajectories(rows, data, args.output_dir / "fig_real_data_trajectories.png")
            print(f"[figure] {args.output_dir / 'fig_real_data_trajectories.png'}")
        if "recognition" in wanted:
            report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
            figure_recognition(report, args.output_dir / "fig_real_data_recognition.png")
            print(f"[figure] {args.output_dir / 'fig_real_data_recognition.png'}")


if __name__ == "__main__":
    main()
