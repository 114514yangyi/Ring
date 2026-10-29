"""Prepare the ``wrist-resting-new-data`` drop for :mod:`build_lab_dataset`.

The drop (``/data/dataset/wrist-resting-new-data``) ships the three CSV files per trial but
no manifest, so this script derives everything ``build_lab_dataset`` needs:

* ``<out>/_meta/manifest.csv`` -- condition / letter / trial / absolute csv paths / per-trial
  clock ``delta``;
* ``<out>/_meta/trial_scale.json`` -- per-trial ``mm_per_px`` for ``--trial-scale-json``;
* ``<out>/_meta/scale_report.json`` -- the evidence behind both (clock gaps, bbox heights).

Two conventions matter:

1. **Clock.** The tablet ``gt_raw.timestamp`` is host epoch and the ring
   ``estimated_measurement_at`` is the same host epoch (``received_at = estimated + ~35 ms``),
   so the absolute mapping ``device_time = gt_time - epoch`` with ``epoch = median(received_at -
   timestamp)`` already aligns the streams to a few tens of milliseconds.  ``delta`` therefore
   has to stay ``0`` -- copying the legacy ``t_down - received_at[0]`` convention would pin the
   pen-down onto the first ring frame and throw away the ~0.67 s of pre-stroke recording.  The
   ``wangtiancheng`` trials are the exception: their clock is ~591 s off, and the offset drifts
   by a few hundred milliseconds between trials, so ``delta`` is measured per trial by
   correlating the pen-speed profile with the ring motion envelope over a +-4 s window.

2. **Scale.** The drop has no guide box, so ``mm_per_px`` cannot be derived from a 3x3/5x5 cm
   box.  The recordings were made on four canvases (400x785, 1024x661, 1080x682, 1080x728) and
   a canvas change rescales the same physical stroke, so the anchor is "one written letter is
   ``--letter-height-mm`` tall": ``mm_per_px = anchor / median bbox height`` per
   ``(user, canvas)``.  This keeps every writer on a comparable millimetre scale while leaving
   the trial-to-trial size variation intact.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from .build_lab_dataset import motion_profile, moving_average, pen_speed_profile, remove_gravity
from .evaluate_real_data import DEFAULT_FS, pen_runs, read_imu, uniform_grid

DEFAULT_RAW_ROOT = Path("/data/dataset/wrist-resting-new-data")
DEFAULT_OUT_ROOT = Path("/data/huyang/datasets/RingLab/wrist_resting_raw")
DEFAULT_CONDITION = "wrist-resting-tablet"
WIDE_GAP_S = 5.0  # |gt_start - ring_start| above this means a broken clock, not a lead-in


def trial_records(raw_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for letter_dir in sorted(raw_root.iterdir()):
        if not letter_dir.is_dir() or len(letter_dir.name) != 1:
            continue
        letter = letter_dir.name.upper()
        for gt_raw in sorted(letter_dir.glob("*_gt_raw.csv")):
            name = gt_raw.name[: -len("_gt_raw.csv")]
            parts = name.rsplit("_", 2)
            if len(parts) != 3:
                continue
            file_letter, user, index = parts
            imu = letter_dir / f"{name}_imu.csv"
            gt_100hz = letter_dir / f"{name}_gt_100hz.csv"
            if not imu.exists() or not gt_100hz.exists():
                continue
            records.append(
                {
                    "trial": name,
                    "letter": letter,
                    "letter_idx": ord(letter) - ord("A"),
                    "user": user,
                    "index": int(index),
                    "imu_path": imu,
                    "gt_raw_path": gt_raw,
                    "gt_100hz_path": gt_100hz,
                    "file_letter": file_letter,
                }
            )
    return records


def gt_summary(path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    ys = [float(row["y"]) for row in rows]
    downs = [float(row["timestamp"]) for row in rows if row["event_type"].strip().lower() == "down"]
    return {
        "rows": len(rows),
        "width": float(rows[0]["width"]),
        "height": float(rows[0]["height"]),
        "bbox_h_px": float(max(ys) - min(ys)) if ys else 0.0,
        "down_t": downs[0] if downs else (float(rows[0]["timestamp"]) if rows else float("nan")),
        "downs": len(downs),
    }


def clock_delta(record: dict[str, object], fs: float, cutoff_hz: float,
                search_half_width: float) -> dict[str, object]:
    """Per-trial clock ``delta`` plus the evidence used to decide it."""

    imu = read_imu(record["imu_path"])  # type: ignore[arg-type]
    info = gt_summary(record["gt_raw_path"])  # type: ignore[arg-type]
    epoch = float(np.median(imu.received_at - imu.t))
    gap = float(info["down_t"]) - float(imu.received_at[0])
    result: dict[str, object] = {"epoch": epoch, "gap_s": gap, "delta": 0.0, "lag": 0.0,
                                 "lag_score": float("nan"), "lag_alt": float("nan")}
    if abs(gap) <= WIDE_GAP_S:
        return result

    # ``gap`` is measured straight from the data (first pen-down vs the first IMU frame
    # timestamp) and it is the whole offset: at ``delta = gap`` the pen window sits inside the
    # recording with 0.98-1.00 coverage.  Correlating a *narrow band* around it was tried and
    # rejected -- the speed profile of a single short letter has no distinctive peak (best
    # score landed on the search boundary for 172/284 trials), because quasi-periodic stroke
    # structure makes the correlation surface ambiguous.  A wide band is worse still: it locks
    # onto a false peak ~2.5 s away (~56% of trials then fail the coverage gate).
    result.update({"delta": gap, "lag": -gap, "lag_score": float("nan"), "lag_alt": gap})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument("--letter-height-mm", type=float, default=30.0)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--gravity-cutoff", type=float, default=1.0)
    parser.add_argument("--users", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress", type=int, default=400)
    args = parser.parse_args()

    records = trial_records(args.raw_root)
    if args.users:
        wanted = set(args.users)
        records = [record for record in records if record["user"] in wanted]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"no trials under {args.raw_root}")
    print(f"[wrist-manifest] {len(records)} trials", flush=True)

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        info = gt_summary(record["gt_raw_path"])  # type: ignore[arg-type]
        clock = clock_delta(record, args.fs, args.gravity_cutoff, search_half_width=0.4)
        rows.append({
            **record,
            **{f"gt_{key}": value for key, value in info.items()},
            **{f"clock_{key}": value for key, value in clock.items()},
        })
        if args.progress and index % args.progress == 0:
            print(f"[wrist-manifest] {index}/{len(records)}", flush=True)

    heights: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        canvas = f"{int(row['gt_width'])}x{int(row['gt_height'])}"
        if float(row["gt_bbox_h_px"]) > 0:
            heights[(str(row["user"]), canvas)].append(float(row["gt_bbox_h_px"]))
    scale_table: dict[tuple[str, str], float] = {}
    for key, values in heights.items():
        median_h = statistics.median(values)
        scale_table[key] = args.letter_height_mm / median_h

    trial_scale: dict[str, float] = {}
    for row in rows:
        canvas = f"{int(row['gt_width'])}x{int(row['gt_height'])}"
        scale = scale_table.get((str(row["user"]), canvas))
        if scale is None:
            raise SystemExit(f"no bbox height for {row['trial']} ({row['user']} {canvas})")
        trial_scale[str(row["trial"])] = scale

    args.out_root.mkdir(parents=True, exist_ok=True)
    meta = args.out_root / "_meta"
    meta.mkdir(exist_ok=True)
    manifest_path = meta / "manifest.csv"
    fieldnames = ["condition", "letter", "letter_idx", "trial", "imu_path", "gt_100hz_path",
                  "gt_raw_path", "delta", "user", "canvas", "device_mm_per_px", "clock_gap_s"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            canvas = f"{int(row['gt_width'])}x{int(row['gt_height'])}"
            writer.writerow({
                "condition": args.condition,
                "letter": row["letter"],
                "letter_idx": row["letter_idx"],
                "trial": row["trial"],
                "imu_path": Path(row["imu_path"]).as_posix(),  # type: ignore[arg-type]
                "gt_100hz_path": Path(row["gt_100hz_path"]).as_posix(),  # type: ignore[arg-type]
                "gt_raw_path": Path(row["gt_raw_path"]).as_posix(),  # type: ignore[arg-type]
                "delta": f"{float(row['clock_delta']):.6f}",
                "user": row["user"],
                "canvas": canvas,
                "device_mm_per_px": f"{trial_scale[str(row['trial'])]:.6f}",
                "clock_gap_s": f"{float(row['clock_gap_s']):.3f}",
            })
    (meta / "trial_scale.json").write_text(json.dumps(trial_scale, indent=2, sort_keys=True),
                                         encoding="utf-8")

    per_user: dict[str, dict[str, object]] = {}
    for (user, canvas), scale in sorted(scale_table.items()):
        entry = per_user.setdefault(user, {"canvases": {}, "trials": 0})
        heights_key = heights[(user, canvas)]
        entry["canvases"][canvas] = {  # type: ignore[index]
            "trials": len(heights_key),
            "median_bbox_h_px": round(statistics.median(heights_key), 2),
            "mm_per_px": round(scale, 6),
        }
        entry["trials"] = int(entry["trials"]) + len(heights_key)  # type: ignore[arg-type]
    clock_rows = [row for row in rows if abs(float(row["clock_gap_s"])) > WIDE_GAP_S]
    report = {
        "raw_root": str(args.raw_root),
        "out_root": str(args.out_root),
        "condition": args.condition,
        "trials": len(rows),
        "users": sorted({str(row["user"]) for row in rows}),
        "letter_height_anchor_mm": args.letter_height_mm,
        "scale_convention": "mm_per_px = letter_height_mm / median(gt_raw bbox height) per (user, canvas)",
        "per_user_scale": per_user,
        "clock_fixed_trials": len(clock_rows),
        "clock_fixed_users": sorted({str(row["user"]) for row in clock_rows}),
        "clock_gap_s": {
            "median": round(statistics.median(float(row["clock_gap_s"]) for row in rows), 4),
            "fixed_median": round(statistics.median(float(row["clock_lag"]) for row in clock_rows), 4)
            if clock_rows else None,
            "fixed_min": round(min(float(row["clock_lag"]) for row in clock_rows), 4) if clock_rows else None,
            "fixed_max": round(max(float(row["clock_lag"]) for row in clock_rows), 4) if clock_rows else None,
            "fixed_alt_spread_max": round(max(
                abs(float(row["clock_lag"]) - float(row["clock_lag_alt"])) for row in clock_rows
            ), 4) if clock_rows else None,
            "delta_note": "delta = 0 for aligned clocks (absolute epoch mapping); "
                          "delta = clock gap (+small correlation correction) for the ~591 s clock-offset trials",
        },
        "canvas_counts": dict(sorted(
            __import__("collections").Counter(
                f"{int(row['gt_width'])}x{int(row['gt_height'])}" for row in rows
            ).items()
        )),
        "multi_stroke_trials": sum(1 for row in rows if int(row["gt_downs"]) > 1),
        "missing_down_trials": sum(1 for row in rows if int(row["gt_downs"]) == 0),
    }
    (meta / "scale_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "trials": len(rows),
                      "users": len(report["users"]),
                      "clock_fixed_trials": report["clock_fixed_trials"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
