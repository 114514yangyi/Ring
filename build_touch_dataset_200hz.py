"""Convert paired 200 Hz letter-writing recordings into touch-model inputs.

The source dataset is described by ``200hz/_meta/manifest.csv``. Its IMU and
touchscreen clocks differ for every trial; the manifest's ``delta`` is added to
IMU ``received_at`` before deriving the binary contact mask from ``down``
through ``up`` (or ``cancel``) events.

Output follows ``paper_dataset.discover_cleaned_samples``:
``user_<split bucket>/0/<id>_x.npy`` and ``<id>_mask.npy``. A deterministic
hash of each trial identity assigns it to one of ten pseudo-users, making the
existing split trial-disjoint rather than writer-disjoint.
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "lin_acc_x", "lin_acc_y", "lin_acc_z", "gyro_x", "gyro_y", "gyro_z"
]
TERMINAL_EVENTS = {"up", "cancel"}


def _contact_interval(gt: pd.DataFrame) -> tuple[float, float] | None:
    """Return the first valid down-to-terminal contact interval."""
    events = gt[["timestamp", "event_type"]].copy()
    events["event_type"] = events["event_type"].astype(str).str.lower()
    for down_index in events.index[events["event_type"] == "down"]:
        terminals = events.loc[down_index + 1 :]
        terminals = terminals[terminals["event_type"].isin(TERMINAL_EVENTS)]
        if not terminals.empty:
            return float(events.at[down_index, "timestamp"]), float(terminals.iloc[0]["timestamp"])
    return None


def _trial_number(row: pd.Series) -> int:
    """Extract the recording repetition from names such as ``a_fk_9``."""
    match = re.search(r"_(\d+)$", str(row.get("trial", "")))
    if match:
        return int(match.group(1))
    return int(row.trial_idx)


def _split_bucket(row: pd.Series) -> int:
    """Assign one independent trial to a stable, stratified split bucket."""
    identity = "|".join(str(row.get(field, "")) for field in ("condition", "letter", "trial"))
    return zlib.crc32(identity.encode("utf-8")) % 10


def convert_dataset(source_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    source_root, output_root = Path(source_root), Path(output_root)
    manifest = pd.read_csv(source_root / "_meta" / "manifest.csv", encoding="utf-8-sig")
    required = {"imu_path", "gt_raw_path", "delta", "trial_idx"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    per_user_id: Counter[int] = Counter()
    for row_index, row in manifest.iterrows():
        try:
            imu = pd.read_csv(source_root / str(row.imu_path))
            gt = pd.read_csv(source_root / str(row.gt_raw_path))
            if not set(FEATURE_COLUMNS + ["received_at"]).issubset(imu.columns):
                raise ValueError("IMU file lacks required feature or received_at columns")
            interval = _contact_interval(gt)
            if interval is None:
                raise ValueError("no down-to-up/cancel contact interval")
            aligned_time = imu["received_at"].to_numpy(dtype=np.float64) + float(row.delta)
            features = imu[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            finite = np.isfinite(features).all(axis=1) & np.isfinite(aligned_time)
            if finite.sum() < 20:
                raise ValueError("fewer than 20 finite IMU frames")
            features, aligned_time = features[finite], aligned_time[finite]
            contact_start, contact_end = interval
            mask = ((aligned_time >= contact_start) & (aligned_time <= contact_end)).astype(np.int8)
            trial_index = _trial_number(row)
            if trial_index < 1:
                raise ValueError(f"invalid trial_idx {trial_index}")
            user_index = _split_bucket(row)
            sample_id = per_user_id[user_index]
            per_user_id[user_index] += 1
            destination = output_root / f"user_{user_index}" / "0"
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / f"{sample_id}_x.npy", features)
            np.save(destination / f"{sample_id}_mask.npy", mask)
            records.append({"source_row": int(row_index), "trial": str(row.get("trial", "")), "trial_number": trial_index, "letter": str(row.get("letter", "")), "condition": str(row.get("condition", "")), "pseudo_user": f"user_{user_index}", "sample_id": str(sample_id), "frames": int(len(features)), "contact_frames": int(mask.sum()), "contact_start": contact_start, "contact_end": contact_end})
        except Exception as error:
            skipped.append({"source_row": int(row_index), "trial": str(row.get("trial", "")), "reason": str(error)})
    summary = {"source_root": str(source_root), "protocol": "trial-disjoint pseudo-user split: deterministic trial hash into 10 buckets, then existing 70/10/20 split", "warning": "pseudo-users encode trial identity, not writer identity; this is not writer-disjoint evaluation", "input_rows": int(len(manifest)), "converted_samples": int(len(records)), "skipped_samples": skipped, "records": records}
    with (output_root / "conversion_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert paired 200 Hz recordings for touch-state training")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_dataset(args.source_root, args.output_root)
    print(f"converted {summary['converted_samples']}/{summary['input_rows']} samples; skipped {len(summary['skipped_samples'])}; output={args.output_root}")


if __name__ == "__main__":
    main()
