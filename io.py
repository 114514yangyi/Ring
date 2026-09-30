from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd


COLUMN_ALIASES = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "TimeStamp (s)": "timestamp",
    "acc_x": "acc_x",
    "acc_y": "acc_y",
    "acc_z": "acc_z",
    "AccX (g)": "acc_x",
    "AccY (g)": "acc_y",
    "AccZ (g)": "acc_z",
    "gyro_x": "gyro_x",
    "gyro_y": "gyro_y",
    "gyro_z": "gyro_z",
    "GyroX (deg/s)": "gyro_x",
    "GyroY (deg/s)": "gyro_y",
    "GyroZ (deg/s)": "gyro_z",
    "lin_acc_x": "lin_acc_x",
    "lin_acc_y": "lin_acc_y",
    "lin_acc_z": "lin_acc_z",
    "LinAccX (g)": "lin_acc_x",
    "LinAccY (g)": "lin_acc_y",
    "LinAccZ (g)": "lin_acc_z",
}


def read_sensor_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    renamed = {}
    for column in df.columns:
        clean = str(column).replace("\ufeff", "").strip()
        renamed[column] = COLUMN_ALIASES.get(clean, clean)
    df = df.rename(columns=renamed)

    required = {"timestamp", "gyro_x", "gyro_y", "gyro_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for column in (
        "timestamp",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "lin_acc_x",
        "lin_acc_y",
        "lin_acc_z",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["timestamp", "gyro_x", "gyro_y", "gyro_z"]).reset_index(drop=True)
    if not {"lin_acc_x", "lin_acc_y", "lin_acc_z"}.issubset(df.columns):
        if {"acc_x", "acc_y", "acc_z"}.issubset(df.columns):
            # For files without linear acceleration, remove each axis' static
            # mean. This is only a fallback and should be replaced upstream.
            raw = df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
            linear = raw - raw.mean(axis=0, keepdims=True)
            df[["lin_acc_x", "lin_acc_y", "lin_acc_z"]] = linear
        else:
            raise ValueError(f"{path} has no acceleration columns")
    return df


def iter_samples(df: pd.DataFrame) -> Iterator[dict[str, Any]]:
    for row in df.to_dict(orient="records"):
        yield row
