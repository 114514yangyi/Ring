from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .detector import DetectorConfig, WritingStateDetector
from .io import iter_samples, read_sensor_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect writing contact, pen-up, and invalid air movement from IMU CSV"
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--events-out", type=Path)
    parser.add_argument("--segments-out", type=Path)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument(
        "--no-implicit-start",
        action="store_true",
        help="require an impact/contact cue before writing_start",
    )
    args = parser.parse_args()

    df = read_sensor_csv(args.csv)
    detector = WritingStateDetector(
        DetectorConfig(
            sample_rate=args.sample_rate,
            allow_implicit_start=not args.no_implicit_start,
        )
    )
    events = detector.process(iter_samples(df))
    event_rows = [event.to_dict() for event in events]
    segment_rows = detector.segments

    print(json.dumps({"events": event_rows, "segments": segment_rows}, ensure_ascii=False, indent=2))
    if args.events_out:
        args.events_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(event_rows).to_csv(args.events_out, index=False, encoding="utf-8-sig")
    if args.segments_out:
        args.segments_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(segment_rows).to_csv(args.segments_out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
