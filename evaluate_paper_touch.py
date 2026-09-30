from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .paper_dataset import (
    PaperTouchWindowDataset,
    build_window_refs,
    class_counts,
    default_user_split,
    discover_cleaned_samples,
)
from .paper_touch import PaperTouchConfig, load_touch_model
from .train_paper_touch import evaluate


def _checkpoint_payload(path: Path, device: str) -> dict[str, Any]:
    payload = torch.load(str(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dict with state_dict/mean/std metadata")
    if "mean" not in payload or "std" not in payload:
        raise ValueError("checkpoint is missing mean/std normalization metadata")
    return payload


def _config_from_payload(payload: dict[str, Any]) -> PaperTouchConfig:
    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return PaperTouchConfig(
        sample_rate=float(cfg.get("sample_rate", 200.0)),
        window_frames=int(cfg.get("window_frames", 20)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


def _samples_by_split(samples, payload: dict[str, Any]) -> dict[str, list]:
    split = payload.get("split")
    if isinstance(split, dict) and all(
        key in split for key in ("train_users", "val_users", "test_users")
    ):
        train_users = set(split["train_users"])
        val_users = set(split["val_users"])
        test_users = set(split["test_users"])
    else:
        train_users, val_users, test_users = default_user_split(
            [sample.user for sample in samples]
        )

    return {
        "train": [sample for sample in samples if sample.user in train_users],
        "val": [sample for sample in samples if sample.user in val_users],
        "test": [sample for sample in samples if sample.user in test_users],
    }


def run(args) -> dict[str, Any]:
    payload = _checkpoint_payload(args.checkpoint, args.device)
    config = _config_from_payload(payload)
    samples = discover_cleaned_samples(args.data_root)
    if not samples:
        raise ValueError(f"no cleaned samples found under {args.data_root}")

    split_samples = _samples_by_split(samples, payload)
    selected_samples = split_samples[args.split]
    if not selected_samples:
        raise ValueError(f"split {args.split!r} has no samples")

    max_per_class = None if args.max_windows_per_class <= 0 else args.max_windows_per_class
    refs = build_window_refs(
        selected_samples,
        config,
        max_per_class=max_per_class,
        seed=args.seed,
    )
    if not refs:
        raise ValueError(f"split {args.split!r} produced no windows")

    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    dataset = PaperTouchWindowDataset(selected_samples, refs, mean, std, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = load_touch_model(args.checkpoint, config=config, device=args.device)
    metrics = evaluate(model, loader, args.device)

    report = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "split": args.split,
        "sample_count": len(selected_samples),
        "window_counts": class_counts(refs),
        "metrics": metrics,
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with args.report_out.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

    print(
        f"{args.split} "
        f"acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"event_f1={metrics['event_macro_f1']:.4f} "
        f"loss={metrics['loss']:.4f}"
    )
    for label, row in metrics["per_class"].items():
        print(
            f"  {label:7s} "
            f"precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} "
            f"f1={row['f1']:.4f} "
            f"support={row['support']}"
        )
    if args.report_out is not None:
        print(f"saved report: {args.report_out}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained WritingRing-style touch-state detector"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("writing_state/models/paper_touch_resnet.pt"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("writing_state/clean_data_delete_g/data"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=15000,
        help="set to 0 to evaluate every window in the selected split",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
