from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .paper_trajectory import load_trajectory_predictor
from .paper_trajectory_dataset import (
    discover_trajectory_samples,
    load_sample_arrays,
    sample_is_valid,
)
from .train_character_classifier import character_split, predict_run_points
from .train_paper_trajectory import set_seed
from .word_dataset import (
    WORD_ACTIONS,
    build_word_examples,
    build_word_recon_examples,
    discover_word_samples,
)
from .word_recognition import evaluate_nearest


DEFAULT_CLEAN_ROOT = Path(
    "/data/huyang/datasets/WritingRing/clean_data_delete_g/data"
)
DEFAULT_RAW_ROOT = Path("/data/huyang/datasets/WritingRing/data")
DEFAULT_RECONSTRUCTION = Path(
    "/data/huyang/trae_projects/Ring/models_trajectory_topology/x3/paper_trajectory.pt"
)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    clean_root = Path(args.data_root)
    raw_root = Path(args.raw_root)
    samples = discover_word_samples(clean_root, raw_root)
    if not samples:
        raise ValueError("no word samples discovered")
    train_samples, _, test_samples, split_info = character_split(
        samples, args.split, seed=args.seed
    )
    gt_train = build_word_examples(
        train_samples, clean_root, args.resample_frames, deskew=args.deskew
    )
    gt_test = build_word_examples(
        test_samples, clean_root, args.resample_frames, deskew=args.deskew
    )

    recon_info: dict[str, Any] = {"used": False}
    recon_train: list[dict] = []
    recon_test: list[dict] = []
    if args.source in {"recon", "both"}:
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = load_trajectory_predictor(args.reconstruction_checkpoint, device=device)
        blocks = {
            (block.user, block.action, block.sample_id): block
            for block in discover_trajectory_samples(clean_root)
            if block.action in WORD_ACTIONS and sample_is_valid(block)
        }
        run_points: dict[tuple[str, int, str], dict] = {}
        needed = {
            (sample.user, sample.action, sample.sample_id)
            for sample in train_samples + test_samples
        }
        for key in needed:
            block = blocks[key]
            x, _, mask, _, _ = load_sample_arrays(block, mmap=False)
            run_points[key] = predict_run_points(
                predictor, np.asarray(x), np.asarray(mask), args.chunk_len
            )
        recon_train = build_word_recon_examples(
            train_samples, run_points, args.resample_frames, deskew=args.deskew
        )
        recon_test = build_word_recon_examples(
            test_samples, run_points, args.resample_frames, deskew=args.deskew
        )
        recon_info = {
            "used": True,
            "checkpoint": str(args.reconstruction_checkpoint),
            "counts": {"train": len(recon_train), "test": len(recon_test)},
        }

    if args.source == "gt":
        training = gt_train
        evaluations = {"gt": gt_test}
    elif args.source == "recon":
        training = recon_train
        evaluations = {"recon": recon_test}
    else:
        training = gt_train + recon_train
        evaluations = {"gt": gt_test, "recon": recon_test}

    test_metrics = {
        name: evaluate_nearest(
            training, examples, topk=args.topk, metric=args.metric, rerank=args.rerank
        )
        for name, examples in evaluations.items()
        if examples
    }

    report = {
        "data_root": str(clean_root),
        "raw_root": str(raw_root),
        "split": split_info,
        "source": args.source,
        "sample_counts": {"train": len(train_samples), "test": len(test_samples)},
        "example_counts": {
            "gt_train": len(gt_train),
            "gt_test": len(gt_test),
            "recon_train": len(recon_train),
            "recon_test": len(recon_test),
        },
        "reconstruction": recon_info,
        "config": {
            "resample_frames": args.resample_frames,
            "metric": args.metric,
            "rerank": args.rerank,
            "deskew": args.deskew,
            "topk": args.topk,
            "seed": args.seed,
        },
        "test": test_metrics,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "word_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    for name, metrics in test_metrics.items():
        print(
            f"test[{name}] overall n={metrics['overall']['n']} "
            f"top1={metrics['overall']['top1']:.4f} top5={metrics['overall']['top5']:.4f} | "
            f"connected {metrics['connected']['top1']:.4f} | "
            f"unconnected {metrics['unconnected']['top1']:.4f} | "
            f"closed {metrics['closed']['top1']:.4f} (n={metrics['closed']['n']})"
        )
    print(f"saved report: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nearest-template word recognition on trajectories"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--reconstruction-checkpoint", type=Path, default=DEFAULT_RECONSTRUCTION
    )
    parser.add_argument("--split", choices=("user", "random"), default="user")
    parser.add_argument("--source", choices=("gt", "recon", "both"), default="both")
    parser.add_argument("--resample-frames", type=int, default=128)
    parser.add_argument("--metric", choices=("euclidean", "dtw"), default="euclidean")
    parser.add_argument("--rerank", type=int, default=50)
    parser.add_argument(
        "--deskew", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--chunk-len", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("models_words"))
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
