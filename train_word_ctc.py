from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .paper_trajectory import load_trajectory_predictor
from .paper_trajectory_dataset import (
    discover_trajectory_samples,
    load_sample_arrays,
    sample_is_valid,
)
from .train_character_classifier import (
    augment_trajectory,
    character_split,
    predict_run_points,
)
from .train_paper_trajectory import set_seed
from .word_ctc import (
    WordCTC,
    greedy_ctc_decode,
    snap_to_vocabulary,
)
from .word_dataset import (
    WORD_ACTIONS,
    build_word_examples,
    build_word_recon_examples,
    discover_word_samples,
)


DEFAULT_CLEAN_ROOT = Path(
    "/data/huyang/datasets/WritingRing/clean_data_delete_g/data"
)
DEFAULT_RAW_ROOT = Path("/data/huyang/datasets/WritingRing/data")
DEFAULT_RECONSTRUCTION = Path(
    "/data/huyang/trae_projects/Ring/models_trajectory_topology/x3/paper_trajectory.pt"
)


def example_target(word: str) -> np.ndarray | None:
    if not word.isalpha() or not word.isascii() or not word.islower():
        return None
    return np.asarray([ord(char) - ord("a") + 1 for char in word], dtype=np.int64)


def collate_ctc_examples(batch: Sequence[tuple[np.ndarray, np.ndarray]]):
    inputs = torch.stack([torch.from_numpy(item[0]).float() for item in batch])
    targets = torch.cat([torch.from_numpy(item[1]).long() for item in batch])
    input_lengths = torch.full(
        (len(batch),), inputs.shape[1], dtype=torch.long
    )
    target_lengths = torch.tensor(
        [len(item[1]) for item in batch], dtype=torch.long
    )
    return inputs, targets, input_lengths, target_lengths


class CtcExampleDataset(Dataset):
    def __init__(self, examples: Sequence[dict], augment: bool = False, seed: int = 0):
        self.examples = list(examples)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        trajectory = example["trajectory"].astype(np.float32)
        if self.augment:
            trajectory = augment_trajectory(trajectory, self.rng).astype(np.float32)
        return trajectory, example["target"]


def decode_examples(
    model: WordCTC,
    examples: Sequence[dict],
    device: str,
    vocabulary: Sequence[str] | set[str],
    batch_size: int = 128,
) -> dict[str, Any]:
    if not examples:
        return {"n": 0, "raw_top1": float("nan"), "snapped_top1": float("nan")}
    loader = DataLoader(
        CtcExampleDataset(examples), batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: (
            torch.stack([torch.from_numpy(item[0]).float() for item in batch]),
            [item[1] for item in batch],
        ),
    )
    model.eval()
    raw_hits = 0
    snapped_hits = 0
    total = 0
    by_group: dict[str, list[int]] = {}
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs.to(device)).cpu().numpy()
            for index in range(inputs.shape[0]):
                example = examples[total]
                decoded = greedy_ctc_decode(logits[:, index, :])
                snapped = snap_to_vocabulary(decoded, vocabulary)
                hit_raw = int(decoded == example["word"])
                hit_snapped = int(snapped == example["word"])
                raw_hits += hit_raw
                snapped_hits += hit_snapped
                by_group.setdefault(example["group"], []).append(hit_snapped)
                total += 1
    metrics = {
        "n": total,
        "raw_top1": raw_hits / total,
        "snapped_top1": snapped_hits / total,
        "vocabulary_size": len(vocabulary),
    }
    for group, hits in by_group.items():
        metrics[group] = {"n": len(hits), "snapped_top1": sum(hits) / len(hits)}
    return metrics


def _with_targets(examples: Sequence[dict]) -> list[dict]:
    filtered = []
    for example in examples:
        target = example_target(example["word"])
        if target is None or len(target) >= 128:
            continue
        item = dict(example)
        item["target"] = target
        filtered.append(item)
    return filtered


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    clean_root = Path(args.data_root)
    raw_root = Path(args.raw_root)
    samples = discover_word_samples(clean_root, raw_root)
    if not samples:
        raise ValueError("no word samples discovered")
    train_samples, val_samples, test_samples, split_info = character_split(
        samples, args.split, seed=args.seed
    )

    gt_train = _with_targets(
        build_word_examples(train_samples, clean_root, args.resample_frames)
    )
    gt_val = _with_targets(
        build_word_examples(val_samples, clean_root, args.resample_frames)
    )
    gt_test = _with_targets(
        build_word_examples(test_samples, clean_root, args.resample_frames)
    )

    recon_info: dict[str, Any] = {"used": False}
    recon_train: list[dict] = []
    recon_val: list[dict] = []
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
            for sample in train_samples + val_samples + test_samples
        }
        for key in needed:
            x, _, mask, _, _ = load_sample_arrays(blocks[key], mmap=False)
            run_points[key] = predict_run_points(
                predictor, np.asarray(x), np.asarray(mask), args.chunk_len
            )
        recon_train = _with_targets(
            build_word_recon_examples(train_samples, run_points, args.resample_frames)
        )
        recon_val = _with_targets(
            build_word_recon_examples(val_samples, run_points, args.resample_frames)
        )
        recon_test = _with_targets(
            build_word_recon_examples(test_samples, run_points, args.resample_frames)
        )
        recon_info = {
            "used": True,
            "checkpoint": str(args.reconstruction_checkpoint),
            "counts": {
                "train": len(recon_train),
                "val": len(recon_val),
                "test": len(recon_test),
            },
        }

    if args.source == "gt":
        train_examples, val_examples, test_examples = gt_train, gt_val, gt_test
    elif args.source == "recon":
        train_examples, val_examples, test_examples = (
            recon_train,
            recon_val,
            recon_test,
        )
    else:
        train_examples = gt_train + recon_train
        val_examples = gt_val + recon_val
        test_examples = gt_test + recon_test
    if not train_examples or not test_examples:
        raise ValueError("empty CTC train/test examples")

    vocabulary = sorted({example["word"] for example in train_examples})

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WordCTC(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    train_loader = DataLoader(
        CtcExampleDataset(train_examples, augment=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ctc_examples,
    )

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for inputs, targets, input_lengths, target_lengths in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(
                logits.log_softmax(dim=-1), targets, input_lengths, target_lengths
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        val_metrics = decode_examples(
            model, val_examples, device, vocabulary, args.eval_batch_size
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / batches if batches else 0.0,
            "val_snapped_top1": val_metrics["snapped_top1"],
            "val_raw_top1": val_metrics["raw_top1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_snapped={row['val_snapped_top1']:.4f} raw={row['val_raw_top1']:.4f}"
        )
        if val_metrics["snapped_top1"] > best_score:
            best_score = val_metrics["snapped_top1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = decode_examples(
        model, test_examples, device, vocabulary, args.eval_batch_size
    )
    for source in ("gt", "recon"):
        subset = [example for example in test_examples if example["source"] == source]
        if subset:
            test_metrics[source] = decode_examples(
                model, subset, device, vocabulary, args.eval_batch_size
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "word_ctc.pt"
    report_path = output_dir / "word_ctc_report.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "config": {"resample_frames": args.resample_frames, "dropout": args.dropout},
            "split": split_info,
        },
        checkpoint_path,
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "data_root": str(clean_root),
        "raw_root": str(raw_root),
        "split": split_info,
        "source": args.source,
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "example_counts": {
            "train": len(train_examples),
            "val": len(val_examples),
            "test": len(test_examples),
        },
        "vocabulary_size": len(vocabulary),
        "reconstruction": recon_info,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": device,
        },
        "history": history,
        "test": test_metrics,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved report: {report_path}")
    print(
        "test overall "
        f"n={test_metrics['n']} raw={test_metrics['raw_top1']:.4f} "
        f"snapped={test_metrics['snapped_top1']:.4f}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CTC letter-sequence word recognizer"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--reconstruction-checkpoint", type=Path, default=DEFAULT_RECONSTRUCTION
    )
    parser.add_argument("--split", choices=("user", "random"), default="user")
    parser.add_argument("--source", choices=("gt", "recon", "both"), default="both")
    parser.add_argument("--resample-frames", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--chunk-len", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("models_words/ctc"))
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
