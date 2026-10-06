from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .character_dataset import (
    CharacterSample,
    build_gt_examples,
    contact_runs,
    discover_character_samples,
    normalize_trajectory,
    resample_trajectory,
)
from .character_model import NUM_CLASSES, CharacterCNN
from .paper_dataset import default_user_split, user_number
from .paper_trajectory import load_trajectory_predictor
from .paper_trajectory_dataset import (
    discover_trajectory_samples,
    load_sample_arrays,
    sample_is_valid,
)
from .train_paper_trajectory import set_seed


DEFAULT_CLEAN_ROOT = Path(
    "/data/huyang/datasets/WritingRing/clean_data_delete_g/data"
)
DEFAULT_RAW_ROOT = Path("/data/huyang/datasets/WritingRing/data")
DEFAULT_RECONSTRUCTION = Path(
    "/data/huyang/trae_projects/Ring/models_trajectory_topology/x3/paper_trajectory.pt"
)


def character_split(
    samples: Sequence[CharacterSample],
    strategy: str = "user",
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> tuple[list[CharacterSample], list[CharacterSample], list[CharacterSample], dict[str, Any]]:
    if strategy == "random":
        order = list(range(len(samples)))
        random.Random(seed).shuffle(order)
        train_count = max(1, int(round(len(samples) * ratios[0])))
        val_count = max(1, int(round(len(samples) * ratios[1])))
        while train_count + val_count >= len(samples):
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                raise ValueError("cannot build a non-empty test split")
        train = [samples[index] for index in order[:train_count]]
        val = [samples[index] for index in order[train_count : train_count + val_count]]
        test = [samples[index] for index in order[train_count + val_count :]]
        info = {"strategy": "random", "seed": seed, "ratios": list(ratios)}
        return train, val, test, info
    if strategy == "user":
        train_users, val_users, test_users = default_user_split(
            [sample.user for sample in samples]
        )
        train = [sample for sample in samples if sample.user in train_users]
        val = [sample for sample in samples if sample.user in val_users]
        test = [sample for sample in samples if sample.user in test_users]
        if not train or not val or not test:
            raise ValueError("empty character split")
        info = {
            "strategy": "user",
            "train_users": sorted(train_users, key=user_number),
            "val_users": sorted(val_users, key=user_number),
            "test_users": sorted(test_users, key=user_number),
        }
        return train, val, test, info
    raise ValueError(f"unknown split strategy: {strategy}")


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    order = np.argsort(-logits, axis=1)[:, :k]
    return float((order == labels[:, None]).any(axis=1).mean())


def confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_value, predicted in zip(labels, predictions):
        matrix[int(true_value), int(predicted)] += 1
    return matrix


def predict_run_points(
    predictor: Any, x_block: np.ndarray, mask: np.ndarray, chunk_len: int = 5000
) -> dict[tuple[int, int], np.ndarray]:
    deltas = np.asarray(predictor.predict_deltas(np.asarray(x_block)), dtype=np.float64)
    points: dict[tuple[int, int], np.ndarray] = {}
    for start, stop in contact_runs(mask, min_frames=1):
        positions = np.zeros((stop - start, 2), dtype=np.float64)
        accumulator = np.zeros(2, dtype=np.float64)
        for index in range(start, stop - 1):
            step = deltas[index] if index < len(deltas) else np.zeros(2)
            if np.isfinite(step).all():
                accumulator = accumulator + step
            positions[index - start + 1] = accumulator
        points[(start, stop)] = positions.astype(np.float32)
    return points


def build_recon_examples(
    samples: Sequence[CharacterSample],
    run_points_by_block: dict[tuple[str, int, str], dict[tuple[int, int], np.ndarray]],
    resample_frames: int = 64,
) -> list[dict]:
    examples: list[dict] = []
    for sample in samples:
        block_points = run_points_by_block.get(
            (sample.user, sample.action, sample.sample_id)
        )
        if not block_points:
            continue
        pieces = []
        for run in sample.runs:
            points = block_points.get(run)
            if points is not None and len(points):
                pieces.append(points - points[0])
        if not pieces:
            continue
        trajectory = normalize_trajectory(
            resample_trajectory(np.concatenate(pieces, axis=0), resample_frames)
        )
        examples.append(
            {
                "trajectory": trajectory,
                "label": sample.label,
                "char": sample.char,
                "user": sample.user,
                "action": sample.action,
                "sample_id": sample.sample_id,
                "runs": sample.runs,
                "source": "recon",
            }
        )
    return examples


def augment_trajectory(
    points: np.ndarray,
    rng: np.random.Generator,
    max_rotation: float = np.deg2rad(10.0),
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    angle = rng.uniform(-max_rotation, max_rotation)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    scale = rng.uniform(*scale_range)
    return (points @ rotation.T) * scale


class CharacterExampleDataset(Dataset):
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
        return torch.from_numpy(trajectory), torch.tensor(
            example["label"], dtype=torch.long
        )


def evaluate_examples(
    model: CharacterCNN,
    examples: Sequence[dict],
    device: str,
    batch_size: int = 256,
) -> dict[str, Any]:
    if not examples:
        return {"n": 0, "top1": float("nan"), "top3": float("nan"), "confusion": None}
    loader = DataLoader(
        CharacterExampleDataset(examples), batch_size=batch_size, shuffle=False
    )
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            all_logits.append(model(inputs.to(device)).cpu().numpy())
            all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = np.argmax(logits, axis=1)
    return {
        "n": int(len(labels)),
        "top1": topk_accuracy(logits, labels, k=1),
        "top3": topk_accuracy(logits, labels, k=3),
        "confusion": confusion_matrix(predictions, labels).tolist(),
    }


def _block_run_points(
    predictor: Any, block: Any, clean_root: Path, chunk_len: int
) -> dict[tuple[int, int], np.ndarray]:
    x, _, mask, _, _ = load_sample_arrays(block, mmap=False)
    return predict_run_points(predictor, np.asarray(x), np.asarray(mask), chunk_len)


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    clean_root = Path(args.data_root)
    raw_root = Path(args.raw_root)
    samples = discover_character_samples(clean_root, raw_root)
    if not samples:
        raise ValueError("no character samples discovered")
    train_samples, val_samples, test_samples, split_info = character_split(
        samples, args.split, seed=args.seed
    )

    def gt_examples(part):
        examples = build_gt_examples(part, clean_root)
        for example in examples:
            example["source"] = "gt"
        return examples

    train_examples = gt_examples(train_samples)
    val_examples = gt_examples(val_samples)
    test_examples = gt_examples(test_samples)

    recon_info: dict[str, Any] = {"used": False}
    if args.source in {"recon", "both"}:
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = load_trajectory_predictor(args.reconstruction_checkpoint, device=device)
        blocks = {
            (block.user, block.action, block.sample_id): block
            for block in discover_trajectory_samples(clean_root)
            if block.action in (0, 1) and sample_is_valid(block)
        }
        run_points: dict[tuple[str, int, str], dict[tuple[int, int], np.ndarray]] = {}
        for key in {
            (sample.user, sample.action, sample.sample_id)
            for part in (train_samples, val_samples, test_samples)
            for sample in part
        }:
            run_points[key] = _block_run_points(
                predictor, blocks[key], clean_root, args.chunk_len
            )
        recon_train = build_recon_examples(train_samples, run_points)
        recon_val = build_recon_examples(val_samples, run_points)
        recon_test = build_recon_examples(test_samples, run_points)
        recon_info = {
            "used": True,
            "checkpoint": str(args.reconstruction_checkpoint),
            "counts": {
                "train": len(recon_train),
                "val": len(recon_val),
                "test": len(recon_test),
            },
        }
        if args.source == "recon":
            train_examples, val_examples, test_examples = (
                recon_train,
                recon_val,
                recon_test,
            )
        else:
            train_examples = train_examples + recon_train
            val_examples = val_examples + recon_val
            test_examples = test_examples + recon_test

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharacterCNN(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        CharacterExampleDataset(train_examples, augment=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        val_metrics = evaluate_examples(model, val_examples, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen if seen else 0.0,
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_top1={row['val_top1']:.4f} val_top3={row['val_top3']:.4f}"
        )
        score = val_metrics["top1"]
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics: dict[str, Any] = {}
    for source in ("gt", "recon"):
        subset = [example for example in test_examples if example["source"] == source]
        if subset:
            test_metrics[source] = evaluate_examples(model, subset, device)
    test_metrics["all"] = evaluate_examples(model, test_examples, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "character_cnn.pt"
    report_path = output_dir / "character_report.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": [chr(ord("A") + index) for index in range(NUM_CLASSES)],
            "config": {
                "resample_frames": 64,
                "dropout": args.dropout,
                "source": args.source,
            },
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
    for source in ("gt", "recon", "all"):
        metrics = test_metrics.get(source)
        if metrics:
            print(
                f"test[{source}] n={metrics['n']} top1={metrics['top1']:.4f} "
                f"top3={metrics['top3']:.4f}"
            )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 26-class handwriting letter classifier on trajectories"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--reconstruction-checkpoint", type=Path, default=DEFAULT_RECONSTRUCTION
    )
    parser.add_argument("--split", choices=("user", "random"), default="user")
    parser.add_argument("--source", choices=("gt", "recon", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--chunk-len", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("models_character"))
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
