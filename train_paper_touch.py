from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .paper_dataset import (
    PaperTouchWindowDataset,
    build_window_refs,
    class_counts,
    compute_normalization,
    default_user_split,
    discover_cleaned_samples,
)
from .paper_touch import TOUCH_LABELS, PaperTouchConfig, WritingRingTouchResNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_samples(samples):
    train_users, val_users, test_users = default_user_split([sample.user for sample in samples])
    train = [sample for sample in samples if sample.user in train_users]
    val = [sample for sample in samples if sample.user in val_users]
    test = [sample for sample in samples if sample.user in test_users]
    return train, val, test, {
        "train_users": sorted(train_users),
        "val_users": sorted(val_users),
        "test_users": sorted(test_users),
    }


def confusion_matrix(num_classes: int) -> np.ndarray:
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def metrics_from_confusion(cm: np.ndarray) -> dict[str, Any]:
    labels = list(TOUCH_LABELS)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    per_class: dict[str, Any] = {}
    for index, label in enumerate(labels):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum() - tp)
        fn = int(cm[index, :].sum() - tp)
        support = int(cm[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "event_macro_f1": float(np.mean([per_class["lift"]["f1"], per_class["press"]["f1"]])),
        "support": total,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def evaluate(model, loader, device: str) -> dict[str, Any]:
    model.eval()
    cm = confusion_matrix(len(TOUCH_LABELS))
    loss_total = 0.0
    item_count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)
            for true_value, pred_value in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cm[int(true_value), int(pred_value)] += 1
            loss_total += float(loss.item()) * len(y)
            item_count += len(y)
    result = metrics_from_confusion(cm)
    result["loss"] = loss_total / item_count if item_count else 0.0
    return result


def train(args) -> dict[str, Any]:
    set_seed(args.seed)
    config = PaperTouchConfig(sample_rate=200.0, window_frames=args.window_frames)
    samples = discover_cleaned_samples(args.data_root)
    if not samples:
        raise ValueError(f"no cleaned samples found under {args.data_root}")
    train_samples, val_samples, test_samples, split_info = split_samples(samples)
    mean, std = compute_normalization(train_samples)

    train_refs = build_window_refs(
        train_samples,
        config,
        max_per_class=args.max_train_windows_per_class,
        seed=args.seed,
    )
    val_refs = build_window_refs(
        val_samples,
        config,
        max_per_class=args.max_eval_windows_per_class,
        seed=args.seed + 1,
    )
    test_refs = build_window_refs(
        test_samples,
        config,
        max_per_class=args.max_eval_windows_per_class,
        seed=args.seed + 2,
    )
    if not train_refs or not val_refs or not test_refs:
        raise ValueError("empty train/val/test refs; check data split")

    train_ds = PaperTouchWindowDataset(train_samples, train_refs, mean, std, config)
    val_ds = PaperTouchWindowDataset(val_samples, val_refs, mean, std, config)
    test_ds = PaperTouchWindowDataset(test_samples, test_refs, mean, std, config)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = args.device
    model = WritingRingTouchResNet(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            count += len(y)

        val_metrics = evaluate(model, val_loader, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": total_loss / count if count else 0.0,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_event_macro_f1": val_metrics["event_macro_f1"],
            "val_loss": val_metrics["loss"],
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"val_acc={epoch_row['val_accuracy']:.4f} "
            f"val_macro_f1={epoch_row['val_macro_f1']:.4f} "
            f"val_event_f1={epoch_row['val_event_macro_f1']:.4f}"
        )
        score = val_metrics["event_macro_f1"]
        if score > best_val:
            best_val = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "paper_touch_resnet.pt"
    report_path = args.output_dir / "paper_touch_report.json"
    payload = {
        "state_dict": model.state_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "labels": list(TOUCH_LABELS),
        "config": {
            "sample_rate": config.sample_rate,
            "window_frames": config.window_frames,
            "window_seconds": config.window_seconds,
            "dropout": args.dropout,
        },
        "split": split_info,
    }
    torch.save(payload, checkpoint_path)

    report = {
        "checkpoint": str(checkpoint_path),
        "data_root": str(args.data_root),
        "split": split_info,
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "window_counts": {
            "train": class_counts(train_refs),
            "val": class_counts(val_refs),
            "test": class_counts(test_refs),
        },
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved report: {report_path}")
    print(
        "test "
        f"acc={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"event_f1={test_metrics['event_macro_f1']:.4f}"
    )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Train WritingRing-style touch-state detector")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("writing_state/clean_data_delete_g/data"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("writing_state/models"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--max-train-windows-per-class", type=int, default=50000)
    parser.add_argument("--max-eval-windows-per-class", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
