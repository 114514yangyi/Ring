"""Train a 26-class letter classifier directly from contact-segment IMU data.

This baseline consumes the derived ``200hz`` conversion manifest produced by
``build_touch_dataset_200hz.py``.  It is separate from the project's existing
trajectory-to-letter classifier: its input is six IMU channels resampled over
the touchscreen-confirmed contact interval.
"""

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

from .paper_dataset import default_user_split, user_number


LABELS = tuple(chr(ord("a") + index) for index in range(26))
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resample_imu(values: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        raise ValueError("cannot resample an empty contact segment")
    if len(values) == 1:
        return np.repeat(values, frames, axis=0)
    positions = np.linspace(0, len(values) - 1, num=frames)
    source = np.arange(len(values))
    return np.stack([np.interp(positions, source, values[:, channel]) for channel in range(6)], axis=1).astype(np.float32)


def load_examples(data_root: str | Path, frames: int) -> list[dict[str, Any]]:
    data_root = Path(data_root)
    manifest_path = data_root / "conversion_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples: list[dict[str, Any]] = []
    for record_index, row in enumerate(payload["records"]):
        label = str(row["letter"]).lower()
        if label not in LABEL_TO_INDEX:
            continue
        path = data_root / row["pseudo_user"] / "0" / f"{row['sample_id']}_x.npy"
        mask_path = data_root / row["pseudo_user"] / "0" / f"{row['sample_id']}_mask.npy"
        x = np.load(path)
        mask = np.load(mask_path).astype(bool)
        contact = np.asarray(x[mask], dtype=np.float32)
        if len(contact) == 0:
            continue
        examples.append({
            "x": resample_imu(contact, frames),
            "label": LABEL_TO_INDEX[label],
            "char": label,
            "user": row["pseudo_user"],
            "trial": row["trial"],
            "source_row": int(row.get("source_row", record_index)),
        })
    if not examples:
        raise ValueError(f"no labeled contact segments found under {data_root}")
    return examples


def split_examples(examples: Sequence[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict], dict[str, list[str]]]:
    train_users, val_users, test_users = default_user_split([row["user"] for row in examples])
    split = {
        "train_users": sorted(train_users, key=user_number),
        "val_users": sorted(val_users, key=user_number),
        "test_users": sorted(test_users, key=user_number),
    }
    return (
        [row for row in examples if row["user"] in train_users],
        [row for row in examples if row["user"] in val_users],
        [row for row in examples if row["user"] in test_users],
        split,
    )


def normalization(examples: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([row["x"] for row in examples], axis=0).astype(np.float64)
    return values.mean(axis=0).astype(np.float32), np.maximum(values.std(axis=0), 1e-6).astype(np.float32)


class LetterDataset(Dataset):
    def __init__(self, examples: Sequence[dict], mean: np.ndarray, std: np.ndarray, augment: bool = False):
        self.examples, self.mean, self.std, self.augment = list(examples), mean, std, augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.examples[index]
        x = (row["x"] - self.mean) / self.std
        if self.augment:
            x = x + np.random.normal(0.0, 0.015, size=x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.tensor(row["label"], dtype=torch.long)


class IMULetterCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128 * 16, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, len(LABELS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 6:
            raise ValueError(f"expected [batch, frames, 6], got {tuple(x.shape)}")
        return self.classifier(self.pool(self.features(x.transpose(1, 2))))


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, Any]:
    model.eval()
    correct1 = correct3 = count = 0
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y = y.to(device)
            prediction = logits.argmax(dim=1)
            correct1 += int((prediction == y).sum())
            correct3 += int((logits.topk(3, dim=1).indices == y[:, None]).any(dim=1).sum())
            count += len(y)
            for truth, guess in zip(y.cpu().numpy(), prediction.cpu().numpy()):
                confusion[int(truth), int(guess)] += 1
    per_class = {LABELS[index]: {"support": int(confusion[index].sum()), "accuracy": float(confusion[index, index] / confusion[index].sum()) if confusion[index].sum() else None} for index in range(len(LABELS))}
    return {"n": count, "top1": correct1 / count if count else 0.0, "top3": correct3 / count if count else 0.0, "confusion": confusion.tolist(), "per_class": per_class}


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    examples = load_examples(args.data_root, args.frames)
    train, val, test, split = split_examples(examples)
    mean, std = normalization(train)
    train_loader = DataLoader(LetterDataset(train, mean, std, augment=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LetterDataset(val, mean, std), batch_size=args.batch_size)
    test_loader = DataLoader(LetterDataset(test, mean, std), batch_size=args.batch_size)
    model = IMULetterCNN(dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state, best_top1, history = None, -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = item_count = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(args.device))
            loss = criterion(logits, y.to(args.device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            item_count += len(y)
        metrics = evaluate(model, val_loader, args.device)
        history.append({"epoch": epoch, "train_loss": loss_sum / item_count, "val_top1": metrics["top1"], "val_top3": metrics["top3"]})
        print(f"epoch {epoch:03d} train_loss={loss_sum/item_count:.4f} val_top1={metrics['top1']:.4f} val_top3={metrics['top3']:.4f}")
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    report = {"data_root": str(args.data_root), "protocol": "trial-disjoint pseudo-user split; not writer-disjoint", "frames": args.frames, "split": split, "sample_counts": {"train": len(train), "val": len(val), "test": len(test)}, "normalization": {"mean": mean.tolist(), "std": std.tolist()}, "history": history, "val": evaluate(model, val_loader, args.device), "test": evaluate(model, test_loader, args.device)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "imu_letter_cnn.pt"
    torch.save({"state_dict": model.state_dict(), "mean": mean.tolist(), "std": std.tolist(), "frames": args.frames, "labels": list(LABELS)}, checkpoint)
    report["checkpoint"] = str(checkpoint)
    with (args.output_dir / "imu_letter_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"saved checkpoint: {checkpoint}")
    print(f"test top1={report['test']['top1']:.4f} top3={report['test']['top3']:.4f} n={report['test']['n']}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct IMU-to-letter baseline on converted 200 Hz data")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
