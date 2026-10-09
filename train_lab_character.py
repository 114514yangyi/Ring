"""Train the 26-class letter CNN on the lab's own 200 Hz dataset.

Reuses the official character pipeline (``CharacterCNN``, 64-point arc-length resampling and
bounding-box normalisation) but sources the examples from the lab dataset built by
``build_lab_dataset.py``.  Both ground-truth pen traces and trajectories reconstructed by a
trajectory checkpoint can be mixed, exactly like ``train_character_classifier.py`` does for
the official data.  The trial split is taken from ``random_split_samples`` with the same seed
as the trajectory training, so the test trials of both models coincide.

Usage::

    python -m writing_state.train_lab_character --data-root /data/huyang/datasets/RingLab/lab200_v1 \
        --trajectory-checkpoint models_lab/lab_traj_scratch/paper_trajectory.pt \
        --output-dir models_character/lab_both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .character_dataset import normalize_trajectory, resample_trajectory
from .character_model import CharacterCNN, RESAMPLE_FRAMES, load_character_model
from .paper_trajectory import integrate_contact_segments, load_trajectory_predictor
from .paper_trajectory_dataset import discover_trajectory_samples, load_sample_arrays
from .train_character_classifier import (
    CharacterExampleDataset,
    augment_trajectory,
    confusion_matrix,
    evaluate_examples,
    topk_accuracy,
)
from .train_paper_trajectory import random_split_samples

REPO_ROOT = Path(__file__).resolve().parent


def load_trial(sample: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x, board, mask, _, _ = load_sample_arrays(sample, mmap=False)
    label = int(Path(sample.board_path).parent.name)  # action directory = letter index
    return np.asarray(x, dtype=np.float64), np.asarray(board, dtype=np.float64), np.asarray(
        mask, dtype=np.float64
    ), label


def gt_example(board: np.ndarray, mask: np.ndarray, label: int, sample: Any) -> dict[str, Any]:
    contact = mask > 0
    points = board[contact]
    trajectory = normalize_trajectory(resample_trajectory(points, RESAMPLE_FRAMES))
    return {"trajectory": trajectory, "label": label, "sample_id": sample.sample_id, "user": sample.user,
            "source": "gt"}


def recon_examples(samples: Sequence[Any], predictor: Any, device: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample in samples:
        x, board, mask, label = load_trial(sample)
        deltas = predictor.predict_deltas(x.astype(np.float32))
        segments = integrate_contact_segments(
            deltas, mask, window_offset=predictor.config.window_offset, min_frames=5
        )
        if not segments:
            continue
        points = np.concatenate([segment.xy for segment in segments], axis=0)
        trajectory = normalize_trajectory(resample_trajectory(points, RESAMPLE_FRAMES))
        out.append({"trajectory": trajectory, "label": label, "sample_id": sample.sample_id,
                    "user": sample.user, "source": "recon"})
    return out


def build_gt_examples(samples: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample in samples:
        _, board, mask, label = load_trial(sample)
        out.append(gt_example(board, mask, label, sample))
    return out


def per_letter(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    out: dict[str, Any] = {}
    for index in range(26):
        chosen = labels == index
        if not chosen.any():
            continue
        top3 = np.argsort(-logits[chosen], axis=1)[:, :3]
        out[chr(ord("A") + index)] = {
            "n": int(chosen.sum()),
            "top1": float((predictions[chosen] == index).mean()),
            "top3": float((top3 == index).any(axis=1).mean()),
        }
    return out


def evaluate_split(model: CharacterCNN, examples: Sequence[dict], device: str) -> dict[str, Any]:
    loader = DataLoader(CharacterExampleDataset(examples), batch_size=256, shuffle=False)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for trajectories, label in loader:
            logits.append(model(trajectories.to(device)).cpu().numpy())
            labels.append(label.numpy())
    logits_array = np.concatenate(logits, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    return {
        "n": int(len(labels_array)),
        "top1": float(topk_accuracy(logits_array, labels_array, 1)),
        "top3": float(topk_accuracy(logits_array, labels_array, 3)),
        "per_letter": per_letter(logits_array, labels_array),
        "confusion": confusion_matrix(logits_array.argmax(axis=1), labels_array).tolist(),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the letter CNN on the lab dataset")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--trajectory-checkpoint", type=Path, default=None,
                        help="when given, reconstructed trajectories are generated for the recon mix")
    parser.add_argument("--init-checkpoint", type=Path,
                        default=REPO_ROOT / "models_character/user_both_ud/character_cnn.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("models_character/lab_both"))
    parser.add_argument("--source", choices=("gt", "recon", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples = discover_trajectory_samples(args.data_root)
    if not samples:
        raise SystemExit(f"no samples under {args.data_root}")
    train_samples, val_samples, test_samples, split_info = random_split_samples(samples, seed=args.seed)
    print(json.dumps({key: len(value) for key, value in
                      (("train", train_samples), ("val", val_samples), ("test", test_samples))}))

    train_examples = build_gt_examples(train_samples) if args.source in ("gt", "both") else []
    val_examples = build_gt_examples(val_samples)
    test_gt = build_gt_examples(test_samples)
    test_recon: list[dict[str, Any]] = []
    if args.source in ("recon", "both") or args.trajectory_checkpoint is not None:
        if args.trajectory_checkpoint is None:
            raise SystemExit("--source recon/both requires --trajectory-checkpoint")
        predictor = load_trajectory_predictor(args.trajectory_checkpoint, device=device)
        train_examples += recon_examples(train_samples, predictor, device)
        test_recon = recon_examples(test_samples, predictor, device)
        val_recon = recon_examples(val_samples, predictor, device)
    else:
        val_recon = []

    model = CharacterCNN()
    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        payload = torch.load(str(args.init_checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(payload.get("state_dict", payload))
        print(f"initialised from {args.init_checkpoint}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(
        CharacterExampleDataset(train_examples, augment=True, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    history: list[dict[str, Any]] = []
    best_top1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for trajectories, labels in loader:
            trajectories, labels = trajectories.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(trajectories), labels)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(labels)
            seen += len(labels)
        scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, "train_loss": total / max(seen, 1)}
        if epoch % 5 == 0 or epoch == args.epochs:
            val = evaluate_examples(model, val_examples + val_recon, device)
            row["val_top1"] = float(val["top1"])
            row["val_top3"] = float(val["top3"])
            if val["top1"] > best_top1:
                best_top1 = float(val["top1"])
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(json.dumps(row))
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    report = {
        "data_root": str(args.data_root),
        "trajectory_checkpoint": str(args.trajectory_checkpoint) if args.trajectory_checkpoint else None,
        "source": args.source,
        "split": split_info,
        "counts": {"train_gt": len(build_gt_examples(train_samples)) if args.source in ("gt", "both") else 0,
                   "train_total": len(train_examples), "test_gt": len(test_gt), "test_recon": len(test_recon)},
        "config": {"resample_frames": RESAMPLE_FRAMES, "epochs": args.epochs, "lr": args.lr,
                   "batch_size": args.batch_size, "dropout": args.dropout},
        "history": history,
    }
    for name, subset in (("test_gt", test_gt), ("test_recon", test_recon), ("val_gt", val_examples)):
        if subset:
            report[name] = evaluate_split(model, subset, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "source": args.source,
                "data_root": str(args.data_root), "labels": [chr(ord("A") + i) for i in range(26)]},
               args.output_dir / "character_cnn.pt")
    with (args.output_dir / "character_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary = {name: {"n": report[name]["n"], "top1": round(report[name]["top1"], 4),
                      "top3": round(report[name]["top3"], 4)} for name in ("test_gt", "test_recon") if name in report}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
