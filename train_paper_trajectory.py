from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .paper_dataset import default_user_split, user_number
from .paper_trajectory import (
    PaperTrajectoryConfig,
    WritingRingTrajectoryNet,
    chunk_slices,
    integrate_contact_segments,
    summarize_segment_metrics,
    trajectory_metrics,
)
from .paper_trajectory_dataset import (
    PaperTrajectoryDataset,
    SegmentRef,
    TrajectorySample,
    build_segment_refs,
    compute_normalization,
    discover_trajectory_samples,
    materialize_segment,
    sample_is_valid,
)


DEFAULT_DATA_ROOT = Path("/data/huyang/datasets/WritingRing/clean_data_delete_g/data")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_split_samples(
    samples: Sequence[TrajectorySample],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> tuple[list[TrajectorySample], list[TrajectorySample], list[TrajectorySample], dict[str, Any]]:
    if len(samples) < 3:
        raise ValueError("need at least three samples for a train/val/test split")
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
    split_info = {
        "strategy": "random",
        "seed": seed,
        "ratios": [float(value) for value in ratios],
    }
    return train, val, test, split_info


def user_split_samples(
    samples: Sequence[TrajectorySample],
    holdout_users: Sequence[int] | None = None,
) -> tuple[list[TrajectorySample], list[TrajectorySample], list[TrajectorySample], dict[str, Any]]:
    users = [sample.user for sample in samples]
    if holdout_users:
        holdout = set(int(value) for value in holdout_users)
        ordered = sorted(set(users), key=user_number)
        test_users = {user for user in ordered if user_number(user) in holdout}
        remaining = [user for user in ordered if user not in test_users]
        if not test_users or len(remaining) < 2:
            raise ValueError("holdout-users leaves no train/val users")
        val_count = max(1, round(len(ordered) * 0.1))
        val_count = min(val_count, len(remaining) - 1)
        val_users = set(remaining[-val_count:])
        train_users = set(remaining) - val_users
    else:
        train_users, val_users, test_users = default_user_split(users)
    train = [sample for sample in samples if sample.user in train_users]
    val = [sample for sample in samples if sample.user in val_users]
    test = [sample for sample in samples if sample.user in test_users]
    if not train or not val or not test:
        raise ValueError("empty train/val/test split")
    split_info = {
        "strategy": "user",
        "train_users": sorted(train_users, key=user_number),
        "val_users": sorted(val_users, key=user_number),
        "test_users": sorted(test_users, key=user_number),
    }
    return train, val, test, split_info


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    schedule: str = "constant",
    eta_min_ratio: float = 0.01,
):
    if schedule == "constant":
        return None
    if schedule == "cosine":
        eta_min = optimizer.param_groups[0]["lr"] * eta_min_ratio
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=eta_min
        )
    raise ValueError(f"unknown lr schedule: {schedule}")


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    finite = torch.isfinite(pred).all(dim=-1)
    weights = mask.to(pred.dtype) * finite.to(pred.dtype)
    denominator = weights.sum() * pred.shape[-1]
    if float(denominator) == 0.0:
        return torch.nan_to_num(pred).sum() * 0.0
    difference = torch.where(
        finite.unsqueeze(-1), pred - target, torch.zeros_like(pred)
    )
    return (difference.pow(2) * weights.unsqueeze(-1)).sum() / denominator


def masked_trajectory_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window: int = 50,
) -> torch.Tensor:
    if window < 2:
        raise ValueError("window must be at least two frames")
    total = pred.shape[1]
    if total < window:
        return torch.nan_to_num(pred).sum() * 0.0
    stride = max(1, window // 2)
    valid = mask.unfold(1, window, stride).min(dim=-1).values > 0.5
    if not bool(valid.any()):
        return torch.nan_to_num(pred).sum() * 0.0
    masked_pred = pred * mask.unsqueeze(-1)
    masked_target = target * mask.unsqueeze(-1)
    cum_pred = torch.cumsum(masked_pred, dim=1)
    cum_target = torch.cumsum(masked_target, dim=1)
    window_pred = cum_pred.unfold(1, window, stride)
    window_target = cum_target.unfold(1, window, stride)
    relative_pred = window_pred - window_pred[..., :1]
    relative_target = window_target - window_target[..., :1]
    errors = (relative_pred - relative_target).pow(2).sum(dim=-1).mean(dim=-1)
    return errors[valid].mean() / window


def _chunked_deltas(
    model: WritingRingTrajectoryNet, x: torch.Tensor, chunk_len: int
) -> torch.Tensor:
    total = x.shape[1]
    output = torch.full((x.shape[0], total, 2), float("nan"), device=x.device)
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    for x_lo, x_hi, out_lo, out_hi in chunk_slices(
        total, chunk_len, model.config.window_offset
    ):
        if state is not None:
            state = (state[0].detach(), state[1].detach())
        chunk_output, state = model(x[:, x_lo:x_hi], state)
        output[:, out_lo:out_hi] = chunk_output[:, out_lo - x_lo : out_hi - x_lo]
    return output


def sample_audit(samples: Sequence[TrajectorySample]) -> dict[str, Any]:
    invalid = [sample for sample in samples if not sample_is_valid(sample)]
    return {
        "discovered": len(samples),
        "invalid_count": len(invalid),
        "invalid_ids": [
            f"{sample.user}/{sample.action}/{sample.sample_id}" for sample in invalid
        ],
    }


def _segment_rows(
    deltas: np.ndarray,
    board: np.ndarray,
    mask: np.ndarray,
    mm_scale: Sequence[float] | None,
    min_frames: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in integrate_contact_segments(deltas, mask, min_frames=min_frames):
        ground_truth = np.asarray(
            board[segment.start : segment.end], dtype=np.float64
        ) - np.asarray(board[segment.start], dtype=np.float64)
        rows.append(trajectory_metrics(segment.xy, ground_truth, mm_scale))
    return rows


def evaluate_model(
    model: WritingRingTrajectoryNet,
    loader: DataLoader,
    chunk_len: int,
    device: str,
    mm_scale: Sequence[float] | None,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for x, _, mask, board in loader:
            x = x.to(device)
            predictions = _chunked_deltas(model, x, chunk_len)
            for index in range(x.shape[0]):
                rows.extend(
                    _segment_rows(
                        predictions[index].cpu().numpy(),
                        board[index].numpy(),
                        mask[index].numpy(),
                        mm_scale,
                    )
                )
    return summarize_segment_metrics(rows)


def train_one_epoch(
    model: WritingRingTrajectoryNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    chunk_len: int,
    device: str,
    update_every: str = "batch",
    aux_weight: float = 0.0,
    aux_window: int = 50,
) -> float:
    if update_every not in {"batch", "chunk"}:
        raise ValueError(f"unknown update_every: {update_every}")
    model.train()
    total_loss = 0.0
    chunks = 0
    for x, y, mask, _ in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        total = x.shape[1]
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        if update_every == "batch":
            optimizer.zero_grad(set_to_none=True)
        for x_lo, x_hi, out_lo, out_hi in chunk_slices(
            total, chunk_len, model.config.window_offset
        ):
            if state is not None:
                state = (state[0].detach(), state[1].detach())
            if update_every == "chunk":
                optimizer.zero_grad(set_to_none=True)
            prediction, state = model(x[:, x_lo:x_hi], state)
            chunk_prediction = prediction[:, out_lo - x_lo : out_hi - x_lo]
            chunk_target = y[:, out_lo:out_hi]
            chunk_mask = mask[:, out_lo:out_hi]
            loss = masked_mse_loss(chunk_prediction, chunk_target, chunk_mask)
            if aux_weight > 0:
                loss = loss + aux_weight * masked_trajectory_loss(
                    chunk_prediction, chunk_target, chunk_mask, window=aux_window
                )
            loss.backward()
            if update_every == "chunk":
                optimizer.step()
            total_loss += float(loss.item())
            chunks += 1
        if update_every == "batch":
            optimizer.step()
    return total_loss / chunks if chunks else 0.0


def save_segment_plots(
    model: WritingRingTrajectoryNet,
    samples: Sequence[TrajectorySample],
    refs: Sequence[SegmentRef],
    mean: np.ndarray,
    std: np.ndarray,
    config: PaperTrajectoryConfig,
    mm_scale: Sequence[float] | None,
    output_dir: Path,
    chunk_len: int,
    device: str,
    limit: int = 6,
) -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping trajectory plots")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    written = 0
    with torch.no_grad():
        for ref in refs:
            x, _, mask, board = materialize_segment(
                samples[ref.sample_index], ref, config.segment_frames
            )
            normalized = (x - mean) / std
            tensor = torch.from_numpy(normalized)[None].to(device)
            deltas = _chunked_deltas(model, tensor, chunk_len)[0].cpu().numpy()
            for segment in integrate_contact_segments(deltas, mask, min_frames=5):
                ground_truth = np.asarray(
                    board[segment.start : segment.end], dtype=np.float64
                ) - np.asarray(board[segment.start], dtype=np.float64)
                figure, axis = plt.subplots(figsize=(3, 3))
                axis.plot(ground_truth[:, 0], ground_truth[:, 1], label="ground truth")
                axis.plot(segment.xy[:, 0], segment.xy[:, 1], label="reconstructed")
                axis.invert_yaxis()
                axis.set_aspect("equal", adjustable="datalim")
                axis.legend(fontsize=7)
                figure.tight_layout()
                figure.savefig(output_dir / f"segment_{written:03d}.png", dpi=120)
                plt.close(figure)
                written += 1
                if written >= limit:
                    return written
    return written


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    config_kwargs: dict[str, Any] = {
        "dropout": args.dropout,
        "tcn_style": args.tcn_style,
        "lstm_layers": args.lstm_layers,
        "bidirectional": args.bidirectional,
    }
    if args.tcn_channels is not None:
        config_kwargs["tcn_channels"] = tuple(
            int(value) for value in args.tcn_channels.split(",")
        )
    if args.board_mm_scale is not None:
        config_kwargs["board_mm_scale"] = tuple(args.board_mm_scale)
    config = PaperTrajectoryConfig(**config_kwargs)
    if config.bidirectional:
        args.chunk_len = max(args.chunk_len, config.segment_frames)
        args.eval_chunk_len = max(args.eval_chunk_len, config.segment_frames)
        args.update_every = "batch"
    discovered = discover_trajectory_samples(args.data_root)
    audit = sample_audit(discovered)
    samples = [sample for sample in discovered if sample_is_valid(sample)]
    if not samples:
        raise ValueError(f"no valid samples found under {args.data_root}")
    if args.limit_samples is not None and args.limit_samples < len(samples):
        step = max(1, len(samples) // args.limit_samples)
        samples = samples[::step][: args.limit_samples]

    if args.split == "random":
        train_samples, val_samples, test_samples, split_info = random_split_samples(
            samples, seed=args.seed
        )
    else:
        holdout_users = (
            [int(value) for value in args.holdout_users.split(",") if value.strip()]
            if args.holdout_users
            else None
        )
        train_samples, val_samples, test_samples, split_info = user_split_samples(
            samples, holdout_users=holdout_users
        )

    mean, std = compute_normalization(train_samples)
    train_refs = build_segment_refs(
        train_samples, config.segment_frames, strategy=args.long_sample_strategy
    )
    val_refs = build_segment_refs(
        val_samples, config.segment_frames, strategy=args.long_sample_strategy
    )
    test_refs = build_segment_refs(
        test_samples, config.segment_frames, strategy=args.long_sample_strategy
    )
    if not train_refs or not val_refs or not test_refs:
        raise ValueError("empty train/val/test refs; check the data split")

    train_dataset = PaperTrajectoryDataset(
        train_samples, train_refs, mean, std, config.segment_frames,
        input_norm=args.input_norm,
    )
    val_dataset = PaperTrajectoryDataset(
        val_samples, val_refs, mean, std, config.segment_frames,
        input_norm=args.input_norm,
    )
    test_dataset = PaperTrajectoryDataset(
        test_samples, test_refs, mean, std, config.segment_frames,
        input_norm=args.input_norm,
    )

    batch_size = 1 if args.long_sample_strategy == "chunks" else args.batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WritingRingTrajectoryNet(config).to(device)
    if args.init_checkpoint is not None:
        payload = torch.load(
            str(args.init_checkpoint), map_location=device, weights_only=True
        )
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args.epochs, args.lr_schedule)

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.chunk_len, device,
            update_every=args.update_every,
            aux_weight=args.aux_weight,
            aux_window=args.aux_window,
        )
        if scheduler is not None:
            scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate_model(
                model, val_loader, args.eval_chunk_len, device, config.board_mm_scale
            )
            row["val"] = val_metrics
            score = val_metrics["normalized_mean"]
            if np.isnan(score):
                score = float("inf")
            if score < best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"epoch {epoch:03d} train_loss={train_loss:.6f} "
                f"val_norm={val_metrics['normalized_mean']:.5f} "
                f"val_mm={val_metrics['mm_mean']:.4f} "
                f"segments={val_metrics['segments']}"
            )
        else:
            print(f"epoch {epoch:03d} train_loss={train_loss:.6f}")
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = evaluate_model(
        model, val_loader, args.eval_chunk_len, device, config.board_mm_scale
    )
    test_metrics = evaluate_model(
        model, test_loader, args.eval_chunk_len, device, config.board_mm_scale
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "paper_trajectory.pt"
    report_path = args.output_dir / "paper_trajectory_report.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "board_mm_scale": list(config.board_mm_scale)
            if config.board_mm_scale is not None
            else None,
            "config": asdict(config),
            "split": split_info,
        },
        checkpoint_path,
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "data_root": str(args.data_root),
        "split": split_info,
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "segment_counts": {
            "train": len(train_refs),
            "val": len(val_refs),
            "test": len(test_refs),
        },
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "config": asdict(config),
        "board_mm_scale": list(config.board_mm_scale)
        if config.board_mm_scale is not None
        else None,
        "sample_audit": audit,
        "y_anchor": 0,
        "y_anchor_note": (
            "verified on official data: y[t] = board[t+1] - board[t] "
            "(frame-level corr 0.97, integrated 0.9995)"
        ),
        "board_scale_note": (
            "clean board coordinates are normalised; mm errors assume the Sensel "
            "Morph active area (240 x 169.5 mm). Raw contacts are normalised too, "
            "so physical calibration is not recoverable from raw data alone."
        ),
        "train": {
            "epochs": args.epochs,
            "batch_size": batch_size,
            "lr": args.lr,
            "chunk_len": args.chunk_len,
            "eval_chunk_len": args.eval_chunk_len,
            "lr_schedule": args.lr_schedule,
            "update_every": args.update_every,
            "aux_weight": args.aux_weight,
            "aux_window": args.aux_window,
            "input_norm": args.input_norm,
            "long_sample_strategy": args.long_sample_strategy,
            "seed": args.seed,
            "device": device,
        },
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    if args.plot_dir is not None:
        written = save_segment_plots(
            model,
            test_samples,
            test_refs,
            mean,
            std,
            config,
            config.board_mm_scale,
            args.plot_dir,
            args.eval_chunk_len,
            device,
        )
        report["plot_count"] = written
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved report: {report_path}")
    print(
        "test "
        f"norm={test_metrics['normalized_mean']:.5f} "
        f"mm={test_metrics['mm_mean']:.4f} "
        f"segments={test_metrics['segments']} "
        f"skipped={test_metrics['skipped_degenerate']}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train WritingRing-style trajectory reconstruction"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("models_trajectory"))
    parser.add_argument("--split", choices=("random", "user"), default="random")
    parser.add_argument(
        "--holdout-users",
        help="comma-separated user numbers forced into the user-split test set",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--update-every", choices=("batch", "chunk"), default="batch")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tcn-style", choices=("valid", "dilated"), default="valid")
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--input-norm", choices=("global", "per_sample"), default="global")
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--aux-window", type=int, default=50)
    parser.add_argument(
        "--tcn-channels",
        help="comma-separated channel widths, e.g. 16,16,32,32,64,128",
    )
    parser.add_argument("--chunk-len", type=int, default=3000)
    parser.add_argument("--eval-chunk-len", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--long-sample-strategy", choices=("pad_trim", "chunks"), default="pad_trim"
    )
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="load weights before training; with --epochs 0 this evaluates the checkpoint",
    )
    parser.add_argument(
        "--board-mm-scale",
        type=float,
        nargs=2,
        default=None,
        help="override board-units-to-mm scale (default: config value)",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
