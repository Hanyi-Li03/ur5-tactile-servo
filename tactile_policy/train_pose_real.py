#!/usr/bin/env python3
"""Train fingertip pose regression from a LeRobot distributed-force dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


try:
    from .data import POSE_NAMES, TactilePoseDataset, fit_normalization, load_lerobot_tactile_dataset
    from .model import MODEL_TYPES, create_model
except ImportError:  # Allows: python tactile_policy/train_pose_real.py
    from data import POSE_NAMES, TactilePoseDataset, fit_normalization, load_lerobot_tactile_dataset
    from model import MODEL_TYPES, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/lhy/ur5-tactile-servo/data/18967_xense_tactile_boltnut_merged"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tactile_policy/outputs/tactile_pose_merged_18967_simple_cnn"))
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="simple_cnn")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    errors = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss_sum += criterion(outputs, targets).item() * inputs.shape[0]
            prediction_pose = outputs * target_std + target_mean
            target_pose = targets * target_std + target_mean
            error = prediction_pose - target_pose
            # Report the shortest Euler-angle difference for robustness near +/-pi.
            error[:, 3:] = torch.atan2(torch.sin(error[:, 3:]), torch.cos(error[:, 3:]))
            errors.append(error.abs().cpu().numpy())
    return loss_sum / len(loader.dataset), np.concatenate(errors).mean(axis=0)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    seed_everything(args.seed)
    device = resolve_device(args.device)

    print(f"Loading dataset: {args.dataset_root.resolve()}")
    inputs, targets = load_lerobot_tactile_dataset(args.dataset_root, args.grid_size)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(targets))
    n_val = max(1, int(round(len(indices) * args.val_ratio)))
    val_indices, train_indices = indices[:n_val], indices[n_val:]
    if len(train_indices) == 0:
        raise ValueError("Training split is empty")

    normalization = fit_normalization(inputs[train_indices], targets[train_indices])
    train_dataset = TactilePoseDataset(inputs, targets, train_indices, normalization)
    val_dataset = TactilePoseDataset(inputs, targets, val_indices, normalization)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        # FCN contains BatchNorm1d, which cannot train on a final batch of one.
        drop_last=args.model_type == "fcn" and len(train_dataset) % args.batch_size == 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(args.model_type, grid_size=args.grid_size).to(device)
    # criterion = nn.SmoothL1Loss(beta=0.5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    target_mean = torch.as_tensor(normalization.target_mean, device=device)
    target_std = torch.as_tensor(normalization.target_std, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    best_val_loss = float("inf")
    stale_epochs = 0
    print(
        f"samples={len(targets)} train={len(train_dataset)} val={len(val_dataset)} "
        f"input={tuple(inputs.shape[1:])} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_inputs.shape[0]
            train_sample_count += batch_inputs.shape[0]

        train_loss = train_loss_sum / train_sample_count
        val_loss, mae = evaluate(
            model, val_loader, criterion, target_mean, target_std, device
        )
        scheduler.step(val_loss)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "position_mae_mm": (mae[:3] * 1000.0).tolist(),
            "rotation_mae_deg": np.rad2deg(mae[3:]).tolist(),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")
        pos_text = ", ".join(f"{name}={value:.3f}" for name, value in zip(POSE_NAMES[:3], metrics["position_mae_mm"]))
        rot_text = ", ".join(f"{name}={value:.3f}" for name, value in zip(POSE_NAMES[3:], metrics["rotation_mae_deg"]))
        print(
            f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f} | "
            f"position MAE mm [{pos_text}] | rotation MAE deg [{rot_text}]"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalization": normalization.as_dict(),
                    "grid_size": args.grid_size,
                    "model_type": args.model_type,
                    "model_image_size": 128,
                    "input_keys": [
                        "observation.tactile_left.distributed_force",
                        "observation.tactile_right.distributed_force",
                    ],
                    "target_key": "observation.fingertip_center_pose_relative_to_nut",
                    "target_names": list(POSE_NAMES),
                    "target_units": ["m", "m", "m", "rad", "rad", "rad"],
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "args": vars(args) | {
                        "dataset_root": str(args.dataset_root),
                        "output_dir": str(args.output_dir),
                    },
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement")
                break

    print(f"Best checkpoint: {best_path.resolve()} (val_loss={best_val_loss:.6f})")


if __name__ == "__main__":
    main()
