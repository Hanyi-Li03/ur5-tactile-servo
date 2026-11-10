#!/usr/bin/env python3
# 修改：此入口改为左右触觉深度图回归，保留原说明供对照。
# """Train fingertip pose regression from a LeRobot distributed-force dataset."""
"""Train relative fingertip pose from embedded left/right tactile depth PNGs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# 修改：PNG 解码依赖已迁入 data.py，此处保留旧导入为注释。
# from io import BytesIO
# from PIL import Image
# import pyarrow.dataset as pads

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# 路径以当前脚本为基准，数据、模型输出与相对导入均留在本仓库内。
POLICY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = POLICY_DIR.parent

try:
    # 修改：统一从 data.py 导入深度处理，原分布力导入保留为注释。
    # from .data import POSE_NAMES, TactilePoseDataset, fit_normalization, load_lerobot_tactile_dataset
    # from .data import POSE_NAMES, POSE_KEY, Normalization, TactilePoseDataset
    from .data import (
        POSE_NAMES, DEPTH_KEYS, load_lerobot_depth_dataset,fit_normalization,
        fit_depth_normalization, TactileDepthPoseDataset,
    )
    # 修改：深度入口直接创建原始尺寸模型；保留旧导入供对照。
    # from .model import MODEL_TYPES, create_model
    from .model import NativeDepthCNN
except ImportError:  # Allows: python tactile_policy/train_pose_real_image.py
    # 修改：统一从 data.py 导入深度处理，原分布力导入保留为注释。
    # from data import POSE_NAMES, TactilePoseDataset, fit_normalization, load_lerobot_tactile_dataset
    # from data import POSE_NAMES, POSE_KEY, Normalization, TactilePoseDataset
    from data import (
        POSE_NAMES, DEPTH_KEYS, load_lerobot_depth_dataset,fit_normalization,
        fit_depth_normalization, TactileDepthPoseDataset,
    )
    # 修改：深度入口直接创建原始尺寸模型；保留旧导入供对照。
    # from model import MODEL_TYPES, create_model
    from model import NativeDepthCNN



# 修改：以下深度处理实现已统一迁入 data.py，旧定义注释保留供对照。
# # 新增：左右字段顺序同时决定输入的第 0、1 通道；pose 使用同一条记录。
# DEPTH_KEYS = ("observation.tactile_left.depth", "observation.tactile_right.depth")
#
#
# def decode_depth_png(value: dict) -> np.ndarray:
#     """一次解码，缓存原始 uint8 深度；不缩放、不逐图拉伸、不做对数压缩。"""
#     if not value or not value.get("bytes"):
#         raise ValueError("Expected embedded depth PNG bytes")
#     with Image.open(BytesIO(value["bytes"])) as image:
#         depth = np.array(image)
#     if depth.ndim == 3:
#         if depth.shape[-1] not in (1, 3) or not np.all(depth == depth[..., :1]):
#             raise ValueError("Depth PNG must have identical grayscale channels")
#         depth = depth[..., 0]
#     if depth.ndim != 2 or depth.dtype != np.uint8:
#         raise ValueError(f"Expected 2D uint8 depth, got {depth.shape}, {depth.dtype}")
#     return np.ascontiguousarray(depth)
#
#
# def load_lerobot_depth_dataset(dataset_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
#     """新增：启动时一次性解码左右图，返回 [N,2,H,W] 和同一行的六维 pose。"""
#     data_dir = Path(dataset_root).expanduser().resolve() / "data"
#     files = sorted(data_dir.rglob("*.parquet"))
#     if not files:
#         raise FileNotFoundError(f"No Parquet files found under {data_dir}")
#     table = pads.dataset([str(path) for path in files], format="parquet").to_table(
#         columns=[*DEPTH_KEYS, POSE_KEY]
#     )
#     if table.num_rows == 0:
#         raise ValueError("Depth dataset is empty")
#     sides = []
#     for key in DEPTH_KEYS:
#         frames = []
#         for index, value in enumerate(table[key].to_pylist()):
#             try:
#                 frames.append(decode_depth_png(value))
#             except (ValueError, OSError, TypeError) as exc:
#                 raise ValueError(f"Invalid {key} at row {index}: {exc}") from exc
#         sides.append(np.stack(frames))
#     inputs = np.stack(sides, axis=1)
#     targets = np.asarray(table[POSE_KEY].to_pylist(), dtype=np.float32)
#     if targets.shape != (len(inputs), 6) or not np.isfinite(targets).all():
#         raise ValueError(f"Expected finite pose targets shaped ({len(inputs)}, 6)")
#     return inputs, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/lhy/ur5-tactile-servo/data/16153_xense_tactile_boltnut_merged"),
    )
    # 修改：深度训练使用独立默认输出目录。
    # parser.add_argument("--output-dir", type=Path, default=Path("tactile_policy/outputs/tactile_pose_merged_14901_tokenizer"))
    # 修改：原始尺寸模型单独输出，避免覆盖之前的 128×128 深度模型。
    # parser.add_argument("--output-dir", type=Path, default=Path("tactile_policy/outputs/tactile_pose_merged_14901_depth_simple_cnn"))
    # 修改：最大值缩放实验使用独立目录，保留旧标准化目录配置。
    # parser.add_argument("--output-dir", type=Path, default=Path("tactile_policy/outputs/tactile_pose_merged_14901_native_depth_cnn_std"))
    parser.add_argument("--output-dir", type=Path, default=Path("tactile_policy/outputs/tactile_pose_merged_16153_depth"))
    parser.add_argument("--grid-size", type=int, default=32)
    # 历史说明：此前本入口限定 simple_cnn；当前改用下方 native_depth_cnn。
    # parser.add_argument("--model-type", choices=MODEL_TYPES, default="simple_cnn")
    # 修改：用独立模型标识区分原始尺寸 CNN 与旧 simple_cnn。
    # parser.add_argument("--model-type", choices=("simple_cnn",), default="simple_cnn")
    parser.add_argument("--model-type", choices=("native_depth_cnn",), default="native_depth_cnn")
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
    # 修改：读取内嵌左右深度图，原始 H/W 保留在内存缓存中。
    # inputs, targets = load_lerobot_tactile_dataset(args.dataset_root, args.grid_size)
    inputs, targets = load_lerobot_depth_dataset(args.dataset_root)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(targets))
    n_val = max(1, int(round(len(indices) * args.val_ratio)))
    val_indices, train_indices = indices[:n_val], indices[n_val:]
    if len(train_indices) == 0:
        raise ValueError("Training split is empty")

    # 修改：不拟合图像均值/标准差，只对训练集 pose 拟合标准化参数。
    # normalization = fit_normalization(inputs[train_indices], targets[train_indices])
    # train_dataset = TactilePoseDataset(inputs, targets, train_indices, normalization)
    # val_dataset = TactilePoseDataset(inputs, targets, val_indices, normalization)
    # normalization = fit_depth_normalization(targets[train_indices])

    # 新处理：左右通道共用训练集最大值作为固定除数
    train_depth_max = float(inputs[train_indices].max())
    input_scale = max(train_depth_max, 1.0)  # 全零时避免除以零

    print(
        f"训练集深度最大灰度值：{train_depth_max:g}；"
        f"左右通道统一除以：{input_scale:g}"
    )

    normalization = fit_depth_normalization(
        targets[train_indices],
        input_scale=input_scale,
    )
    train_dataset = TactileDepthPoseDataset(inputs, targets, train_indices, normalization)
    val_dataset = TactileDepthPoseDataset(inputs, targets, val_indices, normalization)
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

    # 历史说明：此前使用 ResizeInput 缩放到 128×128，下面保留旧调用作为注释。
    # model = create_model(args.model_type, grid_size=args.grid_size).to(device)
    # model = create_model(args.model_type, in_channels=2, grid_size=args.grid_size).to(device)
    # 修改：根据已加载深度的原始 H/W 建模，不经过 ResizeInput。
    model = NativeDepthCNN(in_channels=2, out_dim=6, input_hw=tuple(inputs.shape[-2:])).to(device)
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
                    # 修改：保存真实输入高宽，供推理构建同一 NativeDepthCNN。
                    # "model_image_size": 128,
                    "model_image_size": None,
                    "model_input_hw": list(inputs.shape[-2:]),
                    # 修改：记录真实输入与预处理，推理时不得沿用分布力的 signed_log1p。
                    # "input_keys": [
                    #     "observation.tactile_left.distributed_force",
                    #     "observation.tactile_right.distributed_force",
                    # ],
                    "input_keys": list(DEPTH_KEYS),
                    "input_channels": 2,
                    "input_shape": list(inputs.shape[1:]),
                    "input_preprocessing": {
                        "type": "depth_png_uint8_train_max",
                        "channel_order": ["left", "right"],
                        "png_channel": 0,
                        "divisor": input_scale,
                        "signed_log1p": False,
                        "per_image_normalization": False,
                        # 修改：没有输入缩放；旧插值配置保留为注释。
                        # "resize": {"size": [128, 128], "mode": "bilinear", "align_corners": False},
                        "resize": None,
                    },
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
