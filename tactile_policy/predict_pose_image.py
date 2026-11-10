#!/usr/bin/env python3
"""Run a saved tactile pose model on one sample from a LeRobot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# 此入口也是离线对比脚本的深度尺度校验依赖，所有引用均留在当前仓库。
POLICY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = POLICY_DIR.parent

try:
    # 修改：使用共享深度加载器和原始尺寸模型，原分布力导入保留。
    # from .data import Normalization, POSE_NAMES, load_lerobot_tactile_dataset
    # from .model import create_model
    from .data import Normalization, POSE_NAMES, DEPTH_KEYS, load_lerobot_depth_dataset
    from .model import NativeDepthCNN
except ImportError:
    # 修改：使用共享深度加载器和原始尺寸模型，原分布力导入保留。
    # from data import Normalization, POSE_NAMES, load_lerobot_tactile_dataset
    # from model import create_model
    from data import Normalization, POSE_NAMES, DEPTH_KEYS, load_lerobot_depth_dataset
    from model import NativeDepthCNN



# 新增：检查元数据与实际归一化参数一致，避免模型按错误尺度推理。
def validate_depth_scale(preprocessing: dict, normalization: Normalization) -> float:
    mode = preprocessing.get("type")
    if mode not in ("depth_png_uint8_div255", "depth_png_uint8_train_max"):
        raise ValueError(f"Unsupported depth preprocessing: {mode!r}")
    try:
        divisor = float(preprocessing["divisor"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Depth divisor must be a finite positive number") from exc
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("Depth divisor must be a finite positive number")
    if mode == "depth_png_uint8_div255" and divisor != 255.0:
        raise ValueError("div255 preprocessing requires divisor=255")
    if (normalization.input_mean.shape != (2,)
            or normalization.input_std.shape != (2,)
            or not np.all(normalization.input_mean == 0)
            or not np.isfinite(normalization.input_std).all()
            or not np.all(normalization.input_std > 0)
            or not np.allclose(normalization.input_std, divisor, rtol=1e-6, atol=0)):
        raise ValueError(
            "Checkpoint normalization disagrees with depth scale: "
            "expected input_mean=[0,0] and input_std=[divisor,divisor]. "
            "Check the preprocessing used when this model was trained."
        )
    return divisor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=POLICY_DIR / "outputs" / "tactile_pose_merged_11439_depth" / "best_model.pt")
    parser.add_argument("--dataset-root", type=Path,
                        default=PROJECT_ROOT / "data" / "L1_L2_11439_xense_tactile_boltnut_merged")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    normalization = Normalization.from_dict(checkpoint["normalization"])
    # 历史检查保留：原来仅允许 /255。
    # # 新增：只接受本入口已适配的深度模型和固定 /255 预处理，防止误用分布力权重。
    # if checkpoint.get("model_type") != "native_depth_cnn":
    #     raise ValueError("Expected a native_depth_cnn checkpoint")
    # preprocessing = checkpoint.get("input_preprocessing", {})
    # if (checkpoint.get("input_channels") != 2
    #         or checkpoint.get("input_keys") != list(DEPTH_KEYS)
    #         or preprocessing.get("type") != "depth_png_uint8_div255"
    #         or preprocessing.get("divisor") != 255.0
    #         or preprocessing.get("resize") is not None):
    #     raise ValueError("Checkpoint does not match two-channel, unresized depth /255 inputs")
    # 修改：识别固定 /255 或训练集共同最大值缩放，使用模型保存的尺度。
    if checkpoint.get("model_type") != "native_depth_cnn":
        raise ValueError("Expected a native_depth_cnn checkpoint")
    preprocessing = checkpoint.get("input_preprocessing", {})
    if (checkpoint.get("input_channels") != 2
            or checkpoint.get("input_keys") != list(DEPTH_KEYS)
            or preprocessing.get("resize") is not None):
        raise ValueError("Expected two-channel depth inputs without resizing")
    validate_depth_scale(preprocessing, normalization)
    # if (normalization.input_mean.shape != (2,)
    #         or normalization.input_std.shape != (2,)
    #         or not np.all(normalization.input_mean == 0)
    #         or not np.all(normalization.input_std == 255)):
    #     raise ValueError("Expected saved depth normalization mean=0 and std=255")
    # 修改：同一行读取左右内嵌 PNG 和 pose，不再加载分布力。
    # inputs, targets = load_lerobot_tactile_dataset(
    # args.dataset_root, grid_size=int(checkpoint["grid_size"])
    # )
    inputs, targets = load_lerobot_depth_dataset(args.dataset_root)
    if not -len(inputs) <= args.sample_index < len(inputs):
        raise IndexError(f"sample index {args.sample_index} outside dataset of length {len(inputs)}")

    # 修改：原始深度转 float32，按 checkpoint 保存的固定尺度处理；不重新计算最大值。
    # tactile = inputs[args.sample_index]
    # tactile = (tactile - normalization.input_mean[:, None, None]) / normalization.input_std[:, None, None]
    tactile = inputs[args.sample_index].astype(np.float32)
    tactile = (tactile - normalization.input_mean[:, None, None]) / normalization.input_std[:, None, None]
    # 修改：按 checkpoint 的真实高宽建模，跳过 create_model/ResizeInput。
    # model_type = checkpoint.get("model_type", "simple_cnn")
    # if model_type == "tactile_servo_control_simple_cnn":
    #     model_type = "simple_cnn"
    # model = create_model(
    #     model_type,
    #     grid_size=int(checkpoint["grid_size"]),
    #     image_size=int(checkpoint.get("model_image_size", 128)),
    # ).to(device)
    input_hw = tuple(checkpoint["model_input_hw"])
    if tuple(tactile.shape) != (2, *input_hw):
        raise ValueError(f"Expected depth shape {(2, *input_hw)}, got {tactile.shape}")
    model = NativeDepthCNN(
        in_channels=2, out_dim=len(POSE_NAMES), input_hw=input_hw,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(tactile).unsqueeze(0).to(device)).cpu().numpy()[0]
    prediction = output * normalization.target_std + normalization.target_mean
    target = targets[args.sample_index]

    # 新增：显示所选记录和模型轮数；sample-index 不自动限制为验证集。
    print(f"sample_index={args.sample_index % len(inputs)} input={tuple(tactile.shape)} "
          f"model=native_depth_cnn checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print("                 prediction        target           error")
    for i, name in enumerate(POSE_NAMES):
        scale = 1000.0 if i < 3 else 180.0 / np.pi
        unit = "mm" if i < 3 else "deg"
        error = prediction[i] - target[i]
        if i >= 3:
            error = np.arctan2(np.sin(error), np.cos(error))
        print(f"{name:>5s} ({unit}): {prediction[i] * scale:+11.4f} {target[i] * scale:+11.4f} {error * scale:+11.4f}")


if __name__ == "__main__":
    main()
