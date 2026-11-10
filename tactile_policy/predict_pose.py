#!/usr/bin/env python3
"""Run a saved tactile pose model on one sample from a LeRobot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# 数据和权重均以当前脚本位置解析，直接运行与包方式运行使用同一仓库。
POLICY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = POLICY_DIR.parent

try:
    from .data import Normalization, POSE_NAMES, load_lerobot_tactile_dataset
    from .model import create_model
except ImportError:
    from data import Normalization, POSE_NAMES, load_lerobot_tactile_dataset
    from model import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=POLICY_DIR / "outputs" / "tactile_pose_merged_11439_simple_cnn" / "best_model.pt")
    parser.add_argument("--dataset-root", type=Path,
                        default=PROJECT_ROOT / "data" / "L1_L2_11439_xense_tactile_boltnut_merged")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    normalization = Normalization.from_dict(checkpoint["normalization"])
    inputs, targets = load_lerobot_tactile_dataset(
        args.dataset_root, grid_size=int(checkpoint["grid_size"])
    )
    if not -len(inputs) <= args.sample_index < len(inputs):
        raise IndexError(f"sample index {args.sample_index} outside dataset of length {len(inputs)}")

    tactile = inputs[args.sample_index]
    tactile = (tactile - normalization.input_mean[:, None, None]) / normalization.input_std[:, None, None]
    model_type = checkpoint.get("model_type", "simple_cnn")
    if model_type == "tactile_servo_control_simple_cnn":
        model_type = "simple_cnn"
    model = create_model(
        model_type,
        grid_size=int(checkpoint["grid_size"]),
        image_size=int(checkpoint.get("model_image_size", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(tactile).unsqueeze(0).to(device)).cpu().numpy()[0]
    prediction = output * normalization.target_std + normalization.target_mean
    target = targets[args.sample_index]

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
