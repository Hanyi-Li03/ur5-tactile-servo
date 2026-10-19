"""Loading and preprocessing for the fixed-nut LeRobot tactile dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import torch
from torch.utils.data import Dataset


# LEFT_FORCE_KEY = "observation.tactile_left.force"
# RIGHT_FORCE_KEY = "observation.tactile_right.force"
# POSE_KEY = "observation.tcp_pose"
# POSE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
LEFT_FORCE_KEY = "observation.tactile_left.distributed_force"
RIGHT_FORCE_KEY = "observation.tactile_right.distributed_force"
NUT_POSE_KEY = "observation.nut_pose"
WORLD_POSE_KEY = "observation.fingertip_center_pose"
POSE_KEY = "observation.fingertip_center_pose_relative_to_nut"
POSE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

@dataclass
class Normalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> "Normalization":
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in values.items()})


def signed_log1p(force: np.ndarray) -> np.ndarray:
    """Compress force outliers without discarding the force-vector sign."""
    return np.sign(force) * np.log1p(np.abs(force))


def _force_to_chw(force: np.ndarray, grid_size: int) -> np.ndarray:
    # 新的真实 Xense 数据：(N,H,W,3)
    if force.ndim == 4 and force.shape[-1] == 3:
        return force.transpose(0, 3, 1, 2)

    # 兼容旧的扁平仿真数据：(N,H*W,3)
    expected = grid_size * grid_size
    if force.shape != (force.shape[0], expected, 3):
        raise ValueError(
            f"Expected tactile force shape (N, {expected}, 3), got {force.shape}. "
            "Set --grid-size to match the recorded sensor resolution."
        )
    return force.reshape(-1, grid_size, grid_size, 3).transpose(0, 3, 1, 2)


def load_lerobot_tactile_dataset(
    dataset_root: str | Path,
    grid_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Load left/right force maps and xyz+RPY targets into memory.

    Returns:
        inputs: float32 array shaped (N, 6, H, W), left channels first.
        targets: float32 array shaped (N, 6), in metres and radians.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"LeRobot parquet directory not found: {data_dir}")

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    # Explicit sorting makes the train/validation split and --sample-index
    # reproducible across filesystems.
    table = pads.dataset([str(path) for path in parquet_files], format="parquet").to_table(
        columns=[LEFT_FORCE_KEY, RIGHT_FORCE_KEY, POSE_KEY]
    )
    if table.num_rows == 0:
        raise ValueError(f"No samples found under {data_dir}")

    left = np.asarray(table[LEFT_FORCE_KEY].to_pylist(), dtype=np.float32)
    right = np.asarray(table[RIGHT_FORCE_KEY].to_pylist(), dtype=np.float32)
    targets = np.asarray(table[POSE_KEY].to_pylist(), dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 6:
        raise ValueError(f"Expected {POSE_KEY} shape (N, 6), got {targets.shape}")

    left = _force_to_chw(left, grid_size)
    right = _force_to_chw(right, grid_size)
    inputs = np.concatenate((left, right), axis=1)
    return signed_log1p(inputs).astype(np.float32), targets


def fit_normalization(inputs: np.ndarray, targets: np.ndarray) -> Normalization:
    """Fit statistics on the training split only."""
    input_mean = inputs.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    input_std = inputs.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    target_mean = targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_std = targets.std(axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.maximum(input_std, 1e-6)
    target_std = np.maximum(target_std, 1e-6)
    return Normalization(input_mean, input_std, target_mean, target_std)


class TactilePoseDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        normalization: Normalization,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.input_mean = normalization.input_mean[:, None, None]
        self.input_std = normalization.input_std[:, None, None]
        self.target_mean = normalization.target_mean
        self.target_std = normalization.target_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        image = (self.inputs[index] - self.input_mean) / self.input_std
        target = (self.targets[index] - self.target_mean) / self.target_std
        return torch.from_numpy(image), torch.from_numpy(target)


# 新增：从 train_pose_real_image.py 迁入深度数据处理；以上原有内容保持不变。
# 新增：左右字段顺序同时决定输入的第 0、1 通道；pose 使用同一条记录。
DEPTH_KEYS = ("observation.tactile_left.depth", "observation.tactile_right.depth")


def decode_depth_png(value: dict) -> np.ndarray:
    """一次解码，缓存原始 uint8 深度；不缩放、不逐图拉伸、不做对数压缩。"""
    # 新增：仅深度解码时加载 PNG 依赖，不影响原分布力读取流程。
    from io import BytesIO
    from PIL import Image

    if not value or not value.get("bytes"):
        raise ValueError("Expected embedded depth PNG bytes")
    with Image.open(BytesIO(value["bytes"])) as image:
        depth = np.array(image)
    if depth.ndim == 3:
        if depth.shape[-1] not in (1, 3) or not np.all(depth == depth[..., :1]):
            raise ValueError("Depth PNG must have identical grayscale channels")
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.dtype != np.uint8:
        raise ValueError(f"Expected 2D uint8 depth, got {depth.shape}, {depth.dtype}")
    return np.ascontiguousarray(depth)


def load_lerobot_depth_dataset(dataset_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """新增：启动时一次性解码左右图，返回 [N,2,H,W] 和同一行的六维 pose。"""
    data_dir = Path(dataset_root).expanduser().resolve() / "data"
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {data_dir}")
    table = pads.dataset([str(path) for path in files], format="parquet").to_table(
        columns=[*DEPTH_KEYS, POSE_KEY]
    )
    if table.num_rows == 0:
        raise ValueError("Depth dataset is empty")
    sides = []
    for key in DEPTH_KEYS:
        frames = []
        for index, value in enumerate(table[key].to_pylist()):
            try:
                frames.append(decode_depth_png(value))
            except (ValueError, OSError, TypeError) as exc:
                raise ValueError(f"Invalid {key} at row {index}: {exc}") from exc
        sides.append(np.stack(frames))
    inputs = np.stack(sides, axis=1)
    targets = np.asarray(table[POSE_KEY].to_pylist(), dtype=np.float32)
    if targets.shape != (len(inputs), 6) or not np.isfinite(targets).all():
        raise ValueError(f"Expected finite pose targets shaped ({len(inputs)}, 6)")
    return inputs, targets


def fit_depth_normalization(
    targets: np.ndarray,
    input_scale: float = 255.0,
) -> Normalization:
    """新增：只拟合训练集 pose；图像固定为 (pixel - 0) / input_scale。"""


    return Normalization(
        input_mean=np.zeros(2, dtype=np.float32),
        input_std=np.full(2, input_scale, dtype=np.float32),
        target_mean=targets.mean(axis=0, dtype=np.float64).astype(np.float32),
        target_std=np.maximum(
            targets.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
        ),
    )


class TactileDepthPoseDataset(TactilePoseDataset):
    """新增：复用索引和标签管理，仅替换分布力的输入预处理。"""

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        # 修改：PNG 已在启动时解码；这里使用训练阶段保存的偏移和尺度。
        # 最大值缩放时 input_mean=0、input_std=input_scale；pose 仍按均值/标准差处理。
        image = (self.inputs[index] - self.input_mean) / self.input_std
        # image = self.inputs[index].astype(np.float32) / 255.0
        target = (self.targets[index] - self.target_mean) / self.target_std
        return torch.from_numpy(image), torch.from_numpy(target)
