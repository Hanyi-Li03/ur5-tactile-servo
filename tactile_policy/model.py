"""Models adapted from ``tactile_learning.supervised.models``."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MODEL_TYPES = (
    "fcn", "simple_cnn", "posenet_cnn", "nature_cnn", "resnet", "vit",
    "tactile_tokenizer",
)


def reference_weight_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0.0)


def activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ResizeInput(nn.Module):
    def __init__(self, model: nn.Module, image_size: int = 128) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return self.model(x)


class FCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 6) -> None:
        super().__init__()
        widths = (2048, 1024, 512, 256, 128)
        modules: list[nn.Module] = []
        previous = in_dim
        for index, width in enumerate(widths):
            modules.extend((nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ELU(inplace=True)))
            if index > 0:
                modules.append(nn.Dropout(0.0))
            previous = width
        modules.append(nn.Linear(previous, out_dim))
        self.fc = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class CNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, image_size: int,
                 conv_layers: tuple[int, ...], kernels: tuple[int, ...],
                 fc_layers: tuple[int, ...], activation_name: str) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        previous = in_channels
        for width, kernel in zip(conv_layers, kernels):
            modules.extend((nn.Conv2d(previous, width, kernel, padding=2), nn.BatchNorm2d(width),
                            activation(activation_name), nn.MaxPool2d(2, 2)))
            previous = width
        self.cnn = nn.Sequential(*modules)
        with torch.no_grad():
            feature_dim = self.cnn(torch.zeros(1, in_channels, image_size, image_size)).numel()
        modules = []
        previous = feature_dim
        for index, width in enumerate(fc_layers):
            modules.extend((nn.Linear(previous, width), activation(activation_name)))
            if index > 0:
                modules.append(nn.Dropout(0.0))
            previous = width
        modules.append(nn.Linear(previous, out_dim))
        self.fc = nn.Sequential(*modules)
        self.apply(reference_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(x).flatten(1))


class NatureCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, image_size: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True), nn.Flatten(),
        )
        with torch.no_grad():
            feature_dim = self.cnn(torch.zeros(1, in_channels, image_size, image_size)).shape[1]
        self.fc = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(inplace=True),
                                nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.0),
                                nn.Linear(512, out_dim))
        self.apply(reference_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels))
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride),
                                            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.conv2(self.conv1(x)) + residual)


class ResNet(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, layers: tuple[int, ...] = (2, 2, 2, 2)) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 7, 2, 3),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        stages, previous = [], 64
        for width, count in zip((64, 128, 256, 512), layers):
            stage = [ResidualBlock(previous, width, 1 if width == 64 else 2)]
            stage.extend(ResidualBlock(width, width) for _ in range(1, count))
            stages.append(nn.Sequential(*stage))
            previous = width
        self.stages = nn.Sequential(*stages)
        self.avgpool = nn.AvgPool2d(4, 1)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.maxpool(self.conv1(x)))
        return self.fc(self.avgpool(x).flatten(1))


class VisionTransformer(nn.Module):
    """Native PyTorch equivalent of the reference vit_pytorch configuration."""
    def __init__(self, in_channels: int, out_dim: int, image_size: int = 128) -> None:
        super().__init__()
        patch_size, dim = 32, 128
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.empty(1, num_patches, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, 512, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 6, norm=nn.LayerNorm(dim))
        self.head = nn.Linear(dim, out_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return self.head(self.transformer(x + self.position_embedding).mean(1))


class ResBlock2d(nn.Module):
    """Residual spatial block from TacForeSight's tactile tokenizer."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return x + h


class Downsample2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SingleHandSpatialEncoder(nn.Module):
    """TacForeSight shared hand encoder: (N,36,20,3) -> (N,9,5,64)."""

    def __init__(self, in_channels=3, hand_dim=64, hidden=96, mid=192,
                 dropout=0.1) -> None:
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.Conv2d(hidden, hidden, 1),
        )
        self.res1 = ResBlock2d(hidden, dropout)
        self.down1 = Downsample2d(hidden)
        self.res2 = ResBlock2d(hidden, dropout)
        self.proj2 = nn.Conv2d(hidden, mid, 3, padding=1)
        self.down2 = Downsample2d(mid)
        self.mid = nn.Sequential(
            ResBlock2d(mid, dropout),
            ResBlock2d(mid, dropout),
            ResBlock2d(mid, dropout),
        )
        self.out_norm = nn.GroupNorm(min(8, mid), mid, eps=1e-6)
        self.conv_out = nn.Conv2d(mid, hand_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        h = self.down1(self.res1(self.conv_in(x)))
        h = self.down2(self.proj2(self.res2(h)))
        h = self.mid(h)
        h = self.conv_out(F.silu(self.out_norm(h)))
        return h.permute(0, 2, 3, 1).contiguous()


class PatchTransformerAggregator(nn.Module):
    def __init__(self, dim=384, depth=6, heads=6, mlp_ratio=4.0,
                 dropout=0.1, num_patches=90, pool="cls") -> None:
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")
        self.num_patches = num_patches
        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + num_patches, dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        batch, patches, _ = x.shape
        if patches != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {patches}")
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, x), dim=1)
        tokens = self.norm(self.encoder(self.drop(tokens + self.pos_emb[:, :patches + 1])))
        global_token = tokens[:, 0] if self.pool == "cls" else tokens[:, 1:].mean(dim=1)
        return global_token, tokens


class TactileTokenizer(nn.Module):
    """TacForeSight tokenizer accepting (B,T,36,20,6) tactile sequences."""

    def __init__(self, in_channels=3, hand_dim=64, embed_dim=384,
                 hidden_channels=96, mid_channels=192, depth=6, heads=6,
                 dropout=0.1, pool="cls") -> None:
        super().__init__()
        self.hand_dim = hand_dim
        self.embed_dim = embed_dim
        self.encoder = SingleHandSpatialEncoder(
            in_channels, hand_dim, hidden_channels, mid_channels, dropout
        )
        self.row_pos = nn.Parameter(torch.randn(1, 1, 9, 1, hand_dim) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(1, 1, 1, 5, hand_dim) * 0.02)
        self.left_emb = nn.Parameter(torch.randn(1, 1, 1, 1, hand_dim) * 0.02)
        self.right_emb = nn.Parameter(torch.randn(1, 1, 1, 1, hand_dim) * 0.02)
        self.token_proj = nn.Sequential(nn.LayerNorm(hand_dim), nn.Linear(hand_dim, embed_dim))
        self.transformer = PatchTransformerAggregator(
            embed_dim, depth, heads, 4.0, dropout, 90, pool
        )

    def encode_hands(self, x: torch.Tensor):
        batch, time, height, width, channels = x.shape
        if channels != 6:
            raise ValueError(f"Expected six tactile channels, got {channels}")
        x = x.reshape(batch * time, height, width, channels)
        left = self.encoder(x[..., :3]).reshape(batch, time, 9, 5, self.hand_dim)
        right = self.encoder(x[..., 3:]).reshape(batch, time, 9, 5, self.hand_dim)
        return left, right

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, time, _, _, _ = x.shape
        left, right = self.encode_hands(x)
        left = left + self.row_pos + self.col_pos + self.left_emb
        right = right + self.row_pos + self.col_pos + self.right_emb
        left_tokens = left.reshape(batch * time, 45, self.hand_dim)
        right_tokens = right.reshape(batch * time, 45, self.hand_dim)
        patch_tokens = self.token_proj(torch.cat((left_tokens, right_tokens), dim=1))
        global_token, all_tokens = self.transformer(patch_tokens)
        return {
            "global": global_token.reshape(batch, time, self.embed_dim),
            "tokens": all_tokens.reshape(batch, time, 91, self.embed_dim),
            "left_hand": left,
            "right_hand": right,
        }


class TactileTokenizerPoseRegressor(nn.Module):
    """TacForeSight tactile features followed by a six-DoF regression head."""

    def __init__(self, in_channels=6, out_dim=6, embed_dim=384) -> None:
        super().__init__()
        if in_channels != 6:
            raise ValueError("The tokenizer expects left XYZ + right XYZ (6 channels)")
        self.tokenizer = TactileTokenizer(embed_dim=embed_dim)
        self.pose_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dataset tensors are BCHW. TacForeSight uses BTHWC and was designed
        # for a 36x20 tactile surface. The two stride-2 downsampling layers map
        # both the native Xense 35x20 grid and 36x20 to the required 9x5 token
        # grid, so preserve those samples exactly. Legacy/incompatible spatial
        # sizes still need resizing to keep the fixed 90-token layout.
        # T=1 for the current single-frame data.
        if x.shape[-2:] not in ((35, 20), (36, 20)):
            x = F.interpolate(x, size=(36, 20), mode="bilinear", align_corners=False)
        sequence = x.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        features = self.tokenizer(sequence)["global"][:, 0]
        return self.pose_head(features)


def create_model(model_type: str, in_channels: int = 6, output_dim: int = 6,
                 grid_size: int = 20, image_size: int = 128) -> nn.Module:
    if model_type == "fcn":
        return FCN(in_channels * grid_size * grid_size, output_dim)
    if model_type == "simple_cnn":
        core = CNN(in_channels, output_dim, image_size, (32,) * 4, (11, 9, 7, 5), (512, 512), "relu")
    elif model_type == "posenet_cnn":
        core = CNN(in_channels, output_dim, image_size, (256,) * 5, (3,) * 5, (64,), "elu")
    elif model_type == "nature_cnn":
        core = NatureCNN(in_channels, output_dim, image_size)
    elif model_type == "resnet":
        core = ResNet(in_channels, output_dim)
    elif model_type == "vit":
        core = VisionTransformer(in_channels, output_dim, image_size)
    elif model_type == "tactile_tokenizer":
        return TactileTokenizerPoseRegressor(in_channels, output_dim)
    else:
        raise ValueError(f"Unknown model type {model_type!r}; choose from {MODEL_TYPES}")
    return ResizeInput(core, image_size)


class TactilePoseCNN(nn.Module):
    """Backward-compatible name for ``simple_cnn``."""
    def __init__(self, in_channels: int = 6, output_dim: int = 6, image_size: int = 128) -> None:
        super().__init__()
        self.model = create_model("simple_cnn", in_channels, output_dim, image_size=image_size)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        return self.model(tactile)


# 新增：原始尺寸深度 CNN，独立于原有模型工厂，不改变分布力模型的行为。
class NativeDepthCNN(nn.Module):
    """直接接收左右深度 [B,2,H,W]，不缩放、不补成正方形。"""

    def __init__(self, in_channels: int = 2, out_dim: int = 6,
                 input_hw: tuple[int, int] = (35, 20)) -> None:
        super().__init__()
        self.input_hw = tuple(input_hw)
        self.in_channels = in_channels
        if len(self.input_hw) != 2 or min(self.input_hw) < 4:
            raise ValueError("input_hw must contain height and width >= 4")
        # 新增：3×3 卷积的 padding=1 只保持特征尺寸；两次池化适配小网格。
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 35×20 -> 17×10
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 17×10 -> 8×5
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        # 新增：按真实长方形尺寸计算展平维度，保留特征的空间位置。
        height, width = self.input_hw
        feature_dim = 64 * (height // 4) * (width // 4)
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True), nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 新增：尺寸不匹配时明确报错，禁止隐式插值或改变通道顺序。
        if x.ndim != 4 or x.shape[1] != self.in_channels or tuple(x.shape[-2:]) != self.input_hw:
            raise ValueError(
                f"Expected [B,{self.in_channels},{self.input_hw[0]},{self.input_hw[1]}], "
                f"got {tuple(x.shape)}"
            )
        return self.regressor(self.features(x))
