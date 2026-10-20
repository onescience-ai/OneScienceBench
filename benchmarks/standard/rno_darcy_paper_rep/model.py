"""Flat, self-contained Radon Neural Operator implementation for 2-D Darcy flow.

The implementation is derived from the official RNO release while resolving the
paper-critical differences documented in ``repro_outputs/COMPARABILITY_REPORT.md``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> Callable[[], nn.Module]:
    activations = {
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_layers: int = 1,
        activation: str = "gelu",
        residual: bool = True,
    ):
        super().__init__()
        act = _activation(activation)
        self.residual = residual
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), act())
        self.hidden_layers = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act()) for _ in range(hidden_layers)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layer(inputs)
        for layer in self.hidden_layers:
            update = layer(hidden)
            hidden = hidden + update if self.residual else update
        return self.output_layer(hidden)


class PhysicsAttentionStructured2D(nn.Module):
    """Physics-attention on learnable slice tokens for a regular 2-D grid."""

    def __init__(
        self,
        dim: int,
        heads: int,
        slice_num: int,
        height: int,
        width: int,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden dimension {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head**-0.5
        self.height = height
        self.width = width
        inner_dim = self.dim_head * heads
        padding = kernel_size // 2

        self.project_x = nn.Conv2d(dim, inner_dim, kernel_size, padding=padding)
        self.project_features = nn.Conv2d(dim, inner_dim, kernel_size, padding=padding)
        self.project_slices = nn.Linear(self.dim_head, slice_num)
        nn.init.orthogonal_(self.project_slices.weight)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = values.shape
        nodes = height * width
        return (
            values.permute(0, 2, 3, 1)
            .reshape(batch, nodes, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, nodes, channels = inputs.shape
        if nodes != self.height * self.width:
            raise ValueError(f"Expected {self.height * self.width} nodes, received {nodes}")
        spatial = inputs.reshape(batch, self.height, self.width, channels).permute(0, 3, 1, 2).contiguous()

        feature_values = self._split_heads(self.project_features(spatial))
        coordinate_values = self._split_heads(self.project_x(spatial))
        temperature = self.temperature.clamp(min=0.1, max=5.0)
        slice_weights = torch.softmax(self.project_slices(coordinate_values) / temperature, dim=-1)
        slice_norm = slice_weights.sum(dim=2).clamp_min(1.0e-5)
        slice_tokens = torch.einsum("bhnc,bhng->bhgc", feature_values, slice_weights)
        slice_tokens = slice_tokens / slice_norm.unsqueeze(-1)

        queries = self.to_q(slice_tokens)
        keys = self.to_k(slice_tokens)
        values = self.to_v(slice_tokens)
        attention = torch.softmax(torch.matmul(queries, keys.transpose(-1, -2)) * self.scale, dim=-1)
        attention = self.dropout(attention)
        attended_tokens = torch.matmul(attention, values)

        output = torch.einsum("bhgc,bhng->bhnc", attended_tokens, slice_weights)
        output = output.permute(0, 2, 1, 3).reshape(batch, nodes, -1)
        return self.to_out(output)


class RNOTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        slice_num: int,
        height: int,
        width: int,
        dropout: float,
        mlp_ratio: int,
        output_dim: int,
        last_layer: bool,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = PhysicsAttentionStructured2D(
            dim=hidden_dim,
            heads=num_heads,
            slice_num=slice_num,
            height=height,
            width=width,
            dropout=dropout,
        )
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            hidden_layers=0,
            residual=False,
        )
        if last_layer:
            self.norm_output = nn.LayerNorm(hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.norm_attention(inputs))
        hidden = hidden + self.mlp(self.norm_mlp(hidden))
        if self.last_layer:
            return self.output(self.norm_output(hidden))
        return hidden


def _fourier_filter(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = size // 2
    indices = torch.cat(
        (
            torch.arange(1, half + 1, 2, device=device, dtype=dtype),
            torch.arange(half - 1, 0, -2, device=device, dtype=dtype),
        )
    )
    impulse = torch.zeros(size, device=device, dtype=dtype)
    impulse[0] = 0.25
    impulse[1::2] = -1.0 / (torch.pi * indices).square()
    return 2.0 * torch.fft.fft(impulse).real


class DifferentiableRadon2D(nn.Module):
    """Differentiable Radon projection and filtered backprojection."""

    def __init__(self, num_angles: int, include_pi_endpoint: bool = True):
        super().__init__()
        if include_pi_endpoint:
            angles = torch.linspace(0.0, torch.pi, num_angles)
        else:
            angles = torch.linspace(0.0, torch.pi, num_angles + 1)[:-1]
        self.num_angles = num_angles
        self.register_buffer("angles", angles, persistent=True)

    @staticmethod
    def _padding(height: int, width: int) -> tuple[int, int, int, int, int, int]:
        diagonal = int(math.ceil(math.sqrt(2.0) * max(height, width)))
        pad_height = diagonal - height
        pad_width = diagonal - width
        top = pad_height // 2
        left = pad_width // 2
        return left, pad_width - left, top, pad_height - top, diagonal, diagonal

    def forward_projection(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        left, right, top, bottom, padded_height, padded_width = self._padding(height, width)
        padded = F.pad(image, (left, right, top, bottom), mode="constant", value=0.0)
        projections = []
        for angle in self.angles.to(dtype=image.dtype):
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            zero = torch.zeros((), device=image.device, dtype=image.dtype)
            theta = torch.stack((cosine, sine, zero, -sine, cosine, zero)).reshape(1, 2, 3)
            theta = theta.expand(batch, -1, -1)
            grid = F.affine_grid(
                theta,
                torch.Size((batch, channels, padded_height, padded_width)),
                align_corners=True,
            )
            rotated = F.grid_sample(padded, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            projections.append(rotated.sum(dim=3))
        return torch.stack(projections, dim=-1)

    def filtered_backprojection(self, sinogram: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
        batch, channels, detector_count, angle_count = sinogram.shape
        if angle_count != self.num_angles:
            raise ValueError(f"Expected {self.num_angles} sinogram angles, received {angle_count}")
        height, width = output_shape
        padded_size = max(64, 2 ** math.ceil(math.log2(2 * detector_count)))
        padded = F.pad(sinogram, (0, 0, 0, padded_size - detector_count))
        ramp = _fourier_filter(padded_size, sinogram.device, sinogram.dtype).reshape(1, 1, -1, 1)
        filtered = torch.fft.ifft(torch.fft.fft(padded, dim=2) * ramp, dim=2).real[:, :, :detector_count]

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=sinogram.device, dtype=sinogram.dtype),
            torch.linspace(-1.0, 1.0, width, device=sinogram.device, dtype=sinogram.dtype),
            indexing="ij",
        )
        reconstructed = torch.zeros(
            (batch, channels, height, width), device=sinogram.device, dtype=sinogram.dtype
        )
        angle_coordinates = torch.linspace(
            -1.0, 1.0, self.num_angles, device=sinogram.device, dtype=sinogram.dtype
        )
        for angle, angle_coordinate in zip(self.angles.to(dtype=sinogram.dtype), angle_coordinates):
            detector_coordinate = x_grid * torch.cos(angle) - y_grid * torch.sin(angle)
            projection_coordinate = torch.full_like(detector_coordinate, angle_coordinate)
            grid = torch.stack((projection_coordinate, detector_coordinate), dim=-1).unsqueeze(0)
            backprojected = F.grid_sample(
                filtered,
                grid.expand(batch, -1, -1, -1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            reconstructed = reconstructed + backprojected
        return reconstructed * torch.pi / (2.0 * self.num_angles)


class AngleWeightNetwork(nn.Module):
    """Paper convolutional weighting or the released repository's linear fallback."""

    def __init__(
        self,
        num_angles: int,
        mode: str,
        conv_channels: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.mode = mode
        if mode == "paper_conv":
            if kernel_size % 2 == 0:
                raise ValueError("The angle convolution kernel size must be odd")
            self.network = nn.Sequential(
                nn.Conv1d(1, conv_channels, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv1d(conv_channels, 1, kernel_size, padding=kernel_size // 2),
                nn.Sigmoid(),
            )
        elif mode == "repo_linear":
            self.network = nn.Sequential(
                nn.Linear(num_angles, num_angles),
                nn.GELU(),
                nn.Linear(num_angles, num_angles),
                nn.Softmax(dim=-1),
            )
        else:
            raise ValueError(f"Unsupported angle weighting mode: {mode}")

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        summary = sinogram.mean(dim=(0, 1, 2)).reshape(1, -1)
        if self.mode == "paper_conv":
            weights = self.network(summary.unsqueeze(1)).squeeze(1)
        else:
            weights = self.network(summary)
        return weights.reshape(1, 1, 1, -1)


class RadonResidualBlock(nn.Module):
    def __init__(
        self,
        num_angles: int,
        angle_weighting: str,
        angle_conv_channels: int,
        angle_conv_kernel_size: int,
        include_pi_endpoint: bool,
    ):
        super().__init__()
        self.radon = DifferentiableRadon2D(num_angles, include_pi_endpoint=include_pi_endpoint)
        self.angle_weights = AngleWeightNetwork(
            num_angles,
            mode=angle_weighting,
            conv_channels=angle_conv_channels,
            kernel_size=angle_conv_kernel_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sinogram = self.radon.forward_projection(inputs)
        weighted = sinogram * self.angle_weights(sinogram)
        reconstruction = self.radon.filtered_backprojection(weighted, inputs.shape[-2:])
        return inputs + reconstruction


class RadonNeuralOperator2D(nn.Module):
    """RNO for coefficient-to-solution learning on the Darcy regular grid."""

    def __init__(
        self,
        height: int = 85,
        width: int = 85,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        num_angles: int = 32,
        num_radon_blocks: int = 4,
        slice_num: int = 64,
        ref_grid_size: int = 8,
        unified_pos: bool = True,
        dropout: float = 0.0,
        mlp_ratio: int = 1,
        input_dim: int = 1,
        output_dim: int = 1,
        angle_weighting: str = "paper_conv",
        angle_conv_channels: int = 16,
        angle_conv_kernel_size: int = 3,
        include_pi_endpoint: bool = True,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.hidden_dim = hidden_dim
        self.unified_pos = unified_pos
        self.ref_grid_size = ref_grid_size

        if unified_pos:
            unified_features = self._make_unified_position_features(height, width, ref_grid_size)
            self.register_buffer("unified_position_features", unified_features, persistent=True)
            position_dim = ref_grid_size * ref_grid_size
        else:
            self.register_buffer("unified_position_features", torch.empty(0), persistent=False)
            position_dim = 2

        self.preprocess = MLP(
            input_dim + position_dim,
            hidden_dim * 2,
            hidden_dim,
            hidden_layers=0,
            residual=False,
        )
        self.local_feature = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        self.radon_blocks = nn.ModuleList(
            RadonResidualBlock(
                num_angles=num_angles,
                angle_weighting=angle_weighting,
                angle_conv_channels=angle_conv_channels,
                angle_conv_kernel_size=angle_conv_kernel_size,
                include_pi_endpoint=include_pi_endpoint,
            )
            for _ in range(num_radon_blocks)
        )
        self.fusion = nn.Sequential(nn.Conv2d(2 * hidden_dim, 1, kernel_size=1), nn.GELU())
        self.operator_blocks = nn.ModuleList(
            RNOTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                slice_num=slice_num,
                height=height,
                width=width,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                output_dim=output_dim,
                last_layer=index == num_layers - 1,
            )
            for index in range(num_layers)
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _make_unified_position_features(height: int, width: int, reference_size: int) -> torch.Tensor:
        x_axis = torch.linspace(0.0, 1.0, height)
        y_axis = torch.linspace(0.0, 1.0, width)
        grid_x, grid_y = torch.meshgrid(x_axis, y_axis, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        ref_axis = torch.linspace(0.0, 1.0, reference_size)
        ref_x, ref_y = torch.meshgrid(ref_axis, ref_axis, indexing="ij")
        reference = torch.stack((ref_x, ref_y), dim=-1)
        distances = torch.linalg.vector_norm(
            grid[:, :, None, None, :] - reference[None, None, :, :, :], dim=-1
        )
        return distances.reshape(1, height * width, reference_size * reference_size)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(self, positions: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = field.shape
        if nodes != self.height * self.width:
            raise ValueError(f"Expected {self.height * self.width} nodes, received {nodes}")
        if self.unified_pos:
            position_features = self.unified_position_features.expand(batch, -1, -1)
        else:
            position_features = positions
        hidden = self.preprocess(torch.cat((position_features, field), dim=-1))

        spatial = hidden.reshape(batch, self.height, self.width, self.hidden_dim).permute(0, 3, 1, 2)
        local = self.local_feature(spatial.reshape(batch * self.hidden_dim, 1, self.height, self.width))
        local = local.reshape(batch, self.hidden_dim, self.height, self.width)
        global_features = spatial
        for block in self.radon_blocks:
            global_features = block(global_features)
        fused = self.fusion(torch.cat((global_features, local), dim=1))
        hidden = (spatial + fused).permute(0, 2, 3, 1).reshape(batch, nodes, self.hidden_dim)

        for block in self.operator_blocks:
            hidden = block(hidden)
        return hidden


RNO2D = RadonNeuralOperator2D


def build_model(config: dict) -> RadonNeuralOperator2D:
    model_cfg = config["model"]
    resolution = int(config["data"]["resolution"])
    return RadonNeuralOperator2D(
        height=resolution,
        width=resolution,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        num_angles=int(model_cfg["num_angles"]),
        num_radon_blocks=int(model_cfg["num_radon_blocks"]),
        slice_num=int(model_cfg["slice_num"]),
        ref_grid_size=int(model_cfg["ref_grid_size"]),
        unified_pos=bool(model_cfg["unified_pos"]),
        dropout=float(model_cfg["dropout"]),
        mlp_ratio=int(model_cfg["mlp_ratio"]),
        input_dim=int(model_cfg["input_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        angle_weighting=str(model_cfg["angle_weighting"]),
        angle_conv_channels=int(model_cfg["angle_conv_channels"]),
        angle_conv_kernel_size=int(model_cfg["angle_conv_kernel_size"]),
        include_pi_endpoint=bool(model_cfg["include_pi_endpoint"]),
    )

