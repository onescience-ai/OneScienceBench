"""Independent paper-driven implementation of the 2-D Radon Neural Operator."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class PhysicsAttention2D(nn.Module):
    """Slice/token/deslice Physics-Attention on a regular 2-D grid."""

    def __init__(self, channels: int, heads: int, slices: int, mlp_ratio: float, dropout: float):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.head_dim = channels // heads
        self.slices = slices
        self.norm1 = nn.GroupNorm(1, channels)
        self.feature = nn.Conv2d(channels, channels, 3, padding=1)
        self.slice_queries = nn.Parameter(torch.randn(heads, slices, self.head_dim) / math.sqrt(self.head_dim))
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Linear(channels, 3 * channels)
        self.projection = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        hidden = max(channels, int(channels * mlp_ratio))
        self.norm2 = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, 1), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        features = self.feature(self.norm1(x)).reshape(batch, self.heads, self.head_dim, -1)
        logits = torch.einsum("bhdn,hmd->bhmn", features, self.slice_queries)
        logits = logits / self.temperature.abs().clamp_min(0.05)
        gather = logits.softmax(dim=-1)
        dispatch = logits.softmax(dim=2)
        tokens = torch.einsum("bhmn,bhdn->bhmd", gather, features)
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch, self.slices, channels)
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        query = query.reshape(batch, self.slices, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(batch, self.slices, self.heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.reshape(batch, self.slices, self.heads, self.head_dim).permute(0, 2, 1, 3)
        attention = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        tokens = torch.matmul(attention.softmax(dim=-1), value)
        points = torch.einsum("bhmn,bhmd->bhdn", dispatch, tokens)
        points = points.reshape(batch, channels, height, width)
        x = x + self.dropout(self.projection(points))
        return x + self.mlp(self.norm2(x))


class DifferentiableRadon2D(nn.Module):
    """Bilinear-rotation parallel-beam Radon transform and ramp-filtered backprojection."""

    def __init__(self, angles: int):
        super().__init__()
        if angles < 1:
            raise ValueError("angles must be positive")
        theta = torch.arange(angles, dtype=torch.float32) * (math.pi / angles)
        self.register_buffer("angles", theta, persistent=True)

    @staticmethod
    def _pad_square(x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        height, width = x.shape[-2:]
        diagonal = int(math.ceil(math.sqrt(height * height + width * width)))
        top = (diagonal - height) // 2
        bottom = diagonal - height - top
        left = (diagonal - width) // 2
        right = diagonal - width - left
        return F.pad(x, (left, right, top, bottom)), height, width, diagonal

    def transform(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch, channels, _, _ = x.shape
        padded, height, width, diagonal = self._pad_square(x)
        flat = padded.reshape(batch * channels, 1, diagonal, diagonal)
        projections = []
        for angle in self.angles:
            cosine, sine = torch.cos(angle), torch.sin(angle)
            matrix = torch.stack((cosine, -sine, torch.zeros_like(angle), sine, cosine, torch.zeros_like(angle)))
            matrix = matrix.reshape(1, 2, 3).expand(flat.shape[0], -1, -1)
            grid = F.affine_grid(matrix, flat.shape, align_corners=True)
            rotated = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            projections.append(rotated.sum(dim=-2).reshape(batch, channels, diagonal))
        return torch.stack(projections, dim=-1), (height, width)

    @staticmethod
    def _ramp_filter(sinogram: torch.Tensor) -> torch.Tensor:
        detector = sinogram.shape[-2]
        fft_size = 1 << int(math.ceil(math.log2(max(2, 2 * detector))))
        spectrum = torch.fft.rfft(sinogram, n=fft_size, dim=-2)
        ramp = torch.fft.rfftfreq(fft_size, device=sinogram.device, dtype=sinogram.dtype).abs()
        filtered = torch.fft.irfft(spectrum * ramp.view(1, 1, -1, 1), n=fft_size, dim=-2)
        return filtered[..., :detector, :]

    def filtered_backprojection(self, sinogram: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
        batch, channels, detector, angle_count = sinogram.shape
        if angle_count != self.angles.numel():
            raise ValueError("Sinogram angle count does not match module configuration")
        filtered = self._ramp_filter(sinogram)
        height, width = output_shape
        y = torch.linspace(-(height - 1) / (detector - 1), (height - 1) / (detector - 1), height,
                           device=sinogram.device, dtype=sinogram.dtype)
        x = torch.linspace(-(width - 1) / (detector - 1), (width - 1) / (detector - 1), width,
                           device=sinogram.device, dtype=sinogram.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        reconstruction = sinogram.new_zeros(batch * channels, 1, height, width)
        for index, angle in enumerate(self.angles):
            detector_coordinate = xx * torch.cos(angle) + yy * torch.sin(angle)
            grid = torch.stack((torch.zeros_like(detector_coordinate), detector_coordinate), dim=-1)
            grid = grid.unsqueeze(0).expand(batch * channels, -1, -1, -1)
            line = filtered[..., index].reshape(batch * channels, 1, detector, 1)
            reconstruction += F.grid_sample(line, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        reconstruction *= math.pi / angle_count
        return reconstruction.reshape(batch, channels, height, width)


class SinogramWeightNetwork(nn.Module):
    """Paper Fig. 5 angle-only Conv(1,16)-GELU-Conv(16,1)-Sigmoid weighting."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 3, padding=1, padding_mode="circular"),
            nn.Sigmoid(),
        )

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        batch, channels, detector, angles = sinogram.shape
        lines = sinogram.reshape(batch * channels * detector, 1, angles)
        return self.network(lines).reshape(batch, channels, detector, angles)


class RadonBlock2D(nn.Module):
    def __init__(self, channels: int, angles: int, angle_hidden: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.radon = DifferentiableRadon2D(angles)
        self.weights = SinogramWeightNetwork(angle_hidden)
        self.output = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sinogram, output_shape = self.radon.transform(self.norm(x))
        reconstructed = self.radon.filtered_backprojection(sinogram * self.weights(sinogram), output_shape)
        return x + self.scale * self.output(reconstructed)


class RadonNeuralOperator2D(nn.Module):
    """RNO for Darcy Flow: lifting, 8 Physics-Attention layers, 4 Radon blocks, projection."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        channels: int = 128,
        layers: int = 8,
        heads: int = 8,
        slices: int = 32,
        angles: int = 32,
        radon_blocks: int = 4,
        angle_hidden: int = 16,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("Darcy RNO expects coefficient plus x/y coordinates (in_channels=3)")
        if not 0 <= radon_blocks <= layers:
            raise ValueError("radon_blocks must be between zero and layers")
        self.lifting = nn.Sequential(nn.Conv2d(in_channels, channels, 1), nn.GELU(), nn.Conv2d(channels, channels, 1))
        self.attention_layers = nn.ModuleList(
            PhysicsAttention2D(channels, heads, slices, mlp_ratio, dropout) for _ in range(layers)
        )
        radon_after = {max(0, math.ceil((index + 1) * layers / radon_blocks) - 1) for index in range(radon_blocks)} \
            if radon_blocks else set()
        self.radon_layers = nn.ModuleDict(
            {str(index): RadonBlock2D(channels, angles, angle_hidden) for index in sorted(radon_after)}
        )
        self.projection = nn.Sequential(nn.GroupNorm(1, channels), nn.Conv2d(channels, channels, 1), nn.GELU(),
                                        nn.Conv2d(channels, out_channels, 1))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RadonNeuralOperator2D":
        return cls(**dict(config))

    @staticmethod
    def _coordinates(x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(0.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(x.shape[0], -1, -1, -1)

    def forward(self, coefficient: torch.Tensor) -> torch.Tensor:
        if coefficient.ndim != 4 or coefficient.shape[1] != 1:
            raise ValueError(f"Expected coefficient with shape (B,1,H,W), got {tuple(coefficient.shape)}")
        x = self.lifting(torch.cat((coefficient, self._coordinates(coefficient)), dim=1))
        for index, layer in enumerate(self.attention_layers):
            x = layer(x)
            key = str(index)
            if key in self.radon_layers:
                x = self.radon_layers[key](x)
        return self.projection(x)
