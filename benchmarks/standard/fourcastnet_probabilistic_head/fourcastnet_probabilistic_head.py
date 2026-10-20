from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AFNOBlock(nn.Module):
    """Compact spectral residual block used by the auditable reference model."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.spectral_scale = nn.Parameter(torch.ones(channels))
        self.channel_mixer = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        normalized = self.norm(x)
        spectrum = torch.fft.rfft2(normalized, norm="ortho")
        spectrum = spectrum * self.spectral_scale[None, :, None, None]
        mixed = torch.fft.irfft2(spectrum, s=x.shape[-2:], norm="ortho")
        return x + self.channel_mixer(mixed)


class FourCastNetProbabilisticHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        depth: int = 2,
        min_scale: float = 1e-4,
    ):
        super().__init__()
        if min(in_channels, out_channels, hidden_channels, depth) <= 0:
            raise ValueError("channel counts and depth must be positive")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.min_scale = float(min_scale)
        self.input_projection = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([AFNOBlock(hidden_channels) for _ in range(depth)])
        self.mean_head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.log_scale_head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("x must have shape [B,C,H,W]")
        if x.shape[1] != self.in_channels:
            raise ValueError("input channel count does not match in_channels")
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden)
        mean = self.mean_head(hidden)
        scale = F.softplus(self.log_scale_head(hidden)) + self.min_scale
        return {"mean": mean, "scale": scale}


def gaussian_nll(prediction, target):
    if not isinstance(prediction, dict) or set(prediction) != {"mean", "scale"}:
        raise ValueError("prediction must contain exactly mean and scale")
    mean, scale = prediction["mean"], prediction["scale"]
    if mean.shape != target.shape or scale.shape != target.shape:
        raise ValueError("mean, scale and target must have identical shapes")
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scale must be finite and positive")
    return (0.5 * ((target - mean) / scale).square() + torch.log(scale)).mean()
