from __future__ import annotations

import torch
from torch import nn


class GraphCastConservationProjection(nn.Module):
    """Project selected output variables onto requested area-weighted integrals."""

    def __init__(self, constrained_channels=(0,), eps: float = 1e-8):
        super().__init__()
        channels = tuple(int(index) for index in constrained_channels)
        if not channels or min(channels) < 0 or len(channels) != len(set(channels)):
            raise ValueError("constrained_channels must contain unique non-negative indices")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.constrained_channels = channels
        self.eps = float(eps)

    def forward(self, forecast, target_integrals, area_weights, mask=None):
        if not isinstance(forecast, torch.Tensor) or forecast.ndim != 4:
            raise ValueError("forecast must have shape [B,C,H,W]")
        batch, channels, height, width = forecast.shape
        if max(self.constrained_channels) >= channels:
            raise ValueError("constrained channel index is outside forecast channels")
        if target_integrals.shape != (batch, len(self.constrained_channels)):
            raise ValueError("target_integrals must have shape [B,K]")
        area_weights = torch.as_tensor(area_weights, dtype=forecast.dtype, device=forecast.device)
        if area_weights.shape != (height, width):
            raise ValueError("area_weights must have shape [H,W]")
        if not torch.isfinite(area_weights).all() or (area_weights < 0).any():
            raise ValueError("area_weights must be finite and non-negative")
        if mask is None:
            mask = torch.ones_like(area_weights)
        mask = torch.as_tensor(mask, dtype=forecast.dtype, device=forecast.device)
        if mask.shape != (height, width) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("mask must be a binary [H,W] tensor")
        effective_weights = area_weights * mask
        denominator = effective_weights.sum()
        if not torch.isfinite(denominator) or float(denominator) <= self.eps:
            raise ValueError("area_weights and mask must select positive area")

        output = forecast.clone()
        selected = forecast[:, self.constrained_channels]
        current = (selected * effective_weights[None, None]).sum(dim=(-2, -1))
        correction = (current - target_integrals.to(forecast)) / denominator
        output[:, self.constrained_channels] = selected - correction[:, :, None, None] * mask
        return output


class ResidualGridBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class GraphCastConservationModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        processor_layers: int = 2,
        constrained_channels=(0,),
    ):
        super().__init__()
        if min(in_channels, out_channels, hidden_channels, processor_layers) <= 0:
            raise ValueError("channel counts and processor_layers must be positive")
        self.in_channels = int(in_channels)
        self.encoder = nn.Conv2d(in_channels, hidden_channels, 1)
        self.processor = nn.ModuleList(
            [ResidualGridBlock(hidden_channels) for _ in range(processor_layers)]
        )
        self.decoder = nn.Conv2d(hidden_channels, out_channels, 1)
        self.projection = GraphCastConservationProjection(constrained_channels)

    def forward_raw(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("x must have shape [B,in_channels,H,W]")
        hidden = self.encoder(x)
        for block in self.processor:
            hidden = block(hidden)
        return self.decoder(hidden)

    def forward(self, x, target_integrals, area_weights, mask=None):
        raw = self.forward_raw(x)
        return self.projection(raw, target_integrals, area_weights, mask)
