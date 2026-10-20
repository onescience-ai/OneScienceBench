from __future__ import annotations

import torch
from torch import nn


class NowcastNetMultiHorizon(nn.Module):
    def __init__(self, input_length=4, horizons=(1, 3, 6), hidden_channels=16):
        super().__init__()
        self.input_length = int(input_length)
        self.horizons = tuple(int(item) for item in horizons)
        self.encoder = nn.Sequential(nn.Conv2d(input_length, hidden_channels, 3, padding=1), nn.ReLU())
        self.horizon_heads = nn.ModuleList([nn.Conv2d(hidden_channels, 1, 1) for _ in self.horizons])

    def forward(self, radar_sequence):
        if radar_sequence.ndim == 5:
            radar_sequence = radar_sequence[..., 0]
        if radar_sequence.ndim != 4 or radar_sequence.shape[1] != self.input_length:
            raise ValueError("expected radar sequence (B,T,H,W)")
        features = self.encoder(radar_sequence)
        return torch.cat([head(features) for head in self.horizon_heads], dim=1)
