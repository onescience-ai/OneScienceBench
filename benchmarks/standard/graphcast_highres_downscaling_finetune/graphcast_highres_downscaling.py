from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GraphCastHighResDownscaler(nn.Module):
    def __init__(self, input_channels=5, hidden_channels=16, output_channels=2, upscale_factor=2):
        super().__init__()
        self.grid_encoder = nn.Sequential(nn.Conv2d(input_channels, hidden_channels, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1))
        self.mesh_processor = nn.ModuleList([nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)])
        self.grid_decoder = nn.Conv2d(hidden_channels, output_channels, 1)
        self.upscale_factor = int(upscale_factor)

    def forward(self, grid_features):
        if grid_features.ndim != 4:
            raise ValueError("expected grid features (B,C,H,W)")
        x = self.grid_encoder(grid_features)
        for block in self.mesh_processor:
            x = x + F.gelu(block(x))
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False)
        return self.grid_decoder(x)
