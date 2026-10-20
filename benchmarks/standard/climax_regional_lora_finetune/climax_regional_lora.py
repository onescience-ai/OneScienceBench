from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ClimaXRegionalLoRA(nn.Module):
    """Small, auditable ClimaX-style regional adapter."""

    def __init__(self, input_channels=4, hidden_channels=16, output_channels=2, rank=4):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(input_channels, hidden_channels, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1))
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Conv2d(input_channels, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, hidden_channels, 1, bias=False)
        self.head = nn.Conv2d(hidden_channels, output_channels, 1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("expected (B,C,H,W)")
        features = self.backbone(x) + self.lora_B(self.lora_A(x))
        return self.head(F.gelu(features))
