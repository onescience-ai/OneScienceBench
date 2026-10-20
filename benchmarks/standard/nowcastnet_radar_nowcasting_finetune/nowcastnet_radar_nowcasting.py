from __future__ import annotations

import torch
from torch import nn


class RadarNowcastNet(nn.Module):
    def __init__(self, input_length=4, pred_length=3, hidden_channels=16):
        super().__init__()
        self.input_length, self.pred_length = int(input_length), int(pred_length)
        self.evolution = nn.Sequential(nn.Conv2d(input_length, hidden_channels, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden_channels, pred_length, 1))
        self.generator = nn.Sequential(nn.Conv2d(input_length + pred_length, hidden_channels, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden_channels, pred_length, 1))

    def forward(self, radar_sequence):
        if radar_sequence.ndim == 5:
            radar_sequence = radar_sequence[..., 0]
        if radar_sequence.ndim != 4 or radar_sequence.shape[1] != self.input_length:
            raise ValueError("expected radar sequence (B,T,H,W)")
        evolution = self.evolution(radar_sequence)
        return evolution + self.generator(torch.cat([radar_sequence, evolution], dim=1))
