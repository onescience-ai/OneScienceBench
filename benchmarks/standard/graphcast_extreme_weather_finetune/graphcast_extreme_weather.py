from __future__ import annotations

import torch
from torch import nn


class GraphCastExtremeWeather(nn.Module):
    def __init__(self, input_channels=6, hidden_channels=16, output_channels=1):
        super().__init__()
        self.grid_encoder = nn.Sequential(nn.Conv2d(input_channels, hidden_channels, 3, padding=1), nn.GELU())
        self.processor = nn.ModuleList([nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1) for _ in range(3)])
        self.extreme_head = nn.Conv2d(hidden_channels, output_channels, 1)

    def forward(self, inputs, event_mask=None):
        x = self.grid_encoder(inputs)
        for layer in self.processor:
            x = x + torch.tanh(layer(x))
        output = self.extreme_head(x)
        return output if event_mask is None else output * event_mask
