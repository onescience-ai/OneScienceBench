from __future__ import annotations

import torch
from torch import nn


class NowcastNetDomainAdapter(nn.Module):
    def __init__(self, input_length=4, pred_length=3, hidden_channels=16, adapter_rank=4):
        super().__init__()
        self.input_length, self.pred_length = int(input_length), int(pred_length)
        self.backbone = nn.Conv2d(input_length, hidden_channels, 3, padding=1)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.adapter_down = nn.Conv2d(input_length, adapter_rank, 1, bias=False)
        self.adapter_up = nn.Conv2d(adapter_rank, hidden_channels, 1, bias=False)
        self.head = nn.Conv2d(hidden_channels, pred_length, 1)

    def forward(self, radar_sequence):
        if radar_sequence.ndim == 5:
            radar_sequence = radar_sequence[..., 0]
        if radar_sequence.ndim != 4 or radar_sequence.shape[1] != self.input_length:
            raise ValueError("expected radar sequence (B,T,H,W)")
        features = torch.relu(self.backbone(radar_sequence) + self.adapter_up(self.adapter_down(radar_sequence)))
        return self.head(features)
