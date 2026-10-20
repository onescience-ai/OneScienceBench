from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiscaleFusionBlock(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
        )

    def forward(self, coarse, skip):
        coarse = F.interpolate(
            coarse, size=skip.shape[-3:], mode="trilinear", align_corners=False
        )
        return self.fuse(torch.cat([coarse, skip], dim=1))


class EarthformerMultiscaleDecoder(nn.Module):
    """Top-down 3D decoder for shallow, middle and deep Earthformer features."""

    def __init__(self, encoder_channels=(8, 16, 32), hidden_channels: int = 16, out_channels: int = 2):
        super().__init__()
        encoder_channels = tuple(int(value) for value in encoder_channels)
        if len(encoder_channels) != 3 or min(*encoder_channels, hidden_channels, out_channels) <= 0:
            raise ValueError("three positive encoder channels and positive output sizes are required")
        self.encoder_channels = encoder_channels
        self.lateral_projections = nn.ModuleList(
            [nn.Conv3d(channels, hidden_channels, 1) for channels in encoder_channels]
        )
        self.fusion_blocks = nn.ModuleList(
            [MultiscaleFusionBlock(hidden_channels), MultiscaleFusionBlock(hidden_channels)]
        )
        self.output_head = nn.Conv3d(hidden_channels, out_channels, 1)

    def forward(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 3:
            raise ValueError("features must contain shallow, middle and deep tensors")
        shallow, middle, deep = features
        for index, (feature, channels) in enumerate(zip(features, self.encoder_channels)):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 5:
                raise ValueError("each feature must have shape [B,C,T,H,W]")
            if feature.shape[1] != channels:
                raise ValueError(f"feature {index} has the wrong channel count")
            if feature.shape[:1] + feature.shape[2:3] != shallow.shape[:1] + shallow.shape[2:3]:
                raise ValueError("all features must share batch and time dimensions")
        if shallow.shape[-2] != 2 * middle.shape[-2] or shallow.shape[-1] != 2 * middle.shape[-1]:
            raise ValueError("middle feature must have half the shallow spatial resolution")
        if middle.shape[-2] != 2 * deep.shape[-2] or middle.shape[-1] != 2 * deep.shape[-1]:
            raise ValueError("deep feature must have half the middle spatial resolution")

        projected = [layer(feature) for layer, feature in zip(self.lateral_projections, features)]
        decoded = self.fusion_blocks[0](projected[2], projected[1])
        decoded = self.fusion_blocks[1](decoded, projected[0])
        output = self.output_head(decoded)
        return output.permute(0, 2, 1, 3, 4)
