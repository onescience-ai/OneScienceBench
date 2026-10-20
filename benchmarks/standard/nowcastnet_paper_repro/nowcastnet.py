"""Compact NowCastNet-style evolution plus generative refinement model."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EvolutionNetwork(nn.Module):
    def __init__(self, input_length: int, pred_length: int, base_channels: int = 16):
        super().__init__()
        self.pred_length = pred_length
        self.body = nn.Sequential(nn.Conv2d(input_length, base_channels, 3, padding=1), nn.ReLU(), nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU())
        self.intensity = nn.Conv2d(base_channels, pred_length, 1)
        self.motion = nn.Conv2d(base_channels, pred_length * 2, 1)

    def forward(self, frames: torch.Tensor):
        h = self.body(frames)
        return self.intensity(h), self.motion(h).view(frames.shape[0], self.pred_length, 2, *frames.shape[-2:])


class Net(nn.Module):
    def __init__(self, input_length: int = 4, total_length: int = 8, img_height: int = 32, img_width: int = 32, ngf: int = 16, device: str = "cpu", pretrained_model: str = ""):
        super().__init__()
        if total_length <= input_length or input_length < 1:
            raise ValueError("total_length must exceed input_length")
        self.input_length, self.total_length = input_length, total_length
        self.img_height, self.img_width, self.device_name = img_height, img_width, device
        self.pred_length = total_length - input_length
        self.evolution = EvolutionNetwork(input_length, self.pred_length, ngf)
        self.noise_projector = nn.Conv2d(1, ngf, 1)
        self.generator = nn.Sequential(nn.Conv2d(input_length + self.pred_length + ngf, ngf, 3, padding=1), nn.ReLU(), nn.Conv2d(ngf, self.pred_length, 3, padding=1))
        self.pretrained_model = pretrained_model

    @staticmethod
    def _warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, _, h, w = frame.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=frame.device), torch.linspace(-1, 1, w, device=frame.device), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1).clone()
        grid[..., 0] += flow[:, 0] / max(w / 2.0, 1.0)
        grid[..., 1] += flow[:, 1] / max(h / 2.0, 1.0)
        return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, all_frames: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if all_frames.ndim == 5:
            if all_frames.shape[1] != self.total_length or all_frames.shape[-1] != 1:
                raise ValueError("NowCastNet expects (B,total_length,H,W,C)")
            frames = all_frames[..., 0]
        elif all_frames.ndim == 4 and all_frames.shape[1] == self.total_length:
            frames = all_frames
        else:
            raise ValueError("NowCastNet expects a temporal frame sequence")
        if frames.shape[-2:] != (self.img_height, self.img_width):
            raise ValueError("input grid does not match configured image size")
        history = frames[:, : self.input_length]
        intensity, motion = self.evolution(history)
        last = history[:, -1:].contiguous()
        advected = []
        for i in range(self.pred_length):
            last = self._warp(last, motion[:, i]) + intensity[:, i:i + 1]
            advected.append(last[:, 0])
        evo = torch.stack(advected, dim=1) / 128.0
        if noise is None:
            noise = torch.zeros(frames.shape[0], 1, self.img_height, self.img_width, device=frames.device, dtype=frames.dtype)
        if noise.shape != (frames.shape[0], 1, self.img_height, self.img_width):
            raise ValueError("noise must have shape (B,1,H,W)")
        noise_features = self.noise_projector(noise)
        residual = self.generator(torch.cat([history, evo, noise_features], dim=1))
        return evo + residual
