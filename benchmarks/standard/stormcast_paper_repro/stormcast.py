"""Compact StormCast-style conditional diffusion denoiser reference."""
from __future__ import annotations

import torch
from torch import nn


class StormCast(nn.Module):
    def __init__(self, in_channels: int = 1, cond_channels: int = 4, base_channels: int = 16, num_steps: int = 1000):
        super().__init__()
        if num_steps < 2 or base_channels < 4:
            raise ValueError("num_steps must be >=2 and base_channels >=4")
        self.in_channels, self.cond_channels, self.num_steps = in_channels, cond_channels, num_steps
        self.time_embed = nn.Sequential(nn.Linear(1, base_channels), nn.SiLU(), nn.Linear(base_channels, base_channels))
        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.cond_proj = nn.Conv2d(cond_channels, base_channels, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.GroupNorm(4, base_channels), nn.SiLU(), nn.Conv2d(base_channels, base_channels, 3, padding=1)) for _ in range(2)])
        self.output_proj = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, noisy: torch.Tensor, conditioning: torch.Tensor, timestep: torch.Tensor | int) -> torch.Tensor:
        if noisy.ndim != 4 or conditioning.ndim != 4 or noisy.shape[0] != conditioning.shape[0] or noisy.shape[-2:] != conditioning.shape[-2:]:
            raise ValueError("noisy and conditioning must be (B,C,H,W) with matching batch/grid")
        if noisy.shape[1] != self.in_channels or conditioning.shape[1] != self.cond_channels:
            raise ValueError("channel count does not match StormCast configuration")
        t = torch.as_tensor(timestep, device=noisy.device, dtype=noisy.dtype).reshape(-1, 1)
        if t.numel() == 1:
            t = t.expand(noisy.shape[0], 1)
        if t.shape[0] != noisy.shape[0]:
            raise ValueError("timestep must be scalar or length B")
        h = self.input_proj(noisy) + self.cond_proj(conditioning)
        h = h + self.time_embed((t / float(self.num_steps)).clamp(0, 1)).unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)

    @torch.no_grad()
    def sample(self, conditioning: torch.Tensor, steps: int = 4, generator: torch.Generator | None = None) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        noise = torch.randn(conditioning.shape[0], self.in_channels, *conditioning.shape[-2:], device=conditioning.device, generator=generator)
        for t in torch.linspace(self.num_steps - 1, 0, steps, device=conditioning.device):
            noise = noise - self.forward(noise, conditioning, t) / float(max(steps, 1))
        return noise

