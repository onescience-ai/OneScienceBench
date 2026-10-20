"""Lightweight FengWu multi-branch encoder/fuser/decoder reference."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Fengwu(nn.Module):
    def __init__(self, pressure_level: int = 5, embed_dim: int = 16, img_size: tuple[int, int] = (32, 64), patch_size: int = 2):
        super().__init__()
        if pressure_level < 1 or embed_dim < 4 or patch_size < 1:
            raise ValueError("pressure_level, embed_dim, and patch_size must be valid")
        if len(img_size) != 2 or any(int(n) < 1 for n in img_size):
            raise ValueError("img_size must contain two positive dimensions")
        if any(int(n) % patch_size for n in img_size):
            raise ValueError("img_size dimensions must be divisible by patch_size")
        self.pressure_level, self.img_size, self.patch_size = pressure_level, tuple(img_size), patch_size
        channels = [4] + [pressure_level] * 5
        self.encoders = nn.ModuleList([nn.Conv2d(c, embed_dim, patch_size, patch_size) for c in channels])
        self.fuser = nn.Sequential(nn.Conv3d(embed_dim, embed_dim, 3, padding=1), nn.GELU(), nn.Conv3d(embed_dim, embed_dim, 1))
        self.decoders = nn.ModuleList([nn.Conv2d(embed_dim, c, 1) for c in channels])

    def forward(self, surface: torch.Tensor, z: torch.Tensor, r: torch.Tensor, u: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        inputs = (surface, z, r, u, v, t)
        b, h, w = surface.shape[0], surface.shape[-2], surface.shape[-1]
        if (h, w) != self.img_size or any(x.ndim != 4 or x.shape[0] != b or x.shape[-2:] != (h, w) for x in inputs):
            raise ValueError("all FengWu branches must share configured batch and grid")
        if surface.shape[1] != 4 or any(x.shape[1] != self.pressure_level for x in inputs[1:]):
            raise ValueError("FengWu branch channel contract violated")
        feats = torch.stack([enc(x) for enc, x in zip(self.encoders, inputs)], dim=2)
        fused = self.fuser(feats)
        outs = []
        for i, dec in enumerate(self.decoders):
            branch = fused[:, :, i]
            branch = F.interpolate(branch, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(dec(branch))
        return tuple(outs)
