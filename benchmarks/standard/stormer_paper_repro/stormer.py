"""Small, faithful Stormer-style weather transformer reference.

The paper's global model is represented by patchifying a short history, adding
learned spatial tokens, mixing patches with pre-norm self attention, and
recovering the next weather field.  Defaults are deliberately small so the
reference can be used by a dynamic evaluator.
"""
from __future__ import annotations

import torch
from torch import nn


class StormerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class Stormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        img_size: tuple[int, int] = (16, 32),
        patch_size: int = 4,
        embed_dim: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        history: int = 2,
    ):
        super().__init__()
        if history < 1 or patch_size < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("history, patch_size, and num_heads must be valid")
        if len(img_size) != 2 or any(int(n) < 1 for n in img_size):
            raise ValueError("img_size must contain two positive dimensions")
        if any(int(n) % patch_size for n in img_size):
            raise ValueError("img_size dimensions must be divisible by patch_size")
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.img_size, self.patch_size, self.history = tuple(img_size), int(patch_size), int(history)
        self.grid_size = tuple(n // patch_size for n in img_size)
        self.patch_embed = nn.Conv2d(in_channels * history, embed_dim, patch_size, patch_size)
        self.position = nn.Parameter(torch.zeros(1, self.grid_size[0] * self.grid_size[1], embed_dim))
        self.blocks = nn.ModuleList([StormerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_head = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != self.in_channels * self.history:
                raise ValueError("4D input channel count must equal history*in_channels")
            x = x.reshape(x.shape[0], self.history, self.in_channels, x.shape[-2], x.shape[-1])
        if x.ndim != 5 or x.shape[1:] != (self.history, self.in_channels, *x.shape[-2:]):
            raise ValueError("Stormer expects (B,history,in_channels,H,W)")
        b, _, _, h, w = x.shape
        if (h, w) != self.img_size or h % self.patch_size or w % self.patch_size:
            raise ValueError("input spatial shape must equal img_size and be patch divisible")
        tokens = self.patch_embed(x.reshape(b, -1, h, w)).flatten(2).transpose(1, 2)
        tokens = tokens + self.position
        for block in self.blocks:
            tokens = block(tokens)
        patches = self.patch_head(self.norm(tokens))
        gh, gw = self.grid_size
        patches = patches.view(b, gh, gw, self.out_channels, self.patch_size, self.patch_size)
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(b, self.out_channels, h, w)
