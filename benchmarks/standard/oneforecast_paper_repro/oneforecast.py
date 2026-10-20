"""OneForecast-style variable-aware spatiotemporal transformer reference."""
from __future__ import annotations

import torch
from torch import nn


class OneForecastBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class OneForecast(nn.Module):
    def __init__(self, variables: int = 4, history: int = 2, embed_dim: int = 32, depth: int = 2, heads: int = 4):
        super().__init__()
        if variables < 1 or history < 1 or embed_dim % heads:
            raise ValueError("invalid variables/history or embed_dim/heads")
        self.variables, self.history, self.embed_dim = variables, history, embed_dim
        self.input_projection = nn.Conv2d(history, embed_dim, 1)
        self.variable_embeddings = nn.Parameter(torch.zeros(1, variables, embed_dim))
        self.blocks = nn.ModuleList([OneForecastBlock(embed_dim, heads) for _ in range(depth)])
        self.output_projection = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.history or x.shape[2] != self.variables:
            raise ValueError("OneForecast expects (B,history,variables,H,W)")
        b, _, v, h, w = x.shape
        per_variable = x.permute(0, 2, 1, 3, 4).reshape(b * v, self.history, h, w)
        y = self.input_projection(per_variable)
        tokens = y.flatten(2).transpose(1, 2).reshape(b, v, h * w, self.embed_dim)
        tokens = tokens + self.variable_embeddings.unsqueeze(2)
        tokens = tokens.reshape(b, v * h * w, self.embed_dim)
        for block in self.blocks:
            tokens = block(tokens)
        y = self.output_projection(tokens).reshape(b, v, h, w)
        return y

    @torch.no_grad()
    def rollout(self, history: torch.Tensor, steps: int) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        frames = []
        state = history.clone()
        for _ in range(steps):
            pred = self(state)
            frames.append(pred)
            state = torch.cat([state[:, 1:], pred.unsqueeze(1)], dim=1)
        return torch.stack(frames, dim=1)
