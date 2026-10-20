from __future__ import annotations

import torch
from torch import nn


class IrregularTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, num_frequencies: int = 4):
        super().__init__()
        if embed_dim <= 0 or num_frequencies <= 0:
            raise ValueError("embed_dim and num_frequencies must be positive")
        self.register_buffer(
            "frequency_bands", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        )
        feature_dim = 1 + 2 * num_frequencies
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, time_deltas):
        if not isinstance(time_deltas, torch.Tensor) or time_deltas.ndim != 2:
            raise ValueError("time_deltas must have shape [B,T]")
        if not torch.isfinite(time_deltas).all() or (time_deltas < 0).any():
            raise ValueError("time_deltas must be finite and non-negative")
        angles = time_deltas[..., None] * self.frequency_bands.to(time_deltas)
        features = torch.cat(
            [time_deltas[..., None], torch.sin(angles), torch.cos(angles)], dim=-1
        )
        return self.projection(features)


class EarthformerIrregularTimeModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 16,
        num_heads: int = 4,
        num_frequencies: int = 4,
    ):
        super().__init__()
        if min(in_channels, out_channels, hidden_dim, num_heads) <= 0:
            raise ValueError("channel counts, hidden_dim and num_heads must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.in_channels = int(in_channels)
        self.spatial_encoder = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.time_embedding = IrregularTimeEmbedding(hidden_dim, num_frequencies)
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x, time_deltas, time_mask=None):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("x must have shape [B,T,C,H,W]")
        batch, steps, channels, height, width = x.shape
        if channels != self.in_channels or time_deltas.shape != (batch, steps):
            raise ValueError("input channels or time_deltas shape is invalid")
        if time_mask is None:
            time_mask = torch.ones(batch, steps, dtype=torch.bool, device=x.device)
        if time_mask.shape != (batch, steps) or time_mask.dtype != torch.bool:
            raise ValueError("time_mask must be a boolean [B,T] tensor")
        if not bool(time_mask.any(dim=1).all()):
            raise ValueError("each sample must contain at least one valid time step")

        encoded = self.spatial_encoder(x.reshape(batch * steps, channels, height, width))
        encoded = encoded.reshape(batch, steps, -1, height, width)
        tokens = encoded.mean(dim=(-2, -1)) + self.time_embedding(time_deltas.to(x))
        attended, _ = self.temporal_attention(
            tokens, tokens, tokens, key_padding_mask=~time_mask, need_weights=False
        )
        context = self.norm(tokens + attended)
        fused = encoded + context[..., None, None]
        output = self.decoder(fused.reshape(batch * steps, -1, height, width))
        output = output.reshape(batch, steps, -1, height, width)
        return output * time_mask[:, :, None, None, None]
