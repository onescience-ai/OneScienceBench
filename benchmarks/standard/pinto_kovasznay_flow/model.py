from __future__ import annotations

import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * torch.sigmoid(inputs)


def _activation(name: str) -> nn.Module:
    if name.lower() == "swish":
        return Swish()
    if name.lower() == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, activation: str = "swish") -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        _activation(activation),
        nn.Linear(hidden_dim, output_dim),
        _activation(activation),
    )


class PINTOModel(nn.Module):
    """PyTorch reference for the PINTO Kovasznay flow architecture."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 2,
        activation: str = "swish",
        domain_start: tuple[float, float] = (-0.5, -0.5),
        domain_stop: tuple[float, float] = (1.0, 1.5),
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.register_buffer("domain_start", torch.tensor(domain_start, dtype=torch.float32))
        self.register_buffer("domain_stop", torch.tensor(domain_stop, dtype=torch.float32))

        self.query_lifting = make_mlp(2, self.hidden_dim, self.hidden_dim, activation)
        self.boundary_position_encoder = make_mlp(2, self.hidden_dim, self.hidden_dim, activation)
        self.boundary_value_encoder = make_mlp(3, self.hidden_dim, self.hidden_dim, activation)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.post_attention = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), _activation(activation))
        self.u_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, 1),
        )
        self.v_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, 1),
        )
        self.p_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, 1),
        )

    def normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        start = self.domain_start.to(device=xy.device, dtype=xy.dtype)
        stop = self.domain_stop.to(device=xy.device, dtype=xy.dtype)
        return 2.0 * (xy - start) / (stop - start) - 1.0

    def forward(
        self,
        query_xy: torch.Tensor,
        boundary_xy: torch.Tensor,
        boundary_values: torch.Tensor,
    ) -> torch.Tensor:
        if query_xy.ndim != 3 or query_xy.shape[-1] != 2:
            raise ValueError("query_xy must have shape [B, N, 2].")
        if boundary_xy.ndim != 3 or boundary_xy.shape[-1] != 2:
            raise ValueError("boundary_xy must have shape [B, S, 2].")
        if boundary_values.ndim != 3 or boundary_values.shape[-1] != 3:
            raise ValueError("boundary_values must have shape [B, S, 3] ordered as [u, v, p].")

        query = self.query_lifting(self.normalize_xy(query_xy))
        key = self.boundary_position_encoder(self.normalize_xy(boundary_xy))
        value = self.boundary_value_encoder(boundary_values)
        attended, _ = self.cross_attention(query=query, key=key, value=value, need_weights=True)
        features = self.post_attention(query + attended)
        return torch.cat([self.u_head(features), self.v_head(features), self.p_head(features)], dim=-1)
