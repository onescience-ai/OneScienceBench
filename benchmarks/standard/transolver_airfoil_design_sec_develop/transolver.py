"""Transolver for AirfRANS with a shared backbone and multi-task output heads.

The Transolver backbone is unchanged; only the output projection is split into
velocity, pressure, and viscosity heads while preserving the final 4 channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_


ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


class PhysicsAttention(nn.Module):
    """Physics-Attention on irregular mesh points."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32,
                 dropout: float = 0.0, slice_num: int = 32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        fx_mid = self.in_project_fx(x).reshape(
            b, n, self.heads, self.dim_head
        ).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(
            b, n, self.heads, self.dim_head
        ).permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(dim=2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm[..., None] + 1e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        attn = self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v)

        out = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int,
                 n_layers: int = 1, act: str = "gelu", residual: bool = True):
        super().__init__()
        act_cls = ACTIVATIONS[act.lower()]
        self.residual = residual
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_cls())
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act_cls())
            for _ in range(n_layers)
        )
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for layer in self.layers:
            y = layer(x)
            x = x + y if self.residual else y
        return self.linear_post(x)


class TransolverBlock(nn.Module):
    """Backbone block only; output projection is intentionally outside."""

    def __init__(self, hidden_dim: int, num_heads: int, slice_num: int,
                 mlp_ratio: int = 2, dropout: float = 0.0,
                 act: str = "gelu"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            act=act,
            residual=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transolver(nn.Module):
    """Transolver with a shared backbone and three task-specific output heads."""

    def __init__(self, space_dim: int = 7, out_dim: int = 4,
                 hidden_dim: int = 256, layers: int = 8,
                 heads: int = 8, slice_num: int = 32,
                 mlp_ratio: int = 2, dropout: float = 0.0,
                 act: str = "gelu", ref: int = 8,
                 unified_pos: bool = True):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        input_dim = space_dim + (ref * ref if unified_pos else 0)
        self.preprocess = MLP(
            input_dim, hidden_dim * 2, hidden_dim,
            n_layers=0, act=act, residual=False,
        )
        self.blocks = nn.ModuleList([
            TransolverBlock(
                hidden_dim=hidden_dim,
                num_heads=heads,
                slice_num=slice_num,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                act=act,
            )
            for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        if out_dim != 4:
            raise ValueError("Multi-task AirfRANS head requires out_dim=4: [ux, uy, p, nut].")

        def make_head(task_out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, task_out_dim),
            )

        # Task-specific heads. The shared Transolver backbone is unchanged.
        self.velocity_head = make_head(2)
        self.pressure_head = make_head(1)
        self.viscosity_head = make_head(1)

        self.placeholder = nn.Parameter(torch.rand(hidden_dim) / hidden_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def _reference_distance(self, pos: torch.Tensor) -> torch.Tensor:
        b = pos.shape[0]
        grid_x = torch.linspace(-2.0, 4.0, self.ref, device=pos.device, dtype=pos.dtype)
        grid_y = torch.linspace(-1.5, 1.5, self.ref, device=pos.device, dtype=pos.dtype)
        gx, gy = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).reshape(1, self.ref * self.ref, 2)
        grid = grid.repeat(b, 1, 1)
        return torch.sqrt(((pos[:, :, None, :] - grid[:, None, :, :]) ** 2).sum(dim=-1))

    def forward_features(self, x: torch.Tensor, pos: torch.Tensor):
        """Return shared Transolver backbone features with shape [B, N, hidden_dim]."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if self.unified_pos:
            x = torch.cat([x, self._reference_distance(pos)], dim=-1)

        x = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x, squeeze

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        features, squeeze = self.forward_features(x, pos)

        velocity = self.velocity_head(features)      # [..., 2] -> ux, uy
        pressure = self.pressure_head(features)      # [..., 1] -> p
        viscosity = self.viscosity_head(features)    # [..., 1] -> nut
        out = torch.cat([velocity, pressure, viscosity], dim=-1)

        return out[0] if squeeze else out
