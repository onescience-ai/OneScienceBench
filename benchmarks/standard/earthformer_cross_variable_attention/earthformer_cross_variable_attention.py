from __future__ import annotations

import torch
from torch import nn


class CrossVariableAttentionBlock(nn.Module):
    def __init__(self, num_variables: int, hidden_dim: int, num_heads: int):
        super().__init__()
        if min(num_variables, hidden_dim, num_heads) <= 0:
            raise ValueError("num_variables, hidden_dim and num_heads must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_variables = int(num_variables)
        self.value_projection = nn.Linear(1, hidden_dim)
        self.variable_embedding = nn.Embedding(num_variables, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 1)

    def forward(self, values, valid_mask):
        if values.ndim != 3 or values.shape[1:] != (self.num_variables, 1):
            raise ValueError("values must have shape [N,V,1]")
        if valid_mask.shape != values.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean [N,V] tensor")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("each token must contain at least one valid variable")
        variable_ids = torch.arange(self.num_variables, device=values.device)
        tokens = self.value_projection(values) + self.variable_embedding(variable_ids)[None]
        attended, _ = self.attention(
            tokens, tokens, tokens, key_padding_mask=~valid_mask, need_weights=False
        )
        output = self.output_projection(self.norm(tokens + attended))
        return output * valid_mask[..., None]


class EarthformerCrossVariableAttention(nn.Module):
    def __init__(self, num_variables: int, hidden_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_variables = int(num_variables)
        self.cross_variable_block = CrossVariableAttentionBlock(
            num_variables, hidden_dim, num_heads
        )

    def forward(self, x, variable_mask=None):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("x must have shape [B,T,V,H,W]")
        batch, steps, variables, height, width = x.shape
        if variables != self.num_variables:
            raise ValueError("input variable count does not match num_variables")
        if variable_mask is None:
            variable_mask = torch.ones(
                batch, variables, dtype=torch.bool, device=x.device
            )
        if variable_mask.shape == (variables,):
            variable_mask = variable_mask[None].expand(batch, variables)
        if variable_mask.shape != (batch, variables) or variable_mask.dtype != torch.bool:
            raise ValueError("variable_mask must be boolean [V] or [B,V]")
        values = x.permute(0, 1, 3, 4, 2).reshape(-1, variables, 1)
        expanded_mask = variable_mask[:, None, None, None, :].expand(
            batch, steps, height, width, variables
        ).reshape(-1, variables)
        output = self.cross_variable_block(values, expanded_mask)
        return output.reshape(batch, steps, height, width, variables).permute(0, 1, 4, 2, 3)
