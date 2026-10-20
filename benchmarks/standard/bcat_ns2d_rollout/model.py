from __future__ import annotations

import torch
from torch import nn


def block_lower_triangular_mask(block_size: int, block_num: int, use_float: bool = True) -> torch.Tensor:
    matrix_size = int(block_size) * int(block_num)
    lower = torch.tril(torch.ones(matrix_size, matrix_size, dtype=torch.bool))
    blocks = torch.block_diag(*[torch.ones(int(block_size), int(block_size), dtype=torch.bool) for _ in range(int(block_num))])
    allowed = torch.logical_or(lower, blocks)
    blocked = ~allowed
    if use_float:
        return torch.zeros(matrix_size, matrix_size, dtype=torch.float32).masked_fill(blocked, float("-inf"))
    return blocked


class PatchEmbedder(nn.Module):
    def __init__(self, grid_size: int, channels: int, patch_num: int, dim_emb: int, max_time_len: int = 20) -> None:
        super().__init__()
        if grid_size % patch_num != 0:
            raise ValueError("grid_size must be divisible by patch_num.")
        self.grid_size = int(grid_size)
        self.channels = int(channels)
        self.patch_num = int(patch_num)
        self.patch_resolution = self.grid_size // self.patch_num
        self.patch_dim = self.channels * self.patch_resolution * self.patch_resolution
        self.unfold = nn.Unfold(kernel_size=self.patch_resolution, stride=self.patch_resolution)
        self.fold = nn.Fold(output_size=self.grid_size, kernel_size=self.patch_resolution, stride=self.patch_resolution)
        self.in_proj = nn.Linear(self.patch_dim, int(dim_emb))
        self.time_proj = nn.Sequential(nn.Linear(1, int(dim_emb)), nn.GELU(), nn.Linear(int(dim_emb), int(dim_emb)))
        self.patch_position_embeddings = nn.Parameter(torch.zeros(1, self.patch_num * self.patch_num, int(dim_emb)))
        self.output_proj = nn.Linear(int(dim_emb), self.patch_dim)
        nn.init.trunc_normal_(self.patch_position_embeddings, std=0.02)

    def encode(self, data: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if data.ndim != 5:
            raise ValueError("data must have shape [B, T, H, W, C].")
        batch, steps, height, width, channels = data.shape
        if height != self.grid_size or width != self.grid_size or channels != self.channels:
            raise ValueError("data shape does not match the configured grid and channels.")
        patches = data.permute(0, 1, 4, 2, 3).reshape(batch * steps, channels, height, width)
        patches = self.unfold(patches).transpose(1, 2).reshape(batch, steps * self.patch_num * self.patch_num, self.patch_dim)
        tokens = self.in_proj(patches)
        time_tokens = self.time_proj(times).repeat_interleave(self.patch_num * self.patch_num, dim=1)
        patch_tokens = self.patch_position_embeddings.repeat(1, steps, 1)
        return tokens + time_tokens + patch_tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_len, _ = tokens.shape
        patches_per_step = self.patch_num * self.patch_num
        if token_len % patches_per_step != 0:
            raise ValueError("token length must be divisible by patch_num squared.")
        steps = token_len // patches_per_step
        patches = self.output_proj(tokens).reshape(batch * steps, patches_per_step, self.patch_dim).transpose(1, 2)
        fields = self.fold(patches)
        fields = fields.reshape(batch, steps, self.channels, self.grid_size, self.grid_size)
        return fields.permute(0, 1, 3, 4, 2).contiguous()


class BCATLite(nn.Module):
    """Small BCAT-style block-causal autoregressive model for 2D fluid rollouts."""

    def __init__(
        self,
        grid_size: int = 16,
        channels: int = 3,
        patch_num: int = 4,
        dim_emb: int = 32,
        n_head: int = 4,
        n_layer: int = 2,
        dim_ffn: int = 64,
        dropout: float = 0.0,
        max_time_len: int = 20,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.channels = int(channels)
        self.patch_num = int(patch_num)
        self.dim_emb = int(dim_emb)
        self.n_head = int(n_head)
        self.n_layer = int(n_layer)
        self.embedder = PatchEmbedder(self.grid_size, self.channels, self.patch_num, self.dim_emb, max_time_len=max_time_len)
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim_emb,
            nhead=self.n_head,
            dim_feedforward=int(dim_ffn),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=self.n_layer, norm=nn.LayerNorm(self.dim_emb))
        self.seq_len_per_step = self.patch_num * self.patch_num
        self.register_buffer("mask", block_lower_triangular_mask(self.seq_len_per_step, max_time_len, use_float=True), persistent=False)

    def forward(self, data: torch.Tensor, times: torch.Tensor, input_len: int = 1) -> torch.Tensor:
        if data.ndim != 5:
            raise ValueError("data must have shape [B, T, H, W, C].")
        if times.ndim != 3:
            raise ValueError("times must have shape [B, T, 1].")
        data_in = data[:, :-1]
        time_in = times[:, :-1]
        tokens = self.embedder.encode(data_in, time_in)
        seq_len = tokens.shape[1]
        attn_mask = self.mask[:seq_len, :seq_len].to(device=tokens.device, dtype=tokens.dtype)
        encoded = self.transformer_encoder(tokens, mask=attn_mask)
        input_tokens = (int(input_len) - 1) * self.seq_len_per_step
        return self.embedder.decode(encoded[:, input_tokens:])

    @torch.no_grad()
    def generate(self, data_input: torch.Tensor, times: torch.Tensor, input_len: int = 1) -> torch.Tensor:
        return self.forward(data_input, times, input_len=input_len)
