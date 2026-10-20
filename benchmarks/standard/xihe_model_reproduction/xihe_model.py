from __future__ import annotations

from math import sqrt
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError("expected an int or a two-element sequence")
    return (int(value[0]), int(value[1]))


def _head_count(dim: int, maximum: int = 4) -> int:
    for heads in range(min(maximum, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class _FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float = 2.0):
        super().__init__()
        hidden_dim = max(dim, int(dim * ratio))
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.layers(x)


class _LocalSIE(nn.Module):
    """Window self-attention restricted to ocean tokens."""

    def __init__(self, dim: int, window_size=(4, 4)):
        super().__init__()
        self.dim = dim
        self.window_size = _pair(window_size)
        self.num_heads = _head_count(dim)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _FeedForward(dim)

    def forward(self, x, ocean_mask):
        batch, height, width, channels = x.shape
        win_h, win_w = self.window_size
        pad_h = (-height) % win_h
        pad_w = (-width) % win_w

        residual = x
        normalized = self.norm1(x)
        normalized = F.pad(
            normalized.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)
        ).permute(0, 2, 3, 1)
        padded_mask = F.pad(
            ocean_mask[:, None].float(), (0, pad_w, 0, pad_h)
        ).squeeze(1).bool()
        padded_h, padded_w = normalized.shape[1:3]

        windows = (
            normalized.view(
                batch,
                padded_h // win_h,
                win_h,
                padded_w // win_w,
                win_w,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, win_h * win_w, channels)
        )
        mask_windows = (
            padded_mask.view(
                batch,
                padded_h // win_h,
                win_h,
                padded_w // win_w,
                win_w,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, win_h * win_w)
        )

        qkv = self.qkv(windows).reshape(
            windows.shape[0], windows.shape[1], 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask_windows[:, None, None, :], -1.0e4)
        attention = scores.softmax(dim=-1)
        mixed = (attention @ v).transpose(1, 2).reshape_as(windows)
        mixed = self.proj(mixed) * mask_windows[..., None]

        mixed = (
            mixed.view(
                batch,
                padded_h // win_h,
                padded_w // win_w,
                win_h,
                win_w,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, padded_h, padded_w, channels)
        )
        mixed = mixed[:, :height, :width]
        mask_values = ocean_mask[..., None].to(x.dtype)
        x = (residual + mixed) * mask_values
        x = (x + self.mlp(self.norm2(x))) * mask_values
        return x


class _GroupMixer(nn.Module):
    def __init__(self, num_groups: int, dim: int):
        super().__init__()
        token_hidden = max(num_groups, num_groups * 2)
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_groups, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, num_groups),
        )
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = _FeedForward(dim)

    def forward(self, groups):
        groups = groups + self.token_mlp(
            self.token_norm(groups).transpose(1, 2)
        ).transpose(1, 2)
        return groups + self.channel_mlp(self.channel_norm(groups))


class _GlobalSIE(nn.Module):
    """Group-vector aggregation, global propagation and feature ungrouping."""

    def __init__(self, dim: int, num_groups: int):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.group_vectors = nn.Parameter(torch.empty(num_groups, dim))
        nn.init.trunc_normal_(self.group_vectors, std=0.02)
        self.token_norm = nn.LayerNorm(dim)
        self.group_norm = nn.LayerNorm(dim)
        self.group_mixer = _GroupMixer(num_groups, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_mlp = _FeedForward(dim)

    def forward(self, x, ocean_mask):
        batch, height, width, channels = x.shape
        tokens = self.token_norm(x.reshape(batch, height * width, channels))
        valid = ocean_mask.reshape(batch, height * width)
        groups = self.group_vectors.unsqueeze(0).expand(batch, -1, -1)

        grouping_scores = torch.einsum("bgc,bnc->bgn", groups, tokens) / sqrt(channels)
        grouping_scores = grouping_scores.masked_fill(~valid[:, None, :], -1.0e4)
        grouping_weights = grouping_scores.softmax(dim=-1)
        grouped_features = torch.einsum("bgn,bnc->bgc", grouping_weights, tokens)
        groups = self.group_mixer(groups + grouped_features)

        ungrouping_scores = torch.einsum("bnc,bgc->bng", tokens, self.group_norm(groups))
        ungrouping_weights = ungrouping_scores.mul(channels ** -0.5).softmax(dim=-1)
        global_features = torch.einsum("bng,bgc->bnc", ungrouping_weights, groups)
        global_features = self.output_proj(global_features).reshape(
            batch, height, width, channels
        )

        mask_values = ocean_mask[..., None].to(x.dtype)
        x = (x + global_features) * mask_values
        x = (x + self.output_mlp(self.output_norm(x))) * mask_values
        return x


class _OceanSpecificBlock(nn.Module):
    def __init__(self, dim: int, num_local: int, num_groups: int, window_size):
        super().__init__()
        self.local_sies = nn.ModuleList(
            [_LocalSIE(dim, window_size=window_size) for _ in range(num_local)]
        )
        self.global_sie = _GlobalSIE(dim, num_groups=num_groups)

    def forward(self, x, ocean_mask):
        for local_sie in self.local_sies:
            x = local_sie(x, ocean_mask)
        return self.global_sie(x, ocean_mask)


class _PatchMerging(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim * 4)
        self.reduction = nn.Linear(in_dim * 4, in_dim * 2)

    def forward(self, x, ocean_mask):
        batch, height, width, channels = x.shape
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(
                0, 2, 3, 1
            )
            ocean_mask = F.pad(
                ocean_mask[:, None].float(), (0, pad_w, 0, pad_h)
            ).squeeze(1).bool()

        merged = torch.cat(
            (x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]),
            dim=-1,
        )
        low_mask = F.max_pool2d(ocean_mask[:, None].float(), 2, 2).squeeze(1).bool()
        merged = self.reduction(self.norm(merged))
        return merged * low_mask[..., None].to(merged.dtype), low_mask


class _PatchExpansion(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.expand = nn.Linear(in_dim, out_dim * 4)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, output_size):
        batch, height, width, _ = x.shape
        x = self.expand(x).view(batch, height, width, 2, 2, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, height * 2, width * 2, self.out_dim
        )
        x = x[:, : output_size[0], : output_size[1]]
        return self.norm(x)


class XiheModel(nn.Module):
    """Standalone reproduction of XiHe's masked hierarchical Transformer."""

    local_depths = (1, 2, 2, 2, 1)

    def __init__(
        self,
        img_size=(2041, 4320),
        patch_size=(6, 12),
        in_chans=96,
        out_chans=94,
        embed_dim=192,
        mask=None,
        num_groups=32,
        window_size=(6, 12),
    ):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.embed_dim = int(embed_dim)
        self.num_groups = int(num_groups)

        if min(*self.img_size, *self.patch_size) <= 0:
            raise ValueError("img_size and patch_size values must be positive")
        if self.in_chans <= 0 or self.out_chans <= 0 or self.embed_dim <= 0:
            raise ValueError("channel counts and embed_dim must be positive")
        if self.num_groups <= 0:
            raise ValueError("num_groups must be positive")

        if mask is None:
            mask_tensor = torch.empty(0, dtype=torch.float32)
        else:
            mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
            if mask_tensor.ndim != 2 or tuple(mask_tensor.shape) != self.img_size:
                raise ValueError(f"mask must have shape {self.img_size}")
            mask_tensor = (mask_tensor > 0.5).float()
        self.register_buffer("mask", mask_tensor)

        self.patch_partition = nn.Conv2d(
            self.in_chans,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        high_groups = min(self.num_groups, 32)
        low_groups = min(self.num_groups, 32)
        self.ocean_blocks = nn.ModuleList(
            [
                _OceanSpecificBlock(
                    self.embed_dim,
                    num_local=1,
                    num_groups=high_groups,
                    window_size=window_size,
                ),
                _OceanSpecificBlock(
                    self.embed_dim * 2,
                    num_local=2,
                    num_groups=low_groups,
                    window_size=window_size,
                ),
                _OceanSpecificBlock(
                    self.embed_dim * 2,
                    num_local=2,
                    num_groups=low_groups,
                    window_size=window_size,
                ),
                _OceanSpecificBlock(
                    self.embed_dim * 2,
                    num_local=2,
                    num_groups=low_groups,
                    window_size=window_size,
                ),
                _OceanSpecificBlock(
                    self.embed_dim,
                    num_local=1,
                    num_groups=high_groups,
                    window_size=window_size,
                ),
            ]
        )
        self.downsample = _PatchMerging(self.embed_dim)
        self.upsample = _PatchExpansion(self.embed_dim * 2, self.embed_dim)
        self.skip_projection = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.patch_restoration = nn.ConvTranspose2d(
            self.embed_dim,
            self.out_chans,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def _full_mask(self, x):
        if self.mask.numel() == 0:
            return torch.ones(
                self.img_size, dtype=torch.bool, device=x.device
            )
        return self.mask.to(device=x.device).bool()

    def _partition(self, x):
        height, width = self.img_size
        patch_h, patch_w = self.patch_size
        pad_h = (-height) % patch_h
        pad_w = (-width) % patch_w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return self.patch_partition(x)

    def _token_mask(self, full_mask, grid_size):
        patch_h, patch_w = self.patch_size
        pad_h = grid_size[0] * patch_h - full_mask.shape[0]
        pad_w = grid_size[1] * patch_w - full_mask.shape[1]
        mask = F.pad(full_mask[None, None].float(), (0, pad_w, 0, pad_h))
        return F.max_pool2d(
            mask, kernel_size=self.patch_size, stride=self.patch_size
        ).squeeze(0).squeeze(0).bool()

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("x must be a Tensor with shape (B, in_chans, H, W)")
        batch, channels, height, width = x.shape
        if channels != self.in_chans:
            raise ValueError(f"expected {self.in_chans} input channels, got {channels}")
        if (height, width) != self.img_size:
            raise ValueError(
                f"input spatial size {(height, width)} does not match {self.img_size}"
            )

        features = self._partition(x).permute(0, 2, 3, 1)
        high_size = features.shape[1:3]
        token_mask = self._token_mask(self._full_mask(x), high_size)
        token_mask = token_mask.unsqueeze(0).expand(batch, -1, -1)
        features = features * token_mask[..., None].to(features.dtype)

        skip = self.ocean_blocks[0](features, token_mask)
        features, low_mask = self.downsample(skip, token_mask)
        for block in self.ocean_blocks[1:4]:
            features = block(features, low_mask)
        features = self.upsample(features, high_size)
        features = self.ocean_blocks[4](features, token_mask)
        features = self.skip_projection(torch.cat((features, skip), dim=-1))
        features = features * token_mask[..., None].to(features.dtype)

        output = self.patch_restoration(features.permute(0, 3, 1, 2))
        return output[:, :, :height, :width]
