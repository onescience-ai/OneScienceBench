#!/usr/bin/env python
"""独立的轻量 scGPT 风格单细胞 embedding 推理入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ContinuousValueEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(1, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if bool((values < 0).any()):
            raise ValueError("expression values must be non-negative")
        return self.norm(self.projection(torch.log1p(values).unsqueeze(-1)))


class ScGPTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        attended, _weights = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        return hidden + self.ffn(self.ffn_norm(hidden))


class ScGPTEncoder(nn.Module):
    def __init__(
        self,
        num_genes: int = 128,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        embedding_size: int = 16,
    ) -> None:
        super().__init__()
        self.gene_encoder = nn.Embedding(num_genes, hidden_size, padding_idx=0)
        self.value_encoder = ContinuousValueEncoder(hidden_size)
        self.blocks = nn.ModuleList(
            [ScGPTBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.cell_projection = nn.Linear(hidden_size, embedding_size)

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gene_ids.ndim != 2 or values.shape != gene_ids.shape:
            raise ValueError("gene_ids and values must share shape [batch, genes]")
        if padding_mask is None:
            padding_mask = gene_ids.eq(0)
        hidden = self.gene_encoder(gene_ids) + self.value_encoder(values)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        hidden = self.final_norm(hidden)
        weights = (~padding_mask).to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return torch.nn.functional.normalize(self.cell_projection(pooled), dim=-1)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = (
        payload.get("model_state_dict", payload)
        if isinstance(payload, dict)
        else payload
    )
    model.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = np.load(args.input_npz)
    gene_ids = torch.as_tensor(payload["gene_ids"], dtype=torch.long)
    values = torch.as_tensor(payload["values"], dtype=torch.float32)
    padding_mask = gene_ids.eq(0)
    device = torch.device(args.device)
    model = ScGPTEncoder().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, gene_ids.shape[0], args.batch_size):
            stop = start + args.batch_size
            outputs.append(
                model(
                    gene_ids[start:stop].to(device),
                    values[start:stop].to(device),
                    padding_mask[start:stop].to(device),
                ).cpu()
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=torch.cat(outputs).numpy())


if __name__ == "__main__":
    main()
