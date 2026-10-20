#!/usr/bin/env python
"""独立的轻量 STATE 风格单细胞表达 embedding 推理入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SkipBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.up_projection = nn.Linear(hidden_size, hidden_size * 2)
        self.down_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(hidden)
        update = self.down_projection(
            torch.nn.functional.silu(self.up_projection(normalized))
        )
        return hidden + torch.sigmoid(self.gate(normalized)) * update


class StateEmbeddingModel(nn.Module):
    def __init__(
        self,
        num_genes: int = 64,
        hidden_size: int = 32,
        embedding_size: int = 16,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.gene_embedding = nn.Embedding(num_genes, hidden_size)
        self.value_projection = nn.Linear(1, hidden_size)
        self.blocks = nn.ModuleList([SkipBlock(hidden_size) for _ in range(num_layers)])
        self.pool_gate = nn.Linear(hidden_size, 1)
        self.embedding_projection = nn.Linear(hidden_size, embedding_size)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_genes),
        )

    def encode(self, expression: torch.Tensor) -> torch.Tensor:
        if (
            expression.ndim != 2
            or expression.shape[1] != self.gene_embedding.num_embeddings
        ):
            raise ValueError("expression must have shape [batch, num_genes]")
        if bool((expression < 0).any()):
            raise ValueError("expression must be non-negative")
        gene_ids = torch.arange(expression.shape[1], device=expression.device)
        hidden = self.gene_embedding(gene_ids)[None] + self.value_projection(
            torch.log1p(expression).unsqueeze(-1)
        )
        for block in self.blocks:
            hidden = block(hidden)
        weights = torch.softmax(self.pool_gate(hidden).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1)
        return torch.nn.functional.normalize(self.embedding_projection(pooled), dim=-1)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.decoder(embedding))

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.encode(expression)


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
    expression = torch.as_tensor(payload["expression"], dtype=torch.float32)
    device = torch.device(args.device)
    model = StateEmbeddingModel().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, expression.shape[0], args.batch_size):
            outputs.append(
                model(expression[start : start + args.batch_size].to(device)).cpu()
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=torch.cat(outputs).numpy())


if __name__ == "__main__":
    main()
