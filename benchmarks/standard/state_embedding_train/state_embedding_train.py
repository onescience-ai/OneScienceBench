#!/usr/bin/env python
"""独立的轻量 STATE 风格 masked-expression 重建短训练入口。"""

from __future__ import annotations

import argparse
import json
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
        return self.decode(self.encode(expression))


def load_expression(path: Path) -> torch.Tensor:
    payload = np.load(path)
    expression = torch.as_tensor(payload["expression"], dtype=torch.float32)
    if expression.ndim != 2 or expression.shape[0] == 0 or expression.shape[1] != 64:
        raise ValueError("expression must have nonempty shape [cells, 64]")
    if not bool(torch.isfinite(expression).all()) or bool((expression < 0).any()):
        raise ValueError("expression must be finite and non-negative")
    return expression


def mock_expression() -> torch.Tensor:
    generator = torch.Generator().manual_seed(53)
    return torch.poisson(torch.rand(16, 64, generator=generator) * 3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-npz", type=Path)
    group.add_argument("--mock-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("max_steps, batch_size and mask_ratio are invalid")
    torch.manual_seed(args.seed)
    expression = (
        mock_expression() if args.mock_data else load_expression(args.dataset_npz)
    )
    device = torch.device(args.device)
    model = StateEmbeddingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed + 1)
    rows: list[dict[str, float | int]] = []
    for step in range(1, args.max_steps + 1):
        indices = (
            torch.arange(args.batch_size) + step * args.batch_size
        ) % expression.shape[0]
        target = expression[indices].to(device)
        mask = (
            torch.rand(target.shape, generator=generator).lt(args.mask_ratio).to(device)
        )
        empty_rows = ~mask.any(dim=1)
        mask[empty_rows, 0] = True
        masked = target.masked_fill(mask, 0.0)
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(masked)
        loss = ((torch.log1p(reconstruction) - torch.log1p(target)) ** 2)[mask].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rows.append({"step": step, "train_loss": float(loss.detach().cpu())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": args.max_steps,
            "config": {"num_genes": 64, "mask_ratio": args.mask_ratio},
        },
        args.output_dir / "state_embedding_train.pt",
    )


if __name__ == "__main__":
    main()
