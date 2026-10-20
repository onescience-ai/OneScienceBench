#!/usr/bin/env python
"""独立的轻量 scGPT 风格细胞类型分类短训练入口。"""

from __future__ import annotations

import argparse
import json
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


class ScGPTClassifier(nn.Module):
    def __init__(
        self,
        num_genes: int = 128,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        num_labels: int = 3,
    ) -> None:
        super().__init__()
        self.gene_encoder = nn.Embedding(num_genes, hidden_size, padding_idx=0)
        self.value_encoder = ContinuousValueEncoder(hidden_size)
        self.blocks = nn.ModuleList(
            [ScGPTBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels),
        )

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
        return self.classifier(pooled)


def load_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = np.load(path)
    gene_ids = torch.as_tensor(payload["gene_ids"], dtype=torch.long)
    values = torch.as_tensor(payload["values"], dtype=torch.float32)
    labels = torch.as_tensor(payload["labels"], dtype=torch.long)
    if (
        gene_ids.ndim != 2
        or values.shape != gene_ids.shape
        or labels.shape != gene_ids.shape[:1]
        or gene_ids.shape[0] == 0
        or gene_ids.shape[1] == 0
    ):
        raise ValueError("dataset must contain aligned gene_ids, values, labels")
    if bool((gene_ids < 0).any()) or bool((gene_ids >= 128).any()):
        raise ValueError("gene_ids must be in [0, 128)")
    if bool((labels < 0).any()):
        raise ValueError("labels must be non-negative")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("expression values must be finite and non-negative")
    return gene_ids, values, labels


def mock_dataset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(29)
    gene_ids = torch.randint(1, 96, (12, 10), generator=generator)
    values = torch.rand(12, 10, generator=generator) * 5.0
    labels = torch.arange(12) % 3
    return gene_ids, values, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-npz", type=Path)
    group.add_argument("--mock-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1:
        raise ValueError("max_steps and batch_size must be positive")
    torch.manual_seed(args.seed)
    gene_ids, values, labels = (
        mock_dataset() if args.mock_data else load_dataset(args.dataset_npz)
    )
    num_labels = int(labels.max().item()) + 1
    device = torch.device(args.device)
    model = ScGPTClassifier(num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, float | int]] = []
    for step in range(1, args.max_steps + 1):
        indices = (
            torch.arange(args.batch_size) + step * args.batch_size
        ) % gene_ids.shape[0]
        batch_genes = gene_ids[indices].to(device)
        batch_values = values[indices].to(device)
        batch_labels = labels[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_genes, batch_values, batch_genes.eq(0))
        loss = criterion(logits, batch_labels)
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
            "config": {"num_labels": num_labels},
        },
        args.output_dir / "scgpt_train.pt",
    )


if __name__ == "__main__":
    main()
