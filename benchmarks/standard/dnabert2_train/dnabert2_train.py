#!/usr/bin/env python
"""独立的轻量 DNABERT-2 风格序列分类短训练入口。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn

DNA_VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
CLS_TOKEN = 6


def encode_dna(sequence: str, max_length: int = 64) -> torch.Tensor:
    if max_length < 2:
        raise ValueError("max_length must leave room for CLS and a base")
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("DNA sequence must not be empty")
    invalid = sorted(set(normalized) - set(DNA_VOCAB))
    if invalid:
        raise ValueError(f"invalid DNA symbols: {invalid}")
    return torch.tensor(
        [CLS_TOKEN, *(DNA_VOCAB[base] for base in normalized[: max_length - 1])],
        dtype=torch.long,
    )


def load_sequence_csv(path: Path) -> list[tuple[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"sequence", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain sequence,label columns")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain at least one sample")
    if any(row["sequence"] is None or row["label"] is None for row in rows):
        raise ValueError("CSV must contain sequence,label columns")
    samples = [(row["sequence"], int(row["label"])) for row in rows]
    for sequence, label in samples:
        encode_dna(sequence)
        if label < 0:
            raise ValueError("labels must be non-negative")
    return samples


def collate(
    samples: list[tuple[str, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    encoded = [encode_dna(sequence) for sequence, _label in samples]
    width = max(item.numel() for item in encoded)
    tokens = torch.zeros(len(samples), width, dtype=torch.long)
    for index, item in enumerate(encoded):
        tokens[index, : item.numel()] = item
    labels = torch.tensor([label for _sequence, label in samples], dtype=torch.long)
    return tokens, tokens.ne(0), labels


class ALiBiSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        slopes = torch.tensor(
            [2.0 ** (-8.0 * (head + 1) / num_heads) for head in range(num_heads)]
        )
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, hidden_size = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        scores = torch.matmul(query, key.transpose(-1, -2)) * (self.head_dim**-0.5)
        positions = torch.arange(length, device=hidden.device)
        distance = (positions[:, None] - positions[None, :]).abs().to(hidden.dtype)
        scores = (
            scores
            - self.slopes.to(hidden.dtype)[None, :, None, None] * distance[None, None]
        )
        scores = scores.masked_fill(
            ~attention_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        probabilities = torch.softmax(scores, dim=-1)
        attended = (
            torch.matmul(probabilities, value)
            .transpose(1, 2)
            .reshape(batch, length, hidden_size)
        )
        return self.output_projection(attended)


class DNABert2Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = ALiBiSelfAttention(hidden_size, num_heads)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), attention_mask)
        return hidden + self.mlp(self.mlp_norm(hidden))


class DNABert2Classifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 7,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.blocks = nn.ModuleList(
            [DNABert2Block(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, length]")
        if attention_mask is None:
            attention_mask = tokens.ne(0)
        hidden = self.token_embedding(tokens)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.classifier(self.final_norm(hidden)[:, 0])


def mock_samples() -> list[tuple[str, int]]:
    return [
        ("ACGTACGTACGT", 0),
        ("TTTTGGGGCCCC", 1),
        ("AACCAACCGGTT", 0),
        ("GGGGTTTTAAAA", 1),
        ("ACACACACACAC", 0),
        ("TGTGTGTGTGTG", 1),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-dir", type=Path)
    group.add_argument("--mock-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1:
        raise ValueError("max_steps and batch_size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = (
        mock_samples()
        if args.mock_data
        else load_sequence_csv(args.dataset_dir / "train.csv")
    )
    num_labels = max(label for _sequence, label in samples) + 1
    device = torch.device(args.device)
    model = DNABert2Classifier(num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, float | int]] = []
    for step in range(1, args.max_steps + 1):
        batch = [
            samples[(step * args.batch_size + offset) % len(samples)]
            for offset in range(args.batch_size)
        ]
        tokens, mask, labels = collate(batch)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens.to(device), mask.to(device))
        loss = criterion(logits, labels.to(device))
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
        args.output_dir / "dnabert2_train.pt",
    )


if __name__ == "__main__":
    main()
