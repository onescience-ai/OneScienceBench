#!/usr/bin/env python
"""独立的轻量 DNABERT-2 风格 DNA embedding 推理入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

DNA_VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
CLS_TOKEN = 6


def encode_dna(sequence: str, max_length: int = 64) -> torch.Tensor:
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("DNA sequence must not be empty")
    invalid = sorted(set(normalized) - set(DNA_VOCAB))
    if invalid:
        raise ValueError(f"invalid DNA symbols: {invalid}")
    token_ids = [CLS_TOKEN, *(DNA_VOCAB[base] for base in normalized[: max_length - 1])]
    return torch.tensor(token_ids, dtype=torch.long)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(chunks)))
            identifier = line[1:].strip() or f"sequence_{len(records)}"
            chunks = []
        else:
            if identifier is None:
                raise ValueError("FASTA sequence appeared before its header")
            chunks.append(line)
    if identifier is not None:
        records.append((identifier, "".join(chunks)))
    if not records:
        raise ValueError("FASTA file contains no records")
    return records


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
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
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


class DNABert2Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 7,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.blocks = nn.ModuleList(
            [DNABert2Block(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

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
        return self.final_norm(hidden)

    def embed(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor, pooling: str = "mean"
    ) -> torch.Tensor:
        hidden = self(tokens, attention_mask)
        if pooling == "cls":
            return hidden[:, 0]
        valid = attention_mask.clone()
        valid[:, 0] = False
        if pooling == "max":
            return (
                hidden.masked_fill(~valid.unsqueeze(-1), torch.finfo(hidden.dtype).min)
                .max(dim=1)
                .values
            )
        if pooling != "mean":
            raise ValueError("pooling must be mean, max, or cls")
        weights = valid.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def pad_sequences(
    sequences: list[str], max_length: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_dna(sequence, max_length=max_length) for sequence in sequences]
    width = max(item.numel() for item in encoded)
    tokens = torch.zeros(len(encoded), width, dtype=torch.long)
    for index, item in enumerate(encoded):
        tokens[index, : item.numel()] = item
    return tokens, tokens.ne(0)


def load_checkpoint(model: nn.Module, checkpoint: Path, device: torch.device) -> None:
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    state = (
        payload.get("model_state_dict", payload)
        if isinstance(payload, dict)
        else payload
    )
    model.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pooling", choices=("mean", "max", "cls"), default="mean")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    records = read_fasta(args.input_fasta)
    tokens, mask = pad_sequences([sequence for _identifier, sequence in records])
    model = DNABert2Encoder().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    with torch.inference_mode():
        embeddings = (
            model.embed(tokens.to(device), mask.to(device), args.pooling).cpu().numpy()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        identifiers=np.asarray([identifier for identifier, _sequence in records]),
        embeddings=embeddings,
    )


if __name__ == "__main__":
    main()
