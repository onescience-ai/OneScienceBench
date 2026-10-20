"""Minimal Evo2-style causal language-model training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


VOCAB = "ACGTN"
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}


def encode_sequence(sequence: str) -> torch.Tensor:
    sequence = sequence.strip().upper()
    if not sequence or any(base not in TOKEN_TO_ID for base in sequence):
        raise ValueError("sequences must be non-empty and contain only A/C/G/T/N")
    return torch.tensor([TOKEN_TO_ID[base] for base in sequence], dtype=torch.long)


class Evo2MambaBlock(nn.Module):
    def __init__(self, hidden_size: int = 32, conv_kernel: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, hidden_size * 2)
        self.depthwise_conv = nn.Conv1d(hidden_size, hidden_size, conv_kernel, groups=hidden_size)
        self.ssm_projection = nn.Linear(hidden_size, hidden_size * 3)
        self.delta_projection = nn.Linear(hidden_size, hidden_size)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, hidden_size + 1).float()))
        self.D = nn.Parameter(torch.ones(hidden_size))
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mixed, gate = self.input_projection(self.norm(inputs)).chunk(2, dim=-1)
        kernel = self.depthwise_conv.kernel_size[0]
        mixed = nn.functional.silu(self.depthwise_conv(nn.functional.pad(mixed.transpose(1, 2), (kernel - 1, 0))).transpose(1, 2))
        delta_raw, input_scale, output_scale = self.ssm_projection(mixed).chunk(3, dim=-1)
        delta = nn.functional.softplus(self.delta_projection(delta_raw))
        transition = -torch.exp(self.A_log).view(1, -1)
        state = torch.zeros(inputs.shape[0], inputs.shape[2], device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for index in range(inputs.shape[1]):
            decay = torch.exp(delta[:, index] * transition)
            state = decay * state + delta[:, index] * input_scale[:, index] * mixed[:, index]
            outputs.append(output_scale[:, index] * state + self.D * mixed[:, index])
        return inputs + self.output_projection(torch.stack(outputs, dim=1) * torch.sigmoid(gate))


class Evo2TrainModel(nn.Module):
    def __init__(self, vocab_size: int = 5, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.mamba_layers = nn.ModuleList([Evo2MambaBlock(hidden_size) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 long tensor")
        hidden = self.token_embedding(input_ids)
        for layer in self.mamba_layers:
            hidden = layer(hidden)
        return self.lm_head(self.final_norm(hidden))


class Evo2Trainer:
    """AdamW trainer matching the small contract used by ``train_one_node.py``."""

    def __init__(self, model: Evo2TrainModel, output_dir: str | Path, learning_rate: float = 3e-4, seed: int = 42):
        self.model = model
        self.output_dir = Path(output_dir)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.seed = seed

    def train_step(self, sequence_batch: Iterable[str | torch.Tensor]) -> float:
        tensors = [encode_sequence(item) if isinstance(item, str) else item.long() for item in sequence_batch]
        if not tensors or any(tensor.numel() < 2 for tensor in tensors):
            raise ValueError("a batch must contain sequences of length >= 2")
        length = min(tensor.numel() for tensor in tensors)
        batch = torch.stack([tensor[:length] for tensor in tensors])
        logits = self.model(batch[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach())

    def fit(self, sequences: Iterable[str | torch.Tensor], max_steps: int = 2) -> list[dict[str, float | int]]:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        values = list(sequences)
        if not values:
            raise ValueError("training data must not be empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics = []
        for step in range(1, max_steps + 1):
            loss = self.train_step(values)
            metrics.append({"step": step, "train_loss": loss})
        (self.output_dir / "metrics.jsonl").write_text("".join(json.dumps(item) + "\n" for item in metrics), encoding="utf-8")
        torch.save({"format": "evo2-train-v1", "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "completed_steps": max_steps}, self.output_dir / "evo2_train.pt")
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact Evo2-style model")
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.dataset_config) == bool(args.mock_data):
        raise ValueError("provide exactly one of --dataset-config and --mock-data")
    if args.mock_data:
        sequences = ["ACGTACGT", "CGTACGTA"]
    else:
        config = json.loads(args.dataset_config.read_text(encoding="utf-8")) if args.dataset_config.suffix == ".json" else {"sequences": args.dataset_config.read_text(encoding="utf-8").splitlines()}
        sequences = config.get("sequences", [])
    torch.manual_seed(args.seed)
    trainer = Evo2Trainer(Evo2TrainModel(), args.output_dir, seed=args.seed)
    trainer.fit(sequences, args.max_steps)


if __name__ == "__main__":
    main()
