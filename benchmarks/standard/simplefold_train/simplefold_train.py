"""Standalone lightweight SimpleFold-style flow-matching training.

The implementation preserves the FoldingDiT atom/token conditioning and
velocity-learning contracts while remaining small enough for dynamic tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


REQUIRED_FEATURES = ("ref_pos", "atom_to_token", "atom_pad_mask", "esm_s")


def validate_batch(batch: dict[str, Any]) -> None:
    if not isinstance(batch, dict) or "positions" not in batch or "feats" not in batch:
        raise ValueError("batch must contain positions and feats")
    positions = batch["positions"]
    feats = batch["feats"]
    if not isinstance(positions, torch.Tensor) or positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [batch, atoms, 3]")
    if not isinstance(feats, dict) or any(name not in feats for name in REQUIRED_FEATURES):
        raise ValueError(f"feats must contain {REQUIRED_FEATURES}")
    batch_size, atom_count, _ = positions.shape
    ref_pos = feats["ref_pos"]
    atom_to_token = feats["atom_to_token"]
    atom_pad_mask = feats["atom_pad_mask"]
    esm_s = feats["esm_s"]
    if ref_pos.shape != positions.shape:
        raise ValueError("ref_pos must match positions")
    if atom_to_token.ndim != 3 or atom_to_token.shape[:2] != (batch_size, atom_count):
        raise ValueError("atom_to_token must have shape [batch, atoms, tokens]")
    if atom_pad_mask.shape != (batch_size, atom_count):
        raise ValueError("atom_pad_mask must have shape [batch, atoms]")
    if esm_s.ndim != 3 or esm_s.shape[:2] != (batch_size, atom_to_token.shape[-1]):
        raise ValueError("esm_s must have shape [batch, tokens, esm_dim]")


def make_synthetic_batch(
    batch_size: int = 2,
    atom_count: int = 8,
    token_count: int = 4,
    esm_dim: int = 16,
    seed: int = 17,
) -> dict[str, Any]:
    if min(batch_size, atom_count, token_count, esm_dim) <= 0 or atom_count < token_count:
        raise ValueError("synthetic batch dimensions must be positive and atoms >= tokens")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions = torch.randn(batch_size, atom_count, 3, generator=generator)
    ref_pos = torch.randn(batch_size, atom_count, 3, generator=generator)
    esm_s = torch.randn(batch_size, token_count, esm_dim, generator=generator)
    assignment = torch.arange(atom_count) % token_count
    atom_to_token = nn.functional.one_hot(assignment, token_count).float()
    atom_to_token = atom_to_token.unsqueeze(0).repeat(batch_size, 1, 1)
    atom_pad_mask = torch.ones(batch_size, atom_count, dtype=torch.bool)
    return {
        "positions": positions,
        "feats": {
            "ref_pos": ref_pos,
            "atom_to_token": atom_to_token,
            "atom_pad_mask": atom_pad_mask,
            "esm_s": esm_s,
        },
    }


class FoldingDiT(nn.Module):
    """Compact atom-token velocity model following the SimpleFold contract."""

    def __init__(
        self,
        hidden_size: int = 32,
        num_heads: int = 4,
        esm_dim: int = 16,
        num_trunk_layers: int = 2,
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.esm_dim = esm_dim
        self.coordinate_embedding = nn.Linear(3, hidden_size)
        self.reference_embedding = nn.Linear(3, hidden_size)
        self.self_condition_embedding = nn.Linear(3, hidden_size)
        self.time_embedder = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        atom_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.atom_encoder = nn.TransformerEncoder(atom_layer, num_layers=1)
        self.esm_projection = nn.Linear(esm_dim, hidden_size)
        trunk_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.trunk = nn.TransformerEncoder(trunk_layer, num_layers=num_trunk_layers)
        decoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.atom_decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.velocity_head = nn.Linear(hidden_size, output_channels)

    def forward(
        self,
        noised_pos: torch.Tensor,
        t: torch.Tensor,
        feats: dict[str, torch.Tensor],
        self_cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        validate_batch({"positions": noised_pos, "feats": feats})
        if t.ndim != 1 or t.shape[0] != noised_pos.shape[0]:
            raise ValueError("t must have shape [batch]")
        atom_mask = feats["atom_pad_mask"].bool()
        atom_to_token = feats["atom_to_token"].to(noised_pos.dtype)
        hidden = self.coordinate_embedding(noised_pos)
        hidden = hidden + self.reference_embedding(feats["ref_pos"])
        hidden = hidden + self.time_embedder(t[:, None])[:, None, :]
        if self_cond is not None:
            if self_cond.shape != noised_pos.shape:
                raise ValueError("self_cond must match noised_pos")
            hidden = hidden + self.self_condition_embedding(self_cond)
        hidden = self.atom_encoder(hidden, src_key_padding_mask=~atom_mask)

        token_counts = atom_to_token.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        latent = torch.bmm(atom_to_token.transpose(1, 2), hidden) / token_counts
        if feats["esm_s"].shape[-1] != self.esm_dim:
            raise ValueError(f"esm_s final dimension must equal {self.esm_dim}")
        latent = self.trunk(latent + self.esm_projection(feats["esm_s"]))
        hidden = self.atom_decoder(hidden + torch.bmm(atom_to_token, latent), src_key_padding_mask=~atom_mask)
        velocity = self.velocity_head(hidden) * atom_mask.unsqueeze(-1)
        return {"predict_velocity": velocity, "latent": latent}


class SimpleFoldTrainer:
    def __init__(
        self,
        model: FoldingDiT,
        output_dir: str | Path,
        learning_rate: float = 3e-4,
        max_steps: int = 2,
        grad_clip_norm: float = 1.0,
        seed: int = 42,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.model = model
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.grad_clip_norm = grad_clip_norm
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.last_prediction: torch.Tensor | None = None
        self.last_target: torch.Tensor | None = None

    def train_step(self, batch: dict[str, Any], step: int) -> dict[str, float | int]:
        validate_batch(batch)
        positions = batch["positions"]
        feats = batch["feats"]
        noise = torch.randn(positions.shape, generator=self.generator, dtype=positions.dtype)
        t = torch.rand(positions.shape[0], generator=self.generator, dtype=positions.dtype)
        noised_pos = (1.0 - t[:, None, None]) * noise + t[:, None, None] * positions
        target_velocity = positions - noise
        output = self.model(noised_pos, t, feats)
        prediction = output["predict_velocity"]
        mask = feats["atom_pad_mask"].to(prediction.dtype).unsqueeze(-1)
        loss = (((prediction - target_velocity) ** 2) * mask).sum() / (mask.sum() * 3).clamp_min(1.0)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.last_prediction = prediction.detach().cpu().reshape(-1)
        self.last_target = target_velocity.detach().cpu().reshape(-1)
        rmse = torch.sqrt(torch.mean((self.last_prediction - self.last_target) ** 2))
        mae = torch.mean(torch.abs(self.last_prediction - self.last_target))
        return {
            "step": step,
            "train_loss": float(loss.detach()),
            "rmse": float(rmse),
            "mae": float(mae),
            "grad_norm": float(grad_norm),
        }

    def fit(self, batches: list[dict[str, Any]]) -> list[dict[str, float | int]]:
        if not batches:
            raise ValueError("batches must not be empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.train()
        history = [self.train_step(batches[(step - 1) % len(batches)], step) for step in range(1, self.max_steps + 1)]
        (self.output_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in history),
            encoding="utf-8",
        )
        torch.save(
            {
                "format": "simplefold-flow-train-v1",
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "completed_steps": self.max_steps,
            },
            self.output_dir / "simplefold_train.pt",
        )
        (self.output_dir / "predictions.json").write_text(
            json.dumps({"predictions": self.last_prediction.tolist()}), encoding="utf-8"
        )
        (self.output_dir / "targets.json").write_text(
            json.dumps({"targets": self.last_target.tolist()}), encoding="utf-8"
        )
        return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact SimpleFold FoldingDiT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock-data", action="store_true")
    args = parser.parse_args()
    if not args.mock_data:
        raise ValueError("this standalone entry point requires caller data or --mock-data")
    torch.manual_seed(args.seed)
    model = FoldingDiT()
    trainer = SimpleFoldTrainer(model, args.output_dir, max_steps=args.max_steps, seed=args.seed)
    trainer.fit([make_synthetic_batch(seed=args.seed)])


if __name__ == "__main__":
    main()
