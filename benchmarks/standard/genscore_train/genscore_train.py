#!/usr/bin/env python
"""独立的轻量 GenScore 风格亲和力回归短训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DistanceExpansion(nn.Module):
    def __init__(self, bins: int = 8, cutoff: float = 8.0) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, bins))
        self.gamma = float(bins) / cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2)


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_size: int, radial_bins: int = 8) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.edge_gate = nn.Linear(radial_bins, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        nodes: torch.Tensor,
        coordinates: torch.Tensor,
        mask: torch.Tensor,
        distance_expansion: DistanceExpansion,
    ) -> torch.Tensor:
        distances = torch.cdist(coordinates, coordinates)
        radial = distance_expansion(distances)
        query, key, value = self.query(nodes), self.key(nodes), self.value(nodes)
        attention = torch.matmul(query, key.transpose(-1, -2)) * (
            nodes.shape[-1] ** -0.5
        )
        pair_mask = mask[:, :, None] & mask[:, None, :]
        attention = attention.masked_fill(~pair_mask, torch.finfo(attention.dtype).min)
        probabilities = torch.softmax(attention, dim=-1)
        gates = torch.sigmoid(self.edge_gate(radial))
        messages = (probabilities.unsqueeze(-1) * gates * value[:, None]).sum(dim=2)
        return self.norm(nodes + self.output_projection(messages))


class GenScoreModel(nn.Module):
    def __init__(
        self,
        ligand_features: int = 8,
        pocket_features: int = 10,
        hidden_size: int = 32,
        radial_bins: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.ligand_encoder = nn.Linear(ligand_features, hidden_size)
        self.pocket_encoder = nn.Linear(pocket_features, hidden_size)
        self.distance_expansion = DistanceExpansion(radial_bins)
        self.ligand_layers = nn.ModuleList(
            [GraphMessageLayer(hidden_size, radial_bins) for _ in range(num_layers)]
        )
        self.pocket_layers = nn.ModuleList(
            [GraphMessageLayer(hidden_size, radial_bins) for _ in range(num_layers)]
        )
        self.cross_projection = nn.Sequential(
            nn.Linear(hidden_size * 3 + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1)
        )

    def _pool(self, nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(nodes.dtype).unsqueeze(-1)
        return (nodes * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        ligand_features: torch.Tensor,
        ligand_coordinates: torch.Tensor,
        pocket_features: torch.Tensor,
        pocket_coordinates: torch.Tensor,
        ligand_mask: torch.Tensor | None = None,
        pocket_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ligand_features.ndim != 3 or pocket_features.ndim != 3:
            raise ValueError("feature tensors must have shape [batch, atoms, features]")
        if (
            ligand_coordinates.shape[:2] != ligand_features.shape[:2]
            or ligand_coordinates.shape[-1] != 3
        ):
            raise ValueError("ligand coordinates must align with ligand features")
        if (
            pocket_coordinates.shape[:2] != pocket_features.shape[:2]
            or pocket_coordinates.shape[-1] != 3
        ):
            raise ValueError("pocket coordinates must align with pocket features")
        if ligand_mask is None:
            ligand_mask = torch.ones(
                ligand_features.shape[:2],
                dtype=torch.bool,
                device=ligand_features.device,
            )
        if pocket_mask is None:
            pocket_mask = torch.ones(
                pocket_features.shape[:2],
                dtype=torch.bool,
                device=pocket_features.device,
            )
        ligand = self.ligand_encoder(ligand_features)
        pocket = self.pocket_encoder(pocket_features)
        for layer in self.ligand_layers:
            ligand = layer(
                ligand, ligand_coordinates, ligand_mask, self.distance_expansion
            )
        for layer in self.pocket_layers:
            pocket = layer(
                pocket, pocket_coordinates, pocket_mask, self.distance_expansion
            )
        ligand_pool, pocket_pool = (
            self._pool(ligand, ligand_mask),
            self._pool(pocket, pocket_mask),
        )
        distances = torch.cdist(ligand_coordinates, pocket_coordinates)
        valid = ligand_mask[:, :, None] & pocket_mask[:, None, :]
        closest = (
            distances.masked_fill(~valid, torch.inf).amin(dim=(1, 2)).unsqueeze(-1)
        )
        features = torch.cat(
            [ligand_pool, pocket_pool, ligand_pool * pocket_pool, closest], dim=-1
        )
        return self.score_head(self.cross_projection(features)).squeeze(-1)


def mock_dataset() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    return {
        "ligand_features": torch.randn(10, 5, 8, generator=generator),
        "ligand_coordinates": torch.randn(10, 5, 3, generator=generator),
        "pocket_features": torch.randn(10, 7, 10, generator=generator),
        "pocket_coordinates": torch.randn(10, 7, 3, generator=generator),
        "ligand_mask": torch.ones(10, 5, dtype=torch.bool),
        "pocket_mask": torch.ones(10, 7, dtype=torch.bool),
        "affinity": torch.linspace(-1.0, 1.0, 10),
    }


def load_dataset(path: Path) -> dict[str, torch.Tensor]:
    payload = np.load(path)
    result = {
        name: torch.as_tensor(payload[name], dtype=torch.float32)
        for name in (
            "ligand_features",
            "ligand_coordinates",
            "pocket_features",
            "pocket_coordinates",
            "affinity",
        )
    }
    result["ligand_mask"] = torch.as_tensor(payload["ligand_mask"], dtype=torch.bool)
    result["pocket_mask"] = torch.as_tensor(payload["pocket_mask"], dtype=torch.bool)
    sample_count = result["affinity"].numel()
    if result["affinity"].ndim != 1 or sample_count == 0:
        raise ValueError("affinity must have nonempty shape [samples]")
    for prefix, atom_features in (("ligand", 8), ("pocket", 10)):
        features = result[f"{prefix}_features"]
        coordinates = result[f"{prefix}_coordinates"]
        mask = result[f"{prefix}_mask"]
        if (
            features.ndim != 3
            or features.shape[0] != sample_count
            or features.shape[1] == 0
            or features.shape[2] != atom_features
            or coordinates.shape != (*features.shape[:2], 3)
            or mask.shape != features.shape[:2]
            or not bool(mask.any(dim=1).all())
        ):
            raise ValueError(f"{prefix} features, coordinates and mask must align")
        if not bool(
            torch.isfinite(features).all() and torch.isfinite(coordinates).all()
        ):
            raise ValueError(f"{prefix} features and coordinates must be finite")
    if not bool(torch.isfinite(result["affinity"]).all()):
        raise ValueError("affinity must be finite")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-npz", type=Path)
    group.add_argument("--mock-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1:
        raise ValueError("max_steps and batch_size must be positive")
    torch.manual_seed(args.seed)
    data = mock_dataset() if args.mock_data else load_dataset(args.dataset_npz)
    device = torch.device(args.device)
    model = GenScoreModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rows: list[dict[str, float | int]] = []
    sample_count = data["affinity"].shape[0]
    for step in range(1, args.max_steps + 1):
        indices = (
            torch.arange(args.batch_size) + step * args.batch_size
        ) % sample_count
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            data["ligand_features"][indices].to(device),
            data["ligand_coordinates"][indices].to(device),
            data["pocket_features"][indices].to(device),
            data["pocket_coordinates"][indices].to(device),
            data["ligand_mask"][indices].to(device),
            data["pocket_mask"][indices].to(device),
        )
        loss = torch.nn.functional.mse_loss(
            prediction, data["affinity"][indices].to(device)
        )
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
            "config": {"model": "GenScoreModel"},
        },
        args.output_dir / "genscore_train.pt",
    )


if __name__ == "__main__":
    main()
