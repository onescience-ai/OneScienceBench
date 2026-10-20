#!/usr/bin/env python
"""独立的轻量 GenScore 风格蛋白配体亲和力推理入口。"""

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
        query = self.query(nodes)
        key = self.key(nodes)
        value = self.value(nodes)
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
        ligand_pool = self._pool(ligand, ligand_mask)
        pocket_pool = self._pool(pocket, pocket_mask)
        cross_distance = torch.cdist(ligand_coordinates, pocket_coordinates)
        valid_pairs = ligand_mask[:, :, None] & pocket_mask[:, None, :]
        closest = (
            cross_distance.masked_fill(~valid_pairs, torch.inf)
            .amin(dim=(1, 2))
            .unsqueeze(-1)
        )
        interaction = torch.cat(
            [ligand_pool, pocket_pool, ligand_pool * pocket_pool, closest], dim=-1
        )
        return self.score_head(self.cross_projection(interaction)).squeeze(-1)


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
    parser.add_argument("--complex-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    data = np.load(args.complex_npz)
    device = torch.device(args.device)
    model = GenScoreModel().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    tensors = [
        torch.as_tensor(data[name], dtype=torch.float32, device=device)
        for name in (
            "ligand_features",
            "ligand_coordinates",
            "pocket_features",
            "pocket_coordinates",
        )
    ]
    ligand_mask = torch.as_tensor(data["ligand_mask"], dtype=torch.bool, device=device)
    pocket_mask = torch.as_tensor(data["pocket_mask"], dtype=torch.bool, device=device)
    with torch.inference_mode():
        scores = model(*tensors, ligand_mask, pocket_mask).cpu().tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scores": scores}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
