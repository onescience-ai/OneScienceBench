"""Standalone compact OpenFold-style recycling structure inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def encode_sequence(sequence: str) -> torch.Tensor:
    sequence = sequence.strip().upper()
    if not sequence or any(residue not in AMINO_ACIDS for residue in sequence):
        raise ValueError("protein sequence must be non-empty and contain standard amino acids")
    return torch.tensor([AMINO_ACIDS.index(residue) for residue in sequence], dtype=torch.long)


def make_features(sequence: str, num_recycles: int = 2) -> dict[str, torch.Tensor]:
    tokens = encode_sequence(sequence)
    length = tokens.numel()
    residue_index = torch.arange(length, dtype=torch.float32)
    target_feat = nn.functional.one_hot(tokens, num_classes=20).float()
    target_feat = torch.cat([target_feat, torch.sin(residue_index[:, None] / 10.0), torch.cos(residue_index[:, None] / 10.0)], dim=-1)
    target_feat = target_feat.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, num_recycles)
    msa_feat = target_feat[:, None].repeat(1, 2, 1, 1, 1)
    return {"aatype": tokens[None, :, None].repeat(1, 1, num_recycles), "target_feat": target_feat, "msa_feat": msa_feat, "residue_index": residue_index[None, :, None].repeat(1, 1, num_recycles)}


class InputEmbedder(nn.Module):
    def __init__(self, input_dim: int = 22, single_dim: int = 32, pair_dim: int = 32):
        super().__init__()
        self.single_projection = nn.Linear(input_dim, single_dim)
        self.pair_projection = nn.Linear(input_dim * 2 + 1, pair_dim)

    def forward(self, target_feat: torch.Tensor, residue_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single = self.single_projection(target_feat)
        left = target_feat.unsqueeze(2).expand(-1, -1, target_feat.shape[1], -1)
        right = target_feat.unsqueeze(1).expand(-1, target_feat.shape[1], -1, -1)
        delta = (residue_index[:, :, None] - residue_index[:, None, :]).abs().unsqueeze(-1)
        pair = self.pair_projection(torch.cat([left, right, delta], dim=-1))
        return single, pair


class RecyclingEmbedder(nn.Module):
    def __init__(self, single_dim: int = 32, pair_dim: int = 32):
        super().__init__()
        self.single_norm = nn.LayerNorm(single_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)

    def forward(self, single: torch.Tensor, pair: torch.Tensor, previous_single: torch.Tensor | None, previous_pair: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_single is not None:
            single = single + self.single_norm(previous_single)
        if previous_pair is not None:
            pair = pair + self.pair_norm(previous_pair)
        return single, pair


class EvoformerBlock(nn.Module):
    def __init__(self, single_dim: int = 32, pair_dim: int = 32):
        super().__init__()
        self.single_norm = nn.LayerNorm(single_dim)
        self.single_update = nn.Sequential(nn.Linear(single_dim, single_dim * 2), nn.GELU(), nn.Linear(single_dim * 2, single_dim))
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_update = nn.Sequential(nn.Linear(pair_dim, pair_dim * 2), nn.GELU(), nn.Linear(pair_dim * 2, pair_dim))

    def forward(self, single: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single = single + self.single_update(self.single_norm(single))
        pair = pair + self.pair_update(self.pair_norm(pair))
        return single, pair


class StructureModule(nn.Module):
    def __init__(self, single_dim: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(single_dim)
        self.coordinate_head = nn.Linear(single_dim, 37 * 3)

    def forward(self, single: torch.Tensor) -> dict[str, torch.Tensor]:
        coordinates = self.coordinate_head(self.norm(single)).reshape(single.shape[0], single.shape[1], 37, 3)
        return {"final_atom_positions": coordinates, "final_atom_mask": torch.ones(coordinates.shape[:-1], device=coordinates.device), "final_affine_tensor": coordinates[:, :, 0]}


class OpenFold(nn.Module):
    def __init__(self, num_recycles: int = 2, single_dim: int = 32, pair_dim: int = 32, num_evoformer_blocks: int = 2):
        super().__init__()
        self.num_recycles = num_recycles
        self.input_embedder = InputEmbedder(22, single_dim, pair_dim)
        self.recycling_embedder = RecyclingEmbedder(single_dim, pair_dim)
        self.evoformer = nn.ModuleList([EvoformerBlock(single_dim, pair_dim) for _ in range(num_evoformer_blocks)])
        self.structure_module = StructureModule(single_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target_feat = batch["target_feat"]
        residue_index = batch["residue_index"]
        if target_feat.ndim != 4 or residue_index.ndim != 3 or target_feat.shape[-1] != self.num_recycles:
            raise ValueError("target_feat must be [B,L,22,recycles] and residue_index [B,L,recycles]")
        single = pair = previous_single = previous_pair = None
        for recycle in range(self.num_recycles):
            current_target = target_feat[..., recycle]
            current_index = residue_index[..., recycle]
            single, pair = self.input_embedder(current_target, current_index)
            single, pair = self.recycling_embedder(single, pair, previous_single, previous_pair)
            for block in self.evoformer:
                single, pair = block(single, pair)
            previous_single, previous_pair = single.detach(), pair.detach()
        output = self.structure_module(single)
        output.update({"single": single, "pair": pair, "num_recycles": torch.tensor(self.num_recycles)})
        return output


def write_pdb(coordinates: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if coordinates.ndim == 4:
        coordinates = coordinates[0]
    for index, xyz in enumerate(coordinates[:, 1].tolist(), 1):
        x, y, z = xyz
        lines.append(f"ATOM  {index:5d}  CA  ALA A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenFold-style protein structure inference")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-pdb", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    sequence = "".join(line.strip() for line in Path(args.fasta).read_text(encoding="utf-8").splitlines() if not line.startswith(">"))
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = OpenFold().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    batch = {name: value.to(device) for name, value in make_features(sequence, model.num_recycles).items()}
    with torch.inference_mode():
        output = model(batch)
    write_pdb(output["final_atom_positions"].cpu(), Path(args.output_pdb))


if __name__ == "__main__":
    main()
