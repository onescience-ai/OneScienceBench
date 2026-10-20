"""Dependency-free DiffDock-style ligand pose sampling and scoring."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import torch
from torch import nn


def parse_pdb_coordinates(path: str | Path) -> torch.Tensor:
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                values.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError as error:
                raise ValueError(f"invalid protein coordinate in {path}") from error
    if not values:
        raise ValueError("protein structure contains no atoms")
    return torch.tensor(values, dtype=torch.float32)


def parse_sdf_coordinates(path: str | Path) -> torch.Tensor:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError("invalid SDF")
    try:
        atom_count = int(lines[3][:3])
    except ValueError as error:
        raise ValueError("SDF counts line is invalid") from error
    coordinates = []
    for line in lines[4 : 4 + atom_count]:
        fields = line.split()
        if len(fields) < 3:
            raise ValueError("invalid SDF atom line")
        coordinates.append([float(fields[0]), float(fields[1]), float(fields[2])])
    if not coordinates:
        raise ValueError("SDF contains no atoms")
    return torch.tensor(coordinates, dtype=torch.float32)


class DiffDock(nn.Module):
    """Compact protein-ligand graph score model with tr/rot/tor heads."""

    def __init__(self, hidden_size: int = 32, distance_bins: int = 16, contact_scale: float = 1.0):
        super().__init__()
        self.distance_bins = distance_bins
        self.ligand_encoder = nn.Sequential(nn.Linear(3, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.receptor_encoder = nn.Sequential(nn.Linear(3, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.cross_edge_encoder = nn.Sequential(nn.Linear(distance_bins, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.cross_message = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.translation_head = nn.Linear(hidden_size, 3)
        self.rotation_head = nn.Linear(hidden_size, 3)
        self.torsion_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1))
        self.contact_scale = nn.Parameter(torch.tensor(float(contact_scale)), requires_grad=False)

    def forward(self, protein: torch.Tensor, ligand: torch.Tensor, diffusion_time: float | torch.Tensor = 0.5) -> dict[str, torch.Tensor]:
        if protein.ndim != 2 or ligand.ndim != 2 or protein.shape[-1] != 3 or ligand.shape[-1] != 3:
            raise ValueError("protein and ligand coordinates must have shape [atoms, 3]")
        protein_centered = protein - protein.mean(dim=0, keepdim=True)
        ligand_centered = ligand - ligand.mean(dim=0, keepdim=True)
        ligand_hidden = self.ligand_encoder(ligand_centered)
        receptor_hidden = self.receptor_encoder(protein_centered)
        distances = torch.cdist(ligand, protein)
        centers = torch.linspace(0.0, 20.0, self.distance_bins, device=distances.device, dtype=distances.dtype)
        rbf = torch.exp(-((distances.unsqueeze(-1) - centers) / (20.0 / self.distance_bins)) ** 2)
        edge_hidden = self.cross_edge_encoder(rbf)
        ligand_expanded = ligand_hidden[:, None, :].expand(-1, protein.shape[0], -1)
        receptor_expanded = receptor_hidden[None, :, :].expand(ligand.shape[0], -1, -1)
        messages = self.cross_message(torch.cat([ligand_expanded, receptor_expanded, edge_hidden], dim=-1)).mean(dim=1)
        time = torch.as_tensor(diffusion_time, device=ligand.device, dtype=ligand.dtype).reshape(-1)[0]
        pooled = (ligand_hidden + messages).mean(dim=0) * (1.0 + time)
        tr_pred = self.translation_head(pooled).unsqueeze(0)
        rot_pred = self.rotation_head(pooled).unsqueeze(0)
        if ligand.shape[0] > 1:
            bond_hidden = torch.cat([ligand_hidden[:-1], ligand_hidden[1:]], dim=-1)
            tor_pred = self.torsion_head(bond_hidden).squeeze(-1)
        else:
            tor_pred = ligand.new_empty((0,))
        return {"tr_pred": tr_pred, "rot_pred": rot_pred, "tor_pred": tor_pred}

    def score_pose(self, protein: torch.Tensor, ligand: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        moved = ligand + translation
        distances = torch.cdist(moved, protein)
        return -distances.min(dim=1).values.mean() * self.contact_scale

    @torch.no_grad()
    def sample(self, protein: torch.Tensor, ligand: torch.Tensor, samples: int = 4, seed: int = 0) -> list[dict[str, torch.Tensor | float]]:
        if samples <= 0:
            raise ValueError("samples must be positive")
        generator = torch.Generator(device=protein.device).manual_seed(seed)
        center = protein.mean(dim=0) - ligand.mean(dim=0)
        results = []
        for index in range(samples):
            noise = torch.randn(3, generator=generator, device=protein.device) * (0.05 + 0.02 * index)
            score_outputs = self(protein, ligand, diffusion_time=(index + 1) / samples)
            translation = center + noise + 0.01 * score_outputs["tr_pred"].squeeze(0)
            score = self.score_pose(protein, ligand, translation)
            results.append(
                {
                    "translation": translation.cpu(),
                    "rotation_score": score_outputs["rot_pred"].squeeze(0).cpu(),
                    "torsion_score": score_outputs["tor_pred"].cpu(),
                    "score": float(score.cpu()),
                }
            )
        return results


def write_pose_sdf(ligand: torch.Tensor, translation: torch.Tensor, path: Path) -> None:
    moved = ligand + translation
    lines = ["DiffDock pose", "OneScience evaluation", "", f"{moved.shape[0]:3d}{0:3d}  0  0  0  0  0  0  0  0999 V2000"]
    lines.extend(f"{x:10.4f}{y:10.4f}{z:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0" for x, y, z in moved.tolist())
    lines.extend(["M  END", "$$$$"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="DiffDock-style ligand pose sampling")
    parser.add_argument("--protein-path", required=True)
    parser.add_argument("--ligand-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model = DiffDock()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    protein = parse_pdb_coordinates(args.protein_path)
    ligand = parse_sdf_coordinates(args.ligand_path)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        poses = model.sample(protein, ligand, args.samples, args.seed)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "score", "pose_file"])
        for rank, pose in enumerate(sorted(poses, key=lambda item: float(item["score"]), reverse=True), 1):
            pose_path = output / f"rank{rank}_confidence{float(pose['score']):.4f}.sdf"
            write_pose_sdf(ligand, pose["translation"], pose_path)
            writer.writerow([rank, float(pose["score"]), pose_path.name])


if __name__ == "__main__":
    main()
