"""Standalone TargetDiff-style pocket-conditioned ligand diffusion inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


def parse_pdb_coordinates(path: str | Path) -> torch.Tensor:
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            values.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not values:
        raise ValueError("protein PDB contains no atoms")
    return torch.tensor(values, dtype=torch.float32)


def parse_sdf_coordinates(path: str | Path) -> torch.Tensor:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError("invalid SDF")
    atom_count = int(lines[3][:3])
    values = [[float(field) for field in line.split()[:3]] for line in lines[4 : 4 + atom_count]]
    if not values:
        raise ValueError("ligand SDF contains no atoms")
    return torch.tensor(values, dtype=torch.float32)


class EGNNRefineLayer(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_size * 2 + 1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.node_mlp = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.coordinate_gate = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distance = torch.cdist(positions, positions).unsqueeze(-1)
        pair = self.edge_mlp(torch.cat([hidden[:, None, :].expand(-1, hidden.shape[0], -1), hidden[None, :, :].expand(hidden.shape[0], -1, -1), distance], dim=-1))
        message = pair.mean(dim=1)
        updated = hidden + self.node_mlp(torch.cat([hidden, message], dim=-1))
        direction = positions[:, None, :] - positions[None, :, :]
        coordinate_delta = (pair.mean(dim=-1, keepdim=True) * direction).mean(dim=1)
        return updated, positions + torch.tanh(self.coordinate_gate(updated)) * coordinate_delta


class TargetDiff(nn.Module):
    def __init__(self, protein_feature_dim: int = 3, ligand_vocab: int = 8, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.ligand_vocab = ligand_vocab
        self.protein_atom_emb = nn.Linear(protein_feature_dim, hidden_size)
        self.ligand_atom_emb = nn.Embedding(ligand_vocab, hidden_size)
        self.time_embedding = nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.refine_net = nn.ModuleList([EGNNRefineLayer(hidden_size) for _ in range(num_layers)])
        self.position_head = nn.Linear(hidden_size, 3)
        self.atom_type_head = nn.Linear(hidden_size, ligand_vocab)

    def forward(self, protein_pos: torch.Tensor, protein_v: torch.Tensor, ligand_pos: torch.Tensor, ligand_v: torch.Tensor, time_step: float | torch.Tensor = 0.5) -> dict[str, torch.Tensor]:
        if protein_pos.ndim != 2 or ligand_pos.ndim != 2 or protein_pos.shape[-1] != 3 or ligand_pos.shape[-1] != 3:
            raise ValueError("protein_pos and ligand_pos must have shape [atoms, 3]")
        if ligand_v.ndim != 1 or ligand_v.shape[0] != ligand_pos.shape[0]:
            raise ValueError("ligand_v must have shape [ligand_atoms]")
        protein_hidden = self.protein_atom_emb(protein_v)
        ligand_hidden = self.ligand_atom_emb(ligand_v)
        time = torch.as_tensor(time_step, dtype=ligand_pos.dtype, device=ligand_pos.device).reshape(1, 1)
        ligand_hidden = ligand_hidden + self.time_embedding(time).expand_as(ligand_hidden) + protein_hidden.mean(dim=0, keepdim=True)
        refined_pos = ligand_pos
        for layer in self.refine_net:
            ligand_hidden, refined_pos = layer(ligand_hidden, refined_pos)
        return {"pred_ligand_pos": refined_pos + self.position_head(ligand_hidden), "pred_ligand_v": self.atom_type_head(ligand_hidden), "final_h": torch.cat([protein_hidden, ligand_hidden], dim=0), "final_ligand_h": ligand_hidden}

    def get_diffusion_loss(self, protein_pos: torch.Tensor, protein_v: torch.Tensor, clean_ligand_pos: torch.Tensor, clean_ligand_v: torch.Tensor, time_step: float = 0.5) -> dict[str, torch.Tensor]:
        output = self(protein_pos, protein_v, clean_ligand_pos, clean_ligand_v, time_step)
        loss_pos = nn.functional.mse_loss(output["pred_ligand_pos"], clean_ligand_pos)
        loss_v = nn.functional.cross_entropy(output["pred_ligand_v"], clean_ligand_v)
        return {"loss_pos": loss_pos, "loss_v": loss_v, "loss": loss_pos + loss_v}

    @torch.no_grad()
    def sample_diffusion(self, protein_pos: torch.Tensor, protein_v: torch.Tensor, init_ligand_pos: torch.Tensor, init_ligand_v: torch.Tensor, num_steps: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        generator = torch.Generator(device=init_ligand_pos.device).manual_seed(seed)
        current_pos = init_ligand_pos + torch.randn(init_ligand_pos.shape, generator=generator, device=init_ligand_pos.device) * 0.1
        current_v = init_ligand_v.clone()
        trajectory = []
        for step in range(num_steps):
            output = self(protein_pos, protein_v, current_pos, current_v, (step + 1) / num_steps)
            current_pos = 0.5 * current_pos + 0.5 * output["pred_ligand_pos"]
            current_v = output["pred_ligand_v"].argmax(dim=-1)
            trajectory.append(current_pos.detach().cpu())
        return {"pos": current_pos, "v": current_v, "pos_traj": torch.stack(trajectory)}


def write_sdf(coordinates: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["TargetDiff sample", "OneScience evaluation", "", f"{coordinates.shape[0]:3d}{0:3d}  0  0  0  0  0  0  0  0  0999 V2000"]
    lines.extend(f"{x:10.4f}{y:10.4f}{z:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0" for x, y, z in coordinates.tolist())
    path.write_text("\n".join(lines + ["M  END", "$$$$"]) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TargetDiff-style pocket-conditioned sampling")
    parser.add_argument("--protein-path", required=True)
    parser.add_argument("--ligand-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = TargetDiff().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    protein = parse_pdb_coordinates(args.protein_path).to(device)
    ligand = parse_sdf_coordinates(args.ligand_path).to(device)
    protein_v = protein
    ligand_v = torch.zeros(ligand.shape[0], dtype=torch.long, device=device)
    with torch.inference_mode():
        result = model.sample_diffusion(protein, protein_v, ligand, ligand_v, args.num_steps, args.seed)
    output_dir = Path(args.out_dir)
    write_sdf(result["pos"].cpu(), output_dir / "ligand_sample.sdf")
    (output_dir / "scores.json").write_text(json.dumps({"num_steps": args.num_steps, "seed": args.seed, "final_atom_count": int(result["pos"].shape[0])}), encoding="utf-8")


if __name__ == "__main__":
    main()
