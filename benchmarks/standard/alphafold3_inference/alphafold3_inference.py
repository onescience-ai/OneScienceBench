"""Standalone compact AlphaFold3-style complex structure inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def sequence_from_json(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sequences = []
    for item in payload.get("sequences", []):
        if isinstance(item, dict) and isinstance(item.get("protein"), dict):
            sequences.append(str(item["protein"].get("sequence", "")))
        elif isinstance(item, dict) and isinstance(item.get("dna"), dict):
            sequences.append(str(item["dna"].get("sequence", "")))
    if not sequences and isinstance(payload.get("proteinChain"), dict):
        sequences.append(str(payload["proteinChain"].get("sequence", "")))
    sequence = "".join(sequences).upper()
    if not sequence or any(residue not in AMINO_ACIDS for residue in sequence):
        raise ValueError("input JSON must contain a non-empty standard protein sequence")
    return sequence


def make_features(sequence: str, num_recycles: int = 2) -> dict[str, torch.Tensor]:
    indices = torch.tensor([AMINO_ACIDS.index(residue) for residue in sequence], dtype=torch.long)
    length = len(sequence)
    target = nn.functional.one_hot(indices, num_classes=20).float()
    position = torch.arange(length, dtype=torch.float32)
    target = torch.cat([target, torch.sin(position[:, None] / 10.0), torch.cos(position[:, None] / 10.0)], dim=-1)
    return {
        "target_feat": target[None, :, None, :].repeat(1, 1, num_recycles, 1),
        "residue_index": position[None, :, None].repeat(1, 1, num_recycles),
    }


class TargetEmbedder(nn.Module):
    def __init__(self, target_dim: int = 22, single_dim: int = 32, pair_dim: int = 32):
        super().__init__()
        self.single_projection = nn.Linear(target_dim, single_dim)
        self.pair_projection = nn.Linear(target_dim * 2 + 1, pair_dim)

    def forward(self, target: torch.Tensor, residue_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single = self.single_projection(target)
        left = target[:, :, None, :].expand(-1, -1, target.shape[1], -1)
        right = target[:, None, :, :].expand(-1, target.shape[1], -1, -1)
        delta = (residue_index[:, :, None] - residue_index[:, None, :]).abs().unsqueeze(-1)
        return single, self.pair_projection(torch.cat([left, right, delta], dim=-1))


class PairFormerIteration(nn.Module):
    def __init__(self, single_dim: int = 32, pair_dim: int = 32):
        super().__init__()
        self.single_norm = nn.LayerNorm(single_dim)
        self.single_update = nn.Sequential(nn.Linear(single_dim, single_dim * 2), nn.SiLU(), nn.Linear(single_dim * 2, single_dim))
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_update = nn.Sequential(nn.Linear(pair_dim, pair_dim * 2), nn.SiLU(), nn.Linear(pair_dim * 2, pair_dim))

    def forward(self, single: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single = single + self.single_update(self.single_norm(single))
        pair = pair + self.pair_update(self.pair_norm(pair))
        return single, pair


class DiffusionHead(nn.Module):
    def __init__(self, single_dim: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(single_dim)
        self.atom_position_head = nn.Linear(single_dim, 3)

    def forward(self, single: torch.Tensor, num_samples: int = 2, seed: int = 0) -> torch.Tensor:
        generator = torch.Generator(device=single.device).manual_seed(seed)
        base = self.atom_position_head(self.norm(single))
        noise = torch.randn((num_samples, *base.shape), generator=generator, device=single.device, dtype=single.dtype) * 0.05
        return base.unsqueeze(0) + noise


class ConfidenceHead(nn.Module):
    def __init__(self, single_dim: int = 32):
        super().__init__()
        self.plddt = nn.Linear(single_dim, 1)
        self.resolved = nn.Linear(single_dim, 1)

    def forward(self, single: torch.Tensor) -> dict[str, torch.Tensor]:
        plddt = torch.sigmoid(self.plddt(single)).squeeze(-1)
        resolved = torch.sigmoid(self.resolved(single)).squeeze(-1)
        pair_score = torch.einsum("bli,bmi->blm", single, single).abs()
        pae = 1.0 / (1.0 + pair_score)
        return {"predicted_lddt": plddt, "predicted_experimentally_resolved": resolved, "full_pae": pae, "average_pae": pae.mean(dim=(-1, -2))}


class AlphaFold3(nn.Module):
    def __init__(self, num_recycles: int = 2, num_diffusion_samples: int = 2, single_dim: int = 32, pair_dim: int = 32, num_pairformer_layers: int = 2):
        super().__init__()
        self.num_recycles = num_recycles
        self.num_diffusion_samples = num_diffusion_samples
        self.target_embedder = TargetEmbedder(22, single_dim, pair_dim)
        self.pairformer = nn.ModuleList([PairFormerIteration(single_dim, pair_dim) for _ in range(num_pairformer_layers)])
        self.diffusion_head = DiffusionHead(single_dim)
        self.confidence_head = ConfidenceHead(single_dim)

    def forward(self, batch: dict[str, torch.Tensor], seed: int = 0) -> dict[str, torch.Tensor]:
        target = batch["target_feat"]
        residue_index = batch["residue_index"]
        if target.ndim != 4 or target.shape[-1] != 22 or residue_index.ndim != 3 or target.shape[2] != self.num_recycles:
            raise ValueError("target_feat must be [B,L,recycles,22] and residue_index [B,L,recycles]")
        previous_single = previous_pair = None
        for recycle in range(self.num_recycles):
            single, pair = self.target_embedder(target[:, :, recycle], residue_index[:, :, recycle])
            if previous_single is not None:
                single = single + previous_single
                pair = pair + previous_pair
            for block in self.pairformer:
                single, pair = block(single, pair)
            previous_single, previous_pair = single.detach(), pair.detach()
        positions = self.diffusion_head(single, self.num_diffusion_samples, seed)
        confidence = self.confidence_head(single)
        output = {"diffusion_samples": positions, "atom_positions": positions[0], "single_embeddings": single, "pair_embeddings": pair}
        output.update(confidence)
        return output


def write_pdb(coordinates: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if coordinates.ndim == 3:
        coordinates = coordinates[0]
    lines = [f"ATOM  {index:5d}  CA  ALA A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C" for index, (x, y, z) in enumerate(coordinates.tolist(), 1)]
    path.write_text("\n".join(lines + ["END"]) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaFold3-style multi-molecule inference")
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-data-pipeline", choices=("true", "false"), default="false")
    parser.add_argument("--run-inference", choices=("true", "false"), default="true")
    parser.add_argument("--flash-attention-implementation", choices=("xla", "triton", "cudnn", "cutlass"), default="xla")
    parser.add_argument("--num-recycles", type=int, default=2)
    parser.add_argument("--num-diffusion-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = AlphaFold3(args.num_recycles, args.num_diffusion_samples).to(device)
    model_path = Path(args.model_dir)
    if model_path.is_dir():
        model_path = model_path / "model.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    sequence = sequence_from_json(args.json_path)
    features = {name: value.to(device) for name, value in make_features(sequence, args.num_recycles).items()}
    with torch.inference_mode():
        output = model(features, seed=args.seed)
    output_dir = Path(args.output_dir)
    write_pdb(output["atom_positions"].cpu(), output_dir / "sample_0.pdb")
    (output_dir / "confidence.json").write_text(json.dumps({"average_pae": float(output["average_pae"].mean()), "mean_plddt": float(output["predicted_lddt"].mean())}), encoding="utf-8")
    with (output_dir / "ranking_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "score", "structure_file"])
        writer.writeheader()
        writer.writerow({"rank": 1, "score": float(output["predicted_lddt"].mean()), "structure_file": "sample_0.pdb"})


if __name__ == "__main__":
    main()
