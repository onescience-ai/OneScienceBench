"""Small, standalone ProteinMPNN-style structure-conditioned sequence designer."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"


def parse_ca_coordinates(pdb_path: str | Path) -> torch.Tensor:
    coordinates = []
    for line in Path(pdb_path).read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            try:
                coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError as error:
                raise ValueError(f"invalid CA coordinate in {pdb_path}") from error
    if not coordinates:
        raise ValueError("PDB contains no alpha-carbon atoms")
    return torch.tensor(coordinates, dtype=torch.float32)


def radial_basis(distances: torch.Tensor, bins: int = 16, maximum: float = 20.0) -> torch.Tensor:
    centers = torch.linspace(0.0, maximum, bins, device=distances.device, dtype=distances.dtype)
    width = maximum / bins
    return torch.exp(-((distances.unsqueeze(-1) - centers) / width) ** 2)


class ProteinMPNNMessageLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        neighbor_nodes = nodes[neighbors]
        center_nodes = nodes.unsqueeze(1).expand_as(neighbor_nodes)
        messages = self.message_mlp(torch.cat([center_nodes, neighbor_nodes, edges], dim=-1)).mean(dim=1)
        return self.norm(nodes + self.update_mlp(torch.cat([nodes, messages], dim=-1)))


class ProteinMPNN(nn.Module):
    def __init__(
        self,
        hidden_size: int = 32,
        vocab_size: int = 21,
        k_neighbors: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.node_embedding = nn.Linear(3, hidden_size)
        self.edge_embedding = nn.Linear(19, hidden_size)
        self.encoder_layers = nn.ModuleList(
            [ProteinMPNNMessageLayer(hidden_size) for _ in range(num_encoder_layers)]
        )
        self.sequence_embedding = nn.Embedding(vocab_size + 1, hidden_size)
        self.decoder_layers = nn.ModuleList(
            [ProteinMPNNMessageLayer(hidden_size) for _ in range(num_decoder_layers)]
        )
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def build_geometry_graph(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = coordinates.shape[0]
        if length < 2:
            raise ValueError("at least two residues are required")
        k = min(self.k_neighbors, length - 1)
        distances = torch.cdist(coordinates, coordinates)
        distances.fill_diagonal_(float("inf"))
        neighbor_distances, neighbors = torch.topk(distances, k, largest=False)
        relative = coordinates[neighbors] - coordinates.unsqueeze(1)
        unit = relative / neighbor_distances.unsqueeze(-1).clamp_min(1e-6)
        edge_features = torch.cat([radial_basis(neighbor_distances), unit], dim=-1)
        return neighbors, self.edge_embedding(edge_features)

    def forward(self, coordinates: torch.Tensor, sequence_ids: torch.Tensor | None = None) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [residues, 3]")
        neighbors, edges = self.build_geometry_graph(coordinates)
        nodes = self.node_embedding(coordinates - coordinates.mean(dim=0, keepdim=True))
        for layer in self.encoder_layers:
            nodes = layer(nodes, edges, neighbors)
        if sequence_ids is None:
            sequence_ids = torch.full(
                (coordinates.shape[0],),
                self.sequence_embedding.num_embeddings - 1,
                device=coordinates.device,
                dtype=torch.long,
            )
        if sequence_ids.shape != (coordinates.shape[0],):
            raise ValueError("sequence_ids must have shape [residues]")
        nodes = nodes + self.sequence_embedding(sequence_ids)
        for layer in self.decoder_layers:
            nodes = layer(nodes, edges, neighbors)
        return self.output_projection(nodes)

    @torch.no_grad()
    def sample(self, coordinates: torch.Tensor, num_sequences: int = 1, temperature: float = 0.1, seed: int = 0) -> list[str]:
        if num_sequences <= 0 or temperature <= 0:
            raise ValueError("num_sequences and temperature must be positive")
        generator = torch.Generator(device=coordinates.device).manual_seed(seed)
        designs = []
        mask_index = self.sequence_embedding.num_embeddings - 1
        for _ in range(num_sequences):
            sequence_ids = torch.full(
                (coordinates.shape[0],), mask_index, device=coordinates.device, dtype=torch.long
            )
            for index in range(coordinates.shape[0]):
                probabilities = torch.softmax(self(coordinates, sequence_ids)[index] / temperature, dim=-1)
                sequence_ids[index] = torch.multinomial(probabilities[:20], 1, generator=generator)
            designs.append("".join(AMINO_ACIDS[int(token)] for token in sequence_ids.tolist()))
        return designs


def write_fasta(sequences: list[str], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">design_{index + 1}\n{sequence}\n" for index, sequence in enumerate(sequences)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ProteinMPNN-style sequence design")
    parser.add_argument("--pdb-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-folder", required=True)
    parser.add_argument("--num-seq-per-target", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = ProteinMPNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    coordinates = parse_ca_coordinates(args.pdb_path).to(device)
    with torch.inference_mode():
        sequences = model.sample(coordinates, args.num_seq_per_target, args.temperature, args.seed)
    write_fasta(sequences, Path(args.out_folder) / "seqs" / "designed.fasta")


if __name__ == "__main__":
    main()
