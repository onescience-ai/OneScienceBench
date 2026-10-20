"""Checkpoint, data and model APIs for a NequIX energy-offset adapter.

The scored model is a local PyTorch/e3nn NequIX (``portable_nequix``) reconstructed
from the fixed nequix-mp-1 checkpoint and strictly loaded with ``strict=True``,
wrapped by a thin adapter that adds one trainable scalar energy offset while
keeping the entire base model frozen. No ``onescience`` import is used for the
scored forward/backward.
"""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from portable_nequix import (
    NequixTorch,
    atomic_numbers_to_indices,
    dict_to_pytorch_geometric,
    preprocess_graph,
    scatter,
)

CHECKPOINT_SHA256 = "c6f54523bb73537994bc932f0a1a55c06b1d643e582e9c9ab20d00c35546791c"
DATA_SHA256 = "b32cfaec1fdefb3a823e0bd19b7fab32d3a0c325e644dd388c5cc75117b4143b"
EXPECTED_STATE_COUNT = 107
EXPECTED_STATE_NUMEL = 732710
EXPECTED_BASE_PARAM_COUNT = 38
# Fixed nequix-mp-1 graph constants (from the checkpoint config).
CUTOFF = 6.0
ATOMIC_NUMBERS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 89, 90, 91, 92, 93, 94,
]


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = json.loads(handle.readline().decode())
        zip_bytes = handle.read()
    state_dict = torch.load(io.BytesIO(zip_bytes), map_location="cpu")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("expected a NequIX .pt with a non-empty state_dict")
    return {"config": config, "state_dict": state_dict}


def raw_model_state(payload: dict[str, Any]) -> "OrderedDict[str, torch.Tensor]":
    state = OrderedDict(payload["state_dict"])
    count = len(state)
    numel = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    if count != EXPECTED_STATE_COUNT or numel != EXPECTED_STATE_NUMEL:
        raise ValueError(f"unexpected NequIX state envelope: count={count}, numel={numel}")
    return state


def build_model(payload: dict[str, Any]) -> NequixTorch:
    config = payload["config"]
    return NequixTorch(
        n_species=len(config["atomic_numbers"]),
        hidden_irreps=config["hidden_irreps"],
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        index_weights=config["index_weights"],
        layer_norm=config["layer_norm"],
        shift=config["shift"],
        scale=config["scale"],
        avg_n_neighbors=config["avg_n_neighbors"],
        atom_energies=[config["atom_energies"][str(n)] for n in config["atomic_numbers"]],
        kernel=False,
    )


class NequixEnergyOffsetAdapter(torch.nn.Module):
    """Add one trainable graph energy offset around a frozen NequIX model.

    The base ``NequixTorch.forward`` returns per-atom energy; the per-graph energy
    is obtained by scattering per-atom energy over ``n_graph``. To add a single
    per-graph offset, we add it to the first atom of each graph so the scatter
    contributes it exactly once per graph.
    """

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        dtype = next(base.parameters()).dtype
        self.energy_offset = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    def forward(self, species, positions, edge_attr, edge_index, cell, n_node, n_edge, n_graph):
        energy_per_atom, forces, stress = self.base(
            species, positions, edge_attr, edge_index, cell, n_node, n_edge, n_graph
        )
        # First atom of each graph (n_graph is non-decreasing in a PyG batch).
        is_first = torch.ones(n_graph.shape[0], dtype=torch.bool, device=n_graph.device)
        if n_graph.shape[0] > 1:
            is_first[1:] = n_graph[1:] != n_graph[:-1]
        energy_per_atom = energy_per_atom + self.energy_offset * is_first.to(energy_per_atom.dtype)
        return energy_per_atom, forces, stress


def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    payload = load_checkpoint_payload(checkpoint_path)
    state = raw_model_state(payload)
    base = build_model(payload)
    base.load_state_dict(state, strict=True)
    base._atomic_numbers = list(payload["config"]["atomic_numbers"])
    base.to(device)
    return NequixEnergyOffsetAdapter(base), payload, state


def load_training_structures(path: str | Path):
    import ase.db

    db = ase.db.connect(str(path), readonly=True, use_lock_file=False)
    return [row.toatoms() for row in db.select()]


def _build_graph(atoms, device: str = "cpu"):
    atom_indices = atomic_numbers_to_indices(ATOMIC_NUMBERS)
    graph_dict = preprocess_graph(atoms, atom_indices, CUTOFF, True)
    graph = dict_to_pytorch_geometric(graph_dict)
    graph.n_graph = torch.zeros(graph.x.shape[0], dtype=torch.int64).to(device)
    graph = graph.to(device)
    return graph


def build_finetune_batch(path: str | Path, index: int = 0, device: str = "cpu"):
    """Build a real one-structure energy+force batch from the mptrj_smoke db."""
    structures = load_training_structures(path)
    atoms = structures[int(index)]
    energy = float(atoms.get_potential_energy())
    forces = torch.as_tensor(atoms.get_forces(), dtype=torch.float32, device=device)
    data = _build_graph(atoms, device)
    return {
        "data": data,
        "energy": torch.tensor([[energy]], dtype=torch.float32, device=device),
        "forces": forces,
    }


def predict_energy_forces(model: torch.nn.Module, batch: dict[str, Any]):
    data = batch["data"]
    with torch.enable_grad():
        energy_per_atom, forces, stress = model(
            data.x,
            data.positions,
            data.edge_attr,
            data.edge_index,
            getattr(data, "cell", None),
            data.n_node,
            data.n_edge,
            data.n_graph,
        )
    energy = scatter(energy_per_atom, data.n_graph, dim=0, dim_size=data.n_node.size(0))
    return {
        "energy": energy,
        "forces": forces,
        "stress": stress,
    }


__all__ = [
    "CHECKPOINT_SHA256",
    "DATA_SHA256",
    "EXPECTED_STATE_COUNT",
    "EXPECTED_STATE_NUMEL",
    "EXPECTED_BASE_PARAM_COUNT",
    "NequixEnergyOffsetAdapter",
    "build_finetune_batch",
    "build_model",
    "load_checkpoint_payload",
    "load_model",
    "load_training_structures",
    "predict_energy_forces",
    "raw_model_state",
]
