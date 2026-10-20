"""Data and model APIs for NequIX from-scratch energy-force training.

The scored model is a local PyTorch/e3nn NequIX (``portable_nequix``) built from a
fixed architecture config and deterministically initialized (no checkpoint).
No ``onescience`` import is used for the scored forward/backward.
"""

from __future__ import annotations

import json
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

DATA_SHA256 = "b32cfaec1fdefb3a823e0bd19b7fab32d3a0c325e644dd388c5cc75117b4143b"
EXPECTED_STATE_COUNT = 107
EXPECTED_STATE_NUMEL = 732710
EXPECTED_PARAM_COUNT = 38
SEED = 456
# Fixed nequix-mp-1 graph constants (from the architecture config).
CUTOFF = 6.0
TARGET_PROPERTIES = ["energy", "forces", "stress"]
ATOMIC_NUMBERS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 89, 90, 91, 92, 93, 94,
]


def load_arch_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model(config: dict[str, Any], device: str = "cpu"):
    """Build the local NequIX from the architecture config with a fixed seed."""
    arch = load_arch_config(config["paths"]["arch_config"])
    torch.manual_seed(int(config["training"]["seed"]))
    model = NequixTorch(
        n_species=len(arch["atomic_numbers"]),
        hidden_irreps=arch["hidden_irreps"],
        lmax=arch["lmax"],
        cutoff=arch["cutoff"],
        n_layers=arch["n_layers"],
        radial_basis_size=arch["radial_basis_size"],
        radial_mlp_size=arch["radial_mlp_size"],
        radial_mlp_layers=arch["radial_mlp_layers"],
        radial_polynomial_p=arch["radial_polynomial_p"],
        mlp_init_scale=arch["mlp_init_scale"],
        index_weights=arch["index_weights"],
        layer_norm=arch["layer_norm"],
        shift=arch["shift"],
        scale=arch["scale"],
        avg_n_neighbors=arch["avg_n_neighbors"],
        atom_energies=[arch["atom_energies"][str(n)] for n in arch["atomic_numbers"]],
        kernel=False,
    )
    model._atomic_numbers = list(arch["atomic_numbers"])
    model.to(device).train()
    return model


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


def build_train_batch(path: str | Path, index: int = 0, device: str = "cpu"):
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
    "DATA_SHA256",
    "EXPECTED_STATE_COUNT",
    "EXPECTED_STATE_NUMEL",
    "EXPECTED_PARAM_COUNT",
    "SEED",
    "build_model",
    "build_train_batch",
    "load_arch_config",
    "load_training_structures",
    "predict_energy_forces",
]
