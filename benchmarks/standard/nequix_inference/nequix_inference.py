"""Standalone NequIX inference implementation for the fixed nequix-mp-1 checkpoint.

The scored model is a local PyTorch/e3nn NequIX (``portable_nequix``) reconstructed
from the checkpoint's own config and strictly loaded with ``strict=True``. No
``onescience`` import is used for the scored forward.
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
EXPECTED_STATE_COUNT = 107
EXPECTED_STATE_NUMEL = 732710


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Read the fixed NequIX .pt (JSON config line + torch state_dict zip)."""
    with Path(path).open("rb") as handle:
        config = json.loads(handle.readline().decode())
        zip_bytes = handle.read()
    state_dict = torch.load(io.BytesIO(zip_bytes), map_location="cpu")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("expected a NequIX .pt with a non-empty state_dict")
    return {"config": config, "state_dict": state_dict}


def raw_model_state(payload: dict[str, Any]) -> "OrderedDict[str, torch.Tensor]":
    """Return the complete model state with its envelope verified."""
    state = OrderedDict(payload["state_dict"])
    count = len(state)
    numel = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    if count != EXPECTED_STATE_COUNT or numel != EXPECTED_STATE_NUMEL:
        raise ValueError(f"unexpected NequIX state envelope: count={count}, numel={numel}")
    return state


def build_model(payload: dict[str, Any]) -> NequixTorch:
    """Reconstruct the local NequIX from the checkpoint config (no onescience)."""
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


def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    """Build the local NequIX and strictly load its complete state."""
    payload = load_checkpoint_payload(checkpoint_path)
    state = raw_model_state(payload)
    model = build_model(payload)
    model.load_state_dict(state, strict=True)
    model._atomic_numbers = list(payload["config"]["atomic_numbers"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, payload, state


def _build_graph(atoms, model, device: str = "cpu"):
    atom_indices = atomic_numbers_to_indices(model._atomic_numbers)
    graph_dict = preprocess_graph(atoms, atom_indices, float(model.cutoff), False)
    graph = dict_to_pytorch_geometric(graph_dict)
    graph.n_graph = torch.zeros(graph.x.shape[0], dtype=torch.int64).to(device)
    graph = graph.to(device)
    return graph


def evaluate_structure(model, atoms, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Return energy, forces and full 3x3 stress for one ASE structure."""
    graph = _build_graph(atoms, model, device)
    with torch.enable_grad():
        energy_per_atom, forces, stress = model(
            graph.x,
            graph.positions,
            graph.edge_attr,
            graph.edge_index,
            getattr(graph, "cell", None),
            graph.n_node,
            graph.n_edge,
            graph.n_graph,
        )
    energy = scatter(energy_per_atom, graph.n_graph, dim=0, dim_size=graph.n_node.size(0))
    return {
        "energy": energy.reshape(-1),
        "forces": forces,
        "stress": stress,
    }


__all__ = [
    "CHECKPOINT_SHA256",
    "EXPECTED_STATE_COUNT",
    "EXPECTED_STATE_NUMEL",
    "build_model",
    "evaluate_structure",
    "load_checkpoint_payload",
    "load_model",
    "raw_model_state",
]
