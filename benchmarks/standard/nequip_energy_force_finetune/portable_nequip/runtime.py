"""Portable graph, checkpoint and dataset helpers for the Cu NequIP tasks."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import AtomicDataDict
from .model import build_nequip_model


MODEL_STATE_PREFIX = "model.sole_model."
EXPECTED_MODEL_STATE_COUNT = 33
EXPECTED_MODEL_STATE_NUMEL = 1855
EXPECTED_PARAMETER_COUNT = 11
EXPECTED_PARAMETER_NUMEL = 1680


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("expected a trusted PyTorch Lightning checkpoint with state_dict")
    return payload


def raw_model_state(payload: dict[str, Any]) -> OrderedDict[str, torch.Tensor]:
    source = payload["state_dict"]
    state = OrderedDict(
        (str(key)[len(MODEL_STATE_PREFIX):], value)
        for key, value in source.items()
        if str(key).startswith(MODEL_STATE_PREFIX)
    )
    count = len(state)
    numel = sum(value.numel() for value in state.values())
    if count != EXPECTED_MODEL_STATE_COUNT or numel != EXPECTED_MODEL_STATE_NUMEL:
        raise ValueError(f"unexpected model state envelope: count={count}, numel={numel}")
    return state


def evaluation_model_state(
    payload: dict[str, Any], model: torch.nn.Module
) -> OrderedDict[str, torch.Tensor]:
    """Reproduce EMALightningModule.evaluation_model without importing OneScience."""
    raw = raw_model_state(payload)
    model.load_state_dict(raw, strict=True)
    state = OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in model.state_dict().items()
    )
    parameters = list(model.named_parameters())
    if len(parameters) != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"expected {EXPECTED_PARAMETER_COUNT} model parameters, got {len(parameters)}")
    if sum(parameter.numel() for _, parameter in parameters) != EXPECTED_PARAMETER_NUMEL:
        raise ValueError("unexpected trainable parameter envelope")
    source = payload["state_dict"]
    expected_ema_keys = [f"ema.ema_weight_{index}" for index in range(len(parameters))]
    actual_ema_keys = [str(key) for key in source if str(key).startswith("ema.ema_weight_")]
    if sorted(actual_ema_keys) != sorted(expected_ema_keys):
        raise ValueError(f"unexpected EMA envelope: {actual_ema_keys}")
    for (name, parameter), ema_key in zip(parameters, expected_ema_keys):
        value = source[ema_key]
        if value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise ValueError(f"EMA tensor mismatch for {name}: {ema_key}")
        state[name] = value.detach().clone()
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    if tuple(loaded) != tuple(state) or not all(
        torch.equal(loaded[name].cpu(), state[name].cpu()) for name in state
    ):
        raise RuntimeError("strict EMA evaluation state verification failed")
    return state


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
):
    payload = load_checkpoint_payload(checkpoint_path)
    model = build_nequip_model(seed=456)
    state = evaluation_model_state(payload, model)
    model.to(device).eval()
    return model, payload, state


def load_structures(path: str | Path):
    from ase.io import read

    structures = read(str(path), index=":")
    return list(structures) if isinstance(structures, list) else [structures]


def atoms_to_data(atoms, device: str | torch.device = "cpu"):
    from ase.neighborlist import neighbor_list

    center, neighbor, shifts = neighbor_list("ijS", atoms, 4.0, self_interaction=False)
    positions = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64, device=device)
    cell = torch.as_tensor(np.asarray(atoms.cell), dtype=torch.float64, device=device).view(1, 3, 3)
    edge_index = torch.as_tensor(
        np.stack((center, neighbor), axis=0), dtype=torch.long, device=device
    )
    edge_cell_shift = torch.as_tensor(
        np.asarray(shifts), dtype=torch.float64, device=device
    )
    num_atoms = len(atoms)
    return {
        AtomicDataDict.POSITIONS_KEY: positions,
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: edge_cell_shift,
        AtomicDataDict.CELL_KEY: cell,
        AtomicDataDict.PBC_KEY: torch.as_tensor(
            np.asarray(atoms.pbc), dtype=torch.bool, device=device
        ).view(1, 3),
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros(num_atoms, dtype=torch.long, device=device),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.as_tensor(
            np.asarray(atoms.numbers), dtype=torch.long, device=device
        ),
        AtomicDataDict.BATCH_KEY: torch.zeros(num_atoms, dtype=torch.long, device=device),
        AtomicDataDict.NUM_NODES_KEY: torch.tensor([num_atoms], dtype=torch.long, device=device),
    }


def clone_data(data: dict[str, torch.Tensor]):
    return {key: value.detach().clone() for key, value in data.items()}


def build_energy_force_batch(
    data_path: str | Path, index: int = 0, device: str | torch.device = "cpu"
):
    structures = load_structures(data_path)
    atoms = structures[int(index)]
    energy = (
        float(atoms.info["energy"])
        if "energy" in atoms.info
        else float(atoms.get_potential_energy())
    )
    forces = (
        np.asarray(atoms.arrays["forces"])
        if "forces" in atoms.arrays
        else np.asarray(atoms.get_forces())
    )
    return {
        "data": atoms_to_data(atoms, device),
        "energy": torch.tensor(
            [[energy]], dtype=torch.float64, device=device
        ),
        "forces": torch.as_tensor(
            forces, dtype=torch.float64, device=device
        ),
    }


def predict_energy_forces(model: torch.nn.Module, batch: dict[str, Any]):
    with torch.enable_grad():
        output = model(clone_data(batch["data"]))
    return {
        "energy": output[AtomicDataDict.TOTAL_ENERGY_KEY],
        "forces": output[AtomicDataDict.FORCE_KEY],
        "stress": output[AtomicDataDict.STRESS_KEY],
    }


def evaluate_atoms(model: torch.nn.Module, atoms, device="cpu"):
    data = atoms_to_data(atoms, device)
    with torch.enable_grad():
        output = model(data)
    return {
        "energy": output[AtomicDataDict.TOTAL_ENERGY_KEY].detach().reshape(-1),
        "forces": output[AtomicDataDict.FORCE_KEY].detach(),
        "stress": output[AtomicDataDict.STRESS_KEY].detach(),
    }
