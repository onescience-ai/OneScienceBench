"""Data and model APIs for TACE from-scratch energy-force training.

The scored model is a local PyTorch/e3nn TACE (``portable_tace``) built from a
fixed architecture config and deterministically initialized (no checkpoint).
No ``onescience`` import is used for the scored forward/backward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from portable_tace.models.tace.architectures.e3nn import e3nnTACE
from portable_tace.models.tace.wrappers import TensorModel
from portable_tace.datapipes.materials.tace.graph import from_atoms
from portable_tace.datapipes.materials.tace.quantity import (
    KeySpecification,
    update_keyspec_from_kwargs,
    PROPERTY,
)

DATA_SHA256 = "e540bc7c026cfad063fc970406bb74c0f668007fc3c6d200801ecfd5f1ba600e"
EXPECTED_STATE_COUNT = 456
EXPECTED_STATE_NUMEL = 7105809
EXPECTED_PARAM_COUNT = 184
SEED = 456
# Fixed TACE-OAM-7M graph constants (from the architecture config).
CUTOFF = 6.0
MAX_NEIGHBORS = None
TARGET_PROPERTY = ["energy", "forces", "stress"]
EMBEDDING_PROPERTY = []
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
    """Build the local TACE from the architecture config with a fixed seed."""
    arch = load_arch_config(config["paths"]["arch_config"])
    torch.manual_seed(int(config["training"]["seed"]))
    model = TensorModel(e3nnTACE(**arch["cfg"]))
    model.to(device).train()
    return model


def load_training_structures(path: str | Path):
    from ase.io import read

    structures = read(str(path), index=":")
    return list(structures) if isinstance(structures, list) else [structures]


def _build_graph(atoms, device: str = "cpu"):
    from portable_tace.datapipes.materials.tace.element import TorchElement

    element = TorchElement(ATOMIC_NUMBERS)
    keydict = {f"{p}_key": p for p in PROPERTY}
    key_spec = update_keyspec_from_kwargs(KeySpecification(), keydict)
    data = from_atoms(
        element,
        atoms,
        CUTOFF,
        max_neighbors="inf" if MAX_NEIGHBORS is None else MAX_NEIGHBORS,
        keyspec=key_spec,
        target_property=TARGET_PROPERTY,
        embedding_property=EMBEDDING_PROPERTY,
        neighborlist_backend="matscipy",
    )
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader([data], batch_size=1, shuffle=False)))
    batch.to(device)
    for p in TARGET_PROPERTY:
        for rg in PROPERTY[p]["requires_grad_with"]:
            batch[rg].requires_grad_(True)
    return batch


def build_train_batch(path: str | Path, index: int = 0, device: str = "cpu"):
    """Build a real one-structure energy+force batch from the DMC trajectory."""
    structures = load_training_structures(path)
    atoms = structures[int(index)]
    energy = float(atoms.info["energy_xtb"])
    forces = torch.as_tensor(atoms.arrays["forces_xtb"], dtype=torch.float32, device=device)
    data = _build_graph(atoms, device)
    return {
        "data": data,
        "energy": torch.tensor([[energy]], dtype=torch.float32, device=device),
        "forces": forces,
    }


def predict_energy_forces(model: torch.nn.Module, batch: dict[str, Any]):
    with torch.enable_grad():
        output = model(batch["data"])
    return {
        "energy": output["energy"],
        "forces": output["forces"],
        "stress": output["stress"],
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
