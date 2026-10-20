"""Standalone TACE inference implementation for the fixed TACE-OAM-7M checkpoint.

The scored model is a local PyTorch/e3nn TACE (``portable_tace``) reconstructed
from the checkpoint's own config and strictly loaded with ``strict=True``. No
``onescience`` import is used for the scored forward.
"""

from __future__ import annotations

from collections import OrderedDict
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

CHECKPOINT_SHA256 = "87f6b23c40a5b29c3b0d2520bca104d73c653866e82541919db94979c7f6511c"
EXPECTED_STATE_COUNT = 456
EXPECTED_STATE_NUMEL = 7105809


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Safely read the fixed TACE .pt envelope (state_dict + cfg + statistics)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("expected a TACE .pt envelope with a state_dict")
    for key in ("cfg", "statistics", "target_property", "embedding_property"):
        if key not in payload:
            raise ValueError(f"TACE .pt envelope is missing {key!r}")
    return payload


def raw_model_state(payload: dict[str, Any]) -> "OrderedDict[str, torch.Tensor]":
    """Return the complete model state with its envelope verified."""
    state = OrderedDict(payload["state_dict"])
    count = len(state)
    numel = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    if count != EXPECTED_STATE_COUNT or numel != EXPECTED_STATE_NUMEL:
        raise ValueError(f"unexpected TACE state envelope: count={count}, numel={numel}")
    return state


def build_model(payload: dict[str, Any]) -> TensorModel:
    """Reconstruct the local TACE from the checkpoint config (no onescience)."""
    cfg = payload["cfg"]
    return TensorModel(e3nnTACE(**cfg))


def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    """Build the local TACE and strictly load its complete state."""
    payload = load_checkpoint_payload(checkpoint_path)
    state = raw_model_state(payload)
    model = build_model(payload)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, payload, state


def _build_batch(atoms, model, device: str = "cpu"):
    element = model.get_torch_element()
    target_property = model.get_target_property()
    embedding_property = model.get_embedding_property()
    keydict = {f"{p}_key": p for p in PROPERTY}
    key_spec = update_keyspec_from_kwargs(KeySpecification(), keydict)
    max_neighbors = model.get_max_neighbors()
    cutoff = model.get_cutoff()
    data = from_atoms(
        element,
        atoms,
        cutoff,
        max_neighbors="inf" if max_neighbors is None else max_neighbors,
        keyspec=key_spec,
        target_property=target_property,
        embedding_property=embedding_property,
        neighborlist_backend="matscipy",
    )
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader([data], batch_size=1, shuffle=False)))
    batch.to(device)
    for p in target_property:
        for rg in PROPERTY[p]["requires_grad_with"]:
            batch[rg].requires_grad_(True)
    return batch


def evaluate_structure(model, atoms, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Return energy, forces and full 3x3 stress for one ASE structure."""
    batch = _build_batch(atoms, model, device)
    with torch.enable_grad():
        out = model(batch)
    return {
        "energy": out["energy"].detach().reshape(-1),
        "forces": out["forces"].detach(),
        "stress": out["stress"].detach(),
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
