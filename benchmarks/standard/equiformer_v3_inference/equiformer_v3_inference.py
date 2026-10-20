"""Standalone EquiformerV3 inference implementation for the fixed
omat24-mptrj-salex_gradient checkpoint.

The scored model is a local PyTorch/e3nn EquiformerV3 (``portable_equiformer_v3``)
reconstructed from the checkpoint's own config and strictly loaded with
``strict=True``. No ``onescience`` import is used for the scored forward.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from portable_equiformer_v3.utils.uma.common.registry import registry
from portable_equiformer_v3.utils.uma.common.utils import match_state_dict
from portable_equiformer_v3.utils.uma.normalization.normalizer import create_normalizer
from portable_equiformer_v3.utils.uma.normalization.element_references import create_element_references
import portable_equiformer_v3.models.equiformer_v3  # noqa: F401  (registers model classes)

CHECKPOINT_SHA256 = "429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98"
EXPECTED_STATE_COUNT = 397
EXPECTED_STATE_NUMEL = 30441865


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Read the fixed EquiformerV3 .pt (config + state_dict + normalizers + elementrefs)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload or "config" not in payload:
        raise ValueError("expected an EquiformerV3 .pt with config and state_dict")
    return payload


def raw_model_state(payload: dict[str, Any]) -> "OrderedDict[str, torch.Tensor]":
    """Return the complete model state with its envelope verified."""
    state = OrderedDict(payload["state_dict"])
    count = len(state)
    numel = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    if count != EXPECTED_STATE_COUNT or numel != EXPECTED_STATE_NUMEL:
        raise ValueError(f"unexpected EquiformerV3 state envelope: count={count}, numel={numel}")
    return state


def build_model(payload: dict[str, Any]) -> torch.nn.Module:
    """Reconstruct the local EquiformerV3 from the checkpoint config (no onescience)."""
    model_config = copy.deepcopy(payload["config"]["model"])
    model_name = model_config.pop("name", None)
    if model_name not in {"equiformer_v3", "equiformer_v3_dens"}:
        raise ValueError(f"Unsupported EquiformerV3 model name {model_name!r}")
    return registry.get_model_class(model_name)(**model_config)


def build_transforms(payload: dict[str, Any]) -> tuple[dict, dict]:
    """Build the denormalization normalizers and element references."""
    normalizers = {
        name: create_normalizer(state_dict=state)
        for name, state in payload.get("normalizers", {}).items()
    }
    elementrefs = {
        name: create_element_references(state_dict=state)
        for name, state in payload.get("elementrefs", {}).items()
    }
    return normalizers, elementrefs


def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    """Build the local EquiformerV3 and strictly load its complete state."""
    payload = load_checkpoint_payload(checkpoint_path)
    state = raw_model_state(payload)
    model = build_model(payload)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    state = match_state_dict(model.state_dict(), state)
    model.load_state_dict(state, strict=True)
    normalizers, elementrefs = build_transforms(payload)
    model._normalizers = normalizers
    model._elementrefs = elementrefs
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, payload, state


def _build_batch(atoms, model, device: str = "cpu"):
    from portable_equiformer_v3.datapipes.materials.custom_stack.core.atomic_data import AtomicData
    from portable_equiformer_v3.datapipes.materials.custom_stack.collaters.simple_collater import data_list_collater

    data = AtomicData.from_ase(
        atoms,
        max_neigh=model.max_neighbors,
        radius=model.cutoff,
        r_edges=False, r_energy=False, r_forces=False, r_stress=False,
    )
    return data_list_collater([data], otf_graph=True).to(device)


def _denormalize(name, prediction, batch, model):
    value = prediction
    if name in model._normalizers:
        value = model._normalizers[name].denorm(value)
    if name in model._elementrefs:
        shape = value.shape
        value = value.reshape(int(batch.natoms.numel()), -1)
        value = model._elementrefs[name](value, batch)
        value = value.reshape(shape)
    return value


def evaluate_structure(model, atoms, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Return energy, forces and full 3x3 stress for one ASE structure."""
    batch = _build_batch(atoms, model, device)
    with torch.enable_grad():
        out = model(batch)
    energy = _denormalize("energy", out["energy"], batch, model)
    forces = _denormalize("forces", out["forces"], batch, model)
    stress = _denormalize("stress", out["stress"], batch, model)
    return {
        "energy": energy.reshape(-1),
        "forces": forces,
        "stress": stress.reshape(3, 3),
    }


__all__ = [
    "CHECKPOINT_SHA256",
    "EXPECTED_STATE_COUNT",
    "EXPECTED_STATE_NUMEL",
    "build_model",
    "build_transforms",
    "evaluate_structure",
    "load_checkpoint_payload",
    "load_model",
    "raw_model_state",
]
