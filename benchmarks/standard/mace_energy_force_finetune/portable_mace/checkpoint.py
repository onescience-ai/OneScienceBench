"""Construct the local MACE implementation from a portable JSON configuration."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from e3nn import o3

from .mace import ScaleShiftMACE
from .mace_block import RealAgnosticResidualInteractionBlock


FORMAT_NAME = "mace-portable-state-dict-v1"
INTERACTION_CLASSES = {
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
}


def _irreps(value: list[list[Any]]) -> o3.Irreps:
    return o3.Irreps(
        [(int(multiplicity), (int(irrep[0]), int(irrep[1]))) for multiplicity, irrep in value]
    )


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def build_model_from_config(
    config_path: str | Path,
    dtype: torch.dtype | None = None,
) -> ScaleShiftMACE:
    document = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if document.get("format") != FORMAT_NAME:
        raise ValueError(f"unsupported portable MACE config: {document.get('format')!r}")
    config = dict(document["model_config"])
    interaction_cls = INTERACTION_CLASSES[config.pop("interaction_cls")]
    interaction_cls_first = INTERACTION_CLASSES[config.pop("interaction_cls_first")]
    config.update(
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls_first,
        hidden_irreps=_irreps(config["hidden_irreps"]),
        MLP_irreps=_irreps(config["MLP_irreps"]),
        atomic_energies=np.asarray(config["atomic_energies"]),
        gate=torch.nn.functional.silu,
    )
    if dtype is None:
        dtype = torch.get_default_dtype()
    with _default_dtype(dtype):
        return ScaleShiftMACE(**config)


def load_portable_checkpoint(
    config_path: str | Path,
    weights_path: str | Path,
    device: str | torch.device = "cpu",
) -> ScaleShiftMACE:
    payload = torch.load(weights_path, map_location=device, weights_only=True)
    if payload.get("format") != FORMAT_NAME:
        raise ValueError(f"unsupported portable MACE checkpoint: {payload.get('format')!r}")
    state_dict = payload["state_dict"]
    dtype = next(
        value.dtype for value in state_dict.values() if value.is_floating_point()
    )
    model = build_model_from_config(config_path, dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)
