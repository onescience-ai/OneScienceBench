"""Evaluation helpers for the real MACE-MP-0 affine adapter."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from train_mace_adapter import build_model


def load_model_state(config: dict[str, Any], state_dict: dict[str, torch.Tensor]):
    model = build_model(config)
    model.load_state_dict(state_dict, strict=True)
    return model


def evaluate(model: torch.nn.Module, batches: Iterable[Any]) -> dict[str, float]:
    energy_errors = []
    force_errors = []
    for reference in batches:
        data = (
            reference.to_dict()
            if callable(getattr(reference, "to_dict", None))
            else dict(reference)
        )
        prediction = model(data, training=False, compute_force=True)
        energy_errors.append(
            torch.mean(torch.abs(prediction["energy"] - reference["energy"]))
        )
        force_errors.append(
            torch.mean(torch.abs(prediction["forces"] - reference["forces"]))
        )
    if not energy_errors:
        raise ValueError("batches must not be empty")
    return {
        "energy_mae": float(torch.stack(energy_errors).mean()),
        "forces_mae": float(torch.stack(force_errors).mean()),
    }


__all__ = ["load_model_state", "evaluate"]
