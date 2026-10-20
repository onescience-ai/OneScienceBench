"""Evaluation helpers for a fine-tuned real MACE-MP-0 checkpoint."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from train_mace_finetune import build_model


def checkpoint_path_from_args(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(config.get("checkpoint_path", "mace_finetuned.pt"))


def load_model_state(
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> torch.nn.Module:
    model = build_model(config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
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
        if not isinstance(prediction, Mapping):
            raise TypeError("model output must be a mapping")
        ptr = data.get("ptr")
        if ptr is not None:
            atom_counts = ptr[1:] - ptr[:-1]
        else:
            atom_counts = torch.bincount(
                data["batch"].long(), minlength=prediction["energy"].numel()
            )
        energy_errors.append(
            torch.mean(
                torch.abs(prediction["energy"] - data["energy"])
                / atom_counts.to(prediction["energy"])
            )
        )
        force_errors.append(
            torch.mean(torch.abs(prediction["forces"] - data["forces"]))
        )
    if not energy_errors:
        raise ValueError("batches must not be empty")
    return {
        "energy_mae": float(torch.stack(energy_errors).mean()),
        "forces_mae": float(torch.stack(force_errors).mean()),
    }


__all__ = ["checkpoint_path_from_args", "load_model_state", "evaluate"]
