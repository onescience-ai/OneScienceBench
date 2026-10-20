"""Data and model APIs for deterministic NequIP training from scratch."""

from __future__ import annotations

from pathlib import Path

from portable_nequip.model import build_nequip_model
from portable_nequip.runtime import (
    build_energy_force_batch,
    load_structures,
    predict_energy_forces,
)


DATA_SHA256 = "6671f9b28d37484379f51f1deb23a6f4b165ed628467ee6bffa0648b6d50a5b5"


def load_training_structures(path: str | Path):
    return load_structures(path)


def build_training_batch(path: str | Path, index: int = 0, device: str = "cpu"):
    return build_energy_force_batch(path, index, device)


def build_model(seed: int = 456, device: str = "cpu"):
    return build_nequip_model(seed=seed).to(device)


__all__ = [
    "DATA_SHA256",
    "build_model",
    "build_training_batch",
    "load_training_structures",
    "predict_energy_forces",
]
