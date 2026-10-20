"""Checkpoint, data and model APIs for NequIP energy-force fine-tuning."""

from __future__ import annotations

from pathlib import Path

from portable_nequip.runtime import (
    build_energy_force_batch,
    evaluation_model_state,
    load_checkpoint_payload,
    load_model_from_checkpoint,
    load_structures,
    predict_energy_forces,
    raw_model_state,
)


CHECKPOINT_SHA256 = "3ab4d7d891a36ed6346e661003fc30197d4b615171ccaec92cdd6b7bb1b4049b"
DATA_SHA256 = "6671f9b28d37484379f51f1deb23a6f4b165ed628467ee6bffa0648b6d50a5b5"


def load_training_structures(path: str | Path):
    return load_structures(path)


def build_finetune_batch(path: str | Path, index: int = 0, device: str = "cpu"):
    return build_energy_force_batch(path, index, device)


def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    return load_model_from_checkpoint(checkpoint_path, device)


__all__ = [
    "CHECKPOINT_SHA256",
    "DATA_SHA256",
    "build_finetune_batch",
    "evaluation_model_state",
    "load_checkpoint_payload",
    "load_model",
    "load_training_structures",
    "predict_energy_forces",
    "raw_model_state",
]
