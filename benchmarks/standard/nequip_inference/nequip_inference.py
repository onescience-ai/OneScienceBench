"""Standalone NequIP inference implementation for the fixed Cu smoke checkpoint."""

from __future__ import annotations

from pathlib import Path

from portable_nequip.runtime import (
    evaluate_atoms,
    evaluation_model_state,
    load_checkpoint_payload,
    load_model_from_checkpoint,
    raw_model_state,
)


CHECKPOINT_SHA256 = "3ab4d7d891a36ed6346e661003fc30197d4b615171ccaec92cdd6b7bb1b4049b"


def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    """Build the local architecture and strictly load its complete EMA state."""
    model, payload, state = load_model_from_checkpoint(checkpoint_path, device)
    return model, payload, state


def evaluate_structure(model, atoms, device: str = "cpu"):
    """Return energy, forces and full 3x3 stress for one ASE structure."""
    return evaluate_atoms(model, atoms, device)


__all__ = [
    "CHECKPOINT_SHA256",
    "evaluate_structure",
    "evaluation_model_state",
    "load_checkpoint_payload",
    "load_model",
    "raw_model_state",
]
