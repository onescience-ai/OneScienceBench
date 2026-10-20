"""Training entry points for the real MACE-MP-0 affine adapter."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from mace_element_affine_adapter import (
    MACEElementAffineAdapter,
    load_foundation_model,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration must be a mapping")
    return config


def foundation_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    foundation = dict(config["foundation_model"])
    return Path(foundation["config_path"]), Path(foundation["weights_path"])


def build_model(config: dict[str, Any]) -> MACEElementAffineAdapter:
    config_path, weights_path = foundation_paths(config)
    base = load_foundation_model(config_path, weights_path, device="cpu")
    return MACEElementAffineAdapter(
        model=base,
        atomic_numbers=base.atomic_numbers,
        **dict(config["adapter"]),
    )


def train_step(
    model: MACEElementAffineAdapter,
    reference: Any,
    optimizer: torch.optim.Optimizer,
) -> float:
    data = (
        reference.to_dict()
        if callable(getattr(reference, "to_dict", None))
        else dict(reference)
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(data, training=True, compute_force=True)
    loss = torch.mean(torch.square(prediction["energy"] - reference["energy"]))
    loss = loss + torch.mean(
        torch.square(prediction["forces"] - reference["forces"])
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train(
    config: dict[str, Any],
    batches: Iterable[Any],
    output_dir: str | Path,
) -> dict[str, object]:
    model = build_model(config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        **dict(config["training"]),
    )
    history = [train_step(model, batch, optimizer) for batch in batches]
    if not history:
        raise ValueError("batches must not be empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "mace_element_affine_adapter.pt"
    metrics_path = output_dir / "metrics.json"
    trainable_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    checkpoint = {
        "format": "mace-element-affine-adapter-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "trainable_parameter_names": trainable_parameter_names,
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    metrics_path.write_text(
        json.dumps({"history": history}, indent=2),
        encoding="utf-8",
    )
    return {
        "model": model,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
    }


__all__ = [
    "load_config",
    "foundation_paths",
    "build_model",
    "train_step",
    "train",
]
