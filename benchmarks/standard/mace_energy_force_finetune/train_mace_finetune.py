"""Training entry points for real MACE-MP-0 energy-force fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from mace_energy_force_finetune import (
    MACEEnergyForceFineTuner,
    ScaleShiftMACE,
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


def build_model(config: dict[str, Any]) -> ScaleShiftMACE:
    config_path, weights_path = foundation_paths(config)
    return load_foundation_model(config_path, weights_path, device="cpu")


def build_trainer(
    config: dict[str, Any],
    model: ScaleShiftMACE,
    output_dir: str | Path,
) -> MACEEnergyForceFineTuner:
    return MACEEnergyForceFineTuner(
        model=model,
        output_dir=output_dir,
        **dict(config["training"]),
    )


def train(
    config: dict[str, Any],
    train_batches: Iterable[Any],
    output_dir: str | Path,
) -> dict[str, object]:
    model = build_model(config)
    trainer = build_trainer(config, model, output_dir)
    result = trainer.fit(train_batches)
    return {
        "model": model,
        "history": result["history"],
        "metrics_path": result["metrics_path"],
        "checkpoint_path": result["checkpoint_path"],
    }


__all__ = [
    "load_config",
    "foundation_paths",
    "build_model",
    "build_trainer",
    "train",
]
