"""ASE-facing MatterSim inference entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.io import read

from mattersim import evaluate_structure, load_model


DEFAULT_CONFIG = Path(__file__).with_name("mattersim_inference.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration must be a mapping")
    return config


def load_structures(source: str | Path | Atoms | list[Atoms]) -> list[Atoms]:
    if isinstance(source, Atoms):
        structures = [source]
    elif isinstance(source, (str, Path)):
        structures = read(str(source), index=":")
        if isinstance(structures, Atoms):
            structures = [structures]
    elif isinstance(source, (list, tuple)):
        structures = list(source)
    else:
        raise TypeError("input must be an ASE path, Atoms, or sequence of Atoms")
    if not structures or not all(isinstance(item, Atoms) for item in structures):
        raise ValueError("input must contain at least one valid ASE structure")
    return structures


def evaluate(
    structures: str | Path | Atoms | list[Atoms],
    checkpoint: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible accelerator is unavailable")
    atoms_list = load_structures(structures)
    model = load_model(checkpoint, device=device)
    predictions = [evaluate_structure(model, atoms, device) for atoms in atoms_list]
    return {
        "energies": np.asarray(
            [float(item["energy"].detach().cpu()) for item in predictions],
            dtype=np.float64,
        ),
        "forces": [item["forces"].detach().cpu().numpy() for item in predictions],
        "stresses": np.stack(
            [item["stress"].detach().cpu().numpy() for item in predictions]
        ),
    }


def save_predictions(path: str | Path, result: dict[str, Any]) -> None:
    records = []
    for index, (energy, forces, stress) in enumerate(
        zip(result["energies"], result["forces"], result["stresses"], strict=True)
    ):
        records.append(
            {
                "index": index,
                "energy": float(energy),
                "forces": np.asarray(forces).tolist(),
                "stress": np.asarray(stress).tolist(),
            }
        )
    Path(path).write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone MatterSim inference")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or config["checkpoint"]
    save_predictions(
        args.output,
        evaluate(args.input, checkpoint=checkpoint, device=args.device),
    )


if __name__ == "__main__":
    main()
