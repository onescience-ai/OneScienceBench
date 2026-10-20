from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.io import read

from esen_inference import evaluate_structure, load_checkpoint_payload, load_model


def load_config(path="esen_inference.yaml"):
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("configuration must be a mapping")
    return value


def load_structures(source):
    if isinstance(source, Atoms):
        result = [source]
    elif isinstance(source, (str, Path)):
        result = read(str(source), index=":")
        if isinstance(result, Atoms):
            result = [result]
    elif isinstance(source, (list, tuple)):
        result = list(source)
    else:
        raise TypeError("source must be an ASE path, Atoms, or sequence")
    if not result or not all(isinstance(item, Atoms) for item in result):
        raise ValueError("at least one ASE structure is required")
    return result


def evaluate(structures, checkpoint, jd_path, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested accelerator is unavailable")
    payload = load_checkpoint_payload(checkpoint)
    model = load_model(checkpoint, device=device, jd_path=jd_path)
    values = [evaluate_structure(model, item, payload, device) for item in load_structures(structures)]
    return {
        "energies": np.asarray([float(item["energy"].reshape(-1)[0].detach().cpu()) for item in values]),
        "forces": [item["forces"].detach().cpu().numpy() for item in values],
        "stresses": np.stack([item["stress"].detach().cpu().numpy()[0] for item in values]),
    }


def save_predictions(path, result):
    records = []
    for index, energy in enumerate(result["energies"]):
        records.append({"index": index, "energy": float(energy), "forces": np.asarray(result["forces"][index]).tolist(), "stress": np.asarray(result["stresses"][index]).tolist()})
    Path(path).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="esen_inference.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    result = evaluate(args.input, args.checkpoint or config["checkpoint"], config["jd_path"], args.device)
    save_predictions(args.output, result)


if __name__ == "__main__":
    main()
