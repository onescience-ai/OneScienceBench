"""CLI for the standalone NequIP inference task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from nequip_inference import evaluate_structure, load_model


def load_config(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_structures(source):
    from ase import Atoms
    from ase.io import read

    if isinstance(source, Atoms):
        return [source]
    structures = read(str(source), index=":")
    return list(structures) if isinstance(structures, list) else [structures]


def evaluate(model, structures, device: str = "cpu"):
    records = []
    for index, atoms in enumerate(structures):
        prediction = evaluate_structure(model, atoms, device)
        records.append(
            {
                "structure_index": index,
                "energy_ev": prediction["energy"].cpu().tolist(),
                "forces_ev_per_angstrom": prediction["forces"].cpu().tolist(),
                "stress_ev_per_angstrom3": prediction["stress"].cpu().tolist(),
            }
        )
    return records


def save_predictions(records, output_path: str | Path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("nequip_inference.yaml")))
    parser.add_argument("--checkpoint")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = config["paths"]
    checkpoint = args.checkpoint or paths["checkpoint"]
    source = args.input or paths["input"]
    output = args.output or paths["output"]
    device = args.device or config["inference"]["device"]
    model, _, _ = load_model(checkpoint, device)
    records = evaluate(model, load_structures(source), device)
    save_predictions(records, output)
    return records


if __name__ == "__main__":
    main()
