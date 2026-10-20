"""Full-mesh inference for one paper-first GeoMPNN repetition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config_utils import load_config
from dataset import (
    AirfransRepository,
    NormalizationStats,
    TARGETS,
    make_inference_pair,
)
from model import build_four_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to repeat_XX/ckpt.pt or the repeat_XX directory.",
    )
    parser.add_argument("--split", choices=["test", "test_ood"], default="test")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument(
        "--max-simulations",
        type=int,
        default=None,
        help="Debug-only prefix limit; omit for paper evaluation.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {requested} to cpu", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = select_device(config["runtime"].get("device", "cuda:0"))

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "ckpt.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_config = state.get("config", config)
    if checkpoint_config["model"] != config["model"]:
        raise ValueError("checkpoint model configuration does not match --config")

    normalization_ref = Path(
        state.get("normalization_path", "../preprocessing/normalization.npz")
    )
    normalization_path = (
        normalization_ref
        if normalization_ref.is_absolute()
        else (checkpoint.parent / normalization_ref).resolve()
    )
    stats = NormalizationStats.load(normalization_path)

    repository = AirfransRepository(config["data"])
    names = repository.split_names(args.split)
    expected_key = (
        "test_id_simulations" if args.split == "test" else "test_ood_simulations"
    )
    expected = int(config["experiment"][expected_key])
    if len(names) != expected:
        raise RuntimeError(
            f"resolved {args.split} has {len(names)} simulations; paper requires {expected}"
        )
    if args.max_simulations is not None:
        if args.max_simulations <= 0:
            raise ValueError("--max-simulations must be positive")
        names = names[: args.max_simulations]

    models = build_four_models(
        config["model"],
        knn_query_chunk_size=int(config["runtime"]["knn_query_chunk_size"]),
    ).to(device)
    models.load_state_dict(state["models"] if "models" in state else state)
    models.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "paper": config["experiment"]["paper"],
        "checkpoint": str(checkpoint.resolve()),
        "normalization": str(normalization_path),
        "split": args.split,
        "simulations": [],
    }

    with torch.no_grad():
        for index, name in enumerate(names, start=1):
            graph = stats.transform(repository.load_raw_graph(name), copy=False)
            pair = {
                graph_name: value.to(device)
                for graph_name, value in make_inference_pair(graph).items()
            }
            normalized = torch.cat(
                [models[field](pair) for field in TARGETS],
                dim=1,
            ).cpu().numpy()
            physical = stats.outputs_to_physical(normalized)
            if not np.isfinite(physical).all():
                raise FloatingPointError(f"{name}: prediction contains NaN or infinity")

            payload = {
                "position": graph.pos.numpy(),
                "surface": graph.surf.numpy(),
                **{
                    field: physical[:, field_index]
                    for field_index, field in enumerate(TARGETS)
                },
            }
            output_path = output_dir / f"{name}.npz"
            np.savez_compressed(output_path, **payload)
            manifest["simulations"].append(
                {"name": name, "points": len(graph), "file": output_path.name}
            )
            print(
                f"{args.split}: {index}/{len(names)} {name} points={len(graph)}",
                flush=True,
            )
            del pair, normalized, physical
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved {len(names)} predictions to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
