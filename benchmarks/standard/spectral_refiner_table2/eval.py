"""Evaluate saved SFNO and Spectral-Refiner checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from train import (
    ROOT,
    build_model,
    build_refiner,
    forcing_field,
    load_config,
    load_test_data,
    output_path,
    relative_l2,
    smoke_config,
)
from model import SobolevLoss


def load_model_state(module: torch.nn.Module, checkpoint: str | Path) -> None:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    module.load_state_dict(state, strict=True)


def checkpoint_path_from_args(
    config: dict[str, Any], checkpoint: str | Path | None, output_dir: Path | None = None
) -> Path:
    if checkpoint is not None:
        return Path(checkpoint).expanduser().resolve()
    return output_path(config, "refined_checkpoint", output_dir)


def evaluate(
    config: dict[str, Any],
    smoke: bool = False,
    output_dir: Path | None = None,
    checkpoint: str | Path | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    active = smoke_config(config) if smoke else copy.deepcopy(config)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, targets = load_test_data(active, smoke=smoke)
    if not smoke:
        index = int(active["fine_tuning"].get("sample_index", 0)) % inputs.shape[0]
        inputs, targets = inputs[index : index + 1], targets[index : index + 1]
    inputs, targets = inputs.to(device), targets.to(device)

    model = build_model(active).to(device=device, dtype=inputs.dtype).eval()
    load_model_state(model, output_path(active, "base_checkpoint", output_dir))
    model.add_latent_hook("reduction")
    with torch.no_grad():
        model(inputs, out_steps=targets.shape[-1])
        latent = model.latent_tensors["reduction"].detach()

    refiner = build_refiner(active, model, inputs.shape[0], inputs.shape[1], inputs.dtype, device)
    load_model_state(refiner, checkpoint_path_from_args(active, checkpoint, output_dir))
    forcing = forcing_field(inputs.shape[1], device, inputs.dtype).unsqueeze(0)
    if inputs.shape[0] > 1:
        forcing = forcing.expand(inputs.shape[0], -1, -1)
    loss_fn = SobolevLoss(
        n_grid=inputs.shape[1],
        relative=False,
        time_average=True,
        norm_order=float(active["fine_tuning"]["norm_order"]),
        alpha=float(active["fine_tuning"]["alpha"]),
        freq_cutoff=inputs.shape[1] // 2 + 1,
    ).to(device=device, dtype=inputs.dtype)
    with torch.no_grad():
        result = refiner(latent, inputs, forcing, out_steps=targets.shape[-1])
        metrics = {
            "relative_l2_error": float(relative_l2(result["w"], targets)),
            "residual_hminus1": float(loss_fn(result["residual"])),
        }
    metrics_path = output_path(active, "metrics", output_dir)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"relative_l2_error={metrics['relative_l2_error']:.8e}")
    print(f"residual_hminus1={metrics['residual_hminus1']:.8e}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        load_config(args.config),
        smoke=args.smoke,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
