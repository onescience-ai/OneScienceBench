"""Evaluate a trained RNO checkpoint on the Darcy Flow test split."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ScalarNormalizer, build_darcy_datasets
from model import build_model
from utils import (
    RelativeL2Loss,
    central_difference,
    count_parameters,
    enforce_zero_dirichlet,
    load_checkpoint,
    load_config,
    resolve_device,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint containing model_state_dict")
    parser.add_argument("--data", default=None, help="Override data.root")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--device", default=None, help="Override experiment.device")
    parser.add_argument("--smoke-test", action="store_true", help="Evaluate one real test sample")
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    if args.data is not None:
        config["data"]["root"] = args.data
    if args.device is not None:
        config["experiment"]["device"] = args.device
    return config


@torch.no_grad()
def evaluate_test_split(
    model: torch.nn.Module,
    loader: Iterable,
    normalizer: ScalarNormalizer,
    device: torch.device,
    resolution: int,
) -> Dict[str, float]:
    model.eval()
    relative_l2 = RelativeL2Loss(reduction="sum")
    totals = {
        "relative_l2": 0.0,
        "gradient_relative_l2": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "samples": 0,
        "points": 0,
    }
    spacing = 1.0 / resolution

    for positions, coefficients, targets in loader:
        positions = positions.to(device, non_blocking=True)
        coefficients = coefficients.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        normalized_prediction = model(positions, coefficients.unsqueeze(-1)).squeeze(-1)
        prediction = normalizer.decode(normalized_prediction)

        prediction_bc = enforce_zero_dirichlet(prediction, resolution)
        pred_grad_x, pred_grad_y = central_difference(prediction_bc, resolution, spacing)
        target_grad_x, target_grad_y = central_difference(targets, resolution, spacing)
        batch = targets.shape[0]
        totals["relative_l2"] += float(relative_l2(prediction, targets).item())
        totals["gradient_relative_l2"] += float(
            (relative_l2(pred_grad_x, target_grad_x) + relative_l2(pred_grad_y, target_grad_y)).item()
        )
        totals["mse"] += float(F.mse_loss(prediction, targets, reduction="sum").item())
        totals["mae"] += float(F.l1_loss(prediction, targets, reduction="sum").item())
        totals["samples"] += batch
        totals["points"] += targets.numel()

    if totals["samples"] == 0:
        raise RuntimeError("Evaluation loader is empty")
    return {
        "relative_l2": totals["relative_l2"] / totals["samples"],
        "gradient_relative_l2": totals["gradient_relative_l2"] / totals["samples"],
        "mse": totals["mse"] / totals["points"],
        "mae": totals["mae"] / totals["points"],
        "samples": int(totals["samples"]),
    }


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["experiment"]["device"]))
    output_path = (
        Path(args.output)
        if args.output
        else Path(config["experiment"]["output_dir"]) / "evaluation.json"
    )
    logger = setup_logger(output_path.parent, name="rno_darcy_evaluation")

    _, test_set, _, data_y_normalizer, metadata = build_darcy_datasets(
        config, smoke_test=args.smoke_test
    )
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(Path(args.checkpoint), model, device, strict=True)
    if "y_normalizer" in checkpoint:
        y_normalizer = ScalarNormalizer.from_state_dict(checkpoint["y_normalizer"])
    else:
        y_normalizer = data_y_normalizer
        logger.warning("Checkpoint has no y_normalizer; recomputed it from the training split")
    y_normalizer.to(device)

    loader = DataLoader(
        test_set,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_test_split(
        model,
        loader,
        y_normalizer,
        device,
        resolution=int(config["data"]["resolution"]),
    )
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device),
        "parameter_count": count_parameters(model),
        "data": metadata,
        "metrics": metrics,
        "metric_notes": {
            "relative_l2": "Mean per-sample physical-space relative L2; primary paper metric.",
            "gradient_relative_l2": "Sum of x/y mean per-sample gradient relative L2.",
            "mse": "Pointwise physical-space mean squared error.",
            "mae": "Pointwise physical-space mean absolute error.",
        },
        "smoke_test": bool(args.smoke_test),
    }
    save_json(result, output_path)
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Saved evaluation result to %s", output_path)


if __name__ == "__main__":
    main()
