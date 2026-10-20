"""Evaluate an RNO-Darcy checkpoint on the configured test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import build_darcy_datasets
from model import build_model
from utils import count_parameters, evaluate, load_config, resolve_device, save_json, set_seed, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", default=None, help="Override data.root")
    parser.add_argument("--output", default=None, help="Override experiment.output_dir")
    parser.add_argument("--device", default=None, help="Override experiment.device")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Use two training samples for statistics and evaluate one test sample",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data is not None:
        config["data"]["root"] = args.data
    if args.output is not None:
        config["experiment"]["output_dir"] = args.output
    if args.device is not None:
        config["experiment"]["device"] = args.device
    set_seed(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["experiment"]["device"]))
    output_dir = Path(config["experiment"]["output_dir"])
    if args.smoke_test:
        output_dir = output_dir / "smoke_test"
    logger = setup_logger(output_dir, name="rno_darcy_evaluate")

    # Reuse the existing train-only normalization. Only test_set is evaluated;
    # reading training data here fits statistics, not a validation/model selection.
    _, test_set, _, y_normalizer, metadata = build_darcy_datasets(
        config, smoke_test=args.smoke_test
    )
    test_loader = DataLoader(
        test_set, batch_size=int(config["training"]["batch_size"]), shuffle=False,
        num_workers=int(config["data"]["num_workers"]), pin_memory=device.type == "cuda",
    )
    y_normalizer.to(device)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = evaluate(
        model, test_loader, y_normalizer, device,
        spacing=float(config["loss"].get("grid_spacing", 1.0 / int(config["data"]["resolution"]))),
    )
    report = {
        **metrics,
        "split": "test",
        "data_config": config["data"],
        "data": metadata,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "parameter_count": count_parameters(model),
        "smoke_test": args.smoke_test,
    }
    save_json(report, output_dir / "evaluation.json")
    logger.info("Test metrics: %s", metrics)
    logger.info("Evaluation saved to %s", output_dir / "evaluation.json")


if __name__ == "__main__":
    main()
