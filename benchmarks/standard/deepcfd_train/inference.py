import argparse
import json

import torch

from config_utils import load_config, resolve_path
from dataset import create_dataloaders
from utils import build_model


CHANNELS = ("Ux", "Uy", "p")


def compute_sample_normalized_grid_mse(predictions, targets):
    """Channel grid squared-error sums divided by the number of samples."""
    channel_mse = (predictions - targets).square().sum(dim=(0, 2, 3)) / predictions.shape[0]
    metrics = {
        name + "_MSE": float(value.detach().cpu())
        for name, value in zip(CHANNELS, channel_mse)
    }
    metrics["Total_MSE"] = sum(metrics.values())
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepCFD on the full test split."
    )
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config, config_path = load_config(args.config)
    inference_config = config["inference"]
    checkpoint_path = resolve_path(inference_config["checkpoint_path"])
    result_dir = resolve_path(inference_config["result_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint.get("config", config["model"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_config = dict(config["data"])
    data_config["directory"] = resolve_path(data_config["directory"])
    _, test_loader, _ = create_dataloaders(**data_config)

    sum_sq = torch.zeros(len(CHANNELS), device=device)
    sum_abs = torch.zeros(len(CHANNELS), device=device)
    sum_target_sq = torch.zeros(len(CHANNELS), device=device)
    value_count = 0
    sample_count = 0
    grid_shape = None
    metric_sums = {name: 0.0 for name in ("Ux_MSE", "Uy_MSE", "p_MSE", "Total_MSE")}

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            difference = output - targets
            batch_metrics = compute_sample_normalized_grid_mse(output, targets)
            for name, value in batch_metrics.items():
                metric_sums[name] += value * targets.shape[0]
            current_shape = (int(targets.shape[2]), int(targets.shape[3]))
            if grid_shape is None:
                grid_shape = current_shape
            elif grid_shape != current_shape:
                raise RuntimeError(
                    f"Mixed grid shapes are not supported: {grid_shape} and {current_shape}"
                )

            sum_sq += torch.sum(difference**2, dim=(0, 2, 3))
            sum_abs += torch.sum(torch.abs(difference), dim=(0, 2, 3))
            sum_target_sq += torch.sum(targets**2, dim=(0, 2, 3))
            value_count += targets.shape[0] * targets.shape[2] * targets.shape[3]
            sample_count += targets.shape[0]

    if value_count == 0:
        raise RuntimeError("Evaluation loader is empty")

    grid_points = grid_shape[0] * grid_shape[1]
    mse = sum_sq / value_count
    mae = sum_abs / value_count
    relative_l2 = torch.sqrt(sum_sq / torch.clamp(sum_target_sq, min=1.0e-12))
    paper_style_mse = sum_sq / sample_count

    metrics = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "num_test_samples": sample_count,
        "grid_shape": {
            "height": grid_shape[0],
            "width": grid_shape[1],
            "points": grid_points,
        },
        "mse_note": "mse and pixel_mse are per-pixel averages; paper_style_mse is sum-squared-error per sample and is closer to the DeepCFD paper Table 2 scale.",
        "mse": {name: float(value) for name, value in zip(CHANNELS, mse.cpu())},
        "pixel_mse": {
            name: float(value) for name, value in zip(CHANNELS, mse.cpu())
        },
        "paper_style_mse": {
            name: float(value)
            for name, value in zip(CHANNELS, paper_style_mse.cpu())
        },
        "mae": {name: float(value) for name, value in zip(CHANNELS, mae.cpu())},
        "relative_l2": {
            name: float(value) for name, value in zip(CHANNELS, relative_l2.cpu())
        },
        "total_mse": float(torch.sum(mse).cpu()),
        "paper_style_total_mse": float(torch.sum(paper_style_mse).cpu()),
    }

    metrics.update({name: value / sample_count for name, value in metric_sums.items()})
    metrics["paper_style_mse"] = {name: metrics[name + "_MSE"] for name in CHANNELS}
    metrics["paper_style_total_mse"] = metrics["Total_MSE"]

    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Test samples: {sample_count}")
    print(f"Pixel MSE [Ux, Uy, p]: {mse.cpu().numpy()}")
    print(f"Paper-style total MSE: {metrics['paper_style_total_mse']:.6f}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
