import argparse
import json

import torch

from dataset import create_dataloaders
from metrics import GridMSEAccumulator, compute_sample_normalized_grid_mse
from utils import build_model, load_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate partially shared-decoder DeepCFD on the full test split."
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

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    checkpoint_model_config = checkpoint.get("config")
    if checkpoint_model_config is None:
        model_config = dict(config["model"])
    else:
        model_config = dict(checkpoint_model_config)
        # Checkpoints produced before partial sharing had three fully independent
        # decoder paths and no sharing field in their saved model configuration.
        model_config.setdefault("shared_decoder_stages", 0)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_config = dict(config["data"])
    data_config["directory"] = resolve_path(data_config["directory"])
    _, test_loader, _ = create_dataloaders(**data_config)

    accumulator = GridMSEAccumulator(device)
    grid_shape = None
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            current_shape = (int(targets.shape[2]), int(targets.shape[3]))
            if grid_shape is None:
                grid_shape = current_shape
            elif grid_shape != current_shape:
                raise RuntimeError(
                    f"Mixed grid shapes are not supported: {grid_shape} and {current_shape}"
                )
            predictions = model(inputs)
            batch_metrics = compute_sample_normalized_grid_mse(
                predictions,
                targets,
            )
            accumulator.update_from_metrics(batch_metrics, int(targets.shape[0]))

    mse_metrics = accumulator.compute()
    metrics = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("selection_metric", "Total_MSE"),
        "val_loss": checkpoint.get("val_loss"),
        "num_test_samples": mse_metrics.pop("num_samples"),
        "grid_shape": {
            "height": grid_shape[0],
            "width": grid_shape[1],
            "points": grid_shape[0] * grid_shape[1],
        },
        "metric_definition": (
            "For each sample and channel, squared error is averaged over the HxW "
            "grid; channel MSE is then averaged over samples; Total_MSE is the "
            "sum of Ux_MSE, Uy_MSE and p_MSE."
        ),
        **mse_metrics,
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Test samples: {metrics['num_test_samples']}")
    print(
        "MSE [Ux, Uy, p, Total]: "
        f"{metrics['Ux_MSE']:.6e}, {metrics['Uy_MSE']:.6e}, "
        f"{metrics['p_MSE']:.6e}, {metrics['Total_MSE']:.6e}"
    )
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
