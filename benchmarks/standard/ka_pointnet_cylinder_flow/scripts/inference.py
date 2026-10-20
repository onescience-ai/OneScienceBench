from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_model
from utils.config import ensure_dir, get_device, load_config, set_seed
from utils.data import build_dataloaders, denormalize_targets_torch, prepare_pointnet_cfd_data
from utils.metrics import relative_l2_by_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KA-PointNet inference on a configured split.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default=None, choices=["train", "validation", "test"])
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))
    device = get_device(args.device)

    splits, stats = prepare_pointnet_cfd_data(config)
    loaders = build_dataloaders(config, splits)

    split_name = args.split or config["inference"]["split"]
    checkpoint_path = project_path(config["experiment"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, target in loaders[split_name]:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            predictions.append(denormalize_targets_torch(output, stats).cpu().numpy())
            targets.append(denormalize_targets_torch(target, stats).cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    metrics = relative_l2_by_channel(predictions, targets)

    split_bundle = splits[split_name]
    results_dir = ensure_dir(project_path(config["experiment"]["results_dir"]))
    prediction_path = results_dir / f"{split_name}_predictions.npz"
    np.savez_compressed(
        prediction_path,
        predictions=predictions,
        targets=targets,
        coordinates=split_bundle.coords_physical,
        sample_indices=split_bundle.indices,
    )
    metrics_path = results_dir / f"{split_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Inference split={split_name}")
    print(
        f"rel_l2_u={metrics['u']:.8f} "
        f"rel_l2_v={metrics['v']:.8f} "
        f"rel_l2_p={metrics['p']:.8f} "
        f"rel_l2_mean={metrics['mean']:.8f}"
    )
    print(f"Saved predictions to {prediction_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
