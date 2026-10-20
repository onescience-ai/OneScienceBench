from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import build_ns2d_batch
from train import build_model, load_config


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1).mean() / torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1).mean())


def run_inference(config: dict, output_dir: Path) -> dict[str, float]:
    torch.manual_seed(int(config["experiment"]["seed"]))
    model = build_model(config)
    weight_path = output_dir / config["experiment"]["checkpoint_path"]
    if weight_path.is_file():
        model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=True)
    data_cfg = config["data"]
    batch = build_ns2d_batch(
        batch_size=int(config["inference"]["batch_size"]),
        total_len=int(data_cfg["total_len"]),
        grid_size=int(data_cfg["grid_size"]),
        viscosity=float(data_cfg["viscosity"]),
        seed=int(config["experiment"]["seed"]) + 7,
    )
    model.eval()
    with torch.no_grad():
        pred = model(batch["sequence"], batch["times"], input_len=int(data_cfg["input_len"]))
    metric = {
        "relative_l2": relative_l2(pred, batch["target"]),
        "mse": float(torch.mean((pred - batch["target"]) ** 2).item()),
    }
    results_dir = output_dir / config["experiment"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")
    np.savez(results_dir / "predictions.npz", pred=pred.detach().cpu().numpy(), target=batch["target"].detach().cpu().numpy())
    print("relative_l2={relative_l2:.8f} mse={mse:.8f}".format(**metric))
    return metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    run_inference(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
