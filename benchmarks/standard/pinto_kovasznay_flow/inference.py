from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import build_kovasznay_batch
from train import build_model, load_config


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    errors = {}
    names = ["u", "v", "p"]
    values = []
    for idx, name in enumerate(names):
        numerator = torch.linalg.vector_norm(pred[..., idx] - target[..., idx])
        denominator = torch.linalg.vector_norm(target[..., idx]).clamp_min(1e-12)
        value = float((numerator / denominator).item())
        errors[f"relative_l2_{name}"] = value
        values.append(value)
    errors["relative_l2_mean"] = float(sum(values) / len(values))
    return errors


def run_inference(config: dict, output_dir: Path) -> dict[str, float]:
    torch.manual_seed(int(config["experiment"]["seed"]))
    model = build_model(config).eval()
    checkpoint = output_dir / config["experiment"]["checkpoint_path"]
    if checkpoint.is_file():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    batch = build_kovasznay_batch(
        [float(config["inference"]["nue"])],
        domain_samples=int(config["data"]["smoke_domain_samples"]),
        seq_len=int(config["data"]["smoke_seq_len"]),
        start=tuple(config["data"]["domain_start"]),
        stop=tuple(config["data"]["domain_stop"]),
        seed=int(config["experiment"]["seed"]) + 99,
    )
    with torch.no_grad():
        pred = model(batch["query_xy"], batch["boundary_xy"], batch["boundary_values"])
    metrics = relative_l2(pred, batch["target"])
    results_dir = output_dir / config["experiment"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / "predictions.npz",
        prediction=pred.detach().cpu().numpy(),
        target=batch["target"].detach().cpu().numpy(),
    )
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(" ".join(f"{key}={value:.8f}" for key, value in metrics.items()))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(load_config(args.config), Path(args.output_dir))


if __name__ == "__main__":
    main()
