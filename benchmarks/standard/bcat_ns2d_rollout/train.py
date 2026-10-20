from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from data import build_ns2d_batch
from model import BCATLite


def load_config(path: str | Path = "config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict) -> BCATLite:
    model_cfg = config["model"]
    data_cfg = config["data"]
    return BCATLite(
        grid_size=int(data_cfg["grid_size"]),
        channels=int(data_cfg["channels"]),
        patch_num=int(model_cfg["patch_num"]),
        dim_emb=int(model_cfg["dim_emb"]),
        n_head=int(model_cfg["n_head"]),
        n_layer=int(model_cfg["n_layer"]),
        dim_ffn=int(model_cfg["dim_ffn"]),
        dropout=float(model_cfg["dropout"]),
        max_time_len=max(20, int(data_cfg["total_len"])),
    )


def mse_rollout_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def train_one_step(config: dict, output_dir: Path) -> dict[str, float]:
    torch.manual_seed(int(config["experiment"]["seed"]))
    model = build_model(config)
    data_cfg = config["data"]
    batch = build_ns2d_batch(
        batch_size=int(config["training"]["batch_size"]),
        total_len=int(data_cfg["total_len"]),
        grid_size=int(data_cfg["grid_size"]),
        viscosity=float(data_cfg["viscosity"]),
        seed=int(config["experiment"]["seed"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    optimizer.zero_grad()
    pred = model(batch["sequence"], batch["times"], input_len=int(data_cfg["input_len"]))
    loss = mse_rollout_loss(pred, batch["target"])
    loss.backward()
    optimizer.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / config["experiment"]["checkpoint_path"]
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weight_path)
    results_dir = output_dir / config["experiment"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "train_loss": float(loss.detach().item()),
        "mse": float(loss.detach().item()),
    }
    (results_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("train_loss={train_loss:.8f} mse={mse:.8f}".format(**history))
    return history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train_one_step(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
