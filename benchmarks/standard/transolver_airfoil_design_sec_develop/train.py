"""Train Transolver on AirfRANS.

Usage:
    python train.py config.yaml

Secondary development implemented:
- Data pipeline is unchanged.
- Loss uses surface/volume + field-wise weighted MSE.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import load_dataset, load_names, sample_points
from transolver import Transolver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("--data", default=None, help="Override data.root in config")
    parser.add_argument("--output", default=None, help="Override training.output_dir")
    parser.add_argument("--device", default=None, help="Override training.device")
    return parser.parse_args()


def read_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def field_weighted_mse(pred, target, field_weights):
    """Field-wise MSE for [ux, uy, p, nut]."""
    velocity_mse = torch.mean((pred[..., 0:2] - target[..., 0:2]) ** 2)
    pressure_mse = torch.mean((pred[..., 2] - target[..., 2]) ** 2)
    viscosity_mse = torch.mean((pred[..., 3] - target[..., 3]) ** 2)

    weighted = (
        float(field_weights["velocity"]) * velocity_mse
        + float(field_weights["pressure"]) * pressure_mse
        + float(field_weights["viscosity"]) * viscosity_mse
    )
    return weighted, {
        "velocity_mse": velocity_mse,
        "pressure_mse": pressure_mse,
        "viscosity_mse": viscosity_mse,
    }


def compute_loss(pred, target, surface, loss_cfg):
    """Surface/volume + field-wise weighted MSE.

    L_total = L_volume + surface_weight * L_surface
    L_region = w_u * MSE(ux,uy) + w_p * MSE(p) + w_nut * MSE(nut)
    """
    surface = surface.bool()
    volume = ~surface
    field_weights = loss_cfg["field_weights"]

    surface_loss, surface_fields = field_weighted_mse(
        pred[surface], target[surface], field_weights
    )
    volume_loss, volume_fields = field_weighted_mse(
        pred[volume], target[volume], field_weights
    )

    total_loss = volume_loss + float(loss_cfg["surface_weight"]) * surface_loss

    metrics = {
        "loss": total_loss.detach(),
        "surface_loss": surface_loss.detach(),
        "volume_loss": volume_loss.detach(),
        "surface_velocity_mse": surface_fields["velocity_mse"].detach(),
        "surface_pressure_mse": surface_fields["pressure_mse"].detach(),
        "surface_viscosity_mse": surface_fields["viscosity_mse"].detach(),
        "volume_velocity_mse": volume_fields["velocity_mse"].detach(),
        "volume_pressure_mse": volume_fields["pressure_mse"].detach(),
        "volume_viscosity_mse": volume_fields["viscosity_mse"].detach(),
    }
    return total_loss, metrics


def train_epoch(model, dataset, optimizer, scheduler, device,
                sample_size: int, batch_size: int, loss_cfg: dict):
    model.train()
    sampled = [sample_points(data, sample_size) for data in dataset]
    loader = DataLoader(sampled, batch_size=batch_size, shuffle=True)

    sums = {}
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(data.x, data.pos)
        loss, metrics = compute_loss(pred, data.y, data.surf, loss_cfg)
        loss.backward()
        optimizer.step()
        scheduler.step()

        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.item())

    return {key: value / len(loader) for key, value in sums.items()}


@torch.no_grad()
def validate(model, dataset, device, sample_size: int,
             loss_cfg: dict, repeats: int):
    model.eval()
    repeat_results = []

    for _ in range(repeats):
        sampled = [sample_points(data, sample_size) for data in dataset]
        loader = DataLoader(sampled, batch_size=1, shuffle=False)
        sums = {}

        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.pos)
            _, metrics = compute_loss(pred, data.y, data.surf, loss_cfg)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())

        repeat_results.append({key: value / len(loader) for key, value in sums.items()})

    keys = repeat_results[0].keys()
    return {
        key: float(np.mean([result[key] for result in repeat_results]))
        for key in keys
    }


def main():
    args = parse_args()
    cfg = read_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    train_cfg = cfg["training"]

    data_root = Path(args.data or data_cfg["root"])
    output_dir = Path(args.output or train_cfg["output_dir"])
    device_name = args.device or train_cfg.get("device", "auto")
    device = torch.device(
        "cuda:0" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )

    set_seed(int(train_cfg["seed"]))

    train_names, val_names = load_names(
        data_root,
        task=data_cfg["task"],
        val_ratio=float(data_cfg["val_ratio"]),
    )

    print(f"Loading {len(train_names)} train simulations...")
    train_data, norm_stats = load_dataset(train_names, data_root, fit_stats=True)
    print(f"Loading {len(val_names)} validation simulations...")
    val_data = load_dataset(val_names, data_root, stats=norm_stats)

    model = Transolver(**model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    total_steps = (len(train_data) // batch_size + 1) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(train_cfg["lr"]),
        total_steps=total_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    print(f"device={device}")
    print(f"parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in tqdm(range(epochs), desc="train"):
        train_metrics = train_epoch(
            model=model,
            dataset=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            sample_size=int(train_cfg["sample_points"]),
            batch_size=batch_size,
            loss_cfg=loss_cfg,
        )

        record = {"epoch": epoch}
        record.update({f"train_{k}": float(v) for k, v in train_metrics.items()})

        val_interval = int(train_cfg["val_interval"])
        if epoch == 0 or epoch % val_interval == val_interval - 1:
            val_metrics = validate(
                model=model,
                dataset=val_data,
                device=device,
                sample_size=int(train_cfg["sample_points"]),
                loss_cfg=loss_cfg,
                repeats=int(train_cfg["val_repeats"]),
            )
            record.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
            print("\n", record)

        history.append(record)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    torch.save({
        "model_state_dict": model.state_dict(),
        "normalization": {
            "mean_in": norm_stats[0],
            "std_in": norm_stats[1],
            "mean_out": norm_stats[2],
            "std_out": norm_stats[3],
        },
        "config": cfg,
    }, output_dir / "checkpoint.pt")

    print(f"Saved checkpoint to {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
