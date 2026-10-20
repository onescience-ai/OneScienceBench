"""Train Transolver on the AirfRANS 2D airfoil flow-field task.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --data /path/to/AirfRANS --device cuda:0
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
    parser.add_argument("--config", default="config.yaml")
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


def field_loss(pred, target, surface, surface_weight: float):
    """Original-style volume MSE + lambda * surface MSE."""
    mse = (pred - target) ** 2
    surface_loss = mse[surface].mean()
    volume_loss = mse[~surface].mean()
    total_loss = volume_loss + surface_weight * surface_loss
    return total_loss, surface_loss, volume_loss


def train_epoch(model, dataset, optimizer, scheduler, device,
                sample_size: int, batch_size: int, surface_weight: float):
    model.train()
    sampled = [sample_points(data, sample_size) for data in dataset]
    loader = DataLoader(sampled, batch_size=batch_size, shuffle=True)

    running = np.zeros(3, dtype=np.float64)
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(data.x, data.pos)
        loss, surf_loss, vol_loss = field_loss(pred, data.y, data.surf, surface_weight)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += [loss.item(), surf_loss.item(), vol_loss.item()]

    return running / len(loader)


@torch.no_grad()
def validate(model, dataset, device, sample_size: int,
             surface_weight: float, repeats: int):
    model.eval()
    results = []
    for _ in range(repeats):
        sampled = [sample_points(data, sample_size) for data in dataset]
        loader = DataLoader(sampled, batch_size=1, shuffle=False)
        running = np.zeros(3, dtype=np.float64)
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.pos)
            loss, surf_loss, vol_loss = field_loss(pred, data.y, data.surf, surface_weight)
            running += [loss.item(), surf_loss.item(), vol_loss.item()]
        results.append(running / len(loader))
    return np.stack(results).mean(axis=0)


def main():
    args = parse_args()
    cfg = read_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
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
    # Matches the original Transolver AirfRANS schedule-length convention.
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
            surface_weight=float(train_cfg["surface_weight"]),
        )

        record = {
            "epoch": epoch,
            "train_loss": float(train_metrics[0]),
            "train_surface_mse": float(train_metrics[1]),
            "train_volume_mse": float(train_metrics[2]),
        }

        val_interval = int(train_cfg["val_interval"])
        if epoch == 0 or epoch % val_interval == val_interval - 1:
            val_metrics = validate(
                model=model,
                dataset=val_data,
                device=device,
                sample_size=int(train_cfg["sample_points"]),
                surface_weight=float(train_cfg["surface_weight"]),
                repeats=int(train_cfg["val_repeats"]),
            )
            record.update({
                "val_loss": float(val_metrics[0]),
                "val_surface_mse": float(val_metrics[1]),
                "val_volume_mse": float(val_metrics[2]),
            })
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
