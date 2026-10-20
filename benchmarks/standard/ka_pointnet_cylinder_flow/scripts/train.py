from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.KAPointNet import KAPointNet
from utils.config import ensure_dir, get_device, load_config, set_seed
from utils.data import build_dataloaders, denormalize_targets_torch, prepare_pointnet_cfd_data
from utils.metrics import relative_l2_by_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KA-PointNet on the CFD cylinder benchmark.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run a single forward/eval step without formal training.")
    parser.add_argument("--max-smoke-batches", type=int, default=1)
    return parser.parse_args()


def build_model(config: Dict) -> KAPointNet:
    model_cfg = config["model"]
    data_cfg = config["data"]
    return KAPointNet(
        input_channels=int(model_cfg["input_channels"]),
        output_channels=int(model_cfg["output_channels"]),
        network_scaling=float(model_cfg["network_scaling"]),
        poly_degree=int(model_cfg["poly_degree"]),
        alpha=float(model_cfg["alpha"]),
        beta=float(model_cfg["beta"]),
        num_points=int(data_cfg["num_points"]),
    )


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stats,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions_physical = []
    targets_physical = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            batch_size = inputs.size(0)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            predictions_physical.append(denormalize_targets_torch(outputs, stats).cpu().numpy())
            targets_physical.append(denormalize_targets_torch(targets, stats).cpu().numpy())

    predictions = np.concatenate(predictions_physical, axis=0)
    targets = np.concatenate(targets_physical, axis=0)
    metrics = relative_l2_by_channel(predictions, targets)
    average_loss = total_loss / max(total_examples, 1)
    return average_loss, metrics


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_smoke_test(config: Dict, device: torch.device, max_smoke_batches: int) -> None:
    splits, stats = prepare_pointnet_cfd_data(config)
    loaders = build_dataloaders(config, splits)
    model = build_model(config).to(device)
    criterion = nn.MSELoss()

    for step_index, (inputs, targets) in enumerate(loaders["train"]):
        if step_index >= max_smoke_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        step_seconds = time.perf_counter() - step_started
        outputs_physical = denormalize_targets_torch(outputs, stats).detach().cpu().numpy()
        targets_physical = denormalize_targets_torch(targets, stats).cpu().numpy()
        metrics = relative_l2_by_channel(outputs_physical, targets_physical)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        print(
            "SMOKE_TEST",
            json.dumps(
                {
                    "inputs_shape": list(inputs.shape),
                    "outputs_shape": list(outputs.shape),
                    "loss": float(loss.item()),
                    "rel_l2": metrics,
                    "forward_backward_seconds": step_seconds,
                    "peak_memory_bytes": peak_memory_bytes,
                },
                ensure_ascii=True,
            ),
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))
    device = get_device(args.device)

    if args.smoke_test:
        run_smoke_test(config, device, args.max_smoke_batches)
        return

    splits, stats = prepare_pointnet_cfd_data(config)
    loaders = build_dataloaders(config, splits)

    model = build_model(config).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable_parameters}")
    training_cfg = config["training"]
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        betas=tuple(float(value) for value in training_cfg["betas"]),
        eps=float(training_cfg["eps"]),
    )

    checkpoint_path = project_path(config["experiment"]["checkpoint_path"])
    ensure_dir(checkpoint_path.parent)
    results_dir = ensure_dir(project_path(config["experiment"]["results_dir"]))

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = training_cfg.get("early_stopping_patience")
    history = []

    max_epochs = int(training_cfg["max_epochs"])
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        total_examples = 0

        for inputs, targets in loaders["train"]:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = epoch_loss / max(total_examples, 1)
        val_loss, val_metrics = evaluate(model, loaders["validation"], criterion, device, stats)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rel_u": val_metrics["u"],
            "val_rel_v": val_metrics["v"],
            "val_rel_p": val_metrics["p"],
            "val_rel_mean": val_metrics["mean"],
        }
        history.append(epoch_record)
        print(
            f"Epoch [{epoch + 1}/{max_epochs}] "
            f"train_loss={train_loss:.8f} "
            f"val_loss={val_loss:.8f} "
            f"val_rel_u={val_metrics['u']:.8f} "
            f"val_rel_v={val_metrics['v']:.8f} "
            f"val_rel_p={val_metrics['p']:.8f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "normalization_stats": stats.to_dict(),
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if patience is not None and int(patience) > 0 and epochs_without_improvement >= int(patience):
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    history_path = results_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
