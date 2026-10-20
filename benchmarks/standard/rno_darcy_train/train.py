"""Train and evaluate the paper-configured RNO on Darcy Flow."""

from __future__ import annotations

import argparse
import copy
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from dataset import ScalarNormalizer, build_darcy_datasets
from model import build_model
from utils import (
    RelativeL2Loss,
    atomic_torch_save,
    central_difference,
    count_parameters,
    enforce_zero_dirichlet,
    evaluate,
    load_checkpoint,
    load_config,
    resolve_device,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration")
    parser.add_argument("--data", default=None, help="Override data.root")
    parser.add_argument("--output", default=None, help="Override experiment.output_dir")
    parser.add_argument("--device", default=None, help="Override experiment.device, e.g. cuda:0 or cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--resume", default=None, help="Resume from a training checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate --checkpoint without training")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint used by --eval-only")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use 2 real training samples, 1 real test sample, and one epoch; not a paper result",
    )
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    if args.data is not None:
        config["data"]["root"] = args.data
    if args.output is not None:
        config["experiment"]["output_dir"] = args.output
    if args.device is not None:
        config["experiment"]["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.smoke_test:
        config["training"]["epochs"] = 1
        config["training"]["eval_every"] = 1
        config["training"]["checkpoint_every"] = 1
    return config


def make_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: Dict[str, Any],
    x_normalizer: ScalarNormalizer,
    y_normalizer: ScalarNormalizer,
    history: list[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "x_normalizer": x_normalizer.state_dict(),
        "y_normalizer": y_normalizer.state_dict(),
        "history": history,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    y_normalizer: ScalarNormalizer,
    device: torch.device,
    resolution: int,
    gradient_weight: float,
    max_grad_norm: Optional[float],
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    relative_sum = RelativeL2Loss(reduction="sum")
    totals = {"data_loss": 0.0, "gradient_loss": 0.0, "total_loss": 0.0, "samples": 0}
    spacing = 1.0 / resolution  # follows the released exp_darcy.py

    for positions, coefficients, normalized_targets in loader:
        positions = positions.to(device, non_blocking=True)
        coefficients = coefficients.to(device, non_blocking=True)
        normalized_targets = normalized_targets.to(device, non_blocking=True)
        batch_size = coefficients.shape[0]
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            normalized_prediction = model(positions, coefficients.unsqueeze(-1)).squeeze(-1)
            prediction = y_normalizer.decode(normalized_prediction)
            target = y_normalizer.decode(normalized_targets)
            data_loss = relative_sum(prediction, target)

            prediction_bc = enforce_zero_dirichlet(prediction, resolution)
            pred_grad_x, pred_grad_y = central_difference(prediction_bc, resolution, spacing)
            target_grad_x, target_grad_y = central_difference(target, resolution, spacing)
            gradient_loss = relative_sum(pred_grad_x, target_grad_x) + relative_sum(pred_grad_y, target_grad_y)
            total_loss = data_loss + gradient_weight * gradient_loss

        scaler.scale(total_loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        totals["data_loss"] += float(data_loss.detach().item())
        totals["gradient_loss"] += float(gradient_loss.detach().item())
        totals["total_loss"] += float(total_loss.detach().item())
        totals["samples"] += batch_size

    samples = max(1, totals["samples"])
    return {
        "data_relative_l2": totals["data_loss"] / samples,
        "gradient_relative_l2": totals["gradient_loss"] / samples,
        "total_loss": totals["total_loss"] / samples,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["experiment"]["device"]))
    output_dir = Path(config["experiment"]["output_dir"])
    if args.smoke_test:
        output_dir = output_dir / "smoke_test"
    logger = setup_logger(output_dir)

    logger.info("Loading real Darcy Flow MAT files")
    train_set, test_set, x_normalizer, y_normalizer, data_metadata = build_darcy_datasets(
        config, smoke_test=args.smoke_test
    )
    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=pin_memory,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=pin_memory,
    )
    y_normalizer.to(device)

    model = build_model(config).to(device)
    parameter_count = count_parameters(model)
    logger.info("Device: %s", device)
    logger.info("Trainable parameters: %d", parameter_count)
    logger.info("Data metadata: %s", data_metadata)

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        load_checkpoint(Path(args.checkpoint), model, device, strict=True)
        metrics = evaluate(model, test_loader, y_normalizer, device)
        save_json(metrics, output_dir / "evaluation.json")
        logger.info("Test metrics: %s", metrics)
        return

    training_cfg = config["training"]
    if str(training_cfg["optimizer"]).lower() != "adamw":
        raise ValueError("Paper reproduction currently supports optimizer=AdamW only")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(training_cfg["learning_rate"]),
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    use_amp = bool(training_cfg["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[Dict[str, Any]] = []
    start_epoch = 0

    if args.resume is not None:
        checkpoint = load_checkpoint(Path(args.resume), model, device, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"])
        logger.info("Resuming after epoch %d", start_epoch)

    last_test_metrics: Optional[Dict[str, float]] = None
    started = time.time()
    for epoch_index in range(start_epoch, epochs):
        epoch = epoch_index + 1
        epoch_started = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            y_normalizer=y_normalizer,
            device=device,
            resolution=int(config["data"]["resolution"]),
            gradient_weight=float(config["loss"]["gradient_weight"]),
            max_grad_norm=float(training_cfg["max_grad_norm"])
            if training_cfg["max_grad_norm"] is not None
            else None,
            use_amp=use_amp,
        )
        record: Dict[str, Any] = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - epoch_started,
            "train": train_metrics,
        }
        if epoch % int(training_cfg["eval_every"]) == 0 or epoch == epochs:
            last_test_metrics = evaluate(model, test_loader, y_normalizer, device)
            record["test"] = last_test_metrics
        history.append(record)
        logger.info("Epoch %d/%d | %s", epoch, epochs, record)
        save_json({"history": history}, output_dir / "history.json")

        payload = make_checkpoint(
            model, optimizer, scheduler, epoch, config, x_normalizer, y_normalizer, history
        )
        atomic_torch_save(payload, output_dir / "last.pt")
        if epoch % int(training_cfg["checkpoint_every"]) == 0 or epoch == epochs:
            atomic_torch_save(payload, output_dir / f"epoch_{epoch:04d}.pt")

    summary = {
        "status": "smoke_success" if args.smoke_test else "training_complete",
        "smoke_test": args.smoke_test,
        "scientific_claim": "none_smoke_only" if args.smoke_test else "full_config_run",
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seed": seed,
        "parameter_count": parameter_count,
        "data": data_metadata,
        "model": config["model"],
        "training": config["training"],
        "paper_reference": config["paper_reference"],
        "final_test": last_test_metrics,
        "elapsed_seconds": time.time() - started,
    }
    save_json(summary, output_dir / "summary.json")
    logger.info("Run complete: %s", summary)


if __name__ == "__main__":
    main()

