"""Train the multi-scale, physics-informed RNO on Darcy Flow."""

from __future__ import annotations

import argparse
import copy
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dataset import ScalarNormalizer, build_darcy_datasets
from model import build_model
from utils import (
    RelativeL2Loss,
    atomic_torch_save,
    central_difference,
    count_parameters,
    curriculum_scale,
    darcy_pde_residual_loss,
    enforce_zero_dirichlet,
    evaluate,
    load_checkpoint,
    load_config,
    resolve_device,
    save_json,
    set_seed,
    setup_logger,
    zero_dirichlet_boundary_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None, help="Override data.root")
    parser.add_argument("--output", default=None, help="Override experiment.output_dir")
    parser.add_argument("--device", default=None, help="Override experiment.device")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run one update on two real training samples and evaluate one real test sample",
    )
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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
        config["training"].update(epochs=1, eval_every=1, checkpoint_every=1)
    return config


def make_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: dict[str, Any],
    x_normalizer: ScalarNormalizer,
    y_normalizer: ScalarNormalizer,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
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
    scaler: torch.amp.GradScaler,
    x_normalizer: ScalarNormalizer,
    y_normalizer: ScalarNormalizer,
    device: torch.device,
    resolution: int,
    loss_config: dict[str, Any],
    epoch_index: int,
    epochs: int,
    max_grad_norm: float | None,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    relative_sum = RelativeL2Loss(reduction="sum")
    spacing = float(loss_config.get("grid_spacing", 1.0 / resolution))
    scale = curriculum_scale(
        epoch_index, epochs,
        float(loss_config["curriculum_start_fraction"]),
        float(loss_config["curriculum_end_fraction"]),
    )
    pde_weight = scale * float(loss_config["pde_weight_max"])
    boundary_weight = scale * float(loss_config["boundary_weight_max"])
    gradient_weight = float(loss_config["gradient_weight"])
    totals = {name: 0.0 for name in ("data", "gradient", "pde", "boundary", "total")}
    totals["samples"] = 0

    for positions, fine_coefficients, coarse_coefficients, normalized_targets, importance in loader:
        positions = positions.to(device, non_blocking=True)
        fine_coefficients = fine_coefficients.to(device, non_blocking=True)
        coarse_coefficients = coarse_coefficients.to(device, non_blocking=True)
        normalized_targets = normalized_targets.to(device, non_blocking=True)
        importance = importance.to(device, non_blocking=True)
        batch_size = fine_coefficients.shape[0]
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            normalized_prediction = model(
                positions, fine_coefficients.unsqueeze(-1), coarse_coefficients.unsqueeze(-1)
            ).squeeze(-1)
            prediction = y_normalizer.decode(normalized_prediction)
            target = y_normalizer.decode(normalized_targets)
            coefficient = x_normalizer.decode(fine_coefficients)
            data_loss = relative_sum(prediction, target)
            prediction_bc = enforce_zero_dirichlet(prediction, resolution)
            pred_grad_x, pred_grad_y = central_difference(prediction_bc, resolution, spacing)
            target_grad_x, target_grad_y = central_difference(target, resolution, spacing)
            gradient_loss = relative_sum(pred_grad_x, target_grad_x) + relative_sum(pred_grad_y, target_grad_y)
            pde_loss = darcy_pde_residual_loss(
                prediction, coefficient, importance, resolution, spacing,
                forcing=float(loss_config.get("forcing", 1.0)),
                coefficient_average=str(loss_config.get("coefficient_average", "harmonic")),
            )
            boundary_loss = zero_dirichlet_boundary_loss(prediction, target, resolution)
            total_loss = (
                data_loss + gradient_weight * gradient_loss
                + batch_size * (pde_weight * pde_loss + boundary_weight * boundary_loss)
            )

        scaler.scale(total_loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        totals["data"] += float(data_loss.detach())
        totals["gradient"] += float(gradient_loss.detach())
        totals["pde"] += float(pde_loss.detach()) * batch_size
        totals["boundary"] += float(boundary_loss.detach()) * batch_size
        totals["total"] += float(total_loss.detach())
        totals["samples"] += batch_size

    samples = max(1, int(totals["samples"]))
    return {
        "data_relative_l2": totals["data"] / samples,
        "gradient_relative_l2": totals["gradient"] / samples,
        "pde_residual": totals["pde"] / samples,
        "boundary_loss": totals["boundary"] / samples,
        "total_loss": totals["total"] / samples,
        "physics_curriculum_scale": scale,
        "pde_weight": pde_weight,
        "boundary_weight": boundary_weight,
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

    train_set, test_set, x_normalizer, y_normalizer, metadata = build_darcy_datasets(
        config, smoke_test=args.smoke_test
    )
    generator = torch.Generator().manual_seed(seed)
    loader_args = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    x_normalizer.to(device)
    y_normalizer.to(device)
    model = build_model(config).to(device)
    parameter_count = count_parameters(model)
    logger.info("Device: %s | trainable parameters: %d", device, parameter_count)
    logger.info("Data metadata: %s", metadata)

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        checkpoint = load_checkpoint(Path(args.checkpoint), model, device, strict=True)
        metrics = evaluate(
            model, test_loader, y_normalizer, device,
            spacing=float(config["loss"].get("grid_spacing", 1.0 / int(config["data"]["resolution"]))),
        )
        save_json({
            **metrics,
            "split": "test",
            "data_config": config["data"],
            "data": metadata,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "parameter_count": parameter_count,
            "smoke_test": args.smoke_test,
        }, output_dir / "evaluation.json")
        logger.info("Test metrics: %s", metrics)
        return

    training_cfg = config["training"]
    if str(training_cfg["optimizer"]).lower() != "adamw":
        raise ValueError("Only optimizer=AdamW is currently implemented")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(training_cfg["learning_rate"]),
        epochs=epochs, steps_per_epoch=len(train_loader),
    )
    use_amp = bool(training_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), model, device, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"])

    last_test_metrics = None
    started = time.time()
    for epoch_index in range(start_epoch, epochs):
        epoch_started = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, x_normalizer, y_normalizer,
            device, int(config["data"]["resolution"]), config["loss"], epoch_index, epochs,
            float(training_cfg["max_grad_norm"]) if training_cfg["max_grad_norm"] is not None else None,
            use_amp,
        )
        epoch = epoch_index + 1
        record: dict[str, Any] = {
            "epoch": epoch, "elapsed_seconds": time.time() - epoch_started, "train": train_metrics
        }
        if epoch % int(training_cfg["eval_every"]) == 0 or epoch == epochs:
            last_test_metrics = evaluate(
                model, test_loader, y_normalizer, device,
                spacing=float(config["loss"].get("grid_spacing", 1.0 / int(config["data"]["resolution"]))),
            )
            record["test"] = last_test_metrics
        history.append(record)
        logger.info("Epoch %d/%d | %s", epoch, epochs, record)
        save_json({"history": history}, output_dir / "history.json")
        payload = make_checkpoint(model, optimizer, scheduler, epoch, config, x_normalizer, y_normalizer, history)
        atomic_torch_save(payload, output_dir / "last.pt")
        if epoch % int(training_cfg["checkpoint_every"]) == 0 or epoch == epochs:
            atomic_torch_save(payload, output_dir / f"epoch_{epoch:04d}.pt")

    if last_test_metrics is not None:
        save_json({
            **last_test_metrics,
            "split": "test",
            "data_config": config["data"],
            "data": metadata,
            "checkpoint": str((output_dir / "last.pt").resolve()),
            "checkpoint_epoch": epoch,
            "parameter_count": parameter_count,
            "smoke_test": args.smoke_test,
        }, output_dir / "evaluation.json")

    summary = {
        "status": "smoke_success" if args.smoke_test else "training_complete",
        "smoke_test": args.smoke_test,
        "scientific_claim": "none_smoke_only" if args.smoke_test else "secondary_development_run",
        "device": str(device), "platform": platform.platform(), "torch_version": torch.__version__,
        "seed": seed, "parameter_count": parameter_count, "data": metadata,
        "model": config["model"], "loss": config["loss"], "training": config["training"],
        "paper_reference": config.get("paper_reference"), "final_test": last_test_metrics,
        "elapsed_seconds": time.time() - started,
    }
    save_json(summary, output_dir / "summary.json")
    logger.info("Run complete: %s", summary)


if __name__ == "__main__":
    main()
