"""Train multiscale KNO with curriculum scheduled-sampling rollout loss."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dataset import NavierStokesDataModule
from model import build_model, load_legacy_weights
from utils import (
    atomic_json_dump,
    atomic_torch_save,
    autoregressive_rollout,
    composite_rollout_loss,
    curriculum_state,
    load_config,
    project_path,
    resolve_data_path,
    resolve_device,
    seed_everything,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kno_multiscale")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    spatial_shape: tuple[int, int],
    horizon: int,
    out_dim: int,
    teacher_forcing_ratio: float,
    loss_config: dict[str, Any],
    max_batches: int | None,
    max_grad_norm: float | None,
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in ("loss", "mse", "relative_l2", "spectral")}
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        pos, state, target = (batch[key].to(device) for key in ("pos", "x", "y"))
        optimizer.zero_grad(set_to_none=True)
        prediction = autoregressive_rollout(
            model,
            pos,
            state,
            target,
            horizon,
            out_dim,
            teacher_forcing_ratio,
            training=True,
        )
        loss, components = composite_rollout_loss(
            prediction, target, spatial_shape, horizon, out_dim, loss_config
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        for name, value in components.items():
            totals[name] += float(value.detach())
        batches += 1
    if batches == 0:
        raise RuntimeError("No training batches were processed")
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    spatial_shape: tuple[int, int],
    horizon: int,
    out_dim: int,
    loss_config: dict[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in ("loss", "mse", "relative_l2", "spectral")}
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        pos, state, target = (batch[key].to(device) for key in ("pos", "x", "y"))
        prediction = autoregressive_rollout(
            model, pos, state, target, horizon, out_dim, 0.0, training=False
        )
        loss, components = composite_rollout_loss(
            prediction, target, spatial_shape, horizon, out_dim, loss_config
        )
        totals["loss"] += float(loss)
        for name, value in components.items():
            totals[name] += float(value)
        batches += 1
    if batches == 0:
        raise RuntimeError("No validation batches were processed")
    return {name: value / batches for name, value in totals.items()}


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    spatial_shape: tuple[int, int],
    normalizer: dict[str, float] | None,
    epoch: int,
    best_relative_l2: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": config["model"],
        "data_config": config["datapipe"],
        "training_config": config["training"],
        "spatial_shape": spatial_shape,
        "normalizer": normalizer,
        "best_relative_l2": best_relative_l2,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    training = config["training"]
    if args.device:
        config["common"]["device"] = args.device
    if args.epochs:
        training["epochs"] = args.epochs
    if args.output:
        training["output_dir"] = args.output
    if args.init_checkpoint:
        training["init_checkpoint"] = args.init_checkpoint
    if args.smoke_test:
        training["epochs"] = 1
        training["max_train_batches"] = 1
        training["max_validation_batches"] = 1

    data_path = resolve_data_path(config, PROJECT_ROOT)
    output_dir = project_path(training["output_dir"], PROJECT_ROOT).resolve()
    logger = setup_logging(output_dir)
    seed_everything(int(config["common"]["seed"]))
    device = resolve_device(str(config["common"]["device"]))
    epochs = int(training["epochs"])
    curriculum = training["curriculum"]
    first_horizon, _ = curriculum_state(1, epochs, curriculum)
    data_module = NavierStokesDataModule(config["datapipe"], first_horizon)
    model = build_model(config["model"], data_module.spatial_shape).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["scheduler_step_size"]),
        gamma=float(training["scheduler_gamma"]),
    )
    start_epoch = 1
    best_relative_l2 = float("inf")
    history: list[dict[str, Any]] = []

    if args.resume:
        resume = torch.load(Path(args.resume).expanduser(), map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_relative_l2 = float(resume["best_relative_l2"])
        history = list(resume.get("history", []))
        logger.info("Resumed multiscale checkpoint %s", args.resume)
    else:
        legacy_path = project_path(training["init_checkpoint"], PROJECT_ROOT).resolve()
        legacy = torch.load(legacy_path, map_location="cpu", weights_only=False)
        load_legacy_weights(model, legacy)
        logger.info("Initialized low-frequency branch from %s", legacy_path)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info("device=%s data=%s parameters=%d", device, data_path, parameters)
    out_dim = int(config["datapipe"]["data"]["out_dim"])
    validation_horizon = int(config["datapipe"]["data"]["max_horizon"])
    max_train_batches = training.get("max_train_batches")
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_validation_batches = training.get("max_validation_batches")
    max_validation_batches = (
        None if max_validation_batches is None else int(max_validation_batches)
    )
    max_grad_norm = training.get("max_grad_norm")
    max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        horizon, teacher_ratio = curriculum_state(epoch, epochs, curriculum)
        data_module.set_train_horizon(horizon)
        train_metrics = run_epoch(
            model,
            data_module.train_dataloader(),
            optimizer,
            device,
            data_module.spatial_shape,
            horizon,
            out_dim,
            teacher_ratio,
            training["loss"],
            max_train_batches,
            max_grad_norm,
        )
        scheduler.step()
        validation_metrics = validate(
            model,
            data_module.validation_dataloader(),
            device,
            data_module.spatial_shape,
            validation_horizon,
            out_dim,
            training["loss"],
            max_validation_batches,
        )
        record = {
            "epoch": epoch,
            "rollout_horizon": horizon,
            "teacher_forcing_ratio": teacher_ratio,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        atomic_json_dump(history, output_dir / "history.json")
        logger.info(
            "epoch=%d horizon=%d teacher=%.3f train_loss=%.6e val_loss=%.6e "
            "val_relative_l2=%.6e seconds=%.1f",
            epoch,
            horizon,
            teacher_ratio,
            train_metrics["loss"],
            validation_metrics["loss"],
            validation_metrics["relative_l2"],
            record["seconds"],
        )
        if validation_metrics["relative_l2"] < best_relative_l2:
            best_relative_l2 = validation_metrics["relative_l2"]
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    data_module.spatial_shape,
                    data_module.get_normalizer_state(),
                    epoch,
                    best_relative_l2,
                    history,
                ),
                output_dir / "best.pt",
            )
        atomic_torch_save(
            checkpoint_payload(
                model,
                optimizer,
                scheduler,
                config,
                data_module.spatial_shape,
                data_module.get_normalizer_state(),
                epoch,
                best_relative_l2,
                history,
            ),
            output_dir / "last.pt",
        )

    atomic_json_dump(
        {
            "best_relative_l2": best_relative_l2,
            "epochs_completed": history[-1]["epoch"] if history else 0,
            "parameters": parameters,
            "curriculum_horizons": curriculum["horizons"],
            "output_dir": str(output_dir),
        },
        output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
