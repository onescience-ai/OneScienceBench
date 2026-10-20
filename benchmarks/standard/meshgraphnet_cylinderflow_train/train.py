"""Train and evaluate a standalone MeshGraphNet on DeepMind CylinderFlow.

Examples:
    python train.py --config config.yaml
    python train.py --config config.yaml --smoke-test --device cpu
    torchrun --standalone --nproc_per_node=4 train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset import (
    NORMAL,
    OUTFLOW,
    CylinderFlowDataset,
    NormalizationStats,
    Trajectory,
    _encode_node_type,
    collate_graphs,
)
from model import MeshGraphNet
from utils import (
    load_checkpoint,
    load_config,
    resolve_device,
    save_checkpoint,
    set_seed,
    setup_logging,
    trainable_parameters,
    unwrapped_model,
    write_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", help="Override data.root")
    parser.add_argument("--output", help="Override training.output_dir")
    parser.add_argument("--device", help="Override training.device")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load one trajectory with two steps and run one train/validation epoch",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Evaluate autoregressive rollout using the best checkpoint after training",
    )
    return parser.parse_args()


def distributed_context(requested_device: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = resolve_device(requested_device, local_rank)
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if device.type == "cuda":
            torch.cuda.set_device(device)
    return rank, world_size, local_rank, device


def make_loader(
    dataset: CylinderFlowDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_graphs,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def node_loss(prediction: Tensor, target: Tensor, node_type: Tensor, config: dict[str, Any]) -> Tensor:
    mode = config.get("loss_nodes", "all")
    if mode == "normal_outflow":
        types = node_type.squeeze(-1)
        mask = types.eq(NORMAL) | types.eq(OUTFLOW)
        prediction, target = prediction[mask], target[mask]
    elif mode != "all":
        raise ValueError("training.loss_nodes must be 'all' or 'normal_outflow'")
    criterion = config.get("loss", "mse").lower()
    if criterion == "mse":
        return nn.functional.mse_loss(prediction, target)
    if criterion == "mae":
        return nn.functional.l1_loss(prediction, target)
    raise ValueError("training.loss must be 'mse' or 'mae'")


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def reduce_pair(total: float, count: int, device: torch.device, world_size: int) -> tuple[float, int]:
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), int(values[1].item())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
    training_config: dict[str, Any],
    epoch: int,
    logger: logging.Logger,
    rank: int,
    world_size: int,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    batches = 0
    amp = bool(training_config.get("amp", False)) and device.type == "cuda"
    log_interval = int(training_config.get("log_interval", 1000))

    for batch_index, graph in enumerate(loader, start=1):
        graph = graph.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            prediction = model(graph.node_features, graph.edge_features, graph.edge_index)
            loss = node_loss(prediction, graph.target, graph.node_type, training_config)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1
        total_loss += float(loss.detach())
        batches += 1
        if rank == 0 and batch_index % log_interval == 0:
            logger.info(
                "epoch=%d batch=%d/%d step=%d train_loss=%.8f lr=%.3e",
                epoch + 1,
                batch_index,
                len(loader),
                global_step,
                total_loss / batches,
                optimizer.param_groups[0]["lr"],
            )
    total_loss, batches = reduce_pair(total_loss, batches, device, world_size)
    return total_loss / max(batches, 1), global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    training_config: dict[str, Any],
    world_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    amp = bool(training_config.get("amp", False)) and device.type == "cuda"
    for graph in loader:
        graph = graph.to(device, non_blocking=True)
        with autocast_context(device, amp):
            prediction = model(graph.node_features, graph.edge_features, graph.edge_index)
            loss = node_loss(prediction, graph.target, graph.node_type, training_config)
        total_loss += float(loss)
        batches += 1
    total_loss, batches = reduce_pair(total_loss, batches, device, world_size)
    return total_loss / max(batches, 1)


@torch.no_grad()
def rollout_trajectory(
    model: nn.Module,
    trajectory: Trajectory,
    stats: NormalizationStats,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    steps = min(steps, trajectory.velocity.shape[0] - 1)
    velocity = trajectory.velocity[0].to(device)
    edge_index = trajectory.edge_index.to(device)
    edge_features = stats.normalize(
        trajectory.edge_features.to(device), stats.edge_mean, stats.edge_std
    )
    node_type = trajectory.node_type.to(device)
    node_type_features = _encode_node_type(node_type)
    update_mask = (node_type.squeeze(-1).eq(NORMAL) | node_type.squeeze(-1).eq(OUTFLOW)).unsqueeze(-1)

    velocity_squared_error = 0.0
    pressure_squared_error = 0.0
    for time_index in range(steps):
        node_features = torch.cat(
            (stats.normalize(velocity, stats.velocity_mean, stats.velocity_std), node_type_features),
            dim=-1,
        )
        prediction = model(node_features, edge_features, edge_index)
        velocity_delta = stats.denormalize(
            prediction[:, :2], stats.velocity_diff_mean, stats.velocity_diff_std
        )
        pressure = stats.denormalize(prediction[:, 2:], stats.pressure_mean, stats.pressure_std)
        predicted_velocity = velocity + velocity_delta
        boundary_velocity = trajectory.velocity[time_index + 1].to(device)
        velocity = torch.where(update_mask, predicted_velocity, boundary_velocity)

        velocity_target = trajectory.velocity[time_index + 1].to(device)
        pressure_target = trajectory.pressure[time_index + 1].to(device)
        velocity_squared_error += float(nn.functional.mse_loss(velocity, velocity_target))
        pressure_squared_error += float(nn.functional.mse_loss(pressure, pressure_target))
    return {
        "velocity_rollout_mse": velocity_squared_error / max(steps, 1),
        "pressure_rollout_mse": pressure_squared_error / max(steps, 1),
    }


def build_datasets(
    config: dict[str, Any], smoke_test: bool, logger: logging.Logger
) -> tuple[CylinderFlowDataset, CylinderFlowDataset, NormalizationStats]:
    data_config = config["data"]
    data_root = data_config["root"]
    stats = NormalizationStats.from_directory(data_config.get("stats_dir", Path(data_root) / "stats"))
    train_samples = 1 if smoke_test else int(data_config["train_samples"])
    val_samples = 1 if smoke_test else int(data_config["val_samples"])
    train_steps = 2 if smoke_test else int(data_config["train_steps"])
    val_steps = 2 if smoke_test else int(data_config["val_steps"])
    logger.info("Loading train split: trajectories=%d steps=%d", train_samples, train_steps)
    train_dataset = CylinderFlowDataset(
        data_root,
        "train",
        train_samples,
        train_steps,
        stats,
        noise_std=float(data_config.get("noise_std", 0.0)),
    )
    logger.info("Loading validation split: trajectories=%d steps=%d", val_samples, val_steps)
    val_dataset = CylinderFlowDataset(data_root, "val", val_samples, val_steps, stats)
    return train_dataset, val_dataset, stats


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data:
        config["data"]["root"] = args.data
        config["data"]["stats_dir"] = str(Path(args.data) / "stats")
    if args.output:
        config["training"]["output_dir"] = args.output
    if args.device:
        config["training"]["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.smoke_test:
        config["training"]["epochs"] = 1
        config["training"]["num_workers"] = 0

    training_config = config["training"]
    rank, world_size, local_rank, device = distributed_context(training_config.get("device", "auto"))
    logger = setup_logging(rank)
    set_seed(int(training_config.get("seed", 0)) + rank)
    logger.info("device=%s world_size=%d", device, world_size)

    train_dataset, val_dataset, stats = build_datasets(config, args.smoke_test, logger)
    data_config = config["data"]
    num_workers = 0 if args.smoke_test else int(data_config.get("num_workers", 0))
    train_loader, train_sampler = make_loader(
        train_dataset,
        int(training_config.get("batch_size", 1)),
        num_workers,
        bool(data_config.get("pin_memory", True)),
        True,
        rank,
        world_size,
    )
    val_loader, _ = make_loader(
        val_dataset,
        int(training_config.get("batch_size", 1)),
        num_workers,
        bool(data_config.get("pin_memory", True)),
        False,
        rank,
        world_size,
    )

    model = MeshGraphNet(**config["model"]).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    decay_rate = float(training_config["lr_decay_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: decay_rate**step
    )
    amp_enabled = bool(training_config.get("amp", False)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir = Path(training_config["output_dir"])
    checkpoint_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if bool(training_config.get("resume", True)):
        state = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        if state:
            start_epoch = int(state["epoch"])
            global_step = int(state.get("global_step", 0))
            best_val_loss = float(state.get("best_val_loss", best_val_loss))
            logger.info("Resumed %s at epoch=%d step=%d", checkpoint_path, start_epoch, global_step)

    logger.info("parameters=%d train_items=%d val_items=%d", trainable_parameters(model), len(train_dataset), len(val_dataset))
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    epochs = int(training_config["epochs"])
    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            training_config,
            epoch,
            logger,
            rank,
            world_size,
            global_step,
        )
        val_loss = validate(model, val_loader, device, training_config, world_size)
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        if rank == 0:
            logger.info(
                "epoch=%d/%d train_loss=%.8f val_loss=%.8f seconds=%.1f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                record["seconds"],
            )
            history.append(record)
            write_history(output_dir / "history.json", history)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                stale_epochs = 0
                save_checkpoint(
                    best_path, model, optimizer, scheduler, epoch + 1, global_step, best_val_loss, config
                )
            else:
                stale_epochs += 1
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                best_val_loss,
                config,
            )
        if world_size > 1:
            stop = torch.tensor(
                [stale_epochs > int(training_config.get("patience", 50))], device=device
            )
            dist.broadcast(stop, src=0)
            should_stop = bool(stop.item())
        else:
            should_stop = stale_epochs > int(training_config.get("patience", 50))
        if should_stop:
            logger.info("Early stopping after %d stale epochs", stale_epochs)
            break

    if args.rollout and rank == 0:
        load_checkpoint(best_path, model, device=device)
        evaluation_config = config["evaluation"]
        split = evaluation_config.get("rollout_split", "test")
        count = int(evaluation_config.get("rollout_trajectories", 10))
        steps = int(evaluation_config.get("rollout_steps", 299))
        data_steps = min(steps + 1, int(data_config[f"{split}_steps"]))
        rollout_dataset = CylinderFlowDataset(
            data_config["root"], split, count, data_steps, stats
        )
        metrics = [
            rollout_trajectory(unwrapped_model(model), trajectory, stats, steps, device)
            for trajectory in rollout_dataset.iter_trajectories()
        ]
        mean_metrics = {
            key: sum(metric[key] for metric in metrics) / len(metrics) for key in metrics[0]
        }
        logger.info("rollout_metrics=%s", mean_metrics)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

