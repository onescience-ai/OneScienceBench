"""Train multiscale MeshGraphNet with curriculum pushforward rollout loss."""

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
    GraphSample,
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
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one trajectory and exercise multiscale forward/backward/checkpoint",
    )
    return parser.parse_args()


def distributed_context(requested_device: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = resolve_device(requested_device, local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo", init_method="env://"
        )
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
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_graphs,
            persistent_workers=num_workers > 0,
        ),
        sampler,
    )


def forward_graph(model: nn.Module, graph: GraphSample, node_features: Tensor) -> Tensor:
    return model(
        node_features,
        graph.fine_edge_features,
        graph.fine_edge_index,
        graph.coarse_node_index,
        graph.coarse_edge_features,
        graph.coarse_edge_index,
        graph.down_edge_features,
        graph.down_edge_index,
        graph.up_edge_features,
        graph.up_edge_index,
    )


def node_mask(node_type: Tensor, mode: str) -> Tensor:
    types = node_type.squeeze(-1)
    if mode == "all":
        return torch.ones_like(types, dtype=torch.bool)
    if mode == "normal_outflow":
        return types.eq(NORMAL) | types.eq(OUTFLOW)
    raise ValueError("loss node mode must be 'all' or 'normal_outflow'")


def error_loss(prediction: Tensor, target: Tensor, criterion: str) -> Tensor:
    if criterion == "mse":
        return nn.functional.mse_loss(prediction, target)
    if criterion == "mae":
        return nn.functional.l1_loss(prediction, target)
    raise ValueError("training.loss must be 'mse' or 'mae'")


def teacher_loss(
    prediction: Tensor,
    target: Tensor,
    node_type: Tensor,
    training_config: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    mask = node_mask(node_type, training_config.get("loss_nodes", "all"))
    criterion = training_config.get("loss", "mse").lower()
    velocity_loss = error_loss(prediction[mask, :2], target[mask, :2], criterion)
    pressure_loss = error_loss(prediction[mask, 2:], target[mask, 2:], criterion)
    total = (
        float(training_config.get("velocity_weight", 1.0)) * velocity_loss
        + float(training_config.get("pressure_weight", 0.1)) * pressure_loss
    )
    return total, velocity_loss, pressure_loss


def rollout_training_loss(
    model: nn.Module,
    graph: GraphSample,
    stats: NormalizationStats,
    horizon: int,
    training_config: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    zero = graph.node_features.new_zeros(())
    if horizon <= 1:
        return zero, zero, zero

    state = graph.initial_velocity
    node_types = _encode_node_type(graph.node_type)
    update_mask = (
        graph.node_type.squeeze(-1).eq(NORMAL)
        | graph.node_type.squeeze(-1).eq(OUTFLOW)
    ).unsqueeze(-1)
    for step in range(horizon - 1):
        with torch.no_grad():
            features = torch.cat(
                (
                    stats.normalize(state, stats.velocity_mean, stats.velocity_std),
                    node_types,
                ),
                dim=-1,
            )
            prediction = forward_graph(model, graph, features)
            velocity_delta = stats.denormalize(
                prediction[:, :2], stats.velocity_diff_mean, stats.velocity_diff_std
            )
            state = torch.where(
                update_mask,
                state + velocity_delta,
                graph.velocity_targets[step],
            )

    final_features = torch.cat(
        (
            stats.normalize(state.detach(), stats.velocity_mean, stats.velocity_std),
            node_types,
        ),
        dim=-1,
    )
    final_prediction = forward_graph(model, graph, final_features)
    predicted_velocity = state.detach() + stats.denormalize(
        final_prediction[:, :2], stats.velocity_diff_mean, stats.velocity_diff_std
    )
    predicted_velocity = torch.where(
        update_mask, predicted_velocity, graph.velocity_targets[horizon - 1]
    )
    predicted_pressure = stats.denormalize(
        final_prediction[:, 2:], stats.pressure_mean, stats.pressure_std
    )

    mask = node_mask(
        graph.node_type, training_config.get("rollout_loss_nodes", "normal_outflow")
    )
    velocity_error = (
        predicted_velocity - graph.velocity_targets[horizon - 1]
    ) / (stats.velocity_std.to(predicted_velocity.device) + 1.0e-8)
    pressure_error = (
        predicted_pressure - graph.pressure_targets[horizon - 1]
    ) / (stats.pressure_std.to(predicted_pressure.device) + 1.0e-8)
    criterion = training_config.get("loss", "mse").lower()
    velocity_loss = error_loss(
        velocity_error[mask], torch.zeros_like(velocity_error[mask]), criterion
    )
    pressure_loss = error_loss(
        pressure_error[mask], torch.zeros_like(pressure_error[mask]), criterion
    )
    total = (
        float(training_config.get("velocity_weight", 1.0)) * velocity_loss
        + float(training_config.get("pressure_weight", 0.1)) * pressure_loss
    )
    return total, velocity_loss, pressure_loss


def curriculum_horizon(epoch: int, epochs: int, training_config: dict[str, Any]) -> int:
    horizons = [int(value) for value in training_config.get("rollout_horizons", [1])]
    fractions = [
        float(value) for value in training_config.get("rollout_stage_fractions", [0.0])
    ]
    if len(horizons) != len(fractions) or not fractions or fractions[0] != 0.0:
        raise ValueError("rollout horizons/fractions must have equal length and start at 0.0")
    if fractions != sorted(fractions) or fractions[-1] >= 1.0:
        raise ValueError("rollout_stage_fractions must be sorted values in [0, 1)")
    progress = epoch / max(epochs, 1)
    selected = horizons[0]
    for horizon, start in zip(horizons, fractions):
        if progress >= start:
            selected = horizon
    return selected


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type) if enabled else nullcontext()


def reduce_pair(total: float, count: int, device: torch.device, world_size: int) -> tuple[float, int]:
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0]), int(values[1])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
    stats: NormalizationStats,
    training_config: dict[str, Any],
    epoch: int,
    horizon: int,
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
    rollout_weight = float(training_config.get("rollout_weight", 1.0))
    for batch_index, graph in enumerate(loader, start=1):
        graph = graph.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            prediction = forward_graph(model, graph, graph.node_features)
            one_step, _, _ = teacher_loss(
                prediction, graph.teacher_target, graph.node_type, training_config
            )
            rollout, _, _ = rollout_training_loss(
                model, graph, stats, horizon, training_config
            )
            loss = one_step + rollout_weight * rollout
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1
        total_loss += float(loss.detach())
        batches += 1
        if rank == 0 and batch_index % log_interval == 0:
            logger.info(
                "epoch=%d batch=%d/%d horizon=%d loss=%.8f",
                epoch + 1,
                batch_index,
                len(loader),
                horizon,
                total_loss / batches,
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
            prediction = forward_graph(model, graph, graph.node_features)
            loss, _, _ = teacher_loss(
                prediction, graph.teacher_target, graph.node_type, training_config
            )
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
    if not 1 <= steps < trajectory.velocity.shape[0]:
        raise ValueError("rollout steps must be positive and covered by the trajectory")
    node_type = trajectory.node_type.to(device)
    node_types = _encode_node_type(node_type)
    update_mask = (
        node_type.squeeze(-1).eq(NORMAL) | node_type.squeeze(-1).eq(OUTFLOW)
    ).unsqueeze(-1)
    static = {
        "fine_edge_features": stats.normalize(
            trajectory.fine_edge_features.to(device), stats.edge_mean, stats.edge_std
        ),
        "fine_edge_index": trajectory.fine_edge_index.to(device),
        "coarse_node_index": trajectory.coarse_node_index.to(device),
        "coarse_edge_features": stats.normalize(
            trajectory.coarse_edge_features.to(device), stats.edge_mean, stats.edge_std
        ),
        "coarse_edge_index": trajectory.coarse_edge_index.to(device),
        "down_edge_features": stats.normalize(
            trajectory.down_edge_features.to(device), stats.edge_mean, stats.edge_std
        ),
        "down_edge_index": trajectory.down_edge_index.to(device),
        "up_edge_features": stats.normalize(
            trajectory.up_edge_features.to(device), stats.edge_mean, stats.edge_std
        ),
        "up_edge_index": trajectory.up_edge_index.to(device),
    }
    velocity = trajectory.velocity[0].to(device)
    velocity_mse = pressure_mse = 0.0
    for time_index in range(steps):
        features = torch.cat(
            (stats.normalize(velocity, stats.velocity_mean, stats.velocity_std), node_types),
            dim=-1,
        )
        prediction = model(features, **static)
        velocity_delta = stats.denormalize(
            prediction[:, :2], stats.velocity_diff_mean, stats.velocity_diff_std
        )
        pressure = stats.denormalize(
            prediction[:, 2:], stats.pressure_mean, stats.pressure_std
        )
        target_velocity = trajectory.velocity[time_index + 1].to(device)
        velocity = torch.where(update_mask, velocity + velocity_delta, target_velocity)
        velocity_mse += float(nn.functional.mse_loss(velocity, target_velocity))
        pressure_mse += float(
            nn.functional.mse_loss(
                pressure, trajectory.pressure[time_index + 1].to(device)
            )
        )
    return {
        # Keep MSE for cross-trajectory aggregation; take one square root afterwards.
        "velocity_rollout_mse": velocity_mse / steps,
        "velocity_rollout_rmse": (velocity_mse / steps) ** 0.5,
        "pressure_rollout_mse": pressure_mse / steps,
    }


def build_datasets(
    config: dict[str, Any], smoke_test: bool, logger: logging.Logger
) -> tuple[CylinderFlowDataset, CylinderFlowDataset, NormalizationStats]:
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    data_root = data_config["root"]
    stats = NormalizationStats.from_directory(
        data_config.get("stats_dir", Path(data_root) / "stats")
    )
    train_samples = 1 if smoke_test else int(data_config["train_samples"])
    val_samples = 1 if smoke_test else int(data_config["val_samples"])
    train_steps = 3 if smoke_test else int(data_config["train_steps"])
    val_steps = 2 if smoke_test else int(data_config["val_steps"])
    max_horizon = max(int(value) for value in training_config["rollout_horizons"])
    graph_kwargs = {
        "coarse_ratio": float(model_config["coarse_ratio"]),
        "coarse_knn": int(model_config["coarse_knn"]),
        "cross_scale_knn": int(model_config["cross_scale_knn"]),
    }
    logger.info(
        "Loading train split: trajectories=%d steps=%d max_horizon=%d",
        train_samples,
        train_steps,
        max_horizon,
    )
    train_dataset = CylinderFlowDataset(
        data_root,
        "train",
        train_samples,
        train_steps,
        stats,
        noise_std=float(data_config.get("noise_std", 0.0)),
        max_rollout_horizon=max_horizon,
        **graph_kwargs,
    )
    logger.info("Loading validation split: trajectories=%d steps=%d", val_samples, val_steps)
    val_dataset = CylinderFlowDataset(
        data_root,
        "val",
        val_samples,
        val_steps,
        stats,
        max_rollout_horizon=1,
        **graph_kwargs,
    )
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
        config["training"]["rollout_horizons"] = [2]
        config["training"]["rollout_stage_fractions"] = [0.0]
        config["data"]["num_workers"] = 0

    training_config = config["training"]
    rank, world_size, local_rank, device = distributed_context(
        training_config.get("device", "auto")
    )
    output_dir = Path(training_config["output_dir"])
    logger = setup_logging(rank, output_dir)
    set_seed(int(training_config.get("seed", 0)) + rank)
    logger.info("device=%s world_size=%d", device, world_size)
    train_dataset, val_dataset, stats = build_datasets(config, args.smoke_test, logger)

    data_config = config["data"]
    num_workers = int(data_config.get("num_workers", 0))
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_config["learning_rate"])
    )
    decay_rate = float(training_config["lr_decay_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: decay_rate**step
    )
    amp_enabled = bool(training_config.get("amp", False)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    checkpoint_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    start_epoch = global_step = 0
    best_val_loss = float("inf")
    if bool(training_config.get("resume", True)):
        state = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        if state:
            start_epoch = int(state["epoch"])
            global_step = int(state.get("global_step", 0))
            best_val_loss = float(state.get("best_val_loss", best_val_loss))

    epochs = int(training_config["epochs"])
    logger.info(
        "parameters=%d train_items=%d val_items=%d",
        trainable_parameters(model),
        len(train_dataset),
        len(val_dataset),
    )
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        horizon = curriculum_horizon(epoch, epochs, training_config)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            stats,
            training_config,
            epoch,
            horizon,
            logger,
            rank,
            world_size,
            global_step,
        )
        val_loss = validate(model, val_loader, device, training_config, world_size)
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "rollout_horizon": horizon,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        if rank == 0:
            logger.info(
                "epoch=%d/%d horizon=%d train_loss=%.8f val_loss=%.8f seconds=%.1f",
                epoch + 1,
                epochs,
                horizon,
                train_loss,
                val_loss,
                record["seconds"],
            )
            history.append(record)
            write_history(
                output_dir / "history.json", {"config": config, "history": history}
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stale_epochs = 0
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    best_val_loss,
                    config,
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
            break

    rollout_metrics: dict[str, dict[str, float]] = {}
    if args.rollout and rank == 0:
        if load_checkpoint(best_path, model, device=device) is None:
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        evaluation = config["evaluation"]
        split = evaluation.get("rollout_split", "test")
        count = 1 if args.smoke_test else int(evaluation.get("rollout_trajectories", 10))
        horizons = [1] if args.smoke_test else [
            int(value) for value in evaluation.get(
                "rollout_horizons", [1, 50, int(evaluation.get("rollout_steps", 299))]
            )
        ]
        if not horizons or min(horizons) < 1:
            raise ValueError("evaluation.rollout_horizons must contain positive steps")
        if max(horizons) >= int(data_config[f"{split}_steps"]):
            raise ValueError("evaluation horizons exceed the configured trajectory length")
        graph_kwargs = {
            "coarse_ratio": float(config["model"]["coarse_ratio"]),
            "coarse_knn": int(config["model"]["coarse_knn"]),
            "cross_scale_knn": int(config["model"]["cross_scale_knn"]),
        }
        rollout_dataset = CylinderFlowDataset(
            data_config["root"],
            split,
            count,
            max(horizons) + 1,
            stats,
            max_rollout_horizon=1,
            **graph_kwargs,
        )
        for steps in horizons:
            metrics = [
                rollout_trajectory(unwrapped_model(model), trajectory, stats, steps, device)
                for trajectory in rollout_dataset.iter_trajectories()
            ]
            if not metrics:
                raise ValueError("No trajectories available for rollout evaluation")
            # Preserve equal-trajectory averaging, then convert velocity MSE to RMSE.
            velocity_mse = sum(item["velocity_rollout_mse"] for item in metrics) / len(metrics)
            pressure_mse = sum(item["pressure_rollout_mse"] for item in metrics) / len(metrics)
            rollout_metrics[str(steps)] = {
                "velocity_rmse": velocity_mse ** 0.5,
                "pressure_mse": pressure_mse,
            }
            logger.info("rollout_steps=%d metrics=%s", steps, rollout_metrics[str(steps)])
    if rank == 0:
        write_history(
            output_dir / "summary.json",
            {
                "config": config,
                "epochs_completed": start_epoch + len(history),
                "global_step": global_step,
                "best_checkpoint": str(best_path),
                "best_val_loss": best_val_loss if best_path.is_file() else None,
                "last_epoch_metrics": history[-1] if history else None,
                "smoke_test": args.smoke_test,
                "rollout_executed": bool(rollout_metrics),
                "rollout": rollout_metrics,
            },
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

