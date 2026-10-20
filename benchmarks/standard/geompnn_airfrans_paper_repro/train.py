"""Train the paper-first protocol: eight repetitions and one model per field."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

from config_utils import load_config
from dataset import (
    AirfransRepository,
    TARGETS,
    VOLUME,
    make_train_pair,
    prepare_training_graphs,
)
from model import build_four_models


def set_repeat_seed(base_seed, repeat_index: int) -> int | None:
    """Optionally make a repetition deterministic; the paper does not publish seeds."""
    if base_seed is None:
        return None
    seed = int(base_seed) + int(repeat_index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {requested} to cpu", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def train_one_repeat(
    repeat_index: int,
    train_graphs,
    config: Dict,
    output_root: Path,
    device: torch.device,
) -> Path:
    """Train and save all four scalar-field models for one paper repetition."""
    training = config["training"]
    runtime = config["runtime"]
    run_dir = output_root / f"repeat_{repeat_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    used_seed = set_repeat_seed(runtime.get("base_seed"), repeat_index)
    models = build_four_models(
        config["model"],
        knn_query_chunk_size=int(runtime["knn_query_chunk_size"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in models.parameters())
    print(
        f"repeat={repeat_index:02d} device={device} parameters={parameter_count:,} seed={used_seed}",
        flush=True,
    )

    optimizers = {
        field: torch.optim.Adam(models[field].parameters(), lr=float(training["max_lr"]))
        for field in TARGETS
    }
    if training["onecycle_steps_per_epoch"] != "official_plus_one":
        raise ValueError("only the AIRS official_plus_one scheduler fallback is supported")
    steps_per_epoch = len(train_graphs) // int(training["batch_size"]) + 1
    total_steps = steps_per_epoch * int(training["epochs"])
    schedulers = {
        field: torch.optim.lr_scheduler.OneCycleLR(
            optimizers[field],
            max_lr=float(training["max_lr"]),
            total_steps=total_steps,
        )
        for field in TARGETS
    }

    epochs = int(training["epochs"])
    supervised_points = int(training["supervised_points"])
    log_every = int(training["log_every_epochs"])

    for epoch in range(epochs):
        order = torch.randperm(len(train_graphs)).tolist()
        running_loss = {field: 0.0 for field in TARGETS}

        for graph_index in order:
            pair_cpu = make_train_pair(train_graphs[graph_index], supervised_points)
            pair = {name: graph.to(device) for name, graph in pair_cpu.items()}
            targets = pair[VOLUME].y

            # Paper Sec. 3.1: four independently optimized scalar-field models.
            for target_index, field in enumerate(TARGETS):
                model = models[field]
                model.train()
                optimizer = optimizers[field]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(pair)
                loss = torch.mean(
                    (prediction - targets[:, target_index : target_index + 1]).square()
                )
                loss.backward()
                optimizer.step()
                schedulers[field].step()
                running_loss[field] += float(loss.detach())

        if epoch % log_every == 0 or epoch == epochs - 1:
            fields = " ".join(
                f"{field}={running_loss[field] / len(train_graphs):.6g}"
                for field in TARGETS
            )
            print(f"repeat={repeat_index:02d} epoch={epoch:04d} {fields}", flush=True)

    checkpoint = run_dir / "ckpt.pt"
    torch.save(
        {
            "models": models.state_dict(),
            "config": config,
            "repeat_index": repeat_index,
            "seed": used_seed,
            "normalization_path": "../preprocessing/normalization.npz",
        },
        checkpoint,
    )
    print(f"saved {checkpoint}", flush=True)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--repeat-index",
        type=int,
        default=None,
        help="Train one repeat in [0,7]; omit to run all eight paper repetitions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(args.output or config["runtime"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    repository = AirfransRepository(config["data"])
    counts = repository.split_counts()
    expected_counts = {
        "train": int(config["experiment"]["train_simulations"]),
        "test": int(config["experiment"]["test_id_simulations"]),
        "test_ood": int(config["experiment"]["test_ood_simulations"]),
    }
    if counts != expected_counts:
        raise RuntimeError(f"resolved split counts {counts} do not match paper {expected_counts}")
    print(f"resolved paper splits: {counts}", flush=True)

    normalization_path = output_root / "preprocessing" / "normalization.npz"
    train_graphs, _ = prepare_training_graphs(repository, normalization_path)
    device = select_device(config["runtime"].get("device", "cuda:0"))

    repeat_count = int(config["experiment"]["repeats"])
    if args.repeat_index is None:
        repeat_indices: Iterable[int] = range(repeat_count)
    elif 0 <= args.repeat_index < repeat_count:
        repeat_indices = [args.repeat_index]
    else:
        raise ValueError(f"--repeat-index must be in [0,{repeat_count - 1}]")

    for repeat_index in repeat_indices:
        train_one_repeat(repeat_index, train_graphs, config, output_root, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
