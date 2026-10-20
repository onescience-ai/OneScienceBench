"""Fast structural, data, and GPU smoke validation for paper-first GeoMPNN."""
from __future__ import annotations

import argparse
import torch

from config_utils import load_config
from dataset import (
    AirfransRepository,
    NormalizationStats,
    PRESSURE_INDEX,
    SURFACE,
    VOLUME,
    make_train_pair,
)
from model import GeoMPNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--points", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points <= 0:
        raise ValueError("--points must be positive")
    config = load_config(args.config)
    repository = AirfransRepository(config["data"])

    counts = repository.split_counts()
    expected = {
        "train": config["experiment"]["train_simulations"],
        "test": config["experiment"]["test_id_simulations"],
        "test_ood": config["experiment"]["test_ood_simulations"],
    }
    if counts != expected:
        raise AssertionError(f"split counts {counts} != paper counts {expected}")
    print(f"split_counts={counts}")

    train_name = repository.split_names("train")[0]
    raw_graph = repository.load_raw_graph(train_name)
    surface_positions = raw_graph.pos[raw_graph.surf]
    leading = surface_positions[surface_positions[:, 0].argmin()]
    if not torch.allclose(leading, torch.zeros_like(leading), atol=1e-6):
        raise AssertionError(f"leading edge is not the origin: {leading.tolist()}")
    if torch.any(raw_graph.dist["closest"] < 0):
        raise AssertionError("surface-distance feature must be non-negative")
    print(
        f"sample={train_name} points={len(raw_graph)} surface={int(raw_graph.surf.sum())}"
    )

    stats = NormalizationStats.fit([raw_graph])
    graph = stats.transform(raw_graph, copy=False)
    supervised_points = min(args.points, len(graph))
    pair_cpu = make_train_pair(graph, supervised_points)

    runtime = config["runtime"]
    velocity_model = GeoMPNN(
        config["model"],
        "x-velocity",
        knn_query_chunk_size=int(runtime["knn_query_chunk_size"]),
    )
    pressure_model = GeoMPNN(
        config["model"],
        "pressure",
        knn_query_chunk_size=int(runtime["knn_query_chunk_size"]),
    )
    dimensions = {
        "velocity_node": velocity_model.node_dim,
        "velocity_edge": velocity_model.edge_dim,
        "pressure_node": pressure_model.node_dim,
        "pressure_edge": pressure_model.edge_dim,
    }
    paper_dimensions = {
        "velocity_node": 251,
        "velocity_edge": 115,
        "pressure_node": 449,
        "pressure_edge": 213,
    }
    if dimensions != paper_dimensions:
        raise AssertionError(f"feature dimensions {dimensions} != paper {paper_dimensions}")
    print(f"feature_dimensions={dimensions}")

    requested = args.device or runtime.get("device", "cuda:0")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{requested} requested but CUDA is unavailable")
    device = torch.device(requested)
    pressure_model = pressure_model.to(device)
    pair = {name: value.to(device) for name, value in pair_cpu.items()}

    pressure_model.train()
    prediction = pressure_model(pair)
    target = pair[VOLUME].y[:, PRESSURE_INDEX : PRESSURE_INDEX + 1]
    loss = torch.mean((prediction - target).square())
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise FloatingPointError("forward pass produced NaN or infinity")
    loss.backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in pressure_model.parameters()
    ):
        raise FloatingPointError("backward pass produced a non-finite gradient")

    memory_mb = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    )
    print(
        f"smoke_forward=ok device={device} volume={len(pair[VOLUME])} "
        f"surface={len(pair[SURFACE])} loss={float(loss):.6g} peak_memory_mb={memory_mb:.1f}"
    )


if __name__ == "__main__":
    main()
