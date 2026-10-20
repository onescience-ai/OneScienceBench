"""CylinderFlow TFRecord pipeline with cached multiscale graphs and windows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


NORMAL = 0
OUTFLOW = 5


@dataclass
class GraphSample:
    node_features: Tensor
    fine_edge_features: Tensor
    fine_edge_index: Tensor
    coarse_node_index: Tensor
    coarse_edge_features: Tensor
    coarse_edge_index: Tensor
    down_edge_features: Tensor
    down_edge_index: Tensor
    up_edge_features: Tensor
    up_edge_index: Tensor
    teacher_target: Tensor
    initial_velocity: Tensor
    velocity_targets: Tensor
    pressure_targets: Tensor
    node_type: Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "GraphSample":
        return GraphSample(
            **{
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.__dict__.items()
            }
        )


@dataclass
class Trajectory:
    cells: Tensor
    mesh_pos: Tensor
    node_type: Tensor
    velocity: Tensor
    pressure: Tensor
    fine_edge_index: Tensor
    fine_edge_features: Tensor
    coarse_node_index: Tensor
    coarse_edge_index: Tensor
    coarse_edge_features: Tensor
    down_edge_index: Tensor
    down_edge_features: Tensor
    up_edge_index: Tensor
    up_edge_features: Tensor


class NormalizationStats:
    def __init__(self, edge: dict[str, Any], node: dict[str, Any]) -> None:
        self.edge_mean = torch.as_tensor(edge["edge_mean"], dtype=torch.float32)
        self.edge_std = torch.as_tensor(edge["edge_std"], dtype=torch.float32)
        self.velocity_mean = torch.as_tensor(node["velocity_mean"], dtype=torch.float32)
        self.velocity_std = torch.as_tensor(node["velocity_std"], dtype=torch.float32)
        self.velocity_diff_mean = torch.as_tensor(
            node["velocity_diff_mean"], dtype=torch.float32
        )
        self.velocity_diff_std = torch.as_tensor(
            node["velocity_diff_std"], dtype=torch.float32
        )
        self.pressure_mean = torch.as_tensor(node["pressure_mean"], dtype=torch.float32)
        self.pressure_std = torch.as_tensor(node["pressure_std"], dtype=torch.float32)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "NormalizationStats":
        directory = Path(os.path.expandvars(str(directory))).expanduser()
        edge_path = directory / "edge_stats.json"
        node_path = directory / "node_stats.json"
        if not edge_path.is_file() or not node_path.is_file():
            raise FileNotFoundError(
                f"Expected normalization files {edge_path} and {node_path}"
            )
        with edge_path.open("r", encoding="utf-8") as stream:
            edge = json.load(stream)
        with node_path.open("r", encoding="utf-8") as stream:
            node = json.load(stream)
        return cls(edge, node)

    @staticmethod
    def normalize(value: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return (value - mean.to(value.device)) / (std.to(value.device) + 1.0e-8)

    @staticmethod
    def denormalize(value: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return value * (std.to(value.device) + 1.0e-8) + mean.to(value.device)


def _tensorflow():
    try:
        import tensorflow.compat.v1 as tf
    except ImportError as exc:
        raise ImportError("CylinderFlow TFRecords require TensorFlow") from exc
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    return tf


def _unique_edges(edges: Tensor) -> Tensor:
    bidirectional = torch.cat((edges, edges.flip(0)), dim=1).T
    return torch.unique(bidirectional, dim=0).T.contiguous()


def _cell_edges(cells: np.ndarray) -> Tensor:
    directed = np.concatenate(
        (cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]), axis=0
    )
    return _unique_edges(torch.as_tensor(directed.T.copy(), dtype=torch.long))


def _edge_features(positions: Tensor, edge_index: Tensor) -> Tensor:
    source, destination = edge_index
    displacement = positions[source] - positions[destination]
    return torch.cat(
        (displacement, torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)),
        dim=-1,
    )


def _farthest_point_sample(positions: Tensor, ratio: float) -> Tensor:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("coarse_ratio must be in (0, 1]")
    node_count = positions.shape[0]
    sample_count = min(node_count, max(2, int(round(node_count * ratio))))
    selected = torch.empty(sample_count, dtype=torch.long)
    first = torch.argmin(positions[:, 0])
    selected[0] = first
    min_distance = torch.sum((positions - positions[first]) ** 2, dim=-1)
    for index in range(1, sample_count):
        current = torch.argmax(min_distance)
        selected[index] = current
        current_distance = torch.sum((positions - positions[current]) ** 2, dim=-1)
        min_distance = torch.minimum(min_distance, current_distance)
    return selected


def _knn_edges(positions: Tensor, neighbors: int) -> Tensor:
    node_count = positions.shape[0]
    if node_count < 2:
        raise ValueError("A graph requires at least two nodes")
    neighbors = min(max(1, neighbors), node_count - 1)
    distance = torch.cdist(positions, positions)
    distance.fill_diagonal_(float("inf"))
    nearest = torch.topk(distance, neighbors, largest=False, dim=1).indices
    destination = torch.arange(node_count).repeat_interleave(neighbors)
    source = nearest.reshape(-1)
    return _unique_edges(torch.stack((source, destination)))


def _cross_scale_edges(
    fine_positions: Tensor, coarse_positions: Tensor, neighbors: int
) -> tuple[Tensor, Tensor]:
    neighbors = min(max(1, neighbors), coarse_positions.shape[0])
    nearest = torch.topk(
        torch.cdist(fine_positions, coarse_positions),
        neighbors,
        largest=False,
        dim=1,
    ).indices
    fine_index = torch.arange(fine_positions.shape[0]).repeat_interleave(neighbors)
    coarse_index = nearest.reshape(-1)
    down = torch.stack((fine_index, coarse_index))
    up = torch.stack((coarse_index, fine_index))
    return down, up


def _build_multiscale_graph(
    mesh_pos: Tensor,
    fine_edge_index: Tensor,
    coarse_ratio: float,
    coarse_knn: int,
    cross_scale_knn: int,
) -> dict[str, Tensor]:
    coarse_node_index = _farthest_point_sample(mesh_pos, coarse_ratio)
    coarse_positions = mesh_pos[coarse_node_index]
    coarse_edge_index = _knn_edges(coarse_positions, coarse_knn)
    down_edge_index, up_edge_index = _cross_scale_edges(
        mesh_pos, coarse_positions, cross_scale_knn
    )
    return {
        "fine_edge_features": _edge_features(mesh_pos, fine_edge_index),
        "coarse_node_index": coarse_node_index,
        "coarse_edge_index": coarse_edge_index,
        "coarse_edge_features": _edge_features(coarse_positions, coarse_edge_index),
        "down_edge_index": down_edge_index,
        "down_edge_features": _cross_edge_features(
            mesh_pos, coarse_positions, down_edge_index
        ),
        "up_edge_index": up_edge_index,
        "up_edge_features": _cross_edge_features(
            coarse_positions, mesh_pos, up_edge_index
        ),
    }


def _cross_edge_features(
    source_positions: Tensor, destination_positions: Tensor, edge_index: Tensor
) -> Tensor:
    source, destination = edge_index
    displacement = source_positions[source] - destination_positions[destination]
    return torch.cat(
        (displacement, torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)),
        dim=-1,
    )


def _encode_node_type(node_type: Tensor) -> Tensor:
    squeezed = node_type.squeeze(-1).long()
    mapped = torch.where(squeezed == NORMAL, squeezed, squeezed - 3)
    if mapped.numel() and bool(((mapped < 0) | (mapped > 3)).any()):
        raise ValueError(f"Unexpected CylinderFlow node types: {torch.unique(squeezed)}")
    return F.one_hot(mapped, num_classes=4).float()


def _parse_trajectory(serialized: Any, metadata: dict[str, Any], steps: int) -> dict[str, Any]:
    tf = _tensorflow()
    description = {
        name: tf.io.VarLenFeature(tf.string) for name in metadata["field_names"]
    }
    parsed = tf.io.parse_single_example(serialized, description)
    output: dict[str, Any] = {}
    for name, spec in metadata["features"].items():
        value = tf.reshape(
            tf.io.decode_raw(parsed[name].values, getattr(tf, spec["dtype"])),
            spec["shape"],
        )
        output[name] = value[:steps] if spec["type"] == "dynamic" else value
    return output


def read_trajectories(
    data_root: str | Path,
    split: str,
    samples: int,
    steps: int,
    coarse_ratio: float,
    coarse_knn: int,
    cross_scale_knn: int,
) -> list[Trajectory]:
    data_root = Path(os.path.expandvars(str(data_root))).expanduser()
    with (data_root / "meta.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    trajectory_length = int(metadata["trajectory_length"])
    if not 2 <= steps <= trajectory_length:
        raise ValueError(f"steps must be in [2, {trajectory_length}], got {steps}")

    tf = _tensorflow()
    records = tf.data.TFRecordDataset(str(data_root / f"{split}.tfrecord")).take(samples)
    trajectories: list[Trajectory] = []
    for serialized in records:
        arrays = {
            key: value.numpy()
            for key, value in _parse_trajectory(serialized, metadata, steps).items()
        }
        cells = torch.as_tensor(arrays["cells"][0], dtype=torch.long)
        mesh_pos = torch.as_tensor(arrays["mesh_pos"][0], dtype=torch.float32)
        fine_edge_index = _cell_edges(arrays["cells"][0])
        multiscale = _build_multiscale_graph(
            mesh_pos,
            fine_edge_index,
            coarse_ratio,
            coarse_knn,
            cross_scale_knn,
        )
        trajectories.append(
            Trajectory(
                cells=cells,
                mesh_pos=mesh_pos,
                node_type=torch.as_tensor(arrays["node_type"][0], dtype=torch.long),
                velocity=torch.as_tensor(arrays["velocity"], dtype=torch.float32),
                pressure=torch.as_tensor(arrays["pressure"], dtype=torch.float32),
                fine_edge_index=fine_edge_index,
                **multiscale,
            )
        )
    if len(trajectories) != samples:
        raise ValueError(f"Requested {samples} {split} trajectories, found {len(trajectories)}")
    return trajectories


class CylinderFlowDataset(Dataset[GraphSample]):
    SPLIT_NAMES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        samples: int,
        steps: int,
        stats: NormalizationStats,
        noise_std: float = 0.0,
        max_rollout_horizon: int = 1,
        coarse_ratio: float = 0.25,
        coarse_knn: int = 6,
        cross_scale_knn: int = 3,
    ) -> None:
        if max_rollout_horizon < 1 or max_rollout_horizon >= steps:
            raise ValueError("max_rollout_horizon must be in [1, steps - 1]")
        self.split = split
        self.steps = steps
        self.max_rollout_horizon = max_rollout_horizon
        self.stats = stats
        self.noise_std = float(noise_std) if split == "train" else 0.0
        self.trajectories = read_trajectories(
            data_root,
            self.SPLIT_NAMES[split],
            samples,
            steps,
            coarse_ratio,
            coarse_knn,
            cross_scale_knn,
        )

    def __len__(self) -> int:
        return len(self.trajectories) * (self.steps - self.max_rollout_horizon)

    def __getitem__(self, index: int) -> GraphSample:
        windows_per_trajectory = self.steps - self.max_rollout_horizon
        trajectory_index, time_index = divmod(index, windows_per_trajectory)
        trajectory = self.trajectories[trajectory_index]
        initial_velocity = trajectory.velocity[time_index].clone()
        first_delta = trajectory.velocity[time_index + 1] - initial_velocity

        if self.noise_std > 0.0:
            noise = torch.randn_like(initial_velocity) * self.noise_std
            normal_mask = trajectory.node_type.squeeze(-1).eq(NORMAL).unsqueeze(-1)
            noise = torch.where(normal_mask, noise, torch.zeros_like(noise))
            initial_velocity += noise
            first_delta -= noise

        node_features = torch.cat(
            (
                self.stats.normalize(
                    initial_velocity, self.stats.velocity_mean, self.stats.velocity_std
                ),
                _encode_node_type(trajectory.node_type),
            ),
            dim=-1,
        )
        teacher_target = torch.cat(
            (
                self.stats.normalize(
                    first_delta,
                    self.stats.velocity_diff_mean,
                    self.stats.velocity_diff_std,
                ),
                self.stats.normalize(
                    trajectory.pressure[time_index + 1],
                    self.stats.pressure_mean,
                    self.stats.pressure_std,
                ),
            ),
            dim=-1,
        )
        graph_features = {
            name: self.stats.normalize(value, self.stats.edge_mean, self.stats.edge_std)
            for name, value in {
                "fine_edge_features": trajectory.fine_edge_features,
                "coarse_edge_features": trajectory.coarse_edge_features,
                "down_edge_features": trajectory.down_edge_features,
                "up_edge_features": trajectory.up_edge_features,
            }.items()
        }
        stop = time_index + 1 + self.max_rollout_horizon
        return GraphSample(
            node_features=node_features,
            fine_edge_index=trajectory.fine_edge_index,
            coarse_node_index=trajectory.coarse_node_index,
            coarse_edge_index=trajectory.coarse_edge_index,
            down_edge_index=trajectory.down_edge_index,
            up_edge_index=trajectory.up_edge_index,
            teacher_target=teacher_target,
            initial_velocity=initial_velocity,
            velocity_targets=trajectory.velocity[time_index + 1 : stop],
            pressure_targets=trajectory.pressure[time_index + 1 : stop],
            node_type=trajectory.node_type,
            **graph_features,
        )

    def iter_trajectories(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)


def collate_graphs(samples: Sequence[GraphSample]) -> GraphSample:
    if not samples:
        raise ValueError("Cannot collate an empty graph batch")
    horizons = {sample.velocity_targets.shape[0] for sample in samples}
    if len(horizons) != 1:
        raise ValueError("All samples in a batch must have the same rollout horizon")

    fine_offsets: list[int] = []
    coarse_offsets: list[int] = []
    fine_count = coarse_count = 0
    for sample in samples:
        fine_offsets.append(fine_count)
        coarse_offsets.append(coarse_count)
        fine_count += sample.node_features.shape[0]
        coarse_count += sample.coarse_node_index.shape[0]

    def cat(name: str) -> Tensor:
        return torch.cat([getattr(sample, name) for sample in samples], dim=0)

    return GraphSample(
        node_features=cat("node_features"),
        fine_edge_features=cat("fine_edge_features"),
        fine_edge_index=torch.cat(
            [sample.fine_edge_index + offset for sample, offset in zip(samples, fine_offsets)],
            dim=1,
        ),
        coarse_node_index=torch.cat(
            [sample.coarse_node_index + offset for sample, offset in zip(samples, fine_offsets)]
        ),
        coarse_edge_features=cat("coarse_edge_features"),
        coarse_edge_index=torch.cat(
            [
                sample.coarse_edge_index + offset
                for sample, offset in zip(samples, coarse_offsets)
            ],
            dim=1,
        ),
        down_edge_features=cat("down_edge_features"),
        down_edge_index=torch.cat(
            [
                sample.down_edge_index
                + torch.tensor([[fine_offset], [coarse_offset]])
                for sample, fine_offset, coarse_offset in zip(
                    samples, fine_offsets, coarse_offsets
                )
            ],
            dim=1,
        ),
        up_edge_features=cat("up_edge_features"),
        up_edge_index=torch.cat(
            [
                sample.up_edge_index
                + torch.tensor([[coarse_offset], [fine_offset]])
                for sample, fine_offset, coarse_offset in zip(
                    samples, fine_offsets, coarse_offsets
                )
            ],
            dim=1,
        ),
        teacher_target=cat("teacher_target"),
        initial_velocity=cat("initial_velocity"),
        velocity_targets=torch.cat([sample.velocity_targets for sample in samples], dim=1),
        pressure_targets=torch.cat([sample.pressure_targets for sample in samples], dim=1),
        node_type=cat("node_type"),
    )

