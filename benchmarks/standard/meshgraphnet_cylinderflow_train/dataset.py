"""DeepMind CylinderFlow TFRecord reader and graph preprocessing."""

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
    """Tensor-only graph batch accepted by :class:`model.MeshGraphNet`."""

    node_features: Tensor
    edge_features: Tensor
    edge_index: Tensor
    target: Tensor
    node_type: Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "GraphSample":
        return GraphSample(
            *(value.to(device, non_blocking=non_blocking) for value in self.__dict__.values())
        )


@dataclass
class Trajectory:
    cells: Tensor
    mesh_pos: Tensor
    node_type: Tensor
    velocity: Tensor
    pressure: Tensor
    edge_index: Tensor
    edge_features: Tensor


class NormalizationStats:
    """Normalization tensors shared by train, validation, and rollout."""

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
                f"Expected normalization files {edge_path} and {node_path}. "
                "Generate them with the OneScience data preparation or point data.stats_dir "
                "to the supplied CylinderFlow stats directory."
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
        raise ImportError(
            "Reading the official CylinderFlow TFRecords requires TensorFlow. "
            "Install tensorflow (CPU is sufficient); model training itself uses PyTorch."
        ) from exc
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    return tf


def _unique_bidirectional_edges(cells: np.ndarray) -> Tensor:
    directed = np.concatenate(
        (cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]), axis=0
    )
    directed = np.concatenate((directed, directed[:, ::-1]), axis=0)
    directed = np.unique(directed, axis=0)
    return torch.as_tensor(directed.T.copy(), dtype=torch.long)


def _encode_node_type(node_type: Tensor) -> Tensor:
    """Map CylinderFlow types {0, 4, 5, 6} to four one-hot channels."""
    squeezed = node_type.squeeze(-1).long()
    mapped = torch.where(squeezed == NORMAL, squeezed, squeezed - 3)
    if mapped.numel() and (mapped.min() < 0 or mapped.max() > 3):
        values = sorted(torch.unique(squeezed).tolist())
        raise ValueError(f"Unexpected CylinderFlow node types: {values}")
    return F.one_hot(mapped, num_classes=4).float()


def _parse_trajectory(serialized: Any, metadata: dict[str, Any], steps: int) -> dict[str, Any]:
    tf = _tensorflow()
    description = {name: tf.io.VarLenFeature(tf.string) for name in metadata["field_names"]}
    parsed = tf.io.parse_single_example(serialized, description)
    output: dict[str, Any] = {}
    for name, spec in metadata["features"].items():
        dtype = getattr(tf, spec["dtype"])
        value = tf.reshape(tf.io.decode_raw(parsed[name].values, dtype), spec["shape"])
        if spec["type"] == "dynamic":
            value = value[:steps]
        output[name] = value
    return output


def read_trajectories(
    data_root: str | Path, split: str, samples: int, steps: int
) -> list[Trajectory]:
    """Read a bounded number of trajectories from an official TFRecord split."""
    data_root = Path(os.path.expandvars(str(data_root))).expanduser()
    metadata_path = data_root / "meta.json"
    record_path = data_root / f"{split}.tfrecord"
    if not metadata_path.is_file() or not record_path.is_file():
        raise FileNotFoundError(f"Missing {metadata_path} or {record_path}")

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    trajectory_length = int(metadata["trajectory_length"])
    if not 2 <= steps <= trajectory_length:
        raise ValueError(f"steps must be in [2, {trajectory_length}], got {steps}")

    tf = _tensorflow()
    records = tf.data.TFRecordDataset(str(record_path)).take(samples)
    trajectories: list[Trajectory] = []
    for serialized in records:
        arrays = {
            key: value.numpy() for key, value in _parse_trajectory(serialized, metadata, steps).items()
        }
        cells_np = arrays["cells"][0]
        mesh_pos = torch.as_tensor(arrays["mesh_pos"][0], dtype=torch.float32)
        edge_index = _unique_bidirectional_edges(cells_np)
        source, destination = edge_index
        displacement = mesh_pos[source] - mesh_pos[destination]
        raw_edge_features = torch.cat(
            (displacement, torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)), dim=-1
        )
        trajectories.append(
            Trajectory(
                cells=torch.as_tensor(cells_np, dtype=torch.long),
                mesh_pos=mesh_pos,
                node_type=torch.as_tensor(arrays["node_type"][0], dtype=torch.long),
                velocity=torch.as_tensor(arrays["velocity"], dtype=torch.float32),
                pressure=torch.as_tensor(arrays["pressure"], dtype=torch.float32),
                edge_index=edge_index,
                edge_features=raw_edge_features,
            )
        )
    if len(trajectories) != samples:
        raise ValueError(f"Requested {samples} {split} trajectories, found {len(trajectories)}")
    return trajectories


class CylinderFlowDataset(Dataset[GraphSample]):
    """One-step samples over in-memory CylinderFlow trajectories."""

    SPLIT_NAMES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        samples: int,
        steps: int,
        stats: NormalizationStats,
        noise_std: float = 0.0,
    ) -> None:
        try:
            record_split = self.SPLIT_NAMES[split]
        except KeyError as exc:
            raise ValueError(f"Unknown split {split!r}") from exc
        self.split = split
        self.steps = steps
        self.stats = stats
        self.noise_std = float(noise_std) if split == "train" else 0.0
        self.trajectories = read_trajectories(data_root, record_split, samples, steps)

    def __len__(self) -> int:
        return len(self.trajectories) * (self.steps - 1)

    def __getitem__(self, index: int) -> GraphSample:
        trajectory_index, time_index = divmod(index, self.steps - 1)
        trajectory = self.trajectories[trajectory_index]
        velocity = trajectory.velocity[time_index].clone()
        velocity_delta = trajectory.velocity[time_index + 1] - trajectory.velocity[time_index]

        if self.noise_std > 0.0:
            noise = torch.randn_like(velocity) * self.noise_std
            normal_mask = trajectory.node_type.squeeze(-1).eq(NORMAL).unsqueeze(-1)
            noise = torch.where(normal_mask, noise, torch.zeros_like(noise))
            velocity += noise
            velocity_delta -= noise

        node_features = torch.cat(
            (
                self.stats.normalize(velocity, self.stats.velocity_mean, self.stats.velocity_std),
                _encode_node_type(trajectory.node_type),
            ),
            dim=-1,
        )
        target = torch.cat(
            (
                self.stats.normalize(
                    velocity_delta, self.stats.velocity_diff_mean, self.stats.velocity_diff_std
                ),
                self.stats.normalize(
                    trajectory.pressure[time_index + 1],
                    self.stats.pressure_mean,
                    self.stats.pressure_std,
                ),
            ),
            dim=-1,
        )
        edge_features = self.stats.normalize(
            trajectory.edge_features, self.stats.edge_mean, self.stats.edge_std
        )
        return GraphSample(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=trajectory.edge_index,
            target=target,
            node_type=trajectory.node_type,
        )

    def iter_trajectories(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)


def collate_graphs(samples: Sequence[GraphSample]) -> GraphSample:
    """Batch variable-size graphs by concatenating nodes and offsetting edges."""
    if not samples:
        raise ValueError("Cannot collate an empty graph batch")
    offsets: list[int] = []
    node_count = 0
    for sample in samples:
        offsets.append(node_count)
        node_count += sample.node_features.shape[0]
    edge_index = torch.cat(
        [sample.edge_index + offset for sample, offset in zip(samples, offsets)], dim=1
    )
    return GraphSample(
        node_features=torch.cat([sample.node_features for sample in samples]),
        edge_features=torch.cat([sample.edge_features for sample in samples]),
        edge_index=edge_index,
        target=torch.cat([sample.target for sample in samples]),
        node_type=torch.cat([sample.node_type for sample in samples]),
    )

