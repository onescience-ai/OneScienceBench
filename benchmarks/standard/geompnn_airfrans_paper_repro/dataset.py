"""Raw AirfRANS ingestion and normalization for paper-first GeoMPNN.

The paper fixes the geometric descriptors and target fields. It does not specify
standardization internals, so field normalization follows the AIRS implementation:
training-set population mean/std, unnormalized surface normals, and a separately
standardized signed-log pressure target.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

import numpy as np
import torch

try:
    from airfrans import Simulation
except ImportError as exc:  # Deferred so model-only validation remains available.
    Simulation = None
    _AIRFRANS_IMPORT_ERROR = exc


NOSE, TAIL, CLOSEST = "nose", "tail", "closest"
VOLUME, SURFACE = "volume", "surface"
TARGETS = ["x-velocity", "y-velocity", "pressure", "turbulent_viscosity"]
PRESSURE_INDEX = TARGETS.index("pressure")


@dataclass
class GraphData:
    """Tensor representation of one AirfRANS simulation."""

    name: str
    pos: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    surf: torch.Tensor
    inlet_vel: torch.Tensor
    surface_normal: torch.Tensor
    dist: Dict[str, torch.Tensor]
    coords: Dict[str, torch.Tensor]

    def __len__(self) -> int:
        return len(self.pos)

    def clone(self) -> "GraphData":
        return GraphData(
            name=self.name,
            pos=self.pos.clone(),
            x=self.x.clone(),
            y=self.y.clone(),
            surf=self.surf.clone(),
            inlet_vel=self.inlet_vel.clone(),
            surface_normal=self.surface_normal.clone(),
            dist={key: value.clone() for key, value in self.dist.items()},
            coords={key: value.clone() for key, value in self.coords.items()},
        )

    def to(self, device: torch.device | str) -> "GraphData":
        return GraphData(
            name=self.name,
            pos=self.pos.to(device),
            x=self.x.to(device),
            y=self.y.to(device),
            surf=self.surf.to(device),
            inlet_vel=self.inlet_vel.to(device),
            surface_normal=self.surface_normal.to(device),
            dist={key: value.to(device) for key, value in self.dist.items()},
            coords={key: value.to(device) for key, value in self.coords.items()},
        )

    def index(self, index: torch.Tensor) -> "GraphData":
        return GraphData(
            name=self.name,
            pos=self.pos[index],
            x=self.x[index],
            y=self.y[index],
            surf=self.surf[index],
            inlet_vel=self.inlet_vel[index],
            surface_normal=self.surface_normal[index],
            dist={key: value[index] for key, value in self.dist.items()},
            coords={key: value[index] for key, value in self.coords.items()},
        )


class SignedLog:
    """Paper Sec. 4.5.2 signed logarithm and its exact inverse."""

    @staticmethod
    def scale(value):
        return np.sign(value) * np.log1p(np.abs(value))

    @staticmethod
    def unscale(value):
        return np.sign(value) * np.expm1(np.abs(value))


class _RunningMoments:
    """Numerically stable population moments for a stream of 2-D arrays."""

    def __init__(self, width: int):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError(f"expected (*,{len(self.mean)}) values, got {values.shape}")
        if len(values) == 0:
            return
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / total
        self.count = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("cannot finalize empty normalization statistics")
        std = np.sqrt(self.m2 / self.count)
        std[std < 1e-12] = 1.0
        return self.mean, std


@dataclass(frozen=True)
class NormalizationStats:
    """AIRS-compatible statistics required by the reduced GeoMPNN input."""

    inlet_mean: np.ndarray
    inlet_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    log_pressure_mean: float
    log_pressure_std: float

    @classmethod
    def fit(cls, graphs: Iterable[GraphData]) -> "NormalizationStats":
        inlet = _RunningMoments(2)
        targets = _RunningMoments(len(TARGETS))
        log_pressure = _RunningMoments(1)
        count = 0
        for graph in graphs:
            inlet.update(graph.x.cpu().numpy())
            raw_targets = graph.y.cpu().numpy()
            targets.update(raw_targets)
            log_pressure.update(SignedLog.scale(raw_targets[:, PRESSURE_INDEX : PRESSURE_INDEX + 1]))
            count += 1
        if count == 0:
            raise RuntimeError("no training graphs were supplied")
        inlet_mean, inlet_std = inlet.finalize()
        target_mean, target_std = targets.finalize()
        log_mean, log_std = log_pressure.finalize()
        return cls(
            inlet_mean=inlet_mean,
            inlet_std=inlet_std,
            target_mean=target_mean,
            target_std=target_std,
            log_pressure_mean=float(log_mean[0]),
            log_pressure_std=float(log_std[0]),
        )

    def transform(self, graph: GraphData, *, copy: bool = True) -> GraphData:
        """Normalize one raw graph; optionally reuse storage owned by the caller."""
        raw_pressure = graph.y[:, PRESSURE_INDEX].cpu().numpy().copy()
        output = graph.clone() if copy else graph
        inlet_mean = torch.as_tensor(self.inlet_mean, dtype=output.x.dtype)
        inlet_std = torch.as_tensor(self.inlet_std, dtype=output.x.dtype)
        target_mean = torch.as_tensor(self.target_mean, dtype=output.y.dtype)
        target_std = torch.as_tensor(self.target_std, dtype=output.y.dtype)
        output.x = (output.x - inlet_mean) / inlet_std
        output.y = (output.y - target_mean) / target_std

        normalized_log_pressure = (
            SignedLog.scale(raw_pressure) - self.log_pressure_mean
        ) / self.log_pressure_std
        output.y[:, PRESSURE_INDEX] = torch.as_tensor(
            normalized_log_pressure, dtype=output.y.dtype
        )
        return output

    def outputs_to_physical(self, normalized: np.ndarray) -> np.ndarray:
        """Convert four normalized model outputs to physical AirfRANS fields."""
        normalized = np.asarray(normalized)
        if normalized.ndim != 2 or normalized.shape[1] != len(TARGETS):
            raise ValueError(f"expected prediction shape (N,{len(TARGETS)}), got {normalized.shape}")
        physical = normalized * self.target_std + self.target_mean
        log_pressure = (
            normalized[:, PRESSURE_INDEX] * self.log_pressure_std
            + self.log_pressure_mean
        )
        physical[:, PRESSURE_INDEX] = SignedLog.unscale(log_pressure)
        return physical

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            inlet_mean=self.inlet_mean,
            inlet_std=self.inlet_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            log_pressure_mean=np.array(self.log_pressure_mean),
            log_pressure_std=np.array(self.log_pressure_std),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        with np.load(path) as state:
            return cls(
                inlet_mean=state["inlet_mean"],
                inlet_std=state["inlet_std"],
                target_mean=state["target_mean"],
                target_std=state["target_std"],
                log_pressure_mean=float(state["log_pressure_mean"]),
                log_pressure_std=float(state["log_pressure_std"]),
            )


class AirfransRepository:
    """Resolve paper-sized splits and load raw VTU/VTP simulations on demand."""

    def __init__(self, data_config: Mapping):
        self.root = Path(data_config["root"]).expanduser()
        manifest_path = self.root / data_config.get("manifest", "manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"AirfRANS manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.split_config = data_config["splits"]

    def split_names(self, split: str) -> List[str]:
        if split not in self.split_config:
            raise KeyError(f"unknown split {split!r}; expected one of {list(self.split_config)}")
        spec = self.split_config[split]
        has_key = "key" in spec
        has_intersection = "intersection" in spec
        if has_key == has_intersection:
            raise ValueError(f"split {split!r} must define exactly one of key/intersection")

        if has_key:
            keys = [spec["key"]]
        else:
            keys = list(spec["intersection"])
            if len(keys) < 2:
                raise ValueError(f"split {split!r} intersection requires at least two keys")
        missing = [key for key in keys if key not in self.manifest]
        if missing:
            raise KeyError(f"manifest is missing split keys: {missing}")

        first = list(self.manifest[keys[0]])
        if len(keys) == 1:
            names = first
        else:
            allowed = set(first)
            for key in keys[1:]:
                allowed.intersection_update(self.manifest[key])
            names = [name for name in first if name in allowed]
        if len(names) != len(set(names)):
            raise ValueError(f"manifest split {split!r} contains duplicate simulation names")
        return names

    def split_counts(self) -> Dict[str, int]:
        return {split: len(self.split_names(split)) for split in self.split_config}

    def load_raw_graph(self, name: str) -> GraphData:
        if Simulation is None:
            raise ImportError("install the AirfRANS Python package") from _AIRFRANS_IMPORT_ERROR
        simulation_path = self.root / name
        if not simulation_path.is_dir():
            raise FileNotFoundError(f"simulation directory not found: {simulation_path}")

        simulation = Simulation(root=str(self.root), name=name)
        pos_raw = np.asarray(simulation.position, dtype=np.float32)
        surface = np.asarray(simulation.surface, dtype=bool)
        if surface.ndim != 1 or not surface.any():
            raise RuntimeError(f"{name}: invalid or empty surface mask")

        # Paper Sec. 2.1: move the exact left-most surface point to the origin.
        surface_points = pos_raw[surface]
        leading_edge = surface_points[np.argmin(surface_points[:, 0])]
        pos = pos_raw - leading_edge
        centered_surface = pos[surface]
        trailing_edge = centered_surface[np.argmax(centered_surface[:, 0])]

        inlet = np.asarray(simulation.input_velocity, dtype=np.float32)
        normals = np.asarray(simulation.normals, dtype=np.float32)
        targets = np.concatenate(
            [
                np.asarray(simulation.velocity, dtype=np.float32),
                np.asarray(simulation.pressure, dtype=np.float32),
                np.asarray(simulation.nu_t, dtype=np.float32),
            ],
            axis=1,
        )
        surface_distance = np.abs(np.asarray(simulation.sdf, dtype=np.float32))

        position = torch.from_numpy(pos)
        return GraphData(
            name=name,
            pos=position,
            x=torch.from_numpy(inlet.copy()),
            y=torch.from_numpy(targets),
            surf=torch.from_numpy(surface),
            inlet_vel=torch.from_numpy(inlet),
            surface_normal=torch.from_numpy(normals),
            dist={
                CLOSEST: torch.from_numpy(surface_distance),
                NOSE: position.norm(dim=-1, keepdim=True),
                TAIL: (position - torch.from_numpy(trailing_edge)).norm(dim=-1, keepdim=True),
            },
            coords={
                NOSE: position,
                TAIL: position - torch.from_numpy(trailing_edge),
            },
        )

    def load_split(
        self,
        split: str,
        *,
        limit: Optional[int] = None,
    ) -> List[GraphData]:
        names = self.split_names(split)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            names = names[:limit]
        graphs: List[GraphData] = []
        for index, name in enumerate(names, start=1):
            print(f"load {split}: {index}/{len(names)} {name}", flush=True)
            graphs.append(self.load_raw_graph(name))
        return graphs

    def iter_normalized(
        self,
        split: str,
        stats: NormalizationStats,
        *,
        limit: Optional[int] = None,
    ) -> Iterator[GraphData]:
        names = self.split_names(split)
        if limit is not None:
            names = names[:limit]
        for index, name in enumerate(names, start=1):
            print(f"load {split}: {index}/{len(names)} {name}", flush=True)
            yield stats.transform(self.load_raw_graph(name), copy=False)


def prepare_training_graphs(
    repository: AirfransRepository,
    stats_path: str | Path,
    *,
    limit: Optional[int] = None,
) -> tuple[List[GraphData], NormalizationStats]:
    """Load training graphs once, fit AIRS-style statistics, and normalize."""
    raw_graphs = repository.load_split("train", limit=limit)
    stats = NormalizationStats.fit(raw_graphs)
    stats.save(stats_path)
    normalized_graphs = [
        stats.transform(graph, copy=False) for graph in raw_graphs
    ]
    return normalized_graphs, stats


def make_train_pair(graph: GraphData, supervised_points: int) -> Dict[str, GraphData]:
    """Paper Sec. 3.1/4.2.1: fresh mesh sample plus the complete surface graph."""
    if len(graph) < supervised_points:
        raise ValueError(
            f"{graph.name}: {len(graph)} points is smaller than supervised_points={supervised_points}"
        )
    volume_index = torch.randperm(len(graph))[:supervised_points]
    surface_index = torch.where(graph.surf)[0]
    return {
        VOLUME: graph.index(volume_index),
        SURFACE: graph.index(surface_index),
    }


def make_inference_pair(graph: GraphData) -> Dict[str, GraphData]:
    """Paper Sec. 3.1: use the complete volume and surface meshes."""
    surface_index = torch.where(graph.surf)[0]
    return {
        VOLUME: graph.clone(),
        SURFACE: graph.index(surface_index),
    }
