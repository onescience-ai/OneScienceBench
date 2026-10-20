"""AirfRANS loading, split, normalization and point sampling."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def _reorganize(in_points: np.ndarray, out_points: np.ndarray,
                values: np.ndarray) -> np.ndarray:
    """Match surface values to internal-mesh surface nodes by coordinates."""
    indices = np.zeros(out_points.shape[0], dtype=np.int64)
    for i, point in enumerate(out_points):
        match = np.all(in_points == point, axis=1)
        indices[i] = np.flatnonzero(match)[0]
    return values[indices]


def load_simulation(name: str, data_root: str | Path) -> Data:
    """Load one AirfRANS CFD simulation.

    Input x (7):  [x, y, Uinf_x, Uinf_y, SDF, normal_x, normal_y]
    Target y (4): [U_x, U_y, p, nut]
    """
    sim_dir = Path(data_root) / name
    internal = pv.read(sim_dir / f"{name}_internal.vtu")
    airfoil = pv.read(sim_dir / f"{name}_aerofoil.vtp")

    surface = internal.point_data["U"][:, 0] == 0
    sdf = -internal.point_data["implicit_distance"][:, None]

    parts = name.split("_")
    u_inf = float(parts[2])
    alpha = np.deg2rad(float(parts[3]))
    freestream_vec = np.array([np.cos(alpha), np.sin(alpha)], dtype=np.float32) * u_inf
    freestream = np.repeat(freestream_vec[None, :], internal.n_points, axis=0)

    normals = np.zeros((internal.n_points, 2), dtype=np.float32)
    normals[surface] = _reorganize(
        airfoil.points[:, :2],
        internal.points[surface, :2],
        -airfoil.point_data["Normals"][:, :2],
    ).astype(np.float32)

    pos = internal.points[:, :2].astype(np.float32)
    inputs = np.concatenate([pos, freestream, sdf, normals], axis=1).astype(np.float32)
    targets = np.concatenate([
        internal.point_data["U"][:, :2],
        internal.point_data["p"][:, None],
        internal.point_data["nut"][:, None],
    ], axis=1).astype(np.float32)

    return Data(
        pos=torch.from_numpy(pos),
        x=torch.from_numpy(inputs),
        y=torch.from_numpy(targets),
        surf=torch.from_numpy(surface).bool(),
    )


def load_names(data_root: str | Path, task: str, val_ratio: float = 0.1):
    """Read manifest.json and split the task training list into train/validation."""
    with open(Path(data_root) / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    names = manifest[f"{task}_train"]
    n_val = max(1, int(val_ratio * len(names)))
    return names[:-n_val], names[-n_val:]


def compute_normalization(dataset: Sequence[Data]):
    """Global per-variable mean/std over all training mesh nodes."""
    count = sum(data.x.shape[0] for data in dataset)
    sum_x = sum((data.x.double().sum(dim=0) for data in dataset), torch.zeros(7, dtype=torch.float64))
    sum_y = sum((data.y.double().sum(dim=0) for data in dataset), torch.zeros(4, dtype=torch.float64))
    mean_x = sum_x / count
    mean_y = sum_y / count

    ss_x = sum((((data.x.double() - mean_x) ** 2).sum(dim=0) for data in dataset),
               torch.zeros(7, dtype=torch.float64))
    ss_y = sum((((data.y.double() - mean_y) ** 2).sum(dim=0) for data in dataset),
               torch.zeros(4, dtype=torch.float64))

    return (
        mean_x.float(), torch.sqrt(ss_x / count).float(),
        mean_y.float(), torch.sqrt(ss_y / count).float(),
    )


def apply_normalization(dataset: Sequence[Data], stats) -> None:
    mean_x, std_x, mean_y, std_y = stats
    for data in dataset:
        data.x = (data.x - mean_x) / (std_x + 1e-8)
        data.y = (data.y - mean_y) / (std_y + 1e-8)


def load_dataset(names: Iterable[str], data_root: str | Path,
                 stats=None, fit_stats: bool = False):
    names = list(names)
    dataset = [load_simulation(name, data_root) for name in tqdm(names, desc="load AirfRANS")]
    if fit_stats:
        stats = compute_normalization(dataset)
    if stats is not None:
        apply_normalization(dataset, stats)
    return (dataset, stats) if fit_stats else dataset


def sample_points(data: Data, n_points: int) -> Data:
    """Randomly sample mesh nodes exactly as the original training pipeline does."""
    total = data.x.shape[0]
    if total < n_points:
        raise ValueError(f"{total} mesh nodes < sample_points={n_points}")

    idx = torch.tensor(random.sample(range(total), n_points), dtype=torch.long)
    sampled = data.clone()
    sampled.pos = data.pos[idx]
    sampled.x = data.x[idx]
    sampled.y = data.y[idx]
    sampled.surf = data.surf[idx]
    return sampled
