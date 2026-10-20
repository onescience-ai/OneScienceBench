"""AirfRANS data pipeline: manifest -> VTK parse -> normalized point samples.

Paper: Hu et al., Physics of Fluids 2025 (DOI 10.1063/5.0258928), Section IV.A.

Pipeline:
  - read ``manifest.json`` (full_train / full_test split names)
  - per sample: read ``<sim>/<sim>_internal.vtu`` + ``<sim>/<sim>_aerofoil.vtp``
  - build 7-dim input features ``[pos(2), u_inf(2), sdf(1), normal(2)]``
  - build 4-dim targets ``[u, v, p, nu_t]`` + surface bool mask
  - z-score normalization (stats computed from training split only)
  - train/val point subsampling to 32000 nodes per epoch; test keeps all nodes

This module is pure torch / numpy / pyvista. ``AirflowDatasetBuilder`` parses
the VTK geometry; ``AirflowDataset`` is a torch Dataset that normalizes and
subsamples. ``split_and_compute_stats`` mirrors the datamodule split logic
(90/10 train/val from ``full_train``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:  # pyvista is only needed to parse VTK; keep import optional
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None


class AirflowDatasetBuilder:
    """Parse one AirfRANS simulation into node-level arrays.

    Mirrors the official builder: read VTU (volume mesh + fields) and VTP
    (aerofoil surface), construct 7-dim inputs, 4-dim targets and the surface
    boolean mask.
    """

    def __init__(self, crop: tuple[float, float, float, float] | None = None):
        self.crop = crop

    def build_sample(self, sim_dir: Path, sim_name: str) -> dict[str, np.ndarray]:
        if pv is None:
            raise RuntimeError("pyvista is required to parse AirfRANS VTK files")
        internal = pv.read(f"{sim_dir}/{sim_name}_internal.vtu")
        aerofoil = pv.read(f"{sim_dir}/{sim_name}_aerofoil.vtp")
        internal = internal.compute_cell_sizes(length=False, volume=False)

        if self.crop is not None:
            bounds = (*self.crop, 0, 1)
            internal = internal.clip_box(bounds=bounds, invert=False, crinkle=True)

        surf_bool = internal.point_data["U"][:, 0] == 0
        geom = -internal.point_data["implicit_distance"][:, None]
        u_inf, alpha = (
            float(sim_name.split("_")[2]),
            float(sim_name.split("_")[3]) * np.pi / 180,
        )
        u = (
            np.array([np.cos(alpha), np.sin(alpha)]) * u_inf
        ).reshape(1, 2) * np.ones_like(internal.point_data["U"][:, :1])
        normal = np.zeros_like(u)
        normal[surf_bool] = self._reorganize(
            aerofoil.points[:, :2],
            internal.points[surf_bool, :2],
            -aerofoil.point_data["Normals"][:, :2],
        )

        attr = np.concatenate(
            [
                u,
                geom,
                normal,
                internal.point_data["U"][:, :2],
                internal.point_data["p"][:, None],
                internal.point_data["nut"][:, None],
            ],
            axis=-1,
        )
        pos = internal.points[:, :2]
        init = np.concatenate([pos, attr[:, :5]], axis=1)  # [N, 7]
        target = attr[:, 5:]  # [N, 4] = [u, v, p, nut]

        return {
            "x": init.astype(np.float32),
            "y": target.astype(np.float32),
            "surf": surf_bool,
            "pos": pos.astype(np.float32),
        }

    @staticmethod
    def _reorganize(in_order, out_order, quantity):
        n = out_order.shape[0]
        idx = np.zeros(n)
        point_to_idx = {tuple(in_order[i]): i for i in range(n)}
        for i in range(n):
            idx[i] = point_to_idx[tuple(out_order[i])]
        idx = idx.astype(int)
        assert (in_order[idx] == out_order).all()
        return quantity[idx]


class AirflowDataset(Dataset):
    """Load one node-level sample, normalize, subsample points.

    Normalization statistics come from the training split only and are shared
    by train/val/test. z-score: ``x_bar = (x - mu) / sigma``.
    """

    def __init__(
        self,
        data_dir: Path,
        sample_names: list[str],
        mean_in: np.ndarray,
        std_in: np.ndarray,
        mean_out: np.ndarray,
        std_out: np.ndarray,
        subsampling: int | None = 32000,
        crop: tuple[float, float, float, float] | None = None,
        seed: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_names = sample_names
        self.mean_in = torch.as_tensor(mean_in, dtype=torch.float32)
        self.std_in = torch.as_tensor(std_in, dtype=torch.float32)
        self.mean_out = torch.as_tensor(mean_out, dtype=torch.float32)
        self.std_out = torch.as_tensor(std_out, dtype=torch.float32)
        self.subsampling = subsampling
        self.crop = crop
        self.builder = AirflowDatasetBuilder(crop=crop)
        self._seed = seed
        self._cache: dict[str, dict[str, Any]] = {}

    def _load_raw(self, sim_name: str) -> dict[str, Any]:
        if sim_name in self._cache:
            return self._cache[sim_name]
        sample = self.builder.build_sample(self.data_dir / sim_name, sim_name)
        self._cache[sim_name] = sample
        return sample

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sim_name = self.sample_names[index]
        sample = self._load_raw(sim_name)

        x = torch.as_tensor(sample["x"], dtype=torch.float32)
        y = torch.as_tensor(sample["y"], dtype=torch.float32)
        surf = torch.as_tensor(sample["surf"], dtype=torch.bool)

        n = x.shape[0]
        if self.subsampling is not None and n > self.subsampling:
            g = torch.Generator()
            if self._seed is not None:
                g.manual_seed(self._seed * (index + 1))
            idx = torch.randperm(n, generator=g)[: self.subsampling]
            x, y, surf = x[idx], y[idx], surf[idx]

        x = (x - self.mean_in) / (self.std_in + 1e-9)
        y = (y - self.mean_out) / (self.std_out + 1e-9)
        return x, y, surf


def split_and_compute_stats(
    data_dir: Path,
    stats_dir: Path,
    train_name: str = "full_train",
    test_name: str = "full_test",
    val_split_ratio: float = 0.1,
    subsampling: int = 32000,
    crop: tuple[float, float, float, float] | None = None,
    seed: int = 1,
) -> tuple[list[str], list[str], list[str]]:
    """Read manifest, split train/val, compute z-score stats from train.

    Returns ``(train_names, val_names, test_names)``.
    """
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(data_dir) / "manifest.json"
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    full_train = manifest[train_name]
    full_test = manifest[test_name]
    n = len(full_train)
    n_val = max(1, int(n * val_split_ratio))
    train_names = full_train[: n - n_val]
    val_names = full_train[n - n_val :]

    # Compute stats only if not cached.
    stats = [stats_dir / "mean_in.npy", stats_dir / "std_in.npy"]
    if not all(p.exists() for p in stats):
        builder = AirflowDatasetBuilder(crop=crop)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        rng = np.random.default_rng(seed)
        for sim_name in train_names:
            sample = builder.build_sample(Path(data_dir) / sim_name, sim_name)
            npts = sample["x"].shape[0]
            if subsampling is not None and npts > subsampling:
                idx = rng.choice(npts, size=subsampling, replace=False)
                xs.append(sample["x"][idx])
                ys.append(sample["y"][idx])
            else:
                xs.append(sample["x"])
                ys.append(sample["y"])
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        mean_in, std_in = x.mean(0), x.std(0)
        mean_out, std_out = y.mean(0), y.std(0)
        std_in[std_in < 1e-12] = 1.0
        std_out[std_out < 1e-12] = 1.0
        np.save(stats_dir / "mean_in.npy", mean_in)
        np.save(stats_dir / "std_in.npy", std_in)
        np.save(stats_dir / "mean_out.npy", mean_out)
        np.save(stats_dir / "std_out.npy", std_out)

    return train_names, val_names, test_names


def make_loaders(
    data_dir: Path,
    stats_dir: Path,
    train_names: list[str],
    val_names: list[str],
    test_names: list[str],
    batch_size: int = 1,
    num_workers: int = 0,
    subsampling: int = 32000,
    crop: tuple[float, float, float, float] | None = None,
    seed: int = 1,
):
    """Build train/val/test DataLoaders from the split names."""
    stats_dir = Path(stats_dir)
    mean_in = np.load(stats_dir / "mean_in.npy")
    std_in = np.load(stats_dir / "std_in.npy")
    mean_out = np.load(stats_dir / "mean_out.npy")
    std_out = np.load(stats_dir / "std_out.npy")

    train_ds = AirflowDataset(
        data_dir, train_names, mean_in, std_in, mean_out, std_out,
        subsampling=subsampling, crop=crop, seed=seed,
    )
    val_ds = AirflowDataset(
        data_dir, val_names, mean_in, std_in, mean_out, std_out,
        subsampling=subsampling, crop=crop, seed=seed,
    )
    test_ds = AirflowDataset(
        data_dir, test_names, mean_in, std_in, mean_out, std_out,
        subsampling=None, crop=crop, seed=seed,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader