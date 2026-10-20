"""Multi-resolution Darcy Flow pipeline with cached coarse inputs and interface weights."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class ScalarNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor, eps: float = 1.0e-8) -> "ScalarNormalizer":
        return cls(values.mean(dim=(0, 1), keepdim=True), values.std(dim=(0, 1), keepdim=True) + eps)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values.device) + self.mean.to(values.device)

    def to(self, device: torch.device) -> "ScalarNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "ScalarNormalizer":
        return cls(mean=state["mean"], std=state["std"])


class MultiScaleDarcyFlowDataset(Dataset):
    """Fine/coarse coefficient fields plus dense interface-importance weights."""

    def __init__(
        self,
        positions: torch.Tensor,
        fine_coefficients: torch.Tensor,
        coarse_coefficients: torch.Tensor,
        targets: torch.Tensor,
        importance: torch.Tensor,
    ):
        samples, nodes = fine_coefficients.shape
        if targets.shape != fine_coefficients.shape or importance.shape != fine_coefficients.shape:
            raise ValueError("Fine coefficients, targets, and importance maps must share [samples, nodes]")
        if positions.shape != (nodes, 2) or coarse_coefficients.shape[0] != samples:
            raise ValueError("Invalid position grid or coarse coefficient batch")
        self.positions = positions
        self.fine_coefficients = fine_coefficients
        self.coarse_coefficients = coarse_coefficients
        self.targets = targets
        self.importance = importance

    def __len__(self) -> int:
        return self.fine_coefficients.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.positions,
            self.fine_coefficients[index],
            self.coarse_coefficients[index],
            self.targets[index],
            self.importance[index],
        )


def _load_downsampled_fields(
    path: Path,
    input_key: str,
    target_key: str,
    count: int,
    downsample: int,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Darcy MAT file not found: {path}")
    mat = scipy.io.loadmat(path, variable_names=[input_key, target_key])
    missing = [key for key in (input_key, target_key) if key not in mat]
    if missing:
        raise KeyError(f"Missing MAT variables {missing} in {path}")
    coefficient, target = mat[input_key], mat[target_key]
    if coefficient.ndim != 3 or target.ndim != 3 or coefficient.shape[0] < count:
        raise ValueError(f"Invalid Darcy arrays in {path}: {coefficient.shape}/{target.shape}")
    coefficient = np.asarray(
        coefficient[:count, ::downsample, ::downsample][:, :resolution, :resolution], dtype=np.float32
    ).reshape(count, -1)
    target = np.asarray(
        target[:count, ::downsample, ::downsample][:, :resolution, :resolution], dtype=np.float32
    ).reshape(count, -1)
    del mat
    gc.collect()
    return torch.from_numpy(coefficient), torch.from_numpy(target)


def make_position_grid(resolution: int) -> torch.Tensor:
    axis = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    return torch.from_numpy(np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1))


def _interface_importance(
    physical_coefficient: torch.Tensor,
    resolution: int,
    alpha: float,
    maximum: float,
) -> torch.Tensor:
    field = physical_coefficient.reshape(-1, 1, resolution, resolution)
    grad_x = F.pad((field[..., 2:, :] - field[..., :-2, :]).abs() * 0.5, (0, 0, 1, 1))
    grad_y = F.pad((field[..., :, 2:] - field[..., :, :-2]).abs() * 0.5, (1, 1, 0, 0))
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12)
    scale = magnitude.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
    weights = 1.0 + alpha * magnitude / scale
    return weights.clamp(max=maximum).reshape(physical_coefficient.shape[0], -1)


def _cache_key(
    source: Path,
    split: str,
    count: int,
    resolution: int,
    coarse_resolution: int,
    normalizer: ScalarNormalizer,
    alpha: float,
    maximum: float,
) -> str:
    stat = source.stat()
    payload = {
        "source": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "split": split,
        "count": count,
        "resolution": resolution,
        "coarse_resolution": coarse_resolution,
        "mean": float(normalizer.mean.item()),
        "std": float(normalizer.std.item()),
        "importance_alpha": alpha,
        "importance_max": maximum,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _cached_auxiliary_fields(
    normalized_fine: torch.Tensor,
    physical_fine: torch.Tensor,
    source: Path,
    split: str,
    normalizer: ScalarNormalizer,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    data_cfg = config["data"]
    resolution = int(data_cfg["resolution"])
    coarse_resolution = int(data_cfg["coarse_resolution"])
    alpha = float(data_cfg.get("importance_alpha", 2.0))
    maximum = float(data_cfg.get("importance_max", 8.0))
    key = _cache_key(
        source, split, normalized_fine.shape[0], resolution, coarse_resolution, normalizer, alpha, maximum
    )
    cache_dir = Path(data_cfg.get("cache_dir", "./cache/rno_darcy_multiscale")).expanduser()
    cache_path = cache_dir / f"{split}_{key}.pt"
    if bool(data_cfg.get("cache_enabled", True)) and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        coarse, importance = payload["coarse"], payload["importance"]
        if coarse.shape[0] == normalized_fine.shape[0] and importance.shape == normalized_fine.shape:
            return coarse, importance, str(cache_path)

    fine_grid = normalized_fine.reshape(-1, 1, resolution, resolution)
    coarse = F.interpolate(
        fine_grid, size=(coarse_resolution, coarse_resolution), mode="bilinear", align_corners=True
    ).reshape(normalized_fine.shape[0], -1)
    importance = _interface_importance(physical_fine, resolution, alpha, maximum)
    if bool(data_cfg.get("cache_enabled", True)):
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"coarse": coarse, "importance": importance}, temporary)
        os.replace(temporary, cache_path)
    return coarse, importance, str(cache_path)


def build_darcy_datasets(
    config: Dict[str, Any], smoke_test: bool = False
) -> Tuple[MultiScaleDarcyFlowDataset, MultiScaleDarcyFlowDataset, ScalarNormalizer, ScalarNormalizer, Dict[str, Any]]:
    data_cfg = config["data"]
    root = Path(data_cfg["root"]).expanduser()
    n_train = min(2, int(data_cfg["n_train"])) if smoke_test else int(data_cfg["n_train"])
    n_test = min(1, int(data_cfg["n_test"])) if smoke_test else int(data_cfg["n_test"])
    resolution = int(data_cfg["resolution"])
    downsample = int(data_cfg["downsample"])
    expected = (int(data_cfg["native_resolution"]) - 1) // downsample + 1
    if resolution != expected:
        raise ValueError(f"resolution={resolution} is inconsistent with native/downsample={expected}")

    train_source = root / data_cfg["train_file"]
    test_source = root / data_cfg["test_file"]
    train_physical_x, train_y = _load_downsampled_fields(
        train_source, data_cfg["input_key"], data_cfg["target_key"], n_train, downsample, resolution
    )
    x_normalizer = ScalarNormalizer.fit(train_physical_x)
    y_normalizer = ScalarNormalizer.fit(train_y)
    train_x = x_normalizer.encode(train_physical_x)
    train_targets = y_normalizer.encode(train_y)
    train_coarse, train_importance, train_cache = _cached_auxiliary_fields(
        train_x, train_physical_x, train_source, "train", x_normalizer, config
    )

    test_physical_x, test_targets = _load_downsampled_fields(
        test_source, data_cfg["input_key"], data_cfg["target_key"], n_test, downsample, resolution
    )
    test_x = x_normalizer.encode(test_physical_x)
    test_coarse, test_importance, test_cache = _cached_auxiliary_fields(
        test_x, test_physical_x, test_source, "test", x_normalizer, config
    )
    positions = make_position_grid(resolution)
    train_set = MultiScaleDarcyFlowDataset(
        positions, train_x, train_coarse, train_targets, train_importance
    )
    test_set = MultiScaleDarcyFlowDataset(
        positions, test_x, test_coarse, test_targets, test_importance
    )
    metadata = {
        "train_file": str(train_source),
        "test_file": str(test_source),
        "n_train": n_train,
        "n_test": n_test,
        "resolution": resolution,
        "coarse_resolution": int(data_cfg["coarse_resolution"]),
        "normalization": "global_scalar_train_statistics",
        "train_cache": train_cache,
        "test_cache": test_cache,
        "importance": "coefficient_interface_gradient",
    }
    return train_set, test_set, x_normalizer, y_normalizer, metadata

