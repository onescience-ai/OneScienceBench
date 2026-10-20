"""Darcy Flow data loading for the RNO paper reproduction.

The loader follows the official repository: the first 1,000 samples from
``smooth1`` are used for training and the first 200 samples from ``smooth2``
are used for testing. Statistics are computed from the training split only.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


@dataclass
class ScalarNormalizer:
    """Global scalar normalization used by the official RNO implementation."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor, eps: float = 1.0e-8) -> "ScalarNormalizer":
        mean = values.mean(dim=(0, 1), keepdim=True)
        std = values.std(dim=(0, 1), keepdim=True) + eps
        return cls(mean=mean, std=std)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(values.device)
        std = self.std.to(values.device)
        return values * std + mean

    def to(self, device: torch.device) -> "ScalarNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "ScalarNormalizer":
        return cls(mean=state["mean"], std=state["std"])


class DarcyFlowDataset(Dataset):
    """A flattened regular-grid Darcy dataset with a shared position grid."""

    def __init__(self, positions: torch.Tensor, coefficients: torch.Tensor, targets: torch.Tensor):
        if coefficients.shape != targets.shape:
            raise ValueError(f"Coefficient/target mismatch: {coefficients.shape} vs {targets.shape}")
        if positions.ndim != 2 or positions.shape[-1] != 2:
            raise ValueError(f"Expected positions [N, 2], received {positions.shape}")
        if coefficients.shape[1] != positions.shape[0]:
            raise ValueError("The flattened field size does not match the position grid")
        self.positions = positions
        self.coefficients = coefficients
        self.targets = targets

    def __len__(self) -> int:
        return self.coefficients.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.positions, self.coefficients[index], self.targets[index]


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

    coeff = mat[input_key]
    target = mat[target_key]
    if coeff.ndim != 3 or target.ndim != 3:
        raise ValueError(f"Expected [samples, H, W] arrays, got {coeff.shape} and {target.shape}")
    if coeff.shape[0] < count or target.shape[0] < count:
        raise ValueError(f"Requested {count} samples but {path.name} contains {coeff.shape[0]}")

    coeff = np.asarray(
        coeff[:count, ::downsample, ::downsample][:, :resolution, :resolution],
        dtype=np.float32,
    ).reshape(count, -1)
    target = np.asarray(
        target[:count, ::downsample, ::downsample][:, :resolution, :resolution],
        dtype=np.float32,
    ).reshape(count, -1)
    del mat
    gc.collect()
    return torch.from_numpy(coeff), torch.from_numpy(target)


def make_position_grid(resolution: int) -> torch.Tensor:
    axis = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    return torch.from_numpy(np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1))


def build_darcy_datasets(
    config: Dict[str, Any],
    smoke_test: bool = False,
) -> Tuple[DarcyFlowDataset, DarcyFlowDataset, ScalarNormalizer, ScalarNormalizer, Dict[str, Any]]:
    data_cfg = config["data"]
    root = Path(data_cfg["root"]).expanduser()
    n_train = min(2, int(data_cfg["n_train"])) if smoke_test else int(data_cfg["n_train"])
    n_test = min(1, int(data_cfg["n_test"])) if smoke_test else int(data_cfg["n_test"])
    resolution = int(data_cfg["resolution"])
    downsample = int(data_cfg["downsample"])

    expected_resolution = (int(data_cfg["native_resolution"]) - 1) // downsample + 1
    if resolution != expected_resolution:
        raise ValueError(
            f"resolution={resolution} is inconsistent with native_resolution/downsample "
            f"({expected_resolution})"
        )

    train_x, train_y = _load_downsampled_fields(
        root / data_cfg["train_file"],
        data_cfg["input_key"],
        data_cfg["target_key"],
        n_train,
        downsample,
        resolution,
    )
    x_normalizer = ScalarNormalizer.fit(train_x)
    y_normalizer = ScalarNormalizer.fit(train_y)
    train_x = x_normalizer.encode(train_x)
    train_y = y_normalizer.encode(train_y)

    test_x, test_y = _load_downsampled_fields(
        root / data_cfg["test_file"],
        data_cfg["input_key"],
        data_cfg["target_key"],
        n_test,
        downsample,
        resolution,
    )
    test_x = x_normalizer.encode(test_x)

    positions = make_position_grid(resolution)
    train_set = DarcyFlowDataset(positions, train_x, train_y)
    test_set = DarcyFlowDataset(positions, test_x, test_y)
    metadata = {
        "train_file": str(root / data_cfg["train_file"]),
        "test_file": str(root / data_cfg["test_file"]),
        "n_train": n_train,
        "n_test": n_test,
        "native_resolution": int(data_cfg["native_resolution"]),
        "resolution": resolution,
        "downsample": downsample,
        "input_key": data_cfg["input_key"],
        "target_key": data_cfg["target_key"],
        "normalization": "global_scalar_train_statistics",
    }
    return train_set, test_set, x_normalizer, y_normalizer, metadata

