"""Darcy Flow MAT reader and paper split construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class FieldNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1.0e-6

    @classmethod
    def fit(cls, values: torch.Tensor, eps: float = 1.0e-6) -> "FieldNormalizer":
        if values.ndim != 4:
            raise ValueError(f"Expected NCHW tensor, got {tuple(values.shape)}")
        return cls(values.mean(dim=0, keepdim=True), values.std(dim=0, keepdim=True), eps)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        return (values - mean) / (std + self.eps)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        return values * (std + self.eps) + mean

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FieldNormalizer":
        return cls(torch.as_tensor(state["mean"]), torch.as_tensor(state["std"]), float(state["eps"]))


class DarcyDataset(Dataset):
    def __init__(self, coefficient: torch.Tensor, solution: torch.Tensor, input_normalizer: FieldNormalizer):
        if coefficient.shape != solution.shape:
            raise ValueError(f"Input/target shapes differ: {coefficient.shape} vs {solution.shape}")
        self.coefficient = coefficient
        self.solution = solution
        self.input_normalizer = input_normalizer

    def __len__(self) -> int:
        return self.coefficient.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_normalizer.encode(self.coefficient[index : index + 1]).squeeze(0)
        return x, self.solution[index]


def _load_mat_fields(path: Path, input_key: str, target_key: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Darcy MAT file does not exist: {path}")
    try:
        from scipy.io import loadmat

        fields = loadmat(path, variable_names=[input_key, target_key])
        if input_key not in fields or target_key not in fields:
            raise KeyError(f"Required keys {input_key!r}/{target_key!r} missing from {path}")
        return fields[input_key], fields[target_key]
    except NotImplementedError:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("HDF5-backed MAT input requires h5py") from error
        with h5py.File(path, "r") as handle:
            if input_key not in handle or target_key not in handle:
                raise KeyError(f"Required keys {input_key!r}/{target_key!r} missing from {path}")
            # MATLAB HDF5 dimensions are stored in reverse order.
            return np.asarray(handle[input_key]).transpose(), np.asarray(handle[target_key]).transpose()


def load_split(
    path: str | Path,
    input_key: str,
    target_key: str,
    samples: int,
    subsample: int,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficient, solution = _load_mat_fields(Path(path), input_key, target_key)
    if coefficient.ndim != 3 or solution.ndim != 3:
        raise ValueError(f"Expected sample-first NHW fields, got {coefficient.shape}/{solution.shape}")
    if coefficient.shape != solution.shape or coefficient.shape[0] < samples:
        raise ValueError(f"Invalid shapes or sample count: {coefficient.shape}/{solution.shape}, need {samples}")
    coefficient = coefficient[:samples, ::subsample, ::subsample]
    solution = solution[:samples, ::subsample, ::subsample]
    if coefficient.shape[-2:] != (resolution, resolution):
        raise ValueError(
            f"Subsampling produced {coefficient.shape[-2:]}, expected {(resolution, resolution)}; "
            "check resolution/subsample settings"
        )
    if not np.isfinite(coefficient).all() or not np.isfinite(solution).all():
        raise ValueError(f"Non-finite values found in {path}")
    x = torch.from_numpy(np.ascontiguousarray(coefficient[:, None])).float()
    y = torch.from_numpy(np.ascontiguousarray(solution[:, None])).float()
    return x, y


def build_loaders(
    config: Mapping[str, Any],
    data_root: Optional[str] = None,
    input_normalizer: Optional[FieldNormalizer] = None,
    output_normalizer: Optional[FieldNormalizer] = None,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
) -> tuple[DataLoader, DataLoader, FieldNormalizer, FieldNormalizer]:
    data_cfg = config["data"]
    training_cfg = config["training"]
    root = Path(data_root or data_cfg["root"])
    n_train = int(train_samples or data_cfg["train_samples"])
    n_test = int(test_samples or data_cfg["test_samples"])
    common = dict(
        input_key=data_cfg["input_key"],
        target_key=data_cfg["target_key"],
        subsample=int(data_cfg["subsample"]),
        resolution=int(data_cfg["resolution"]),
    )
    train_x, train_y = load_split(root / data_cfg["train_file"], samples=n_train, **common)
    test_x, test_y = load_split(root / data_cfg["test_file"], samples=n_test, **common)
    eps = float(data_cfg.get("normalization_eps", 1.0e-6))
    input_normalizer = input_normalizer or FieldNormalizer.fit(train_x, eps)
    output_normalizer = output_normalizer or FieldNormalizer.fit(train_y, eps)
    train_set = DarcyDataset(train_x, train_y, input_normalizer)
    test_set = DarcyDataset(test_x, test_y, input_normalizer)
    workers = int(training_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_set,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, input_normalizer, output_normalizer


def synthetic_loaders(
    resolution: int = 17, train_samples: int = 8, test_samples: int = 4, batch_size: int = 2
) -> tuple[DataLoader, DataLoader, FieldNormalizer, FieldNormalizer]:
    """Deterministic PDE-shaped tensors used only by the explicit --smoke mode."""
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(train_samples + test_samples, 1, resolution, resolution, generator=generator) + 1.0
    y = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)
    input_norm = FieldNormalizer.fit(x[:train_samples])
    output_norm = FieldNormalizer.fit(y[:train_samples])
    train = DarcyDataset(x[:train_samples], y[:train_samples], input_norm)
    test = DarcyDataset(x[train_samples:], y[train_samples:], input_norm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
        input_norm,
        output_norm,
    )
