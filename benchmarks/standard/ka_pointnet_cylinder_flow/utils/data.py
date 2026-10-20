from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class NormalizationStats:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    p_min: float
    p_max: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SplitBundle:
    name: str
    inputs: np.ndarray
    targets: np.ndarray
    coords_physical: np.ndarray
    targets_physical: np.ndarray
    indices: np.ndarray


class PointNetCFDDataset(Dataset):
    def __init__(self, split_bundle: SplitBundle) -> None:
        self.split_bundle = split_bundle

    def __len__(self) -> int:
        return int(self.split_bundle.inputs.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.from_numpy(self.split_bundle.inputs[index].T.copy()).float()
        targets = torch.from_numpy(self.split_bundle.targets[index].T.copy()).float()
        return inputs, targets


def _load_split_indices(root: Path, filename: str) -> np.ndarray:
    return np.load(root / filename).astype(np.int64)


def _normalize_to_minus_one_one(values: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    return 2.0 * (values - value_min) / (value_max - value_min) - 1.0


def _denormalize_from_minus_one_one(values: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    return 0.5 * (values + 1.0) * (value_max - value_min) + value_min


def prepare_pointnet_cfd_data(config: Dict) -> Tuple[Dict[str, SplitBundle], NormalizationStats]:
    data_cfg = config["data"]
    root = Path(data_cfg["root"])
    raw = np.load(root / data_cfg["cfd_file"]).astype(np.float32)

    input_columns = data_cfg["input_columns"]
    output_columns = data_cfg["output_columns"]

    coords_physical = raw[:, :, [input_columns["x"], input_columns["y"]]].copy()
    targets_physical = raw[:, :, [output_columns["u"], output_columns["v"], output_columns["p"]]].copy()

    training_idx = _load_split_indices(root, data_cfg["training_idx_file"])
    validation_idx = _load_split_indices(root, data_cfg["validation_idx_file"])
    test_idx = _load_split_indices(root, data_cfg["test_idx_file"])

    x_min = float(coords_physical[:, :, 0].min())
    x_max = float(coords_physical[:, :, 0].max())
    y_min = float(coords_physical[:, :, 1].min())
    y_max = float(coords_physical[:, :, 1].max())

    train_targets_physical = targets_physical[training_idx]
    u_min = float(train_targets_physical[:, :, 0].min())
    u_max = float(train_targets_physical[:, :, 0].max())
    v_min = float(train_targets_physical[:, :, 1].min())
    v_max = float(train_targets_physical[:, :, 1].max())
    p_min = float(train_targets_physical[:, :, 2].min())
    p_max = float(train_targets_physical[:, :, 2].max())

    stats = NormalizationStats(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
        p_min=p_min,
        p_max=p_max,
    )

    inputs = coords_physical.copy()
    inputs[:, :, 0] = _normalize_to_minus_one_one(inputs[:, :, 0], x_min, x_max)
    inputs[:, :, 1] = _normalize_to_minus_one_one(inputs[:, :, 1], y_min, y_max)

    targets = targets_physical.copy()
    targets[:, :, 0] = _normalize_to_minus_one_one(targets[:, :, 0], u_min, u_max)
    targets[:, :, 1] = _normalize_to_minus_one_one(targets[:, :, 1], v_min, v_max)
    targets[:, :, 2] = _normalize_to_minus_one_one(targets[:, :, 2], p_min, p_max)

    split_map = {
        "train": training_idx,
        "validation": validation_idx,
        "test": test_idx,
    }
    bundles: Dict[str, SplitBundle] = {}
    for split_name, split_indices in split_map.items():
        bundles[split_name] = SplitBundle(
            name=split_name,
            inputs=inputs[split_indices],
            targets=targets[split_indices],
            coords_physical=coords_physical[split_indices],
            targets_physical=targets_physical[split_indices],
            indices=split_indices,
        )

    return bundles, stats


def build_dataloaders(config: Dict, splits: Dict[str, SplitBundle]) -> Dict[str, DataLoader]:
    train_cfg = config["training"]
    num_workers = int(train_cfg.get("num_workers", 0))

    loaders = {
        "train": DataLoader(
            PointNetCFDDataset(splits["train"]),
            batch_size=int(train_cfg["train_batch_size"]),
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        ),
        "validation": DataLoader(
            PointNetCFDDataset(splits["validation"]),
            batch_size=int(train_cfg["eval_batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            PointNetCFDDataset(splits["test"]),
            batch_size=int(config["inference"]["batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        ),
    }
    return loaders


def denormalize_targets_numpy(values: np.ndarray, stats: NormalizationStats, channel_axis: int = 1) -> np.ndarray:
    mins = np.array([stats.u_min, stats.v_min, stats.p_min], dtype=np.float32)
    maxs = np.array([stats.u_max, stats.v_max, stats.p_max], dtype=np.float32)
    shape = [1] * values.ndim
    shape[channel_axis] = 3
    mins = mins.reshape(shape)
    maxs = maxs.reshape(shape)
    return _denormalize_from_minus_one_one(values, mins, maxs)


def denormalize_targets_torch(values: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    mins = torch.tensor([stats.u_min, stats.v_min, stats.p_min], device=values.device, dtype=values.dtype).view(1, 3, 1)
    maxs = torch.tensor([stats.u_max, stats.v_max, stats.p_max], device=values.device, dtype=values.dtype).view(1, 3, 1)
    return 0.5 * (values + 1.0) * (maxs - mins) + mins
