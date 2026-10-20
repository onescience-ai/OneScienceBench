"""Dynamic sliding-window pipeline for multistep KNO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class NavierStokesWindowDataset(Dataset[dict[str, Tensor]]):
    """Expose every valid history/future window for the active rollout horizon."""

    _RAW_CACHE: dict[Path, np.ndarray] = {}

    def __init__(
        self,
        config: dict[str, Any],
        split: str,
        horizon: int,
        normalizer_state: dict[str, float] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        source, data = config["source"], config["data"]
        path = (Path(source["data_dir"]).expanduser() / source["file_name"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Navier--Stokes data file not found: {path}")
        raw = self._load_raw(path)
        ntrain, ntest = int(data["ntrain"]), int(data["ntest"])
        if raw.shape[0] < ntrain + ntest:
            raise ValueError(f"Requested {ntrain + ntest} trajectories, found {raw.shape[0]}")
        self.t_in = int(data["t_in"])
        self.out_dim = int(data["out_dim"])
        self.max_horizon = int(data["max_horizon"])
        if self.out_dim != 1:
            raise ValueError("The benchmark contains one scalar field; out_dim must be 1")
        r1, r2 = int(data["downsamplex"]), int(data["downsampley"])
        if min(r1, r2, self.t_in, self.max_horizon) < 1:
            raise ValueError("Downsampling, history, and horizon values must be positive")

        height = int((raw.shape[1] - 1) / r1 + 1)
        width = int((raw.shape[2] - 1) / r2 + 1)
        self.spatial_shape = (height, width)
        training = raw[:ntrain, ::r1, ::r2, :][:, :height, :width, :]
        testing = raw[-ntest:, ::r1, ::r2, :][:, :height, :width, :]
        self.trajectories = training if split == "train" else testing
        self.frame_count = int(self.trajectories.shape[-1])
        if self.t_in + self.max_horizon > self.frame_count:
            raise ValueError(
                f"t_in + max_horizon={self.t_in + self.max_horizon} exceeds "
                f"trajectory length {self.frame_count}"
            )

        self.normalizer: dict[str, Tensor] | None = None
        if bool(data["normalize"]):
            if normalizer_state is None:
                values = torch.from_numpy(training).float()
                self.normalizer = {
                    "mean": values.mean(),
                    "std": values.std().clamp_min(1.0e-6),
                }
            else:
                self.normalizer = {
                    "mean": torch.as_tensor(normalizer_state["mean"], dtype=torch.float32),
                    "std": torch.as_tensor(normalizer_state["std"], dtype=torch.float32).clamp_min(1.0e-6),
                }

        grid_x = np.linspace(0, 1, width, dtype=np.float32)
        grid_y = np.linspace(0, 1, height, dtype=np.float32)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        self.pos = torch.from_numpy(np.c_[mesh_x.ravel(), mesh_y.ravel()].astype("float32"))
        self.set_horizon(horizon)

    @classmethod
    def _load_raw(cls, path: Path) -> np.ndarray:
        if path not in cls._RAW_CACHE:
            payload = scipy.io.loadmat(path)
            if "u" not in payload:
                raise KeyError(f"MAT file does not contain variable 'u': {path}")
            cls._RAW_CACHE[path] = payload["u"].astype("float32")
        return cls._RAW_CACHE[path]

    def set_horizon(self, horizon: int) -> None:
        horizon = int(horizon)
        if not 1 <= horizon <= self.max_horizon:
            raise ValueError(f"horizon must be in [1, {self.max_horizon}], got {horizon}")
        self.horizon = horizon
        self.windows_per_trajectory = self.frame_count - self.t_in - horizon + 1
        if self.windows_per_trajectory < 1:
            raise ValueError("Active horizon leaves no valid temporal windows")

    def get_normalizer_state(self) -> dict[str, float] | None:
        if self.normalizer is None:
            return None
        return {key: float(value.cpu()) for key, value in self.normalizer.items()}

    def decode_solution(self, values: Tensor) -> Tensor:
        if self.normalizer is None:
            return values
        mean = self.normalizer["mean"].to(values.device, values.dtype)
        std = self.normalizer["std"].to(values.device, values.dtype)
        return values * std + mean

    def __len__(self) -> int:
        return len(self.trajectories) * self.windows_per_trajectory

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        trajectory_index, start = divmod(index, self.windows_per_trajectory)
        trajectory = torch.from_numpy(self.trajectories[trajectory_index]).float()
        history = trajectory[..., start : start + self.t_in]
        future = trajectory[
            ..., start + self.t_in : start + self.t_in + self.horizon
        ]
        if self.normalizer is not None:
            history = (history - self.normalizer["mean"]) / self.normalizer["std"]
            future = (future - self.normalizer["mean"]) / self.normalizer["std"]
        point_count = self.spatial_shape[0] * self.spatial_shape[1]
        return {
            "pos": self.pos,
            "x": history.reshape(point_count, self.t_in * self.out_dim),
            "y": future.reshape(point_count, self.horizon * self.out_dim),
            "trajectory_index": torch.tensor(trajectory_index),
            "window_start": torch.tensor(start),
        }


class NavierStokesDataModule:
    """Own training/validation datasets and rebuild loaders after horizon changes."""

    def __init__(self, config: dict[str, Any], initial_horizon: int = 1) -> None:
        self.config = config
        self.train_dataset = NavierStokesWindowDataset(
            config, "train", initial_horizon
        )
        self.validation_dataset = NavierStokesWindowDataset(
            config,
            "val",
            int(config["data"]["max_horizon"]),
            self.train_dataset.get_normalizer_state(),
        )

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.train_dataset.spatial_shape

    def set_train_horizon(self, horizon: int) -> None:
        self.train_dataset.set_horizon(horizon)

    def get_normalizer_state(self) -> dict[str, float] | None:
        return self.train_dataset.get_normalizer_state()

    def _loader(self, dataset: NavierStokesWindowDataset, shuffle: bool) -> DataLoader:
        loader = self.config["dataloader"]
        return DataLoader(
            dataset,
            batch_size=int(loader["batch_size"]),
            shuffle=shuffle,
            num_workers=int(loader["num_workers"]),
            pin_memory=bool(loader["pin_memory"]),
            drop_last=bool(loader.get("drop_last", False)) if shuffle else False,
            persistent_workers=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def validation_dataloader(self) -> DataLoader:
        return self._loader(self.validation_dataset, shuffle=False)
