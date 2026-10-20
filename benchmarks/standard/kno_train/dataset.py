"""Standalone Navier--Stokes MAT data pipeline used by KNO.

This module intentionally depends only on NumPy, SciPy, and PyTorch.  Its split,
downsampling, coordinate, and normalization semantics match the checkpoint's
original OneScience data pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset


class NavierStokesDataset(Dataset[dict[str, torch.Tensor]]):
    """Transolver Navier--Stokes benchmark split with flattened grid samples."""

    _RAW_CACHE: dict[Path, np.ndarray] = {}

    def __init__(
        self,
        config: dict[str, Any],
        mode: str,
        normalizer_state: dict[str, float] | None = None,
    ) -> None:
        if mode not in {"train", "test", "val"}:
            raise ValueError(f"Unsupported split: {mode}")
        self.mode = mode
        source = config["source"]
        data = config["data"]
        data_file = (Path(source["data_dir"]).expanduser() / source["file_name"]).resolve()
        if not data_file.is_file():
            raise FileNotFoundError(f"Navier--Stokes data file not found: {data_file}")

        raw = self._load_raw(data_file)
        r1, r2 = int(data["downsamplex"]), int(data["downsampley"])
        t_in, t_out, out_dim = int(data["t_in"]), int(data["t_out"]), int(data["out_dim"])
        ntrain, ntest = int(data["ntrain"]), int(data["ntest"])
        if min(r1, r2, t_in, t_out, ntrain, ntest) < 1:
            raise ValueError("Split, time-window, and downsampling values must be positive")
        if out_dim != 1:
            raise ValueError("The benchmark contains one scalar field; out_dim must be 1")
        if raw.shape[0] < ntrain + ntest:
            raise ValueError(f"Requested {ntrain + ntest} samples, file contains {raw.shape[0]}")
        if raw.shape[-1] < t_in + t_out:
            raise ValueError(f"Requested {t_in + t_out} frames, file contains {raw.shape[-1]}")

        height = int((raw.shape[1] - 1) / r1 + 1)
        width = int((raw.shape[2] - 1) / r2 + 1)
        self.spatial_shape = (height, width)
        train_raw = raw[:ntrain, ::r1, ::r2, :][:, :height, :width, :]
        test_raw = raw[-ntest:, ::r1, ::r2, :][:, :height, :width, :]
        selected = train_raw if mode == "train" else test_raw
        count = selected.shape[0]
        self.x = torch.from_numpy(selected[..., :t_in].reshape(count, -1, t_in * out_dim))
        self.y = torch.from_numpy(
            selected[..., t_in : t_in + t_out].reshape(count, -1, t_out * out_dim)
        )

        self.normalizer: dict[str, torch.Tensor] | None = None
        if bool(data["normalize"]):
            if normalizer_state is None:
                train_x = torch.from_numpy(train_raw[..., :t_in].reshape(ntrain, -1, t_in * out_dim))
                train_y = torch.from_numpy(
                    train_raw[..., t_in : t_in + t_out].reshape(ntrain, -1, t_out * out_dim)
                )
                values = torch.cat((train_x.reshape(-1), train_y.reshape(-1)))
                self.normalizer = {"mean": values.mean(), "std": values.std().clamp_min(1.0e-6)}
            else:
                self.normalizer = {
                    "mean": torch.as_tensor(normalizer_state["mean"], dtype=torch.float32),
                    "std": torch.as_tensor(normalizer_state["std"], dtype=torch.float32).clamp_min(1.0e-6),
                }
            self.x = (self.x - self.normalizer["mean"]) / self.normalizer["std"]
            self.y = (self.y - self.normalizer["mean"]) / self.normalizer["std"]

        grid_x = np.linspace(0, 1, width, dtype=np.float32)
        grid_y = np.linspace(0, 1, height, dtype=np.float32)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        self.pos = torch.from_numpy(np.c_[mesh_x.ravel(), mesh_y.ravel()].astype("float32"))

    @classmethod
    def _load_raw(cls, path: Path) -> np.ndarray:
        if path not in cls._RAW_CACHE:
            payload = scipy.io.loadmat(path)
            if "u" not in payload:
                raise KeyError(f"MAT file does not contain variable 'u': {path}")
            cls._RAW_CACHE[path] = payload["u"].astype("float32")
        return cls._RAW_CACHE[path]

    def get_normalizer_state(self) -> dict[str, float] | None:
        if self.normalizer is None:
            return None
        return {key: float(value.cpu()) for key, value in self.normalizer.items()}

    def decode_solution(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return tensor
        mean = self.normalizer["mean"].to(device=tensor.device, dtype=tensor.dtype)
        std = self.normalizer["std"].to(device=tensor.device, dtype=tensor.dtype)
        return tensor * std + mean

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"pos": self.pos, "x": self.x[index], "y": self.y[index]}


class NavierStokesDatapipe:
    """Small compatibility wrapper with the original checkpoint data contract."""

    def __init__(
        self,
        config: dict[str, Any],
        normalizer_state: dict[str, float] | None = None,
    ) -> None:
        self.config = config
        self.train_dataset = NavierStokesDataset(config, "train", normalizer_state)
        state = self.train_dataset.get_normalizer_state()
        self.test_dataset = NavierStokesDataset(config, "test", state)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.train_dataset.spatial_shape

    def get_normalizer_state(self) -> dict[str, float] | None:
        return self.train_dataset.get_normalizer_state()

    def decode_solution(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.train_dataset.decode_solution(tensor)

    def _loader(self, dataset: NavierStokesDataset, shuffle: bool) -> DataLoader:
        loader = self.config["dataloader"]
        return DataLoader(
            dataset,
            batch_size=int(loader["batch_size"]),
            shuffle=shuffle,
            num_workers=int(loader["num_workers"]),
            pin_memory=bool(loader["pin_memory"]),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
