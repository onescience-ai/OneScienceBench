"""Navier--Stokes data pipeline for the G-FNO experiment.

Responsibilities in this file only:
1. read MATLAB/HDF5 data;
2. validate and preprocess trajectory layout;
3. reproduce the official train/valid/test split;
4. optional train-only-fitted normalization;
5. build teacher-forcing and rollout datasets/loaders.
"""

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset


class UnitGaussianNormalizer:
    """Channel-wise Gaussian normalizer fitted on training data only.

    For NS there is one physical channel (vorticity), so mean/std are scalars.
    Statistics are computed over N, X, Y and T dimensions, never using
    validation/test samples.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    @classmethod
    def fit(cls, x: torch.Tensor, eps: float = 1e-6) -> "UnitGaussianNormalizer":
        if x.ndim != 5:
            raise ValueError(f"expected [N,X,Y,T,C], got {tuple(x.shape)}")
        reduce_dims = tuple(range(x.ndim - 1))
        mean = x.mean(dim=reduce_dims, keepdim=True)
        std = x.std(dim=reduce_dims, keepdim=True, unbiased=False)
        return cls(mean=mean, std=std, eps=eps)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * (std + self.eps) + mean

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.cpu(),
            "std": self.std.cpu(),
            "eps": self.eps,
        }


class NSDataset(Dataset):
    """Teacher-forcing dataset for training and rollout dataset for evaluation.

    Input trajectory layout:
        [trajectory, x, y, time, channel]

    train=True:
        each trajectory is expanded into T_out one-step pairs
        previous T_in frames -> next frame

    train=False:
        one item per trajectory
        first T_in frames -> all T_out future frames
    """

    def __init__(
        self,
        data: torch.Tensor,
        T_in: int = 10,
        T_out: int = 20,
        *,
        train: bool = True,
    ) -> None:
        if data.ndim != 5:
            raise ValueError(f"expected [N,X,Y,T,C], got {tuple(data.shape)}")
        if data.shape[-2] < T_in + T_out:
            raise ValueError(
                f"need at least {T_in + T_out} frames, got {data.shape[-2]}"
            )

        self.data = data[..., : T_in + T_out, :]
        self.T_in = T_in
        self.T_out = T_out
        self.train = train

    def __len__(self) -> int:
        return len(self.data) * self.T_out if self.train else len(self.data)

    def __getitem__(self, idx: int):
        if not self.train:
            pde = self.data[idx]
            x = pde[..., : self.T_in, :]
            y = pde[..., self.T_in : self.T_in + self.T_out, :]
            return x, y

        traj_idx = idx // self.T_out
        step = idx % self.T_out
        t = self.T_in + step
        pde = self.data[traj_idx]
        x = pde[..., t - self.T_in : t, :]
        y = pde[..., t, :]
        return x, y


def load_ns_mat(path: str, key: str = "u") -> torch.Tensor:
    """Read ``u`` from classic MAT or MATLAB v7.3/HDF5.

    Returns:
        float32 tensor [N, X, Y, T, 1]
    """

    path = os.path.expanduser(os.path.expandvars(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        # MATLAB v7.3 files are HDF5-backed and their axes are reversed.
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"key '{key}' not found in {path}")
            data = np.array(f[key])
        data = np.transpose(data, axes=range(data.ndim - 1, -1, -1))
    except OSError:
        mat = scipy.io.loadmat(path)
        if key not in mat:
            raise KeyError(f"key '{key}' not found in {path}")
        data = mat[key]

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(
            f"expected MAT variable '{key}' as [N,X,Y,T], got {data.shape}"
        )
    return torch.from_numpy(data[..., None])


def official_ns_split(
    data: torch.Tensor,
    ntrain: int = 1000,
    nvalid: int = 100,
    ntest: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the official non-SWE split used by G-FNO experiments.py.

    train = first ntrain trajectories
    valid = nvalid trajectories immediately before the final test block
    test  = final ntest trajectories
    """

    required = ntrain + nvalid + ntest
    if len(data) < required:
        raise ValueError(f"have {len(data)} trajectories, need at least {required}")

    train = data[:ntrain]
    test = data[-ntest:]
    valid = data[-(ntest + nvalid) : -ntest]
    return train, valid, test


def preprocess_ns(
    data: torch.Tensor,
    *,
    T_in: int,
    T_out: int,
    expected_resolution: Optional[int] = 64,
) -> torch.Tensor:
    """Basic validation/cropping before splitting."""

    if data.ndim != 5:
        raise ValueError(f"expected [N,X,Y,T,C], got {tuple(data.shape)}")
    if data.shape[-2] < T_in + T_out:
        raise ValueError(
            f"need >= {T_in + T_out} time frames, got {data.shape[-2]}"
        )
    if expected_resolution is not None:
        if data.shape[1] != expected_resolution or data.shape[2] != expected_resolution:
            raise ValueError(
                f"expected {expected_resolution}x{expected_resolution}, "
                f"got {tuple(data.shape[1:3])}"
            )
    return data[..., : T_in + T_out, :].contiguous()


@dataclass
class NSDataBundle:
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    train_trajectories: torch.Tensor
    valid_trajectories: torch.Tensor
    test_trajectories: torch.Tensor
    normalizer: Optional[UnitGaussianNormalizer]


def build_dataloaders(
    *,
    data_path: str,
    T_in: int = 10,
    T_out: int = 20,
    ntrain: int = 1000,
    nvalid: int = 100,
    ntest: int = 100,
    batch_size: int = 20,
    num_workers: int = 0,
    expected_resolution: Optional[int] = 64,
    normalization: str = "none",
    normalization_eps: float = 1e-6,
    pin_memory: bool = True,
) -> NSDataBundle:
    """Complete NS data preparation entry point.

    ``normalization='none'`` reproduces the official G-FNO NS setup.
    ``normalization='unit_gaussian'`` is provided for controlled experiments;
    its statistics are fitted only on the training split.
    """

    data = load_ns_mat(data_path)
    data = preprocess_ns(
        data,
        T_in=T_in,
        T_out=T_out,
        expected_resolution=expected_resolution,
    )
    train, valid, test = official_ns_split(data, ntrain, nvalid, ntest)

    normalizer: Optional[UnitGaussianNormalizer] = None
    normalization = normalization.lower()
    if normalization == "unit_gaussian":
        normalizer = UnitGaussianNormalizer.fit(train, eps=normalization_eps)
        train = normalizer.encode(train)
        valid = normalizer.encode(valid)
        test = normalizer.encode(test)
    elif normalization != "none":
        raise ValueError(
            f"unknown normalization='{normalization}', use 'none' or 'unit_gaussian'"
        )

    train_dataset = NSDataset(train, T_in=T_in, T_out=T_out, train=True)
    valid_dataset = NSDataset(valid, T_in=T_in, T_out=T_out, train=False)
    test_dataset = NSDataset(test, T_in=T_in, T_out=T_out, train=False)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return NSDataBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_trajectories=train,
        valid_trajectories=valid,
        test_trajectories=test,
        normalizer=normalizer,
    )
