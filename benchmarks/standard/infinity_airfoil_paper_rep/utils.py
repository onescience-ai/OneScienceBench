"""Shared utilities for the flattened INFINITY AirfRANS experiment.

This module intentionally contains only cross-cutting infrastructure used by
training and inference. Data-specific preprocessing stays in ``datset.py`` and
model-specific logic stays in ``model.py``.
"""
from __future__ import annotations

from pathlib import Path
import random
from typing import Iterator, Mapping

import numpy as np
import torch
from torch import Tensor, nn
import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in config file {path!s}.")
    return config


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device, with an explicit CUDA-to-CPU fallback."""
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def checkpoint_dir(config: Mapping) -> Path:
    """Return and create the configured checkpoint directory."""
    path = Path(config["runtime"].get("checkpoint_dir", "checkpoints"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_index_batches(
    n_items: int,
    batch_size: int,
    generator: torch.Generator,
) -> Iterator[list[int]]:
    """Yield shuffled index batches without materializing a DataLoader."""
    order = torch.randperm(n_items, generator=generator).tolist()
    for start in range(0, n_items, batch_size):
        yield order[start : start + batch_size]


def fit_code_stats(codes: Mapping[str, Tensor], eps: float = 1e-8) -> dict[str, dict[str, Tensor]]:
    """Fit per-latent-dimension z-score statistics from training codes."""
    return {
        field: {
            "mean": tensor.mean(dim=0),
            "std": tensor.std(dim=0).clamp_min(eps),
        }
        for field, tensor in codes.items()
    }


def normalize_codes(codes: Mapping[str, Tensor], stats: Mapping) -> dict[str, Tensor]:
    """Z-score latent codes using train-set statistics."""
    return {
        field: (tensor - stats[field]["mean"]) / stats[field]["std"]
        for field, tensor in codes.items()
    }


def move_code_stats(stats: Mapping, device: torch.device) -> dict[str, dict[str, Tensor]]:
    """Move latent-code normalization statistics to a target device."""
    return {
        field: {
            "mean": values["mean"].to(device),
            "std": values["std"].to(device),
        }
        for field, values in stats.items()
    }


def freeze_module(module: nn.Module) -> nn.Module:
    """Switch a module to evaluation mode and disable parameter gradients."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def decode_in_chunks(
    model: nn.Module,
    coordinates: Tensor,
    code: Tensor,
    chunk_size: int,
) -> Tensor:
    """Decode a large coordinate set in memory-safe chunks and return CPU data."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = []
    with torch.no_grad():
        for start in range(0, coordinates.shape[0], chunk_size):
            chunks.append(model(coordinates[start : start + chunk_size], code).cpu())
    return torch.cat(chunks, dim=0)
