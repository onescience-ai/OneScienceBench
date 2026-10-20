"""Shared configuration, reproducibility, metrics, and persistence utilities."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-sample relative L2, matching Eq. (25) of the paper."""
    pred = prediction.reshape(prediction.shape[0], -1)
    truth = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.vector_norm(pred - truth, dim=1)
    denominator = torch.linalg.vector_norm(truth, dim=1).clamp_min(1.0e-12)
    return (numerator / denominator).mean()


def spatial_gradient(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Second-order interior / first-order boundary finite differences."""
    gx = torch.empty_like(field)
    gy = torch.empty_like(field)
    gx[..., 1:-1, :] = 0.5 * (field[..., 2:, :] - field[..., :-2, :])
    gx[..., 0, :] = field[..., 1, :] - field[..., 0, :]
    gx[..., -1, :] = field[..., -1, :] - field[..., -2, :]
    gy[..., :, 1:-1] = 0.5 * (field[..., :, 2:] - field[..., :, :-2])
    gy[..., :, 0] = field[..., :, 1] - field[..., :, 0]
    gy[..., :, -1] = field[..., :, -1] - field[..., :, -2]
    return gx, gy


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px, py = spatial_gradient(prediction)
    tx, ty = spatial_gradient(target)
    return 0.5 * (relative_l2(px, tx) + relative_l2(py, ty))


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def append_jsonl(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def append_csv(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(payload)


def prepare_output_dirs(config: Mapping[str, Any], project_dir: str | Path) -> Dict[str, Path]:
    output = config["output"]
    root = Path(output["root"])
    if not root.is_absolute():
        root = Path(project_dir) / root
    paths = {
        "root": root,
        "checkpoints": root / output["checkpoints"],
        "logs": root / output["logs"],
        "results": root / output["results"],
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def move_batch(batch: Iterable[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch)
