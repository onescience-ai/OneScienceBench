"""Configuration, device, metric, and autoregressive rollout helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("root"), dict):
        raise ValueError(f"Config must contain a root mapping: {path}")
    return config["root"]


def project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout(
    model: torch.nn.Module,
    pos: torch.Tensor,
    state: torch.Tensor,
    t_out: int,
    out_dim: int,
    teacher_forcing: bool = False,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    predictions = []
    for step in range(t_out):
        prediction = model(pos, state)
        predictions.append(prediction)
        next_frame = target[..., step * out_dim : (step + 1) * out_dim] if teacher_forcing and target is not None else prediction
        state = torch.cat((state[..., out_dim:], next_frame), dim=-1)
    return torch.cat(predictions, dim=-1)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.vector_norm(prediction - target, dim=1)
    denominator = torch.linalg.vector_norm(target, dim=1).clamp_min(1.0e-8)
    return numerator / denominator
