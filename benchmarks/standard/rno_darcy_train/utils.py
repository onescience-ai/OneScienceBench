"""Configuration, reproducibility, metrics, logging, and checkpoint utilities."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA/DCU device requested but torch.cuda.is_available() is false: {requested}")
    return device


def setup_logger(output_dir: Path, name: str = "rno_darcy") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class RelativeL2Loss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", eps: float = 1.0e-12):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = prediction.shape[0]
        numerator = torch.linalg.vector_norm((prediction - target).reshape(batch, -1), dim=1)
        denominator = torch.linalg.vector_norm(target.reshape(batch, -1), dim=1).clamp_min(self.eps)
        values = numerator / denominator
        if self.reduction == "sum":
            return values.sum()
        if self.reduction == "mean":
            return values.mean()
        return values


def enforce_zero_dirichlet(field: torch.Tensor, resolution: int) -> torch.Tensor:
    shaped = field.reshape(field.shape[0], resolution, resolution).clone()
    shaped[:, 0, :] = 0.0
    shaped[:, -1, :] = 0.0
    shaped[:, :, 0] = 0.0
    shaped[:, :, -1] = 0.0
    return shaped.reshape(field.shape[0], -1)


def central_difference(field: torch.Tensor, resolution: int, spacing: float) -> Tuple[torch.Tensor, torch.Tensor]:
    shaped = field.reshape(field.shape[0], resolution, resolution, 1)
    padded = F.pad(shaped, (0, 0, 1, 1, 1, 1), mode="constant", value=0.0)
    grad_x = (padded[:, 1:-1, 2:, :] - padded[:, 1:-1, :-2, :]) / (2.0 * spacing)
    grad_y = (padded[:, 2:, 1:-1, :] - padded[:, :-2, 1:-1, :]) / (2.0 * spacing)
    return grad_x, grad_y


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    y_normalizer: Any,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    rel_loss = RelativeL2Loss(reduction="sum")
    totals = {"relative_l2": 0.0, "mse": 0.0, "mae": 0.0, "samples": 0}
    for positions, coefficients, targets in loader:
        positions = positions.to(device, non_blocking=True)
        coefficients = coefficients.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction_normalized = model(positions, coefficients.unsqueeze(-1)).squeeze(-1)
        prediction = y_normalizer.decode(prediction_normalized)
        batch = targets.shape[0]
        totals["relative_l2"] += float(rel_loss(prediction, targets).item())
        totals["mse"] += float(F.mse_loss(prediction, targets, reduction="sum").item())
        totals["mae"] += float(F.l1_loss(prediction, targets, reduction="sum").item())
        totals["samples"] += batch

    if totals["samples"] == 0:
        raise RuntimeError("Evaluation loader is empty")
    points = totals["samples"] * targets.shape[1]
    return {
        "relative_l2": totals["relative_l2"] / totals["samples"],
        "mse": totals["mse"] / points,
        "mae": totals["mae"] / points,
        "samples": int(totals["samples"]),
    }


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device, strict: bool = True) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint

