"""Shared configuration, metric, physics-loss, logging and checkpoint utilities."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
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
        raise RuntimeError(f"CUDA/DCU requested but unavailable: {requested}")
    return device


def setup_logger(output_dir: Path, name: str = "rno_darcy_multiscale") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "train.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class RelativeL2Loss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", eps: float = 1.0e-12):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction, self.eps = reduction, eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = prediction.shape[0]
        numerator = torch.linalg.vector_norm((prediction - target).reshape(batch, -1), dim=1)
        denominator = torch.linalg.vector_norm(target.reshape(batch, -1), dim=1).clamp_min(self.eps)
        values = numerator / denominator
        return values.sum() if self.reduction == "sum" else values.mean() if self.reduction == "mean" else values


def enforce_zero_dirichlet(field: torch.Tensor, resolution: int) -> torch.Tensor:
    shaped = field.reshape(field.shape[0], resolution, resolution).clone()
    shaped[:, (0, -1), :] = 0.0
    shaped[:, :, (0, -1)] = 0.0
    return shaped.reshape(field.shape[0], -1)


def central_difference(field: torch.Tensor, resolution: int, spacing: float) -> tuple[torch.Tensor, torch.Tensor]:
    shaped = field.reshape(field.shape[0], resolution, resolution, 1)
    padded = F.pad(shaped, (0, 0, 1, 1, 1, 1), mode="constant", value=0.0)
    grad_x = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * spacing)
    grad_y = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * spacing)
    return grad_x, grad_y


def curriculum_scale(epoch: int, total_epochs: int, start_fraction: float, end_fraction: float) -> float:
    """Linear 0-to-1 curriculum between configured fractions of training."""
    if total_epochs <= 1:
        return 1.0
    progress = epoch / max(total_epochs - 1, 1)
    if progress <= start_fraction:
        return 0.0
    if progress >= end_fraction:
        return 1.0
    return (progress - start_fraction) / max(end_fraction - start_fraction, 1.0e-12)


def darcy_pde_residual_loss(
    prediction: torch.Tensor,
    coefficient: torch.Tensor,
    importance: torch.Tensor,
    resolution: int,
    spacing: float,
    forcing: float = 1.0,
    coefficient_average: str = "harmonic",
) -> torch.Tensor:
    """Finite-volume residual for ``-div(a grad(u)) = f`` on interior cells."""
    u = prediction.reshape(-1, resolution, resolution)
    a = coefficient.reshape(-1, resolution, resolution).clamp_min(1.0e-8)
    if coefficient_average == "harmonic":
        ax = 2.0 * a[:, :, :-1] * a[:, :, 1:] / (a[:, :, :-1] + a[:, :, 1:]).clamp_min(1.0e-8)
        ay = 2.0 * a[:, :-1, :] * a[:, 1:, :] / (a[:, :-1, :] + a[:, 1:, :]).clamp_min(1.0e-8)
    elif coefficient_average == "arithmetic":
        ax = 0.5 * (a[:, :, :-1] + a[:, :, 1:])
        ay = 0.5 * (a[:, :-1, :] + a[:, 1:, :])
    else:
        raise ValueError(f"Unsupported coefficient average: {coefficient_average}")
    flux_x = ax * (u[:, :, 1:] - u[:, :, :-1]) / spacing
    flux_y = ay * (u[:, 1:, :] - u[:, :-1, :]) / spacing
    divergence = (
        (flux_x[:, 1:-1, 1:] - flux_x[:, 1:-1, :-1])
        + (flux_y[:, 1:, 1:-1] - flux_y[:, :-1, 1:-1])
    ) / spacing
    residual = -divergence - forcing
    weights = importance.reshape(-1, resolution, resolution)[:, 1:-1, 1:-1]
    return (weights * residual.square()).sum() / weights.sum().clamp_min(1.0) / max(forcing**2, 1.0e-12)


def zero_dirichlet_boundary_loss(
    prediction: torch.Tensor, target: torch.Tensor, resolution: int
) -> torch.Tensor:
    prediction = prediction.reshape(-1, resolution, resolution)
    pred_boundary = torch.cat(
        (prediction[:, 0], prediction[:, -1], prediction[:, 1:-1, 0], prediction[:, 1:-1, -1]), dim=1
    )
    return pred_boundary.square().mean()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: Iterable, y_normalizer: Any, device: torch.device,
    spacing: float | None = None,
) -> dict[str, float]:
    """Report physical-space metrics; gradient Relative L2 follows the training convention.

    The gradient metric is the sum of x/y relative errors, averaged over samples.
    It uses the existing zero-boundary prediction and central differences, without
    applying the training loss's 0.1 weight. The original three metrics are unchanged.
    """
    model.eval()
    rel_loss = RelativeL2Loss(reduction="sum")
    totals = {"relative_l2": 0.0, "gradient_relative_l2": 0.0, "mse": 0.0, "mae": 0.0, "samples": 0}
    point_count = 0
    for positions, fine_coefficients, coarse_coefficients, targets, _importance in loader:
        positions = positions.to(device, non_blocking=True)
        fine_coefficients = fine_coefficients.to(device, non_blocking=True)
        coarse_coefficients = coarse_coefficients.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction_n = model(
            positions, fine_coefficients.unsqueeze(-1), coarse_coefficients.unsqueeze(-1)
        ).squeeze(-1)
        prediction = y_normalizer.decode(prediction_n)
        totals["relative_l2"] += float(rel_loss(prediction, targets).item())
        resolution = int(targets.shape[1] ** 0.5)
        grid_spacing = 1.0 / resolution if spacing is None else spacing
        prediction_bc = enforce_zero_dirichlet(prediction, resolution)
        pred_grad_x, pred_grad_y = central_difference(prediction_bc, resolution, grid_spacing)
        target_grad_x, target_grad_y = central_difference(targets, resolution, grid_spacing)
        gradient_relative_l2 = rel_loss(pred_grad_x, target_grad_x) + rel_loss(pred_grad_y, target_grad_y)
        totals["gradient_relative_l2"] += float(gradient_relative_l2.item())
        totals["mse"] += float(F.mse_loss(prediction, targets, reduction="sum").item())
        totals["mae"] += float(F.l1_loss(prediction, targets, reduction="sum").item())
        totals["samples"] += targets.shape[0]
        point_count += targets.numel()
    if totals["samples"] == 0:
        raise RuntimeError("Evaluation loader is empty")
    return {
        "relative_l2": totals["relative_l2"] / totals["samples"],
        "gradient_relative_l2": totals["gradient_relative_l2"] / totals["samples"],
        "mse": totals["mse"] / point_count,
        "mae": totals["mae"] / point_count,
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


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device, strict: bool = True) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint
