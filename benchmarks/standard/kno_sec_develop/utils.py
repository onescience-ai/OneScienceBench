"""Training utilities for curriculum rollout and composite physical/spectral loss."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("root"), dict):
        raise ValueError(f"Config must contain a root mapping: {path}")
    return payload["root"]


def project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def resolve_data_path(config: dict[str, Any], project_root: Path) -> Path:
    source = config["datapipe"]["source"]
    directory = project_path(source["data_dir"], project_root).resolve()
    source["data_dir"] = str(directory)
    return directory / source["file_name"]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def curriculum_state(
    epoch: int, total_epochs: int, config: dict[str, Any]
) -> tuple[int, float]:
    """Select rollout horizon and a linearly decayed teacher-forcing ratio."""

    horizons = [int(value) for value in config["horizons"]]
    fractions = [float(value) for value in config["stage_fractions"]]
    if len(horizons) != len(fractions) or not horizons:
        raise ValueError("horizons and stage_fractions must have equal non-zero length")
    if fractions[0] != 0.0 or fractions != sorted(fractions) or fractions[-1] >= 1.0:
        raise ValueError("stage_fractions must be sorted, start at 0, and remain below 1")
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    horizon = horizons[0]
    for candidate, start in zip(horizons, fractions):
        if progress >= start:
            horizon = candidate
    start_ratio = float(config["teacher_forcing_start"])
    end_ratio = float(config["teacher_forcing_end"])
    teacher_forcing_ratio = start_ratio + progress * (end_ratio - start_ratio)
    return horizon, min(1.0, max(0.0, teacher_forcing_ratio))


def autoregressive_rollout(
    model: nn.Module,
    pos: Tensor,
    state: Tensor,
    target: Tensor,
    horizon: int,
    out_dim: int,
    teacher_forcing_ratio: float,
    training: bool,
) -> Tensor:
    """Roll out with per-sample scheduled sampling and differentiable feedback."""

    predictions = []
    for step in range(horizon):
        prediction = model(pos, state)
        predictions.append(prediction)
        target_frame = target[..., step * out_dim : (step + 1) * out_dim]
        if training and teacher_forcing_ratio > 0.0:
            selector = torch.rand(
                (state.shape[0], 1, 1), device=state.device
            ) < teacher_forcing_ratio
            feedback = torch.where(selector, target_frame, prediction)
        else:
            feedback = prediction
        state = torch.cat((state[..., out_dim:], feedback), dim=-1)
    return torch.cat(predictions, dim=-1)


def relative_l2(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.vector_norm(prediction - target, dim=1)
    denominator = torch.linalg.vector_norm(target, dim=1).clamp_min(1.0e-8)
    return numerator / denominator


def spectral_relative_l2(
    prediction: Tensor, target: Tensor, spatial_shape: tuple[int, int]
) -> Tensor:
    height, width = spatial_shape
    channels = prediction.shape[-1]
    prediction_grid = prediction.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
    target_grid = target.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
    prediction_spectrum = torch.fft.rfft2(prediction_grid)
    target_spectrum = torch.fft.rfft2(target_grid)
    difference = (prediction_spectrum - target_spectrum).reshape(prediction.shape[0], -1)
    reference = target_spectrum.reshape(target.shape[0], -1)
    return torch.linalg.vector_norm(difference, dim=1) / torch.linalg.vector_norm(
        reference, dim=1
    ).clamp_min(1.0e-8)


def composite_rollout_loss(
    prediction: Tensor,
    target: Tensor,
    spatial_shape: tuple[int, int],
    horizon: int,
    out_dim: int,
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine time-weighted MSE, physical Relative-L2, and spectral error."""

    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    far_weight = float(config.get("far_step_weight", 2.0))
    time_weights = torch.linspace(1.0, far_weight, horizon, device=prediction.device)
    time_weights = time_weights / time_weights.sum()
    mse_terms, relative_terms, spectral_terms = [], [], []
    for step in range(horizon):
        selection = slice(step * out_dim, (step + 1) * out_dim)
        predicted_frame = prediction[..., selection]
        target_frame = target[..., selection]
        mse_terms.append(F.mse_loss(predicted_frame, target_frame))
        relative_terms.append(relative_l2(predicted_frame, target_frame).mean())
        spectral_terms.append(
            spectral_relative_l2(predicted_frame, target_frame, spatial_shape).mean()
        )
    mse = torch.sum(torch.stack(mse_terms) * time_weights)
    relative = torch.sum(torch.stack(relative_terms) * time_weights)
    spectral = torch.sum(torch.stack(spectral_terms) * time_weights)
    total = (
        float(config["mse_weight"]) * mse
        + float(config["relative_l2_weight"]) * relative
        + float(config["spectral_weight"]) * spectral
    )
    return total, {"mse": mse, "relative_l2": relative, "spectral": spectral}


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)
