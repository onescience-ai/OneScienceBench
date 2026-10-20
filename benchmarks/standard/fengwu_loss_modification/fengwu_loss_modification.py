"""Weighted loss replacement for the FengWu training interface."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class FengwuWeightedLoss(nn.Module):
    """Compute grouped MSE for FengWu's six named variable tensors."""

    variable_names = ("surface", "z", "r", "u", "v", "t")

    def __init__(
        self,
        pressure_level: int = 37,
        surface_weight: float = 1.0,
        upper_air_weight: float = 1.0,
        level_weights: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__()
        if isinstance(pressure_level, bool) or not isinstance(pressure_level, int):
            raise TypeError("pressure_level must be an integer")
        if pressure_level <= 0:
            raise ValueError("pressure_level must be positive")
        self.pressure_level = pressure_level
        self.surface_weight = self._weight(surface_weight, "surface_weight")
        self.upper_air_weight = self._weight(upper_air_weight, "upper_air_weight")

        if level_weights is None:
            weights = None
        else:
            weights = torch.as_tensor(level_weights, dtype=torch.float32)
            if weights.ndim != 1 or weights.numel() != pressure_level:
                raise ValueError("level_weights must match pressure_level")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("level_weights must be finite and non-negative")
            if weights.sum() <= 0:
                raise ValueError("level_weights must have a positive sum")
        self.register_buffer("level_weights", weights)

    @staticmethod
    def _weight(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def forward(
        self,
        predictions: Sequence[Tensor],
        targets: Sequence[Tensor],
    ) -> Tensor:
        if not isinstance(predictions, (tuple, list)) or not isinstance(
            targets, (tuple, list)
        ):
            raise TypeError("predictions and targets must be tuple or list")
        if len(predictions) != len(self.variable_names) or len(targets) != len(
            self.variable_names
        ):
            raise ValueError("predictions and targets must contain six tensors")

        losses = []
        reference_geometry = None
        for index, (prediction, target) in enumerate(zip(predictions, targets)):
            name = self.variable_names[index]
            if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
                raise TypeError(f"{name} prediction and target must be tensors")
            if prediction.ndim != 4 or target.ndim != 4:
                raise ValueError(f"{name} tensors must have shape (B, C, H, W)")
            if prediction.shape != target.shape:
                raise ValueError(f"{name} prediction and target shapes must match")

            expected_channels = 4 if index == 0 else self.pressure_level
            if prediction.shape[1] != expected_channels:
                raise ValueError(f"{name} has an invalid channel count")
            geometry = (prediction.shape[0], prediction.shape[2:])
            if reference_geometry is None:
                reference_geometry = geometry
            elif geometry != reference_geometry:
                raise ValueError("all variables must share batch and grid dimensions")

            squared_error = (prediction - target).square()
            if index > 0 and self.level_weights is not None:
                level_error = squared_error.mean(dim=(0, 2, 3))
                weights = self.level_weights.to(level_error)
                losses.append((level_error * weights).mean())
            else:
                losses.append(squared_error.mean())

        upper_air_loss = torch.stack(losses[1:]).mean()
        return self.surface_weight * losses[0] + self.upper_air_weight * upper_air_loss
