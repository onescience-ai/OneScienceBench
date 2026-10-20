from __future__ import annotations

import torch
from torch import nn


class GraphCastHorizonWeightedLoss(nn.Module):
    """Weighted GraphCast loss for [batch, horizon, variable, lat, lon]."""

    def __init__(self, horizon_weights, variable_weights=None, eps: float = 1e-8):
        super().__init__()
        horizon_weights = torch.as_tensor(horizon_weights, dtype=torch.float32)
        if horizon_weights.ndim != 1 or horizon_weights.numel() == 0:
            raise ValueError("horizon_weights must be a non-empty 1D sequence")
        if not torch.isfinite(horizon_weights).all() or (horizon_weights < 0).any():
            raise ValueError("horizon_weights must be finite and non-negative")
        if float(horizon_weights.sum()) <= 0:
            raise ValueError("horizon_weights must contain a positive value")

        if variable_weights is None:
            variable_weights = torch.empty(0, dtype=torch.float32)
        variable_weights = torch.as_tensor(variable_weights, dtype=torch.float32)
        if variable_weights.ndim != 1:
            raise ValueError("variable_weights must be a 1D sequence")
        if not torch.isfinite(variable_weights).all() or (variable_weights < 0).any():
            raise ValueError("variable_weights must be finite and non-negative")
        if variable_weights.numel() and float(variable_weights.sum()) <= 0:
            raise ValueError("variable_weights must contain a positive value")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.register_buffer("horizon_weights", horizon_weights)
        self.register_buffer("variable_weights", variable_weights)
        self.eps = float(eps)

    def forward(self, predictions, targets, latitude_weights=None):
        if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("predictions and targets must be tensors")
        if predictions.shape != targets.shape or predictions.ndim != 5:
            raise ValueError("predictions and targets must share shape [B,T,C,H,W]")
        batch, horizons, variables, height, width = predictions.shape
        if horizons != self.horizon_weights.numel():
            raise ValueError("prediction horizon count does not match horizon_weights")
        if self.variable_weights.numel() not in (0, variables):
            raise ValueError("variable_weights length must match the variable dimension")

        if latitude_weights is None:
            latitude_weights = predictions.new_ones(height)
        latitude_weights = torch.as_tensor(
            latitude_weights, dtype=predictions.dtype, device=predictions.device
        )
        if latitude_weights.ndim == 1:
            if latitude_weights.numel() != height:
                raise ValueError("1D latitude_weights length must equal H")
            latitude_weights = latitude_weights[:, None].expand(height, width)
        elif latitude_weights.shape != (height, width):
            raise ValueError("latitude_weights must have shape [H] or [H,W]")
        if not torch.isfinite(latitude_weights).all() or (latitude_weights < 0).any():
            raise ValueError("latitude_weights must be finite and non-negative")
        if float(latitude_weights.sum()) <= 0:
            raise ValueError("latitude_weights must contain a positive value")

        horizon = self.horizon_weights.to(predictions).view(1, horizons, 1, 1, 1)
        if self.variable_weights.numel():
            variable = self.variable_weights.to(predictions).view(1, 1, variables, 1, 1)
        else:
            variable = predictions.new_ones(1, 1, variables, 1, 1)
        latitude = latitude_weights.view(1, 1, 1, height, width)
        weights = horizon * variable * latitude
        squared_error = (predictions - targets).square()
        numerator = (squared_error * weights).sum(dim=(1, 2, 3, 4))
        denominator = weights.sum().clamp_min(self.eps)
        return (numerator / denominator).mean()
