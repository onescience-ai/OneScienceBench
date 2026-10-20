"""Masked temporal-fusion style probabilistic station forecaster."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


class TFTProbabilisticForecastModel(nn.Module):
    def __init__(self, variables: int = 4, time_features: int = 3, context_length: int = 12, forecast_horizon: int = 6, d_model: int = 16):
        super().__init__()
        if min(variables, time_features, context_length, forecast_horizon, d_model) <= 0:
            raise ValueError("invalid TFT configuration")
        self.variables, self.time_features, self.context_length, self.forecast_horizon, self.d_model = variables, time_features, context_length, forecast_horizon, d_model
        self.variable_encoder = nn.Linear(variables, d_model)
        self.time_encoder = nn.Linear(time_features, d_model)
        self.temporal_attention = nn.MultiheadAttention(d_model, 2, batch_first=True)
        self.quantile_head = nn.Linear(d_model, variables * 3)
        self.quantile_levels = (0.1, 0.5, 0.9)

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"values", "time_features", "observed_mask"}
        if not isinstance(batch, dict) or not required.issubset(batch):
            raise ValueError("batch requires values, time_features and observed_mask")
        values, times, mask = batch["values"], batch["time_features"], batch["observed_mask"]
        if not all(isinstance(item, torch.Tensor) for item in (values, times, mask)):
            raise ValueError("values, time_features and observed_mask must be tensors")
        if values.ndim != 3 or values.shape[1:] != (self.context_length, self.variables):
            raise ValueError("invalid values/time_features shape")
        if times.shape != (values.shape[0], self.context_length, self.time_features):
            raise ValueError("invalid values/time_features shape")
        if mask.shape != values.shape or mask.dtype != torch.bool or not mask.flatten(1).any(dim=1).all():
            raise ValueError("observed_mask must be boolean with at least one observation per sample")
        if not torch.isfinite(values).all() or not torch.isfinite(times).all():
            raise ValueError("inputs must be finite")
        masked = values * mask.to(values.dtype)
        encoded = self.variable_encoder(masked) + self.time_encoder(times)
        attended, _ = self.temporal_attention(encoded, encoded, encoded, key_padding_mask=~mask.any(dim=2))
        context = attended[:, -1:].expand(-1, self.forecast_horizon, -1)
        raw = self.quantile_head(context).view(values.shape[0], self.forecast_horizon, self.variables, 3)
        q10 = raw[..., 0]
        upper = torch.full_like(q10, float("inf"))
        q50 = torch.nextafter(q10 + torch.nn.functional.softplus(raw[..., 1]), upper)
        q90 = torch.nextafter(q50 + torch.nn.functional.softplus(raw[..., 2]), upper)
        quantiles = torch.stack((q10, q50, q90), dim=-1)
        return {"quantiles": quantiles, "point": q50}

    forward = predict


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> TFTProbabilisticForecastModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"model_state_dict"}:
        raise ValueError("checkpoint must contain exactly model_state_dict")
    model = TFTProbabilisticForecastModel().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()
