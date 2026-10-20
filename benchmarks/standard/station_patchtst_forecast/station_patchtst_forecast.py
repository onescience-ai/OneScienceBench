"""Compact PatchTST-style multi-station forecaster."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


class PatchTSTForecastModel(nn.Module):
    def __init__(self, variables: int = 4, context_length: int = 12, patch_length: int = 4, forecast_horizon: int = 6, d_model: int = 16):
        super().__init__()
        if min(variables, context_length, patch_length, forecast_horizon, d_model) <= 0 or patch_length > context_length:
            raise ValueError("invalid PatchTST configuration")
        self.variables, self.context_length, self.patch_length, self.forecast_horizon, self.d_model = variables, context_length, patch_length, forecast_horizon, d_model
        self.patch_projection = nn.Linear(patch_length, d_model)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 2, 2 * d_model, batch_first=True), 2)
        self.head = nn.Linear(d_model, forecast_horizon)

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.context_length or values.shape[2] != self.variables:
            raise ValueError("values must have shape [B,context_length,variables]")
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")
        x = values.transpose(1, 2)
        patches = x.unfold(-1, self.patch_length, self.patch_length).contiguous()
        tokens = self.patch_projection(patches).reshape(values.shape[0] * self.variables, -1, self.d_model)
        encoded = self.encoder(tokens)[:, -1]
        forecast = self.head(encoded).view(values.shape[0], self.variables, self.forecast_horizon).transpose(1, 2)
        return forecast

    forward = predict


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> PatchTSTForecastModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"model_state_dict"}:
        raise ValueError("checkpoint must contain exactly model_state_dict")
    model = PatchTSTForecastModel().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()
