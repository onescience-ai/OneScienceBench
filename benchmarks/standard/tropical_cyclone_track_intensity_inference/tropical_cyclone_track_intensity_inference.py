"""Storm-centric track and intensity inference reference model."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


class TropicalCycloneTrackIntensityModel(nn.Module):
    def __init__(self, history_features: int = 4, hidden_dim: int = 24, lead_steps: int = 4):
        super().__init__()
        if history_features != 4 or hidden_dim <= 0 or lead_steps <= 0:
            raise ValueError("invalid cyclone model configuration")
        self.history_features, self.hidden_dim, self.lead_steps = history_features, hidden_dim, lead_steps
        self.history_encoder = nn.GRU(history_features, hidden_dim, batch_first=True)
        self.track_head = nn.Linear(hidden_dim, lead_steps * 2)
        self.intensity_head = nn.Linear(hidden_dim, lead_steps * 2)
        self.landfall_head = nn.Linear(hidden_dim, lead_steps)

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        if history.ndim != 3 or history.shape[2] != self.history_features or history.shape[1] < 2:
            raise ValueError("history must have shape [B,T,4] with T>=2")
        if not torch.isfinite(history).all():
            raise ValueError("history must be finite")
        encoded, _ = self.history_encoder(history)
        last = encoded[:, -1]
        track_delta = self.track_head(last).view(-1, self.lead_steps, 2) * 2.0
        last_position = history[:, -1, :2].unsqueeze(1)
        track = last_position + torch.cumsum(track_delta, dim=1)
        track = torch.stack((track[..., 0].clamp(-90, 90), ((track[..., 1] + 180) % 360) - 180), dim=-1)
        intensity_delta = self.intensity_head(last).view(-1, self.lead_steps, 2)
        last_intensity = history[:, -1, 2:4].unsqueeze(1)
        intensity = last_intensity + torch.cumsum(intensity_delta, dim=1)
        return {"track": track, "central_pressure": intensity[..., 0], "max_wind": intensity[..., 1], "landfall_probability": torch.sigmoid(self.landfall_head(last))}

    def infer(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            return self.forward(history)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> TropicalCycloneTrackIntensityModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"model_state_dict"}:
        raise ValueError("checkpoint must contain exactly model_state_dict")
    model = TropicalCycloneTrackIntensityModel().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()
