"""Compact multi-spectral satellite cloud-property inference model."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


class SatelliteCloudPropertyModel(nn.Module):
    def __init__(self, bands: int = 5, hidden_dim: int = 16, cloud_types: int = 4):
        super().__init__()
        if bands <= 0 or hidden_dim <= 0 or cloud_types <= 1:
            raise ValueError("invalid model dimensions")
        self.bands, self.hidden_dim, self.cloud_types = bands, hidden_dim, cloud_types
        self.temporal_encoder = nn.Conv2d(bands, hidden_dim, 3, padding=1)
        self.stem = nn.Sequential(nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU())
        self.heads = nn.ModuleDict({
            "cloud_top_temperature": nn.Conv2d(hidden_dim, 1, 1),
            "cloud_type_logits": nn.Conv2d(hidden_dim, cloud_types, 1),
            "precipitation_probability": nn.Conv2d(hidden_dim, 1, 1),
        })

    def forward(self, spectral: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if not isinstance(spectral, torch.Tensor) or spectral.ndim != 5 or spectral.shape[2] != self.bands:
            raise ValueError("spectral must have shape [B,T,bands,H,W]")
        if not torch.isfinite(spectral).all():
            raise ValueError("spectral contains non-finite values")
        if valid_mask is not None:
            if valid_mask.shape != spectral.shape[:2] or valid_mask.dtype != torch.bool or not valid_mask.any(dim=1).all():
                raise ValueError("valid_mask must be boolean with one valid frame per sample")
            weights = valid_mask[:, :, None, None, None].to(spectral.dtype)
            spectral = (spectral * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        else:
            spectral = spectral.mean(1)
        features = self.stem(self.temporal_encoder(spectral))
        return {
            "cloud_top_temperature": self.heads["cloud_top_temperature"](features).squeeze(1),
            "cloud_type_logits": self.heads["cloud_type_logits"](features).permute(0, 2, 3, 1),
            "precipitation_probability": torch.sigmoid(self.heads["precipitation_probability"](features).squeeze(1)),
        }

    def infer(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | object]:
        if not isinstance(batch, dict) or "spectral" not in batch or "latitude" not in batch or "longitude" not in batch:
            raise ValueError("batch requires spectral, latitude and longitude")
        output = self.forward(batch["spectral"], batch.get("valid_mask"))
        output["latitude"], output["longitude"] = batch["latitude"], batch["longitude"]
        return output


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> SatelliteCloudPropertyModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"model_state_dict"}:
        raise ValueError("checkpoint must contain exactly model_state_dict")
    model = SatelliteCloudPropertyModel().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()
