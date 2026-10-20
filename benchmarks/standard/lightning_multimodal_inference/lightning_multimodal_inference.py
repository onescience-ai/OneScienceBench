"""Multi-modal satellite/NWP/terrain lightning probability inference."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


class LightningMultimodalModel(nn.Module):
    def __init__(self, nwp_channels: int = 3, hidden_dim: int = 16):
        super().__init__()
        if nwp_channels <= 0 or hidden_dim <= 0:
            raise ValueError("invalid dimensions")
        self.nwp_channels, self.hidden_dim = nwp_channels, hidden_dim
        self.satellite_encoder = nn.Conv2d(1, hidden_dim, 3, padding=1)
        self.nwp_encoder = nn.Conv2d(nwp_channels, hidden_dim, 1)
        self.terrain_encoder = nn.Conv2d(1, hidden_dim, 1)
        self.fusion = nn.Sequential(nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU())
        self.head = nn.Conv2d(hidden_dim, 1, 1)

    def infer(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"satellite_ir", "nwp", "terrain"}
        if not isinstance(batch, dict) or not required.issubset(batch):
            raise ValueError("batch requires satellite_ir, nwp and terrain")
        sat, nwp, terrain = batch["satellite_ir"], batch["nwp"], batch["terrain"]
        if sat.ndim != 4 or nwp.ndim != 4 or terrain.ndim != 4 or sat.shape[1] < 1 or nwp.shape[1] != self.nwp_channels or terrain.shape[1] != 1:
            raise ValueError("invalid modality shapes")
        if sat.shape[0] != nwp.shape[0] or sat.shape[2:] != nwp.shape[2:] or terrain.shape != (sat.shape[0], 1, *sat.shape[2:]):
            raise ValueError("modalities must share batch and grid")
        if not all(torch.isfinite(x).all() for x in (sat, nwp, terrain)):
            raise ValueError("modalities must be finite")
        feature = self.satellite_encoder(sat.mean(1, keepdim=True)) + self.nwp_encoder(nwp) + self.terrain_encoder(terrain)
        probability = torch.sigmoid(self.head(self.fusion(feature))).squeeze(1)
        return {"probability": probability}

    forward = infer


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> LightningMultimodalModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"model_state_dict"}:
        raise ValueError("checkpoint must contain exactly model_state_dict")
    model = LightningMultimodalModel().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()
