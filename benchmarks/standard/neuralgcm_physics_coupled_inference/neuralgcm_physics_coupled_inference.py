"""Small physics-coupled atmospheric rollout reference implementation."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


class NeuralGCMInference(nn.Module):
    def __init__(self, channels: int = 4, hidden_dim: int = 16, dt: float = 0.1):
        super().__init__()
        if channels <= 0 or hidden_dim <= 0 or not (0.0 < dt <= 1.0):
            raise ValueError("channels, hidden_dim and dt are invalid")
        self.channels, self.hidden_dim, self.dt = channels, hidden_dim, float(dt)
        self.encoder = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.processor = nn.Sequential(nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU())
        self.decoder = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, state: torch.Tensor, forcing: torch.Tensor | None = None) -> torch.Tensor:
        if not isinstance(state, torch.Tensor) or state.ndim != 4 or state.shape[1] != self.channels:
            raise ValueError("state must have shape [B,channels,H,W]")
        if not torch.isfinite(state).all():
            raise ValueError("state contains non-finite values")
        model_input = state
        if forcing is not None:
            if forcing.shape != state.shape or not torch.isfinite(forcing).all():
                raise ValueError("forcing must match state and be finite")
            model_input = state + 0.01 * forcing
        tendency = self.decoder(self.processor(self.encoder(model_input)))
        return state + self.dt * torch.tanh(tendency)

    def rollout(self, state: torch.Tensor, steps: int, forcing: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        current = state.clone()
        outputs = []
        with torch.no_grad():
            for _ in range(steps):
                current = self.forward(current, forcing)
                outputs.append(current.clone())
        return torch.stack(outputs, dim=0)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> NeuralGCMInference:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"model_state_dict"}:
        raise ValueError("checkpoint must contain exactly model_state_dict")
    model = NeuralGCMInference().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()
