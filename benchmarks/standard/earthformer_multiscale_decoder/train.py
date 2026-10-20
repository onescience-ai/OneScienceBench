from __future__ import annotations

import torch

from earthformer_multiscale_decoder import EarthformerMultiscaleDecoder


def build_model(config=None):
    config = dict(config or {})
    return EarthformerMultiscaleDecoder(
        encoder_channels=tuple(config.get("encoder_channels", [8, 16, 32])),
        hidden_channels=int(config.get("hidden_channels", 16)),
        out_channels=int(config.get("out_channels", 2)),
    )


def train_step(model, features, targets, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(features)
    loss = torch.mean((prediction - targets).square())
    loss.backward()
    optimizer.step()
    return float(loss.detach())
