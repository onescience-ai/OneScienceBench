from __future__ import annotations

import torch

from graphcast_conservation_projection import GraphCastConservationModel


def build_model(config=None):
    config = dict(config or {})
    return GraphCastConservationModel(
        in_channels=int(config.get("in_channels", 3)),
        out_channels=int(config.get("out_channels", 2)),
        hidden_channels=int(config.get("hidden_channels", 16)),
        processor_layers=int(config.get("processor_layers", 2)),
        constrained_channels=tuple(config.get("constrained_channels", [0])),
    )


def train_step(model, inputs, targets, target_integrals, area_weights, optimizer, mask=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs, target_integrals, area_weights, mask)
    loss = torch.mean((prediction - targets).square())
    loss.backward()
    optimizer.step()
    return float(loss.detach())
