from __future__ import annotations

import torch

from earthformer_irregular_time_embedding import EarthformerIrregularTimeModel


def build_model(config=None):
    config = dict(config or {})
    return EarthformerIrregularTimeModel(
        in_channels=int(config.get("in_channels", 3)),
        out_channels=int(config.get("out_channels", 2)),
        hidden_dim=int(config.get("hidden_dim", 16)),
        num_heads=int(config.get("num_heads", 4)),
        num_frequencies=int(config.get("num_frequencies", 4)),
    )


def train_step(model, inputs, time_deltas, targets, optimizer, time_mask=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs, time_deltas, time_mask)
    if time_mask is None:
        loss = torch.mean((prediction - targets).square())
    else:
        weights = time_mask[:, :, None, None, None].to(prediction)
        loss = ((prediction - targets).square() * weights).sum() / weights.expand_as(prediction).sum()
    loss.backward()
    optimizer.step()
    return float(loss.detach())
