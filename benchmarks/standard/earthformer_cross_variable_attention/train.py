from __future__ import annotations

import torch

from earthformer_cross_variable_attention import EarthformerCrossVariableAttention


def build_model(config=None):
    config = dict(config or {})
    return EarthformerCrossVariableAttention(
        num_variables=int(config.get("num_variables", 4)),
        hidden_dim=int(config.get("hidden_dim", 16)),
        num_heads=int(config.get("num_heads", 4)),
    )


def train_step(model, inputs, targets, optimizer, variable_mask=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs, variable_mask)
    if variable_mask is None:
        loss = torch.mean((prediction - targets).square())
    else:
        if variable_mask.ndim == 1:
            variable_mask = variable_mask[None].expand(inputs.shape[0], -1)
        weights = variable_mask[:, None, :, None, None].to(prediction)
        loss = ((prediction - targets).square() * weights).sum() / weights.expand_as(prediction).sum()
    loss.backward()
    optimizer.step()
    return float(loss.detach())
