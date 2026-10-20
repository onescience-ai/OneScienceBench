from __future__ import annotations

from graphcast_horizon_weighted_loss import GraphCastHorizonWeightedLoss


def build_loss(config=None):
    config = dict(config or {})
    return GraphCastHorizonWeightedLoss(
        horizon_weights=config.get("horizon_weights", [1.0, 2.0, 4.0]),
        variable_weights=config.get("variable_weights", [1.0, 0.5]),
        eps=float(config.get("eps", 1e-8)),
    )


def train_step(loss_fn, predictions, targets, optimizer, latitude_weights=None):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(predictions, targets, latitude_weights)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
