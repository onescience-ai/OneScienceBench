from __future__ import annotations

from fourcastnet_probabilistic_head import FourCastNetProbabilisticHead, gaussian_nll


def build_model(config=None):
    config = dict(config or {})
    return FourCastNetProbabilisticHead(
        in_channels=int(config.get("in_channels", 4)),
        out_channels=int(config.get("out_channels", 2)),
        hidden_channels=int(config.get("hidden_channels", 16)),
        depth=int(config.get("depth", 2)),
        min_scale=float(config.get("min_scale", 1e-4)),
    )


def train_step(model, inputs, targets, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = gaussian_nll(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
