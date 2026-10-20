from __future__ import annotations

import torch

from fourcastnet_probabilistic_head import gaussian_nll


@torch.no_grad()
def evaluate(model, inputs, targets):
    model.eval()
    prediction = model(inputs)
    return {
        "rmse": float(torch.sqrt(torch.mean((prediction["mean"] - targets).square()))),
        "nll": float(gaussian_nll(prediction, targets)),
    }
