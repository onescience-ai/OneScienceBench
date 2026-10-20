import torch
from fengwu import Fengwu

def evaluate(model: Fengwu, inputs, targets) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(*inputs)
        err = torch.cat([(a - b).reshape(-1) for a, b in zip(pred, targets)])
    return {"mae": float(err.abs().mean()), "rmse": float(err.square().mean().sqrt())}

