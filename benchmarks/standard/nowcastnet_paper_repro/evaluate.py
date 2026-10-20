import torch
from nowcastnet import Net

def evaluate(model: Net, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        err = model(inputs) - targets
    return {"mae": float(err.abs().mean()), "rmse": float(err.square().mean().sqrt())}

