import torch
from stormer import Stormer

def evaluate(model: Stormer, inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        error = model(inputs) - targets
    return {"mae": float(error.abs().mean()), "rmse": float(error.square().mean().sqrt())}

