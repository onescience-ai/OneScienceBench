import torch
from stormcast import StormCast

def evaluate(model: StormCast, noisy: torch.Tensor, cond: torch.Tensor, target_noise: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        err = model(noisy, cond, 10) - target_noise
    return {"mae": float(err.abs().mean()), "rmse": float(err.square().mean().sqrt())}

