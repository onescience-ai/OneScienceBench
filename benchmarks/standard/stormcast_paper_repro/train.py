from __future__ import annotations
import torch
from stormcast import StormCast

def train_step(model: StormCast, noisy: torch.Tensor, cond: torch.Tensor, target_noise: torch.Tensor, optimizer=None) -> float:
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((model(noisy, cond, 10) - target_noise) ** 2)
    loss.backward()
    optimizer.step()
    return float(loss.detach())

