from __future__ import annotations
import torch
from nowcastnet import Net

def train_step(model: Net, inputs: torch.Tensor, targets: torch.Tensor, optimizer=None) -> float:
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((model(inputs) - targets) ** 2)
    loss.backward()
    optimizer.step()
    return float(loss.detach())

