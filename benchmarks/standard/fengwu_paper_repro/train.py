from __future__ import annotations
import torch
from fengwu import Fengwu

def train_step(model: Fengwu, inputs, targets, optimizer=None) -> float:
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    pred = model(*inputs)
    loss = sum(torch.mean((a - b) ** 2) for a, b in zip(pred, targets))
    loss.backward()
    optimizer.step()
    return float(loss.detach())

