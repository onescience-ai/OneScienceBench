import torch
from nowcastnet_radar_nowcasting import RadarNowcastNet


def train_step(model, inputs, targets, optimizer=None):
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean(); loss.backward(); optimizer.step()
    return float(loss.detach())


def build_model(config=None):
    return RadarNowcastNet(**(config or {}))
