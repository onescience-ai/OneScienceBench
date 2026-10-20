import torch
from graphcast_extreme_weather import GraphCastExtremeWeather


def weighted_mse(prediction, target, event_mask=None):
    weights = 1.0 + 4.0 * event_mask if event_mask is not None else 1.0
    return ((prediction - target).square() * weights).mean()


def train_step(model, inputs, targets, event_mask=None, optimizer=None):
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = weighted_mse(model(inputs), targets, event_mask); loss.backward(); optimizer.step()
    return float(loss.detach())


def build_model(config=None):
    return GraphCastExtremeWeather(**(config or {}))
