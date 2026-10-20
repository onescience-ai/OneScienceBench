import torch
from nowcastnet_domain_adaptation import NowcastNetDomainAdapter


def train_step(model, inputs, targets, optimizer=None):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer or torch.optim.AdamW(trainable, lr=1e-3)
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean(); loss.backward(); optimizer.step()
    return float(loss.detach())


def build_model(config=None):
    return NowcastNetDomainAdapter(**(config or {}))
