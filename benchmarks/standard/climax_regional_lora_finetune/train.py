import torch
from climax_regional_lora import ClimaXRegionalLoRA


def train_step(model, inputs, targets, optimizer=None):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer or torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
    return float(loss.detach())


def build_model(config=None):
    return ClimaXRegionalLoRA(**(config or {}))
