import torch


def evaluate(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        error = model(inputs) - targets
    return {"rmse": float(error.square().mean().sqrt()), "mae": float(error.abs().mean())}
