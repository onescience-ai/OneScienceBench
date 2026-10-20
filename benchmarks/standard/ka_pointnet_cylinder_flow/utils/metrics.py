from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def relative_l2_by_channel(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    numerators = np.linalg.norm(predictions - targets, axis=-1)
    denominators = np.linalg.norm(targets, axis=-1) + 1e-12
    channel_errors = (numerators / denominators).mean(axis=0)
    return {
        "u": float(channel_errors[0]),
        "v": float(channel_errors[1]),
        "p": float(channel_errors[2]),
        "mean": float(channel_errors.mean()),
    }


def mse_value(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float(torch.mean((predictions - targets) ** 2).item())
