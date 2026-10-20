"""Runtime constants used by the standalone GemNet-T implementation."""

from functools import lru_cache
from pathlib import Path

import torch


MAX_ATOMIC_NUM = 100
MODELS_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache
def get_pyg_device() -> torch.device:
    return get_device()
