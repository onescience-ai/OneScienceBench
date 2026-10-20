"""Independent local NequIX torch-backend implementation for this evaluation task.

JAX-free. The e3nn 0.4.4 compatibility shim is applied before the model is
imported (the model uses Irreps.regroup/filter/slice_by_mul from e3nn 0.5.x).
"""
from __future__ import annotations

from . import _e3nn_compat  # noqa: F401
from .layer_norm import RMSLayerNorm
from .load import load_model
from .model import (
    Activation,
    Gate,
    MLP,
    NequixTorch,
    NequixTorchConvolution,
    bessel_basis,
    get_optimizer_param_groups,
    polynomial_cutoff,
    save_model,
    scatter,
)
from .data import (
    AseDBDataset,
    Dataset,
    IndexDataset,
    atomic_numbers_to_indices,
    dict_to_pytorch_geometric,
    preprocess_graph,
)

__all__ = [
    "Activation",
    "AseDBDataset",
    "Dataset",
    "Gate",
    "IndexDataset",
    "MLP",
    "NequixTorch",
    "NequixTorchConvolution",
    "RMSLayerNorm",
    "atomic_numbers_to_indices",
    "bessel_basis",
    "dict_to_pytorch_geometric",
    "get_optimizer_param_groups",
    "load_model",
    "polynomial_cutoff",
    "preprocess_graph",
    "save_model",
    "scatter",
]
