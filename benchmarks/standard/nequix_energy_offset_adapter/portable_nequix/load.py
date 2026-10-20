"""JAX-free loader for NequIX torch checkpoints (``.pt``).

The official NequIX ``load_model`` reads the config line and then calls
``torch.load(f)`` on the same file object. On torch 2.5.x (the OneScience
matchem build) that fails with "failed finding central directory" because the
zip reader cannot seek past the config-line prefix on a positioned file object.
The state_dict itself is intact, so we read the config line, then the remaining
zip bytes into a ``BytesIO`` buffer, which ``torch.load`` reads reliably.

This module is intentionally JAX-free so it can be imported and used on the
DCU/ROCm target platform (where JAX does not run).
"""
from __future__ import annotations

import io
import json

import torch

from .model import NequixTorch


def _build_model(config: dict, use_kernel: bool) -> NequixTorch:
    return NequixTorch(
        n_species=len(config["atomic_numbers"]),
        hidden_irreps=config["hidden_irreps"],
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        index_weights=config["index_weights"],
        layer_norm=config["layer_norm"],
        shift=config["shift"],
        scale=config["scale"],
        avg_n_neighbors=config["avg_n_neighbors"],
        atom_energies=[config["atom_energies"][str(n)] for n in config["atomic_numbers"]],
        kernel=use_kernel,
    )


def load_model(path: str, use_kernel: bool = False) -> tuple[NequixTorch, dict]:
    """Load a NequIX torch checkpoint (config JSON line + torch state_dict zip).

    Args:
        path: path to a ``.pt`` file produced by the NequIX torch backend.
        use_kernel: whether to build the OpenEquivariance-kernel variant. Must be
            ``False`` on DCU/ROCm (kernels are CUDA-only).

    Returns:
        ``(model, config)`` with the state_dict loaded (``strict=False``; the
        ``.tp.`` weights are recomputed and not stored in the non-kernel form).
    """
    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        zip_bytes = f.read()

    model = _build_model(config, use_kernel)
    state_dict = torch.load(io.BytesIO(zip_bytes), map_location="cpu")
    # The non-kernel form does not store the separable tensor-product weights;
    # they are recomputed from the radial MLP, so allow them to be missing.
    state_dict = {k: v for k, v in state_dict.items() if ".tp." not in k}
    model.load_state_dict(state_dict, strict=False)
    return model, config
