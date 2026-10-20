"""Real MACE-MP-0 model with a trainable element-wise affine adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from portable_mace import ScaleShiftMACE, load_portable_checkpoint


def load_foundation_model(
    config_path: str | Path,
    weights_path: str | Path,
    device: str | torch.device = "cpu",
) -> ScaleShiftMACE:
    return load_portable_checkpoint(config_path, weights_path, device=device)


_DERIVATIVE_KEYS = (
    "forces",
    "virials",
    "stress",
    "edge_forces",
    "hessian",
    "atomic_virials",
    "atomic_stresses",
)


class MACEElementAffineAdapter(nn.Module):
    """Add a trainable energy scale and per-element reference-energy shift."""

    def __init__(
        self,
        model: nn.Module,
        atomic_numbers: Sequence[int] | torch.Tensor,
        trainable: bool = True,
        freeze_base: bool = True,
        energy_scale_init: float = 1.0,
        element_shift_init: float = 0.0,
    ) -> None:
        super().__init__()
        table = torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1)
        if table.numel() == 0 or torch.any(table <= 0):
            raise ValueError("atomic_numbers must contain positive integers")
        if torch.unique(table).numel() != table.numel():
            raise ValueError("atomic_numbers must not contain duplicates")
        self.model = model
        if freeze_base:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.register_buffer("atomic_numbers", table)
        dtype = next(
            parameter.dtype
            for parameter in self.model.parameters()
            if parameter.is_floating_point()
        )
        scale = torch.tensor(float(energy_scale_init), dtype=dtype)
        shift = torch.full((table.numel(),), float(element_shift_init), dtype=dtype)
        if trainable:
            self.energy_scale = nn.Parameter(scale)
            self.element_shift = nn.Parameter(shift)
        else:
            self.register_buffer("energy_scale", scale)
            self.register_buffer("element_shift", shift)

    def forward(
        self,
        data: Mapping[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        output = self.model(
            data,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            **kwargs,
        )
        if not isinstance(output, Mapping):
            raise TypeError("MACE model must return a mapping")
        if "energy" not in output or not isinstance(output["energy"], torch.Tensor):
            raise ValueError("MACE output must contain tensor energy")
        node_attrs = data.get("node_attrs")
        batch = data.get("batch")
        if not isinstance(node_attrs, torch.Tensor) or node_attrs.ndim != 2:
            raise ValueError("data['node_attrs'] must have shape (N, num_elements)")
        if node_attrs.shape[1] != self.atomic_numbers.numel():
            raise ValueError("node_attrs columns must match atomic_numbers")
        if not isinstance(batch, torch.Tensor) or batch.shape != (node_attrs.shape[0],):
            raise ValueError("data['batch'] must have shape (N,)")

        species = torch.argmax(node_attrs, dim=-1)
        node_shift = self.element_shift[species]
        energy = output["energy"]
        graph_shift = torch.zeros_like(energy).index_add(
            0, batch.to(dtype=torch.long), node_shift
        )
        adapted: dict[str, torch.Tensor | None] = dict(output)
        adapted["energy"] = self.energy_scale * energy + graph_shift
        node_energy = adapted.get("node_energy")
        if isinstance(node_energy, torch.Tensor):
            expanded_shift = node_shift
            while expanded_shift.ndim < node_energy.ndim:
                expanded_shift = expanded_shift.unsqueeze(-1)
            adapted["node_energy"] = self.energy_scale * node_energy + expanded_shift
        interaction_energy = adapted.get("interaction_energy")
        if isinstance(interaction_energy, torch.Tensor):
            adapted["interaction_energy"] = self.energy_scale * interaction_energy
        for key in _DERIVATIVE_KEYS:
            value = adapted.get(key)
            if isinstance(value, torch.Tensor):
                adapted[key] = self.energy_scale * value
        return adapted


__all__ = [
    "ScaleShiftMACE",
    "load_foundation_model",
    "MACEElementAffineAdapter",
]
