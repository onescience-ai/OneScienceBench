"""Atomic energy scaling used by the fixed MatterSim checkpoint."""

from __future__ import annotations

import torch
from torch import nn


class AtomScaling(nn.Module):
    def __init__(
        self,
        max_z: int = 94,
        init_scale: torch.Tensor | float | None = None,
        init_shift: torch.Tensor | float | None = None,
        trainable_scale: bool = False,
        trainable_shift: bool = False,
        device: str = "cpu",
        **_kwargs,
    ) -> None:
        super().__init__()
        scale = self._expand(init_scale, max_z, 1.0)
        shift = self._expand(init_shift, max_z, 0.0)
        if trainable_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        if trainable_shift:
            self.shift = nn.Parameter(shift)
        else:
            self.register_buffer("shift", shift)
        self.to(device)

    @staticmethod
    def _expand(value, max_z: int, default: float) -> torch.Tensor:
        if value is None:
            return torch.full((max_z + 1,), default, dtype=torch.float32)
        if isinstance(value, (int, float)):
            return torch.full((max_z + 1,), float(value), dtype=torch.float32)
        value = torch.as_tensor(value, dtype=torch.float32)
        if tuple(value.shape) != (max_z + 1,):
            raise ValueError("atomic scaling tensors have the wrong shape")
        return value

    def forward(
        self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        return self.scale[atomic_numbers] * atomic_energies + self.shift[atomic_numbers]
