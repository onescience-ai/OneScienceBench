"""Reference MACE energy-force loss with force-weight warmup."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from onescience.modules.loss.mace_loss import (
    mean_squared_error_forces,
    weighted_mean_squared_error_energy,
)


class MACEForceCurriculumLoss(nn.Module):
    """Preserve native MACE weighting while warming up the force objective."""

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        warmup_steps: int = 100,
    ) -> None:
        super().__init__()
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        self.warmup_steps = int(warmup_steps)
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self,
        reference: Any,
        prediction: dict[str, torch.Tensor],
        step: int | torch.Tensor,
        ddp: bool | None = None,
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(reference, prediction, ddp)
        loss_forces = mean_squared_error_forces(reference, prediction, ddp)
        progress = torch.as_tensor(
            step,
            dtype=loss_forces.dtype,
            device=loss_forces.device,
        ).clamp(min=0, max=self.warmup_steps) / self.warmup_steps
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * progress * loss_forces
        )


__all__ = ["MACEForceCurriculumLoss"]
