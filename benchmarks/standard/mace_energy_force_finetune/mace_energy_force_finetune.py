"""Real MACE-MP-0 energy-force fine-tuning without OneScience imports."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from portable_mace import ScaleShiftMACE, load_portable_checkpoint


def load_foundation_model(
    config_path: str | Path,
    weights_path: str | Path,
    device: str | torch.device = "cpu",
) -> ScaleShiftMACE:
    """Build the independent MACE structure and strictly load real MP-0 weights."""

    return load_portable_checkpoint(config_path, weights_path, device=device)


def _field(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, Mapping):
        result = value[name]
    else:
        try:
            result = value[name]
        except (KeyError, TypeError):
            result = getattr(value, name)
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    return result


def _move_reference(reference: Any, device: torch.device) -> Any:
    move = getattr(reference, "to", None)
    return move(device) if callable(move) else reference


def _model_input(reference: Any) -> dict[str, torch.Tensor]:
    to_dict = getattr(reference, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(reference, Mapping):
        return dict(reference)
    raise TypeError("reference must provide to_dict() or implement Mapping")


class MACEEnergyForceFineTuner:
    """Fine-tune a real MACE-MP-0 model with native weighted energy-force loss."""

    def __init__(
        self,
        model: torch.nn.Module,
        output_dir: str | Path,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 0.0,
        max_steps: int = 1,
        energy_weight: float = 1.0,
        forces_weight: float = 10.0,
        grad_clip_norm: float = 10.0,
        device: str = "cpu",
        seed: int = 20260804,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if energy_weight < 0 or forces_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if energy_weight == 0 and forces_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        if grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")

        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = int(max_steps)
        self.energy_weight = float(energy_weight)
        self.forces_weight = float(forces_weight)
        self.grad_clip_norm = float(grad_clip_norm)
        self.completed_steps = 0

        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        self.trainable_parameter_names = [
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    def _loss_terms(
        self,
        reference: Any,
        prediction: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy = _field(reference, "energy")
        forces = _field(reference, "forces")
        ptr = _field(reference, "ptr").to(dtype=torch.long)
        weight = _field(reference, "weight").to(dtype=energy.dtype)
        energy_weight = _field(reference, "energy_weight").to(dtype=energy.dtype)
        forces_weight = _field(reference, "forces_weight").to(dtype=forces.dtype)
        predicted_energy = _field(prediction, "energy")
        predicted_forces = _field(prediction, "forces")

        num_atoms = (ptr[1:] - ptr[:-1]).to(dtype=energy.dtype)
        if energy.shape != predicted_energy.shape or forces.shape != predicted_forces.shape:
            raise ValueError("MACE prediction and reference shapes do not match")
        if num_atoms.numel() != energy.numel() or torch.any(num_atoms <= 0):
            raise ValueError("ptr does not describe the energy graphs")

        energy_loss = (
            weight
            * energy_weight
            * torch.square((energy - predicted_energy) / num_atoms)
        ).mean()
        atoms_per_graph = ptr[1:] - ptr[:-1]
        atom_weight = torch.repeat_interleave(weight, atoms_per_graph).unsqueeze(-1)
        atom_force_weight = torch.repeat_interleave(
            forces_weight, atoms_per_graph
        ).unsqueeze(-1)
        force_loss = (
            atom_weight
            * atom_force_weight
            * torch.square(forces - predicted_forces)
        ).mean()
        total = self.energy_weight * energy_loss + self.forces_weight * force_loss
        return total, energy_loss, force_loss

    def train_step(self, reference: Any) -> dict[str, float]:
        self.model.train()
        reference = _move_reference(reference, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(
            _model_input(reference),
            training=True,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )
        if not isinstance(prediction, Mapping):
            raise TypeError("MACE model must return a mapping")
        loss, energy_loss, forces_loss = self._loss_terms(reference, prediction)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss is not finite")
        loss.backward()
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("gradient norm is not finite")
        self.optimizer.step()
        self.completed_steps += 1

        metrics = {
            "step": float(self.completed_steps),
            "train_loss": float(loss.detach().cpu()),
            "energy_loss": float(energy_loss.detach().cpu()),
            "forces_loss": float(forces_loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite metrics: {metrics}")
        return metrics

    def save_checkpoint(self) -> Path:
        path = self.output_dir / "mace_finetuned.pt"
        torch.save(
            {
                "format": "mace-mp0-energy-force-finetune-v1",
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "trainable_parameter_names": list(self.trainable_parameter_names),
                "completed_steps": self.completed_steps,
                "energy_weight": self.energy_weight,
                "forces_weight": self.forces_weight,
            },
            path,
        )
        return path

    def fit(self, train_batches: Iterable[Any]) -> dict[str, object]:
        batches = tuple(train_batches)
        if not batches:
            raise ValueError("train_batches must not be empty")
        metrics_path = self.output_dir / "metrics.jsonl"
        history: list[dict[str, float]] = []
        with metrics_path.open("w", encoding="utf-8") as handle:
            for index in range(self.max_steps):
                metrics = self.train_step(batches[index % len(batches)])
                history.append(metrics)
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        checkpoint_path = self.save_checkpoint()
        return {
            "history": history,
            "metrics_path": str(metrics_path),
            "checkpoint_path": str(checkpoint_path),
        }


__all__ = [
    "ScaleShiftMACE",
    "load_foundation_model",
    "MACEEnergyForceFineTuner",
]
