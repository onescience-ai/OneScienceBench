"""LoRA-only fine-tuning utilities for the official Microsoft Aurora model."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import torch
from aurora import Batch


LORA_PARAMETER_MARKERS = ("lora_A", "lora_B")


def is_lora_parameter(name: str) -> bool:
    """Return whether an Aurora parameter belongs to a LoRA adapter."""
    return any(marker in name for marker in LORA_PARAMETER_MARKERS)


def _prediction_target_pairs(prediction: Batch, target: Batch):
    for group_name in ("surf_vars", "atmos_vars"):
        predicted = prediction.get(group_name, {}) if isinstance(prediction, Mapping) else getattr(prediction, group_name)
        expected = target.get(group_name, {}) if isinstance(target, Mapping) else getattr(target, group_name)
        if set(predicted) != set(expected):
            raise ValueError(
                f"{group_name} keys differ: {sorted(predicted)} != {sorted(expected)}"
            )
        for name in sorted(predicted):
            if predicted[name].shape != expected[name].shape:
                raise ValueError(
                    f"{group_name}.{name} shape differs: "
                    f"{tuple(predicted[name].shape)} != {tuple(expected[name].shape)}"
                )
            yield predicted[name], expected[name]


def latitude_weighted_mse(prediction: Batch, target: Batch) -> torch.Tensor:
    """Average latitude-weighted MSE over all surface and atmospheric variables."""
    losses: list[torch.Tensor] = []
    metadata = prediction.get("metadata") if isinstance(prediction, Mapping) else prediction.metadata
    latitude = metadata.lat
    for predicted, expected in _prediction_target_pairs(prediction, target):
        weights = torch.cos(torch.deg2rad(latitude.to(predicted.device, predicted.dtype)))
        weights = weights.clamp_min(0)
        weights = weights / weights.mean().clamp_min(torch.finfo(predicted.dtype).eps)
        shape = [1] * predicted.ndim
        shape[-2] = weights.numel()
        losses.append(((predicted - expected).square() * weights.reshape(shape)).mean())
    if not losses:
        raise ValueError("prediction and target contain no trainable weather variables")
    return torch.stack(losses).mean()


def weather_metrics(prediction: Batch, target: Batch) -> dict[str, float]:
    """Compute unweighted RMSE and MAE over aligned Aurora output tensors."""
    differences = [
        (predicted.detach().float() - expected.detach().float()).reshape(-1)
        for predicted, expected in _prediction_target_pairs(prediction, target)
    ]
    difference = torch.cat(differences)
    return {
        "rmse": float(torch.sqrt(torch.mean(difference.square())).cpu()),
        "mae": float(torch.mean(torch.abs(difference)).cpu()),
    }


def flatten_weather_batch(batch: Batch) -> list[float]:
    """Flatten outputs in a stable surface-then-atmosphere variable order."""
    values: list[torch.Tensor] = []
    for group_name in ("surf_vars", "atmos_vars"):
        group = getattr(batch, group_name)
        values.extend(group[name].detach().float().cpu().reshape(-1) for name in sorted(group))
    if not values:
        raise ValueError("weather batch contains no output values")
    return torch.cat(values).tolist()


class AuroraLoRAFineTuner:
    """Load an Aurora checkpoint and update only its built-in LoRA adapters."""

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: str | Path,
        output_dir: str | Path,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 0.0,
        max_steps: int = 5,
        grad_clip_norm: float = 1.0,
        device: str = "cuda",
        use_activation_checkpointing: bool = False,
        seed: int = 20260804,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Aurora checkpoint not found: {self.checkpoint_path}")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = int(max_steps)
        self.grad_clip_norm = float(grad_clip_norm)
        self.device = torch.device(device)
        self.completed_steps = 0

        torch.manual_seed(seed)
        self.model = model
        load_local = getattr(self.model, "load_checkpoint_local", None)
        if not callable(load_local):
            raise TypeError("model must provide Aurora load_checkpoint_local(path, strict=...)")
        load_local(str(self.checkpoint_path), strict=False)

        self.trainable_parameter_names: list[str] = []
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(is_lora_parameter(name))
            if parameter.requires_grad:
                self.trainable_parameter_names.append(name)
        if not self.trainable_parameter_names:
            raise ValueError("model contains no lora_A/lora_B parameters; construct it with use_lora=True")
        if not any("lora_A" in name for name in self.trainable_parameter_names):
            raise ValueError("model contains no lora_A parameters")
        if not any("lora_B" in name for name in self.trainable_parameter_names):
            raise ValueError("model contains no lora_B parameters")

        if use_activation_checkpointing:
            configure = getattr(self.model, "configure_activation_checkpointing", None)
            if not callable(configure):
                raise TypeError("model does not support activation checkpointing")
            configure()

        self.model.to(self.device)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    def train_step(self, input_batch: Batch, target_batch: Batch) -> dict[str, float]:
        """Execute one real LoRA optimisation step and return finite training metrics."""
        self.model.train()
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(input_batch)
        loss = latitude_weighted_mse(prediction, target_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {float(loss.detach().cpu())}")
        loss.backward()
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm: {float(grad_norm.detach().cpu())}")
        self.optimizer.step()
        self.completed_steps += 1

        metrics = weather_metrics(prediction, target_batch)
        metrics.update(
            {
                "step": self.completed_steps,
                "epoch": 0,
                "train_loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
        )
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise FloatingPointError(f"non-finite training metrics: {metrics}")
        return metrics

    def _predict(self, input_batch: Batch) -> Batch:
        self.model.eval()
        with torch.no_grad():
            return self.model(input_batch.to(self.device))

    def save_checkpoint(self) -> Path:
        """Save the complete fine-tuned state and enough metadata to audit LoRA scope."""
        path = self.output_dir / "aurora_lora_finetuned.pt"
        torch.save(
            {
                "format": "aurora-lora-finetune-v1",
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "trainable_parameter_names": list(self.trainable_parameter_names),
                "completed_steps": self.completed_steps,
            },
            path,
        )
        return path

    def fit(self, train_batches: Iterable[tuple[Batch, Batch]]) -> dict[str, object]:
        """Cycle over batches for ``max_steps`` and write deterministic evaluation artifacts."""
        batches = tuple(train_batches)
        if not batches:
            raise ValueError("train_batches must not be empty")

        metrics_path = self.output_dir / "metrics.jsonl"
        history: list[dict[str, float]] = []
        with metrics_path.open("w", encoding="utf-8") as handle:
            for index in range(self.max_steps):
                input_batch, target_batch = batches[index % len(batches)]
                metrics = self.train_step(input_batch, target_batch)
                history.append(metrics)
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")

        final_input, final_target = batches[(self.max_steps - 1) % len(batches)]
        final_target = final_target.to(self.device)
        final_prediction = self._predict(final_input)
        prediction_path = self.output_dir / "predictions.json"
        target_path = self.output_dir / "targets.json"
        prediction_path.write_text(
            json.dumps({"predictions": flatten_weather_batch(final_prediction)}),
            encoding="utf-8",
        )
        target_path.write_text(
            json.dumps({"targets": flatten_weather_batch(final_target)}),
            encoding="utf-8",
        )
        checkpoint_path = self.save_checkpoint()
        return {
            "history": history,
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "prediction_path": str(prediction_path),
            "target_path": str(target_path),
        }
