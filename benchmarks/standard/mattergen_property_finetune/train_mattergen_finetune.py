"""Deterministic adapter-only MatterGen property fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mattergen_property_finetune import (
    is_adapter_parameter,
    load_config,
    load_pretrained_model,
)
from portable_mattergen import CrystalBatch


DEFAULT_CONFIG = Path(__file__).with_name("mattergen_property_finetune.yaml")


def build_finetune_batch(seed: int = 20260819) -> dict[str, Any]:
    """Build two deterministic noisy crystals with one scalar property each."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    clean_pos = torch.tensor(
        [
            [0.05, 0.10, 0.15],
            [0.35, 0.45, 0.25],
            [0.70, 0.20, 0.55],
            [0.20, 0.75, 0.80],
            [0.10, 0.20, 0.30],
            [0.55, 0.65, 0.40],
            [0.85, 0.15, 0.70],
        ],
        dtype=torch.float32,
    )
    clean_cell = torch.tensor(
        [
            [[5.1, 0.0, 0.0], [0.2, 5.3, 0.0], [0.1, 0.3, 5.5]],
            [[4.8, 0.0, 0.0], [0.4, 5.0, 0.0], [0.2, 0.1, 5.2]],
        ],
        dtype=torch.float32,
    )
    clean_atomic_numbers = torch.tensor(
        [6, 8, 14, 26, 12, 29, 16], dtype=torch.long
    )
    num_atoms = torch.tensor([4, 3], dtype=torch.long)
    timesteps = torch.tensor([0.2, 0.8], dtype=torch.float32)
    property_values = torch.tensor([0.12, 0.68], dtype=torch.float32)
    batch_index = torch.repeat_interleave(torch.arange(2), num_atoms)

    pos_noise = torch.randn(clean_pos.shape, generator=generator) * 0.035
    raw_cell_noise = torch.randn(clean_cell.shape, generator=generator) * 0.018
    cell_noise = 0.5 * (raw_cell_noise + raw_cell_noise.transpose(1, 2))
    noisy_pos = torch.remainder(
        clean_pos + pos_noise * timesteps[batch_index, None], 1.0
    )
    noisy_cell = clean_cell + cell_noise * timesteps[:, None, None]
    atom_loss_mask = torch.tensor(
        [False, True, False, False, False, True, False], dtype=torch.bool
    )
    noisy_atomic_numbers = clean_atomic_numbers.clone()
    noisy_atomic_numbers[atom_loss_mask] = 101
    batch = CrystalBatch(
        pos=noisy_pos,
        cell=noisy_cell,
        atomic_numbers=noisy_atomic_numbers,
        num_atoms=num_atoms,
        dft_mag_density=property_values,
        use_unconditional_dft_mag_density=torch.zeros(2, 1, dtype=torch.bool),
    )
    return {
        "batch": batch,
        "timesteps": timesteps,
        "targets": {
            "pos": -pos_noise,
            "cell": -cell_noise,
            "atomic_numbers": clean_atomic_numbers - 1,
        },
        "atom_loss_mask": atom_loss_mask,
    }


def compute_materials_loss(
    output: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    atom_loss_mask: torch.Tensor,
    loss_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = {
        "pos": F.mse_loss(output["pos"], targets["pos"], reduction="sum"),
        "cell": F.mse_loss(output["cell"], targets["cell"], reduction="sum"),
        "atomic_numbers": F.cross_entropy(
            output["atomic_numbers"][atom_loss_mask],
            targets["atomic_numbers"][atom_loss_mask],
            reduction="sum",
        ),
    }
    total = sum(
        components[name] * float(loss_weights[name]) for name in components
    )
    return total, components


def _fixture_to_device(fixture: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "batch": fixture["batch"].to(device),
        "timesteps": fixture["timesteps"].to(device),
        "targets": {
            key: value.to(device) for key, value in fixture["targets"].items()
        },
        "atom_loss_mask": fixture["atom_loss_mask"].to(device),
    }


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    fixture: dict[str, Any],
    loss_weights: Mapping[str, float],
    gradient_clip_val: float,
    step: int,
) -> dict[str, float | int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(fixture["batch"], fixture["timesteps"])
    loss, components = compute_materials_loss(
        output,
        fixture["targets"],
        fixture["atom_loss_mask"],
        loss_weights,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("fine-tuning loss is not finite")
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters_with_grad = sum(parameter.grad is not None for parameter in trainable)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=float(gradient_clip_val),
        error_if_nonfinite=True,
    )
    optimizer.step()
    metrics: dict[str, float | int] = {
        "step": int(step),
        "loss": float(loss.detach().cpu()),
        "pos_loss": float(components["pos"].detach().cpu()),
        "cell_loss": float(components["cell"].detach().cpu()),
        "atomic_numbers_loss": float(components["atomic_numbers"].detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "parameters_with_grad": int(parameters_with_grad),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError(f"non-finite fine-tuning metrics: {metrics}")
    return metrics


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible accelerator is not available")
    return torch.device("cuda:0" if device == "cuda" else "cpu")


def train(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = "finetune_output",
    device: str = "auto",
) -> dict[str, Any]:
    config = load_config(config_path)
    finetuning = config["finetuning"]
    if bool(finetuning.get("full_finetuning", False)):
        raise ValueError("this task requires adapter-only fine-tuning")
    steps = int(finetuning["steps"])
    if steps != 2:
        raise ValueError("this evaluation task requires exactly two fine-tuning steps")
    selected_device = _resolve_device(device)
    model = load_pretrained_model(config, device=selected_device)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(
        not is_adapter_parameter(name) for name in trainable_names
    ):
        raise RuntimeError("only layerwise property-adapter parameters may be trainable")
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(finetuning["learning_rate"]),
    )
    fixture = _fixture_to_device(
        build_finetune_batch(seed=int(finetuning["seed"])), selected_device
    )
    history = [
        train_step(
            model=model,
            optimizer=optimizer,
            fixture=fixture,
            loss_weights=finetuning["loss_weights"],
            gradient_clip_val=float(finetuning["gradient_clip_val"]),
            step=step,
        )
        for step in range(1, steps + 1)
    ]

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = config["artifacts"]
    checkpoint_path = destination / artifacts["checkpoint"]
    metrics_path = destination / artifacts["metrics"]
    log_path = destination / artifacts["log"]
    payload = {
        "format": "mattergen-dft-mag-density-adapter-v1",
        "completed_steps": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "trainable_parameter_names": trainable_names,
        "config": config,
        "history": history,
    }
    torch.save(payload, checkpoint_path)
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in history
    )
    metrics_path.write_text(serialized, encoding="utf-8")
    log_path.write_text(serialized, encoding="utf-8")
    return {
        "device": str(selected_device),
        "model": model,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default="finetune_output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    result = train(arguments.config, arguments.output_dir, arguments.device)
    printable = {key: value for key, value in result.items() if key != "model"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "build_finetune_batch",
    "compute_materials_loss",
    "main",
    "train",
    "train_step",
]
