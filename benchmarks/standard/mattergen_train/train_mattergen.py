"""Deterministic one-step MatterGen training on a compact crystal fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from mattergen_denoiser import build_model, load_config
from portable_mattergen import CrystalBatch


DEFAULT_CONFIG = Path(__file__).with_name("mattergen_train.yaml")


def build_synthetic_batch(seed: int = 20260818) -> dict[str, Any]:
    """Build two deterministic noisy crystals and their denoising targets."""
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
    timesteps = torch.tensor([0.125, 0.875], dtype=torch.float32)
    batch_index = torch.repeat_interleave(torch.arange(2), num_atoms)

    pos_noise = torch.randn(clean_pos.shape, generator=generator) * 0.04
    raw_cell_noise = torch.randn(clean_cell.shape, generator=generator) * 0.02
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
    )
    targets = {
        "pos": -pos_noise,
        "cell": -cell_noise,
        "atomic_numbers": clean_atomic_numbers - 1,
    }
    return {
        "batch": batch,
        "timesteps": timesteps,
        "targets": targets,
        "atom_loss_mask": atom_loss_mask,
    }


def compute_materials_loss(
    output: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    atom_loss_mask: torch.Tensor,
    loss_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the three MatterGen denoising losses and their weighted sum."""
    pos_loss = F.mse_loss(output["pos"], targets["pos"], reduction="sum")
    cell_loss = F.mse_loss(output["cell"], targets["cell"], reduction="sum")
    atomic_numbers_loss = F.cross_entropy(
        output["atomic_numbers"][atom_loss_mask],
        targets["atomic_numbers"][atom_loss_mask],
        reduction="sum",
    )
    components = {
        "pos": pos_loss,
        "cell": cell_loss,
        "atomic_numbers": atomic_numbers_loss,
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
) -> dict[str, float | int]:
    """Run one real forward/backward/Adam update and return scalar metrics."""
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
        raise FloatingPointError("training loss is not finite")
    loss.backward()
    parameters_with_grad = sum(
        parameter.grad is not None for parameter in model.parameters()
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(gradient_clip_val),
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {
        "step": 1,
        "loss": float(loss.detach().cpu()),
        "pos_loss": float(components["pos"].detach().cpu()),
        "cell_loss": float(components["cell"].detach().cpu()),
        "atomic_numbers_loss": float(
            components["atomic_numbers"].detach().cpu()
        ),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "parameters_with_grad": int(parameters_with_grad),
    }


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
    output_dir: str | Path = "training_output",
    device: str = "auto",
) -> dict[str, Any]:
    """Execute the fixed one-step training run and write all artifacts."""
    config = load_config(config_path)
    training = config["training"]
    if int(training["steps"]) != 1:
        raise ValueError("this evaluation task requires exactly one training step")
    torch.manual_seed(int(training["seed"]))
    selected_device = _resolve_device(device)
    model = build_model(config).to(selected_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"])
    )
    fixture = _fixture_to_device(
        build_synthetic_batch(seed=int(training["seed"])), selected_device
    )
    metrics = train_step(
        model=model,
        optimizer=optimizer,
        fixture=fixture,
        loss_weights=training["loss_weights"],
        gradient_clip_val=float(training["gradient_clip_val"]),
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_config = config["artifacts"]
    checkpoint_path = destination / artifact_config["checkpoint"]
    metrics_path = destination / artifact_config["metrics"]
    log_path = destination / artifact_config["log"]
    torch.save(
        {
            "step": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_path.write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "device": str(selected_device),
        "metrics": metrics,
        "checkpoint": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default="training_output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    result = train(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        device=arguments.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
