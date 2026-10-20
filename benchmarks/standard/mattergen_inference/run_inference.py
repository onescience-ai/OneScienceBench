"""Command-line and Python inference entry points for MatterGen denoising."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mattergen_denoiser import load_config, load_model
from portable_mattergen import CrystalBatch


OUTPUT_KEYS = ("pos", "cell", "atomic_numbers")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU device was requested but is unavailable")
    return torch.device(requested)


def validate_inputs(batch: Mapping[str, torch.Tensor], timesteps: torch.Tensor) -> None:
    required = {"pos", "cell", "atomic_numbers", "num_atoms"}
    if set(batch) != required:
        raise ValueError(f"batch keys must be exactly {sorted(required)}")
    pos = batch["pos"]
    cell = batch["cell"]
    atom_types = batch["atomic_numbers"]
    num_atoms = batch["num_atoms"]
    if pos.ndim != 2 or pos.shape[1] != 3 or not pos.is_floating_point():
        raise ValueError("pos must be a floating tensor with shape [N, 3]")
    if cell.ndim != 3 or cell.shape[1:] != (3, 3) or not cell.is_floating_point():
        raise ValueError("cell must be a floating tensor with shape [B, 3, 3]")
    if atom_types.shape != (pos.shape[0],) or atom_types.dtype != torch.long:
        raise ValueError("atomic_numbers must be int64 with shape [N]")
    if num_atoms.shape != (cell.shape[0],) or num_atoms.dtype != torch.long:
        raise ValueError("num_atoms must be int64 with shape [B]")
    if int(num_atoms.sum()) != pos.shape[0] or torch.any(num_atoms <= 0):
        raise ValueError("num_atoms must be positive and sum to N")
    if torch.any((atom_types < 1) | (atom_types > 101)):
        raise ValueError("atomic_numbers must be in [1, 101]")
    if timesteps.shape != (cell.shape[0],) or not timesteps.is_floating_point():
        raise ValueError("timesteps must be floating with shape [B]")
    floating = (pos, cell, timesteps)
    if not all(bool(torch.isfinite(value).all()) for value in floating):
        raise ValueError("floating inputs must contain finite values")


def evaluate(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    timesteps: torch.Tensor,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    validate_inputs(batch, timesteps)
    target = torch.device(device)
    crystal_batch = CrystalBatch.from_mapping(batch).to(target)
    model = model.to(target).eval()
    with torch.no_grad():
        prediction = model(crystal_batch, timesteps.to(target))
    result = {key: prediction[key].detach().cpu() for key in OUTPUT_KEYS}
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise ValueError("model output contains NaN or Inf")
    return result


def load_input_npz(path: str | Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    with np.load(path, allow_pickle=False) as arrays:
        expected = {"pos", "cell", "atomic_numbers", "num_atoms", "timesteps"}
        if set(arrays.files) != expected:
            raise ValueError(f"input NPZ keys must be exactly {sorted(expected)}")
        batch = {
            "pos": torch.as_tensor(arrays["pos"], dtype=torch.float32),
            "cell": torch.as_tensor(arrays["cell"], dtype=torch.float32),
            "atomic_numbers": torch.as_tensor(arrays["atomic_numbers"], dtype=torch.long),
            "num_atoms": torch.as_tensor(arrays["num_atoms"], dtype=torch.long),
        }
        timesteps = torch.as_tensor(arrays["timesteps"], dtype=torch.float32)
    return batch, timesteps


def save_output_npz(path: str | Path, output: Mapping[str, torch.Tensor]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        **{key: output[key].detach().cpu().numpy() for key in OUTPUT_KEYS},
    )
    return output_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run one MatterGen denoising step")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    config = load_config(args.config)
    model = load_model(config, args.checkpoint, device=device)
    batch, timesteps = load_input_npz(args.input)
    return save_output_npz(args.output, evaluate(model, batch, timesteps, device))


if __name__ == "__main__":
    main()
