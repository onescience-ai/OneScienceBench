"""Strict loading and differentiable inference for the fixed MatterSim model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from portable_mattersim import M3Gnet, atoms_to_input


CHECKPOINT_SHA256 = "28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be"
EXPECTED_MODEL_ARGS = {
    "num_blocks": 3,
    "units": 128,
    "max_l": 4,
    "max_n": 4,
    "cutoff": 5.0,
    "max_z": 94,
    "threebody_cutoff": 4.0,
}
EXPECTED_STATE_ENTRIES = 89
EXPECTED_STATE_NUMEL = 890_246


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 does not match the fixed MatterSim model")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    required = {
        "model_name",
        "model",
        "model_args",
        "optimizer",
        "ema",
        "scheduler",
        "last_epoch",
        "validation_metrics",
        "description",
    }
    if set(payload) != required:
        raise ValueError("checkpoint top-level schema is not exact")
    if payload["model_name"] != "m3gnet":
        raise ValueError("checkpoint model_name must be m3gnet")
    if payload["model_args"] != EXPECTED_MODEL_ARGS:
        raise ValueError("checkpoint model_args do not match the fixed 1M model")
    state = payload["model"]
    if not isinstance(state, dict) or len(state) != EXPECTED_STATE_ENTRIES:
        raise ValueError("checkpoint model state entry count is incorrect")
    if sum(value.numel() for value in state.values()) != EXPECTED_STATE_NUMEL:
        raise ValueError("checkpoint model state element count is incorrect")
    return payload


def build_model(model_args: dict[str, Any], device: str | torch.device = "cpu") -> M3Gnet:
    if dict(model_args) != EXPECTED_MODEL_ARGS:
        raise ValueError("model_args must match the fixed MatterSim 1M architecture")
    return M3Gnet(device=str(device), **model_args).to(device)


def load_model(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> M3Gnet:
    payload = load_checkpoint_payload(checkpoint)
    model = build_model(payload["model_args"], device=device)
    state = {name: value.to(device) for name, value in payload["model"].items()}
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict checkpoint loading reported incompatible keys")
    model.eval()
    return model


def evaluate_structure(
    model: M3Gnet,
    atoms,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    graph = atoms_to_input(
        atoms,
        cutoff=float(model.model_args["cutoff"]),
        threebody_cutoff=float(model.model_args["threebody_cutoff"]),
        device=device,
    )
    atom_pos = graph["atom_pos"].detach().requires_grad_(True)
    strain = torch.zeros_like(graph["cell"], requires_grad=True)
    identity = torch.eye(3, device=graph["cell"].device)[None]
    graph["cell"] = torch.matmul(graph["cell"], identity + strain)
    graph["atom_pos"] = torch.einsum(
        "bi,bij->bj", atom_pos, (identity + strain)[graph["batch"]]
    )
    volume = torch.linalg.det(graph["cell"])
    energy = model(graph)
    force_grad, stress_grad = torch.autograd.grad(
        energy,
        (atom_pos, strain),
        grad_outputs=torch.ones_like(energy),
        create_graph=False,
    )
    output = {
        "energy": energy.reshape(()),
        "forces": -force_grad,
        # ASE exposes full stress tensors in eV / Angstrom^3.
        "stress": stress_grad[0] / volume[0],
    }
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("MatterSim produced non-finite predictions")
    return output


__all__ = [
    "M3Gnet",
    "build_model",
    "checkpoint_sha256",
    "evaluate_structure",
    "load_checkpoint_payload",
    "load_model",
]
