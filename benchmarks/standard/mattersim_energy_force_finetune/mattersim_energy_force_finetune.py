"""Local MatterSim M3GNet and fixed-checkpoint fine-tuning primitives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ase.io import read

from portable_mattersim import M3Gnet, atoms_to_input


CHECKPOINT_SHA256 = "28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be"
DATASET_SHA256 = "46c7ad6375ca6596a39fc5c15c8a0a76e7ead90ec74189b370fae3fcc726e779"
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
EXPECTED_TOP_LEVEL_KEYS = {
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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 does not match the fixed MatterSim 1M model")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("checkpoint top-level schema is not exact")
    if payload["model_name"] != "m3gnet":
        raise ValueError("checkpoint model_name must be m3gnet")
    if payload["model_args"] != EXPECTED_MODEL_ARGS:
        raise ValueError("checkpoint model_args do not match the fixed 1M architecture")
    state = payload["model"]
    if not isinstance(state, dict) or len(state) != EXPECTED_STATE_ENTRIES:
        raise ValueError("checkpoint model state entry count is incorrect")
    if sum(value.numel() for value in state.values()) != EXPECTED_STATE_NUMEL:
        raise ValueError("checkpoint model state element count is incorrect")
    return payload


def build_model(
    model_args: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> M3Gnet:
    args = EXPECTED_MODEL_ARGS if model_args is None else dict(model_args)
    if args != EXPECTED_MODEL_ARGS:
        raise ValueError("model_args must match the fixed MatterSim 1M architecture")
    return M3Gnet(device=str(device), **args).to(device)


def load_pretrained_model(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> M3Gnet:
    payload = load_checkpoint_payload(checkpoint)
    model = build_model(payload["model_args"], device=device)
    state = {name: value.to(device) for name, value in payload["model"].items()}
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint loading reported incompatible keys")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def load_training_structures(path: str | Path):
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if file_sha256(dataset_path) != DATASET_SHA256:
        raise ValueError("training dataset SHA256 does not match high_level_water.xyz")
    structures = read(dataset_path, index=":")
    if not isinstance(structures, list) or len(structures) != 10:
        raise ValueError("the fixed training dataset must contain ten structures")
    return structures


def build_finetune_batch(
    dataset_path: str | Path,
    structure_index: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    structures = load_training_structures(dataset_path)
    if not 0 <= int(structure_index) < len(structures):
        raise IndexError("structure_index is outside the fixed dataset")
    atoms = structures[int(structure_index)]
    energy = torch.tensor(
        [float(atoms.get_potential_energy())], dtype=torch.float32, device=device
    )
    forces = torch.as_tensor(
        atoms.get_forces(), dtype=torch.float32, device=device
    )
    if forces.shape != (len(atoms), 3):
        raise ValueError("force labels must have shape [N, 3]")
    if not bool(torch.isfinite(energy).all() and torch.isfinite(forces).all()):
        raise ValueError("training labels must be finite")
    graph = atoms_to_input(
        atoms,
        cutoff=EXPECTED_MODEL_ARGS["cutoff"],
        threebody_cutoff=EXPECTED_MODEL_ARGS["threebody_cutoff"],
        device=device,
    )
    return {
        "graph": graph,
        "energy": energy,
        "forces": forces,
        "num_atoms": torch.tensor([len(atoms)], dtype=torch.long, device=device),
        "structure_index": int(structure_index),
    }


def predict_energy_forces(
    model: M3Gnet,
    batch: dict[str, Any],
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    graph = dict(batch["graph"])
    positions = graph["atom_pos"].detach().clone().requires_grad_(True)
    graph["atom_pos"] = positions
    energy = model(graph)
    forces = -torch.autograd.grad(
        energy,
        positions,
        grad_outputs=torch.ones_like(energy),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    output = {"energy": energy, "forces": forces}
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("MatterSim produced non-finite energy or forces")
    return output


def compute_energy_force_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, Any],
    force_loss_ratio: float = 1.0,
    huber_delta: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    atoms_per_graph = batch["num_atoms"].to(prediction["energy"].dtype)
    energy_loss = F.huber_loss(
        prediction["energy"] / atoms_per_graph,
        batch["energy"] / atoms_per_graph,
        delta=float(huber_delta),
    )
    force_loss = F.huber_loss(
        prediction["forces"], batch["forces"], delta=float(huber_delta)
    )
    total = energy_loss + float(force_loss_ratio) * force_loss
    return total, {"energy_loss": energy_loss, "force_loss": force_loss}


class MatterSimEnergyForceFineTuner:
    """One-batch full-parameter fine-tuning helper used by the training entrypoint."""

    def __init__(
        self,
        model: M3Gnet,
        lr: float = 2.0e-4,
        force_loss_ratio: float = 1.0,
        huber_delta: float = 0.01,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.force_loss_ratio = float(force_loss_ratio)
        self.huber_delta = float(huber_delta)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr), eps=1e-7)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = predict_energy_forces(self.model, batch, create_graph=True)
        loss, components = compute_energy_force_loss(
            prediction,
            batch,
            force_loss_ratio=self.force_loss_ratio,
            huber_delta=self.huber_delta,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        self.optimizer.step()
        values = {
            "loss": float(loss.detach().cpu()),
            "energy_loss": float(components["energy_loss"].detach().cpu()),
            "force_loss": float(components["force_loss"].detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise FloatingPointError("training metrics must be finite")
        return values


__all__ = [
    "M3Gnet",
    "MatterSimEnergyForceFineTuner",
    "build_finetune_batch",
    "build_model",
    "compute_energy_force_loss",
    "file_sha256",
    "load_checkpoint_payload",
    "load_pretrained_model",
    "load_training_structures",
    "predict_energy_forces",
]
