from __future__ import annotations

from pathlib import Path

import torch

from portable_esen.model import (
    CHECKPOINT,
    JD_PATH,
    ESENModel,
    atoms_to_data,
    build_model,
    checkpoint_sha256,
    load_checkpoint_payload,
    load_model,
)


def load_training_structures(path, limit=None):
    from ase.db import connect
    db = connect(str(path))
    structures = [db.get(index).toatoms() for index in range(1, len(db) + 1)]
    return structures if limit is None else structures[: int(limit)]


def build_finetune_batch(path, structure_index=0, device="cpu"):
    structures = load_training_structures(path)
    atoms = structures[int(structure_index)]
    data = atoms_to_data(atoms, device=device)
    energy = torch.tensor([float(atoms.get_potential_energy())], dtype=torch.float32, device=device)
    forces = torch.as_tensor(atoms.get_forces(), dtype=torch.float32, device=device)
    return {"data": data, "energy": energy, "forces": forces, "natoms": len(atoms), "structure_index": int(structure_index)}


def trainable_named_parameters(model):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "denoising_linear" not in name
    ]


def compute_energy_force_loss(prediction, batch, payload, huber_delta=0.01, force_loss_ratio=1.0):
    normalizer = payload.get("config", {}).get("dataset", {}).get("transforms", {}).get("normalizer", {})
    e_stdev = float(normalizer.get("energy", {}).get("stdev", 1.0))
    f_stdev = float(normalizer.get("forces", {}).get("stdev", 1.0))
    refs = payload.get("elementrefs", {}).get("energy", {}).get("element_references", torch.zeros(119))
    refs = torch.as_tensor(refs, dtype=batch["data"]["atomic_numbers"].dtype, device=batch["data"]["atomic_numbers"].device)
    energy_target = (batch["energy"] - refs[batch["data"]["atomic_numbers"]].sum().reshape(1)) / e_stdev
    force_target = batch["forces"] / f_stdev
    energy_loss = torch.nn.functional.huber_loss(
        prediction["energy"].reshape_as(energy_target) / batch["natoms"],
        energy_target / batch["natoms"],
        delta=huber_delta,
    )
    force_loss = torch.nn.functional.huber_loss(prediction["forces"], force_target, delta=huber_delta)
    total = energy_loss + force_loss_ratio * force_loss
    return total, {"energy_loss": energy_loss.detach(), "force_loss": force_loss.detach()}


def predict_energy_forces(model, batch):
    prediction = model(batch["data"])
    stress = prediction.get("stress")
    if isinstance(stress, torch.Tensor) and stress.ndim == 2 and stress.shape[-1] == 9:
        prediction = dict(prediction)
        prediction["stress"] = stress.reshape(stress.shape[0], 3, 3)
    return prediction


__all__ = ["CHECKPOINT", "JD_PATH", "ESENModel", "checkpoint_sha256", "load_checkpoint_payload", "build_model", "load_model", "load_training_structures", "build_finetune_batch", "trainable_named_parameters", "compute_energy_force_loss", "predict_energy_forces"]
