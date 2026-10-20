from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .esen_dens import MLP_EFS_Head, eSEN_DeNS_Backbone


CHECKPOINT = "/public/share/sugonhpcapp01/onestore/onemodels/eSEN/esen_30m_mptrj.pt"
JD_PATH = "/public/share/sugonhpcapp01/onestore/onemodels/UMA/checkpoint/Jd.pt"
CHECKPOINT_SHA256 = "f44aecdfbdb74575f8861c08ee6aa86939cce4571e73ac508666190cb67f064b"


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__


class ESENModel(nn.Module):
    def __init__(self, jd_path: str = JD_PATH):
        super().__init__()
        self.backbone = eSEN_DeNS_Backbone(
            max_num_elements=100, sphere_channels=128, lmax=3, mmax=2,
            otf_graph=False, max_neighbors=300, use_pbc=True, use_pbc_single=True,
            cutoff=6.0, edge_channels=128, distance_function="gaussian",
            num_distance_basis=10, direct_forces=False, regress_forces=True,
            regress_stress=True, num_layers=10, hidden_channels=128,
            norm_type="rms_norm_sh", act_type="gate", mlp_type="spectral",
            use_envelope=True, jd_path=jd_path,
        )
        self.output_heads = nn.ModuleDict({"mptrj": MLP_EFS_Head(self.backbone)})

    def forward(self, data):
        return self.output_heads["mptrj"](data, self.backbone(data))


def checkpoint_sha256(path: str | Path = CHECKPOINT) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_payload(path: str | Path = CHECKPOINT) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise TypeError("eSEN checkpoint must contain a tensor state_dict")
    return payload


def build_model(device="cpu", jd_path: str = JD_PATH):
    return ESENModel(jd_path).to(device)


def load_model(checkpoint=CHECKPOINT, device="cpu", jd_path: str = JD_PATH):
    payload = load_checkpoint_payload(checkpoint)
    state = payload["state_dict"]
    local_state = {str(key)[len("module."):]: value for key, value in state.items() if str(key).startswith("module.")}
    if len(local_state) != len(state):
        raise ValueError("checkpoint state keys must use the module. prefix")
    model = build_model(device, jd_path)
    model.load_state_dict(local_state, strict=True)
    return model, payload


def _normalizer(payload, name):
    item = payload.get("normalizers", {}).get(name, {})
    mean, scale = item.get("mean", 0.0), item.get("rmsd", item.get("stdev", 1.0))
    mean = float(mean) if not torch.is_tensor(mean) else float(mean.item())
    scale = float(scale) if not torch.is_tensor(scale) else float(scale.item())
    return mean, scale


def energy_references(payload, atomic_numbers):
    state = payload.get("elementrefs", {}).get("energy", {})
    refs = state.get("element_references", torch.zeros(119)) if isinstance(state, dict) else torch.zeros(119)
    refs = torch.as_tensor(refs, dtype=torch.float32, device=atomic_numbers.device)
    return refs[atomic_numbers].sum().reshape(1)


def atoms_to_data(atoms, device="cpu"):
    import numpy as np
    from ase.neighborlist import neighbor_list
    if len(atoms) == 0 or not bool(np.all(atoms.pbc)):
        raise ValueError("eSEN fixture requires a non-empty fully periodic structure")
    i, j, shifts = neighbor_list("ijS", atoms, 6.0, self_interaction=False)
    if len(i) == 0:
        raise ValueError("structure has no neighbors")
    data = AttrDict(
        pos=torch.as_tensor(atoms.positions, dtype=torch.float32, device=device),
        cell=torch.as_tensor(atoms.cell.array[None], dtype=torch.float32, device=device),
        atomic_numbers=torch.as_tensor(atoms.numbers, dtype=torch.long, device=device),
        edge_index=torch.as_tensor([j, i], dtype=torch.long, device=device),
        cell_offsets=torch.as_tensor(shifts, dtype=torch.float32, device=device),
        nedges=torch.tensor([len(i)], dtype=torch.long, device=device),
        natoms=torch.tensor([len(atoms)], dtype=torch.long, device=device),
        batch=torch.zeros(len(atoms), dtype=torch.long, device=device),
        batch_full=torch.zeros(len(atoms), dtype=torch.long, device=device),
        atomic_numbers_full=torch.as_tensor(atoms.numbers, dtype=torch.long, device=device),
        num_graphs=1,
    )
    return data


def predict_energy_forces(model, atoms, payload, device="cpu"):
    data = atoms_to_data(atoms, device)
    raw = model(data)
    out = {}
    for name in ("energy", "forces", "stress"):
        mean, stdev = _normalizer(payload, name)
        out[name] = raw[name] * stdev + mean
    out["energy"] = out["energy"] + energy_references(payload, data["atomic_numbers"])
    out["stress"] = out["stress"].reshape(-1, 3, 3)
    return out


__all__ = ["ESENModel", "AttrDict", "checkpoint_sha256", "load_checkpoint_payload", "build_model", "load_model", "atoms_to_data", "predict_energy_forces"]
