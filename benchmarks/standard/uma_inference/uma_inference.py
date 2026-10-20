from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from atomic_data import AtomicData, atomicdata_list_to_batch
from base import HydraModel, HydraModelV2
from uma_common_utils import load_model_and_weights_from_checkpoint
from uma_escn_md import eSCNMDBackbone  # noqa: F401 - imported for registry side effects
from uma_escn_moe import eSCNMDMoeBackbone  # noqa: F401 - imported for registry side effects
from uma_head import (
    DatasetSpecificMoEWrapper,
    DatasetSpecificSingleHeadWrapper,
    Linear_Energy_Head,
    Linear_Force_Head,
    MLP_EFS_Head,
    MLP_Energy_Head,
    MLP_Stress_Head,
)
from uma_registry import registry


DEFAULT_CHECKPOINT_DIR = Path(
    "/public/share/sugonhpcapp01/onestore/onemodels/UMA/checkpoint"
)
DEFAULT_CHECKPOINT_CANDIDATES = (
    "uma-s-1p1_converted.pt",
    "uma-s-1p1.pt",
)

_HEAD_REGISTRY = {
    "DatasetSpecificMoEWrapper": DatasetSpecificMoEWrapper,
    "DatasetSpecificSingleHeadWrapper": DatasetSpecificSingleHeadWrapper,
    "Linear_Energy_Head": Linear_Energy_Head,
    "Linear_Force_Head": Linear_Force_Head,
    "MLP_EFS_Head": MLP_EFS_Head,
    "MLP_Energy_Head": MLP_Energy_Head,
    "MLP_Stress_Head": MLP_Stress_Head,
    "dataset_specific_moe_wrapper": DatasetSpecificMoEWrapper,
    "dataset_specific_single_head_wrapper": DatasetSpecificSingleHeadWrapper,
    "linear_energy_head": Linear_Energy_Head,
    "linear_force_head": Linear_Force_Head,
    "mlp_efs_head": MLP_EFS_Head,
    "mlp_energy_head": MLP_Energy_Head,
    "mlp_stress_head": MLP_Stress_Head,
}


def _ensure_registry() -> None:
    for name, cls in _HEAD_REGISTRY.items():
        registry.mapping["model_name_mapping"].setdefault(name, cls)
    registry.mapping["model_name_mapping"].setdefault("hydra", HydraModel)


def _resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        path = (DEFAULT_CHECKPOINT_DIR / candidate).resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No default UMA checkpoint found under {DEFAULT_CHECKPOINT_DIR}"
    )


def _infer_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    _ensure_registry()
    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    model = load_model_and_weights_from_checkpoint(str(resolved_checkpoint))
    model.eval()
    model.to(_infer_device(device))
    return model


def _guess_dataset_name(model: torch.nn.Module, sample: Any) -> str | None:
    if isinstance(sample, dict):
        value = sample.get("dataset") or sample.get("task_name")
        if isinstance(value, str) and value:
            return value
    info = getattr(sample, "info", None)
    if isinstance(info, dict):
        value = info.get("dataset") or info.get("task_name")
        if isinstance(value, str) and value:
            return value

    backbone = getattr(model, "backbone", None)
    dataset_list = getattr(backbone, "dataset_list", None)
    if dataset_list:
        try:
            first = next(iter(dataset_list))
        except TypeError:
            first = None
        if isinstance(first, str) and first:
            return first

    dataset_embedding = getattr(backbone, "dataset_embedding", None)
    dataset_emb_dict = getattr(dataset_embedding, "dataset_emb_dict", None)
    if dataset_emb_dict:
        keys = list(dataset_emb_dict.keys())
        if keys:
            return str(keys[0])
    return None


def _to_atomic_data(
    sample: Any,
    model: torch.nn.Module,
) -> AtomicData:
    if isinstance(sample, AtomicData):
        return sample

    if isinstance(sample, dict):
        if "atoms" in sample:
            return _to_atomic_data(sample["atoms"], model)
        if "ase_atoms" in sample:
            return _to_atomic_data(sample["ase_atoms"], model)
        required = {"pos", "atomic_numbers", "cell", "pbc", "natoms", "edge_index", "cell_offsets", "nedges", "charge", "spin", "fixed", "tags"}
        if required.issubset(sample):
            return AtomicData.from_dict(sample)

    if hasattr(sample, "get_positions") and hasattr(sample, "get_atomic_numbers"):
        backbone = getattr(model, "backbone", None)
        radius = float(getattr(backbone, "cutoff", 6.0))
        max_neigh = int(getattr(backbone, "max_neighbors", 300))
        task_name = _guess_dataset_name(model, sample)
        molecule_cell_size = None
        cell = sample.get_cell()
        # AtomicData.from_ase(..., r_edges=True) eventually uses pymatgen to
        # build neighbors. Non-periodic molecules read from plain xyz files
        # typically have a singular/zero cell, so give them a finite vacuum box.
        if cell is None or abs(float(cell.volume)) < 1e-8:
            molecule_cell_size = max(radius + 4.0, 12.0)
        data = AtomicData.from_ase(
            sample,
            r_edges=True,
            radius=radius,
            max_neigh=max_neigh,
            molecule_cell_size=molecule_cell_size,
            task_name=task_name,
        )
        if task_name and getattr(data, "dataset", None) is not None and isinstance(data.dataset, str):
            data.dataset = [data.dataset]
        return data

    raise TypeError(
        "Unsupported sample type for UMA inference. Expected AtomicData, ASE Atoms, "
        f"or a dict-like LMDB sample, got {type(sample)}"
    )


def _to_batched_atomic_data(
    sample: Any,
    model: torch.nn.Module,
) -> AtomicData:
    if isinstance(sample, (list, tuple)):
        data_list = [_to_atomic_data(item, model) for item in sample]
        if not data_list:
            raise ValueError("Input sample list is empty")
        return atomicdata_list_to_batch(data_list)
    return _to_atomic_data(sample, model)


def _move_atomic_data_to_device(data: AtomicData, device: torch.device) -> AtomicData:
    return data.to(device)


def _unwrap_prediction_tree(output: Any) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                next_prefix = key if not prefix else f"{prefix}.{key}"
                walk(next_prefix, item)
        else:
            flat[prefix] = value

    walk("", output)
    return flat


def _pick_prediction(flat: dict[str, Any], target: str) -> torch.Tensor:
    exact = [value for key, value in flat.items() if key.split(".")[-1] == target]
    if exact:
        return exact[0]
    suffix = [value for key, value in flat.items() if key.endswith(target)]
    if suffix:
        return suffix[0]
    raise KeyError(f"Could not find `{target}` in model output keys: {sorted(flat)}")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


@torch.no_grad()
def predict(
    sample: Any,
    model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    if model is None:
        model = load_model(checkpoint_path=checkpoint_path, device=device)

    runtime_device = _infer_device(device) if device is not None else next(model.parameters()).device
    data = _to_batched_atomic_data(sample, model)
    data = _move_atomic_data_to_device(data, runtime_device)

    output = model(data)
    flat = _unwrap_prediction_tree(output)
    energy = _to_numpy(_pick_prediction(flat, "energy")).reshape(-1)
    forces = _to_numpy(_pick_prediction(flat, "forces"))
    return {
        "energy": energy,
        "forces": forces,
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UMA eSCN-MoE inference helper")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument(
        "--xyz",
        type=str,
        default=None,
        help="Optional ASE-readable structure file for a quick single-sample inference run",
    )
    return parser


def _main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    model = load_model(checkpoint_path=args.checkpoint, device=args.device)
    if args.xyz is None:
        print(model.__class__.__name__)
        return

    from ase.io import read

    atoms = read(args.xyz)
    prediction = predict(atoms, model=model, device=args.device)
    print(
        {
            "energy": prediction["energy"].tolist(),
            "forces_shape": list(prediction["forces"].shape),
        }
    )


if __name__ == "__main__":
    _main()
