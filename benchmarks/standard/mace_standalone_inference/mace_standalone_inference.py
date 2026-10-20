"""Reference runner for standalone inference with a portable MACE TorchScript model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list


METADATA_FILENAME = "mace_portable_metadata.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_atoms(atoms: Atoms) -> Atoms:
    if not isinstance(atoms, Atoms):
        raise TypeError(f"expected ase.Atoms, got {type(atoms)!r}")
    if len(atoms) == 0:
        raise ValueError("empty structures are not supported")
    numbers = np.asarray(atoms.get_atomic_numbers())
    positions = np.asarray(atoms.get_positions())
    cell = np.asarray(atoms.cell.array)
    if numbers.shape != (len(atoms),) or np.any(numbers <= 0):
        raise ValueError("atomic numbers must be positive integers")
    if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        raise ValueError("positions must have shape (N, 3) and contain finite values")
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError("cell must have shape (3, 3) and contain finite values")
    if np.any(atoms.pbc) and abs(float(np.linalg.det(cell))) < 1.0e-12:
        raise ValueError("periodic structures require a non-singular cell")
    return atoms.copy()


def _read_ase(source: str | Path | Sequence[Atoms] | Atoms) -> list[Atoms]:
    if isinstance(source, Atoms):
        return [_validate_atoms(source)]
    if isinstance(source, (str, Path)):
        frames = read(str(source), index=":")
        structures = frames if isinstance(frames, list) else [frames]
    else:
        structures = list(source)
    if not structures:
        raise ValueError("ASE input contains no structures")
    return [_validate_atoms(atoms) for atoms in structures]


def _hdf5_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("HDF5 input file must end in .h5 or .hdf5")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"HDF5 input is unavailable: {path}")
    files = sorted((*path.glob("*.h5"), *path.glob("*.hdf5")))
    if not files:
        raise ValueError(f"HDF5 directory contains no .h5 or .hdf5 files: {path}")
    return files


def _atoms_from_hdf5_group(group: h5py.Group) -> Atoms:
    numbers = np.asarray(group["atomic_numbers"][()], dtype=np.int64)
    positions = np.asarray(group["positions"][()], dtype=np.float64)
    cell = (
        np.asarray(group["cell"][()], dtype=np.float64)
        if "cell" in group
        else np.zeros((3, 3), dtype=np.float64)
    )
    pbc = np.asarray(group["pbc"][()], dtype=bool) if "pbc" in group else False
    return _validate_atoms(Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc))


def _read_hdf5(path: Path) -> list[Atoms]:
    structures: list[Atoms] = []
    for filename in _hdf5_files(path):
        with h5py.File(filename, "r") as handle:
            groups: list[h5py.Group] = []
            if "atomic_numbers" in handle and "positions" in handle:
                groups.append(handle)

            def collect(_: str, obj: h5py.Group | h5py.Dataset) -> None:
                if (
                    isinstance(obj, h5py.Group)
                    and "atomic_numbers" in obj
                    and "positions" in obj
                ):
                    groups.append(obj)

            handle.visititems(collect)
            structures.extend(_atoms_from_hdf5_group(group) for group in groups)
    if not structures:
        raise ValueError(f"no HDF5 structures found in {path}")
    return structures


def _read_structures(
    source: str | Path | Sequence[Atoms] | Atoms,
    data_format: str,
) -> list[Atoms]:
    if data_format == "ase":
        return _read_ase(source)
    if data_format == "hdf5":
        if not isinstance(source, (str, Path)):
            raise TypeError("HDF5 input_path must be a file or directory path")
        return _read_hdf5(Path(source))
    raise ValueError("data_format must be 'ase' or 'hdf5'")


def _model_dtype(metadata: dict[str, Any]) -> torch.dtype:
    name = metadata.get("dtype")
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported portable model dtype: {name!r}")


def _build_graph(
    atoms: Atoms,
    metadata: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dtype = _model_dtype(metadata)
    model_numbers = [int(value) for value in metadata["atomic_numbers"]]
    index_by_number = {number: index for index, number in enumerate(model_numbers)}
    numbers = atoms.get_atomic_numbers().astype(np.int64)
    unknown = sorted(set(numbers.tolist()) - set(index_by_number))
    if unknown:
        raise ValueError(f"portable model does not support atomic numbers: {unknown}")

    senders, receivers, unit_shifts = neighbor_list(
        "ijS",
        atoms,
        cutoff=float(metadata["r_max"]),
        self_interaction=False,
    )
    edge_index = np.stack((senders, receivers), axis=0).astype(np.int64, copy=False)
    unit_shifts = np.asarray(unit_shifts, dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    shifts = unit_shifts @ cell

    node_attrs = np.zeros((len(atoms), len(model_numbers)), dtype=np.float64)
    columns = [index_by_number[int(number)] for number in numbers]
    node_attrs[np.arange(len(atoms)), columns] = 1.0

    return {
        "positions": torch.as_tensor(atoms.get_positions(), dtype=dtype, device=device),
        "node_attrs": torch.as_tensor(node_attrs, dtype=dtype, device=device),
        "edge_index": torch.as_tensor(edge_index, dtype=torch.long, device=device),
        "batch": torch.zeros(len(atoms), dtype=torch.long, device=device),
        "cell": torch.as_tensor(cell, dtype=dtype, device=device),
        "shifts": torch.as_tensor(shifts, dtype=dtype, device=device),
        "unit_shifts": torch.as_tensor(unit_shifts, dtype=dtype, device=device),
        "head": torch.zeros(1, dtype=torch.long, device=device),
        "ptr": torch.tensor([0, len(atoms)], dtype=torch.long, device=device),
    }


def _load_portable_model(
    model_path: Path,
    device: torch.device,
) -> tuple[torch.jit.ScriptModule, dict[str, Any]]:
    extra_files = {METADATA_FILENAME: ""}
    model = torch.jit.load(
        str(model_path),
        map_location=device,
        _extra_files=extra_files,
    )
    try:
        metadata = json.loads(extra_files[METADATA_FILENAME])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("portable model metadata is missing or invalid") from exc
    if metadata.get("format") != "mace-portable-torchscript-v1":
        raise ValueError(f"unexpected portable model format: {metadata.get('format')!r}")
    required = {"r_max", "atomic_numbers", "heads", "dtype"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"portable model metadata is missing fields: {sorted(missing)}")
    if len(metadata["heads"]) != 1:
        raise ValueError("this runner requires a single-head portable model")
    model.eval()
    return model, metadata


class MACEStandaloneInferenceRunner:
    """Run portable MACE energy/force inference without model-package imports."""

    def __init__(
        self,
        model_path: str | Path,
        input_path: str | Path | Sequence[Atoms] | Atoms,
        output_dir: str | Path,
        data_format: str = "ase",
        device: str = "cpu",
    ) -> None:
        if data_format not in {"ase", "hdf5"}:
            raise ValueError("data_format must be 'ase' or 'hdf5'")
        self.model_path = Path(model_path).expanduser().resolve()
        self.input_path = (
            Path(input_path).expanduser().resolve()
            if isinstance(input_path, (str, Path))
            else input_path
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.data_format = data_format
        self.device = torch.device(device)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"portable model is unavailable: {self.model_path}")
        if isinstance(self.input_path, Path) and not self.input_path.exists():
            raise FileNotFoundError(f"input is unavailable: {self.input_path}")

    def run(self) -> Path:
        model, metadata = _load_portable_model(self.model_path, self.device)
        structures = _read_structures(self.input_path, self.data_format)
        predictions = []
        with torch.enable_grad():
            for index, atoms in enumerate(structures):
                output = model(
                    _build_graph(atoms, metadata, self.device),
                    training=False,
                    compute_force=True,
                    compute_virials=False,
                    compute_stress=False,
                    compute_displacement=False,
                    compute_hessian=False,
                    compute_edge_forces=False,
                    compute_atomic_stresses=False,
                    lammps_mliap=False,
                )
                energy = output.get("energy")
                forces = output.get("forces")
                if energy is None or forces is None:
                    raise ValueError("portable model did not return energy and forces")
                energy_values = energy.detach().cpu().reshape(-1)
                force_values = forces.detach().cpu().numpy()
                if energy_values.numel() != 1:
                    raise ValueError(f"expected one graph energy, got shape {tuple(energy.shape)}")
                if force_values.shape != (len(atoms), 3):
                    raise ValueError(
                        f"expected force shape {(len(atoms), 3)}, got {force_values.shape}"
                    )
                if not np.isfinite(force_values).all() or not torch.isfinite(energy_values).all():
                    raise ValueError("model produced non-finite energy or forces")
                predictions.append(
                    {
                        "index": index,
                        "atomic_numbers": atoms.get_atomic_numbers().astype(int).tolist(),
                        "energy": float(energy_values[0].item()),
                        "forces": force_values.astype(np.float64, copy=False).tolist(),
                    }
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in predictions),
            encoding="utf-8",
        )
        manifest = {
            "model_path": str(self.model_path),
            "model_sha256": _sha256(self.model_path),
            "model_format": metadata["format"],
            "source_model_sha256": metadata.get("source_model_sha256"),
            "data_format": self.data_format,
            "num_structures": len(predictions),
            "device": str(self.device),
            "dtype": metadata["dtype"],
            "r_max": float(metadata["r_max"]),
        }
        (self.output_dir / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.output_dir
