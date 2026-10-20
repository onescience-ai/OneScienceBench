"""JAX-free data utilities for the NequIX torch backend.

Provides the graph preprocessing, the PyTorch-Geometric conversion, and the
ASE-database dataset used by the NequIX torch training/inference path. This
module is intentionally free of JAX/jraph so it can be imported and used on the
DCU/ROCm target platform. (The upstream NequIX ``data.py`` imports jax/jraph at
module level and contains a jraph-based dynamic-batch ``DataLoader`` that is only
used by the JAX training path; the torch path uses ``torch_geometric.loader.
DataLoader`` instead.)
"""
from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from pathlib import Path

import ase
import ase.db
import numpy as np
from ase.geometry import complete_cell
import matscipy.neighbours


def preprocess_graph(
    atoms: ase.Atoms,
    atom_indices: dict[int, int],
    cutoff: float,
    targets: bool,
) -> dict:
    """Build a NequIX graph dict from an ASE structure.

    Args:
        atoms: ASE structure (positions, cell, pbc, and optionally forces/energy).
        atom_indices: mapping of atomic number -> species index.
        cutoff: neighbor-list cutoff radius.
        targets: if True, include energy/forces/stress labels (for training).
    """
    cell = complete_cell(atoms.cell)  # avoids singular cell
    src, dst, shift = matscipy.neighbours.neighbour_list(
        "ijS", positions=atoms.positions, cell=cell, pbc=atoms.pbc, cutoff=cutoff
    )
    graph_dict = {
        "n_node": np.array([len(atoms)]).astype(np.int32),
        "n_edge": np.array([len(src)]).astype(np.int32),
        "senders": dst.astype(np.int32),
        "receivers": src.astype(np.int32),
        "species": np.array([atom_indices[n] for n in atoms.get_atomic_numbers()]).astype(np.int32),
        "positions": atoms.positions.astype(np.float32),
        "shifts": shift.astype(np.float32),
        "cell": atoms.cell.astype(np.float32) if atoms.pbc.all() else None,
    }
    if targets:
        graph_dict["forces"] = atoms.get_forces().astype(np.float32)
        graph_dict["energy"] = np.array([atoms.get_potential_energy()]).astype(np.float32)
        try:
            graph_dict["stress"] = atoms.get_stress(voigt=False).astype(np.float32)
        except ase.calculators.calculator.PropertyNotImplementedError:
            pass
    return graph_dict


def dict_to_pytorch_geometric(graph_dict: dict):
    """Convert a NequIX graph dict to a ``torch_geometric.data.Data`` object."""
    import torch
    from torch_geometric.data import Data

    species = torch.from_numpy(graph_dict["species"]).long()
    positions = torch.from_numpy(graph_dict["positions"])
    edge_index = torch.stack(
        [torch.from_numpy(graph_dict["senders"]), torch.from_numpy(graph_dict["receivers"])], dim=0
    ).long()
    energy = None if "energy" not in graph_dict else torch.from_numpy(graph_dict["energy"])
    forces = None if "forces" not in graph_dict else torch.from_numpy(graph_dict["forces"])
    stress = (
        None if "stress" not in graph_dict else torch.from_numpy(graph_dict["stress"])[None, :, :]
    )
    edge_attr = torch.from_numpy(graph_dict["shifts"])
    cell = (
        torch.from_numpy(graph_dict["cell"])[None, :, :] if graph_dict["cell"] is not None else None
    )
    n_node = torch.from_numpy(graph_dict["n_node"])
    n_edge = torch.from_numpy(graph_dict["n_edge"])

    return Data(
        n_node=n_node,
        n_edge=n_edge,
        energy=energy,
        forces=forces,
        stress=stress,
        x=species,
        positions=positions,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cell=cell,
    )


def atomic_numbers_to_indices(atomic_numbers: list[int]) -> dict[int, int]:
    """Map atomic numbers to contiguous species indices (sorted)."""
    return {n: i for i, n in enumerate(sorted(atomic_numbers))}


class Dataset(ABC):
    def __init__(self, backend: str = "torch"):
        self.backend = backend

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def _get_graph_dict(self, idx: int) -> dict: ...

    def __getitem__(self, idx):
        graph = self._get_graph_dict(idx)
        return dict_to_pytorch_geometric(graph)

    def split(self, valid_frac: float, seed: int = 42):
        """Split into (train, valid) index subsets."""
        import numpy as _np

        rng = _np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        n_valid = int(round(valid_frac * len(self)))
        valid_idx = idx[:n_valid]
        train_idx = idx[n_valid:]
        return IndexDataset(self, train_idx), IndexDataset(self, valid_idx)


class IndexDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        super().__init__(backend=base.backend)
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def _get_graph_dict(self, i):
        return self.base._get_graph_dict(int(self.indices[i]))


class AseDBDataset(Dataset):
    """Load structures (with energy/forces/stress labels) from ``.aselmdb`` files.

    Accepts a single ``.aselmdb`` file or a directory (all ``*.aselmdb`` under it
    are concatenated).
    """

    def __init__(
        self,
        file_path: str,
        atomic_numbers: list[int],
        cutoff: float = 5.0,
        backend: str = "torch",
    ):
        super().__init__(backend=backend)
        self.atomic_indices = atomic_numbers_to_indices(atomic_numbers)
        self.file_path = Path(file_path)
        self.cutoff = cutoff
        if self.file_path.is_dir():
            files = sorted(self.file_path.rglob("*.aselmdb"))
            self.dbs = [ase.db.connect(fp, readonly=True, use_lock_file=False) for fp in files]
        else:
            self.dbs = [ase.db.connect(file_path, readonly=True, use_lock_file=False)]
        self.db_ids = [db.ids for db in self.dbs]
        self.id_cumulative = np.cumsum([len(ids) for ids in self.db_ids])

    def __len__(self):
        return self.id_cumulative[-1]

    def _get_graph_dict(self, idx: int):
        db_idx = bisect.bisect(self.id_cumulative, idx)
        if db_idx > 0:
            idx = idx - self.id_cumulative[db_idx - 1]
        atoms = self.dbs[db_idx]._get_row(self.db_ids[db_idx][idx]).toatoms()
        return preprocess_graph(atoms, self.atomic_indices, self.cutoff, True)
