"""ASE-to-MatterSim graph conversion without OneScience or PyG."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ase import Atoms
from pymatgen.optimization.neighbors import find_points_in_spheres


def _neighbors(
    atoms: Atoms, cutoff: float, numerical_tol: float = 1.0e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.ascontiguousarray(atoms.positions, dtype=float)
    cell = np.ascontiguousarray(atoms.cell.array, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=np.int64)
    senders, receivers, images, distances = find_points_in_spheres(
        positions,
        positions,
        r=float(cutoff),
        pbc=pbc,
        lattice=cell,
        tol=float(numerical_tol),
    )
    senders = senders.astype(np.int64)
    receivers = receivers.astype(np.int64)
    images = images.astype(np.int64)
    distances = distances.astype(float)
    keep = (senders != receivers) | (distances > numerical_tol)
    return senders[keep], receivers[keep], images[keep], distances[keep]


def _three_body_indices(
    edge_index: np.ndarray,
    distances: np.ndarray,
    num_atoms: int,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_edge_ids = np.flatnonzero(distances <= float(cutoff))
    outgoing: list[list[int]] = [[] for _ in range(num_atoms)]
    for edge_id in valid_edge_ids.tolist():
        outgoing[int(edge_index[0, edge_id])].append(edge_id)

    triples: list[tuple[int, int]] = []
    counts = np.zeros(edge_index.shape[1], dtype=np.int64)
    for edge_ids in outgoing:
        for first in edge_ids:
            counts[first] = len(edge_ids) - 1
            for second in edge_ids:
                if first != second:
                    triples.append((first, second))
    indices = np.asarray(triples, dtype=np.int64).reshape(-1, 2)
    return indices, counts


def _normalized_atoms(source: Atoms, cutoff: float, threebody_cutoff: float) -> Atoms:
    if not isinstance(source, Atoms):
        raise TypeError("each structure must be an ase.Atoms instance")
    if len(source) == 0:
        raise ValueError("empty structures are not supported")
    atoms = source.copy()
    if not np.isfinite(atoms.positions).all():
        raise ValueError("positions must be finite")
    if np.any(atoms.numbers < 1) or np.any(atoms.numbers > 94):
        raise ValueError("atomic numbers must be in [1, 94]")
    if not np.any(atoms.pbc):
        span = np.ptp(atoms.positions, axis=0)
        side = float(max(span.max() + 5.0 * max(cutoff, threebody_cutoff), 1.0))
        atoms.set_cell(np.eye(3) * side)
        atoms.set_pbc(True)
    elif abs(float(np.linalg.det(atoms.cell.array))) < 1.0e-8:
        raise ValueError("periodic structures require a non-singular cell")
    atoms.set_scaled_positions(np.mod(atoms.get_scaled_positions(), 1.0))
    return atoms


def atoms_to_input(
    source: Atoms,
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    atoms = _normalized_atoms(source, cutoff, threebody_cutoff)
    senders, receivers, images, distances = _neighbors(atoms, cutoff)
    edge_index_np = np.stack((senders, receivers), axis=0)
    triples_np, triple_counts_np = _three_body_indices(
        edge_index_np, distances, len(atoms), threebody_cutoff
    )

    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)
    three_body_indices = torch.as_tensor(
        triples_np, dtype=torch.long, device=device
    )
    triple_counts = torch.as_tensor(
        triple_counts_np, dtype=torch.long, device=device
    ).unsqueeze(-1)
    num_atoms = torch.tensor([len(atoms)], dtype=torch.long, device=device)
    num_bonds = torch.tensor([edge_index.shape[1]], dtype=torch.long, device=device)
    num_three_body = torch.tensor(
        [three_body_indices.shape[0]], dtype=torch.long, device=device
    )
    edge_map = torch.repeat_interleave(
        torch.arange(edge_index.shape[1], device=device), triple_counts.reshape(-1)
    )

    return {
        "atom_pos": torch.as_tensor(atoms.positions, dtype=torch.float32, device=device),
        "cell": torch.as_tensor(
            atoms.cell.array[None], dtype=torch.float32, device=device
        ),
        "pbc_offsets": torch.as_tensor(images, dtype=torch.float32, device=device),
        "atom_attr": torch.as_tensor(
            atoms.numbers[:, None], dtype=torch.float32, device=device
        ),
        "edge_index": edge_index,
        "three_body_indices": three_body_indices,
        "num_three_body": num_three_body,
        "num_bonds": num_bonds,
        "num_triple_ij": triple_counts,
        "num_atoms": num_atoms,
        "num_graphs": torch.tensor(1, dtype=torch.long, device=device),
        "batch": torch.zeros(len(atoms), dtype=torch.long, device=device),
        "total_num_atoms": len(atoms),
        "total_num_bonds": edge_index.shape[1],
        "bond_index_bias": torch.zeros(
            (three_body_indices.shape[0], 1), dtype=torch.long, device=device
        ),
        "three_body_edge_map": edge_map,
    }


__all__ = ["atoms_to_input"]
