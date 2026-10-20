"""Standalone MatterGen GemNet-T denoiser."""

from __future__ import annotations

import math

import torch
from torch import nn

from .batch import CrystalBatch
from .utils.globals import MAX_ATOMIC_NUM


class NoiseLevelEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        encoded = torch.zeros(
            (timesteps.shape[0], self.d_model),
            device=self.div_term.device,
            dtype=self.div_term.dtype,
        )
        encoded[:, 0::2] = torch.sin(timesteps[:, None] * self.div_term[None])
        encoded[:, 1::2] = torch.cos(timesteps[:, None] * self.div_term[None])
        return self.dropout(encoded)


class GemNetTDenoiser(nn.Module):
    """Predict coordinate, lattice, and atom-type diffusion updates."""

    def __init__(
        self,
        gemnet: nn.Module,
        hidden_dim: int = 512,
        denoise_atom_types: bool = True,
        atom_type_diffusion: str = "mask",
    ) -> None:
        super().__init__()
        self.gemnet = gemnet
        self.noise_level_encoding = NoiseLevelEncoding(hidden_dim)
        self.hidden_dim = hidden_dim
        self.denoise_atom_types = denoise_atom_types
        self.atom_type_diffusion = atom_type_diffusion
        self.property_embeddings = nn.ModuleDict()
        with_mask_type = denoise_atom_types and "mask" in atom_type_diffusion
        self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM + int(with_mask_type))

    def forward(self, batch: CrystalBatch, timesteps: torch.Tensor) -> CrystalBatch:
        frac_coords = batch["pos"]
        lattice = batch["cell"]
        atom_types = batch["atomic_numbers"]
        num_atoms = batch["num_atoms"]
        batch_index = batch.get_batch_idx("pos")

        latent = self.noise_level_encoding(timesteps).to(lattice.device)
        output = self.gemnet(
            z=latent,
            frac_coords=frac_coords,
            atom_types=atom_types,
            num_atoms=num_atoms,
            batch=batch_index,
            lengths=None,
            angles=None,
            lattice=lattice,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
        )
        atom_logits = self.fc_atom(output.node_embeddings)
        fractional_position_update = (
            lattice.inverse().transpose(1, 2)[batch_index]
            @ output.forces.unsqueeze(-1)
        ).squeeze(-1)
        return batch.replace(
            pos=fractional_position_update,
            cell=output.stress,
            atomic_numbers=atom_logits,
        )


__all__ = ["GemNetTDenoiser", "NoiseLevelEncoding"]
