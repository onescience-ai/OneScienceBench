"""Property-conditioned MatterGen adapter modules."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .batch import CrystalBatch
from .denoiser import GemNetTDenoiser, NoiseLevelEncoding


class ScalarPropertyEmbedding(nn.Module):
    """Encode one scalar crystal property with MatterGen's sinusoidal encoder."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.conditional_embedding_module = NoiseLevelEncoding(self.hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        use_unconditional: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flattened = values.reshape(-1)
        encoded = self.conditional_embedding_module(flattened)
        if use_unconditional is None:
            return encoded
        mask = use_unconditional.reshape(-1, 1).to(
            device=encoded.device, dtype=torch.bool
        )
        return torch.where(mask, torch.zeros_like(encoded), encoded)


class GemNetTAdapterMixin:
    """Layerwise condition injection matching MatterGen's GemNetT adapter."""

    condition_on_adapt: tuple[str, ...]
    cond_adapt_layers: nn.ModuleDict
    cond_mixin_layers: nn.ModuleDict

    def initialize_condition_adapter(
        self,
        condition_on_adapt: tuple[str, ...],
        hidden_dim: int,
        num_blocks: int,
    ) -> None:
        self.condition_on_adapt = tuple(condition_on_adapt)
        self.cond_adapt_layers = nn.ModuleDict()
        self.cond_mixin_layers = nn.ModuleDict()
        for condition in self.condition_on_adapt:
            self.cond_adapt_layers[condition] = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(num_blocks)
                ]
            )
            mixin_layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_blocks)]
            )
            for layer in mixin_layers:
                nn.init.zeros_(layer.weight)
            self.cond_mixin_layers[condition] = mixin_layers

    def apply_condition_adapter(
        self,
        h: torch.Tensor,
        block_index: int,
        batch: torch.Tensor,
        cond_adapt: Mapping[str, torch.Tensor] | None,
        cond_adapt_mask: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if cond_adapt is None or cond_adapt_mask is None:
            raise ValueError("property-conditioned GemNet requires adapter inputs")
        adapted = torch.zeros_like(h)
        for condition in self.condition_on_adapt:
            if condition not in cond_adapt or condition not in cond_adapt_mask:
                raise KeyError(f"missing adapter condition: {condition}")
            per_atom = cond_adapt[condition][batch]
            use_conditional = 1.0 - cond_adapt_mask[condition][batch].to(h.dtype)
            hidden = self.cond_adapt_layers[condition][block_index](
                torch.cat((h, per_atom), dim=-1)
            )
            mixed = self.cond_mixin_layers[condition][block_index](hidden)
            adapted = adapted + use_conditional.reshape(-1, 1) * mixed
        return h + adapted


class MatterGenPropertyAdapter(GemNetTDenoiser):
    """GemNet-T denoiser conditioned on one or more scalar properties."""

    def __init__(
        self,
        gemnet: nn.Module,
        property_name: str,
        hidden_dim: int = 512,
        denoise_atom_types: bool = True,
        atom_type_diffusion: str = "mask",
    ) -> None:
        super().__init__(
            gemnet=gemnet,
            hidden_dim=hidden_dim,
            denoise_atom_types=denoise_atom_types,
            atom_type_diffusion=atom_type_diffusion,
        )
        self.property_name = str(property_name)
        self.property_embeddings_adapt = nn.ModuleDict(
            {self.property_name: ScalarPropertyEmbedding(hidden_dim)}
        )

    def forward(self, batch: CrystalBatch, timesteps: torch.Tensor) -> CrystalBatch:
        property_values = batch[self.property_name]
        if property_values.shape[0] != batch["num_atoms"].numel():
            raise ValueError("property values must contain one scalar per crystal")
        unconditional_key = f"use_unconditional_{self.property_name}"
        use_unconditional = batch.get(
            unconditional_key,
            torch.zeros(
                property_values.shape[0],
                1,
                dtype=torch.bool,
                device=property_values.device,
            ),
        )
        condition = self.property_embeddings_adapt[self.property_name](
            property_values, use_unconditional
        )

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
            cond_adapt={self.property_name: condition},
            cond_adapt_mask={self.property_name: use_unconditional},
        )
        atom_logits = self.fc_atom(output.node_embeddings)
        position_update = (
            lattice.inverse().transpose(1, 2)[batch_index]
            @ output.forces.unsqueeze(-1)
        ).squeeze(-1)
        return CrystalBatch(
            pos=position_update,
            cell=output.stress,
            atomic_numbers=atom_logits,
        )


__all__ = [
    "GemNetTAdapterMixin",
    "MatterGenPropertyAdapter",
    "ScalarPropertyEmbedding",
]
