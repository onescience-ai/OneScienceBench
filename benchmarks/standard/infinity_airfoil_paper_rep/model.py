"""Core INFINITY components for the 2-D AirfRANS airfoil experiment.

Paper-authoritative choices:
- one modulated INR per modality: d, n, vx, vy, p, nu_t;
- Fourier-feature INR backbone;
- additive (shift) modulation;
- a linear latent-to-modulation hypernetwork;
- CAVIA-style auto-decoding with three inner steps (configured externally);
- an MLP mapping [z_d, z_n, Vx, Vy] to output-field latent codes.

Architecture sizes are not reported in the paper and therefore come from the
public author repository. This file is an independent, flattened rewrite rather
than a copy of the repository implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

GEOMETRY_FIELDS = ("sdf", "n")
PHYSICS_FIELDS = ("vx", "vy", "p", "nu")
ALL_FIELDS = GEOMETRY_FIELDS + PHYSICS_FIELDS


class FourierEncoding(nn.Module):
    """NeRF-style log-spaced Fourier encoding used by the official code."""

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int,
        max_frequency: float,
        base_frequency: float,
        include_input: bool,
    ) -> None:
        super().__init__()
        bands = base_frequency ** torch.linspace(0.0, max_frequency, num_frequencies)
        self.register_buffer("bands", bands, persistent=True)
        self.include_input = include_input
        self.input_dim = input_dim
        self.out_dim = 2 * input_dim * num_frequencies + (input_dim if include_input else 0)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [N,{self.input_dim}] coordinates, got {tuple(x.shape)}")
        phase = (x[:, None, :] * self.bands[None, :, None]).reshape(x.shape[0], -1)
        encoded = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return torch.cat((x, encoded), dim=-1) if self.include_input else encoded


class ShiftModulatedINR(nn.Module):
    """Fourier MLP with per-hidden-layer additive modulation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        num_frequencies: int,
        max_frequency: float,
        base_frequency: float,
        include_input: bool,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must include at least one hidden layer and one output layer")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.encoding = FourierEncoding(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
            base_frequency=base_frequency,
            include_input=include_input,
        )
        dims_in = [self.encoding.out_dim] + [hidden_dim] * (depth - 1)
        dims_out = [hidden_dim] * (depth - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims_in, dims_out))

        # The paper explicitly states that h(z) is linear.
        self.hypernetwork = nn.Linear(latent_dim, hidden_dim * (depth - 1))

    def forward(self, coordinates: Tensor, code: Tensor) -> Tensor:
        if code.ndim == 2:
            if code.shape[0] != 1:
                raise ValueError("This flattened implementation adapts one airfoil case at a time.")
            code = code[0]
        if code.ndim != 1 or code.shape[0] != self.latent_dim:
            raise ValueError(f"Expected code [{self.latent_dim}], got {tuple(code.shape)}")

        shifts = self.hypernetwork(code).view(self.depth - 1, self.hidden_dim)
        h = self.encoding(coordinates)
        for layer_id, layer in enumerate(self.layers[:-1]):
            h = F.relu(layer(h) + shifts[layer_id])
        return self.layers[-1](h)


class MetaINR(nn.Module):
    """An INR plus the learnable CAVIA inner-loop step size alpha."""

    def __init__(self, inr: ShiftModulatedINR, inner_lr: float = 1e-2) -> None:
        super().__init__()
        self.inr = inr
        self.inner_lr = nn.Parameter(torch.tensor(float(inner_lr)))

    @property
    def latent_dim(self) -> int:
        return self.inr.latent_dim

    def forward(self, coordinates: Tensor, code: Tensor) -> Tensor:
        return self.inr(coordinates, code)


class Swish(nn.Module):
    """Swish variant used by the author's public residual MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x * F.softplus(self.beta)) / 1.1


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.act1 = Swish()
        self.act2 = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.fc2(self.act1(self.fc1(x)))
        return x + self.act2(self.dropout(residual))


class CodeProcessor(nn.Module):
    """g_psi: [z_d, z_n, Vx, Vy] -> [z_vx, z_vy, z_p, z_nu]."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = 2 * latent_dim + 2
        output_dim = 4 * latent_dim
        self.latent_dim = latent_dim
        self.blocks = nn.Sequential(
            *(ResidualBlock(input_dim, hidden_dim, dropout) for _ in range(depth))
        )
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.blocks(x))

    def split_output(self, output: Tensor) -> dict[str, Tensor]:
        chunks = output.split(self.latent_dim, dim=-1)
        return dict(zip(PHYSICS_FIELDS, chunks))


@dataclass(frozen=True)
class INRBuildSpec:
    hidden_dim: int
    depth: int
    num_frequencies: int
    max_frequency: float
    base_frequency: float
    include_input: bool


def _physics_uses_sdf(config: Mapping) -> bool:
    return bool(config["model"].get("physics_decoder_condition_on_sdf", False))


def build_inr(field: str, config: Mapping) -> MetaINR:
    """Build one field-specific modulated INR from a loaded YAML dictionary."""
    if field not in ALL_FIELDS:
        raise KeyError(f"Unknown field {field!r}; expected one of {ALL_FIELDS}")

    inr_cfg = config["model"]["inr"][field]
    latent_dim = int(config["model"]["latent_dim"])
    input_dim = 2
    if field in PHYSICS_FIELDS and _physics_uses_sdf(config):
        input_dim = 3
    output_dim = 2 if field == "n" else 1

    inr = ShiftModulatedINR(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        hidden_dim=int(inr_cfg["hidden_dim"]),
        depth=int(inr_cfg["depth"]),
        num_frequencies=int(inr_cfg["num_frequencies"]),
        max_frequency=float(inr_cfg["max_frequency"]),
        base_frequency=float(inr_cfg["base_frequency"]),
        include_input=bool(inr_cfg["include_input"]),
    )
    inner_lr = float(config["optimization"]["inr"][field]["inner_lr"])
    return MetaINR(inr, inner_lr=inner_lr)


def build_processor(config: Mapping) -> CodeProcessor:
    cfg = config["model"]["processor"]
    return CodeProcessor(
        latent_dim=int(config["model"]["latent_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )


def adapt_code(
    model: MetaINR,
    coordinates: Tensor,
    target: Tensor,
    steps: int,
    *,
    create_graph: bool,
) -> Tensor:
    """CAVIA auto-decoding from a zero code using differentiable inner steps."""
    code = torch.zeros(model.latent_dim, device=coordinates.device, dtype=coordinates.dtype)
    code.requires_grad_(True)
    for _ in range(steps):
        reconstruction = model(coordinates, code)
        loss = F.mse_loss(reconstruction, target)
        (grad,) = torch.autograd.grad(loss, code, create_graph=create_graph)
        code = code - model.inner_lr * grad
    return code
