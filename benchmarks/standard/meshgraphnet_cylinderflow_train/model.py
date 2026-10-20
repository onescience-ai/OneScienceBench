"""Pure-PyTorch MeshGraphNet, independent of OneScience and DGL."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


def _activation(name: str) -> type[nn.Module]:
    activations = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}; choose {tuple(activations)}") from exc


class MLP(nn.Module):
    """MLP used by the encoder, processor, and decoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least one")

        activation_cls = _activation(activation)
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current_dim, hidden_dim), activation_cls()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class ProcessorBlock(nn.Module):
    """One residual edge-update/node-update message-passing block."""

    def __init__(
        self,
        latent_dim: int,
        hidden_layers: int,
        activation: str,
        aggregation: Literal["sum", "mean"],
    ) -> None:
        super().__init__()
        if aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be 'sum' or 'mean'")
        self.aggregation = aggregation
        self.edge_mlp = MLP(
            input_dim=3 * latent_dim,
            output_dim=latent_dim,
            hidden_dim=latent_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=True,
        )
        self.node_mlp = MLP(
            input_dim=2 * latent_dim,
            output_dim=latent_dim,
            hidden_dim=latent_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=True,
        )

    def forward(
        self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        source, destination = edge_index
        edge_input = torch.cat(
            (edge_features, node_features[source], node_features[destination]), dim=-1
        )
        edge_features = edge_features + self.edge_mlp(edge_input)

        messages = node_features.new_zeros((node_features.shape[0], edge_features.shape[-1]))
        messages.index_add_(0, destination, edge_features)
        if self.aggregation == "mean":
            counts = node_features.new_zeros((node_features.shape[0], 1))
            counts.index_add_(
                0, destination, torch.ones_like(destination, dtype=node_features.dtype).unsqueeze(-1)
            )
            messages = messages / counts.clamp_min_(1.0)

        node_input = torch.cat((node_features, messages), dim=-1)
        node_features = node_features + self.node_mlp(node_input)
        return node_features, edge_features


class MeshGraphNet(nn.Module):
    """Encode-process-decode graph network from Pfaff et al. (2021)."""

    def __init__(
        self,
        node_input_dim: int = 6,
        edge_input_dim: int = 3,
        output_dim: int = 3,
        latent_dim: int = 128,
        processor_steps: int = 15,
        mlp_hidden_layers: int = 2,
        activation: str = "relu",
        aggregation: Literal["sum", "mean"] = "sum",
    ) -> None:
        super().__init__()
        if processor_steps < 1:
            raise ValueError("processor_steps must be at least one")

        self.node_encoder = MLP(
            node_input_dim,
            latent_dim,
            latent_dim,
            mlp_hidden_layers,
            activation,
            layer_norm=True,
        )
        self.edge_encoder = MLP(
            edge_input_dim,
            latent_dim,
            latent_dim,
            mlp_hidden_layers,
            activation,
            layer_norm=True,
        )
        self.processor = nn.ModuleList(
            ProcessorBlock(latent_dim, mlp_hidden_layers, activation, aggregation)
            for _ in range(processor_steps)
        )
        self.decoder = MLP(
            latent_dim,
            output_dim,
            latent_dim,
            mlp_hidden_layers,
            activation,
            layer_norm=False,
        )

    def forward(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor) -> Tensor:
        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)
        for block in self.processor:
            node_latent, edge_latent = block(node_latent, edge_latent, edge_index)
        return self.decoder(node_latent)

