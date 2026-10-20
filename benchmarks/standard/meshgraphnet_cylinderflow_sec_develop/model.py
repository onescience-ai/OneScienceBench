"""Pure-PyTorch multiscale MeshGraphNet for CylinderFlow."""

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


def aggregate_messages(
    messages: Tensor,
    destination: Tensor,
    target_count: int,
    aggregation: Literal["sum", "mean"],
) -> Tensor:
    output = messages.new_zeros((target_count, messages.shape[-1]))
    output.index_add_(0, destination, messages)
    if aggregation == "mean":
        counts = messages.new_zeros((target_count, 1))
        counts.index_add_(
            0,
            destination,
            torch.ones((destination.numel(), 1), dtype=messages.dtype, device=messages.device),
        )
        output = output / counts.clamp_min_(1.0)
    return output


class ProcessorBlock(nn.Module):
    """Residual edge update followed by residual node update."""

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
            3 * latent_dim,
            latent_dim,
            latent_dim,
            hidden_layers,
            activation,
            layer_norm=True,
        )
        self.node_mlp = MLP(
            2 * latent_dim,
            latent_dim,
            latent_dim,
            hidden_layers,
            activation,
            layer_norm=True,
        )

    def forward(
        self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        source, destination = edge_index
        edge_features = edge_features + self.edge_mlp(
            torch.cat(
                (edge_features, node_features[source], node_features[destination]), dim=-1
            )
        )
        received = aggregate_messages(
            edge_features, destination, node_features.shape[0], self.aggregation
        )
        node_features = node_features + self.node_mlp(
            torch.cat((node_features, received), dim=-1)
        )
        return node_features, edge_features


class CrossScaleTransfer(nn.Module):
    """Edge-aware residual transfer between fine and coarse node sets."""

    def __init__(
        self,
        edge_input_dim: int,
        latent_dim: int,
        hidden_layers: int,
        activation: str,
        aggregation: Literal["sum", "mean"],
    ) -> None:
        super().__init__()
        self.aggregation = aggregation
        self.edge_encoder = MLP(
            edge_input_dim,
            latent_dim,
            latent_dim,
            hidden_layers,
            activation,
            layer_norm=True,
        )
        self.message_mlp = MLP(
            2 * latent_dim,
            latent_dim,
            latent_dim,
            hidden_layers,
            activation,
            layer_norm=True,
        )
        self.fusion_mlp = MLP(
            2 * latent_dim,
            latent_dim,
            latent_dim,
            hidden_layers,
            activation,
            layer_norm=True,
        )

    def forward(
        self,
        source_nodes: Tensor,
        target_nodes: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        source, destination = edge_index
        edge_latent = self.edge_encoder(edge_features)
        messages = self.message_mlp(
            torch.cat((source_nodes[source], edge_latent), dim=-1)
        )
        received = aggregate_messages(
            messages, destination, target_nodes.shape[0], self.aggregation
        )
        return target_nodes + self.fusion_mlp(
            torch.cat((target_nodes, received), dim=-1)
        )


class MeshGraphNet(nn.Module):
    """Fine-to-coarse-to-fine MeshGraphNet with unchanged encoder/decoder roles."""

    def __init__(
        self,
        node_input_dim: int = 6,
        edge_input_dim: int = 3,
        output_dim: int = 3,
        latent_dim: int = 128,
        mlp_hidden_layers: int = 2,
        activation: str = "relu",
        aggregation: Literal["sum", "mean"] = "sum",
        fine_steps_before: int = 5,
        coarse_steps: int = 5,
        fine_steps_after: int = 5,
        coarse_ratio: float = 0.25,
        coarse_knn: int = 6,
        cross_scale_knn: int = 3,
    ) -> None:
        super().__init__()
        if min(fine_steps_before, coarse_steps, fine_steps_after) < 1:
            raise ValueError("All processor stage lengths must be at least one")
        self.multiscale_config = {
            "coarse_ratio": coarse_ratio,
            "coarse_knn": coarse_knn,
            "cross_scale_knn": cross_scale_knn,
        }
        self.node_encoder = MLP(
            node_input_dim,
            latent_dim,
            latent_dim,
            mlp_hidden_layers,
            activation,
            layer_norm=True,
        )
        self.fine_edge_encoder = MLP(
            edge_input_dim,
            latent_dim,
            latent_dim,
            mlp_hidden_layers,
            activation,
            layer_norm=True,
        )
        self.coarse_edge_encoder = MLP(
            edge_input_dim,
            latent_dim,
            latent_dim,
            mlp_hidden_layers,
            activation,
            layer_norm=True,
        )
        block_args = (latent_dim, mlp_hidden_layers, activation, aggregation)
        self.fine_processor_before = nn.ModuleList(
            ProcessorBlock(*block_args) for _ in range(fine_steps_before)
        )
        self.down_transfer = CrossScaleTransfer(
            edge_input_dim, latent_dim, mlp_hidden_layers, activation, aggregation
        )
        self.coarse_processor = nn.ModuleList(
            ProcessorBlock(*block_args) for _ in range(coarse_steps)
        )
        self.up_transfer = CrossScaleTransfer(
            edge_input_dim, latent_dim, mlp_hidden_layers, activation, aggregation
        )
        self.fine_processor_after = nn.ModuleList(
            ProcessorBlock(*block_args) for _ in range(fine_steps_after)
        )
        self.decoder = MLP(
            latent_dim,
            output_dim,
            latent_dim,
            mlp_hidden_layers,
            activation,
            layer_norm=False,
        )

    def forward(
        self,
        node_features: Tensor,
        fine_edge_features: Tensor,
        fine_edge_index: Tensor,
        coarse_node_index: Tensor,
        coarse_edge_features: Tensor,
        coarse_edge_index: Tensor,
        down_edge_features: Tensor,
        down_edge_index: Tensor,
        up_edge_features: Tensor,
        up_edge_index: Tensor,
    ) -> Tensor:
        fine_nodes = self.node_encoder(node_features)
        fine_edges = self.fine_edge_encoder(fine_edge_features)
        for block in self.fine_processor_before:
            fine_nodes, fine_edges = block(fine_nodes, fine_edges, fine_edge_index)

        coarse_nodes = fine_nodes[coarse_node_index]
        coarse_nodes = self.down_transfer(
            fine_nodes, coarse_nodes, down_edge_features, down_edge_index
        )
        coarse_edges = self.coarse_edge_encoder(coarse_edge_features)
        for block in self.coarse_processor:
            coarse_nodes, coarse_edges = block(
                coarse_nodes, coarse_edges, coarse_edge_index
            )

        fine_nodes = self.up_transfer(
            coarse_nodes, fine_nodes, up_edge_features, up_edge_index
        )
        for block in self.fine_processor_after:
            fine_nodes, fine_edges = block(fine_nodes, fine_edges, fine_edge_index)
        return self.decoder(fine_nodes)

