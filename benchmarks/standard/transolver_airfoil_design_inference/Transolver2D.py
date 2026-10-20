"""Standalone Transolver2D implementation for AirfRANS inference.

The module intentionally keeps the parameter hierarchy used by the published
checkpoint while avoiding imports from the wider OneScience package.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from timm.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_


class StandardMLP(nn.Module):
    """Minimal MLP whose layer names match the training implementation."""

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str | Callable[..., Any] = "relu",
        output_activation: str | Callable[..., Any] | None = None,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        dropout_position: str = "after",
        norm_layer: Optional[str] = None,
        norm_position: str = "before",
        use_skip_connection: bool | str = False,
        skip_strength: float = 1.0,
        **_: Any,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if (
            input_dim <= 0
            or output_dim <= 0
            or any(dim <= 0 for dim in hidden_dims)
        ):
            raise ValueError("MLP dimensions must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")
        if dropout_position not in {"before", "after"}:
            raise ValueError("dropout_position must be 'before' or 'after'")
        if norm_position not in {"before", "after"}:
            raise ValueError("norm_position must be 'before' or 'after'")

        self.hidden_dims = hidden_dims
        self.output_dim = int(output_dim)
        self.dropout_rate = float(dropout_rate)
        self.dropout_position = dropout_position
        self.norm_position = norm_position
        self.use_skip_connection = use_skip_connection
        self.skip_strength = float(skip_strength)
        self.activation = self._make_activation(activation)
        self.output_activation = (
            self._make_activation(output_activation)
            if output_activation is not None
            else None
        )

        dimensions = (int(input_dim), *hidden_dims, self.output_dim)
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=use_bias)
            for in_features, out_features in zip(dimensions, dimensions[1:])
        )
        self.norm_layers = nn.ModuleList(
            self._make_norm(norm_layer, dim) for dim in hidden_dims
        )
        self.dropout_layers = nn.ModuleList(
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
            for _ in hidden_dims
        )

    @classmethod
    def _make_activation(
        cls, activation: str | Callable[..., Any]
    ) -> nn.Module | Callable[..., Any]:
        if isinstance(activation, str):
            try:
                activation_type = cls.ACTIVATIONS[activation]
            except KeyError as error:
                raise ValueError(f"Unsupported activation: {activation}") from error
            return activation_type()
        return activation() if isinstance(activation, type) else activation

    @staticmethod
    def _make_norm(norm_layer: Optional[str], dim: int) -> nn.Module:
        if norm_layer is None:
            return nn.Identity()
        if norm_layer == "layer_norm":
            return nn.LayerNorm(dim)
        if norm_layer == "batch_norm":
            return nn.BatchNorm1d(dim)
        raise ValueError(f"Unsupported normalization: {norm_layer}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape
        features = inputs
        if features.dim() > 2:
            features = features.reshape(-1, features.shape[-1])

        for index, hidden_dim in enumerate(self.hidden_dims):
            identity = (
                features
                if self.use_skip_connection and features.shape[-1] == hidden_dim
                else None
            )
            if self.dropout_position == "before":
                features = self.dropout_layers[index](features)
            features = self.layers[index](features)
            if self.norm_position == "before":
                features = self.norm_layers[index](features)
            features = self.activation(features)
            if self.norm_position == "after":
                features = self.norm_layers[index](features)
            if self.dropout_position == "after":
                features = self.dropout_layers[index](features)
            if identity is not None and self.use_skip_connection == "residual":
                features = identity + self.skip_strength * features

        features = self.layers[-1](features)
        if self.output_activation is not None:
            features = self.output_activation(features)
        if len(original_shape) > 2:
            features = features.reshape(*original_shape[:-1], self.output_dim)
        return features


class PhysicsAttentionIrregularMesh(nn.Module):
    """Slice-attention-deslice operator for an irregular mesh."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        **_: Any,
    ) -> None:
        super().__init__()
        if min(dim, heads, dim_head, slice_num) <= 0:
            raise ValueError("Attention dimensions must be positive")

        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = inputs.shape

        projected_values = self.in_project_fx(inputs).reshape(
            batch_size, num_points, self.heads, self.dim_head
        )
        projected_values = projected_values.permute(0, 2, 1, 3).contiguous()
        projected_positions = self.in_project_x(inputs).reshape(
            batch_size, num_points, self.heads, self.dim_head
        )
        projected_positions = projected_positions.permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(
            self.in_project_slice(projected_positions) / self.temperature
        )
        slice_norm = slice_weights.sum(dim=2)
        slice_tokens = torch.einsum(
            "bhnc,bhng->bhgc", projected_values, slice_weights
        )
        slice_tokens = slice_tokens / (slice_norm[..., None] + 1.0e-5)

        queries = self.to_q(slice_tokens)
        keys = self.to_k(slice_tokens)
        values = self.to_v(slice_tokens)
        attention = self.softmax(
            torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        )
        attention = self.dropout(attention)
        output_tokens = torch.matmul(attention, values)

        output = torch.einsum(
            "bhgc,bhng->bhnc", output_tokens, slice_weights
        )
        output = output.permute(0, 2, 1, 3).reshape(
            batch_size, num_points, -1
        )
        return self.to_out(output)


class _AttentionWrapper(nn.Module):
    """Keep the checkpoint's ``Attn.attentioner`` parameter hierarchy."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.attentioner = PhysicsAttentionIrregularMesh(**kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.attentioner(inputs)


class _MLPWrapper(nn.Module):
    """Keep the checkpoint's ``mlp.mlp`` parameter hierarchy."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.mlp = StandardMLP(**kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class TransolverBlock(nn.Module):
    """Pre-normalized physical-attention block with a residual MLP."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: float = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
        geotype: str = "unstructured",
        **_: Any,
    ) -> None:
        super().__init__()
        if geotype != "unstructured":
            raise ValueError(
                "The standalone AirfRANS model only supports unstructured meshes"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.last_layer = bool(last_layer)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = _AttentionWrapper(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = _MLPWrapper(
            input_dim=hidden_dim,
            hidden_dims=[int(hidden_dim * mlp_ratio)],
            output_dim=hidden_dim,
            activation=act,
            dropout_rate=dropout,
            use_bias=True,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.Attn(self.ln_1(features)) + features
        features = self.mlp(self.ln_2(features)) + features
        if self.last_layer:
            return self.mlp2(self.ln_3(features))
        return features


# Preserve the public name used by the original model source.
Transolver_block = TransolverBlock


class Transolver2D(nn.Module):
    """Transolver model used by the AirfRANS airfoil checkpoint."""

    def __init__(
        self,
        space_dim: int = 1,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: float = 1,
        fun_dim: int = 1,
        out_dim: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = False,
    ) -> None:
        super().__init__()
        if min(n_layers, n_hidden, n_head, out_dim, slice_num, ref) <= 0:
            raise ValueError("Model dimensions must be positive")
        if space_dim < 0 or fun_dim < 0:
            raise ValueError("space_dim and fun_dim must be non-negative")
        if n_hidden % n_head != 0:
            raise ValueError("n_hidden must be divisible by n_head")

        self.__name__ = "Transolver"
        self.ref = int(ref)
        self.unified_pos = bool(unified_pos)
        self.n_hidden = int(n_hidden)
        self.space_dim = int(space_dim)

        input_dim = fun_dim + space_dim
        if self.unified_pos:
            input_dim += self.ref * self.ref
        self.preprocess = StandardMLP(
            input_dim=input_dim,
            hidden_dims=[n_hidden * 2],
            output_dim=n_hidden,
            activation=act,
            use_bias=True,
            use_skip_connection=False,
        )
        self.blocks = nn.ModuleList(
            TransolverBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                last_layer=layer_index == n_layers - 1,
                geotype="unstructured",
            )
            for layer_index in range(n_layers)
        )

        self.initialize_weights()
        self.placeholder = nn.Parameter(
            torch.rand(n_hidden, dtype=torch.float) / n_hidden
        )

    def initialize_weights(self) -> None:
        """Apply the initialization scheme used during training."""
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def get_grid(self, positions: torch.Tensor) -> torch.Tensor:
        """Return distances from mesh points to the fixed reference grid."""
        if positions.shape[-1] != 2:
            raise ValueError("AirfRANS positions must have two coordinates")

        batch_size = positions.shape[0]
        grid_x = torch.tensor(np.linspace(-2, 4, self.ref), dtype=torch.float)
        grid_x = grid_x.reshape(1, self.ref, 1, 1).repeat(
            batch_size, 1, self.ref, 1
        )
        grid_y = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        grid_y = grid_y.reshape(1, 1, self.ref, 1).repeat(
            batch_size, self.ref, 1, 1
        )
        reference_grid = torch.cat((grid_x, grid_y), dim=-1)
        reference_grid = reference_grid.to(positions.device).reshape(
            batch_size, self.ref**2, 2
        )

        distances = torch.sqrt(
            torch.sum(
                (positions[:, :, None, :] - reference_grid[:, None, :, :]) ** 2,
                dim=-1,
            )
        )
        return distances.reshape(
            batch_size, positions.shape[1], self.ref * self.ref
        ).contiguous()

    def forward(self, data: Any) -> torch.Tensor:
        features = data.x[None, :, :]
        if self.unified_pos:
            grid_features = self.get_grid(data.pos[None, :, :])
            features = torch.cat((features, grid_features), dim=-1)

        features = self.preprocess(features)
        features = features + self.placeholder[None, None, :]
        for block in self.blocks:
            features = block(features)
        return features[0]


__all__ = [
    "PhysicsAttentionIrregularMesh",
    "StandardMLP",
    "Transolver2D",
    "TransolverBlock",
    "Transolver_block",
]
