"""AirfRANS surrogate networks: MTL Decoder I (surface / volume separation).

Paper: Hu et al., "Enhancing Airfoil Design Optimization Surrogate Models
Using Multi-Task Learning", Physics of Fluids 2025 (DOI 10.1063/5.0258928).

Architecture (paper Table I + official repo):
    Encoder  : MLP 7 -> 64 -> 64 -> 8        (ReLU, no batch norm)
    Backbone : MLP 8 -> 64 -> 64 -> 64 -> 8  (ReLU, batch norm)
    Decoder I:
        Surface decoder : 8 -> 64 -> 64 -> 1  (predicts p|S)
        Volume decoder  : 8 -> 64 -> 64 -> 4  (predicts u, v, p, nu_t)

The ``state_dict`` keys match the official repository (``encoder.lins.*``,
``nn.lins.*``, ``nn.norms.*``, ``decoder_surf.lins.*``, ``decoder_vol.lins.*``)
so pre-trained checkpoints load directly.
"""

from __future__ import annotations

from itertools import chain
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Point-wise MLP defined by an explicit channel list.

    Args:
        channel_list: layer widths, e.g. ``[7, 64, 64, 8]``.
        batch_norm: apply BatchNorm1d on hidden channels (before ReLU).
        dropout: dropout probability applied after ReLU (hidden layers only).
        relu_first: apply ReLU before batch norm (default False = norm then ReLU).

    ``state_dict`` keys are ``lins.{i}.weight`` / ``norms.{i}.weight``,
    matching the official checkpoint. BatchNorm uses ``track_running_stats=False``
    (per-batch statistics), which suits batch_size=1 + 32000 pts/sample.
    """

    def __init__(
        self,
        channel_list: Sequence[int],
        batch_norm: bool = False,
        dropout: float = 0.0,
        relu_first: bool = False,
        **kwargs,
    ):
        super().__init__()
        channel_list = list(channel_list)
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first
        self.batch_norm = batch_norm

        self.lins = nn.ModuleList()
        for dims in zip(channel_list[:-1], channel_list[1:]):
            self.lins.append(nn.Linear(*dims))

        self.norms = nn.ModuleList()
        for dim in zip(channel_list[1:-1]):
            self.norms.append(
                nn.BatchNorm1d(dim, track_running_stats=False)
                if batch_norm
                else nn.Identity()
            )

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.relu_first:
                x = x.relu_()
            x = norm(x)
            if not self.relu_first:
                x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.channel_list)[1:-1]})"


class AirflowNN(nn.Module):
    """Baseline single-decoder network (whole-domain prediction)."""

    def __init__(
        self,
        nb_hidden_layers: int = 3,
        size_hidden_layers: int = 64,
        bn_bool: bool = False,
        encoder: Sequence[int] = (7, 64, 64, 8),
        decoder: Sequence[int] = (8, 64, 64, 4),
    ):
        super(AirflowNN, self).__init__()
        self.nb_hidden_layers = nb_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.bn_bool = bn_bool
        self.activation = nn.ReLU()

        self.encoder = MLP(list(encoder), batch_norm=False)
        self.decoder = MLP(list(decoder), batch_norm=False)
        self.dim_enc = encoder[-1]

        self.nn = MLP(
            [self.dim_enc]
            + [self.size_hidden_layers] * self.nb_hidden_layers
            + [self.dim_enc],
            batch_norm=self.bn_bool,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        z = self.encoder(data)
        z = self.nn(z)
        return self.decoder(z)

    def shared_parameters(self) -> Iterator[torch.nn.Parameter]:
        return chain(self.encoder.parameters(), self.nn.parameters())

    def task_specific_parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.decoder.parameters()

    def last_shared_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter(())


class AirflowNNMTLDecoderI(nn.Module):
    """MTL architecture with Decoder I: separate surface / volume decoders.

    forward returns ``(out_surf, out_vol)`` where the split is driven by the
    ``surf`` boolean mask (True = aerofoil surface node).
    """

    def __init__(
        self,
        nb_hidden_layers: int,
        size_hidden_layers: int,
        bn_bool: bool,
        encoder: Sequence[int],
        decoder_surf: Sequence[int],
        decoder_vol: Sequence[int],
    ):
        super(AirflowNNMTLDecoderI, self).__init__()

        self.bn_bool = bn_bool
        self.activation = nn.ReLU()

        self.encoder = MLP(list(encoder), batch_norm=False)
        self.decoder_surf = MLP(list(decoder_surf), batch_norm=False)
        self.decoder_vol = MLP(list(decoder_vol), batch_norm=False)

        self.dim_enc = encoder[-1]

        self.nn = MLP(
            [self.dim_enc]
            + [size_hidden_layers] * nb_hidden_layers
            + [self.dim_enc],
            batch_norm=self.bn_bool,
        )

    def forward(self, data: torch.Tensor, surf: torch.Tensor):
        z = self.encoder(data)
        z = self.nn(z)
        return self.decoder_surf(z[surf]), self.decoder_vol(z[~surf])

    def shared_parameters(self) -> Iterator[torch.nn.Parameter]:
        return chain(self.encoder.parameters(), self.nn.parameters())

    def reset(self):
        self.encoder.zero_grad(set_to_none=False)
        self.nn.zero_grad(set_to_none=False)

    def task_specific_parameters(self) -> Iterator[torch.nn.Parameter]:
        return chain(
            self.decoder_surf.parameters(),
            self.decoder_vol.parameters(),
        )

    def last_shared_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter(())