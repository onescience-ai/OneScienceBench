"""Multiscale Koopman Neural Operator for 2-D Navier--Stokes dynamics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class EncoderConv2D(nn.Module):
    """Lift channels-last history frames into Koopman latent channels."""

    def __init__(self, input_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.layer = nn.Conv2d(input_channels, latent_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layer(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class DecoderConv2D(nn.Module):
    """Project Koopman latent channels to a one-step physical field."""

    def __init__(self, output_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.layer = nn.Conv2d(latent_channels, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layer(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MultiScaleKoopmanOperator2D(nn.Module):
    """Parallel low/high Fourier Koopman branches with gated residual fusion.

    ``koopman_matrix`` deliberately retains the legacy parameter name and shape,
    so a single-scale checkpoint initializes the low-frequency branch exactly.
    The high branch models the rectangular frequency band outside the low modes.
    """

    def __init__(
        self,
        op_size: int,
        low_modes_x: int = 12,
        low_modes_y: int = 12,
        high_modes_x: int = 24,
        high_modes_y: int = 24,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        values = (op_size, low_modes_x, low_modes_y, high_modes_x, high_modes_y)
        if min(values) < 1:
            raise ValueError("Operator size and Fourier modes must be positive")
        if high_modes_x < low_modes_x or high_modes_y < low_modes_y:
            raise ValueError("High-frequency limits must cover the low-frequency limits")
        self.op_size = int(op_size)
        self.low_modes_x = int(low_modes_x)
        self.low_modes_y = int(low_modes_y)
        self.high_modes_x = int(high_modes_x)
        self.high_modes_y = int(high_modes_y)
        scale = 1.0 / (self.op_size * self.op_size)
        self.koopman_matrix = nn.Parameter(
            scale
            * torch.rand(
                self.op_size,
                self.op_size,
                self.low_modes_x,
                self.low_modes_y,
                dtype=torch.cfloat,
            )
        )
        self.high_koopman_matrix = nn.Parameter(
            torch.zeros(
                self.op_size,
                self.op_size,
                self.high_modes_x,
                self.high_modes_y,
                dtype=torch.cfloat,
            )
        )
        self.high_gate_logits = nn.Parameter(
            torch.full((self.op_size,), float(gate_init), dtype=torch.float32)
        )

    @staticmethod
    def _time_marching(inputs: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("btxy,tfxy->bfxy", inputs, weights)

    def forward(self, inputs: Tensor) -> Tensor:
        spectrum = torch.fft.rfft2(inputs)
        low_spectrum = torch.zeros_like(spectrum)
        high_spectrum = torch.zeros_like(spectrum)
        max_x = max(1, spectrum.shape[-2] // 2)
        low_x = min(self.low_modes_x, max_x)
        low_y = min(self.low_modes_y, spectrum.shape[-1])
        high_x = min(self.high_modes_x, max_x)
        high_y = min(self.high_modes_y, spectrum.shape[-1])

        low_weights = self.koopman_matrix[:, :, :low_x, :low_y]
        low_spectrum[:, :, :low_x, :low_y] = self._time_marching(
            spectrum[:, :, :low_x, :low_y], low_weights
        )
        low_spectrum[:, :, -low_x:, :low_y] = self._time_marching(
            spectrum[:, :, -low_x:, :low_y], low_weights
        )

        high_weights = self.high_koopman_matrix[:, :, :high_x, :high_y]
        positive_band = self._time_marching(
            spectrum[:, :, :high_x, :high_y], high_weights
        )
        negative_band = self._time_marching(
            spectrum[:, :, -high_x:, :high_y], high_weights
        )
        positive_mask = positive_band.new_ones((high_x, high_y))
        negative_mask = negative_band.new_ones((high_x, high_y))
        positive_mask[:low_x, :low_y] = 0
        negative_mask[-low_x:, :low_y] = 0
        high_spectrum[:, :, :high_x, :high_y] = positive_band * positive_mask
        high_spectrum[:, :, -high_x:, :high_y] = negative_band * negative_mask

        gate = torch.sigmoid(self.high_gate_logits).to(spectrum.dtype).view(1, -1, 1, 1)
        fused_spectrum = low_spectrum + gate * high_spectrum
        return torch.fft.irfft2(fused_spectrum, s=inputs.shape[-2:])


class KNO2D(nn.Module):
    """Encode--evolve--decode KNO with the original shortcut path."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        op_size: int,
        low_modes_x: int = 12,
        low_modes_y: int = 12,
        high_modes_x: int = 24,
        high_modes_y: int = 24,
        decompose: int = 6,
        linear_type: bool = True,
        normalization: bool = False,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if decompose < 1:
            raise ValueError("decompose must be positive")
        self.decompose = int(decompose)
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = MultiScaleKoopmanOperator2D(
            op_size,
            low_modes_x,
            low_modes_y,
            high_modes_x,
            high_modes_y,
            gate_init,
        )
        self.w0 = nn.Conv2d(op_size, op_size, kernel_size=1)
        self.linear_type = bool(linear_type)
        self.normalization = bool(normalization)
        if self.normalization:
            self.norm_layer = nn.BatchNorm2d(op_size)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        encoded = torch.tanh(self.enc(inputs))
        reconstruction = self.dec(encoded)
        latent = encoded.permute(0, 3, 1, 2)
        shortcut_source = latent
        for _ in range(self.decompose):
            update = self.koopman_layer(latent)
            latent = latent + update if self.linear_type else torch.tanh(latent + update)
        shortcut = self.w0(shortcut_source)
        if self.normalization:
            shortcut = self.norm_layer(shortcut)
        decoded = torch.tanh(shortcut + latent).permute(0, 2, 3, 1)
        return self.dec(decoded), reconstruction


class KNO2DNavierStokes(nn.Module):
    """Flattened-grid adapter used by the training and rollout pipeline."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        spatial_shape: tuple[int, int],
        op_size: int = 32,
        low_modes_x: int = 12,
        low_modes_y: int = 12,
        high_modes_x: int = 24,
        high_modes_y: int = 24,
        decompose: int = 6,
        linear_type: bool = True,
        normalization: bool = False,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if output_channels != 1:
            raise ValueError("KNO2DNavierStokes expects output_channels=1")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.spatial_shape = tuple(int(value) for value in spatial_shape)
        if len(self.spatial_shape) != 2 or min(self.spatial_shape) < 1:
            raise ValueError("spatial_shape must contain two positive dimensions")
        self.kno = KNO2D(
            EncoderConv2D(self.input_channels, op_size),
            DecoderConv2D(self.output_channels, op_size),
            op_size,
            low_modes_x,
            low_modes_y,
            high_modes_x,
            high_modes_y,
            decompose,
            linear_type,
            normalization,
            gate_init,
        )

    def forward(self, pos: Tensor, fields: Tensor) -> Tensor:
        batch_size, point_count, channels = fields.shape
        height, width = self.spatial_shape
        if pos.shape != (batch_size, point_count, 2):
            raise ValueError(f"Expected pos [B,N,2], got {tuple(pos.shape)}")
        if point_count != height * width or channels != self.input_channels:
            raise ValueError(
                f"Expected fields [B,{height * width},{self.input_channels}], "
                f"got {tuple(fields.shape)}"
            )
        prediction, _ = self.kno(fields.reshape(batch_size, height, width, channels))
        return prediction.reshape(batch_size, point_count, self.output_channels)


def build_model(model_config: Mapping[str, Any], spatial_shape: tuple[int, int]) -> KNO2DNavierStokes:
    """Build the multiscale model while accepting legacy ``modes_x/y`` config."""

    low_x = int(model_config.get("low_modes_x", model_config.get("modes_x", 12)))
    low_y = int(model_config.get("low_modes_y", model_config.get("modes_y", 12)))
    return KNO2DNavierStokes(
        input_channels=int(model_config["input_channels"]),
        output_channels=int(model_config["output_channels"]),
        spatial_shape=spatial_shape,
        op_size=int(model_config["op_size"]),
        low_modes_x=low_x,
        low_modes_y=low_y,
        high_modes_x=int(model_config.get("high_modes_x", 2 * low_x)),
        high_modes_y=int(model_config.get("high_modes_y", 2 * low_y)),
        decompose=int(model_config["decompose"]),
        linear_type=bool(model_config["linear_type"]),
        normalization=bool(model_config["normalization"]),
        gate_init=float(model_config.get("gate_init", 0.0)),
    )


def checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
    """Resolve common checkpoint state keys and strip an optional DDP prefix."""

    state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint model state must be a mapping")
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_legacy_weights(model: nn.Module, checkpoint: Mapping[str, Any]) -> None:
    """Strictly migrate all legacy parameters, leaving only new high-band keys."""

    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    allowed_missing = {
        "kno.koopman_layer.high_koopman_matrix",
        "kno.koopman_layer.high_gate_logits",
    }
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Legacy checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
