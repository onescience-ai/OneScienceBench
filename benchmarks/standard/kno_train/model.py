from __future__ import annotations

import torch
import torch.nn as nn


class EncoderMLP(nn.Module):
    def __init__(self, t_len: int, op_size: int):
        super().__init__()
        self.layer = nn.Linear(t_len, op_size)

    def forward(self, x):
        return self.layer(x)


class DecoderMLP(nn.Module):
    def __init__(self, t_len: int, op_size: int):
        super().__init__()
        self.layer = nn.Linear(op_size, t_len)

    def forward(self, x):
        return self.layer(x)


class EncoderConv1D(nn.Module):
    def __init__(self, t_len: int, op_size: int):
        super().__init__()
        self.layer = nn.Conv1d(t_len, op_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer(x)
        return x.permute(0, 2, 1)


class DecoderConv1D(nn.Module):
    def __init__(self, t_len: int, op_size: int):
        super().__init__()
        self.layer = nn.Conv1d(op_size, t_len, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer(x)
        return x.permute(0, 2, 1)


class EncoderConv2D(nn.Module):
    def __init__(self, t_len: int, op_size: int):
        super().__init__()
        self.layer = nn.Conv2d(t_len, op_size, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.layer(x)
        return x.permute(0, 2, 3, 1)


class DecoderConv2D(nn.Module):
    def __init__(self, t_len: int, op_size: int):
        super().__init__()
        self.layer = nn.Conv2d(op_size, t_len, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.layer(x)
        return x.permute(0, 2, 3, 1)


class KoopmanOperator1D(nn.Module):
    def __init__(self, op_size: int, modes_x: int = 16):
        super().__init__()
        self.op_size = int(op_size)
        self.modes_x = int(modes_x)
        if min(self.op_size, self.modes_x) < 1:
            raise ValueError("op_size and modes_x must be positive")
        scale = 1.0 / (self.op_size * self.op_size)
        self.koopman_matrix = nn.Parameter(
            scale * torch.rand(self.op_size, self.op_size, self.modes_x, dtype=torch.cfloat)
        )

    def time_marching(self, input_tensor, weights):
        return torch.einsum("btx,tfx->bfx", input_tensor, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device=x.device)
        modes_x = min(self.modes_x, x_ft.shape[-1])
        out_ft[:, :, :modes_x] = self.time_marching(
            x_ft[:, :, :modes_x],
            self.koopman_matrix[:, :, :modes_x],
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class KNO1D(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        op_size: int,
        modes_x: int = 16,
        decompose: int = 4,
        linear_type: bool = True,
        normalization: bool = False,
    ):
        super().__init__()
        self.op_size = int(op_size)
        self.decompose = int(decompose)
        if self.decompose < 1:
            raise ValueError("decompose must be positive")
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = KoopmanOperator1D(self.op_size, modes_x=modes_x)
        self.w0 = nn.Conv1d(self.op_size, self.op_size, 1)
        self.linear_type = bool(linear_type)
        self.normalization = bool(normalization)
        if self.normalization:
            self.norm_layer = nn.BatchNorm1d(self.op_size)

    def forward(self, x):
        x_reconstruct = self.dec(torch.tanh(self.enc(x)))
        x = torch.tanh(self.enc(x)).permute(0, 2, 1)
        x_w = x
        for _ in range(self.decompose):
            x1 = self.koopman_layer(x)
            x = x + x1 if self.linear_type else torch.tanh(x + x1)
        shortcut = self.w0(x_w)
        if self.normalization:
            shortcut = self.norm_layer(shortcut)
        x = torch.tanh(shortcut + x).permute(0, 2, 1)
        return self.dec(x), x_reconstruct


class KoopmanOperator2D(nn.Module):
    def __init__(self, op_size: int, modes_x: int = 12, modes_y: int = 12):
        super().__init__()
        self.op_size = int(op_size)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        if min(self.op_size, self.modes_x, self.modes_y) < 1:
            raise ValueError("op_size, modes_x and modes_y must be positive")
        scale = 1.0 / (self.op_size * self.op_size)
        self.koopman_matrix = nn.Parameter(
            scale * torch.rand(
                self.op_size,
                self.op_size,
                self.modes_x,
                self.modes_y,
                dtype=torch.cfloat,
            )
        )

    def time_marching(self, input_tensor, weights):
        return torch.einsum("btxy,tfxy->bfxy", input_tensor, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device=x.device)
        modes_x = min(self.modes_x, max(1, x_ft.shape[-2] // 2))
        modes_y = min(self.modes_y, x_ft.shape[-1])
        weights = self.koopman_matrix[:, :, :modes_x, :modes_y]
        out_ft[:, :, :modes_x, :modes_y] = self.time_marching(
            x_ft[:, :, :modes_x, :modes_y],
            weights,
        )
        out_ft[:, :, -modes_x:, :modes_y] = self.time_marching(
            x_ft[:, :, -modes_x:, :modes_y],
            weights,
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class KNO2D(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        op_size: int,
        modes_x: int = 12,
        modes_y: int = 12,
        decompose: int = 6,
        linear_type: bool = True,
        normalization: bool = False,
    ):
        super().__init__()
        self.op_size = int(op_size)
        self.decompose = int(decompose)
        if self.decompose < 1:
            raise ValueError("decompose must be positive")
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = KoopmanOperator2D(self.op_size, modes_x=modes_x, modes_y=modes_y)
        self.w0 = nn.Conv2d(self.op_size, self.op_size, 1)
        self.linear_type = bool(linear_type)
        self.normalization = bool(normalization)
        if self.normalization:
            self.norm_layer = nn.BatchNorm2d(self.op_size)

    def forward(self, x):
        x_reconstruct = self.dec(torch.tanh(self.enc(x)))
        x = torch.tanh(self.enc(x)).permute(0, 3, 1, 2)
        x_w = x
        for _ in range(self.decompose):
            x1 = self.koopman_layer(x)
            x = x + x1 if self.linear_type else torch.tanh(x + x1)
        shortcut = self.w0(x_w)
        if self.normalization:
            shortcut = self.norm_layer(shortcut)
        x = torch.tanh(shortcut + x).permute(0, 2, 3, 1)
        return self.dec(x), x_reconstruct


class KNO2DNavierStokes(nn.Module):
    """KNO2D adapter for flattened Navier-Stokes batches.

    ``pos`` is accepted for API parity with FNO but is not used by the Koopman
    operator. ``fx`` is shaped ``(B, H*W, t_in*out_dim)`` and the output is
    shaped ``(B, H*W, out_dim)`` for one-step autoregressive rollout.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        spatial_shape,
        op_size: int = 32,
        modes_x: int = 12,
        modes_y: int = 12,
        decompose: int = 6,
        linear_type: bool = True,
        normalization: bool = False,
    ):
        super().__init__()
        if int(output_channels) != 1:
            raise ValueError("KNO2DNavierStokes currently expects output_channels=1.")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        if len(self.spatial_shape) != 2 or min(self.spatial_shape) < 1:
            raise ValueError("spatial_shape must contain two positive dimensions")
        if self.input_channels < 1:
            raise ValueError("input_channels must be positive")
        encoder = EncoderConv2D(self.input_channels, int(op_size))
        decoder = DecoderConv2D(self.output_channels, int(op_size))
        self.kno = KNO2D(
            encoder=encoder,
            decoder=decoder,
            op_size=int(op_size),
            modes_x=int(modes_x),
            modes_y=int(modes_y),
            decompose=int(decompose),
            linear_type=bool(linear_type),
            normalization=bool(normalization),
        )

    def forward(self, pos, fx):
        batch_size, num_points, channels = fx.shape
        height, width = self.spatial_shape
        if pos.shape[:2] != (batch_size, num_points) or pos.shape[-1] != 2:
            raise ValueError(
                f"Expected pos [B, {num_points}, 2], got {tuple(pos.shape)}"
            )
        if num_points != height * width:
            raise ValueError(f"Expected {height * width} points, got {num_points}.")
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}.")
        x = fx.reshape(batch_size, height, width, channels)
        pred, _ = self.kno(x)
        return pred.reshape(batch_size, num_points, self.output_channels)
