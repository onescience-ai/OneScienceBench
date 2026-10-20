"""G-FNO2d-p4 model used for the Navier--Stokes experiment.

The implementation is a focused reorganization of the official
AIRS/OpenPDE/G-FNO GFNO2d-p4 path. It keeps the paper's four G-Fourier
layers and p4/C4 rotational group structure, while removing unrelated
p4m, 3D, SWE, steerable and baseline model code.

Core model class is intentionally named ``xx`` as requested.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricGrid2D(nn.Module):
    """Rotation-symmetric radial coordinate appended to the input field."""

    grid_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, X, Y, C]
        batch, size_x, size_y = x.shape[:3]
        gridx = torch.linspace(0, 1, size_x, device=x.device, dtype=x.dtype)
        gridy = torch.linspace(0, 1, size_y, device=x.device, dtype=x.dtype)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(batch, 1, size_y, 1)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(batch, size_x, 1, 1)

        midx = 0.5
        midy = (size_y - 1) / (2 * (size_x - 1))
        radial = (gridx - midx) ** 2 + (gridy - midy) ** 2
        return torch.cat((x, radial), dim=-1)


class P4Conv2d(nn.Module):
    """p4/C4 group convolution.

    The four group elements correspond to rotations by
    0, 90, 180 and 270 degrees. Internally, the group axis is flattened
    into the PyTorch channel dimension.
    """

    group_size = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        bias: bool = True,
        first_layer: bool = False,
        last_layer: bool = False,
        spectral: bool = False,
        hermitian: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_y = kernel_size
        self.kernel_size_x = kernel_size // 2 + 1 if hermitian else kernel_size
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.hermitian = hermitian

        dtype = torch.cfloat if spectral else torch.float

        if first_layer or last_layer:
            self.W = nn.Parameter(
                torch.empty(
                    out_channels,
                    1,
                    in_channels,
                    kernel_size,
                    self.kernel_size_x,
                    dtype=dtype,
                )
            )
        elif hermitian:
            self.W = nn.ParameterDict(
                {
                    "y0_modes": nn.Parameter(
                        torch.empty(
                            out_channels,
                            1,
                            in_channels,
                            4,
                            self.kernel_size_x - 1,
                            1,
                            dtype=dtype,
                        )
                    ),
                    "yposx_modes": nn.Parameter(
                        torch.empty(
                            out_channels,
                            1,
                            in_channels,
                            4,
                            self.kernel_size_y,
                            self.kernel_size_x - 1,
                            dtype=dtype,
                        )
                    ),
                    "00_modes": nn.Parameter(
                        torch.empty(
                            out_channels,
                            1,
                            in_channels,
                            4,
                            1,
                            1,
                            dtype=torch.float,
                        )
                    ),
                }
            )
        else:
            self.W = nn.Parameter(
                torch.empty(
                    out_channels,
                    1,
                    in_channels,
                    4,
                    kernel_size,
                    self.kernel_size_x,
                    dtype=dtype,
                )
            )

        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1)) if bias else None
        self._eval_weight_built = False
        self.reset_parameters()
        self._build_weight()

    def reset_parameters(self) -> None:
        if isinstance(self.W, nn.ParameterDict):
            for value in self.W.values():
                nn.init.kaiming_uniform_(value, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def _base_weight(self) -> torch.Tensor:
        if not self.hermitian:
            return self.W

        # Build the Hermitian-compatible kernel used with rfft2.
        weight = torch.cat(
            [
                self.W["y0_modes"],
                self.W["00_modes"].cfloat(),
                self.W["y0_modes"].flip(dims=(-2,)).conj(),
            ],
            dim=-2,
        )
        weight = torch.cat([weight, self.W["yposx_modes"]], dim=-1)
        weight = torch.cat(
            [weight[..., 1:].conj().rot90(k=2, dims=[-2, -1]), weight],
            dim=-1,
        )
        return weight

    def _build_weight(self) -> None:
        # Cache transformed kernels during evaluation only. During training the
        # transformed kernel must be rebuilt because parameters change.
        if not self.training and self._eval_weight_built:
            return

        base = self._base_weight()

        if self.first_layer or self.last_layer:
            # Lifting/projection between planar fields and p4 feature maps.
            weight = base.repeat(1, 4, 1, 1, 1)
            for k in range(1, 4):
                weight[:, k] = weight[:, k].rot90(k=k, dims=[-2, -1])

            if self.first_layer:
                weight = weight.reshape(
                    -1,
                    self.in_channels,
                    self.kernel_size_y,
                    self.kernel_size_y,
                )
                bias = self.B.repeat_interleave(4, dim=1) if self.B is not None else None
            else:
                weight = weight.transpose(2, 1).reshape(
                    self.out_channels,
                    -1,
                    self.kernel_size_y,
                    self.kernel_size_y,
                )
                bias = self.B
        else:
            # Group-to-group convolution: spatial rotation plus cyclic shift
            # of the input group index.
            weight = base.repeat(1, 4, 1, 1, 1, 1)
            for k in range(1, 4):
                weight[:, k] = weight[:, k - 1].rot90(dims=[-2, -1])
                weight[:, k] = torch.cat(
                    [weight[:, k, :, -1].unsqueeze(2), weight[:, k, :, :-1]],
                    dim=2,
                )

            weight = weight.reshape(
                self.out_channels * 4,
                self.in_channels * 4,
                self.kernel_size_y,
                self.kernel_size_y,
            )
            bias = self.B.repeat_interleave(4, dim=1) if self.B is not None else None

        if self.hermitian:
            weight = weight[..., -self.kernel_size_x :]

        self.weight = weight
        self.bias = bias
        self._eval_weight_built = not self.training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._build_weight()
        x = F.conv2d(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x


class P4SpectralConv2d(nn.Module):
    """Group-equivariant Fourier convolution over retained low modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        self.conv = P4Conv2d(
            in_channels,
            out_channels,
            kernel_size=2 * modes - 1,
            bias=False,
            spectral=True,
            hermitian=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modes > x.size(-1) // 2 + 1:
            raise ValueError(
                f"modes={self.modes} is too large for spatial width={x.size(-1)}"
            )

        self.conv._build_weight()
        weights = self.conv.weight.transpose(0, 1)

        batch = x.shape[0]
        freq0_y = (
            torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2], device=x.device)) == 0
        ).nonzero().item()

        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[
            ...,
            freq0_y - self.modes + 1 : freq0_y + self.modes,
            : self.modes,
        ]

        out_ft = torch.zeros(
            batch,
            weights.shape[1],
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[
            ...,
            freq0_y - self.modes + 1 : freq0_y + self.modes,
            : self.modes,
        ] = torch.einsum("bixy,ioxy->boxy", x_ft, weights)

        return torch.fft.irfft2(
            torch.fft.ifftshift(out_ft, dim=-2),
            s=(x.size(-2), x.size(-1)),
        )


class P4MLP2d(nn.Module):
    """Pointwise G-MLP: 1x1 G-Conv -> GELU -> 1x1 G-Conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        *,
        last_layer: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = P4Conv2d(in_channels, mid_channels, kernel_size=1)
        self.fc2 = P4Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            last_layer=last_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GroupNorm2d(nn.Module):
    """Normalization respecting the explicit four-element group axis."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.width, 4, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        return x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])


class GFNO(nn.Module):
    """Core G-FNO2d-p4 model for the paper's standard NS experiment.

    Architecture:
        symmetric lifting P_G
        -> 4 x [G-Fourier + G-MLP + 1x1 G-Conv residual]
        -> group projection Q_G

    Expected input:  [B, X, Y, T_in, C]
    Returned output: [B, X, Y, 1, C]
    """

    def __init__(
        self,
        num_channels: int = 1,
        modes: int = 12,
        width: int = 10,
        initial_step: int = 10,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.modes = modes
        self.width = width
        self.initial_step = initial_step
        self.grid = SymmetricGrid2D()

        # P_G: lift input history + symmetric coordinate into group features.
        self.p = P4Conv2d(
            in_channels=num_channels * initial_step + self.grid.grid_dim,
            out_channels=width,
            kernel_size=1,
            first_layer=True,
        )

        # Four G-Fourier layers, matching the paper/official implementation.
        self.spectral = nn.ModuleList(
            [P4SpectralConv2d(width, width, modes) for _ in range(4)]
        )
        self.mlp = nn.ModuleList(
            [P4MLP2d(width, width, width) for _ in range(4)]
        )
        self.skip = nn.ModuleList(
            [P4Conv2d(width, width, kernel_size=1) for _ in range(4)]
        )
        self.norm = GroupNorm2d(width)

        # Q_G: project group features back to the scalar PDE field.
        self.q = P4MLP2d(
            width,
            num_channels,
            width * 4,
            last_layer=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B,X,Y,T,C], got {tuple(x.shape)}")
        if x.shape[-2] != self.initial_step:
            raise ValueError(
                f"expected T_in={self.initial_step}, got {x.shape[-2]}"
            )
        if x.shape[-1] != self.num_channels:
            raise ValueError(
                f"expected channels={self.num_channels}, got {x.shape[-1]}"
            )

        # Flatten history and physical channels before lifting.
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        x = self.grid(x)
        x = x.permute(0, 3, 1, 2)
        x = self.p(x)

        for i in range(4):
            spectral_branch = self.norm(self.spectral[i](self.norm(x)))
            spectral_branch = self.mlp[i](spectral_branch)
            x = spectral_branch + self.skip[i](x)
            if i < 3:
                x = F.gelu(x)

        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x.unsqueeze(-2)

