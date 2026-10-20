"""Multi-scale Radon Neural Operator for two-dimensional Darcy flow.

The original encoder, transformer decoder and Radon primitives live in the
flat sibling module ``rno_layers.py``.  This module replaces the single-scale
processor with fine/coarse Radon branches and learnable gated fusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rno_layers import AngleWeightNetwork, DifferentiableRadon2D, MLP, RNOTransformerBlock, _fourier_filter


class CachedDifferentiableRadon2D(DifferentiableRadon2D):
    """Radon transform with non-persistent sampling-grid caches.

    Grids depend only on resolution, angle, dtype and device.  Keeping them out
    of ``state_dict`` preserves checkpoint portability while avoiding repeated
    grid construction during training and inference.
    """

    def __init__(self, num_angles: int, include_pi_endpoint: bool = True, cache_grids: bool = True):
        super().__init__(num_angles=num_angles, include_pi_endpoint=include_pi_endpoint)
        self.cache_grids = cache_grids
        self._projection_grids: dict[tuple, torch.Tensor] = {}
        self._backprojection_grids: dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _key(index: int, height: int, width: int, tensor: torch.Tensor) -> tuple:
        return index, height, width, str(tensor.device), str(tensor.dtype)

    def _projection_grid(
        self, index: int, angle: torch.Tensor, height: int, width: int, reference: torch.Tensor
    ) -> torch.Tensor:
        key = self._key(index, height, width, reference)
        if self.cache_grids and key in self._projection_grids:
            return self._projection_grids[key]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
        theta = torch.stack((cosine, sine, zero, -sine, cosine, zero)).reshape(1, 2, 3)
        grid = F.affine_grid(theta, torch.Size((1, 1, height, width)), align_corners=True)
        if self.cache_grids:
            self._projection_grids[key] = grid.detach()
        return grid

    def _backprojection_grid(
        self, index: int, angle: torch.Tensor, height: int, width: int, reference: torch.Tensor
    ) -> torch.Tensor:
        key = self._key(index, height, width, reference)
        if self.cache_grids and key in self._backprojection_grids:
            return self._backprojection_grids[key]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype),
            torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype),
            indexing="ij",
        )
        detector = x_grid * torch.cos(angle) - y_grid * torch.sin(angle)
        angle_coordinate = 0.0 if self.num_angles == 1 else -1.0 + 2.0 * index / (self.num_angles - 1)
        angle_plane = torch.full_like(detector, angle_coordinate)
        grid = torch.stack((angle_plane, detector), dim=-1).unsqueeze(0)
        if self.cache_grids:
            self._backprojection_grids[key] = grid.detach()
        return grid

    def forward_projection(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        left, right, top, bottom, padded_height, padded_width = self._padding(height, width)
        padded = F.pad(image, (left, right, top, bottom), mode="constant", value=0.0)
        projections = []
        for index, angle in enumerate(self.angles.to(device=image.device, dtype=image.dtype)):
            grid = self._projection_grid(index, angle, padded_height, padded_width, image)
            rotated = F.grid_sample(
                padded, grid.expand(batch, -1, -1, -1), mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )
            projections.append(rotated.sum(dim=3))
        return torch.stack(projections, dim=-1)

    def filtered_backprojection(self, sinogram: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
        batch, channels, detector_count, angle_count = sinogram.shape
        if angle_count != self.num_angles:
            raise ValueError(f"Expected {self.num_angles} angles, received {angle_count}")
        height, width = output_shape
        padded_size = max(64, 2 ** math.ceil(math.log2(2 * detector_count)))
        padded = F.pad(sinogram, (0, 0, 0, padded_size - detector_count))
        ramp = _fourier_filter(padded_size, sinogram.device, sinogram.dtype).reshape(1, 1, -1, 1)
        filtered = torch.fft.ifft(torch.fft.fft(padded, dim=2) * ramp, dim=2).real
        filtered = filtered[:, :, :detector_count]
        reconstruction = torch.zeros(
            (batch, channels, height, width), device=sinogram.device, dtype=sinogram.dtype
        )
        for index, angle in enumerate(self.angles.to(device=sinogram.device, dtype=sinogram.dtype)):
            grid = self._backprojection_grid(index, angle, height, width, sinogram)
            reconstruction = reconstruction + F.grid_sample(
                filtered, grid.expand(batch, -1, -1, -1), mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )
        return reconstruction * torch.pi / (2.0 * self.num_angles)


class CachedRadonResidualBlock(nn.Module):
    def __init__(
        self,
        num_angles: int,
        angle_weighting: str,
        angle_conv_channels: int,
        angle_conv_kernel_size: int,
        include_pi_endpoint: bool,
        cache_grids: bool,
    ):
        super().__init__()
        self.radon = CachedDifferentiableRadon2D(
            num_angles, include_pi_endpoint=include_pi_endpoint, cache_grids=cache_grids
        )
        self.angle_weights = AngleWeightNetwork(
            num_angles, mode=angle_weighting, conv_channels=angle_conv_channels,
            kernel_size=angle_conv_kernel_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sinogram = self.radon.forward_projection(inputs)
        reconstruction = self.radon.filtered_backprojection(
            sinogram * self.angle_weights(sinogram), inputs.shape[-2:]
        )
        return inputs + reconstruction


class MultiScaleRadonProcessor(nn.Module):
    """Parallel fine/coarse Radon propagation followed by gated residual fusion."""

    def __init__(
        self,
        hidden_dim: int,
        coarse_resolution: int,
        fine_blocks: int,
        coarse_blocks: int,
        fine_angles: int,
        coarse_angles: int,
        angle_weighting: str,
        angle_conv_channels: int,
        angle_conv_kernel_size: int,
        include_pi_endpoint: bool,
        cache_grids: bool,
    ):
        super().__init__()
        self.coarse_resolution = coarse_resolution
        block_args = dict(
            angle_weighting=angle_weighting,
            angle_conv_channels=angle_conv_channels,
            angle_conv_kernel_size=angle_conv_kernel_size,
            include_pi_endpoint=include_pi_endpoint,
            cache_grids=cache_grids,
        )
        self.fine_branch = nn.Sequential(
            *(CachedRadonResidualBlock(num_angles=fine_angles, **block_args) for _ in range(fine_blocks))
        )
        self.coarse_lift = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.coarse_branch = nn.Sequential(
            *(CachedRadonResidualBlock(num_angles=coarse_angles, **block_args) for _ in range(coarse_blocks))
        )
        self.gate = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1)
        self.correction = nn.Sequential(
            nn.Conv2d(3 * hidden_dim, hidden_dim, kernel_size=1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

    def forward(
        self, fine_features: torch.Tensor, local_features: torch.Tensor, coarse_field: torch.Tensor
    ) -> torch.Tensor:
        fine = self.fine_branch(fine_features)
        coarse_seed = F.interpolate(
            fine_features, size=(self.coarse_resolution, self.coarse_resolution),
            mode="bilinear", align_corners=True,
        )
        coarse_seed = (coarse_seed + self.coarse_lift(coarse_field)) * (2.0**-0.5)
        coarse = self.coarse_branch(coarse_seed)
        coarse_up = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=True)
        gate = torch.sigmoid(self.gate(torch.cat((fine, coarse_up), dim=1)))
        fused = gate * fine + (1.0 - gate) * coarse_up
        correction = self.correction(torch.cat((fused, local_features, fine_features), dim=1))
        return fine_features + correction


class RadonNeuralOperator2D(nn.Module):
    """Multi-scale RNO mapping permeability coefficients to Darcy pressure."""

    def __init__(
        self, height: int = 85, width: int = 85, coarse_resolution: int = 43,
        hidden_dim: int = 128, num_heads: int = 8, num_layers: int = 8,
        num_angles: int = 32, fine_radon_blocks: int = 2, coarse_radon_blocks: int = 2,
        coarse_num_angles: int = 16, slice_num: int = 64, ref_grid_size: int = 8,
        unified_pos: bool = True, dropout: float = 0.0, mlp_ratio: int = 1,
        input_dim: int = 1, output_dim: int = 1, angle_weighting: str = "paper_conv",
        angle_conv_channels: int = 16, angle_conv_kernel_size: int = 3,
        include_pi_endpoint: bool = True, cache_radon_grids: bool = True,
    ):
        super().__init__()
        self.height, self.width, self.hidden_dim = height, width, hidden_dim
        self.coarse_resolution, self.unified_pos = coarse_resolution, unified_pos
        if unified_pos:
            features = self._make_unified_position_features(height, width, ref_grid_size)
            self.register_buffer("unified_position_features", features, persistent=True)
            position_dim = ref_grid_size * ref_grid_size
        else:
            self.register_buffer("unified_position_features", torch.empty(0), persistent=False)
            position_dim = 2
        self.preprocess = MLP(input_dim + position_dim, hidden_dim * 2, hidden_dim, hidden_layers=0, residual=False)
        self.local_feature = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        self.processor = MultiScaleRadonProcessor(
            hidden_dim, coarse_resolution, fine_radon_blocks, coarse_radon_blocks,
            num_angles, coarse_num_angles, angle_weighting, angle_conv_channels,
            angle_conv_kernel_size, include_pi_endpoint, cache_radon_grids,
        )
        self.operator_blocks = nn.ModuleList(
            RNOTransformerBlock(
                hidden_dim, num_heads, slice_num, height, width, dropout, mlp_ratio,
                output_dim, index == num_layers - 1,
            )
            for index in range(num_layers)
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _make_unified_position_features(height: int, width: int, reference_size: int) -> torch.Tensor:
        x, y = torch.meshgrid(torch.linspace(0.0, 1.0, height), torch.linspace(0.0, 1.0, width), indexing="ij")
        rx, ry = torch.meshgrid(
            torch.linspace(0.0, 1.0, reference_size), torch.linspace(0.0, 1.0, reference_size), indexing="ij"
        )
        grid, reference = torch.stack((x, y), dim=-1), torch.stack((rx, ry), dim=-1)
        distances = torch.linalg.vector_norm(grid[:, :, None, None] - reference[None, None], dim=-1)
        return distances.reshape(1, height * width, reference_size * reference_size)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, positions: torch.Tensor, field: torch.Tensor, coarse_field: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, nodes, _ = field.shape
        if nodes != self.height * self.width:
            raise ValueError(f"Expected {self.height * self.width} fine nodes, received {nodes}")
        if coarse_field is None:
            coarse_field = F.interpolate(
                field.transpose(1, 2).reshape(batch, 1, self.height, self.width),
                size=(self.coarse_resolution, self.coarse_resolution), mode="bilinear", align_corners=True,
            )
        else:
            if coarse_field.ndim == 2:
                coarse_field = coarse_field.unsqueeze(-1)
            expected = self.coarse_resolution * self.coarse_resolution
            if coarse_field.shape[1] != expected:
                raise ValueError(f"Expected {expected} coarse nodes, received {coarse_field.shape[1]}")
            coarse_field = coarse_field.transpose(1, 2).reshape(
                batch, 1, self.coarse_resolution, self.coarse_resolution
            )
        position_features = self.unified_position_features.expand(batch, -1, -1) if self.unified_pos else positions
        hidden = self.preprocess(torch.cat((position_features, field), dim=-1))
        spatial = hidden.reshape(batch, self.height, self.width, self.hidden_dim).permute(0, 3, 1, 2)
        local = self.local_feature(spatial.reshape(batch * self.hidden_dim, 1, self.height, self.width))
        local = local.reshape(batch, self.hidden_dim, self.height, self.width)
        hidden = self.processor(spatial, local, coarse_field)
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch, nodes, self.hidden_dim)
        for block in self.operator_blocks:
            hidden = block(hidden)
        return hidden


MultiScaleRNO2D = RadonNeuralOperator2D
RNO2D = RadonNeuralOperator2D


def build_model(config: dict) -> RadonNeuralOperator2D:
    model_cfg, data_cfg = config["model"], config["data"]
    resolution = int(data_cfg["resolution"])
    return RadonNeuralOperator2D(
        height=resolution, width=resolution,
        coarse_resolution=int(data_cfg["coarse_resolution"]),
        hidden_dim=int(model_cfg["hidden_dim"]), num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]), num_angles=int(model_cfg["num_angles"]),
        fine_radon_blocks=int(model_cfg["fine_radon_blocks"]),
        coarse_radon_blocks=int(model_cfg["coarse_radon_blocks"]),
        coarse_num_angles=int(model_cfg["coarse_num_angles"]),
        slice_num=int(model_cfg["slice_num"]), ref_grid_size=int(model_cfg["ref_grid_size"]),
        unified_pos=bool(model_cfg["unified_pos"]), dropout=float(model_cfg["dropout"]),
        mlp_ratio=int(model_cfg["mlp_ratio"]), input_dim=int(model_cfg["input_dim"]),
        output_dim=int(model_cfg["output_dim"]), angle_weighting=str(model_cfg["angle_weighting"]),
        angle_conv_channels=int(model_cfg["angle_conv_channels"]),
        angle_conv_kernel_size=int(model_cfg["angle_conv_kernel_size"]),
        include_pi_endpoint=bool(model_cfg["include_pi_endpoint"]),
        cache_radon_grids=bool(model_cfg.get("cache_radon_grids", True)),
    )
