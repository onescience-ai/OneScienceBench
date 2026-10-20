from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JacobiKANLayer(nn.Module):
    """Shared KAN layer using Jacobi polynomial bases."""

    def __init__(self, input_dim: int, output_dim: int, degree: int, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.jacobi_coeffs = nn.Parameter(torch.empty(self.input_dim, self.output_dim, self.degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1.0 / (self.input_dim * (self.degree + 1)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_points = inputs.shape
        x = torch.tanh(inputs.permute(0, 2, 1).contiguous())

        polynomials = [torch.ones_like(x)]
        if self.degree > 0:
            polynomials.append(((self.alpha - self.beta) + (self.alpha + self.beta + 2.0) * x) / 2.0)

        for order in range(2, self.degree + 1):
            order_float = float(order)
            a_term = (2.0 * order_float + self.alpha + self.beta - 1.0) * (
                2.0 * order_float + self.alpha + self.beta
            ) / ((2.0 * order_float) * (order_float + self.alpha + self.beta))
            b_term = (
                (2.0 * order_float + self.alpha + self.beta - 1.0)
                * (self.alpha**2 - self.beta**2)
                / (
                    (2.0 * order_float)
                    * (order_float + self.alpha + self.beta)
                    * (2.0 * order_float + self.alpha + self.beta - 2.0)
                )
            )
            c_term = (
                -2.0
                * (order_float + self.alpha - 1.0)
                * (order_float + self.beta - 1.0)
                * (2.0 * order_float + self.alpha + self.beta)
                / (
                    (2.0 * order_float)
                    * (order_float + self.alpha + self.beta)
                    * (2.0 * order_float + self.alpha + self.beta - 2.0)
                )
            )
            polynomials.append(
                (a_term * x + b_term) * polynomials[order - 1]
                + c_term * polynomials[order - 2]
            )

        jacobi = torch.stack(polynomials, dim=-1)
        jacobi = jacobi.permute(0, 2, 3, 1)
        return torch.einsum("bids,iod->bos", jacobi, self.jacobi_coeffs)


class KAPointNet(nn.Module):
    """KA-PointNet segmentation branch for 2D CFD field prediction."""

    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 3,
        network_scaling: float = 1.0,
        poly_degree: int = 5,
        alpha: float = 1.0,
        beta: float = 1.0,
        num_points: int = 1024,
    ) -> None:
        super().__init__()
        width_64 = int(64 * network_scaling)
        width_128 = int(128 * network_scaling)
        width_256 = int(256 * network_scaling)
        width_512 = int(512 * network_scaling)
        width_1024 = int(1024 * network_scaling)

        self.num_points = int(num_points)

        self.jacobikan1 = JacobiKANLayer(input_channels, width_64, poly_degree, alpha, beta)
        self.jacobikan2 = JacobiKANLayer(width_64, width_64, poly_degree, alpha, beta)

        self.jacobikan3 = JacobiKANLayer(width_64, width_64, poly_degree, alpha, beta)
        self.jacobikan4 = JacobiKANLayer(width_64, width_128, poly_degree, alpha, beta)
        self.jacobikan5 = JacobiKANLayer(width_128, width_1024, poly_degree, alpha, beta)

        self.jacobikan6 = JacobiKANLayer(width_1024 + width_64, width_512, poly_degree, alpha, beta)
        self.jacobikan7 = JacobiKANLayer(width_512, width_256, poly_degree, alpha, beta)
        self.jacobikan8 = JacobiKANLayer(width_256, width_128, poly_degree, alpha, beta)

        self.jacobikan9 = JacobiKANLayer(width_128, width_128, poly_degree, alpha, beta)
        self.jacobikan10 = JacobiKANLayer(width_128, output_channels, poly_degree, alpha, beta)

        self.bn1 = nn.BatchNorm1d(width_64)
        self.bn2 = nn.BatchNorm1d(width_64)
        self.bn3 = nn.BatchNorm1d(width_64)
        self.bn4 = nn.BatchNorm1d(width_128)
        self.bn5 = nn.BatchNorm1d(width_1024)
        self.bn6 = nn.BatchNorm1d(width_512)
        self.bn7 = nn.BatchNorm1d(width_256)
        self.bn8 = nn.BatchNorm1d(width_128)
        self.bn9 = nn.BatchNorm1d(width_128)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.jacobikan1(inputs))
        x = self.bn2(self.jacobikan2(x))
        local_feature = x

        x = self.bn3(self.jacobikan3(x))
        x = self.bn4(self.jacobikan4(x))
        x = self.bn5(self.jacobikan5(x))

        global_feature = F.max_pool1d(x, kernel_size=self.num_points)
        global_feature = global_feature.view(-1, global_feature.size(1), 1).expand(-1, -1, self.num_points)

        x = torch.cat([local_feature, global_feature], dim=1)
        x = self.bn6(self.jacobikan6(x))
        x = self.bn7(self.jacobikan7(x))
        x = self.bn8(self.jacobikan8(x))
        x = self.bn9(self.jacobikan9(x))
        return self.jacobikan10(x)
