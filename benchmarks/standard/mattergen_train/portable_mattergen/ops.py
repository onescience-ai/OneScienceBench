"""Small differentiable PyTorch replacements for optional PyG native operators."""

from __future__ import annotations

import torch
from torch import nn


def _size_as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size=None,
    reduce: str = "sum",
) -> torch.Tensor:
    if dim != 0:
        raise ValueError("portable scatter currently supports dim=0 only")
    if reduce not in {"sum", "add"}:
        raise ValueError(f"unsupported scatter reduction: {reduce}")
    index = index.to(device=src.device, dtype=torch.long)
    if dim_size is None:
        size = int(index.max().item()) + 1 if index.numel() else 0
    else:
        size = _size_as_int(dim_size)
    output = src.new_zeros((size, *src.shape[1:]))
    expanded_index = index.reshape((-1,) + (1,) * (src.ndim - 1)).expand_as(src)
    return output.scatter_add(0, expanded_index, src)


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size=None,
) -> torch.Tensor:
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    reduce: str = "sum",
) -> torch.Tensor:
    if reduce not in {"sum", "add"}:
        raise ValueError(f"unsupported segment reduction: {reduce}")
    rows = []
    for start, stop in zip(indptr[:-1], indptr[1:]):
        begin = _size_as_int(start)
        end = _size_as_int(stop)
        if begin == end:
            rows.append(src.new_zeros(src.shape[1:]))
        else:
            rows.append(src[begin:end].sum(dim=0))
    if rows:
        return torch.stack(rows, dim=0)
    return src.new_zeros((0, *src.shape[1:]))


def segment_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size=None,
    reduce: str = "sum",
) -> torch.Tensor:
    return scatter(src, index, dim=0, dim_size=dim_size, reduce=reduce)


class GaussianSmearing(nn.Module):
    """API-compatible implementation of PyG's GaussianSmearing layer."""

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / float(offset[1] - offset[0]) ** 2
        if trainable:
            self.offset = nn.Parameter(offset)
        else:
            self.register_buffer("offset", offset)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        difference = distance.reshape(-1, 1) - self.offset.reshape(1, -1)
        return torch.exp(self.coeff * difference.pow(2))


__all__ = [
    "GaussianSmearing",
    "scatter",
    "scatter_add",
    "segment_coo",
    "segment_csr",
]
