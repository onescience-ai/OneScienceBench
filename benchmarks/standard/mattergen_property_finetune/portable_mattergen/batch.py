"""Minimal tensor container for batched periodic crystals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class CrystalBatch(dict[str, Any]):
    """Dictionary-compatible crystal batch with MatterGen batching helpers."""

    def __init__(self, **values: Any) -> None:
        super().__init__(values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def replace(self, **values: Any) -> "CrystalBatch":
        updated = dict(self)
        updated.update(values)
        return type(self)(**updated)

    def get_batch_idx(self, field_name: str) -> torch.LongTensor:
        if field_name not in {"pos", "atomic_numbers"}:
            raise KeyError(f"unsupported batched field: {field_name}")
        num_atoms = self["num_atoms"]
        return torch.repeat_interleave(
            torch.arange(num_atoms.numel(), device=num_atoms.device),
            num_atoms,
        )

    def to(self, device: str | torch.device) -> "CrystalBatch":
        return type(self)(
            **{
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.items()
            }
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CrystalBatch":
        return cls(**dict(values))


ChemGraph = CrystalBatch

__all__ = ["ChemGraph", "CrystalBatch"]
