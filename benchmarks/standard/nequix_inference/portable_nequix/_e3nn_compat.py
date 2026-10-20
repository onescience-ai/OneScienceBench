"""e3nn compatibility for the NequIX torch backend on e3nn < 0.5.8.

NequIX's torch model (``model.py``) calls ``Irreps.regroup()``,
``Irreps.filter()`` and ``Irreps.slice_by_mul``, all added in e3nn 0.5.x. The
OneScience matchem environment ships e3nn 0.4.4, which lacks them. Rather than
upgrading the shared e3nn (which would risk the other MatChem models: MACE,
eSEN, NequIP, EquiformerV3), we add the smallest integration-side compatibility
patch. Semantics are taken from e3nn 0.5.9 / e3nn_jax:

- ``regroup()``: ``sort()`` then ``simplify()`` (merge identical irreps after
  sorting).
- ``filter(keep=..., drop=..., lmax=...)``: keep/drop by irrep set or callable,
  or by ``lmax``; ``keep`` and ``drop`` are mutually exclusive.
- ``slice_by_mul``: index by channel (multiplicity) index, returning an
  ``Irreps``.

The patch is a no-op when the installed e3nn already provides the methods.
"""
from __future__ import annotations

from e3nn import o3


def _irrep_set(spec) -> set:
    """Normalize a keep/drop spec to a set of ``e3nn.o3.Irrep``."""
    if isinstance(spec, str):
        spec = o3.Irreps(spec)
    if isinstance(spec, o3.Irrep):
        spec = [spec]
    out = set()
    for item in spec:
        if isinstance(item, str):
            out.add(o3.Irrep(item))
        elif isinstance(item, o3.Irrep):
            out.add(item)
        else:
            raise TypeError(f"unsupported irrep spec element: {item!r}")
    return out


def _make_regroup():
    def _regroup(self):
        # e3nn 0.4.x sort() returns a Sort namedtuple (irreps, inv, perm);
        # [0] is the sorted Irreps.
        return self.sort()[0].simplify()

    return _regroup


def _make_filter():
    def _filter(self, keep=None, *, drop=None, lmax=None):
        if keep is None and drop is None and lmax is None:
            return self
        if keep is not None and drop is not None:
            raise ValueError("Cannot specify both keep and drop")
        if keep is not None and lmax is not None:
            raise ValueError("Cannot specify both keep and lmax")
        if drop is not None and lmax is not None:
            raise ValueError("Cannot specify both drop and lmax")

        if keep is not None:
            if callable(keep):
                return o3.Irreps([mul_ir for mul_ir in self if keep(mul_ir)])
            keep_set = _irrep_set(keep)
            return o3.Irreps([(mul, ir) for mul, ir in self if ir in keep_set])

        if drop is not None:
            if callable(drop):
                return o3.Irreps([mul_ir for mul_ir in self if not drop(mul_ir)])
            drop_set = _irrep_set(drop)
            return o3.Irreps([(mul, ir) for mul, ir in self if ir not in drop_set])

        return o3.Irreps([(mul, ir) for mul, ir in self if ir.l <= lmax])

    return _filter


class _MulIndexSliceHelper:
    """Replicates e3nn 0.5.x ``Irreps.slice_by_mul`` (index by channel index)."""

    def __init__(self, irreps):
        self.irreps = irreps

    def __getitem__(self, index):
        if not isinstance(index, slice):
            raise IndexError("Irreps.slice_by_mul only supports slices.")
        start, stop, stride = index.indices(self.irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError("Irreps.slice_by_mul does not support strides.")
        out = []
        i = 0
        for mul, ir in self.irreps:
            if start <= i and i + mul <= stop:
                out.append((mul, ir))
            elif start < i + mul and i < stop:
                out.append((min(stop, i + mul) - max(start, i), ir))
            i += mul
        return o3.Irreps(out)


def _make_slice_by_mul():
    @property
    def _slice_by_mul(self):
        return _MulIndexSliceHelper(self)

    return _slice_by_mul


def ensure_irreps_compat() -> None:
    """Monkey-patch missing ``Irreps`` methods if the installed e3nn lacks them."""
    if not hasattr(o3.Irreps, "regroup"):
        o3.Irreps.regroup = _make_regroup()
    if not hasattr(o3.Irreps, "filter"):
        o3.Irreps.filter = _make_filter()
    if not hasattr(o3.Irreps, "slice_by_mul"):
        o3.Irreps.slice_by_mul = _make_slice_by_mul()


ensure_irreps_compat()
