"""TACE model architectures exposed by OneScience."""

from .e3nn import e3nnTACE

try:
    from .cartesian import cartTACE
except Exception:  # Optional Cartesian backend dependencies.
    cartTACE = None

try:
    from ..scf import SCFTACE
except Exception:  # Optional SCF dependencies.
    SCFTACE = None

__all__ = ["cartTACE", "e3nnTACE", "SCFTACE"]
