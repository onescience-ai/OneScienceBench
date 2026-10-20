"""Independent local NequIP implementation used by this evaluation task."""

from .model import build_nequip_model
from .runtime import *

__all__ = ["build_nequip_model"]
