"""Backward-compatible TACE model exports.

Implementation is owned by ``portable_tace.models.tace.architectures`` and
``portable_tace.modules.layer.tace``.  Keep this shim for existing Hydra configs
and checkpoints that reference ``portable_tace.models.tace.models``.
"""

from ..architectures import SCFTACE, cartTACE, e3nnTACE
from ..wrappers import TensorModel
from portable_tace.utils.tace.runtime import CompileTensorModel

__all__ = [
    "cartTACE",
    "e3nnTACE",
    "TensorModel",
    "CompileTensorModel",
    "SCFTACE",
]
