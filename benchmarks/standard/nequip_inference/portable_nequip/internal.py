"""Small runtime helpers required by the vendored NequIP modules."""

from __future__ import annotations

import contextlib
import contextvars
from typing import Union

import torch
from e3nn import get_optimization_defaults, set_optimization_defaults


_GLOBAL_DTYPE = torch.float64
NEQUIP_CUSTOM_OPS_LIBS_KEY = "nequip_custom_ops_libs"
_TORCH_GE_2_7 = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 7)
_CONDITIONAL_TORCHSCRIPT_MODE = contextvars.ContextVar(
    "_CONDITIONAL_TORCHSCRIPT_MODE", default=True
)


@contextlib.contextmanager
def torch_default_dtype(dtype: torch.dtype):
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original)


@contextlib.contextmanager
def conditional_torchscript_mode(enabled: bool):
    original_e3nn = get_optimization_defaults()["jit_script_fx"]
    token = _CONDITIONAL_TORCHSCRIPT_MODE.set(bool(enabled))
    set_optimization_defaults(jit_script_fx=bool(enabled))
    try:
        yield
    finally:
        set_optimization_defaults(jit_script_fx=original_e3nn)
        _CONDITIONAL_TORCHSCRIPT_MODE.reset(token)


def conditional_torchscript_jit(
    module: torch.nn.Module,
) -> Union[torch.jit.ScriptModule, torch.nn.Module]:
    if _CONDITIONAL_TORCHSCRIPT_MODE.get():
        return torch.jit.script(module)
    return module
