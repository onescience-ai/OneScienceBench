from __future__ import annotations

import os
from functools import wraps
from pathlib import Path


class _Registry:
    def __init__(self):
        self.models = {}

    def register_model(self, name):
        def decorator(cls):
            self.models[name] = cls
            return cls
        return decorator


registry = _Registry()


def conditional_grad(_context):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return function(*args, **kwargs)
        return wrapped
    return decorator


def resolve_esen_jd_path(path=None):
    candidate = path or os.environ.get("ONESCIENCE_UMA_JD_PATH")
    if not candidate:
        raise FileNotFoundError("Jd.pt path is required")
    candidate = Path(candidate).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return str(candidate)


def cg_change_mat(*_args, **_kwargs):
    raise NotImplementedError("rank-2 head is outside this task")


def irreps_sum(*_args, **_kwargs):
    raise NotImplementedError("rank-2 head is outside this task")
