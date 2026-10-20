"""V2 adapter for the legacy GraphCast conservation projection evaluator."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

_BASE_EVALUATOR_DIR = None
for _parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    _base_dir = _parent / 'benchmarks' / 'evaluators' / 'base'
    if _base_dir.is_dir():
        _BASE_EVALUATOR_DIR = _base_dir
        break
if _BASE_EVALUATOR_DIR is None:
    raise FileNotFoundError('Could not locate repository evaluators/base directory.')
sys.path.insert(0, str(_BASE_EVALUATOR_DIR))
from base_evaluator import *


def _repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / 'benchmarks' / 'datasets').is_dir():
            return candidate
    raise FileNotFoundError('Could not locate repository root containing benchmarks/datasets.')


candidate_env = os.environ.get("candidate_env")
stardard_name = os.environ.get("stardard_name")

def _candidate_model_path() -> Path:
    if not candidate_env:
        raise RuntimeError("candidate_env is not configured")
    candidate_path = os.environ.get(candidate_env)
    if not candidate_path:
        raise RuntimeError(f"{candidate_env} is not configured")
    return Path(candidate_path).expanduser().resolve()

def _reference_model_path() -> Path:
    return (
        _repo_root()
        / "benchmarks"
        / "standard"
        / stardard_name
    ).resolve()

REQUIRED_FILES = ("graphcast_conservation_projection.py", "train.py")
ENV_NAME = "GRAPHCAST_CONSERVATION_PROJECTION_SUBMISSION_DIR"
HINTS = ("graphcast_conservation_projection", "graphcast_conservation_projection_V1")
ALLOWED = {"file", "data", "train_script", "model_structure", "forward_executable", "forward_equivalence", "traning"}

CANDIDATE_DIR = _candidate_model_path()
REFERENCE_DIR = _reference_model_path()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "file:graphcast_conservation_projection.py",
        "file:train.py",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        for name in REQUIRED_FILES:
            ok = (submission / name).is_file()
            msg = str(submission / name)
            update_check(current_index, ok, msg)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["data:integral_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        x = torch.randn(2, 2, 6, 8)
        area = torch.ones(6, 8)
        target_integrals = torch.tensor([[1.5, -0.5], [0.25, 0.75]])
        ok = target_integrals.shape == (2, 2) and area.shape == (6, 8)
        msg = "target integrals and area weights"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["train_script:public_entrypoints"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        train = load_module(
            "graphcast_conservation_projection_train_submission",
            CANDIDATE_DIR / REQUIRED_FILES[1],
        )
        ok = callable(getattr(train, "build_model", None)) and callable(
            getattr(train, "train_step", None)
        )
        msg = "build_model/train_step"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["model_structure:encoder_processor_decoder_projection"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn

        module = load_module(
            "graphcast_conservation_projection_submission",
            CANDIDATE_DIR / REQUIRED_FILES[0],
        )
        projection_cls = getattr(module, "GraphCastConservationProjection")
        model_cls = getattr(module, "GraphCastConservationModel")
        projection = projection_cls((0, 1))
        model = model_cls(
            2,
            2,
            hidden_channels=8,
            processor_layers=2,
            constrained_channels=(0, 1),
        )
        module_order = list(model._modules)
        processor_types_ok = all(
            block.__class__.__name__ == "ResidualGridBlock"
            for block in model.processor
        )
        ok = (
            module_order == ["encoder", "processor", "decoder", "projection"]
            and isinstance(model.encoder, nn.Conv2d)
            and isinstance(model.processor, nn.ModuleList)
            and processor_types_ok
            and isinstance(model.decoder, nn.Conv2d)
            and isinstance(model.projection, nn.Module)
        )
        msg = f"module_order={module_order}; {model}"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_executable:shape_finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn

        module = load_module(
            "graphcast_conservation_projection_submission",
            CANDIDATE_DIR / REQUIRED_FILES[0],
        )
        projection_cls = getattr(module, "GraphCastConservationProjection")
        model_cls = getattr(module, "GraphCastConservationModel")
        projection = projection_cls((0, 1))
        model = model_cls(
            2,
            2,
            hidden_channels=8,
            processor_layers=2,
            constrained_channels=(0, 1),
        )
        x = torch.randn(2, 2, 6, 8)
        area = torch.ones(6, 8)
        target_integrals = torch.tensor([[1.5, -0.5], [0.25, 0.75]])
        raw = model.forward_raw(x)
        output = projection(raw, target_integrals, area)
        ok = output.shape == raw.shape and bool(torch.isfinite(output).all())
        msg = str(tuple(output.shape))
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "forward_equivalence:conservation_oracle",
        "forward_equivalence:identity_when_conserved",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn

        module = load_module(
            "graphcast_conservation_projection_submission",
            CANDIDATE_DIR / REQUIRED_FILES[0],
        )
        projection_cls = getattr(module, "GraphCastConservationProjection")
        model_cls = getattr(module, "GraphCastConservationModel")
        projection = projection_cls((0, 1))
        model = model_cls(
            2,
            2,
            hidden_channels=8,
            processor_layers=2,
            constrained_channels=(0, 1),
        )
        x = torch.randn(2, 2, 6, 8)
        area = torch.ones(6, 8)
        target_integrals = torch.tensor([[1.5, -0.5], [0.25, 0.75]])
        raw = model.forward_raw(x)
        output = projection(raw, target_integrals, area)
        integrals = (output * area).sum(dim=(-2, -1))
        ok = bool(torch.allclose(integrals, target_integrals, atol=1e-5))
        msg = f"integrals={integrals.tolist()}"
        update_check(current_index, ok, msg)
        current_index += 1
        identity = projection(
            raw,
            (raw[:, (0, 1)] * area).sum(dim=(-2, -1)),
            area,
        )
        ok = bool(torch.allclose(identity, raw, atol=1e-5))
        msg = "zero correction is identity"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["traning:finite_loss_and_update"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn

        module = load_module(
            "graphcast_conservation_projection_submission",
            CANDIDATE_DIR / REQUIRED_FILES[0],
        )
        train = load_module(
            "graphcast_conservation_projection_train_submission",
            CANDIDATE_DIR / REQUIRED_FILES[1],
        )
        model_cls = getattr(module, "GraphCastConservationModel")
        model = model_cls(
            2,
            2,
            hidden_channels=8,
            processor_layers=2,
            constrained_channels=(0, 1),
        )
        x = torch.randn(2, 2, 6, 8)
        area = torch.ones(6, 8)
        target_integrals = torch.tensor([[1.5, -0.5], [0.25, 0.75]])
        raw = model.forward_raw(x)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]
        loss = train.train_step(
            model,
            x,
            torch.zeros_like(raw),
            target_integrals,
            area,
            optimizer,
        )
        changed = any(
            not torch.equal(a, b)
            for a, b in zip(before, model.parameters())
        )
        ok = bool(loss == loss and changed)
        msg = f"loss={loss} changed={changed}"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
