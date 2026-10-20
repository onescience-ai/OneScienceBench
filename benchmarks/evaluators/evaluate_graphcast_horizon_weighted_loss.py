"""V2 evaluator adapter for the GraphCast horizon weighted loss task."""
from __future__ import annotations

from pathlib import Path
import sys
import importlib.util
import json
import os
import traceback
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


CANDIDATE_DIR = _candidate_model_path()
REFERENCE_DIR = _reference_model_path()

REQUIRED_FILES = ("graphcast_horizon_weighted_loss.py", "train.py")
ENV_NAME = "GRAPHCAST_HORIZON_WEIGHTED_LOSS_SUBMISSION_DIR"
HINTS = ("graphcast_horizon_weighted_loss", "graphcast_horizon_weighted_loss_V1")
ALLOWED = {"file", "data", "train_script", "model_structure", "forward_executable", "forward_equivalence", "traning"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["data:weighted_batch_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        predictions = torch.zeros(2, 3, 2, 3, 4, requires_grad=True)
        targets = torch.ones_like(predictions)
        latitude = torch.tensor([1.0, 2.0, 1.0])
        update_check(current_index, predictions.shape == targets.shape and latitude.shape == (3,), "[B,T,C,H,W] and latitude weights")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["file:graphcast_horizon_weighted_loss.py", "file:train.py"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        for name in REQUIRED_FILES:
            submission_path = CANDIDATE_DIR / name
            update_check(current_index, submission_path.is_file(), str(submission_path))
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
            "graphcast_horizon_weighted_loss_train_submission",
            CANDIDATE_DIR / REQUIRED_FILES[1],
        )
        update_check(
            current_index,
            callable(getattr(train, "build_loss", None)) and callable(getattr(train, "train_step", None)),
            "build_loss/train_step",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["model_structure:is_loss_module"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn
        module = load_module(
            "graphcast_horizon_weighted_loss_submission",
            CANDIDATE_DIR / REQUIRED_FILES[0],
        )
        cls = getattr(module, "GraphCastHorizonWeightedLoss")
        loss_fn = cls([1.0, 2.0, 4.0], [1.0, 0.5])
        update_check(current_index, isinstance(loss_fn, nn.Module), str(type(loss_fn)))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_executable:scalar_finite", "forward_executable:invalid_horizon_rejected"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        module = load_module(
            "graphcast_horizon_weighted_loss_submission",
            CANDIDATE_DIR / REQUIRED_FILES[0],
        )
        cls = getattr(module, "GraphCastHorizonWeightedLoss")
        loss_fn = cls([1.0, 2.0, 4.0], [1.0, 0.5])
        predictions = torch.zeros(2, 3, 2, 3, 4, requires_grad=True)
        targets = torch.ones_like(predictions)
        latitude = torch.tensor([1.0, 2.0, 1.0])
        actual = loss_fn(predictions, targets, latitude)
        update_check(current_index, actual.ndim == 0 and bool(torch.isfinite(actual)), str(float(actual.detach())))
        current_index += 1
        try:
            loss_fn(torch.zeros(1, 2, 2, 2, 2), torch.zeros(1, 2, 2, 2, 2), torch.ones(2))
            update_check(current_index, False, "invalid horizon was accepted")
            current_index += 1
        except ValueError:
            update_check(current_index, True, "ValueError")
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_equivalence:weighted_mse_oracle"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        module = load_module(
            "graphcast_horizon_weighted_loss_submission",
            CANDIDATE_DIR / REQUIRED_FILES[0],
        )
        cls = getattr(module, "GraphCastHorizonWeightedLoss")
        loss_fn = cls([1.0, 2.0, 4.0], [1.0, 0.5])
        predictions = torch.zeros(2, 3, 2, 3, 4)
        targets = torch.ones_like(predictions)
        latitude = torch.tensor([1.0, 2.0, 1.0])
        oracle_weights = torch.tensor([1., 2., 4.]).view(1, 3, 1, 1, 1) * torch.tensor([1., .5]).view(1, 1, 2, 1, 1) * latitude.view(1, 1, 1, 3, 1).expand(1, 1, 1, 3, 4)
        expected = (((predictions - targets).square() * oracle_weights).sum(dim=(1, 2, 3, 4)) / oracle_weights.sum()).mean()
        actual = loss_fn(predictions, targets, latitude)
        update_check(current_index, bool(torch.allclose(actual, expected, atol=1e-6)), f"actual={float(actual.detach()):.8f} expected={float(expected.detach()):.8f}")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["traning:backward_has_gradient", "traning:optimizer_step_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        module = load_module(
            "graphcast_horizon_weighted_loss_submission",
            CANDIDATE_DIR / REQUIRED_FILES[0],
        )
        train = load_module(
            "graphcast_horizon_weighted_loss_train_submission",
            CANDIDATE_DIR / REQUIRED_FILES[1],
        )
        cls = getattr(module, "GraphCastHorizonWeightedLoss")
        loss_fn = cls([1.0, 2.0, 4.0], [1.0, 0.5])
        predictions = torch.zeros(2, 3, 2, 3, 4, requires_grad=True)
        targets = torch.ones_like(predictions)
        latitude = torch.tensor([1.0, 2.0, 1.0])
        actual = loss_fn(predictions, targets, latitude)
        actual.backward()
        update_check(current_index, predictions.grad is not None and bool(torch.isfinite(predictions.grad).all()), "gradient is finite")
        current_index += 1
        optimizer = torch.optim.AdamW([predictions], lr=1e-3)
        before = predictions.detach().clone()
        train_loss = train.train_step(loss_fn, predictions, targets, optimizer, latitude)
        changed = not torch.equal(before, predictions.detach())
        update_check(current_index, bool(torch.isfinite(torch.as_tensor(train_loss))) and changed, f"loss={float(train_loss):.8f} changed={changed}")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
