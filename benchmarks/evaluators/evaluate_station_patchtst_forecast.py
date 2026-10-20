"""V2 adapter for station PatchTST forecast evaluation."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict

import numpy as np

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


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["file:required"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        task = "station_patchtst_forecast"
        filename = "station_patchtst_forecast.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        ok = candidate_path.is_file()
        msg = str(candidate_path)
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "checkpoint_loading:state_values",
        "checkpoint_loading:malformed_rejected",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch as t

        task = "station_patchtst_forecast"
        cls = "PatchTSTForecastModel"
        filename = "station_patchtst_forecast.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        reference_path = (REFERENCE_DIR / filename).resolve()
        candidate_spec = importlib.util.spec_from_file_location(
            task + "_candidate", str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules[task + "_candidate"] = candidate
        candidate_spec.loader.exec_module(candidate)
        standard_spec = importlib.util.spec_from_file_location(
            task + "_standard", str(reference_path)
        )
        if standard_spec is None or standard_spec.loader is None:
            raise ImportError(str(reference_path))
        standard = importlib.util.module_from_spec(standard_spec)
        sys.modules[task + "_standard"] = standard
        standard_spec.loader.exec_module(standard)
        with tempfile.TemporaryDirectory() as td:
            good, bad = Path(td) / "good.pt", Path(td) / "bad.pt"
            ref = getattr(standard, cls)()
            t.save({"model_state_dict": ref.state_dict()}, good)
            loaded = candidate.load_checkpoint(good, "cpu")
            same = tuple(loaded.state_dict()) == tuple(ref.state_dict()) and all(
                t.equal(loaded.state_dict()[k], v)
                for k, v in ref.state_dict().items()
            )
            t.save({}, bad)
            try:
                candidate.load_checkpoint(bad, "cpu")
            except Exception:
                rejected = True
            else:
                rejected = False
        detail = {"state_values": same, "malformed_rejected": rejected}
        ok = same and rejected
        msg = json.dumps({"state_values": detail["state_values"]}, sort_keys=True)
        update_check(current_index, bool(ok and detail["state_values"]), msg)
        current_index += 1
        msg = json.dumps(
            {"malformed_rejected": detail["malformed_rejected"]}, sort_keys=True
        )
        update_check(current_index, bool(ok and detail["malformed_rejected"]), msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "model_structure:module_order",
        "model_structure:patch_shape",
        "model_structure:forecast_head",
        "model_structure:encoder_layers",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        task = "station_patchtst_forecast"
        cls = "PatchTSTForecastModel"
        filename = "station_patchtst_forecast.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_spec = importlib.util.spec_from_file_location(
            task + "_candidate", str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules[task + "_candidate"] = candidate
        candidate_spec.loader.exec_module(candidate)
        model = getattr(candidate, cls)()
        modules = list(getattr(model, "_modules", {}))
        expected = ["patch_projection", "encoder", "head"]
        ok = modules == expected
        msg = json.dumps({"actual": modules, "expected": expected}, sort_keys=True)
        update_check(current_index, ok, msg)
        current_index += 1
        ok = model.patch_projection.in_features == 4
        msg = json.dumps(
            {"in_features": model.patch_projection.in_features}, sort_keys=True
        )
        update_check(current_index, ok, msg)
        current_index += 1
        ok = model.head.out_features == 6
        msg = json.dumps({"out_features": model.head.out_features}, sort_keys=True)
        update_check(current_index, ok, msg)
        current_index += 1
        ok = len(model.encoder.layers) == 2
        msg = json.dumps({"layers": len(model.encoder.layers)}, sort_keys=True)
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "evaluation_interface:predict",
        "evaluation_interface:module_load_checkpoint",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        task = "station_patchtst_forecast"
        cls = "PatchTSTForecastModel"
        filename = "station_patchtst_forecast.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_spec = importlib.util.spec_from_file_location(
            task + "_candidate", str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules[task + "_candidate"] = candidate
        candidate_spec.loader.exec_module(candidate)
        names = ("predict",)
        obj = getattr(candidate, cls, None)
        name = names[0]
        ok = callable(getattr(obj, name, None))
        msg = json.dumps({"class": cls, "callable": ok}, sort_keys=True)
        update_check(current_index, ok, msg)
        current_index += 1
        ok = callable(getattr(candidate, "load_checkpoint", None))
        msg = json.dumps({"callable": ok}, sort_keys=True)
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "forward_executable:output_shape",
        "forward_executable:nonoverlapping_patch_contract",
        "forward_executable:input_shape_rejected",
        "forward_executable:input_immutable",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch as t

        task = "station_patchtst_forecast"
        cls = "PatchTSTForecastModel"
        filename = "station_patchtst_forecast.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_spec = importlib.util.spec_from_file_location(
            task + "_candidate", str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules[task + "_candidate"] = candidate
        candidate_spec.loader.exec_module(candidate)
        m = getattr(candidate, cls)().eval()
        x = t.randn(3, 12, 4)
        y = m.predict(x)
        ok = y.shape == (3, 6, 4)
        msg = json.dumps(
            {"shape": tuple(y.shape), "finite": bool(t.isfinite(y).all())},
            sort_keys=True,
        )
        update_check(current_index, ok, msg)
        current_index += 1
        m = getattr(candidate, cls)()
        ok = (
            m.context_length % m.patch_length == 0
            and m.patch_projection.in_features == m.patch_length
        )
        msg = json.dumps(
            {
                "context_length": m.context_length,
                "patch_length": m.patch_length,
                "nonoverlap_possible": m.context_length % m.patch_length == 0,
            },
            sort_keys=True,
        )
        update_check(current_index, ok, msg)
        current_index += 1
        m = getattr(candidate, cls)()
        wrong_context = wrong_vars = False
        try:
            m.predict(t.randn(1, 11, 4))
        except ValueError:
            wrong_context = True
        try:
            m.predict(t.randn(1, 12, 3))
        except ValueError:
            wrong_vars = True
        ok = wrong_context and wrong_vars
        msg = json.dumps(
            {"context": wrong_context, "variables": wrong_vars}, sort_keys=True
        )
        update_check(current_index, ok, msg)
        current_index += 1
        m = getattr(candidate, cls)()
        x = t.randn(1, 12, 4)
        y = x.clone()
        m.predict(x)
        ok = t.equal(x, y)
        msg = "input unchanged"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["metric:forecast_rmse", "metric:forecast_mae"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch as t

        task = "station_patchtst_forecast"
        cls = "PatchTSTForecastModel"
        filename = "station_patchtst_forecast.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        reference_path = (REFERENCE_DIR / filename).resolve()
        candidate_spec = importlib.util.spec_from_file_location(
            task + "_candidate", str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules[task + "_candidate"] = candidate
        candidate_spec.loader.exec_module(candidate)
        standard_spec = importlib.util.spec_from_file_location(
            task + "_standard", str(reference_path)
        )
        if standard_spec is None or standard_spec.loader is None:
            raise ImportError(str(reference_path))
        standard = importlib.util.module_from_spec(standard_spec)
        sys.modules[task + "_standard"] = standard
        standard_spec.loader.exec_module(standard)
        ref = getattr(standard, cls)().eval()
        cand = getattr(candidate, cls)().eval()
        cand.load_state_dict(ref.state_dict(), strict=True)
        x = t.linspace(-1, 1, 96).reshape(2, 12, 4)
        errors = {"forecast": cand.predict(x) - ref.predict(x)}
        error = errors["forecast"]
        e = error.detach()
        rmse = float(t.sqrt(t.mean(e.square())))
        mae = float(t.mean(e.abs()))
        ok = rmse <= 1e-6
        msg = json.dumps({"rmse": rmse, "atol": 1e-6}, sort_keys=True)
        update_check(current_index, ok, msg)
        current_index += 1
        ok = mae <= 1e-6
        msg = json.dumps({"mae": mae, "atol": 1e-6}, sort_keys=True)
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
