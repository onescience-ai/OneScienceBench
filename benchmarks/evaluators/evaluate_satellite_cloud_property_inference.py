"""V2 adapter for satellite cloud property inference evaluation."""
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
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["file:required"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        task = config.get("task", "satellite_cloud_property_inference")
        filename = "satellite_cloud_property_inference.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        ok, detail = candidate_path.is_file(), str(candidate_path)
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "checkpoint_loading:state_values",
        "checkpoint_loading:malformed_rejected",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        def torch():
            import torch as _torch
            return _torch

        task = config.get("task", "satellite_cloud_property_inference")
        cls = "SatelliteCloudPropertyModel"
        filename = "satellite_cloud_property_inference.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        standard_path = (REFERENCE_DIR / filename).resolve()
        candidate = load_module(task + "_candidate", candidate_path)
        standard = load_module(task + "_standard", standard_path)
        with tempfile.TemporaryDirectory() as td:
            good, bad = Path(td) / "good.pt", Path(td) / "bad.pt"
            ref = getattr(standard, cls)()
            t = torch()
            t.save({"model_state_dict": ref.state_dict()}, good)
            loaded = candidate.load_checkpoint(good, "cpu")
            same = tuple(loaded.state_dict()) == tuple(ref.state_dict()) and all(
                t.equal(loaded.state_dict()[k], v) for k, v in ref.state_dict().items()
            )
            t.save({}, bad)
            try:
                candidate.load_checkpoint(bad, "cpu")
            except Exception:
                rejected = True
            else:
                rejected = False
        ok = same and rejected
        detail = {"state_values": same}
        update_check(current_index, bool(ok and same), json.dumps(detail, sort_keys=True))
        current_index += 1
        detail = {"malformed_rejected": rejected}
        update_check(current_index, bool(ok and rejected), json.dumps(detail, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "model_structure:module_order",
        "model_structure:head_order",
        "model_structure:head_shapes",
        "model_structure:temporal_input",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        task = config.get("task", "satellite_cloud_property_inference")
        cls = "SatelliteCloudPropertyModel"
        filename = "satellite_cloud_property_inference.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate = load_module(task + "_candidate", candidate_path)
        model = getattr(candidate, cls)()
        modules = list(getattr(model, "_modules", {}))
        expected = ["temporal_encoder", "stem", "heads"]
        ok, detail = modules == expected, {"actual": modules, "expected": expected}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        heads = list(model.heads)
        ok, detail = heads == ["cloud_top_temperature", "cloud_type_logits", "precipitation_probability"], {"heads": heads}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        out_channels = [model.heads[x].out_channels for x in heads]
        ok, detail = out_channels == [1, 4, 1], {"out_channels": out_channels}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        in_channels = getattr(model.temporal_encoder, "in_channels", None)
        ok, detail = in_channels == 5, {"in_channels": in_channels}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "evaluation_interface:forward",
        "evaluation_interface:infer",
        "evaluation_interface:module_load_checkpoint",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        task = config.get("task", "satellite_cloud_property_inference")
        cls = "SatelliteCloudPropertyModel"
        filename = "satellite_cloud_property_inference.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate = load_module(task + "_candidate", candidate_path)
        obj = getattr(candidate, cls, None)
        name = "forward"
        ok, detail = callable(getattr(obj, name, None)), {"class": cls, "callable": callable(getattr(obj, name, None))}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        name = "infer"
        ok, detail = callable(getattr(obj, name, None)), {"class": cls, "callable": callable(getattr(obj, name, None))}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        ok, detail = callable(getattr(candidate, "load_checkpoint", None)), {"callable": callable(getattr(candidate, "load_checkpoint", None))}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "forward_executable:output_shapes",
        "forward_executable:outputs_finite",
        "forward_executable:probability_range",
        "forward_executable:coordinates_preserved",
        "forward_executable:invalid_mask_rejected",
        "forward_executable:invalid_spectral_rejected",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        def torch():
            import torch as _torch
            return _torch

        def finite(x):
            return bool(np.isfinite(x).all()) if isinstance(x, np.ndarray) else bool(torch().isfinite(x).all())

        task = config.get("task", "satellite_cloud_property_inference")
        cls = "SatelliteCloudPropertyModel"
        filename = "satellite_cloud_property_inference.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate = load_module(task + "_candidate", candidate_path)
        t = torch()

        def outputs():
            m = getattr(candidate, cls)().eval()
            spectral = t.randn(2, 3, 5, 8, 10)
            mask = t.tensor([[1, 1, 0], [1, 0, 0]], dtype=t.bool)
            lat, lon = t.arange(8), t.arange(10)
            out = m.infer({"spectral": spectral, "latitude": lat, "longitude": lon, "valid_mask": mask})
            ok = out.keys() == {"cloud_top_temperature", "cloud_type_logits", "precipitation_probability", "latitude", "longitude"}
            ok = ok and out["cloud_top_temperature"].shape == (2, 8, 10) and out["cloud_type_logits"].shape == (2, 8, 10, 4) and out["precipitation_probability"].shape == (2, 8, 10)
            return ok, {"keys": sorted(out), "shapes": {k: list(v.shape) for k, v in out.items() if hasattr(v, "shape")}}

        ok, detail = outputs()
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1

        def output_finite():
            out = getattr(candidate, cls)().infer({"spectral": t.randn(1, 2, 5, 4, 5), "latitude": t.arange(4), "longitude": t.arange(5)})
            tensors = [value for value in out.values() if isinstance(value, t.Tensor)]
            return all(finite(value) for value in tensors), "all tensor outputs finite"

        ok, detail = output_finite()
        update_check(current_index, ok, detail)
        current_index += 1
        ok, detail = (lambda p: (bool((p >= 0).all() and (p <= 1).all()), "probability in [0,1]"))(getattr(candidate, cls)().infer({"spectral": t.randn(1, 2, 5, 4, 5), "latitude": t.arange(4), "longitude": t.arange(5)})["precipitation_probability"])
        update_check(current_index, ok, detail)
        current_index += 1
        ok, detail = (lambda m, lat, lon, o: (t.equal(o["latitude"], lat) and t.equal(o["longitude"], lon), "coordinates unchanged"))(getattr(candidate, cls)(), t.arange(4), t.arange(5), getattr(candidate, cls)().infer({"spectral": t.randn(1, 2, 5, 4, 5), "latitude": t.arange(4), "longitude": t.arange(5)}))
        update_check(current_index, ok, detail)
        current_index += 1

        def invalid_mask():
            try:
                getattr(candidate, cls)().forward(t.randn(1, 2, 5, 4, 5), t.zeros(1, 2, dtype=t.bool))
            except ValueError:
                return True, "all-invalid mask rejected"
            return False, "all-invalid mask accepted"

        ok, detail = invalid_mask()
        update_check(current_index, ok, detail)
        current_index += 1

        def invalid_bands():
            try:
                getattr(candidate, cls)().forward(t.randn(1, 2, 4, 4, 5))
            except ValueError:
                return True, "wrong spectral channels rejected"
            return False, "wrong spectral channels accepted"

        ok, detail = invalid_bands()
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "metric:cloud_top_temperature_rmse",
        "metric:cloud_top_temperature_mae",
        "metric:cloud_type_logits_rmse",
        "metric:cloud_type_logits_mae",
        "metric:precipitation_probability_rmse",
        "metric:precipitation_probability_mae",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        def torch():
            import torch as _torch
            return _torch

        task = config.get("task", "satellite_cloud_property_inference")
        cls = "SatelliteCloudPropertyModel"
        filename = "satellite_cloud_property_inference.py"
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        standard_path = (REFERENCE_DIR / filename).resolve()
        candidate = load_module(task + "_candidate", candidate_path)
        standard = load_module(task + "_standard", standard_path)
        t = torch()
        ref = getattr(standard, cls)().eval()
        cand = getattr(candidate, cls)().eval()
        cand.load_state_dict(ref.state_dict(), strict=True)
        x = t.linspace(0, 1, 420).reshape(1, 2, 5, 6, 7)
        batch = {"spectral": x, "latitude": t.arange(6), "longitude": t.arange(7)}
        aa, bb = cand.infer(batch), ref.infer(batch)
        errors = {k: aa[k] - bb[k] for k in ("cloud_top_temperature", "cloud_type_logits", "precipitation_probability")}
        for name, error in errors.items():
            e = error.detach()
            rmse = float(t.sqrt(t.mean(e.square())))
            mae = float(t.mean(e.abs()))
            ok, detail = rmse <= 1e-6, {"rmse": rmse, "atol": 1e-6}
            update_check(current_index, ok, json.dumps(detail, sort_keys=True))
            current_index += 1
            ok, detail = mae <= 1e-6, {"mae": mae, "atol": 1e-6}
            update_check(current_index, ok, json.dumps(detail, sort_keys=True))
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
