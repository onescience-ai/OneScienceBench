"""V2 adapter for neuralgcm physics-coupled inference evaluation."""
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
    CHECKPOINT_NAMES = ['file:required']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        filename = config.get('required_files', ['neuralgcm_physics_coupled_inference.py'])[0]
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        ok, detail = candidate_path.is_file(), str(candidate_path)
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'checkpoint_loading:state_values',
        'checkpoint_loading:malformed_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

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

        cls = config.get('class_name', 'NeuralGCMInference')
        source_file = config.get('source_file', 'neuralgcm_physics_coupled_inference.py')
        candidate_path = (CANDIDATE_DIR / source_file).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        standard_path = (REFERENCE_DIR / source_file).resolve()
        standard_path.relative_to(REFERENCE_DIR)
        candidate = load_module('neuralgcm_physics_coupled_inference_candidate', candidate_path)
        standard = load_module('neuralgcm_physics_coupled_inference_standard', standard_path)
        with tempfile.TemporaryDirectory() as td:
            good, bad = Path(td) / 'good.pt', Path(td) / 'bad.pt'
            ref = getattr(standard, cls)()
            t = torch()
            t.save({'model_state_dict': ref.state_dict()}, good)
            loaded = candidate.load_checkpoint(good, 'cpu')
            same = tuple(loaded.state_dict()) == tuple(ref.state_dict()) and all(
                t.equal(loaded.state_dict()[k], v) for k, v in ref.state_dict().items()
            )
            t.save({}, bad)
            try:
                candidate.load_checkpoint(bad, 'cpu')
            except Exception:
                rejected = True
            else:
                rejected = False
        ok = same and rejected
        update_check(current_index, bool(ok and same), json.dumps({'state_values': same}, sort_keys=True))
        current_index += 1
        update_check(current_index, bool(ok and rejected), json.dumps({'malformed_rejected': rejected}, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'model_structure:module_order',
        'model_structure:encoder_shape',
        'model_structure:decoder_shape',
        'model_structure:processor_conv',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

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

        cls = config.get('class_name', 'NeuralGCMInference')
        source_file = config.get('source_file', 'neuralgcm_physics_coupled_inference.py')
        candidate_path = (CANDIDATE_DIR / source_file).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        standard_path = (REFERENCE_DIR / source_file).resolve()
        standard_path.relative_to(REFERENCE_DIR)
        candidate = load_module('neuralgcm_physics_coupled_inference_candidate', candidate_path)
        standard = load_module('neuralgcm_physics_coupled_inference_standard', standard_path)
        model = getattr(candidate, cls)()
        modules = list(getattr(model, '_modules', {}))
        expected = ['encoder', 'processor', 'decoder']
        ok, detail = modules == expected, {'actual': modules, 'expected': expected}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        in_channels = getattr(model.encoder, 'in_channels', None)
        ok, detail = in_channels == 4, {'in_channels': in_channels}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        out_channels = getattr(model.decoder, 'out_channels', None)
        ok, detail = out_channels == 4, {'out_channels': out_channels}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        ok, detail = any(isinstance(x, torch().nn.Conv2d) for x in model.processor.modules()), 'processor_has_conv'
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'evaluation_interface:forward',
        'evaluation_interface:rollout',
        'evaluation_interface:module_load_checkpoint',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

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

        cls = config.get('class_name', 'NeuralGCMInference')
        source_file = config.get('source_file', 'neuralgcm_physics_coupled_inference.py')
        candidate_path = (CANDIDATE_DIR / source_file).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        standard_path = (REFERENCE_DIR / source_file).resolve()
        standard_path.relative_to(REFERENCE_DIR)
        candidate = load_module('neuralgcm_physics_coupled_inference_candidate', candidate_path)
        standard = load_module('neuralgcm_physics_coupled_inference_standard', standard_path)
        obj = getattr(candidate, cls, None)
        name = 'forward'
        callable_method = callable(getattr(obj, name, None))
        ok, detail = callable_method, {'class': cls, 'callable': callable_method}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        name = 'rollout'
        callable_method = callable(getattr(obj, name, None))
        ok, detail = callable_method, {'class': cls, 'callable': callable_method}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        callable_loader = callable(getattr(candidate, 'load_checkpoint', None))
        ok, detail = callable_loader, {'callable': callable_loader}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'forward_executable:forward_shape',
        'forward_executable:forward_finite',
        'forward_executable:rollout_shape',
        'forward_executable:rollout_no_grad',
        'forward_executable:input_immutable',
        'forward_executable:forcing_effect',
        'forward_executable:invalid_steps_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

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

        cls = config.get('class_name', 'NeuralGCMInference')
        source_file = config.get('source_file', 'neuralgcm_physics_coupled_inference.py')
        candidate_path = (CANDIDATE_DIR / source_file).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        standard_path = (REFERENCE_DIR / source_file).resolve()
        standard_path.relative_to(REFERENCE_DIR)
        candidate = load_module('neuralgcm_physics_coupled_inference_candidate', candidate_path)
        standard = load_module('neuralgcm_physics_coupled_inference_standard', standard_path)
        t = torch()
        m = getattr(candidate, cls)().eval()
        x = t.randn(2, 4, 8, 12)
        original = x.clone()
        forcing = t.ones_like(x)
        one = m.forward(x)
        rollout = m.rollout(x, 3)
        zero = m.rollout(x, 2, t.zeros_like(x))
        driven = m.rollout(x, 2, forcing)
        ok, detail = one.shape == x.shape, {'shape': tuple(one.shape)}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        ok, detail = finite(getattr(candidate, cls)().forward(t.randn(2, 4, 8, 12))), 'finite'
        update_check(current_index, ok, detail)
        current_index += 1
        ok, detail = getattr(candidate, cls)().rollout(t.randn(2, 4, 8, 12), 3).shape == (3, 2, 4, 8, 12), 'shape=(3,2,4,8,12)'
        update_check(current_index, ok, detail)
        current_index += 1
        ok, detail = not getattr(candidate, cls)().rollout(t.randn(1, 4, 4, 4), 2).requires_grad, 'requires_grad=False'
        update_check(current_index, ok, detail)
        current_index += 1
        m = getattr(candidate, cls)().eval()
        x = t.randn(1, 4, 4, 4)
        y = x.clone()
        m.rollout(x, 2)
        ok, detail = t.equal(x, y), 'input unchanged'
        update_check(current_index, ok, detail)
        current_index += 1
        m = getattr(candidate, cls)().eval()
        state = t.zeros(1, 4, 4, 4)
        zero = t.zeros_like(state)
        one = t.ones_like(state)
        a, b = m.rollout(state, 2, zero), m.rollout(state, 2, one)
        ok, detail = not t.allclose(a, b), 'forcing changes rollout'
        update_check(current_index, ok, detail)
        current_index += 1
        try:
            getattr(candidate, cls)().rollout(t.randn(1, 4, 4, 4), 0)
        except ValueError:
            ok, detail = True, 'steps=0 rejected'
        else:
            ok, detail = False, 'steps=0 accepted'
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_metrics(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'metric:rollout_rmse',
        'metric:rollout_mae',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

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

        cls = config.get('class_name', 'NeuralGCMInference')
        source_file = config.get('source_file', 'neuralgcm_physics_coupled_inference.py')
        candidate_path = (CANDIDATE_DIR / source_file).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        standard_path = (REFERENCE_DIR / source_file).resolve()
        standard_path.relative_to(REFERENCE_DIR)
        candidate = load_module('neuralgcm_physics_coupled_inference_candidate', candidate_path)
        standard = load_module('neuralgcm_physics_coupled_inference_standard', standard_path)
        t = torch()
        ref = getattr(standard, cls)().eval()
        cand = getattr(candidate, cls)().eval()
        cand.load_state_dict(ref.state_dict(), strict=True)
        x = t.linspace(-1, 1, 384).reshape(1, 4, 8, 12)
        a, b = cand.rollout(x, 3), ref.rollout(x, 3)
        errors = {'rollout': a - b}
        error = errors['rollout']
        e = error.detach()
        rmse = float(t.sqrt(t.mean(e.square())))
        mae = float(t.mean(e.abs()))
        ok, detail = rmse <= 1e-6, {'rmse': rmse, 'atol': 1e-6}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        ok, detail = mae <= 1e-6, {'mae': mae, 'atol': 1e-6}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
