"""V2 adapter for tropical cyclone track and intensity inference."""
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
        filename = config.get('model_file', 'tropical_cyclone_track_intensity_inference.py')
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        ok = candidate_path.is_file()
        update_check(current_index, ok, str(candidate_path))
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
        def torch():
            import torch as _torch
            return _torch

        filename = config.get('model_file', 'tropical_cyclone_track_intensity_inference.py')
        cls = config.get('model_class', 'TropicalCycloneTrackIntensityModel')
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        candidate_spec = importlib.util.spec_from_file_location(
            'tropical_cyclone_track_intensity_inference_candidate', str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules['tropical_cyclone_track_intensity_inference_candidate'] = candidate
        candidate_spec.loader.exec_module(candidate)
        standard_path = (REFERENCE_DIR / filename).resolve()
        standard_path.relative_to(REFERENCE_DIR)
        standard_spec = importlib.util.spec_from_file_location(
            'tropical_cyclone_track_intensity_inference_standard', str(standard_path)
        )
        if standard_spec is None or standard_spec.loader is None:
            raise ImportError(str(standard_path))
        standard = importlib.util.module_from_spec(standard_spec)
        sys.modules['tropical_cyclone_track_intensity_inference_standard'] = standard
        standard_spec.loader.exec_module(standard)
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
        update_check(
            current_index,
            bool(ok and same),
            json.dumps({'state_values': same}, sort_keys=True),
        )
        current_index += 1
        update_check(
            current_index,
            bool(ok and rejected),
            json.dumps({'malformed_rejected': rejected}, sort_keys=True),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'model_structure:module_order',
        'model_structure:track_head_shape',
        'model_structure:intensity_head_shape',
        'model_structure:landfall_head_shape',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        filename = config.get('model_file', 'tropical_cyclone_track_intensity_inference.py')
        cls = config.get('model_class', 'TropicalCycloneTrackIntensityModel')
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        candidate_spec = importlib.util.spec_from_file_location(
            'tropical_cyclone_track_intensity_inference_candidate', str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules['tropical_cyclone_track_intensity_inference_candidate'] = candidate
        candidate_spec.loader.exec_module(candidate)
        model = getattr(candidate, cls)()
        modules = list(getattr(model, '_modules', {}))
        expected = ['history_encoder', 'track_head', 'intensity_head', 'landfall_head']
        update_check(
            current_index,
            modules == expected,
            json.dumps({'actual': modules, 'expected': expected}, sort_keys=True),
        )
        current_index += 1
        update_check(
            current_index,
            model.track_head.out_features == 8,
            json.dumps({'out_features': model.track_head.out_features}, sort_keys=True),
        )
        current_index += 1
        update_check(
            current_index,
            model.intensity_head.out_features == 8,
            json.dumps({'out_features': model.intensity_head.out_features}, sort_keys=True),
        )
        current_index += 1
        update_check(
            current_index,
            model.landfall_head.out_features == 4,
            json.dumps({'out_features': model.landfall_head.out_features}, sort_keys=True),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'evaluation_interface:forward',
        'evaluation_interface:infer',
        'evaluation_interface:module_load_checkpoint',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        filename = config.get('model_file', 'tropical_cyclone_track_intensity_inference.py')
        cls = config.get('model_class', 'TropicalCycloneTrackIntensityModel')
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        candidate_spec = importlib.util.spec_from_file_location(
            'tropical_cyclone_track_intensity_inference_candidate', str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules['tropical_cyclone_track_intensity_inference_candidate'] = candidate
        candidate_spec.loader.exec_module(candidate)
        names = ('forward', 'infer')
        obj = getattr(candidate, cls, None)
        name = names[0]
        callable_method = callable(getattr(obj, name, None))
        update_check(
            current_index,
            callable_method,
            json.dumps({'class': cls, 'callable': callable_method}, sort_keys=True),
        )
        current_index += 1
        name = names[1]
        callable_method = callable(getattr(obj, name, None))
        update_check(
            current_index,
            callable_method,
            json.dumps({'class': cls, 'callable': callable_method}, sort_keys=True),
        )
        current_index += 1
        callable_function = callable(getattr(candidate, 'load_checkpoint', None))
        update_check(
            current_index,
            callable_function,
            json.dumps({'callable': callable_function}, sort_keys=True),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'forward_executable:output_shapes',
        'forward_executable:physical_ranges',
        'forward_executable:short_history_rejected',
        'forward_executable:input_immutable',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def torch():
            import torch as _torch
            return _torch

        def finite(x):
            return bool(np.isfinite(x).all()) if isinstance(x, np.ndarray) else bool(torch().isfinite(x).all())

        filename = config.get('model_file', 'tropical_cyclone_track_intensity_inference.py')
        cls = config.get('model_class', 'TropicalCycloneTrackIntensityModel')
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        candidate_spec = importlib.util.spec_from_file_location(
            'tropical_cyclone_track_intensity_inference_candidate', str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules['tropical_cyclone_track_intensity_inference_candidate'] = candidate
        candidate_spec.loader.exec_module(candidate)
        t = torch()
        m = getattr(candidate, cls)().eval()
        x = t.randn(2, 6, 4)
        out = m.infer(x)
        shapes = {k: tuple(v.shape) for k, v in out.items()}
        ok = shapes == {
            'track': (2, 4, 2),
            'central_pressure': (2, 4),
            'max_wind': (2, 4),
            'landfall_probability': (2, 4),
        }
        update_check(current_index, ok, json.dumps({'shapes': shapes}, sort_keys=True))
        current_index += 1
        out = getattr(candidate, cls)().infer(t.randn(2, 6, 4))
        track, p = out['track'], out['landfall_probability']
        ok = finite(track) and finite(out['central_pressure']) and finite(out['max_wind']) and finite(p) and bool(
            (track[..., 0] >= -90).all()
            and (track[..., 0] <= 90).all()
            and (track[..., 1] >= -180).all()
            and (track[..., 1] < 180).all()
            and (p >= 0).all()
            and (p <= 1).all()
        )
        update_check(current_index, ok, 'all physical ranges finite')
        current_index += 1
        m = getattr(candidate, cls)()
        short = False
        try:
            m.forward(t.randn(1, 1, 4))
        except ValueError:
            short = True
        update_check(
            current_index,
            short,
            json.dumps({'short_history_rejected': short}, sort_keys=True),
        )
        current_index += 1
        x = t.randn(1, 6, 4)
        y = x.clone()
        m.infer(x)
        bad = t.equal(x, y)
        update_check(
            current_index,
            bad,
            json.dumps({'input_immutable': bad}, sort_keys=True),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'metric:landfall_brier_score',
        'metric:track_rmse',
        'metric:track_mae',
        'metric:central_pressure_rmse',
        'metric:central_pressure_mae',
        'metric:max_wind_rmse',
        'metric:max_wind_mae',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def torch():
            import torch as _torch
            return _torch

        filename = config.get('model_file', 'tropical_cyclone_track_intensity_inference.py')
        cls = config.get('model_class', 'TropicalCycloneTrackIntensityModel')
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        candidate_spec = importlib.util.spec_from_file_location(
            'tropical_cyclone_track_intensity_inference_candidate', str(candidate_path)
        )
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules['tropical_cyclone_track_intensity_inference_candidate'] = candidate
        candidate_spec.loader.exec_module(candidate)
        standard_path = (REFERENCE_DIR / filename).resolve()
        standard_path.relative_to(REFERENCE_DIR)
        standard_spec = importlib.util.spec_from_file_location(
            'tropical_cyclone_track_intensity_inference_standard', str(standard_path)
        )
        if standard_spec is None or standard_spec.loader is None:
            raise ImportError(str(standard_path))
        standard = importlib.util.module_from_spec(standard_spec)
        sys.modules['tropical_cyclone_track_intensity_inference_standard'] = standard
        standard_spec.loader.exec_module(standard)
        t = torch()
        ref = getattr(standard, cls)().eval()
        cand = getattr(candidate, cls)().eval()
        cand.load_state_dict(ref.state_dict(), strict=True)
        x = t.linspace(-1, 1, 48).reshape(2, 6, 4)
        aa, bb = cand.infer(x), ref.infer(x)
        errors = {k: aa[k] - bb[k] for k in ('track', 'central_pressure', 'max_wind')}
        target = t.zeros_like(bb['landfall_probability'].detach())
        brier = t.mean((aa['landfall_probability'].detach() - target) ** 2)
        ref_brier = t.mean((bb['landfall_probability'].detach() - target) ** 2)
        ok = bool(t.abs(brier - ref_brier) <= 1e-6)
        update_check(
            current_index,
            ok,
            json.dumps(
                {
                    'brier_score': float(brier),
                    'oracle_brier_score': float(ref_brier),
                    'target_source': 'fixed synthetic binary target',
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        for name, error in errors.items():
            e = error.detach()
            rmse = float(t.sqrt(t.mean(e.square())))
            mae = float(t.mean(e.abs()))
            update_check(
                current_index,
                rmse <= 1e-6,
                json.dumps({'rmse': rmse, 'atol': 1e-6}, sort_keys=True),
            )
            current_index += 1
            update_check(
                current_index,
                mae <= 1e-6,
                json.dumps({'mae': mae, 'atol': 1e-6}, sort_keys=True),
            )
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
