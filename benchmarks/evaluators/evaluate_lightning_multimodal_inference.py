'''V2 evaluator adapter for Lightning multimodal inference.'''
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
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:required']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        filename = 'lightning_multimodal_inference.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        ok, detail = candidate_path.is_file(), str(candidate_path)
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
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

        cls = 'LightningMultimodalModel'
        filename = 'lightning_multimodal_inference.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        reference_path = (REFERENCE_DIR / filename).resolve()
        candidate = load_module('lightning_multimodal_inference_candidate', candidate_path)
        standard = load_module('lightning_multimodal_inference_standard', reference_path)
        with tempfile.TemporaryDirectory() as td:
            good, bad = Path(td) / 'good.pt', Path(td) / 'bad.pt'
            ref = getattr(standard, cls)()
            t = torch()
            t.save({'model_state_dict': ref.state_dict()}, good)
            loaded = candidate.load_checkpoint(good, 'cpu')
            same = tuple(loaded.state_dict()) == tuple(ref.state_dict()) and all(t.equal(loaded.state_dict()[k], v) for k, v in ref.state_dict().items())
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
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'model_structure:module_order',
        'model_structure:encoder_channels',
        'model_structure:head_channels',
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

        task = 'lightning_multimodal_inference'
        cls = 'LightningMultimodalModel'
        filename = 'lightning_multimodal_inference.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate = load_module('lightning_multimodal_inference_candidate', candidate_path)
        model = getattr(candidate, cls)()
        modules = list(getattr(model, '_modules', {}))
        expected = {
            'neuralgcm_physics_coupled_inference': ['encoder', 'processor', 'decoder'],
            'satellite_cloud_property_inference': ['temporal_encoder', 'stem', 'heads'],
            'lightning_multimodal_inference': ['satellite_encoder', 'nwp_encoder', 'terrain_encoder', 'fusion', 'head'],
            'tropical_cyclone_track_intensity_inference': ['history_encoder', 'track_head', 'intensity_head', 'landfall_head'],
            'station_patchtst_forecast': ['patch_projection', 'encoder', 'head'],
            'station_tft_probabilistic_forecast': ['variable_encoder', 'time_encoder', 'temporal_attention', 'quantile_head'],
        }.get(task)
        ok = modules == expected
        detail = {'actual': modules, 'expected': expected}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        ok = model.nwp_encoder.in_channels == 3 and model.terrain_encoder.in_channels == 1
        detail = {'nwp': model.nwp_encoder.in_channels, 'terrain': model.terrain_encoder.in_channels}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        ok = model.head.out_channels == 1
        detail = {'out_channels': model.head.out_channels}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
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
        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        cls = 'LightningMultimodalModel'
        filename = 'lightning_multimodal_inference.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate = load_module('lightning_multimodal_inference_candidate', candidate_path)
        obj = getattr(candidate, cls, None)
        name = 'forward'
        ok = callable(getattr(obj, name, None))
        detail = {'class': cls, 'callable': callable(getattr(obj, name, None))}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        name = 'infer'
        ok = callable(getattr(obj, name, None))
        detail = {'class': cls, 'callable': callable(getattr(obj, name, None))}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        ok = callable(getattr(candidate, 'load_checkpoint', None))
        detail = {'callable': callable(getattr(candidate, 'load_checkpoint', None))}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'forward_executable:output_shape',
        'forward_executable:probability_finite',
        'forward_executable:probability_range',
        'forward_executable:malformed_modalities_rejected',
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

        cls = 'LightningMultimodalModel'
        filename = 'lightning_multimodal_inference.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        candidate = load_module('lightning_multimodal_inference_candidate', candidate_path)
        t = torch()

        def output():
            m = getattr(candidate, cls)().eval(); batch = {'satellite_ir': t.randn(2, 4, 8, 10), 'nwp': t.randn(2, 3, 8, 10), 'terrain': t.randn(2, 1, 8, 10)}; p = m.infer(batch)['probability']; return p.shape == (2, 8, 10), {'shape': list(p.shape), 'finite': finite(p)}

        ok, detail = output()
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        ok, detail = (lambda p: (finite(p), 'finite'))(getattr(candidate, cls)().infer({'satellite_ir': t.randn(1, 2, 4, 5), 'nwp': t.randn(1, 3, 4, 5), 'terrain': t.randn(1, 1, 4, 5)})['probability'])
        update_check(current_index, ok, detail)
        current_index += 1
        ok, detail = (lambda p: (bool((p >= 0).all() and (p <= 1).all()), 'probability in [0,1]'))(getattr(candidate, cls)().infer({'satellite_ir': t.randn(1, 2, 4, 5), 'nwp': t.randn(1, 3, 4, 5), 'terrain': t.randn(1, 1, 4, 5)})['probability'])
        update_check(current_index, ok, detail)
        current_index += 1

        def malformed():
            m = getattr(candidate, cls)(); base = {'satellite_ir': t.randn(1, 2, 4, 5), 'nwp': t.randn(1, 3, 4, 5), 'terrain': t.randn(1, 1, 4, 5)}; missing = mismatch = False
            try: m.infer({'satellite_ir': base['satellite_ir'], 'nwp': base['nwp']})
            except ValueError: missing = True
            bad = dict(base); bad['terrain'] = t.randn(1, 1, 3, 5)
            try: m.infer(bad)
            except ValueError: mismatch = True
            return missing and mismatch, {'missing_modality': missing, 'grid_mismatch': mismatch}

        ok, detail = malformed()
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_metrics(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['metric:brier_score', 'metric:accuracy']
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

        task = 'lightning_multimodal_inference'
        cls = 'LightningMultimodalModel'
        filename = 'lightning_multimodal_inference.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        reference_path = (REFERENCE_DIR / filename).resolve()
        candidate = load_module('lightning_multimodal_inference_candidate', candidate_path)
        standard = load_module('lightning_multimodal_inference_standard', reference_path)
        t = torch(); ref = getattr(standard, cls)().eval(); cand = getattr(candidate, cls)().eval(); cand.load_state_dict(ref.state_dict(), strict=True)
        batch = {'satellite_ir': t.linspace(-1, 1, 336).reshape(2, 4, 6, 7), 'nwp': t.zeros(2, 3, 6, 7), 'terrain': t.ones(2, 1, 6, 7)}; aa, bb = cand.infer(batch)['probability'], ref.infer(batch)['probability']; target = (t.arange(bb.numel(), device=bb.device).reshape_as(bb) % 2).to(bb.dtype); ref_brier = t.mean((bb.detach() - target) ** 2); brier = t.mean((aa.detach() - target) ** 2); ref_accuracy = t.mean(((bb.detach() >= .5) == (target >= .5)).to(t.float32)); accuracy = t.mean(((aa.detach() >= .5) == (target >= .5)).to(t.float32))
        ok = bool(t.abs(brier - ref_brier) <= 1e-6)
        detail = {'brier_score': float(brier), 'oracle_brier_score': float(ref_brier), 'target_source': 'fixed synthetic binary target'}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
        ok = bool(t.abs(accuracy - ref_accuracy) <= 1e-6)
        detail = {'accuracy': float(accuracy), 'oracle_accuracy': float(ref_accuracy), 'target_source': 'fixed synthetic binary target'}
        update_check(current_index, ok, json.dumps(detail, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
