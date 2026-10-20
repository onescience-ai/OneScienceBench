"""V2 evaluator adapter for neural_enkf_analysis_update."""
from pathlib import Path
import sys
import importlib.util
import json
import os
import tempfile
import traceback
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


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check checkpoint loading behavior."""
    CHECKPOINT_NAMES = [
        'checkpoint_loading:state_values',
        'checkpoint_loading:unexpected_key_rejected',
        'checkpoint_loading:bad_shape_rejected',
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

        cls = 'NeuralEnKFAnalyzer'
        filename = 'neural_enkf_analysis_update.py'
        candidate = load_module(
            'neural_enkf_analysis_update_candidate', CANDIDATE_DIR / filename
        )
        standard = load_module(
            'neural_enkf_analysis_update_standard', REFERENCE_DIR / filename
        )
        with tempfile.TemporaryDirectory() as td:
            good, bad_key, bad_shape = Path(td) / 'good.npz', Path(td) / 'bad_key.npz', Path(td) / 'bad_shape.npz'
            ref = standard.NeuralEnKFAnalyzer()
            np.savez(good, gain=ref.gain, observation_matrix=ref.observation_matrix)
            np.savez(bad_key, gain=ref.gain)
            np.savez(bad_shape, gain=np.zeros((1, 1)), observation_matrix=ref.observation_matrix)
            loaded = candidate.NeuralEnKFAnalyzer().load_checkpoint(good)
            same = np.array_equal(loaded.gain, ref.gain) and np.array_equal(loaded.observation_matrix, ref.observation_matrix)
            rejected_key = rejected_shape = False
            try:
                candidate.NeuralEnKFAnalyzer().load_checkpoint(bad_key)
            except Exception:
                rejected_key = True
            try:
                candidate.NeuralEnKFAnalyzer().load_checkpoint(bad_shape)
            except Exception:
                rejected_shape = True
        ok = same and rejected_key and rejected_shape
        update_check(current_index, bool(ok and same), json.dumps({'state_values': same}, sort_keys=True))
        current_index += 1
        update_check(current_index, bool(ok and rejected_key), json.dumps({'unexpected_key_rejected': rejected_key}, sort_keys=True))
        current_index += 1
        update_check(current_index, bool(ok and rejected_shape), json.dumps({'bad_shape_rejected': rejected_shape}, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check numerical equivalence against the reference analyzer."""
    CHECKPOINT_NAMES = ['metric:analysis_rmse', 'metric:analysis_mae']
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

        cls = 'NeuralEnKFAnalyzer'
        filename = 'neural_enkf_analysis_update.py'
        candidate = load_module(
            'neural_enkf_analysis_update_candidate', CANDIDATE_DIR / filename
        )
        standard = load_module(
            'neural_enkf_analysis_update_standard', REFERENCE_DIR / filename
        )
        p = np.arange(60, dtype=float).reshape(3, 5, 4) / 10
        o = np.array([[1., 2.], [3., 4.], [5., 6.]])
        a = candidate.NeuralEnKFAnalyzer().analyze(p, o)
        b = standard.NeuralEnKFAnalyzer().analyze(p, o)
        errs = {'analysis': a - b}
        details = {
            k: {
                'rmse': float(np.sqrt(np.mean(v * v))),
                'mae': float(np.mean(np.abs(v))),
                'max_abs_error': float(np.max(np.abs(v))),
            }
            for k, v in errs.items()
        }['analysis']
        update_check(
            current_index,
            bool(details['max_abs_error'] <= 1e-10),
            json.dumps(
                {
                    'rmse': details['rmse'],
                    'max_abs_error': details['max_abs_error'],
                    'atol': 1e-10,
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        update_check(
            current_index,
            bool(details['max_abs_error'] <= 1e-10),
            json.dumps(
                {
                    'mae': details['mae'],
                    'max_abs_error': details['max_abs_error'],
                    'atol': 1e-10,
                },
                sort_keys=True,
            ),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the required candidate source file."""
    CHECKPOINT_NAMES = ['file:required']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        filename = 'neural_enkf_analysis_update.py'
        ok = (CANDIDATE_DIR / filename).is_file()
        detail = str(CANDIDATE_DIR / filename)
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the analyzer interface."""
    CHECKPOINT_NAMES = [
        'evaluation_interface:analyze',
        'evaluation_interface:load_checkpoint',
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

        cls = 'NeuralEnKFAnalyzer'
        filename = 'neural_enkf_analysis_update.py'
        candidate = load_module(
            'neural_enkf_analysis_update_candidate', CANDIDATE_DIR / filename
        )
        names = ('analyze', 'load_checkpoint')
        obj = getattr(candidate, cls, None)
        ok = callable(getattr(obj, names[0], None))
        update_check(
            current_index,
            ok,
            json.dumps({'class': cls, 'callable': ok}, sort_keys=True),
        )
        current_index += 1
        ok = callable(getattr(obj, names[1], None))
        update_check(
            current_index,
            ok,
            json.dumps({'class': cls, 'callable': ok}, sort_keys=True),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check analyzer execution behavior."""
    CHECKPOINT_NAMES = [
        'forward_executable:output_shape_finite',
        'forward_executable:all_missing_identity',
        'forward_executable:partial_mask_semantics',
        'forward_executable:malformed_inputs_rejected',
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

        filename = 'neural_enkf_analysis_update.py'
        candidate = load_module(
            'neural_enkf_analysis_update_candidate', CANDIDATE_DIR / filename
        )
        t = torch()
        cls = 'NeuralEnKFAnalyzer'
        m = getattr(candidate, cls)()
        prior = np.zeros((3, 5, 4))
        obs = np.ones((3, 2))
        mask = np.array([[True, True], [True, False], [False, False]])
        out = m.analyze(prior, obs, mask)
        ok = out.shape == prior.shape and np.isfinite(out).all()
        update_check(
            current_index,
            bool(ok),
            json.dumps({'shape': out.shape, 'finite': bool(np.isfinite(out).all())}, sort_keys=True),
        )
        current_index += 1
        m = getattr(candidate, cls)()
        prior = np.zeros((3, 5, 4))
        observations = np.ones((3, 2))
        ok = np.array_equal(
            m.analyze(prior, observations, np.zeros((3, 2), dtype=bool)), prior
        )
        update_check(current_index, bool(ok), 'all missing returns prior')
        current_index += 1
        m = getattr(candidate, cls)()
        p = np.zeros((1, 3, 4))
        o = np.ones((1, 2))
        out = m.analyze(p, o, np.array([[True, False]]))
        ok = np.any(out[..., 0] != 0) and np.all(out[..., 1:] == 0)
        update_check(
            current_index,
            bool(ok),
            json.dumps(
                {
                    'observed_updated': bool(np.any(out[..., 0] != 0)),
                    'unobserved_unchanged': bool(np.all(out[..., 1:] == 0)),
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        m = getattr(candidate, cls)()
        rejected_dtype = rejected_shape = False
        try:
            m.analyze(
                np.zeros((1, 2, 4)),
                np.zeros((1, 2)),
                np.ones((1, 2), dtype=float),
            )
        except ValueError:
            rejected_dtype = True
        try:
            m.analyze(np.zeros((1, 2, 3)), np.zeros((1, 2)))
        except ValueError:
            rejected_shape = True
        ok = rejected_dtype and rejected_shape
        update_check(
            current_index,
            bool(ok),
            json.dumps(
                {'mask_dtype': rejected_dtype, 'prior_shape': rejected_shape},
                sort_keys=True,
            ),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the analyzer array shapes."""
    CHECKPOINT_NAMES = [
        'model_structure:gain_shape',
        'model_structure:observation_matrix_shape',
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

        cls = 'NeuralEnKFAnalyzer'
        filename = 'neural_enkf_analysis_update.py'
        candidate = load_module(
            'neural_enkf_analysis_update_candidate', CANDIDATE_DIR / filename
        )
        model = getattr(candidate, cls)()
        ok = model.gain.shape == (4, 2)
        update_check(
            current_index,
            bool(ok),
            json.dumps({'shape': model.gain.shape}, sort_keys=True),
        )
        current_index += 1
        ok = model.observation_matrix.shape == (2, 4)
        update_check(
            current_index,
            bool(ok),
            json.dumps({'shape': model.observation_matrix.shape}, sort_keys=True),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
