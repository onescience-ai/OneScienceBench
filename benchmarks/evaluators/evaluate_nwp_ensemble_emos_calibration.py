"""V2 adapter for nwp_ensemble_emos_calibration."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

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
        filename = 'nwp_ensemble_emos_calibration.py'
        sub = CANDIDATE_DIR
        ok = (sub / filename).is_file()
        detail = str(sub / filename)
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'checkpoint_loading:state_values',
        'checkpoint_loading:unexpected_key_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = 'NWPEnsembleEMOS'
        filename = 'nwp_ensemble_emos_calibration.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        spec = importlib.util.spec_from_file_location(
            'nwp_ensemble_emos_calibration_candidate', str(candidate_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['nwp_ensemble_emos_calibration_candidate'] = candidate
        spec.loader.exec_module(candidate)
        with tempfile.TemporaryDirectory() as td:
            good, bad = Path(td) / 'good.npz', Path(td) / 'bad.npz'
            np.savez(good, bias=np.array(1.25), scale=np.array(0.75))
            np.savez(bad, bias=np.array(1.25), extra=np.array(0.75))
            loaded = candidate.NWPEnsembleEMOS().load_checkpoint(good)
            same = loaded.bias == 1.25 and loaded.scale == 0.75
            update_check(
                current_index,
                bool(same),
                json.dumps({'state_values': same}, sort_keys=True),
            )
            current_index += 1
            try:
                candidate.NWPEnsembleEMOS().load_checkpoint(bad)
                rejected = False
            except Exception:
                rejected = True
            update_check(
                current_index,
                rejected,
                json.dumps({'unexpected_key_rejected': rejected}, sort_keys=True),
            )
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'model_structure:quantile_levels',
        'model_structure:positive_scale',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = 'NWPEnsembleEMOS'
        filename = 'nwp_ensemble_emos_calibration.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        spec = importlib.util.spec_from_file_location(
            'nwp_ensemble_emos_calibration_candidate', str(candidate_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['nwp_ensemble_emos_calibration_candidate'] = candidate
        spec.loader.exec_module(candidate)
        model = getattr(candidate, cls)()
        levels = np.asarray(model.quantile_levels)
        ok = levels.shape == (3,) and np.allclose(levels, [0.1, 0.5, 0.9])
        update_check(
            current_index,
            ok,
            json.dumps({'levels': levels.tolist()}, sort_keys=True),
        )
        current_index += 1
        ok = np.isfinite(model.scale) and model.scale > 0
        update_check(
            current_index,
            bool(ok),
            json.dumps({'scale': model.scale}, sort_keys=True),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'evaluation_interface:predict',
        'evaluation_interface:load_checkpoint',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = 'NWPEnsembleEMOS'
        filename = 'nwp_ensemble_emos_calibration.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        spec = importlib.util.spec_from_file_location(
            'nwp_ensemble_emos_calibration_candidate', str(candidate_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['nwp_ensemble_emos_calibration_candidate'] = candidate
        spec.loader.exec_module(candidate)
        obj = getattr(candidate, cls, None)
        ok = callable(getattr(obj, 'predict', None))
        update_check(
            current_index,
            ok,
            json.dumps({'class': cls, 'callable': ok}, sort_keys=True),
        )
        current_index += 1
        ok = callable(getattr(obj, 'load_checkpoint', None))
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
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'forward_executable:output_shapes',
        'forward_executable:spread_positive',
        'forward_executable:quantiles_ordered',
        'forward_executable:member_count_rejected',
        'forward_executable:nonfinite_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = 'NWPEnsembleEMOS'
        filename = 'nwp_ensemble_emos_calibration.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        spec = importlib.util.spec_from_file_location(
            'nwp_ensemble_emos_calibration_candidate', str(candidate_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['nwp_ensemble_emos_calibration_candidate'] = candidate
        spec.loader.exec_module(candidate)
        m = getattr(candidate, cls)()
        x = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
        out = m.predict(x)
        perm = m.predict(x[:, ::-1])
        ok = (
            set(out) == {'mean', 'spread', 'quantiles'}
            and out['mean'].shape == (2, 3)
            and out['spread'].shape == (2, 3)
            and out['quantiles'].shape == (2, 3, 3)
        )
        permutation_invariant = all(np.allclose(out[k], perm[k]) for k in out)
        update_check(
            current_index,
            ok and permutation_invariant,
            json.dumps(
                {
                    'shapes': {k: list(v.shape) for k, v in out.items()},
                    'permutation_invariant': permutation_invariant,
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        ok = bool(
            (
                getattr(candidate, cls)().predict(
                    np.arange(30.).reshape(2, 5, 3)
                )['spread']
                > 0
            ).all()
        )
        update_check(current_index, ok, 'spread > 0')
        current_index += 1
        q = getattr(candidate, cls)().predict(
            np.arange(30.).reshape(2, 5, 3)
        )['quantiles']
        ok = bool((np.diff(q, axis=-1) > 0).all())
        update_check(current_index, ok, 'quantiles strictly ordered')
        current_index += 1
        m = getattr(candidate, cls)()
        rejected_m = False
        try:
            m.predict(np.zeros((1, 1, 3)))
        except ValueError:
            rejected_m = True
        update_check(
            current_index,
            rejected_m,
            json.dumps({'M<2': rejected_m}, sort_keys=True),
        )
        current_index += 1
        rejected_nan = False
        try:
            m.predict(np.array([[np.nan, 0.0], [1.0, 2.0]]))
        except ValueError:
            rejected_nan = True
        update_check(
            current_index,
            rejected_nan,
            json.dumps({'nonfinite': rejected_nan}, sort_keys=True),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'metric:mean_rmse',
        'metric:mean_mae',
        'metric:spread_rmse',
        'metric:spread_mae',
        'metric:quantiles_rmse',
        'metric:quantiles_mae',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = 'NWPEnsembleEMOS'
        filename = 'nwp_ensemble_emos_calibration.py'
        candidate_path = (CANDIDATE_DIR / filename).resolve()
        spec = importlib.util.spec_from_file_location(
            'nwp_ensemble_emos_calibration_candidate', str(candidate_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(str(candidate_path))
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['nwp_ensemble_emos_calibration_candidate'] = candidate
        spec.loader.exec_module(candidate)
        standard_path = (REFERENCE_DIR / filename).resolve()
        spec = importlib.util.spec_from_file_location(
            'nwp_ensemble_emos_calibration_standard', str(standard_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(str(standard_path))
        standard = importlib.util.module_from_spec(spec)
        sys.modules['nwp_ensemble_emos_calibration_standard'] = standard
        spec.loader.exec_module(standard)
        x = np.linspace(-2, 2, 48).reshape(2, 4, 6)
        a = candidate.NWPEnsembleEMOS(.3, .7).predict(x)
        b = standard.NWPEnsembleEMOS(.3, .7).predict(x)
        errs = {k: np.asarray(a[k]) - np.asarray(b[k]) for k in b}
        details = {
            k: {
                'rmse': float(np.sqrt(np.mean(v * v))),
                'mae': float(np.mean(np.abs(v))),
                'max_abs_error': float(np.max(np.abs(v))),
            }
            for k, v in errs.items()
        }
        ok = details['mean']['max_abs_error'] <= 1e-10
        update_check(
            current_index,
            ok,
            json.dumps(
                {
                    'rmse': details['mean']['rmse'],
                    'max_abs_error': details['mean']['max_abs_error'],
                    'atol': 1e-10,
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        ok = details['mean']['max_abs_error'] <= 1e-10
        update_check(
            current_index,
            ok,
            json.dumps(
                {
                    'mae': details['mean']['mae'],
                    'max_abs_error': details['mean']['max_abs_error'],
                    'atol': 1e-10,
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        ok = details['spread']['max_abs_error'] <= 1e-10
        update_check(
            current_index,
            ok,
            json.dumps(
                {
                    'rmse': details['spread']['rmse'],
                    'max_abs_error': details['spread']['max_abs_error'],
                    'atol': 1e-10,
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        ok = details['spread']['max_abs_error'] <= 1e-10
        update_check(
            current_index,
            ok,
            json.dumps(
                {
                    'mae': details['spread']['mae'],
                    'max_abs_error': details['spread']['max_abs_error'],
                    'atol': 1e-10,
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        ok = details['quantiles']['max_abs_error'] <= 1e-10
        update_check(
            current_index,
            ok,
            json.dumps(
                {
                    'rmse': details['quantiles']['rmse'],
                    'max_abs_error': details['quantiles']['max_abs_error'],
                    'atol': 1e-10,
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        ok = details['quantiles']['max_abs_error'] <= 1e-10
        update_check(
            current_index,
            ok,
            json.dumps(
                {
                    'mae': details['quantiles']['mae'],
                    'max_abs_error': details['quantiles']['max_abs_error'],
                    'atol': 1e-10,
                },
                sort_keys=True,
            ),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
