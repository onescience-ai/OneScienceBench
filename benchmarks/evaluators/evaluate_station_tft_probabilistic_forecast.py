"""V2 adapter for station TFT probabilistic forecast evaluation."""
from pathlib import Path
import sys

import importlib.util
import json
import os
import tempfile
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
    return bool(torch().isfinite(x).all())


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
        filename = "station_tft_probabilistic_forecast.py"
        sub = CANDIDATE_DIR
        ok = (sub / filename).is_file()
        detail = str(sub / filename)
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "checkpoint_loading:state_dict_keys_equal",
        "checkpoint_loading:state_tensor_values_equal",
        "checkpoint_loading:malformed_checkpoint_must_raise",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = "TFTProbabilisticForecastModel"
        candidate_path = (CANDIDATE_DIR / "station_tft_probabilistic_forecast.py").resolve()
        standard_path = (REFERENCE_DIR / "station_tft_probabilistic_forecast.py").resolve()
        candidate = load_module("station_tft_probabilistic_forecast_candidate", candidate_path)
        standard = load_module("station_tft_probabilistic_forecast_standard", standard_path)
        with tempfile.TemporaryDirectory() as td:
            good, bad = Path(td) / "good.pt", Path(td) / "bad.pt"
            ref = getattr(standard, cls)()
            t = torch()
            t.save({"model_state_dict": ref.state_dict()}, good)
            loaded = candidate.load_checkpoint(good, "cpu")
            state_dict_keys_equal = tuple(loaded.state_dict()) == tuple(ref.state_dict())
            state_tensor_values_equal = all(
                t.equal(loaded.state_dict()[k], v) for k, v in ref.state_dict().items()
            )
            same = state_dict_keys_equal and state_tensor_values_equal
            update_check(
                current_index,
                state_dict_keys_equal,
                json.dumps({"state_dict_keys_equal": state_dict_keys_equal}, sort_keys=True),
            )
            current_index += 1
            update_check(
                current_index,
                state_tensor_values_equal,
                json.dumps({"state_tensor_values_equal": state_tensor_values_equal}, sort_keys=True),
            )
            current_index += 1
            t.save({}, bad)
            try:
                candidate.load_checkpoint(bad, "cpu")
            except Exception:
                rejected = True
            else:
                rejected = False
            ok = same and rejected
            update_check(
                current_index,
                rejected,
                json.dumps({"malformed_rejected": rejected}, sort_keys=True),
            )
            current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "model_structure:module_order",
        "model_structure:attention_batch_first",
        "model_structure:quantile_head",
        "model_structure:quantile_levels",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = "TFTProbabilisticForecastModel"
        candidate_path = (CANDIDATE_DIR / "station_tft_probabilistic_forecast.py").resolve()
        candidate = load_module("station_tft_probabilistic_forecast_candidate", candidate_path)
        model = getattr(candidate, cls)()
        modules = list(getattr(model, "_modules", {}))
        expected = ["variable_encoder", "time_encoder", "temporal_attention", "quantile_head"]
        module_order_ok = modules == expected
        update_check(
            current_index,
            module_order_ok,
            json.dumps({"actual": modules, "expected": expected}, sort_keys=True),
        )
        current_index += 1
        attention_batch_first_ok = model.temporal_attention.batch_first is True
        update_check(
            current_index,
            attention_batch_first_ok,
            json.dumps({"batch_first": model.temporal_attention.batch_first}, sort_keys=True),
        )
        current_index += 1
        quantile_head_ok = model.quantile_head.out_features == 12
        update_check(
            current_index,
            quantile_head_ok,
            json.dumps({"out_features": model.quantile_head.out_features}, sort_keys=True),
        )
        current_index += 1
        quantile_levels_ok = tuple(model.quantile_levels) == (0.1, 0.5, 0.9)
        update_check(
            current_index,
            quantile_levels_ok,
            json.dumps({"levels": model.quantile_levels}, sort_keys=True),
        )
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "evaluation_interface:predict",
        "evaluation_interface:module_load_checkpoint",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = "TFTProbabilisticForecastModel"
        candidate_path = (CANDIDATE_DIR / "station_tft_probabilistic_forecast.py").resolve()
        candidate = load_module("station_tft_probabilistic_forecast_candidate", candidate_path)
        obj = getattr(candidate, cls, None)
        name = "predict"
        callable_method = callable(getattr(obj, name, None))
        update_check(
            current_index,
            callable_method,
            json.dumps({"class": cls, "callable": callable_method}, sort_keys=True),
        )
        current_index += 1
        callable_loader = callable(getattr(candidate, "load_checkpoint", None))
        update_check(
            current_index,
            callable_loader,
            json.dumps({"callable": callable_loader}, sort_keys=True),
        )
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "forward_executable:quantiles_shape",
        "forward_executable:point_shape",
        "forward_executable:quantiles_finite",
        "forward_executable:quantile_10_lt_50",
        "forward_executable:quantile_50_lt_90",
        "forward_executable:empty_mask_rejected",
        "forward_executable:mask_dtype_rejected",
        "forward_executable:input_immutable",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = "TFTProbabilisticForecastModel"
        candidate_path = (CANDIDATE_DIR / "station_tft_probabilistic_forecast.py").resolve()
        candidate = load_module("station_tft_probabilistic_forecast_candidate", candidate_path)
        t = torch()
        m = getattr(candidate, cls)().eval()
        batch = {
            "values": t.randn(2, 12, 4),
            "time_features": t.randn(2, 12, 3),
            "observed_mask": t.ones(2, 12, 4, dtype=t.bool),
        }
        out = m.predict(batch)
        q, p = out["quantiles"], out["point"]
        shapes = {"quantiles": tuple(q.shape), "point": tuple(p.shape)}
        quantiles_shape_ok = q.shape == (2, 6, 4, 3)
        update_check(
            current_index,
            quantiles_shape_ok,
            json.dumps(shapes, sort_keys=True),
        )
        current_index += 1
        point_shape_ok = p.shape == (2, 6, 4)
        update_check(
            current_index,
            point_shape_ok,
            json.dumps(shapes, sort_keys=True),
        )
        current_index += 1
        out = getattr(candidate, cls)().predict(
            {
                "values": t.randn(2, 12, 4),
                "time_features": t.randn(2, 12, 3),
                "observed_mask": t.ones(2, 12, 4, dtype=t.bool),
            }
        )
        q = out["quantiles"]
        quantiles_finite_ok = finite(q)
        update_check(current_index, quantiles_finite_ok, "finite quantiles")
        current_index += 1
        quantile_10_lt_50_ok = bool((q[..., 0] < q[..., 1]).all())
        update_check(current_index, quantile_10_lt_50_ok, "quantile 0.1 < quantile 0.5")
        current_index += 1
        quantile_50_lt_90_ok = bool((q[..., 1] < q[..., 2]).all())
        update_check(current_index, quantile_50_lt_90_ok, "quantile 0.5 < quantile 0.9")
        current_index += 1
        m = getattr(candidate, cls)()
        values = t.randn(1, 12, 4)
        times = t.randn(1, 12, 3)
        no = {
            "values": values,
            "time_features": times,
            "observed_mask": t.zeros(1, 12, 4, dtype=t.bool),
        }
        bad = dict(no)
        bad["observed_mask"] = t.zeros(1, 12, 4)
        try:
            m.predict(no)
        except ValueError:
            rejected_empty = True
        else:
            rejected_empty = False
        update_check(
            current_index,
            rejected_empty,
            json.dumps({"empty_mask": rejected_empty}, sort_keys=True),
        )
        current_index += 1
        try:
            m.predict(bad)
        except ValueError:
            rejected_dtype = True
        else:
            rejected_dtype = False
        update_check(
            current_index,
            rejected_dtype,
            json.dumps({"mask_dtype": rejected_dtype}, sort_keys=True),
        )
        current_index += 1
        m = getattr(candidate, cls)()
        batch = {
            "values": t.randn(1, 12, 4),
            "time_features": t.randn(1, 12, 3),
            "observed_mask": t.ones(1, 12, 4, dtype=t.bool),
        }
        before = {k: v.clone() for k, v in batch.items()}
        m.predict(batch)
        immutable_ok = all(t.equal(batch[k], before[k]) for k in batch)
        update_check(current_index, immutable_ok, "inputs unchanged")
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["metric:pinball_loss", "metric:mae"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        cls = "TFTProbabilisticForecastModel"
        candidate_path = (CANDIDATE_DIR / "station_tft_probabilistic_forecast.py").resolve()
        standard_path = (REFERENCE_DIR / "station_tft_probabilistic_forecast.py").resolve()
        candidate = load_module("station_tft_probabilistic_forecast_candidate", candidate_path)
        standard = load_module("station_tft_probabilistic_forecast_standard", standard_path)
        t = torch()
        ref = getattr(standard, cls)().eval()
        cand = getattr(candidate, cls)().eval()
        cand.load_state_dict(ref.state_dict(), strict=True)
        batch = {
            "values": t.linspace(-1, 1, 96).reshape(2, 12, 4),
            "time_features": t.zeros(2, 12, 3),
            "observed_mask": t.ones(2, 12, 4, dtype=t.bool),
        }
        aa, bb = cand.predict(batch), ref.predict(batch)
        target = t.zeros_like(bb["point"].detach())
        ref_q = bb["quantiles"].detach()
        q = aa["quantiles"].detach()
        levels = t.tensor([0.1, 0.5, 0.9])
        diff = target.unsqueeze(-1) - q
        ref_diff = target.unsqueeze(-1) - ref_q
        pinball = t.maximum(levels * diff, (levels - 1) * diff).mean()
        ref_pinball = t.maximum(levels * ref_diff, (levels - 1) * ref_diff).mean()
        mae = (aa["point"].detach() - target).abs().mean()
        ref_mae = (bb["point"].detach() - target).abs().mean()
        pinball_ok = bool(t.abs(pinball - ref_pinball) <= 1e-6)
        update_check(
            current_index,
            pinball_ok,
            json.dumps(
                {
                    "pinball_loss": float(pinball),
                    "oracle_pinball_loss": float(ref_pinball),
                    "target_source": "fixed synthetic point target",
                },
                sort_keys=True,
            ),
        )
        current_index += 1
        mae_ok = bool(t.abs(mae - ref_mae) <= 1e-6)
        update_check(
            current_index,
            mae_ok,
            json.dumps(
                {
                    "mae": float(mae),
                    "oracle_mae": float(ref_mae),
                    "target_source": "fixed synthetic point target",
                },
                sort_keys=True,
            ),
        )
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks
