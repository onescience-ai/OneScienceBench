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
    params = config.get("params", config)
    required_files = params.get("required_files", [])
    phase = params.get("phase", "files")
    if isinstance(required_files, str):
        required_files = [required_files]
    if not required_files:
        return [("files.config", False, "required_files is missing or empty")]
    CHECKPOINT_NAMES = ["file:required"] if len(required_files) == 1 else [
        f"{phase}.{name}" for name in required_files
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    for index, relative in enumerate(required_files):
        try:
            path = (CANDIDATE_DIR / relative).resolve()
            valid = path.is_file()
            if valid and path.suffix.lower() not in {".pt", ".pth", ".bin"}:
                valid = bool(path.read_text(encoding="utf-8").strip())
            checks[index] = (CHECKPOINT_NAMES[index], valid, str(path))
        except Exception:
            checks[index] = (CHECKPOINT_NAMES[index], False, traceback.format_exc())
    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = [
        "checkpoint_loading:state_dict_keys_equal",
        "checkpoint_loading:state_tensor_values_equal",
        "checkpoint_loading:malformed_checkpoint_must_raise",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    current_index = 0
    try:
        model_file = params["model_file"]
        model_class_name = params["model_class"]
        candidate = load_module("station_tft_probabilistic_forecast_candidate", (CANDIDATE_DIR / model_file).resolve())
        model_cls = getattr(candidate, model_class_name)
        candidate_config = _load_candidate_config(params.get("config_name")) if params.get("config_name") else {}
        context = {"config": candidate_config}
        resolved_args = _resolve_args(params.get("model_args", {}), context)
        source_model = model_cls(**resolved_args)
        with tempfile.TemporaryDirectory() as td:
            good = Path(td) / Path(params.get("checkpoint_file", "good.pt")).name
            bad = Path(td) / Path(params.get("malformed_checkpoint_file", "bad.pt")).name
            t = torch()
            source_state = source_model.state_dict()
            t.save({"model_state_dict": source_state}, good)
            loaded = candidate.load_checkpoint(good, "cpu")
            loaded_state = loaded.state_dict()
            keys_ok = tuple(loaded_state.keys()) == tuple(source_state.keys())
            checks[0] = (CHECKPOINT_NAMES[0], keys_ok, json.dumps({"keys_equal": keys_ok}, sort_keys=True))
            values_ok = keys_ok and all(t.equal(loaded_state[key], value) for key, value in source_state.items())
            checks[1] = (CHECKPOINT_NAMES[1], values_ok, json.dumps({"values_equal": values_ok}, sort_keys=True))
            t.save({}, bad)
            try:
                candidate.load_checkpoint(bad, "cpu")
            except Exception:
                rejected = True
            else:
                rejected = False
            checks[2] = (CHECKPOINT_NAMES[2], rejected, json.dumps({"malformed_rejected": rejected, "strict": params.get("strict", True)}, sort_keys=True))
    except Exception:
        checks[current_index] = (CHECKPOINT_NAMES[current_index], False, traceback.format_exc())
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = [
        "model_structure:module_order",
        "model_structure:attention_batch_first",
        "model_structure:quantile_head",
        "model_structure:quantile_levels",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    try:
        candidate = load_module("station_tft_probabilistic_forecast_candidate", (CANDIDATE_DIR / params["model_file"]).resolve())
        model_cls = getattr(candidate, params["model_class"])
        candidate_config = _load_candidate_config(params.get("config_name")) if params.get("config_name") else {}
        context = {"config": candidate_config, **candidate_config}
        model = model_cls(**_resolve_args(params.get("model_args", {}), context)).eval()
    except Exception:
        checks[0] = (CHECKPOINT_NAMES[0], False, traceback.format_exc())
        return checks
    rules = params.get("rules", [])
    for index, rule in enumerate(rules[:len(CHECKPOINT_NAMES)]):
        try:
            rule_params = rule.get("params", {})
            if rule.get("type") == "required_modules":
                actual = set(dict(model.named_children()))
                expected = set(rule_params.get("names", []))
                passed = expected.issubset(actual)
                message = json.dumps({"required": sorted(expected), "actual": sorted(actual)}, sort_keys=True)
            else:
                target = model
                for part in rule_params.get("target_path", "").split(".") if rule_params.get("target_path", "") else []:
                    target = target[int(part)] if part.isdigit() else getattr(target, part)
                actual = getattr(target, rule_params["attribute"])
                expected = rule_params.get("value")
                passed = actual == expected
                if isinstance(actual, tuple):
                    passed = list(actual) == expected
                message = json.dumps({"actual": actual, "expected": expected}, sort_keys=True, default=str)
            checks[index] = (CHECKPOINT_NAMES[index], bool(passed), message)
        except Exception:
            checks[index] = (CHECKPOINT_NAMES[index], False, traceback.format_exc())
    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    interfaces = params.get("required_interfaces", [])
    CHECKPOINT_NAMES = [
        "evaluation_interface:predict" if "predict" in name else "evaluation_interface:module_load_checkpoint"
        for name in interfaces
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    modules = []
    for filename in params.get("files", []):
        try:
            modules.append(load_module("station_tft_probabilistic_forecast_interface", (CANDIDATE_DIR / filename).resolve()))
        except Exception:
            continue
    for index, interface in enumerate(interfaces):
        try:
            found = False
            for module in modules:
                if "." in interface:
                    class_name, method_name = interface.split(".", 1)
                    found = found or callable(getattr(getattr(module, class_name, None), method_name, None))
                else:
                    found = found or callable(getattr(module, interface, None))
            checks[index] = (CHECKPOINT_NAMES[index], found, f"{interface}: {found}")
        except Exception:
            checks[index] = (CHECKPOINT_NAMES[index], False, traceback.format_exc())
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
