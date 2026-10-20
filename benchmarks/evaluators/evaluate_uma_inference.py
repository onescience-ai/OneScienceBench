"""V2 UMA evaluator adapter."""
from __future__ import annotations
import importlib.util
import inspect
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

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

REQUIRED_FILES = ("uma.py", "uma_infer.py", "config.yaml")
CORE_CONFIG_KEYS = {"checkpoint", "device", "dtype", "dataset", "batch_size", "lmax", "mmax", "norm_type", "act_type", "ff_type"}
FORBIDDEN_NESTED_CONFIG_KEYS = {"model", "backbone", "heads"}
DEFAULT_CHECKPOINT_DIR = Path("/public/share/sugonhpcapp01/onestore/onemodels/UMA/checkpoint")
PREFERRED_CHECKPOINTS = ("uma-s-1p1_converted.pt", "uma-s-1p1.pt")
INFERENCE_ONLY_CONFIG_KEYS = {"checkpoint", "checkpoint_path", "device", "dtype", "batch_size", "input", "input_path", "output", "output_path", "num_workers"}
_RUNTIME_MODEL = None
_RUNTIME_CHECKPOINT = None
_RUNTIME_MODEL_MODULE = None
_RUNTIME_INFER_MODULE = None
_RUNTIME_STANDARD_MODEL = None
_RUNTIME_STANDARD_INFER = None
_RUNTIME_STANDARD_CHECKPOINT = None
_RUNTIME_STANDARD_DEVICE = None


def _legacy():
    path = _repo_root() / "evaluation" / "eval_datasets" / "codegen" / "evaluators" / "evaluate_uma_inference.py"
    spec = importlib.util.spec_from_file_location("uma_legacy_evaluator", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import legacy evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device(config):
    return str(config.get("device", "cpu"))


def _records(names, records, checks, update_check, current_index):
    for name, passed, message in records:
        if current_index >= len(names):
            break
        update_check(current_index, bool(passed), str(message))
        current_index += 1
    return current_index


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["file:uma.py", "file:uma_infer.py", "file:config.yaml", "file:only_requested_source_files", "file:required_artifacts"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        authored = {path.name for path in CANDIDATE_DIR.iterdir() if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml"}}
        extras = sorted(authored - set(REQUIRED_FILES))
        for relative in REQUIRED_FILES:
            update_check(current_index, (CANDIDATE_DIR / relative).is_file(), str(CANDIDATE_DIR / relative))
            current_index += 1
        update_check(current_index, not extras, f"extra Python/YAML files={extras}")
        current_index += 1
        update_check(current_index, CANDIDATE_DIR.is_dir() and all((CANDIDATE_DIR / relative).is_file() for relative in REQUIRED_FILES), str(CANDIDATE_DIR))
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["file:required_flat_keys", "file:flat_mapping", "file:lmax", "file:mmax", "file:norm_type", "file:act_type", "file:ff_type", "file:default_checkpoint_directory", "file:positive_batch_size", "file:exact_filename"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import yaml
        config_path = CANDIDATE_DIR / "config.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("config.yaml must contain a mapping at its root")
        missing = sorted(CORE_CONFIG_KEYS - set(payload))
        update_check(current_index, not missing, f"missing={missing}"); current_index += 1
        nested = sorted(key for key, value in payload.items() if isinstance(value, dict))
        forbidden = sorted(FORBIDDEN_NESTED_CONFIG_KEYS & set(payload))
        update_check(current_index, not nested and not forbidden, f"nested_values={nested}, forbidden_keys={forbidden}"); current_index += 1
        for key, expected in {"lmax": 2, "mmax": 2, "norm_type": "rms_norm_sh", "act_type": "gate", "ff_type": "spectral"}.items():
            actual = payload.get(key)
            update_check(current_index, actual == expected, f"expected={expected!r}, actual={actual!r}"); current_index += 1
        checkpoint = payload.get("checkpoint")
        matched = isinstance(checkpoint, (str, os.PathLike)) and Path(checkpoint).expanduser().resolve() == DEFAULT_CHECKPOINT_DIR.resolve()
        update_check(current_index, matched, f"expected={DEFAULT_CHECKPOINT_DIR}, actual={checkpoint!r}"); current_index += 1
        update_check(current_index, isinstance(payload.get("batch_size"), int) and payload["batch_size"] > 0, repr(payload.get("batch_size"))); current_index += 1
        update_check(current_index, config_path.name == "config.yaml", str(config_path)); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["checkpoint:load_pt"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    global _RUNTIME_MODEL, _RUNTIME_CHECKPOINT, _RUNTIME_MODEL_MODULE, _RUNTIME_INFER_MODULE
    try:
        legacy = _legacy()
        model_module, infer_module = legacy._load_submission_modules(CANDIDATE_DIR)
        _RUNTIME_MODEL_MODULE, _RUNTIME_INFER_MODULE = model_module, infer_module
        checkpoint_path = legacy.resolve_checkpoint_path(config.get("checkpoint"))
        model = legacy._invoke_load_model(infer_module, checkpoint_path=checkpoint_path, device=_device(config), dtype=str(config.get("dtype", "float32")))
        import torch
        ok = isinstance(model, torch.nn.Module)
        _RUNTIME_MODEL = model if ok else None
        _RUNTIME_CHECKPOINT = checkpoint_path
        update_check(current_index, ok, f"loaded {checkpoint_path} as {type(model).__name__}"); current_index += 1
    except Exception:
        _RUNTIME_MODEL = None
        _RUNTIME_CHECKPOINT = None
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["model_struct:runtime_model_available"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    global _RUNTIME_STANDARD_MODEL, _RUNTIME_STANDARD_INFER, _RUNTIME_STANDARD_CHECKPOINT, _RUNTIME_STANDARD_DEVICE
    try:
        legacy = _legacy()
        device = _device(config)
        records = legacy._check_model_structure(CANDIDATE_DIR, _RUNTIME_MODEL_MODULE, _RUNTIME_INFER_MODULE, config, device)
        CHECKPOINT_NAMES[:] = [record[0] for record in records] or ["model_struct:runtime_model_available"]
        checks[:] = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
        for _name, passed, message in records:
            update_check(current_index, passed, message); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "evaluation_interface:uma_class",
        "evaluation_interface:uma_is_nn_module",
        "evaluation_interface:load_model",
        "evaluation_interface:predict",
        "evaluation_interface:load_model_device_dtype",
        "evaluation_interface:predict_existing_model",
        "evaluation_interface:construct_from_flat_config",
        "evaluation_interface:runtime_model_available",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        legacy = _legacy()
        try:
            model_module, infer_module = legacy._load_submission_modules(CANDIDATE_DIR)
        except Exception:
            model_module, infer_module = _RUNTIME_MODEL_MODULE, _RUNTIME_INFER_MODULE
        device = _device(config)
        records = legacy._check_evaluation_interface(CANDIDATE_DIR, model_module, infer_module, config, device)
        for record in records:
            if record[0] in CHECKPOINT_NAMES:
                update_check(current_index, record[1], record[2]); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "evaluation_interface:runtime_model_available",
        "evaluation_interface:mapping_input_contract",
        "evaluation_interface:energy_keeps_autograd",
        "evaluation_interface:force_is_negative_energy_gradient",
        "evaluation_interface:atomicdata_like_object",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        legacy = _legacy()
        model_module, infer_module = _RUNTIME_MODEL_MODULE, _RUNTIME_INFER_MODULE
        records = legacy._check_evaluation_interface(CANDIDATE_DIR, model_module, infer_module, config, _device(config))
        for record in records:
            if record[0] in CHECKPOINT_NAMES:
                update_check(current_index, record[1], record[2]); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["metric:candidate_inference_interface"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        legacy = _legacy()
        records = legacy._check_evaluate_oracle(CANDIDATE_DIR, _RUNTIME_INFER_MODULE, config, _device(config))
        record = next((item for item in records if item[0] == CHECKPOINT_NAMES[0]), records[0])
        update_check(current_index, record[1], record[2]); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks
