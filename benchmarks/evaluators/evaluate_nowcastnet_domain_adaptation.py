"""V2 adapter for the legacy NowcastNet domain-adaptation evaluator."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

import torch

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
    CHECKPOINT_NAMES = [
        "file:nowcastnet_domain_adaptation.py",
        "file:train.py",
        "file:evaluate.py",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = {
            "model_file": config.get("model_file", "nowcastnet_domain_adaptation.py"),
            "class_name": config.get("class_name", "NowcastNetDomainAdapter"),
            "env": config.get("env", "NOWCASTNET_DOMAIN_SUBMISSION_DIR"),
            "shape": config.get("shape", ((2, 4, 8, 8), (2, 3, 8, 8))),
            "kwargs": config.get("kwargs", {}),
            "kind": "domain",
        }
        for name in (case["model_file"], "train.py", "evaluate.py"):
            path = (CANDIDATE_DIR / name).resolve()
            path.relative_to(CANDIDATE_DIR)
            ok = path.is_file()
            detail = str(path)
            if ok:
                try:
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                    detail = f"syntax=ok, bytes={path.stat().st_size}"
                except Exception as exc:
                    ok, detail = False, f"{type(exc).__name__}: {exc}"
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_padding_and_mask(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["data:synthetic_weather_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = {
            "model_file": config.get("model_file", "nowcastnet_domain_adaptation.py"),
            "class_name": config.get("class_name", "NowcastNetDomainAdapter"),
            "env": config.get("env", "NOWCASTNET_DOMAIN_SUBMISSION_DIR"),
            "shape": config.get("shape", ((2, 4, 8, 8), (2, 3, 8, 8))),
            "kwargs": config.get("kwargs", {}),
            "kind": "domain",
        }
        input_shape, target_shape = case["shape"]
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets)
        mask[..., :2, :2] = 1
        ok = bool(torch.isfinite(inputs).all() and torch.isfinite(targets).all())
        msg = f"input={input_shape}, target={target_shape}, mask={tuple(mask.shape)}"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["train_script:adamw_backward_step"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        train_path = (CANDIDATE_DIR / "train.py").resolve()
        train_path.relative_to(CANDIDATE_DIR)
        if str(train_path.parent) not in sys.path:
            sys.path.insert(0, str(train_path.parent))
        spec = importlib.util.spec_from_file_location("weather_finetune_train", train_path)
        if spec is None or spec.loader is None:
            raise ImportError(train_path)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules["weather_finetune_train"] = train_module
        spec.loader.exec_module(train_module)
        source = Path(train_module.__file__).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
            node = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "train_step"
            )
            calls = {
                n.func.id
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                else (
                    n.func.attr
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    else ""
                )
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
            }
            parameters = {arg.arg for arg in node.args.args}
            ok = (
                {"model", "inputs", "targets"}.issubset(parameters)
                and {"backward", "step"}.issubset(calls)
                and "AdamW" in source
            )
            msg = f"parameters={sorted(parameters)}, calls={sorted(calls)}"
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["eval_script:metric_interface"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        eval_path = (CANDIDATE_DIR / "evaluate.py").resolve()
        eval_path.relative_to(CANDIDATE_DIR)
        if str(eval_path.parent) not in sys.path:
            sys.path.insert(0, str(eval_path.parent))
        spec = importlib.util.spec_from_file_location("weather_finetune_eval", eval_path)
        if spec is None or spec.loader is None:
            raise ImportError(eval_path)
        eval_module = importlib.util.module_from_spec(spec)
        sys.modules["weather_finetune_eval"] = eval_module
        spec.loader.exec_module(eval_module)
        try:
            source = Path(eval_module.__file__).read_text(encoding="utf-8")
            ok = "def evaluate" in source and "rmse" in source and "mae" in source
            msg = "evaluate/rmse/mae"
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["model_structure:required_module_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = {
            "model_file": config.get("model_file", "nowcastnet_domain_adaptation.py"),
            "class_name": config.get("class_name", "NowcastNetDomainAdapter"),
            "env": config.get("env", "NOWCASTNET_DOMAIN_SUBMISSION_DIR"),
            "shape": config.get("shape", ((2, 4, 8, 8), (2, 3, 8, 8))),
            "kwargs": config.get("kwargs", {}),
            "kind": "domain",
        }
        input_shape, target_shape = case["shape"]
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets)
        mask[..., :2, :2] = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_path = (CANDIDATE_DIR / case["model_file"]).resolve()
        train_path = (CANDIDATE_DIR / "train.py").resolve()
        model_path.relative_to(CANDIDATE_DIR)
        train_path.relative_to(CANDIDATE_DIR)
        if str(model_path.parent) not in sys.path:
            sys.path.insert(0, str(model_path.parent))
        spec = importlib.util.spec_from_file_location("weather_finetune_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules["weather_finetune_model"] = model_module
        spec.loader.exec_module(model_module)
        if str(train_path.parent) not in sys.path:
            sys.path.insert(0, str(train_path.parent))
        spec = importlib.util.spec_from_file_location("weather_finetune_train", train_path)
        if spec is None or spec.loader is None:
            raise ImportError(train_path)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules["weather_finetune_train"] = train_module
        spec.loader.exec_module(train_module)
        model = train_module.build_model(case["kwargs"]).to(device)
        children = dict(model.named_children())
        with torch.no_grad():
            backbone = model.backbone(inputs)
            low_rank = model.adapter_down(inputs)
            adapted = model.adapter_up(low_rank)
            output = model.head(torch.relu(backbone + adapted))
        ok = (
            list(children) == ["backbone", "adapter_down", "adapter_up", "head"]
            and all(not p.requires_grad for p in model.backbone.parameters())
            and backbone.shape == adapted.shape == (2, 16, 8, 8)
            and low_rank.shape == (2, 4, 8, 8)
            and output.shape == (2, 3, 8, 8)
        )
        msg = (
            f"children={list(children)}, backbone={tuple(backbone.shape)}, "
            f"low_rank={tuple(low_rank.shape)}, adapted={tuple(adapted.shape)}, "
            f"output={tuple(output.shape)}"
        )
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_executable:shape_and_finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = {
            "model_file": config.get("model_file", "nowcastnet_domain_adaptation.py"),
            "class_name": config.get("class_name", "NowcastNetDomainAdapter"),
            "env": config.get("env", "NOWCASTNET_DOMAIN_SUBMISSION_DIR"),
            "shape": config.get("shape", ((2, 4, 8, 8), (2, 3, 8, 8))),
            "kwargs": config.get("kwargs", {}),
            "kind": "domain",
        }
        input_shape, target_shape = case["shape"]
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets)
        mask[..., :2, :2] = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_path = (CANDIDATE_DIR / case["model_file"]).resolve()
        train_path = (CANDIDATE_DIR / "train.py").resolve()
        model_path.relative_to(CANDIDATE_DIR)
        train_path.relative_to(CANDIDATE_DIR)
        if str(model_path.parent) not in sys.path:
            sys.path.insert(0, str(model_path.parent))
        spec = importlib.util.spec_from_file_location("weather_finetune_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules["weather_finetune_model"] = model_module
        spec.loader.exec_module(model_module)
        if str(train_path.parent) not in sys.path:
            sys.path.insert(0, str(train_path.parent))
        spec = importlib.util.spec_from_file_location("weather_finetune_train", train_path)
        if spec is None or spec.loader is None:
            raise ImportError(train_path)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules["weather_finetune_train"] = train_module
        spec.loader.exec_module(train_module)
        model = train_module.build_model(case["kwargs"]).to(device)
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        ok = output.shape == targets.shape and bool(torch.isfinite(output).all())
        msg = f"output={tuple(output.shape)}, expected={tuple(targets.shape)}"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["traning:real_backward_optimizer_update"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = {
            "model_file": config.get("model_file", "nowcastnet_domain_adaptation.py"),
            "class_name": config.get("class_name", "NowcastNetDomainAdapter"),
            "env": config.get("env", "NOWCASTNET_DOMAIN_SUBMISSION_DIR"),
            "shape": config.get("shape", ((2, 4, 8, 8), (2, 3, 8, 8))),
            "kwargs": config.get("kwargs", {}),
            "kind": "domain",
        }
        input_shape, target_shape = case["shape"]
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets)
        mask[..., :2, :2] = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_path = (CANDIDATE_DIR / case["model_file"]).resolve()
        train_path = (CANDIDATE_DIR / "train.py").resolve()
        model_path.relative_to(CANDIDATE_DIR)
        train_path.relative_to(CANDIDATE_DIR)
        if str(model_path.parent) not in sys.path:
            sys.path.insert(0, str(model_path.parent))
        spec = importlib.util.spec_from_file_location("weather_finetune_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules["weather_finetune_model"] = model_module
        spec.loader.exec_module(model_module)
        if str(train_path.parent) not in sys.path:
            sys.path.insert(0, str(train_path.parent))
        spec = importlib.util.spec_from_file_location("weather_finetune_train", train_path)
        if spec is None or spec.loader is None:
            raise ImportError(train_path)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules["weather_finetune_train"] = train_module
        spec.loader.exec_module(train_module)
        model = train_module.build_model(case["kwargs"]).to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )
        before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        try:
            kwargs = {"optimizer": optimizer}
            loss = train_module.train_step(model, inputs, targets, **kwargs)
            after = [p.detach() for p in model.parameters() if p.requires_grad]
            changed = any(
                not torch.equal(left, right) for left, right in zip(before, after)
            )
            ok = isinstance(loss, (float, int)) and loss == loss and changed
            msg = f"loss={loss}, parameter_changed={changed}"
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
