"""V2 adapter for the OneForecast paper reproduction evaluator."""

import ast
import copy
import importlib.util
import inspect
import json
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict

import torch

warnings.filterwarnings(
    "ignore", message=".*not compiled with memory efficient attention.*"
)


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
TASK_NAME = "oneforecast_paper_repro"
MODEL_FILE = "oneforecast.py"
MODEL_CLASS = "OneForecast"
ENV_NAME = "ONEFORECAST_SUBMISSION_DIR"
REQUIRED_FILES = (MODEL_FILE, "train.py", "evaluate.py")
CHECK_TYPES = {
    "file", "data", "train_script", "eval_script", "model_structure",
    "forward_executable", "forward_equivalence", "traning", "artifacts",
}
SEED = 20260819


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _function(tree: ast.AST, name: str):
    return next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name), None)


def _calls(node: ast.AST) -> set[str]:
    result = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                result.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                result.add(child.func.attr)
    return result


def _subsequence(actual: list[str], expected: tuple[str, ...]) -> bool:
    positions = [actual.index(name) for name in expected if name in actual]
    return len(positions) == len(expected) and positions == sorted(positions)


def _rejects(callable_) -> bool:
    try:
        callable_()
    except (ValueError, TypeError):
        return True
    except Exception:
        return False
    return False



def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["data:multivariable_history_batch"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.get("seed", SEED))
        fixture = {
            "batch": torch.randn(2, 2, 3, 8, 8, device=device),
            "target": torch.randn(2, 3, 8, 8, device=device),
        }
        inputs, targets = fixture["batch"], fixture["target"]
        ok = (
            inputs.shape == (2, 2, 3, 8, 8)
            and targets.shape == (2, 3, 8, 8)
            and inputs.dtype == torch.float32
            and targets.dtype == torch.float32
            and bool(torch.isfinite(inputs).all())
            and bool(torch.isfinite(targets).all())
        )
        update_check(current_index, ok, f"input={tuple(inputs.shape)}, target={tuple(targets.shape)}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["train_script:adamw_mse_backward"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = (CANDIDATE_DIR / config.get("file", "train.py")).resolve()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        node = _function(tree, "train_step")
        if node is None:
            raise AttributeError("train_step is missing")
        parameters = [arg.arg for arg in node.args.args]
        calls = _calls(node)
        ok = all(name in parameters for name in ("model", "inputs", "targets")) and {"AdamW", "backward", "step"}.issubset(calls)
        update_check(current_index, ok, f"parameters={parameters}, calls={sorted(calls)}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_metrics(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["eval_script:mae_rmse_interface"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = (CANDIDATE_DIR / config.get("file", "evaluate.py")).resolve()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        node = _function(tree, "evaluate")
        if node is None:
            raise AttributeError("evaluate is missing")
        parameters = [arg.arg for arg in node.args.args]
        calls = _calls(node)
        strings = {child.value.lower() for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)}
        ok = all(name in parameters for name in ("model", "inputs", "targets")) and {"mae", "rmse"}.issubset(strings) and {"abs", "mean", "sqrt"}.issubset(calls)
        update_check(current_index, ok, f"parameters={parameters}, metrics={sorted(strings & {'mae', 'rmse'})}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["model_structure:variable_token_transformer_order_and_shapes"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_file = config.get("model_file", MODEL_FILE)
        model_class_name = config.get("model_class", MODEL_CLASS)
        model_args = config.get("model_args", {"variables": 3, "history": 2, "embed_dim": 16, "depth": 2, "heads": 4})
        module = _load_module("oneforecast_submission_model", (CANDIDATE_DIR / model_file).resolve())
        model_class = getattr(module, model_class_name)
        if not inspect.isclass(model_class):
            raise TypeError(f"{model_class_name} is not a class")
        model = model_class(**model_args).to(device)
        children = list(dict(model.named_children()))
        blocks = getattr(model, "blocks")
        embeddings = getattr(model, "variable_embeddings")
        ok = (
            _subsequence(children, ("input_projection", "blocks", "output_projection"))
            and isinstance(model.input_projection, torch.nn.Conv2d)
            and (model.input_projection.in_channels, model.input_projection.out_channels) == (2, 16)
            and tuple(embeddings.shape) == (1, 3, 16)
            and isinstance(blocks, torch.nn.ModuleList)
            and len(blocks) == 2
            and all(_subsequence(list(dict(block.named_children())), ("norm1", "attn", "norm2", "mlp")) for block in blocks)
            and isinstance(model.output_projection, torch.nn.Linear)
            and model.output_projection.out_features == 1
            and callable(getattr(model, "rollout", None))
        )
        detail = f"children={children}, blocks={len(blocks)}, embeddings={tuple(embeddings.shape)}"
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_executable:forward_rollout_and_validation"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.get("seed", SEED))
        inputs = torch.randn(2, 2, 3, 8, 8, device=device)
        target = torch.randn(2, 3, 8, 8, device=device)
        fixture = {"batch": inputs, "target": target}
        model_file = config.get("model_file", MODEL_FILE)
        model_class_name = config.get("model_class", MODEL_CLASS)
        model_args = config.get("model_args", {"variables": 3, "history": 2, "embed_dim": 16, "depth": 2, "heads": 4})
        module = _load_module("oneforecast_submission_model", (CANDIDATE_DIR / model_file).resolve())
        model_class = getattr(module, model_class_name)
        if not inspect.isclass(model_class):
            raise TypeError(f"{model_class_name} is not a class")
        model = model_class(**model_args).to(device)
        inputs = fixture["batch"]
        original = inputs.clone()
        model.eval()
        with torch.no_grad():
            output = model(inputs)
            rollout = model.rollout(inputs, 3)
        rejects = {
            "rank": _rejects(lambda: model(inputs[:, 0])),
            "history": _rejects(lambda: model(inputs[:, :1])),
            "variables": _rejects(lambda: model(inputs[:, :, :2])),
            "steps": _rejects(lambda: model.rollout(inputs, 0)),
        }
        unchanged = torch.equal(inputs, original)
        ok = (
            output.shape == fixture["target"].shape
            and rollout.shape == (2, 3, 3, 8, 8)
            and bool(torch.isfinite(output).all())
            and bool(torch.isfinite(rollout).all())
            and unchanged
            and all(rejects.values())
        )
        detail = f"forward={tuple(output.shape)}, rollout={tuple(rollout.shape)}, input_unchanged={unchanged}, rejects={rejects}"
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_equivalence:strict_state_reload"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.get("seed", SEED))
        fixture = {
            "batch": torch.randn(2, 2, 3, 8, 8, device=device),
            "target": torch.randn(2, 3, 8, 8, device=device),
        }
        model_file = config.get("model_file", MODEL_FILE)
        model_class_name = config.get("model_class", MODEL_CLASS)
        model_args = config.get("model_args", {"variables": 3, "history": 2, "embed_dim": 16, "depth": 2, "heads": 4})
        module = _load_module("oneforecast_submission_model", (CANDIDATE_DIR / model_file).resolve())
        model_class = getattr(module, model_class_name)
        if not inspect.isclass(model_class):
            raise TypeError(f"{model_class_name} is not a class")
        model = model_class(**model_args).to(device)
        model.eval()
        clone = copy.deepcopy(model).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            left, right = model(fixture["batch"]), clone(fixture["batch"])
        ok = left.shape == right.shape and torch.allclose(left, right, atol=1e-6, rtol=1e-5)
        update_check(current_index, ok, f"max_abs_diff={float((left-right).abs().max())}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["traning:one_real_adamw_step"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.get("seed", SEED))
        fixture = {
            "batch": torch.randn(2, 2, 3, 8, 8, device=device),
            "target": torch.randn(2, 3, 8, 8, device=device),
        }
        model_file = config.get("model_file", MODEL_FILE)
        model_class_name = config.get("model_class", MODEL_CLASS)
        model_args = config.get("model_args", {"variables": 3, "history": 2, "embed_dim": 16, "depth": 2, "heads": 4})
        module = _load_module("oneforecast_submission_model", (CANDIDATE_DIR / model_file).resolve())
        model_class = getattr(module, model_class_name)
        if not inspect.isclass(model_class):
            raise TypeError(f"{model_class_name} is not a class")
        model = model_class(**model_args).to(device)
        model.train()
        parameters = [item for item in model.parameters() if item.requires_grad]
        before = [item.detach().clone() for item in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        output = model(fixture["batch"])
        loss = (output - fixture["target"]).square().mean()
        loss.backward()
        gradients = [item.grad for item in parameters if item.grad is not None]
        gradients_ok = bool(gradients) and all(bool(torch.isfinite(item).all()) for item in gradients)
        optimizer.step()
        changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, parameters))
        ok = bool(torch.isfinite(loss)) and gradients_ok and changed
        update_check(current_index, ok, f"loss={float(loss.detach()):.6g}, gradients={len(gradients)}, changed={changed}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["artifacts:importable_entry_points"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        train_file = config.get("train_file", "train.py")
        evaluate_file = config.get("evaluate_file", "evaluate.py")
        train = _load_module("oneforecast_submission_train", (CANDIDATE_DIR / train_file).resolve())
        evaluation = _load_module("oneforecast_submission_evaluate", (CANDIDATE_DIR / evaluate_file).resolve())
        ok = callable(getattr(train, "train_step", None)) and callable(getattr(evaluation, "evaluate", None))
        update_check(current_index, ok, "train_step and evaluate import successfully")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks
