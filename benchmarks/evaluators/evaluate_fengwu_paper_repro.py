"""V2 adapter for the FengWu paper reproduction evaluator."""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
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


def _tensors(value) -> list[torch.Tensor]:
    if isinstance(value, (tuple, list)) and len(value) == 6 and all(isinstance(item, torch.Tensor) for item in value):
        return list(value)
    raise TypeError("FengWu output must be a six-tensor tuple/list")


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["file:fengwu.py", "file:train.py", "file:evaluate.py"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config) if isinstance(config, dict) else {}
        required_files = tuple(params.get("required_files", ("fengwu.py", "train.py", "evaluate.py")))
        forbidden = tuple(params.get("forbidden_tokens", ("codegen.evaluators", "codegen.standard")))
        encoding = params.get("encoding", "utf-8")
        for name in required_files:
            path = CANDIDATE_DIR / name
            ok, detail = path.is_file(), str(path)
            try:
                if ok:
                    source = path.read_text(encoding=encoding)
                    ast.parse(source, filename=str(path))
                    found = [token for token in forbidden if token in source]
                    ok = bool(source.strip()) and not found
                    detail = f"syntax=ok, lines={len(source.splitlines())}, forbidden={found}"
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "data:input_shapes",
        "data:target_shapes",
        "data:finite_values",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config) if isinstance(config, dict) else {}
        seed = params.get("seed", 20260819)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        inputs = (torch.randn(2, 4, 8, 8, device=device),) + tuple(torch.randn(2, 3, 8, 8, device=device) for _ in range(5))
        targets = tuple(torch.randn_like(item) for item in inputs)
        fixture = {"batch": inputs, "target": targets}
        input_shapes = [tuple(item.shape) for item in fixture["batch"]]
        target_shapes = [tuple(item.shape) for item in fixture["target"]]
        expected = [tuple(item) for item in params.get("input_shapes", [(2, 4, 8, 8)] + [(2, 3, 8, 8)] * 5)]
        expected_targets = [tuple(item) for item in params.get("target_shapes", expected)]
        ok = input_shapes == expected
        update_check(current_index, ok, f"inputs={input_shapes}")
        current_index += 1
        ok = target_shapes == expected_targets
        update_check(current_index, ok, f"targets={target_shapes}")
        current_index += 1
        ok = all(bool(torch.isfinite(item).all()) for item in (*inputs, *targets))
        update_check(current_index, ok, f"inputs={input_shapes}, targets={target_shapes}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_metrics(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "eval_script:parse",
        "eval_script:function",
        "eval_script:parameters",
        "eval_script:metrics",
        "eval_script:calls",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config) if isinstance(config, dict) else {}
        file_name = params.get("file", "evaluate.py")
        function_name = params.get("function", "evaluate")
        required_parameters = tuple(params.get("required_parameters", ("model", "inputs", "targets")))
        required_metric_strings = tuple(params.get("required_metric_strings", ("mae", "rmse")))
        required_calls = tuple(params.get("required_calls", ("abs", "mean", "sqrt")))
        path = CANDIDATE_DIR / file_name
        source = path.read_text(encoding=params.get("encoding", "utf-8"))
        tree = ast.parse(source, filename=str(path))
        update_check(current_index, True, "syntax=ok")
        current_index += 1
        node = _function(tree, function_name)
        ok = node is not None
        update_check(current_index, ok, f"function={function_name}")
        current_index += 1
        if node is None:
            raise AttributeError(f"{function_name} is missing")
        parameters = [arg.arg for arg in node.args.args]
        calls = _calls(node)
        strings = {child.value.lower() for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)}
        ok = all(name in parameters for name in required_parameters)
        update_check(current_index, ok, f"parameters={parameters}")
        current_index += 1
        ok = set(required_metric_strings).issubset(strings)
        update_check(current_index, ok, f"metrics={sorted(strings & set(required_metric_strings))}")
        current_index += 1
        ok = set(required_calls).issubset(calls)
        update_check(current_index, ok, f"calls={sorted(calls)}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "artifacts:train_import",
        "artifacts:evaluate_import",
        "artifacts:train_step_callable",
        "artifacts:evaluate_callable",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config) if isinstance(config, dict) else {}
        files = tuple(params.get("files", ("train.py", "evaluate.py")))
        train_file = files[0] if files else "train.py"
        evaluate_file = files[1] if len(files) > 1 else "evaluate.py"
        train = _load_module("fengwu_submission_train", CANDIDATE_DIR / train_file)
        update_check(current_index, True, "train.py imported successfully")
        current_index += 1
        evaluation = _load_module("fengwu_submission_evaluate", CANDIDATE_DIR / evaluate_file)
        update_check(current_index, True, "evaluate.py imported successfully")
        current_index += 1
        train_step = getattr(train, "train_step", None)
        ok = callable(train_step)
        update_check(current_index, ok, "train_step")
        current_index += 1
        evaluate = getattr(evaluation, "evaluate", None)
        ok = callable(evaluate)
        update_check(current_index, ok, "evaluate")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "model_structure:model_class",
        "model_structure:children_subsequence",
        "model_structure:encoder_module_list",
        "model_structure:decoder_module_list",
        "model_structure:encoder_decoder_lengths",
        "model_structure:encoder_channels",
        "model_structure:decoder_channels",
        "model_structure:encoder_kernel_stride",
        "model_structure:fuser_conv3d",
        "model_structure:invalid_grid_rejected",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config) if isinstance(config, dict) else {}
        model_file = params.get("model_file", "fengwu.py")
        model_class_name = params.get("model_class", "Fengwu")
        constructor_args = dict(params.get("constructor_args", {"pressure_level": 3, "embed_dim": 8, "img_size": (8, 8), "patch_size": 2}))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        module = _load_module("fengwu_submission_model_structure", CANDIDATE_DIR / model_file)
        model_class = getattr(module, model_class_name)
        ok = inspect.isclass(model_class)
        update_check(current_index, ok, f"model_class={model_class_name}")
        current_index += 1
        if not ok:
            raise TypeError(f"{model_class_name} is not a class")
        model = model_class(**constructor_args).to(device)
        children = list(dict(model.named_children()))
        encoders, decoders = getattr(model, "encoders"), getattr(model, "decoders")
        fuser = getattr(model, "fuser")
        encoder_channels = [item.in_channels for item in encoders]
        decoder_channels = [item.out_channels for item in decoders]
        conv3d_layers = [item for item in fuser.modules() if isinstance(item, torch.nn.Conv3d)]
        invalid_grid_args = dict(constructor_args)
        invalid_grid_args["img_size"] = (7, 8)
        invalid_grid_rejected = _rejects(lambda: model.__class__(**invalid_grid_args))
        ok = _subsequence(children, tuple(params.get("children_subsequence", ("encoders", "fuser", "decoders"))))
        update_check(current_index, ok, f"children={children}")
        current_index += 1
        ok = isinstance(encoders, torch.nn.ModuleList)
        update_check(current_index, ok, str(type(encoders)))
        current_index += 1
        ok = isinstance(decoders, torch.nn.ModuleList)
        update_check(current_index, ok, str(type(decoders)))
        current_index += 1
        ok = len(encoders) == len(decoders) == int(params.get("encoder_decoder_length", 6))
        update_check(current_index, ok, f"encoders={len(encoders)}, decoders={len(decoders)}")
        current_index += 1
        ok = encoder_channels == list(params.get("encoder_in_channels", [4, 3, 3, 3, 3, 3]))
        update_check(current_index, ok, f"encoder_channels={encoder_channels}")
        current_index += 1
        ok = decoder_channels == list(params.get("decoder_out_channels", [4, 3, 3, 3, 3, 3]))
        update_check(current_index, ok, f"decoder_channels={decoder_channels}")
        current_index += 1
        kernel_size = tuple(params.get("encoder_kernel_size", (2, 2)))
        stride = tuple(params.get("encoder_stride", (2, 2)))
        ok = all(item.kernel_size == kernel_size and item.stride == stride for item in encoders)
        update_check(current_index, ok, f"kernel_size={kernel_size}, stride={stride}")
        current_index += 1
        ok = len(conv3d_layers) >= int(params.get("fuser_conv3d_min", 2))
        update_check(current_index, ok, f"conv3d={len(conv3d_layers)}")
        current_index += 1
        ok = invalid_grid_rejected
        update_check(current_index, ok, f"invalid_grid_rejected={invalid_grid_rejected}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "forward_executable:six_tensor_output",
        "forward_executable:shapes",
        "forward_executable:finite",
        "forward_executable:batch_rejected",
        "forward_executable:channels_rejected",
        "forward_executable:grid_rejected",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config) if isinstance(config, dict) else {}
        model_file = params.get("model_file", "fengwu.py")
        model_class_name = params.get("model_class", "Fengwu")
        constructor_args = dict(params.get("constructor_args", {"pressure_level": 3, "embed_dim": 8, "img_size": (8, 8), "patch_size": 2}))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(params.get("seed", 20260819))
        inputs = (torch.randn(2, 4, 8, 8, device=device),) + tuple(torch.randn(2, 3, 8, 8, device=device) for _ in range(5))
        targets = tuple(torch.randn_like(item) for item in inputs)
        fixture = {"batch": inputs, "target": targets}
        module = _load_module("fengwu_submission_model_forward", CANDIDATE_DIR / model_file)
        model_class = getattr(module, model_class_name)
        if not inspect.isclass(model_class):
            raise TypeError(f"{model_class_name} is not a class")
        model = model_class(**constructor_args).to(device)
        model.eval()
        with torch.no_grad():
            outputs = _tensors(model(*fixture["batch"]))
        update_check(current_index, True, "six tensor tuple/list")
        current_index += 1
        shapes = [tuple(item.shape) for item in outputs]
        expected_shapes = [tuple(item.shape) for item in fixture["target"]]
        ok = shapes == expected_shapes
        update_check(current_index, ok, f"shapes={shapes}")
        current_index += 1
        ok = all(bool(torch.isfinite(item).all()) for item in outputs)
        update_check(current_index, ok, f"shapes={shapes}")
        current_index += 1
        invalid_batch = list(inputs)
        invalid_batch[2] = invalid_batch[2][:1]
        invalid_channels = list(inputs)
        invalid_channels[3] = torch.randn(2, 2, 8, 8, device=inputs[0].device)
        invalid_grid = list(inputs)
        invalid_grid[4] = invalid_grid[4][:, :, :6]
        rejects = {
            "batch": _rejects(lambda: model(*invalid_batch)),
            "channels": _rejects(lambda: model(*invalid_channels)),
            "grid": _rejects(lambda: model(*invalid_grid)),
        }
        update_check(current_index, rejects["batch"], f"rejects={rejects}")
        current_index += 1
        update_check(current_index, rejects["channels"], f"rejects={rejects}")
        current_index += 1
        update_check(current_index, rejects["grid"], f"rejects={rejects}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "forward_equivalence:six_tensor_outputs",
        "forward_equivalence:allclose",
        "forward_equivalence:max_abs_diff",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config) if isinstance(config, dict) else {}
        model_file = params.get("model_file", "fengwu.py")
        model_class_name = params.get("model_class", "Fengwu")
        constructor_args = dict(params.get("constructor_args", {"pressure_level": 3, "embed_dim": 8, "img_size": (8, 8), "patch_size": 2}))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(params.get("seed", 20260819))
        inputs = (torch.randn(2, 4, 8, 8, device=device),) + tuple(torch.randn(2, 3, 8, 8, device=device) for _ in range(5))
        module = _load_module("fengwu_submission_model_equivalence", CANDIDATE_DIR / model_file)
        model_class = getattr(module, model_class_name)
        if not inspect.isclass(model_class):
            raise TypeError(f"{model_class_name} is not a class")
        model = model_class(**constructor_args).to(device)
        model.eval()
        clone = copy.deepcopy(model).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            left, right = _tensors(model(*inputs)), _tensors(clone(*inputs))
        ok = len(left) == len(right) == int(params.get("expected_output_count", 6))
        update_check(current_index, ok, f"left={len(left)}, right={len(right)}")
        current_index += 1
        atol = params.get("allclose", {}).get("atol", 1e-6)
        rtol = params.get("allclose", {}).get("rtol", 1e-5)
        ok = all(torch.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right))
        update_check(current_index, ok, f"atol={atol}, rtol={rtol}")
        current_index += 1
        max_diff = max(float((a - b).abs().max()) for a, b in zip(left, right))
        update_check(current_index, max_diff <= atol + rtol * max(1.0, max_diff), f"max_abs_diff={max_diff}")
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    params = config.get("params", config) if isinstance(config, dict) else {}
    static_mode = "function" in params or "required_calls" in params
    CHECKPOINT_NAMES = [
        "train_script:parse",
        "train_script:function",
        "train_script:parameters",
        "train_script:calls",
    ] if static_mode else [
        "traning:finite_loss",
        "traning:finite_gradients",
        "traning:parameter_changed",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        if static_mode:
            files = tuple(params.get("files", ("train.py",)))
            path = CANDIDATE_DIR / (files[0] if files else "train.py")
            source = path.read_text(encoding=params.get("encoding", "utf-8"))
            tree = ast.parse(source, filename=str(path))
            update_check(current_index, True, "syntax=ok")
            current_index += 1
            function_name = params.get("function", "train_step")
            node = _function(tree, function_name)
            ok = node is not None
            update_check(current_index, ok, f"function={function_name}")
            current_index += 1
            if node is None:
                raise AttributeError(f"{function_name} is missing")
            parameters = [arg.arg for arg in node.args.args]
            required_parameters = tuple(params.get("required_parameters", ("model", "inputs", "targets")))
            calls = _calls(node)
            ok = all(name in parameters for name in required_parameters)
            update_check(current_index, ok, f"parameters={parameters}")
            current_index += 1
            required_calls = tuple(params.get("required_calls", ("AdamW", "backward", "step")))
            ok = set(required_calls).issubset(calls)
            update_check(current_index, ok, f"calls={sorted(calls)}")
            current_index += 1
        else:
            model_file = params.get("model_file", "fengwu.py")
            model_class_name = params.get("model_class", "Fengwu")
            constructor_args = dict(params.get("constructor_args", {"pressure_level": 3, "embed_dim": 8, "img_size": (8, 8), "patch_size": 2}))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(params.get("seed", 20260819))
            inputs = (torch.randn(2, 4, 8, 8, device=device),) + tuple(torch.randn(2, 3, 8, 8, device=device) for _ in range(5))
            targets = tuple(torch.randn_like(item) for item in inputs)
            module = _load_module("fengwu_submission_model_training", CANDIDATE_DIR / model_file)
            model_class = getattr(module, model_class_name)
            if not inspect.isclass(model_class):
                raise TypeError(f"{model_class_name} is not a class")
            model = model_class(**constructor_args).to(device)
            model.train()
            parameters = [item for item in model.parameters() if item.requires_grad]
            before = [item.detach().clone() for item in parameters]
            optimizer = torch.optim.AdamW(parameters, lr=params.get("learning_rate", 1e-3))
            optimizer.zero_grad(set_to_none=True)
            outputs, targets = _tensors(model(*inputs)), _tensors(targets)
            loss = sum((output - target).square().mean() for output, target in zip(outputs, targets))
            loss.backward()
            gradients = [item.grad for item in parameters if item.grad is not None]
            gradients_ok = bool(gradients) and all(bool(torch.isfinite(item).all()) for item in gradients)
            optimizer.step()
            changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, parameters))
            ok = bool(torch.isfinite(loss))
            update_check(current_index, ok, f"loss={float(loss.detach()):.6g}")
            current_index += 1
            update_check(current_index, gradients_ok, f"gradients={len(gradients)}")
            current_index += 1
            update_check(current_index, changed, f"changed={changed}")
            current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks
