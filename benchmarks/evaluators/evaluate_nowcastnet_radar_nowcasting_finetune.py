"""V2 adapter for the legacy NowcastNet radar evaluator."""
from __future__ import annotations

from pathlib import Path
import sys

import ast
import importlib.util
import inspect
import json
import os
import traceback
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

def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["data:inputs_finite", "data:targets_finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    shape_spec = params.get("shape", {})
    input_shape = shape_spec.get("inputs") if isinstance(shape_spec, dict) else None
    target_shape = shape_spec.get("targets") if isinstance(shape_spec, dict) else None
    if input_shape is None or target_shape is None:
        checks[0] = (CHECKPOINT_NAMES[0], False, "shape.inputs and shape.targets are required")
        return checks

    try:
        inputs = torch.randn(*tuple(input_shape))
        checks[0] = (
            CHECKPOINT_NAMES[0],
            bool(torch.isfinite(inputs).all()),
            f"input shape={tuple(inputs.shape)}",
        )
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))

    try:
        targets = torch.randn(*tuple(target_shape))
        checks[1] = (
            CHECKPOINT_NAMES[1],
            bool(torch.isfinite(targets).all()),
            f"target shape={tuple(targets.shape)}",
        )
    except Exception as error:
        checks[1] = (CHECKPOINT_NAMES[1], False, str(error))
    return checks


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    required_files = params.get("required_files", [])
    if isinstance(required_files, str):
        required_files = [required_files]
    checkpoint_names = []
    for name in required_files:
        checkpoint_names.extend([f"file:{name}:exists", f"file:{name}:syntax"])
    if not checkpoint_names:
        return [("files.config", False, "required_files is missing or empty")]
    checks = [(name, False, "Not executed due to earlier failure") for name in checkpoint_names]

    for offset, name in enumerate(required_files):
        exists_index = offset * 2
        syntax_index = exists_index + 1
        try:
            path = (CANDIDATE_DIR / name).resolve()
            exists = path.is_file()
            checks[exists_index] = (
                checkpoint_names[exists_index],
                exists and path.stat().st_size > 0,
                str(path),
            )
            if not exists or path.stat().st_size == 0:
                checks[syntax_index] = (checkpoint_names[syntax_index], False, "file is missing or empty")
                continue
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            checks[syntax_index] = (checkpoint_names[syntax_index], True, "syntax/content valid")
        except Exception as error:
            checks[syntax_index] = (checkpoint_names[syntax_index], False, str(error))
    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = [
        "train_script:parameter_model", "train_script:parameter_inputs",
        "train_script:parameter_targets", "train_script:call_backward",
        "train_script:call_step", "train_script:source_adamw",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    files = params.get("files", [])
    if isinstance(files, str):
        files = [files]
    try:
        sources = [((CANDIDATE_DIR / file_name).read_text(encoding="utf-8"), file_name) for file_name in files]
        trees = [(ast.parse(source, filename=file_name), source) for source, file_name in sources]
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
        return checks

    calls = []
    source_text = "\n".join(source for _, source in trees)
    for tree, _ in trees:
        calls.extend(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    call_names = {
        node.func.id.lower() if isinstance(node.func, ast.Name) else node.func.attr.lower()
        for node in calls
        if isinstance(node.func, (ast.Name, ast.Attribute))
    }
    has_optimizer = any(name in {"adamw", "adam", "sgd", "rmsprop"} for name in call_names)
    has_model_parameters = "parameters" in call_names or "model" in source_text.lower()
    lowered = source_text.lower()
    has_mse = (
        "mse" in lowered
        or "mseloss" in lowered
        or ("square" in lowered and "mean" in lowered)
        or ("pow" in lowered and "mean" in lowered)
    )
    details = f"files={list(files)}, calls={sorted(call_names)}"
    results = [
        has_model_parameters and has_optimizer,
        has_mse,
        has_mse,
        "backward" in call_names,
        "step" in call_names,
        "adamw" in call_names or "adamw" in lowered,
    ]
    for index, passed in enumerate(results):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), details)
    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["eval_script:evaluate_function", "eval_script:rmse_metric", "eval_script:mae_metric"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    files = params.get("files", [])
    if isinstance(files, str):
        files = [files]
    required = params.get("required_interfaces", [])
    modules = []
    source_text = ""
    for file_name in files:
        try:
            path = (CANDIDATE_DIR / file_name).resolve()
            spec = importlib.util.spec_from_file_location(f"candidate_interface_{Path(file_name).stem}", path)
            if spec is None or spec.loader is None:
                raise ImportError(str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules.append(module)
            source_text += "\n" + path.read_text(encoding="utf-8")
        except Exception as error:
            source_text += f"\n{error}"

    evaluate_required = "evaluate" in required or not required
    evaluate_found = any(callable(getattr(module, "evaluate", None)) for module in modules)
    checks[0] = (CHECKPOINT_NAMES[0], evaluate_found if evaluate_required else True, "evaluate callable")
    lowered = source_text.lower()
    rmse_semantics = any(token in lowered for token in ("rmse", "root_mean", "sqrt", "mean_squared"))
    mae_semantics = any(token in lowered for token in ("mae", "mean_absolute", "absolute_error"))
    checks[1] = (
        CHECKPOINT_NAMES[1],
        bool(evaluate_found and rmse_semantics),
        "RMSE computation/field detected",
    )
    checks[2] = (
        CHECKPOINT_NAMES[2],
        bool(evaluate_found and mae_semantics),
        "MAE computation/field detected",
    )
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = [
        "model_structure:children_in_order", "model_structure:pred_length",
        "model_structure:evolution_matches_generator", "model_structure:generator_shape",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    model = None
    try:
        model_path = (CANDIDATE_DIR / params["model_file"]).resolve()
        spec = importlib.util.spec_from_file_location(f"candidate_structure_{model_path.stem}", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(str(model_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model_class = getattr(module, params["model_class"], None)
        if not callable(model_class):
            raise TypeError(f"model class {params['model_class']} is not callable")
        model = model_class(**params.get("model_args", {})).to(DEVICE).eval()
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
        return checks

    def output_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for item in value.values():
                found = output_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for item in value:
                found = output_tensor(item)
                if found is not None:
                    return found
        try:
            return torch.as_tensor(value)
        except Exception:
            return None

    rules = params.get("rules", [])
    for index, rule in enumerate(rules[:len(CHECKPOINT_NAMES)]):
        try:
            rule_params = rule.get("params", {})
            rule_type = rule.get("type", "")
            if rule_type in {"required_attributes", "children_count", "children_order"}:
                attributes = rule_params.get("attributes", ["evolution", "generator"])
                passed = all(hasattr(model, name) for name in attributes)
                message = f"required attributes={attributes}"
            elif rule_type in {"callable_attributes", "required_interfaces"}:
                attributes = rule_params.get("attributes", ["evolution", "generator"])
                passed = all(callable(getattr(model, name, None)) for name in attributes)
                message = f"callable attributes={attributes}"
            elif rule_type in {"optional_attribute_equals", "attribute_equals"}:
                attribute = rule_params.get("attribute", "pred_length")
                expected = rule_params.get("value", 3)
                actual = getattr(model, attribute, expected)
                passed = actual == expected
                message = f"{attribute}={actual}, expected={expected}"
            elif rule_type in {"forward_shape", "output_shape", "generator_shape"}:
                shape = tuple(rule_params["input_shape"])
                expected = tuple(rule_params["expected_shape"])
                with torch.no_grad():
                    output = output_tensor(model(torch.randn(*shape, device=DEVICE)))
                passed = output is not None and tuple(output.shape) == expected
                message = f"output={None if output is None else tuple(output.shape)}, expected={expected}"
            else:
                passed = False
                message = f"unsupported structure rule: {rule_type}"
            checks[index] = (CHECKPOINT_NAMES[index], bool(passed), message)
        except Exception as error:
            checks[index] = (CHECKPOINT_NAMES[index], False, str(error))
    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["forward_executable:output_shape", "forward_executable:output_finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    output = None
    try:
        model_path = (CANDIDATE_DIR / params["model_file"]).resolve()
        spec = importlib.util.spec_from_file_location(f"candidate_forward_{model_path.stem}", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(str(model_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        context = {}
        for hook in params.get("pre_forward_hooks", []):
            function = getattr(module, hook["function"], None)
            if not callable(function):
                raise AttributeError(f"hook {hook['function']} is not callable")
            hook_args = dict(hook.get("args", {}))
            result = function(**hook_args)
            if hook.get("save_as"):
                context[hook["save_as"]] = result
        model_class = getattr(module, params["model_class"], None)
        if not callable(model_class):
            raise TypeError(f"model class {params['model_class']} is not callable")
        model = model_class(**params.get("model_args", {})).to(DEVICE).eval()
        input_shape = tuple(params["input_shape"])
        inputs = torch.randn(*input_shape, device=DEVICE)
        forward_args = params.get("forward_args", {})
        if forward_args:
            resolved = {
                key: (context.get(value[1:]) if isinstance(value, str) and value.startswith("$") else value)
                for key, value in forward_args.items()
            }
            resolved = {
                key: (inputs if value is None and key == "input" else value)
                for key, value in resolved.items()
            }
            with torch.no_grad():
                output = model(**resolved)
        else:
            with torch.no_grad():
                output = model(inputs)
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
        return checks

    def output_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for item in value.values():
                found = output_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for item in value:
                found = output_tensor(item)
                if found is not None:
                    return found
        try:
            return torch.as_tensor(value, device=DEVICE)
        except Exception:
            return None

    tensor = output_tensor(output)
    expected_shape = params.get("out_shape")
    shape_ok = tensor is not None and (expected_shape is None or tuple(tensor.shape) == tuple(expected_shape))
    checks[0] = (
        CHECKPOINT_NAMES[0],
        bool(shape_ok),
        f"output={None if tensor is None else tuple(tensor.shape)}, expected={expected_shape}",
    )
    try:
        finite_ok = tensor is not None and bool(torch.isfinite(tensor).all())
        checks[1] = (CHECKPOINT_NAMES[1], finite_ok, "finite output" if finite_ok else "output is not finite")
    except Exception as error:
        checks[1] = (CHECKPOINT_NAMES[1], False, str(error))
    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["traning:loss_python_numeric", "traning:loss_not_nan", "traning:parameter_changed"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    candidate = params.get("candidate", {})
    try:
        train_path = (CANDIDATE_DIR / candidate["file"]).resolve()
        train_spec = importlib.util.spec_from_file_location(f"candidate_training_{train_path.stem}", train_path)
        if train_spec is None or train_spec.loader is None:
            raise ImportError(str(train_path))
        train_module = importlib.util.module_from_spec(train_spec)
        sys.modules[train_spec.name] = train_module
        train_spec.loader.exec_module(train_module)

        model_path = (CANDIDATE_DIR / candidate["model_file"]).resolve()
        model_spec = importlib.util.spec_from_file_location(f"candidate_training_model_{model_path.stem}", model_path)
        if model_spec is None or model_spec.loader is None:
            raise ImportError(str(model_path))
        model_module = importlib.util.module_from_spec(model_spec)
        sys.modules[model_spec.name] = model_module
        model_spec.loader.exec_module(model_module)
        model_class = getattr(model_module, candidate["model_class"], None)
        if not callable(model_class):
            raise TypeError(f"model class {candidate['model_class']} is not callable")
        model = model_class(**candidate.get("model_args", {})).to(DEVICE).train()
        inputs = torch.randn(*tuple(candidate["input_shape"]), device=DEVICE)
        targets = torch.randn(*tuple(candidate["target_shape"]), device=DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(candidate.get("optimizer", {}).get("lr", 1e-3)))

        function_name = candidate.get("function")
        train_function = getattr(train_module, function_name, None) if function_name else None
        if not callable(train_function):
            functions = [item for item in vars(train_module).values() if inspect.isfunction(item)]
            train_function = next(
                (item for item in functions if "train" in item.__name__.lower() or "backward" in inspect.getsource(item)),
                None,
            )
        if not callable(train_function):
            raise AttributeError("no training callable found")

        before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
        signature = inspect.signature(train_function)
        positional_fallback = [model, inputs, targets, optimizer]
        call_args = []
        call_kwargs = {}
        for parameter in signature.parameters.values():
            if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                continue
            name = parameter.name.lower()
            if "optim" in name:
                value = optimizer
            elif "model" in name or name in {"net", "module"}:
                value = model
            elif "input" in name or name in {"x", "data"}:
                value = inputs
            elif "target" in name or "label" in name or name in {"y", "truth"}:
                value = targets
            elif parameter.default is not inspect.Parameter.empty:
                continue
            elif positional_fallback:
                value = positional_fallback.pop(0)
            else:
                raise TypeError(f"cannot resolve required training argument {parameter.name}")
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                call_kwargs[parameter.name] = value
            else:
                call_args.append(value)
        result = train_function(*call_args, **call_kwargs)
        after = [parameter.detach() for parameter in model.parameters() if parameter.requires_grad]
        changed = bool(any(not torch.equal(left, right) for left, right in zip(before, after)))

        def numeric(value):
            if isinstance(value, (float, int)):
                return float(value)
            if isinstance(value, torch.Tensor):
                return float(value.detach().mean().cpu()) if value.numel() else None
            if isinstance(value, dict):
                for key in ("loss", "objective", "cost"):
                    if key in value:
                        found = numeric(value[key])
                        if found is not None:
                            return found
                for item in value.values():
                    found = numeric(item)
                    if found is not None:
                        return found
            if isinstance(value, (list, tuple)):
                for item in value:
                    found = numeric(item)
                    if found is not None:
                        return found
            return None

        loss_value = numeric(result)
        if loss_value is None:
            with torch.no_grad():
                prediction = model(inputs)
                if isinstance(prediction, (list, tuple)):
                    prediction = next(item for item in prediction if isinstance(item, torch.Tensor))
                loss_value = float((prediction - targets).square().mean().detach().cpu())
        numeric_ok = isinstance(loss_value, (float, int))
        finite_ok = numeric_ok and loss_value == loss_value and abs(loss_value) != float("inf")
        detail = f"loss={loss_value}, parameter_changed={changed}"
        checks[0] = (CHECKPOINT_NAMES[0], bool(numeric_ok), detail)
        checks[1] = (CHECKPOINT_NAMES[1], bool(finite_ok), detail)
        checks[2] = (CHECKPOINT_NAMES[2], changed, detail)
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
    return checks
