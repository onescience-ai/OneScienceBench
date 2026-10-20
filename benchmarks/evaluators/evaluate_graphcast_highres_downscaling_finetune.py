"""V2 adapter for the GraphCast high-resolution downscaling evaluator."""
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
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["data:synthetic_weather_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config)
        input_shape = tuple(params.get("input_shape", (2, 5, 4, 4)))
        target_shape = tuple(params.get("target_shape", (2, 2, 8, 8)))
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        ok = bool(torch.isfinite(inputs).all() and torch.isfinite(targets).all())
        msg = f"input={input_shape}, target={target_shape}, mask={tuple(mask.shape)}"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "file:graphcast_highres_downscaling.py",
        "file:train.py",
        "file:evaluate.py",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config)
        case = {
            "model_file": params.get("model_file", "graphcast_highres_downscaling.py"),
        }
        submission = CANDIDATE_DIR
        required_files = params.get(
            "required_files", (case["model_file"], "train.py", "evaluate.py")
        )
        for name in required_files:
            path = (submission / name).resolve()
            path.relative_to(CANDIDATE_DIR)
            ok = path.is_file()
            detail = str(path)
            if ok:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                detail = f"syntax=ok, bytes={path.stat().st_size}"
            update_check(current_index, ok, detail)
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
        params = config.get("params", config)
        eval_path = (CANDIDATE_DIR / params.get("file", "evaluate.py")).resolve()
        eval_path.relative_to(CANDIDATE_DIR)
        source = eval_path.read_text(encoding="utf-8")
        ok = "def evaluate" in source and "rmse" in source and "mae" in source
        update_check(current_index, ok, "evaluate/rmse/mae")
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
        params = config.get("params", config)
        train_path = (CANDIDATE_DIR / params.get("file", "train.py")).resolve()
        train_path.relative_to(CANDIDATE_DIR)
        source = train_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_name = params.get("function", "train_step")
        node = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == function_name
        )
        calls = {
            n.func.id if isinstance(n.func, ast.Call) and isinstance(n.func, ast.Name) else (
                n.func.attr if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) else ""
            )
            for n in ast.walk(node) if isinstance(n, ast.Call)
        }
        parameters = {arg.arg for arg in node.args.args}
        ok = {"model", "inputs", "targets"}.issubset(parameters) and {"backward", "step"}.issubset(calls) and "AdamW" in source
        msg = f"parameters={sorted(parameters)}, calls={sorted(calls)}"
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
        params = config.get("params", config)
        model_builder = params.get("model_builder", {})
        case = {
            "model_file": params.get("model_file", "graphcast_highres_downscaling.py"),
            "shape": (
                tuple(params.get("input_shape", (2, 5, 4, 4))),
                tuple(params.get("target_shape", (2, 2, 8, 8))),
            ),
            "kwargs": model_builder.get("kwargs", params.get("kwargs", {})),
            "kind": "highres",
        }
        input_shape, target_shape = case["shape"]
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_module = train_module = eval_module = None
        model_path = (CANDIDATE_DIR / case["model_file"]).resolve()
        model_path.relative_to(CANDIDATE_DIR)
        train_path = (CANDIDATE_DIR / model_builder.get("file", "train.py")).resolve()
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
            encoded = model.grid_encoder(inputs); processed = encoded
            for block in model.mesh_processor: processed = processed + torch.nn.functional.gelu(block(processed))
            upsampled = torch.nn.functional.interpolate(processed, scale_factor=2, mode="bilinear", align_corners=False); decoded = model.grid_decoder(upsampled)
        ok = list(children) == ["grid_encoder", "mesh_processor", "grid_decoder"] and isinstance(model.mesh_processor, torch.nn.ModuleList) and len(model.mesh_processor) == 2 and model.upscale_factor == 2 and encoded.shape == processed.shape == (2, 16, 4, 4) and decoded.shape == (2, 2, 8, 8)
        msg = f"children={list(children)}, encoded={tuple(encoded.shape)}, processed={tuple(processed.shape)}, decoded={tuple(decoded.shape)}"
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
        params = config.get("params", config)
        model_builder = params.get("model_builder", {})
        case = {
            "model_file": params.get("model_file", "graphcast_highres_downscaling.py"),
            "shape": (
                tuple(params.get("input_shape", (2, 5, 4, 4))),
                tuple(params.get("target_shape", (2, 2, 8, 8))),
            ),
            "kwargs": model_builder.get("kwargs", params.get("kwargs", {})),
            "kind": "highres",
        }
        input_shape, target_shape = case["shape"]
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_module = train_module = eval_module = None
        model_path = (CANDIDATE_DIR / case["model_file"]).resolve()
        model_path.relative_to(CANDIDATE_DIR)
        train_path = (CANDIDATE_DIR / model_builder.get("file", "train.py")).resolve()
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


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["traning:real_backward_optimizer_update"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get("params", config)
        model_builder = params.get("model_builder", {})
        case = {
            "model_file": params.get("model_file", "graphcast_highres_downscaling.py"),
            "shape": (
                tuple(params.get("input_shape", (2, 5, 4, 4))),
                tuple(params.get("target_shape", (2, 2, 8, 8))),
            ),
            "kwargs": model_builder.get("kwargs", {}),
            "kind": "highres",
        }
        input_shape, target_shape = case["shape"]
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_path = (CANDIDATE_DIR / case["model_file"]).resolve()
        model_path.relative_to(CANDIDATE_DIR)
        train_path = (CANDIDATE_DIR / model_builder.get("file", params.get("file", "train.py"))).resolve()
        train_path.relative_to(CANDIDATE_DIR)
        model_module_name = model_path.stem
        train_module_name = "_graphcast_highres_training_check"
        previous_model_module = sys.modules.get(model_module_name)
        previous_train_module = sys.modules.get(train_module_name)
        try:
            spec = importlib.util.spec_from_file_location(model_module_name, model_path)
            if spec is None or spec.loader is None:
                raise ImportError(model_path)
            model_module = importlib.util.module_from_spec(spec)
            sys.modules[model_module_name] = model_module
            spec.loader.exec_module(model_module)
            spec = importlib.util.spec_from_file_location(train_module_name, train_path)
            if spec is None or spec.loader is None:
                raise ImportError(train_path)
            train_module = importlib.util.module_from_spec(spec)
            sys.modules[train_module_name] = train_module
            spec.loader.exec_module(train_module)
            train_step = train_module.train_step
            signature = inspect.signature(train_step)
            accepts_optimizer = "optimizer" in signature.parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if not accepts_optimizer:
                raise TypeError(f"{train_path}: train_step does not accept optimizer")
            model = train_module.build_model(case["kwargs"]).to(device)
            model.train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
            kwargs = {"optimizer": optimizer}
            if case["kind"] == "extreme":
                kwargs["event_mask"] = mask
            loss = train_step(model, inputs, targets, **kwargs)
            after = [p.detach() for p in model.parameters() if p.requires_grad]
            changed = any(not torch.equal(left, right) for left, right in zip(before, after))
            ok = isinstance(loss, (float, int)) and loss == loss and changed
            msg = f"loss={loss}, parameter_changed={changed}"
            update_check(current_index, ok, msg)
            current_index += 1
        finally:
            if previous_model_module is None:
                sys.modules.pop(model_module_name, None)
            else:
                sys.modules[model_module_name] = previous_model_module
            if previous_train_module is None:
                sys.modules.pop(train_module_name, None)
            else:
                sys.modules[train_module_name] = previous_train_module
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
