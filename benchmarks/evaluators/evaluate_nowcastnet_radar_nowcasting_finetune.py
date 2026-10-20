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
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'data:inputs_finite',
        'data:targets_finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        input_shape, target_shape = ((2, 4, 8, 8), (2, 3, 8, 8))
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets)
        mask[..., :2, :2] = 1
        ok = bool(torch.isfinite(inputs).all() and torch.isfinite(targets).all())
        detail = f'input={input_shape}, target={target_shape}, mask={tuple(mask.shape)}'
        update_check(current_index, bool(torch.isfinite(inputs).all()), detail)
        current_index += 1
        update_check(current_index, bool(torch.isfinite(targets).all()), detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'file:nowcastnet_radar_nowcasting.py:exists',
        'file:nowcastnet_radar_nowcasting.py:syntax',
        'file:train.py:exists',
        'file:train.py:syntax',
        'file:evaluate.py:exists',
        'file:evaluate.py:syntax',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        case = {
            'model_file': 'nowcastnet_radar_nowcasting.py',
            'class_name': 'RadarNowcastNet',
            'env': 'NOWCASTNET_RADAR_SUBMISSION_DIR',
            'shape': ((2, 4, 8, 8), (2, 3, 8, 8)),
            'kwargs': {},
            'kind': 'radar',
        }
        for name in (case['model_file'], 'train.py', 'evaluate.py'):
            path = submission / name
            ok = path.is_file()
            detail = str(path)
            update_check(current_index, ok, detail)
            current_index += 1
            if ok:
                try:
                    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                    detail = f'syntax=ok, bytes={path.stat().st_size}'
                except Exception as exc:
                    ok, detail = False, f'{type(exc).__name__}: {exc}'
            else:
                detail = f'file does not exist: {path}'
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'train_script:parameter_model',
        'train_script:parameter_inputs',
        'train_script:parameter_targets',
        'train_script:call_backward',
        'train_script:call_step',
        'train_script:source_adamw',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        train_path = CANDIDATE_DIR / 'train.py'
        source = train_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'train_step')
        calls = {
            n.func.id if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            else (n.func.attr if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) else '')
            for n in ast.walk(node) if isinstance(n, ast.Call)
        }
        parameters = {arg.arg for arg in node.args.args}
        ok = {'model', 'inputs', 'targets'}.issubset(parameters) and {'backward', 'step'}.issubset(calls) and 'AdamW' in source
        detail = f'parameters={sorted(parameters)}, calls={sorted(calls)}'
        update_check(current_index, 'model' in parameters, detail)
        current_index += 1
        update_check(current_index, 'inputs' in parameters, detail)
        current_index += 1
        update_check(current_index, 'targets' in parameters, detail)
        current_index += 1
        update_check(current_index, 'backward' in calls, detail)
        current_index += 1
        update_check(current_index, 'step' in calls, detail)
        current_index += 1
        update_check(current_index, 'AdamW' in source, detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'eval_script:evaluate_function',
        'eval_script:rmse_metric',
        'eval_script:mae_metric',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        eval_path = CANDIDATE_DIR / 'evaluate.py'
        source = eval_path.read_text(encoding='utf-8')
        ok = 'def evaluate' in source and 'rmse' in source and 'mae' in source
        detail = 'evaluate/rmse/mae'
        update_check(current_index, 'def evaluate' in source, detail)
        current_index += 1
        update_check(current_index, 'rmse' in source, detail)
        current_index += 1
        update_check(current_index, 'mae' in source, detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'model_structure:children_in_order',
        'model_structure:pred_length',
        'model_structure:evolution_matches_generator',
        'model_structure:generator_shape',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = {
            'model_file': 'nowcastnet_radar_nowcasting.py',
            'class_name': 'RadarNowcastNet',
            'env': 'NOWCASTNET_RADAR_SUBMISSION_DIR',
            'shape': ((2, 4, 8, 8), (2, 3, 8, 8)),
            'kwargs': {},
            'kind': 'radar',
        }
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        inputs = torch.randn(*case['shape'][0]).to(device)
        model_path = CANDIDATE_DIR / case['model_file']
        if str(model_path.parent) not in sys.path:
            sys.path.insert(0, str(model_path.parent))
        spec = importlib.util.spec_from_file_location('weather_finetune_model', model_path)
        if spec is None or spec.loader is None:
            raise ImportError(model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules['weather_finetune_model'] = model_module
        spec.loader.exec_module(model_module)
        train_path = CANDIDATE_DIR / 'train.py'
        if str(train_path.parent) not in sys.path:
            sys.path.insert(0, str(train_path.parent))
        spec = importlib.util.spec_from_file_location('weather_finetune_train', train_path)
        if spec is None or spec.loader is None:
            raise ImportError(train_path)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['weather_finetune_train'] = train_module
        spec.loader.exec_module(train_module)
        model = train_module.build_model(case['kwargs']).to(device)
        children = dict(model.named_children())
        kind = case['kind']
        if kind == 'radar':
            with torch.no_grad():
                evolution = model.evolution(inputs)
                generated = model.generator(torch.cat([inputs, evolution], dim=1))
            ok = list(children) == ['evolution', 'generator'] and model.pred_length == 3 and evolution.shape == generated.shape == (2, 3, 8, 8)
            detail = f'children={list(children)}, evolution={tuple(evolution.shape)}, generated={tuple(generated.shape)}'
        update_check(current_index, list(children) == ['evolution', 'generator'], detail)
        current_index += 1
        update_check(current_index, model.pred_length == 3, detail)
        current_index += 1
        update_check(current_index, evolution.shape == generated.shape, detail)
        current_index += 1
        update_check(current_index, generated.shape == (2, 3, 8, 8), detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'forward_executable:output_shape',
        'forward_executable:output_finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = {
            'model_file': 'nowcastnet_radar_nowcasting.py',
            'class_name': 'RadarNowcastNet',
            'env': 'NOWCASTNET_RADAR_SUBMISSION_DIR',
            'shape': ((2, 4, 8, 8), (2, 3, 8, 8)),
            'kwargs': {},
            'kind': 'radar',
        }
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = case['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets)
        mask[..., :2, :2] = 1
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_path = CANDIDATE_DIR / case['model_file']
        if str(model_path.parent) not in sys.path:
            sys.path.insert(0, str(model_path.parent))
        spec = importlib.util.spec_from_file_location('weather_finetune_model', model_path)
        if spec is None or spec.loader is None:
            raise ImportError(model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules['weather_finetune_model'] = model_module
        spec.loader.exec_module(model_module)
        train_path = CANDIDATE_DIR / 'train.py'
        if str(train_path.parent) not in sys.path:
            sys.path.insert(0, str(train_path.parent))
        spec = importlib.util.spec_from_file_location('weather_finetune_train', train_path)
        if spec is None or spec.loader is None:
            raise ImportError(train_path)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['weather_finetune_train'] = train_module
        spec.loader.exec_module(train_module)
        model = train_module.build_model(case['kwargs']).to(device)
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        ok = output.shape == targets.shape and bool(torch.isfinite(output).all())
        detail = f'output={tuple(output.shape)}, expected={tuple(targets.shape)}'
        update_check(current_index, output.shape == targets.shape, detail)
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output).all()), detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'traning:loss_python_numeric',
        'traning:loss_not_nan',
        'traning:parameter_changed',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = {
            'model_file': 'nowcastnet_radar_nowcasting.py',
            'class_name': 'RadarNowcastNet',
            'env': 'NOWCASTNET_RADAR_SUBMISSION_DIR',
            'shape': ((2, 4, 8, 8), (2, 3, 8, 8)),
            'kwargs': {},
            'kind': 'radar',
        }
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = case['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets)
        mask[..., :2, :2] = 1
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_path = CANDIDATE_DIR / case['model_file']
        if str(model_path.parent) not in sys.path:
            sys.path.insert(0, str(model_path.parent))
        spec = importlib.util.spec_from_file_location('weather_finetune_model', model_path)
        if spec is None or spec.loader is None:
            raise ImportError(model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules['weather_finetune_model'] = model_module
        spec.loader.exec_module(model_module)
        train_path = CANDIDATE_DIR / 'train.py'
        if str(train_path.parent) not in sys.path:
            sys.path.insert(0, str(train_path.parent))
        spec = importlib.util.spec_from_file_location('weather_finetune_train', train_path)
        if spec is None or spec.loader is None:
            raise ImportError(train_path)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['weather_finetune_train'] = train_module
        spec.loader.exec_module(train_module)
        model = train_module.build_model(case['kwargs']).to(device)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        kwargs = {'optimizer': optimizer}
        if case['kind'] == 'extreme':
            kwargs['event_mask'] = mask
        loss = train_module.train_step(model, inputs, targets, **kwargs)
        after = [p.detach() for p in model.parameters() if p.requires_grad]
        changed = any(not torch.equal(left, right) for left, right in zip(before, after))
        ok = isinstance(loss, (float, int)) and loss == loss and changed
        detail = f'loss={loss}, parameter_changed={changed}'
        update_check(current_index, isinstance(loss, (float, int)), detail)
        current_index += 1
        update_check(current_index, loss == loss, detail)
        current_index += 1
        update_check(current_index, changed, detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks
