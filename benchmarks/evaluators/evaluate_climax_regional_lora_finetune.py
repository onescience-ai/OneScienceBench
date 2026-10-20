'''V2 adapter for the ClimaX regional LoRA fine-tuning evaluator.'''
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

CASE = {
    'model_file': 'climax_regional_lora.py',
    'class_name': 'ClimaXRegionalLoRA',
    'env': 'CLIMAX_REGIONAL_LORA_FINETUNE_SUBMISSION_DIR',
    'shape': ((2, 4, 8, 8), (2, 2, 8, 8)),
    'kwargs': {},
    'kind': 'lora',
}


def _load(name, path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['data:synthetic_weather_contract']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = CASE['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        ok = bool(torch.isfinite(inputs).all() and torch.isfinite(targets).all())
        update_check(current_index, ok, f'input={input_shape}, target={target_shape}, mask={tuple(mask.shape)}')
        current_index += 1
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:climax_regional_lora.py', 'file:train.py', 'file:evaluate.py']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        submission = CANDIDATE_DIR
        for name in (CASE['model_file'], 'train.py', 'evaluate.py'):
            path = (submission / name).resolve()
            ok = path.is_file()
            detail = str(path)
            if ok:
                try:
                    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                    detail = f'syntax=ok, bytes={path.stat().st_size}'
                except Exception:
                    ok, detail = False, traceback.format_exc()
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_metrics(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['eval_script:metric_interface']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        source = (CANDIDATE_DIR / 'evaluate.py').resolve().read_text(encoding='utf-8')
        ok = 'def evaluate' in source and 'rmse' in source and 'mae' in source
        update_check(current_index, ok, 'evaluate/rmse/mae')
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['train_script:modules_importable', 'train_script:build_model']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        submission = CANDIDATE_DIR
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_module = train_module = eval_module = None
        model_module = _load('weather_finetune_model', (submission / CASE['model_file']).resolve())
        train_module = _load('weather_finetune_train', (submission / 'train.py').resolve())
        eval_module = _load('weather_finetune_eval', (submission / 'evaluate.py').resolve())
        update_check(current_index, bool(model_module and train_module and eval_module), 'modules importable')
        current_index += 1
        if model_module and train_module:
            model = train_module.build_model(CASE['kwargs']).to(device)
            update_check(current_index, True, str(type(model)))
            current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['forward_executable:shape_and_finite']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        submission = CANDIDATE_DIR
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = CASE['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        train_module = _load('weather_finetune_train', (submission / 'train.py').resolve())
        model = train_module.build_model(CASE['kwargs']).to(device)
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        ok = output.shape == targets.shape and bool(torch.isfinite(output).all())
        update_check(current_index, ok, f'output={tuple(output.shape)}, expected={tuple(targets.shape)}')
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['model_structure:required_module_contract']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        submission = CANDIDATE_DIR
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = CASE['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        train_module = _load('weather_finetune_train', (submission / 'train.py').resolve())
        model = train_module.build_model(CASE['kwargs']).to(device)
        children = dict(model.named_children())
        with torch.no_grad():
            backbone = model.backbone(inputs); low_rank = model.lora_A(inputs); adapted = model.lora_B(low_rank); output = model.head(torch.nn.functional.gelu(backbone + adapted))
        ok = list(children).index('backbone') < list(children).index('lora_A') < list(children).index('lora_B') < list(children).index('head') and all(name in dict(model.named_parameters()) for name in ('lora_A.weight', 'lora_B.weight')) and all(not p.requires_grad for p in model.backbone.parameters()) and backbone.shape == adapted.shape == (2, 16, 8, 8) and output.shape == (2, 2, 8, 8)
        detail = f'children={list(children)}, backbone={tuple(backbone.shape)}, low_rank={tuple(low_rank.shape)}, adapted={tuple(adapted.shape)}, output={tuple(output.shape)}'
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['train_script:adamw_backward_step']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        source = (CANDIDATE_DIR / 'train.py').resolve().read_text(encoding='utf-8')
        tree = ast.parse(source)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'train_step')
        calls = {n.func.id if isinstance(n.func, ast.Call) and isinstance(n.func, ast.Name) else (n.func.attr if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) else '') for n in ast.walk(node) if isinstance(n, ast.Call)}
        parameters = {arg.arg for arg in node.args.args}
        ok = {'model', 'inputs', 'targets'}.issubset(parameters) and {'backward', 'step'}.issubset(calls) and 'AdamW' in source
        update_check(current_index, ok, f'parameters={sorted(parameters)}, calls={sorted(calls)}')
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['traning:real_backward_optimizer_update']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        submission = CANDIDATE_DIR
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = CASE['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        train_module = _load('weather_finetune_train', (submission / 'train.py').resolve())
        model = train_module.build_model(CASE['kwargs']).to(device)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        kwargs = {'optimizer': optimizer}
        if CASE['kind'] == 'extreme':
            kwargs['event_mask'] = mask
        loss = train_module.train_step(model, inputs, targets, **kwargs)
        after = [p.detach() for p in model.parameters() if p.requires_grad]
        changed = any(not torch.equal(left, right) for left, right in zip(before, after))
        ok = isinstance(loss, (float, int)) and loss == loss and changed
        update_check(current_index, ok, f'loss={loss}, parameter_changed={changed}')
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks
