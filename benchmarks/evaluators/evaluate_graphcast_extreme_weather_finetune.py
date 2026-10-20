'''V2 adapter for graphcast extreme weather fine-tuning.'''
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

CASE = {
    'model_file': 'graphcast_extreme_weather.py',
    'class_name': 'GraphCastExtremeWeather',
    'env': 'GRAPHCAST_EXTREME_SUBMISSION_DIR',
    'shape': ((2, 6, 8, 8), (2, 1, 8, 8)),
    'kwargs': {},
    'kind': 'extreme',
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


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:graphcast_extreme_weather.py', 'file:train.py', 'file:evaluate.py']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = CASE
        submission = CANDIDATE_DIR
        for name in (case['model_file'], 'train.py', 'evaluate.py'):
            path = submission / name
            ok = path.is_file()
            detail = str(path)
            if ok:
                try:
                    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                    detail = f'syntax=ok, bytes={path.stat().st_size}'
                except Exception as exc:
                    ok, detail = False, f'{type(exc).__name__}: {exc}'
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['data:synthetic_weather_contract']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        case = CASE
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = case['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        ok = bool(torch.isfinite(inputs).all() and torch.isfinite(targets).all())
        detail = f'input={input_shape}, target={target_shape}, mask={tuple(mask.shape)}'
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
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
        case = CASE
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_module = train_module = eval_module = None
        model_module = _load('weather_finetune_model', CANDIDATE_DIR / case['model_file'])
        train_module = _load('weather_finetune_train', CANDIDATE_DIR / 'train.py')
        eval_module = _load('weather_finetune_eval', CANDIDATE_DIR / 'evaluate.py')
        update_check(current_index, True, 'modules imported')
        current_index += 1
        model = train_module.build_model(case['kwargs']).to(device)
        update_check(current_index, True, f'model={type(model).__name__}')
        current_index += 1
    except Exception as e:
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
        train_module = _load('weather_finetune_train', CANDIDATE_DIR / 'train.py')
        source = Path(train_module.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'train_step')
        calls = {n.func.id if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) else (n.func.attr if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) else '') for n in ast.walk(node) if isinstance(n, ast.Call)}
        parameters = {arg.arg for arg in node.args.args}
        ok = {'model', 'inputs', 'targets'}.issubset(parameters) and {'backward', 'step'}.issubset(calls) and 'AdamW' in source
        update_check(current_index, ok, f'parameters={sorted(parameters)}, calls={sorted(calls)}')
        current_index += 1
    except Exception as e:
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
        eval_module = _load('weather_finetune_eval', CANDIDATE_DIR / 'evaluate.py')
        source = Path(eval_module.__file__).read_text(encoding='utf-8')
        ok = 'def evaluate' in source and 'rmse' in source and 'mae' in source
        update_check(current_index, ok, 'evaluate/rmse/mae')
        current_index += 1
    except Exception as e:
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
        case = CASE
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = case['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_module = train_module = eval_module = None
        model_module = _load('weather_finetune_model', CANDIDATE_DIR / case['model_file'])
        train_module = _load('weather_finetune_train', CANDIDATE_DIR / 'train.py')
        eval_module = _load('weather_finetune_eval', CANDIDATE_DIR / 'evaluate.py')
        model = train_module.build_model(case['kwargs']).to(device)
        children = dict(model.named_children())
        kind = case['kind']
        if kind == 'lora':
            with torch.no_grad():
                backbone = model.backbone(inputs); low_rank = model.lora_A(inputs); adapted = model.lora_B(low_rank); output = model.head(torch.nn.functional.gelu(backbone + adapted))
            ok = list(children).index('backbone') < list(children).index('lora_A') < list(children).index('lora_B') < list(children).index('head') and all(name in dict(model.named_parameters()) for name in ('lora_A.weight', 'lora_B.weight')) and all(not p.requires_grad for p in model.backbone.parameters()) and backbone.shape == adapted.shape == (2, 16, 8, 8) and output.shape == (2, 2, 8, 8)
            detail = f'children={list(children)}, backbone={tuple(backbone.shape)}, low_rank={tuple(low_rank.shape)}, adapted={tuple(adapted.shape)}, output={tuple(output.shape)}'
        elif kind == 'highres':
            with torch.no_grad():
                encoded = model.grid_encoder(inputs); processed = encoded
                for block in model.mesh_processor: processed = processed + torch.nn.functional.gelu(block(processed))
                upsampled = torch.nn.functional.interpolate(processed, scale_factor=2, mode='bilinear', align_corners=False); decoded = model.grid_decoder(upsampled)
            ok = list(children) == ['grid_encoder', 'mesh_processor', 'grid_decoder'] and isinstance(model.mesh_processor, torch.nn.ModuleList) and len(model.mesh_processor) == 2 and model.upscale_factor == 2 and encoded.shape == processed.shape == (2, 16, 4, 4) and decoded.shape == (2, 2, 8, 8)
            detail = f'children={list(children)}, encoded={tuple(encoded.shape)}, processed={tuple(processed.shape)}, decoded={tuple(decoded.shape)}'
        elif kind == 'extreme':
            with torch.no_grad():
                encoded = model.grid_encoder(inputs); processed = encoded
                for layer in model.processor: processed = processed + torch.tanh(layer(processed))
                output = model.extreme_head(processed)
            ok = list(children) == ['grid_encoder', 'processor', 'extreme_head'] and len(model.processor) == 3 and encoded.shape == processed.shape == (2, 16, 8, 8) and output.shape == (2, 1, 8, 8)
            detail = f'children={list(children)}, encoded={tuple(encoded.shape)}, processed={tuple(processed.shape)}, output={tuple(output.shape)}'
        elif kind == 'radar':
            with torch.no_grad():
                evolution = model.evolution(inputs); generated = model.generator(torch.cat([inputs, evolution], dim=1))
            ok = list(children) == ['evolution', 'generator'] and model.pred_length == 3 and evolution.shape == generated.shape == (2, 3, 8, 8)
            detail = f'children={list(children)}, evolution={tuple(evolution.shape)}, generated={tuple(generated.shape)}'
        elif kind == 'domain':
            with torch.no_grad():
                backbone = model.backbone(inputs); low_rank = model.adapter_down(inputs); adapted = model.adapter_up(low_rank); output = model.head(torch.relu(backbone + adapted))
            ok = list(children) == ['backbone', 'adapter_down', 'adapter_up', 'head'] and all(not p.requires_grad for p in model.backbone.parameters()) and backbone.shape == adapted.shape == (2, 16, 8, 8) and low_rank.shape == (2, 4, 8, 8) and output.shape == (2, 3, 8, 8)
            detail = f'children={list(children)}, backbone={tuple(backbone.shape)}, low_rank={tuple(low_rank.shape)}, adapted={tuple(adapted.shape)}, output={tuple(output.shape)}'
        else:
            with torch.no_grad():
                encoded = model.encoder(inputs); head_outputs = [head(encoded) for head in model.horizon_heads]
            ok = list(children) == ['encoder', 'horizon_heads'] and isinstance(model.horizon_heads, torch.nn.ModuleList) and len(model.horizon_heads) == 3 and tuple(model.horizons) == (1, 3, 6) and encoded.shape == (2, 16, 8, 8) and all(item.shape == (2, 1, 8, 8) for item in head_outputs)
            detail = f'children={list(children)}, horizons={model.horizons}, encoded={tuple(encoded.shape)}, heads={[tuple(item.shape) for item in head_outputs]}'
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
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
        case = CASE
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = case['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_module = train_module = eval_module = None
        model_module = _load('weather_finetune_model', CANDIDATE_DIR / case['model_file'])
        train_module = _load('weather_finetune_train', CANDIDATE_DIR / 'train.py')
        eval_module = _load('weather_finetune_eval', CANDIDATE_DIR / 'evaluate.py')
        model = train_module.build_model(case['kwargs']).to(device)
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        ok = output.shape == targets.shape and bool(torch.isfinite(output).all())
        update_check(current_index, ok, f'output={tuple(output.shape)}, expected={tuple(targets.shape)}')
        current_index += 1
    except Exception as e:
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
        case = CASE
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape, target_shape = case['shape']
        inputs = torch.randn(*input_shape)
        targets = torch.randn(*target_shape)
        mask = torch.zeros_like(targets); mask[..., :2, :2] = 1
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        model_module = train_module = eval_module = None
        model_module = _load('weather_finetune_model', CANDIDATE_DIR / case['model_file'])
        train_module = _load('weather_finetune_train', CANDIDATE_DIR / 'train.py')
        eval_module = _load('weather_finetune_eval', CANDIDATE_DIR / 'evaluate.py')
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
        update_check(current_index, ok, f'loss={loss}, parameter_changed={changed}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
