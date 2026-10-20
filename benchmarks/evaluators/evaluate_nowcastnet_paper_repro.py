"""V2 adapter for the legacy NowCastNet evaluator."""

import ast
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
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

TASK_NAME = 'nowcastnet_paper_repro'
MODEL_FILE = 'nowcastnet.py'
MODEL_CLASS = 'Net'
ENV_NAME = 'NOWCASTNET_SUBMISSION_DIR'
REQUIRED_FILES = (MODEL_FILE, 'train.py', 'evaluate.py')
CHECK_TYPES = {
    'file', 'data', 'train_script', 'eval_script', 'model_structure',
    'forward_executable', 'forward_equivalence', 'traning', 'artifacts',
}
SEED = 20260819


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import {path}')
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
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'data:batch_shape', 'data:target_shape', 'data:noise_shape',
        'data:batch_finite', 'data:target_finite', 'data:noise_finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(SEED)
        fixture = {
            'batch': torch.randn(2, 6, 8, 8, 1, device=device),
            'target': torch.randn(2, 3, 8, 8, device=device),
            'noise': torch.randn(2, 1, 8, 8, device=device),
        }
        inputs, targets, noise = fixture['batch'], fixture['target'], fixture['noise']
        ok = inputs.shape == (2, 6, 8, 8, 1)
        update_check(current_index, ok, f'input={tuple(inputs.shape)}')
        current_index += 1
        ok = targets.shape == (2, 3, 8, 8)
        update_check(current_index, ok, f'target={tuple(targets.shape)}')
        current_index += 1
        ok = noise.shape == (2, 1, 8, 8)
        update_check(current_index, ok, f'noise={tuple(noise.shape)}')
        current_index += 1
        ok = bool(torch.isfinite(inputs).all())
        update_check(current_index, ok, f'input_finite={ok}')
        current_index += 1
        ok = bool(torch.isfinite(targets).all())
        update_check(current_index, ok, f'target_finite={ok}')
        current_index += 1
        ok = bool(torch.isfinite(noise).all())
        update_check(current_index, ok, f'noise_finite={ok}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'train_script:train_step_exists', 'train_script:parameter_model', 'train_script:parameter_inputs', 'train_script:parameter_targets',
        'train_script:call_AdamW', 'train_script:call_backward', 'train_script:call_step',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = CANDIDATE_DIR / 'train.py'
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        node = _function(tree, 'train_step')
        if node is None:
            raise AttributeError('train_step is missing')
        parameters = [arg.arg for arg in node.args.args]
        calls = _calls(node)
        update_check(current_index, True, f'parameters={parameters}, calls={sorted(calls)}')
        current_index += 1
        ok = 'model' in parameters
        update_check(current_index, ok, f'parameters={parameters}')
        current_index += 1
        ok = 'inputs' in parameters
        update_check(current_index, ok, f'parameters={parameters}')
        current_index += 1
        ok = 'targets' in parameters
        update_check(current_index, ok, f'parameters={parameters}')
        current_index += 1
        ok = 'AdamW' in calls
        update_check(current_index, ok, f'calls={sorted(calls)}')
        current_index += 1
        ok = 'backward' in calls
        update_check(current_index, ok, f'calls={sorted(calls)}')
        current_index += 1
        ok = 'step' in calls
        update_check(current_index, ok, f'calls={sorted(calls)}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_metrics(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'eval_script:evaluate_exists', 'eval_script:parameter_model', 'eval_script:parameter_inputs', 'eval_script:parameter_targets',
        'eval_script:metric_mae', 'eval_script:metric_rmse', 'eval_script:call_abs', 'eval_script:call_mean', 'eval_script:call_sqrt',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = CANDIDATE_DIR / 'evaluate.py'
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        node = _function(tree, 'evaluate')
        if node is None:
            raise AttributeError('evaluate is missing')
        parameters = [arg.arg for arg in node.args.args]
        calls = _calls(node)
        strings = {child.value.lower() for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)}
        update_check(current_index, True, f'parameters={parameters}, metrics={sorted(strings & {"mae", "rmse"})}')
        current_index += 1
        ok = 'model' in parameters
        update_check(current_index, ok, f'parameters={parameters}')
        current_index += 1
        ok = 'inputs' in parameters
        update_check(current_index, ok, f'parameters={parameters}')
        current_index += 1
        ok = 'targets' in parameters
        update_check(current_index, ok, f'parameters={parameters}')
        current_index += 1
        ok = 'mae' in strings
        update_check(current_index, ok, f'metrics={sorted(strings & {"mae", "rmse"})}')
        current_index += 1
        ok = 'rmse' in strings
        update_check(current_index, ok, f'metrics={sorted(strings & {"mae", "rmse"})}')
        current_index += 1
        ok = 'abs' in calls
        update_check(current_index, ok, f'calls={sorted(calls)}')
        current_index += 1
        ok = 'mean' in calls
        update_check(current_index, ok, f'calls={sorted(calls)}')
        current_index += 1
        ok = 'sqrt' in calls
        update_check(current_index, ok, f'calls={sorted(calls)}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'model_structure:top_level_order', 'model_structure:evolution_order',
        'model_structure:intensity_conv2d', 'model_structure:intensity_out_channels',
        'model_structure:motion_conv2d', 'model_structure:motion_out_channels',
        'model_structure:noise_projector_conv2d', 'model_structure:noise_projector_in_channels',
        'model_structure:pred_length', 'model_structure:warp_callable',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        module = _load_module('nowcastnet_submission_model', CANDIDATE_DIR / MODEL_FILE)
        model_class = getattr(module, MODEL_CLASS)
        if not inspect.isclass(model_class):
            raise TypeError(f'{MODEL_CLASS} is not a class')
        model = model_class(input_length=3, total_length=6, img_height=8, img_width=8, ngf=8).to(device)
        children = list(dict(model.named_children()))
        evolution = getattr(model, 'evolution')
        evolution_children = list(dict(evolution.named_children()))
        detail = f'children={children}, evolution_children={evolution_children}, pred_length={model.pred_length}'
        ok = _subsequence(children, ('evolution', 'noise_projector', 'generator'))
        update_check(current_index, ok, detail)
        current_index += 1
        ok = _subsequence(evolution_children, ('body', 'intensity', 'motion'))
        update_check(current_index, ok, detail)
        current_index += 1
        ok = isinstance(evolution.intensity, torch.nn.Conv2d)
        update_check(current_index, ok, detail)
        current_index += 1
        ok = evolution.intensity.out_channels == 3
        update_check(current_index, ok, detail)
        current_index += 1
        ok = isinstance(evolution.motion, torch.nn.Conv2d)
        update_check(current_index, ok, detail)
        current_index += 1
        ok = evolution.motion.out_channels == 6
        update_check(current_index, ok, detail)
        current_index += 1
        ok = isinstance(model.noise_projector, torch.nn.Conv2d)
        update_check(current_index, ok, detail)
        current_index += 1
        ok = model.noise_projector.in_channels == 1
        update_check(current_index, ok, detail)
        current_index += 1
        ok = model.pred_length == 3
        update_check(current_index, ok, detail)
        current_index += 1
        ok = callable(getattr(model, '_warp', None))
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'forward_executable:output_shape', 'forward_executable:default_output_finite',
        'forward_executable:zero_noise_matches_default', 'forward_executable:channel_first_matches_default',
        'forward_executable:noise_changes_output', 'forward_executable:rejects_bad_length',
        'forward_executable:rejects_bad_grid', 'forward_executable:rejects_bad_channels',
        'forward_executable:rejects_bad_noise',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(SEED)
        fixture = {
            'batch': torch.randn(2, 6, 8, 8, 1, device=device),
            'target': torch.randn(2, 3, 8, 8, device=device),
            'noise': torch.randn(2, 1, 8, 8, device=device),
        }
        module = _load_module('nowcastnet_submission_model', CANDIDATE_DIR / MODEL_FILE)
        model_class = getattr(module, MODEL_CLASS)
        if not inspect.isclass(model_class):
            raise TypeError(f'{MODEL_CLASS} is not a class')
        model = model_class(input_length=3, total_length=6, img_height=8, img_width=8, ngf=8).to(device)
        inputs, noise = fixture['batch'], fixture['noise']
        model.eval()
        with torch.no_grad():
            default_output = model(inputs)
            zero_output = model(inputs, torch.zeros_like(noise))
            noisy_output = model(inputs, noise)
            channel_first = model(inputs[..., 0])
        rejects = {
            'length': _rejects(lambda: model(inputs[:, :5])),
            'grid': _rejects(lambda: model(inputs[:, :, :6])),
            'channels': _rejects(lambda: model(inputs.repeat(1, 1, 1, 1, 2))),
            'noise': _rejects(lambda: model(inputs, noise[:, :, :6])),
        }
        zero_matches_default = torch.allclose(default_output, zero_output, atol=1e-6, rtol=1e-5)
        channel_first_matches = torch.allclose(default_output, channel_first, atol=1e-6, rtol=1e-5)
        noise_delta = float((noisy_output - zero_output).abs().max())
        detail = f'shape={tuple(default_output.shape)}, zero_default={zero_matches_default}, channel_first={channel_first_matches}, noise_delta={noise_delta}, rejects={rejects}'
        ok = default_output.shape == fixture['target'].shape
        update_check(current_index, ok, detail)
        current_index += 1
        ok = bool(torch.isfinite(default_output).all())
        update_check(current_index, ok, detail)
        current_index += 1
        ok = zero_matches_default
        update_check(current_index, ok, detail)
        current_index += 1
        ok = channel_first_matches
        update_check(current_index, ok, detail)
        current_index += 1
        ok = noise_delta > 1e-7
        update_check(current_index, ok, detail)
        current_index += 1
        ok = rejects['length']
        update_check(current_index, ok, detail)
        current_index += 1
        ok = rejects['grid']
        update_check(current_index, ok, detail)
        current_index += 1
        ok = rejects['channels']
        update_check(current_index, ok, detail)
        current_index += 1
        ok = rejects['noise']
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['forward_equivalence:same_shape', 'forward_equivalence:strict_state_reload_allclose']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(SEED)
        fixture = {
            'batch': torch.randn(2, 6, 8, 8, 1, device=device),
            'target': torch.randn(2, 3, 8, 8, device=device),
            'noise': torch.randn(2, 1, 8, 8, device=device),
        }
        module = _load_module('nowcastnet_submission_model', CANDIDATE_DIR / MODEL_FILE)
        model_class = getattr(module, MODEL_CLASS)
        if not inspect.isclass(model_class):
            raise TypeError(f'{MODEL_CLASS} is not a class')
        model = model_class(input_length=3, total_length=6, img_height=8, img_width=8, ngf=8).to(device)
        model.eval()
        clone = copy.deepcopy(model).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            left = model(fixture['batch'], fixture['noise'])
            right = clone(fixture['batch'], fixture['noise'])
        same_shape = left.shape == right.shape
        update_check(current_index, same_shape, f'left_shape={tuple(left.shape)}, right_shape={tuple(right.shape)}')
        current_index += 1
        ok = same_shape and torch.allclose(left, right, atol=1e-6, rtol=1e-5)
        max_abs_diff = float((left - right).abs().max())
        update_check(current_index, ok, f'max_abs_diff={max_abs_diff}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'traning:loss_finite', 'traning:gradients_present',
        'traning:gradients_finite', 'traning:parameter_changed',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(SEED)
        fixture = {
            'batch': torch.randn(2, 6, 8, 8, 1, device=device),
            'target': torch.randn(2, 3, 8, 8, device=device),
            'noise': torch.randn(2, 1, 8, 8, device=device),
        }
        module = _load_module('nowcastnet_submission_model', CANDIDATE_DIR / MODEL_FILE)
        model_class = getattr(module, MODEL_CLASS)
        if not inspect.isclass(model_class):
            raise TypeError(f'{MODEL_CLASS} is not a class')
        model = model_class(input_length=3, total_length=6, img_height=8, img_width=8, ngf=8).to(device)
        model.train()
        parameters = [item for item in model.parameters() if item.requires_grad]
        before = [item.detach().clone() for item in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        output = model(fixture['batch'], fixture['noise'])
        loss = (output - fixture['target']).square().mean()
        loss.backward()
        gradients = [item.grad for item in parameters if item.grad is not None]
        gradients_ok = bool(gradients) and all(bool(torch.isfinite(item).all()) for item in gradients)
        optimizer.step()
        changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, parameters))
        detail = f'loss={float(loss.detach()):.6g}, gradients={len(gradients)}, changed={changed}'
        ok = bool(torch.isfinite(loss))
        update_check(current_index, ok, detail)
        current_index += 1
        ok = bool(gradients)
        update_check(current_index, ok, detail)
        current_index += 1
        ok = gradients_ok
        update_check(current_index, ok, detail)
        current_index += 1
        ok = changed
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['artifacts:train_step_callable', 'artifacts:evaluate_callable']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        train = _load_module('nowcastnet_submission_train', CANDIDATE_DIR / 'train.py')
        evaluation = _load_module('nowcastnet_submission_evaluate', CANDIDATE_DIR / 'evaluate.py')
        ok = callable(getattr(train, 'train_step', None))
        update_check(current_index, ok, 'train_step imports successfully')
        current_index += 1
        ok = callable(getattr(evaluation, 'evaluate', None))
        update_check(current_index, ok, 'evaluate imports successfully')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
