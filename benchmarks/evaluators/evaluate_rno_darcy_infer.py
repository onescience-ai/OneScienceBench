"""V2 adapter for the legacy RNO-Darcy inference evaluator."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import math
import os
import sys
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

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


TASK_NAME = 'rno_darcy_infer'
MODEL_CLASS = 'RadonNeuralOperator2D'
REQUIRED_FILES = ('model.py', 'inference.py', 'inference_summary.json')
PYTHON_FILES = ('model.py', 'inference.py')
CHECKPOINT_PATH = Path('/public/share/sugonhpcapp01/onestore/paper_repro/weight/rno/best_model.pth')
DATA_FILE = Path('/public/share/sugonhpcapp01/onestore/onedatasets/CFD_Benchmark/darcy/piececonst_r421_N1024_smooth2.mat')
EXPECTED_MODEL_CONFIG = {
    'in_channels': 3, 'out_channels': 1, 'channels': 128, 'layers': 8,
    'heads': 8, 'slices': 32, 'angles': 32, 'radon_indices': (1, 3, 5, 7),
    'angle_hidden': 16, 'mlp_ratio': 4.0,
}
ATOL = 1.0e-6
RTOL = 1.0e-5

def _read(path: Path) -> str:
    return path.read_text(encoding='utf-8')

def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise TypeError(f'{path.name} must contain a JSON object')
    return value

def _load_module(path: Path, role: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    name = f'_{TASK_NAME}_{role}_{abs(hash(path))}'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module

@lru_cache(maxsize=1)
def _checkpoint():
    import torch
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    try:
        value = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    except TypeError:
        value = torch.load(CHECKPOINT_PATH, map_location='cpu')
    if not isinstance(value, Mapping):
        raise TypeError('checkpoint must be a mapping')
    state = next((value.get(key) for key in ('model', 'model_state_dict', 'state_dict') if isinstance(value.get(key), Mapping)), None)
    if not state:
        raise KeyError('checkpoint has no model state dictionary')
    for key in ('input_normalizer', 'output_normalizer'):
        normalizer = value.get(key)
        if not isinstance(normalizer, Mapping) or not {'mean', 'std'}.issubset(normalizer):
            raise KeyError(f'checkpoint has no valid {key}')
    return value, state

def _infer_model_config(state: Mapping[str, Any]) -> dict[str, Any]:
    lifting = state['lifting.0.weight']
    channels, in_channels = map(int, lifting.shape[:2])
    layer_indices = sorted({int(key.split('.')[1]) for key in state if key.startswith('attention_layers.')})
    slice_shape = tuple(int(value) for value in state['attention_layers.0.slice_queries'].shape)
    heads, slices, head_dim = slice_shape
    if layer_indices != list(range(len(layer_indices))) or heads * head_dim != channels:
        raise ValueError('inconsistent attention tensor names or shapes')
    radon_indices = tuple(sorted({int(key.split('.')[1]) for key in state if key.startswith('radon_layers.')}))
    first_radon = str(radon_indices[0])
    hidden = int(state['attention_layers.0.mlp.0.weight'].shape[0])
    return {
        'in_channels': in_channels,
        'out_channels': int(state['projection.3.weight'].shape[0]),
        'channels': channels, 'layers': len(layer_indices), 'heads': heads,
        'slices': slices,
        'angles': int(state[f'radon_layers.{first_radon}.radon.angles'].numel()),
        'radon_indices': radon_indices,
        'angle_hidden': int(state[f'radon_layers.{first_radon}.weights.network.0.weight'].shape[0]),
        'mlp_ratio': hidden / channels,
        'dropout': 0.0,
    }

def _construct_model(module: Any, state: Mapping[str, Any]):
    saved = _checkpoint()[0].get('config', {}).get('model', {})
    factory = getattr(module, 'build_model_from_state_dict', None)
    if callable(factory):
        try:
            built = factory(state, saved)
        except TypeError:
            built = factory(state)
        model = built[0] if isinstance(built, tuple) else built
    else:
        model_class = getattr(module, MODEL_CLASS, None)
        if not inspect.isclass(model_class):
            raise AttributeError(f'provide {MODEL_CLASS} or build_model_from_state_dict() in model.py')
        config = _infer_model_config(state)
        signature = inspect.signature(model_class)
        accepts_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values())
        aliases = {
            **config, 'hidden_dim': config['channels'], 'num_layers': config['layers'],
            'num_heads': config['heads'], 'num_slices': config['slices'],
            'num_angles': config['angles'], 'radon_blocks': len(config['radon_indices']),
            'angle_hidden_channels': config['angle_hidden'],
        }
        kwargs = config if accepts_kwargs else {key: value for key, value in aliases.items() if key in signature.parameters}
        model = model_class(**kwargs)
    import torch
    if not isinstance(model, torch.nn.Module):
        raise TypeError('model builder did not return torch.nn.Module')
    model.load_state_dict(state, strict=True)
    return model.cpu()

def _runtime(root: Path, role: str):
    module = _load_module(root / 'model.py', role)
    checkpoint, state = _checkpoint()
    return module, _construct_model(module, state), checkpoint, state

def _canonical_output(output: Any, batch: int, height: int, width: int):
    import torch
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, Mapping):
        output = next((output[key] for key in ('pred', 'prediction', 'output', 'sol') if key in output), None)
    if not torch.is_tensor(output):
        raise TypeError('model output must be a tensor')
    if tuple(output.shape) == (batch, 1, height, width):
        return output
    if tuple(output.shape) == (batch, height, width, 1):
        return output.permute(0, 3, 1, 2)
    if tuple(output.shape) == (batch, height * width, 1):
        return output.transpose(1, 2).reshape(batch, 1, height, width)
    if tuple(output.shape) == (batch, height, width):
        return output[:, None]
    raise ValueError(f'unexpected output shape: {tuple(output.shape)}')

def _positions(batch: int, height: int, width: int, like: Any):
    import torch
    y = torch.linspace(0.0, 1.0, height, device=like.device, dtype=like.dtype)
    x = torch.linspace(0.0, 1.0, width, device=like.device, dtype=like.dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), dim=-1).reshape(1, -1, 2).repeat(batch, 1, 1)

def _forward(model: Any, coefficient: Any):
    batch, _, height, width = coefficient.shape
    positions = _positions(batch, height, width, coefficient)
    field = coefficient.flatten(2).transpose(1, 2)
    errors: list[str] = []
    for arguments in ((coefficient,), (positions, field), (field, positions)):
        try:
            return _canonical_output(model(*arguments), batch, height, width)
        except Exception as exc:
            errors.append(f'{len(arguments)} arg(s): {type(exc).__name__}: {exc}')
    raise RuntimeError('supported forward interfaces failed; ' + ' | '.join(errors))

def _normalizer_statistics(checkpoint: Mapping[str, Any], name: str):
    import torch
    state = checkpoint[name]
    mean, std = torch.as_tensor(state['mean']).float(), torch.as_tensor(state['std']).float()
    eps = float(state.get('eps', 1.0e-6))
    if tuple(mean.shape) != (1, 1, 85, 85) or mean.shape != std.shape:
        raise ValueError(f'invalid {name} statistic shapes: {mean.shape}/{std.shape}')
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
        raise ValueError(f'invalid {name} statistic values')
    return mean, std, eps

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['required:model.py', 'required:inference.py', 'required:inference_summary.json', 'syntax:model.py', 'syntax:inference.py', 'submission_independence']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        root = CANDIDATE_DIR
        path = root / 'model.py'; ok = path.is_file(); update_check(current_index, ok, str(path)); current_index += 1
        path = root / 'inference.py'; ok = path.is_file(); update_check(current_index, ok, str(path)); current_index += 1
        path = root / 'inference_summary.json'; ok = path.is_file(); update_check(current_index, ok, str(path)); current_index += 1
        path = root / 'model.py'
        try:
            ast.parse(_read(path), filename=str(path)); ok, msg = True, 'syntax is valid'
        except Exception as exc:
            ok, msg = False, f'{type(exc).__name__}: {exc}'
        update_check(current_index, ok, msg); current_index += 1
        path = root / 'inference.py'
        try:
            ast.parse(_read(path), filename=str(path)); ok, msg = True, 'syntax is valid'
        except Exception as exc:
            ok, msg = False, f'{type(exc).__name__}: {exc}'
        update_check(current_index, ok, msg); current_index += 1
        source = chr(10).join(_read(root / name) for name in PYTHON_FILES if (root / name).is_file())
        ok = 'codegen.standard' not in source and 'codegen.evaluators' not in source
        update_check(current_index, ok, 'must not import evaluator/reference code'); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_data_transformation(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['smooth2 MAT path', 'coeff/sol fields', 'first 200 samples', '421 to 85 stride-5', 'batch size 8', 'shuffle false', 'first test batch', 'MAT keys']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        root = CANDIDATE_DIR
        source = ''.join(_read(root / 'inference.py').lower().split())
        summary = _load_json(root / 'inference_summary.json')
        conditions = {
            'smooth2 MAT path': Path(str(summary.get('data_file', ''))).resolve() == DATA_FILE or DATA_FILE.name.lower() in source,
            'coeff/sol fields': 'coeff' in source and 'sol' in source,
            'first 200 samples': summary.get('selected_samples') == 200 and any(token in source for token in ('[:samples', '[:args.samples', '[0:samples')),
            '421 to 85 stride-5': summary.get('subsample_stride') == 5 and summary.get('downsampled_shape') == [200, 1, 85, 85],
            'batch size 8': summary.get('batch_size') == 8,
            'shuffle false': summary.get('shuffle') is False and 'shuffle=false' in source,
            'first test batch': summary.get('evaluated_batch_index') == 0 and summary.get('evaluated_samples') == 8,
            'MAT keys': summary.get('mat_keys') == {'input': 'coeff', 'target': 'sol'},
        }
        ok = conditions['smooth2 MAT path']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['coeff/sol fields']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['first 200 samples']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['421 to 85 stride-5']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['batch size 8']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['shuffle false']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['first test batch']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['MAT keys']; update_check(current_index, ok, str(ok)); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['main guard', 'checkpoint path', 'state names/shapes reconstruction', 'strict checkpoint load', 'checkpoint normalizers', 'evaluation mode', 'physical-space decode', 'MSE and per-sample Relative L2', 'summary output']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        root = CANDIDATE_DIR
        text = _read(root / 'inference.py')
        combined = text + chr(10) + _read(root / 'model.py')
        source = ''.join(combined.lower().split())
        summary = _load_json(root / 'inference_summary.json')
        conditions = {
            'main guard': 'if__name__=="__main__"' in source.replace("'", '"'),
            'checkpoint path': Path(str(summary.get('checkpoint', ''))).resolve() == CHECKPOINT_PATH or str(CHECKPOINT_PATH).lower() in source,
            'state names/shapes reconstruction': 'state' in source and '.shape' in source,
            'strict checkpoint load': 'strict=true' in source,
            'checkpoint normalizers': 'input_normalizer' in source and 'output_normalizer' in source,
            'evaluation mode': '.eval(' in source and ('no_grad' in source or 'inference_mode' in source),
            'physical-space decode': 'decode(' in source or ('output_normalizer' in source and 'std' in source and 'mean' in source),
            'MSE and per-sample Relative L2': all(token in source for token in ('mse', 'relative_l2', 'flatten(1)')),
            'summary output': 'inference_summary.json' in source and 'json' in source,
        }
        ok = conditions['main guard']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['checkpoint path']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['state names/shapes reconstruction']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['strict checkpoint load']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['checkpoint normalizers']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['evaluation mode']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['physical-space decode']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['MSE and per-sample Relative L2']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['summary output']; update_check(current_index, ok, str(ok)); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_eval_script_contract(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['checkpoint CLI', 'data CLI', 'output CLI', 'device selection']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        text = _read(CANDIDATE_DIR / 'inference.py').lower()
        conditions = {
            'checkpoint CLI': '--checkpoint' in text,
            'data CLI': '--data' in text,
            'output CLI': '--output' in text,
            'device selection': '--device' in text or 'torch.device' in text,
        }
        ok = conditions['checkpoint CLI']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['data CLI']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['output CLI']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['device selection']; update_check(current_index, ok, str(ok)); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['checkpoint-derived config', 'exact state keys and shapes', '178 state tensors']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        _, model, _, state = _runtime(CANDIDATE_DIR, 'structure')
        derived = _infer_model_config(state)
        actual_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        expected_shapes = {name: tuple(value.shape) for name, value in state.items()}
        conditions = {
            'checkpoint-derived config': all(derived[key] == value for key, value in EXPECTED_MODEL_CONFIG.items()),
            'exact state keys and shapes': actual_shapes == expected_shapes,
            '178 state tensors': len(actual_shapes) == 178,
        }
        ok = conditions['checkpoint-derived config']; update_check(current_index, ok, str({'condition': ok, 'derived': derived})); current_index += 1
        ok = conditions['exact state keys and shapes']; update_check(current_index, ok, str({'condition': ok, 'derived': derived})); current_index += 1
        ok = conditions['178 state tensors']; update_check(current_index, ok, str({'condition': ok, 'derived': derived})); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['output shape (B,1,H,W)', 'finite output', 'input unchanged']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        torch.manual_seed(20260826)
        _, model, _, _ = _runtime(CANDIDATE_DIR, 'forward')
        model.eval()
        coefficient = torch.randn(2, 1, 9, 11)
        original = coefficient.clone()
        with torch.inference_mode():
            output = _forward(model, coefficient)
        conditions = {
            'output shape (B,1,H,W)': tuple(output.shape) == (2, 1, 9, 11),
            'finite output': bool(torch.isfinite(output).all()),
            'input unchanged': bool(torch.equal(coefficient, original)),
        }
        ok = conditions['output shape (B,1,H,W)']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['finite output']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['input unchanged']; update_check(current_index, ok, str(ok)); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['reference_forward']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        torch.manual_seed(20260826)
        _, candidate, _, _ = _runtime(CANDIDATE_DIR, 'candidate')
        _, reference, _, _ = _runtime(REFERENCE_DIR, 'reference')
        candidate.eval()
        reference.eval()
        coefficient = torch.randn(1, 1, 9, 11)
        with torch.inference_mode():
            actual = _forward(candidate, coefficient)
            expected = _forward(reference, coefficient)
        difference = (actual.double() - expected.double()).abs()
        equivalent = bool(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL))
        detail = {'max_abs_error': float(difference.max()), 'mse': float(difference.square().mean()), 'atol': ATOL, 'rtol': RTOL}
        update_check(current_index, equivalent, str(detail)); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_metrics(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['finite prediction', 'finite MSE', 'finite Relative L2']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        torch.manual_seed(20260826)
        _, model, checkpoint, _ = _runtime(CANDIDATE_DIR, 'pipeline')
        input_mean, input_std, input_eps = _normalizer_statistics(checkpoint, 'input_normalizer')
        output_mean, output_std, output_eps = _normalizer_statistics(checkpoint, 'output_normalizer')
        height, width = 9, 11
        input_mean, input_std = input_mean[..., :height, :width], input_std[..., :height, :width]
        output_mean, output_std = output_mean[..., :height, :width], output_std[..., :height, :width]
        physical_input = input_mean + 0.25 * input_std
        normalized_input = (physical_input - input_mean) / (input_std + input_eps)
        model.eval()
        with torch.inference_mode():
            normalized_prediction = _forward(model, normalized_input)
            prediction = normalized_prediction * (output_std + output_eps) + output_mean
        difference = prediction - output_mean
        mse = difference.square().flatten(1).mean(dim=1)
        relative_l2 = torch.linalg.vector_norm(difference.flatten(1), dim=1) / torch.linalg.vector_norm(output_mean.flatten(1), dim=1).clamp_min(1.0e-12)
        detail = {'mse': float(mse.mean()), 'relative_l2': float(relative_l2.mean())}
        ok = bool(torch.isfinite(prediction).all()); update_check(current_index, ok, str(detail)); current_index += 1
        ok = bool(torch.isfinite(mse).all()); update_check(current_index, ok, str(detail)); current_index += 1
        ok = bool(torch.isfinite(relative_l2).all()); update_check(current_index, ok, str(detail)); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_output_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['strict load recorded', 'checkpoint tensor count', 'physical metric space', 'prediction shape', 'finite per-sample metrics', 'batch means are consistent', 'matches standard inference']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        root = CANDIDATE_DIR
        standard_root = REFERENCE_DIR
        summary = _load_json(root / 'inference_summary.json')
        expected = _load_json(standard_root / 'inference_summary.json')
        mse = [float(value) for value in summary.get('mse_per_sample', [])]
        relative = [float(value) for value in summary.get('relative_l2_per_sample', [])]
        scalar_values = [*mse, *relative, summary.get('mse_batch_mean'), summary.get('relative_l2_batch_mean')]
        finite = all(isinstance(value, (int, float)) and math.isfinite(value) for value in scalar_values)
        means_match = len(mse) == len(relative) == 8 and math.isclose(sum(mse) / 8, float(summary['mse_batch_mean']), rel_tol=1e-6, abs_tol=1e-12) and math.isclose(sum(relative) / 8, float(summary['relative_l2_batch_mean']), rel_tol=1e-6, abs_tol=1e-12)
        baseline_match = math.isclose(float(summary.get('mse_batch_mean', math.inf)), float(expected['mse_batch_mean']), rel_tol=2e-4, abs_tol=1e-12) and math.isclose(float(summary.get('relative_l2_batch_mean', math.inf)), float(expected['relative_l2_batch_mean']), rel_tol=2e-4, abs_tol=1e-7)
        conditions = {
            'strict load recorded': summary.get('strict_load') is True,
            'checkpoint tensor count': summary.get('checkpoint_state_tensor_count') == 178,
            'physical metric space': 'physical' in str(summary.get('metrics_space', '')).lower(),
            'prediction shape': summary.get('prediction_shape') == [8, 1, 85, 85],
            'finite per-sample metrics': finite,
            'batch means are consistent': means_match,
            'matches standard inference': baseline_match,
        }
        ok = conditions['strict load recorded']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['checkpoint tensor count']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['physical metric space']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['prediction shape']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['finite per-sample metrics']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['batch means are consistent']; update_check(current_index, ok, str(ok)); current_index += 1
        ok = conditions['matches standard inference']; update_check(current_index, ok, str(ok)); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks
