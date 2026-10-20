"""V2 adapter for the Transolver AirfRANS legacy evaluator."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

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
        'required:transolver.py', 'parse:transolver.py',
        'required:dataset.py', 'parse:dataset.py',
        'required:train.py', 'parse:train.py',
        'required:config.yaml', 'parse:config.yaml',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        generated_dir = CANDIDATE_DIR
        reference_dir = REFERENCE_DIR
        required_files = ('transolver.py', 'dataset.py', 'train.py', 'config.yaml')
        for name in required_files:
            path = generated_dir / name
            exists = path.is_file()
            update_check(current_index, exists, str(path))
            current_index += 1
            if not exists:
                current_index += 1
                continue
            try:
                if path.suffix == '.py':
                    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                else:
                    import yaml
                    value = yaml.safe_load(path.read_text(encoding='utf-8'))
                    if not isinstance(value, Mapping):
                        raise TypeError(f'configuration must be a mapping: {path}')
                update_check(current_index, True, 'valid')
            except Exception as exc:
                update_check(current_index, False, f'{type(exc).__name__}: {exc}')
            current_index += 1
        del reference_dir
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_datasset_pipeline(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'data:configured_path', 'data:manifest_split',
        'data:last_ten_percent_validation', 'data:vtk_files',
        'data:required_fields', 'data:seven_inputs', 'data:four_targets',
        'data:surface_mask', 'data:training_statistics',
        'data:shared_normalization', 'data:random_32000_sampling',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        directory = CANDIDATE_DIR
        reference_dir = REFERENCE_DIR
        source = (directory / 'dataset.py').read_text(encoding='utf-8')
        lower = source.lower()
        import yaml
        yaml_config = yaml.safe_load((directory / 'config.yaml').read_text(encoding='utf-8'))
        if not isinstance(yaml_config, Mapping):
            raise TypeError(f'configuration must be a mapping: {directory / "config.yaml"}')
        data_config = next((yaml_config[name] for name in ('data', 'dataset') if isinstance(yaml_config.get(name), Mapping)), {})
        training_config = next((yaml_config[name] for name in ('training', 'train') if isinstance(yaml_config.get(name), Mapping)), {})
        sample_points = next((training_config[name] for name in ('sample_points', 'subsampling', 'n_points') if name in training_config), None)
        root_value = next((data_config[name] for name in ('root', 'data_root', 'data_path', 'path') if name in data_config), None)
        update_check(current_index, bool(root_value) and 'transolver-airfoil-design/dataset' in str(root_value).lower(), f'configured_root={root_value!r}')
        current_index += 1
        update_check(current_index, 'manifest.json' in lower and 'train' in lower and 'val' in lower, 'manifest-based train/validation split')
        current_index += 1
        update_check(current_index, ('0.1' in lower or '10' in lower or 'val_ratio' in lower) and ('[-' in source or 'validation' in lower or 'n_val' in lower), 'the final 10% of the selected training list is validation')
        current_index += 1
        update_check(current_index, 'internal.vtu' in lower and 'aerofoil.vtp' in lower, 'internal VTU and aerofoil VTP are read')
        current_index += 1
        update_check(current_index, all(token in lower for token in ('implicit_distance', 'normal', 'nut')) and ('point_data' in lower or 'point_arrays' in lower), 'U, p, nut, implicit_distance, and Normals are consumed')
        current_index += 1
        update_check(current_index, ('concatenate' in lower or 'torch.cat' in lower) and ('freestream' in lower or 'u_inf' in lower) and 'sdf' in lower and 'normal' in lower, '[x,y,u_inf_x,u_inf_y,SDF,normal_x,normal_y]')
        current_index += 1
        update_check(current_index, ('target' in lower or '"y"' in lower or "['y']" in lower) and 'nut' in lower and ('pressure' in lower or '"p"' in lower or "['p']" in lower), '[ux,uy,p,nut]')
        current_index += 1
        update_check(current_index, 'surf' in lower and ('bool' in lower or 'mask' in lower), 'surface mask remains aligned with sampled nodes')
        current_index += 1
        update_check(current_index, ('mean' in lower and 'std' in lower) and ('fit_stats' in lower or 'compute_normal' in lower or 'training' in lower), 'normalization statistics are derived from training data')
        current_index += 1
        update_check(current_index, 'stats' in lower or 'normalization' in lower or 'coef_norm' in lower, 'validation consumes reusable training statistics')
        current_index += 1
        try:
            sample_points_match = int(sample_points) == 32_000
        except (TypeError, ValueError):
            sample_points_match = sample_points == 32_000
        update_check(current_index, sample_points_match and any(token in lower for token in ('random.sample', 'randperm', 'choice', 'randint')), f'configured_sample_points={sample_points!r}')
        current_index += 1
        del reference_dir
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'train:config_interface', 'train:model_defaults', 'train:training_defaults',
        'train:model_usage', 'train:volume_plus_surface_mse', 'train:adam_onecycle',
        'train:optimization_loop', 'train:scheduler_per_step',
        'train:resampled_validation', 'train:history_and_checkpoint',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        directory = CANDIDATE_DIR
        reference_dir = REFERENCE_DIR
        source = (directory / 'train.py').read_text(encoding='utf-8')
        lower = source.lower()
        import yaml
        yaml_config = yaml.safe_load((directory / 'config.yaml').read_text(encoding='utf-8'))
        if not isinstance(yaml_config, Mapping):
            raise TypeError(f'configuration must be a mapping: {directory / "config.yaml"}')
        model_config = yaml_config.get('model') if isinstance(yaml_config.get('model'), Mapping) else {}
        training_config = next((yaml_config[name] for name in ('training', 'train') if isinstance(yaml_config.get(name), Mapping)), {})
        model_defaults = {'space_dim': 7, 'hidden_dim': 256, 'layers': 8, 'heads': 8, 'mlp_ratio': 2, 'out_dim': 4, 'slice_num': 32, 'ref': 8, 'unified_pos': True, 'dropout': 0.0}
        model_aliases = {'space_dim': ('space_dim',), 'hidden_dim': ('hidden_dim', 'n_hidden'), 'layers': ('layers', 'n_layers'), 'heads': ('heads', 'n_heads', 'n_head'), 'mlp_ratio': ('mlp_ratio',), 'out_dim': ('out_dim',), 'slice_num': ('slice_num',), 'ref': ('ref',), 'unified_pos': ('unified_pos',), 'dropout': ('dropout',)}
        train_defaults = {'epochs': 398, 'batch_size': 1, 'learning_rate': 1.0e-3, 'sample_points': 32_000, 'validation_interval': 10, 'validation_repeats': 20}
        train_aliases = {'epochs': ('epochs',), 'batch_size': ('batch_size',), 'learning_rate': ('learning_rate', 'lr'), 'sample_points': ('sample_points', 'subsampling', 'n_points'), 'validation_interval': ('validation_interval', 'val_interval'), 'validation_repeats': ('validation_repeats', 'val_repeats')}
        parameter_checks = []
        for canonical, expected in model_defaults.items():
            alias = next((name for name in model_aliases[canonical] if name in model_config), None)
            actual = model_config.get(alias) if alias is not None else None
            if isinstance(expected, float):
                try:
                    same = math.isclose(float(actual), expected, rel_tol=1.0e-9, abs_tol=1.0e-12)
                except (TypeError, ValueError):
                    same = False
            elif isinstance(expected, bool):
                same = actual is expected or str(actual).lower() == str(expected).lower()
            else:
                try:
                    same = int(actual) == expected
                except (TypeError, ValueError):
                    same = actual == expected
            if alias is None or not same:
                parameter_checks.append(f'{canonical}={actual!r}')
        act = next((model_config[name] for name in ('act', 'activation') if name in model_config), 'gelu' if 'gelu' in (directory / 'transolver.py').read_text(encoding='utf-8').lower() else None)
        if str(act).lower() != 'gelu':
            parameter_checks.append(f'activation={act!r}')
        training_mismatches = []
        for canonical, expected in train_defaults.items():
            alias = next((name for name in train_aliases[canonical] if name in training_config), None)
            actual = training_config.get(alias) if alias is not None else None
            if isinstance(expected, float):
                try:
                    same = math.isclose(float(actual), expected, rel_tol=1.0e-9, abs_tol=1.0e-12)
                except (TypeError, ValueError):
                    same = False
            else:
                try:
                    same = int(actual) == expected
                except (TypeError, ValueError):
                    same = actual == expected
            if alias is None or not same:
                training_mismatches.append(f'{canonical}={actual!r}')
        update_check(current_index, 'argparse' in lower and 'config' in lower and 'yaml' in lower, 'python train.py config.yaml or an equivalent --config interface')
        current_index += 1
        update_check(current_index, not parameter_checks, f'mismatches={parameter_checks}')
        current_index += 1
        update_check(current_index, not training_mismatches, f'mismatches={training_mismatches}')
        current_index += 1
        update_check(current_index, 'transolver' in lower and ('model(' in lower or 'transolver(' in lower), 'training instantiates the configured Transolver')
        current_index += 1
        update_check(current_index, 'surface' in lower and ('volume' in lower or 'vol_' in lower) and ('mse' in lower or 'square' in lower or '** 2' in source), 'volume MSE + surface MSE')
        current_index += 1
        update_check(current_index, ('torch.optim.adam' in lower or 'optim.adam' in lower) and 'onecyclelr' in lower, 'Adam optimizer and OneCycleLR scheduler')
        current_index += 1
        update_check(current_index, 'zero_grad' in lower and 'backward' in lower and 'optimizer.step' in lower, 'zero-grad/backward/optimizer-step')
        current_index += 1
        update_check(current_index, 'scheduler.step' in lower, 'OneCycleLR advances during optimization')
        current_index += 1
        update_check(current_index, ('no_grad' in lower or 'inference_mode' in lower) and 'repeat' in lower and ('valid' in lower or 'val_' in lower), '20 random-resampling validation passes every 10 epochs')
        current_index += 1
        update_check(current_index, 'torch.save' in lower and 'checkpoint' in lower and ('history' in lower or 'json' in lower), 'training records and checkpoints are persisted')
        current_index += 1
        del reference_dir
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['eval:optional_absent', 'eval:checkpoint_loading', 'eval:inference_mode', 'eval:field_metrics']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        directory = CANDIDATE_DIR
        reference_dir = REFERENCE_DIR
        path = next((directory / name for name in ('evaluate.py', 'eval.py', 'inference.py', 'test.py') if (directory / name).is_file()), None)
        if path is None:
            update_check(current_index, True, 'no optional evaluation script supplied')
            current_index += 1
            for _ in range(3):
                update_check(current_index, True, 'not applicable: no optional evaluation script supplied')
                current_index += 1
        else:
            update_check(current_index, True, f'optional evaluation script supplied: {path}')
            current_index += 1
            source = path.read_text(encoding='utf-8').lower()
            update_check(current_index, 'torch.load' in source and 'load_state_dict' in source, str(path))
            current_index += 1
            update_check(current_index, '.eval(' in source and ('no_grad' in source or 'inference_mode' in source), str(path))
            current_index += 1
            update_check(current_index, 'mse' in source and any(name in source for name in ('ux', 'uy', 'pressure', 'nut')), str(path))
            current_index += 1
        del reference_dir
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['model:torch_module', 'model:eight_blocks', 'model:physics_attention_per_block', 'model:mlp_per_block', 'model:layer_norms', 'model:input_71_channels', 'model:hidden_256', 'model:eight_heads', 'model:thirty_two_slices', 'model:four_outputs']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        import torch
        import torch.nn as nn
        reference_dir = REFERENCE_DIR
        path = CANDIDATE_DIR / 'transolver.py'
        spec = importlib.util.spec_from_file_location('transolver_airfoil_train_candidate_structure', path)
        if spec is None or spec.loader is None:
            raise ImportError(f'could not load module from {path}')
        directory = str(path.parent)
        while directory in sys.path:
            sys.path.remove(directory)
        sys.path.insert(0, directory)
        module = importlib.util.module_from_spec(spec)
        sys.modules['transolver_airfoil_train_candidate_structure'] = module
        spec.loader.exec_module(module)
        model_class = next((getattr(module, name) for name in ('Transolver', 'Model', 'Transolver2D') if inspect.isclass(getattr(module, name, None))), None)
        if model_class is None:
            raise TypeError(f'no Transolver model class found in {module.__file__}')
        if not issubclass(model_class, torch.nn.Module):
            raise TypeError(f'{model_class.__name__} is not torch.nn.Module')
        defaults = {'space_dim': 7, 'hidden_dim': 256, 'layers': 8, 'heads': 8, 'mlp_ratio': 2, 'out_dim': 4, 'slice_num': 32, 'ref': 8, 'unified_pos': True, 'dropout': 0.0}
        aliases = {'space_dim': ('space_dim',), 'hidden_dim': ('hidden_dim', 'n_hidden'), 'layers': ('layers', 'n_layers'), 'heads': ('heads', 'n_heads', 'n_head'), 'mlp_ratio': ('mlp_ratio',), 'out_dim': ('out_dim',), 'slice_num': ('slice_num',), 'ref': ('ref',), 'unified_pos': ('unified_pos',), 'dropout': ('dropout',)}
        signature = inspect.signature(model_class)
        accepts_kwargs = any(value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values())
        kwargs = {}
        for canonical, expected in defaults.items():
            selected = next((name for name in aliases[canonical] if name in signature.parameters), None)
            if selected is None and accepts_kwargs:
                selected = aliases[canonical][0]
            if selected is not None:
                kwargs[selected] = expected
        for name in ('act', 'activation'):
            if name in signature.parameters:
                kwargs[name] = 'gelu'
                break
        model = model_class(**kwargs).cpu()
        blocks = []
        for name in ('blocks', 'transformer', 'layers'):
            value = getattr(model, name, None)
            if value is not None and hasattr(value, '__len__'):
                try:
                    members = list(value)
                except TypeError:
                    continue
                if len(members) == 8:
                    blocks = members
                    break
        if not blocks:
            for member in model.modules():
                if isinstance(member, nn.ModuleList) and len(member) == 8:
                    blocks = list(member)
                    break
        linears = [member for member in model.modules() if isinstance(member, nn.Linear)]
        layer_norms = [member for member in model.modules() if isinstance(member, nn.LayerNorm)]
        attention_blocks = [block for block in blocks if any('attention' in type(member).__name__.lower() for member in block.modules())]
        residual_mlps = [block for block in blocks if any('mlp' in type(member).__name__.lower() or 'feedforward' in type(member).__name__.lower() for member in block.modules())]
        head_values = [getattr(member, name) for member in model.modules() for name in ('heads', 'num_heads', 'n_heads', 'n_head') if hasattr(member, name) and isinstance(getattr(member, name), int)]
        slice_values = [member.out_features for member in linears if member.out_features == 32 and member.in_features == 256 // 8]
        update_check(current_index, isinstance(model, nn.Module), type(model).__name__)
        current_index += 1
        update_check(current_index, len(blocks) == 8, f'block_count={len(blocks)}')
        current_index += 1
        update_check(current_index, len(attention_blocks) == 8, f'attention_blocks={len(attention_blocks)}')
        current_index += 1
        update_check(current_index, len(residual_mlps) == 8, f'mlp_blocks={len(residual_mlps)}')
        current_index += 1
        update_check(current_index, len(layer_norms) >= 16, f'layer_norm_count={len(layer_norms)}')
        current_index += 1
        update_check(current_index, any(member.in_features == 7 + 8 * 8 for member in linears), 'input MLP consumes 7 raw + 64 distance features')
        current_index += 1
        update_check(current_index, any(member.out_features == 256 for member in linears), 'hidden width 256')
        current_index += 1
        update_check(current_index, 8 in head_values, f'head_values={head_values}')
        current_index += 1
        update_check(current_index, bool(slice_values), f'matching_slice_projections={len(slice_values)}')
        current_index += 1
        update_check(current_index, any(member.out_features == 4 for member in linears), 'four aerodynamic output channels')
        current_index += 1
        del reference_dir
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['forward:return_tensor', 'forward:shape', 'forward:finite']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        import torch
        reference_dir = REFERENCE_DIR
        path = CANDIDATE_DIR / 'transolver.py'
        spec = importlib.util.spec_from_file_location('transolver_airfoil_train_candidate_forward', path)
        if spec is None or spec.loader is None:
            raise ImportError(f'could not load module from {path}')
        directory = str(path.parent)
        while directory in sys.path:
            sys.path.remove(directory)
        sys.path.insert(0, directory)
        module = importlib.util.module_from_spec(spec)
        sys.modules['transolver_airfoil_train_candidate_forward'] = module
        spec.loader.exec_module(module)
        model_class = next((getattr(module, name) for name in ('Transolver', 'Model', 'Transolver2D') if inspect.isclass(getattr(module, name, None))), None)
        if model_class is None or not issubclass(model_class, torch.nn.Module):
            raise TypeError('no torch.nn.Module Transolver model class found')
        defaults = {'space_dim': 7, 'hidden_dim': 256, 'layers': 8, 'heads': 8, 'mlp_ratio': 2, 'out_dim': 4, 'slice_num': 32, 'ref': 8, 'unified_pos': True, 'dropout': 0.0}
        signature = inspect.signature(model_class)
        kwargs = {name: value for name, value in defaults.items() if name in signature.parameters or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())}
        if 'act' in signature.parameters:
            kwargs['act'] = 'gelu'
        elif 'activation' in signature.parameters:
            kwargs['activation'] = 'gelu'
        torch.manual_seed(20260819)
        model = model_class(**kwargs).cpu().eval()
        generator = torch.Generator(device='cpu').manual_seed(20260819)
        x = torch.randn(64, 7, generator=generator)
        pos = torch.empty(64, 2)
        pos[:, 0] = torch.linspace(-2.0, 4.0, 64)
        pos[:, 1] = torch.linspace(-1.5, 1.5, 64).flip(0)
        with torch.no_grad():
            output = model(x, pos)
        tensor_output = isinstance(output, torch.Tensor)
        shape = tuple(output.shape) if tensor_output else None
        finite = tensor_output and bool(torch.isfinite(output).all())
        update_check(current_index, tensor_output, f'type={type(output).__name__}')
        current_index += 1
        update_check(current_index, shape == (64, 4), f'actual={shape} expected={(64, 4)}')
        current_index += 1
        update_check(current_index, finite, 'all output values must be finite')
        current_index += 1
        del reference_dir
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['equivalence:parameter_shapes', 'equivalence:forward_values']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        import torch
        import torch.nn as nn
        def load_model(path, name):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f'could not load module from {path}')
            directory = str(path.parent)
            while directory in sys.path:
                sys.path.remove(directory)
            sys.path.insert(0, directory)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            model_class = next((getattr(module, item) for item in ('Transolver', 'Model', 'Transolver2D') if inspect.isclass(getattr(module, item, None))), None)
            if model_class is None or not issubclass(model_class, torch.nn.Module):
                raise TypeError('no torch.nn.Module Transolver model class found')
            defaults = {'space_dim': 7, 'hidden_dim': 256, 'layers': 8, 'heads': 8, 'mlp_ratio': 2, 'out_dim': 4, 'slice_num': 32, 'ref': 8, 'unified_pos': True, 'dropout': 0.0}
            signature = inspect.signature(model_class)
            accepts_kwargs = any(value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values())
            kwargs = {item: value for item, value in defaults.items() if item in signature.parameters or accepts_kwargs}
            if 'act' in signature.parameters:
                kwargs['act'] = 'gelu'
            elif 'activation' in signature.parameters:
                kwargs['activation'] = 'gelu'
            return model_class(**kwargs).cpu().eval()
        generated = load_model(CANDIDATE_DIR / 'transolver.py', 'transolver_airfoil_train_candidate_equivalence')
        reference = load_model(REFERENCE_DIR / 'transolver.py', 'transolver_airfoil_train_reference_equivalence')
        generated_shapes = Counter(tuple(parameter.shape) for parameter in generated.parameters())
        reference_shapes = Counter(tuple(parameter.shape) for parameter in reference.parameters())
        for model in (generated, reference):
            handled = set()
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        values = torch.arange(module.weight.numel(), dtype=torch.float64).reshape(module.weight.shape)
                        module.weight.copy_(((values.remainder(41) - 20.0) / 4000.0).to(dtype=module.weight.dtype))
                        handled.add(id(module.weight))
                        if module.bias is not None:
                            module.bias.zero_()
                            handled.add(id(module.bias))
                    elif isinstance(module, nn.LayerNorm):
                        if module.weight is not None:
                            module.weight.fill_(1.0)
                            handled.add(id(module.weight))
                        if module.bias is not None:
                            module.bias.zero_()
                            handled.add(id(module.bias))
                    elif isinstance(module, nn.BatchNorm1d):
                        if module.weight is not None:
                            module.weight.fill_(1.0)
                            handled.add(id(module.weight))
                        if module.bias is not None:
                            module.bias.zero_()
                            handled.add(id(module.bias))
                        module.running_mean.zero_()
                        module.running_var.fill_(1.0)
                for parameter in model.parameters():
                    if id(parameter) in handled:
                        continue
                    if tuple(parameter.shape) == (1, 8, 1, 1):
                        parameter.fill_(0.5)
                    elif parameter.ndim <= 1:
                        parameter.fill_(0.01)
                    else:
                        values = torch.arange(parameter.numel(), dtype=torch.float64).reshape(parameter.shape)
                        parameter.copy_(((values.remainder(37) - 18.0) / 5000.0).to(parameter.dtype))
        generator = torch.Generator(device='cpu').manual_seed(20260819)
        x = torch.randn(64, 7, generator=generator)
        pos = torch.empty(64, 2)
        pos[:, 0] = torch.linspace(-2.0, 4.0, 64)
        pos[:, 1] = torch.linspace(-1.5, 1.5, 64).flip(0)
        with torch.no_grad():
            actual = generated(x.clone(), pos.clone())
            expected = reference(x.clone(), pos.clone())
        same_shape = isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor) and tuple(actual.shape) == tuple(expected.shape) == (64, 4)
        max_difference = float((actual - expected).abs().max()) if same_shape else float('inf')
        equivalent = same_shape and bool(torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5))
        update_check(current_index, generated_shapes == reference_shapes, f'generated={sum(generated_shapes.values())} reference={sum(reference_shapes.values())} shape_delta={generated_shapes - reference_shapes or reference_shapes - generated_shapes}')
        current_index += 1
        update_check(current_index, equivalent, f'max_abs_diff={max_difference:.3e} atol={1.0e-5} rtol={1.0e-5}')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['training:parameters', 'training:forward_shape', 'training:finite_loss', 'training:finite_gradients', 'training:parameter_updates']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        import torch
        reference_dir = REFERENCE_DIR
        path = CANDIDATE_DIR / 'transolver.py'
        spec = importlib.util.spec_from_file_location('transolver_airfoil_train_candidate_training', path)
        if spec is None or spec.loader is None:
            raise ImportError(f'could not load module from {path}')
        directory = str(path.parent)
        while directory in sys.path:
            sys.path.remove(directory)
        sys.path.insert(0, directory)
        module = importlib.util.module_from_spec(spec)
        sys.modules['transolver_airfoil_train_candidate_training'] = module
        spec.loader.exec_module(module)
        model_class = next((getattr(module, name) for name in ('Transolver', 'Model', 'Transolver2D') if inspect.isclass(getattr(module, name, None))), None)
        if model_class is None or not issubclass(model_class, torch.nn.Module):
            raise TypeError('no torch.nn.Module Transolver model class found')
        defaults = {'space_dim': 7, 'hidden_dim': 256, 'layers': 8, 'heads': 8, 'mlp_ratio': 2, 'out_dim': 4, 'slice_num': 32, 'ref': 8, 'unified_pos': True, 'dropout': 0.0}
        signature = inspect.signature(model_class)
        accepts_kwargs = any(value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values())
        kwargs = {name: value for name, value in defaults.items() if name in signature.parameters or accepts_kwargs}
        if 'act' in signature.parameters:
            kwargs['act'] = 'gelu'
        elif 'activation' in signature.parameters:
            kwargs['activation'] = 'gelu'
        torch.manual_seed(20260819)
        model = model_class(**kwargs).cpu().train()
        parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        update_check(current_index, bool(parameters), 'trainable parameters found' if parameters else 'no trainable parameters')
        current_index += 1
        if parameters:
            before = {name: value.detach().clone() for name, value in parameters.items()}
            optimizer = torch.optim.Adam(parameters.values(), lr=1.0e-3)
            generator = torch.Generator(device='cpu').manual_seed(20260819)
            x = torch.randn(32, 7, generator=generator)
            pos = torch.empty(32, 2)
            pos[:, 0] = torch.linspace(-2.0, 4.0, 32)
            pos[:, 1] = torch.linspace(-1.5, 1.5, 32).flip(0)
            target = torch.randn(32, 4, generator=generator)
            surface = torch.zeros(32, dtype=torch.bool)
            surface[:12] = True
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, pos)
            squared_error = (prediction - target).square()
            volume_loss = squared_error[~surface].mean()
            surface_loss = squared_error[surface].mean()
            loss = volume_loss + surface_loss
            loss.backward()
            finite_gradients = {name for name, parameter in parameters.items() if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())}
            optimizer.step()
            changed = {name for name, parameter in parameters.items() if not torch.equal(before[name], parameter.detach())}
            gradient_fraction = len(finite_gradients) / len(parameters)
            update_fraction = len(changed) / len(parameters)
            update_check(current_index, tuple(prediction.shape) == (32, 4), f'shape={tuple(prediction.shape)}')
            current_index += 1
            update_check(current_index, bool(torch.isfinite(loss)), f'volume={float(volume_loss.detach()):.6g} surface={float(surface_loss.detach()):.6g}')
            current_index += 1
            update_check(current_index, gradient_fraction >= 0.9, f'finite_gradient_parameters={len(finite_gradients)}/{len(parameters)}')
            current_index += 1
            update_check(current_index, update_fraction >= 0.9, f'updated_parameters={len(changed)}/{len(parameters)}')
            current_index += 1
        del reference_dir
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['artifacts:checkpoint', 'artifacts:training_history', 'artifacts:normalization_state']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        directory = CANDIDATE_DIR
        reference_dir = REFERENCE_DIR
        train_source = (directory / 'train.py').read_text(encoding='utf-8').lower()
        data_source = (directory / 'dataset.py').read_text(encoding='utf-8').lower()
        project_source = '\n'.join(path.read_text(encoding='utf-8') for path in sorted(directory.rglob('*.py')) if not any(part.startswith('.') for part in path.relative_to(directory).parts)).lower()
        update_check(current_index, 'torch.save' in train_source and 'checkpoint' in train_source and ('.pt' in train_source or '.pth' in train_source), 'checkpoint save declaration')
        current_index += 1
        update_check(current_index, 'history' in train_source and any(token in train_source for token in ('json.dump', 'jsonl', 'csv')), 'machine-readable training history')
        current_index += 1
        update_check(current_index, 'normalization' in project_source and 'mean' in data_source and 'std' in data_source and ('torch.save' in project_source or 'np.save' in project_source), 'training normalization statistics are persisted')
        current_index += 1
        del reference_dir
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks
