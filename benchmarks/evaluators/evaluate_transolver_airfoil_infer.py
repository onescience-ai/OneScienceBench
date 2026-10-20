'''V2 evaluator adapter for the Transolver2D airfoil inference evaluator.'''

from pathlib import Path
import sys
import ast
import importlib.util
import os
import traceback
from types import SimpleNamespace
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


REQUIRED_FILES = ('Transolver2D.py', 'AirfRANSDataSet.py')
MODEL_CONFIG = {'space_dim': 7, 'fun_dim': 0, 'n_hidden': 256, 'n_layers': 8, 'n_head': 8, 'mlp_ratio': 2, 'out_dim': 4, 'slice_num': 32, 'ref': 8, 'unified_pos': True}
CHECKPOINT_PATH = Path('/public/share/sugonhpcapp01/onestore/onedatasets/Transolver-Airfoil-Design/weight/best.pth')
POINT_COUNT = 64
SEED = 42
ATOL = 1.0e-5
RTOL = 1.0e-5

def _load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法加载模块: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _load_module_safely(name: str, path: Path) -> tuple[Any, str | None]:
    try:
        return _load_module(name, path), None
    except BaseException as exc:
        return None, f'{type(exc).__name__}: {exc}'

def _load_checkpoint(torch, checkpoint_path: Path) -> Mapping[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise TypeError('checkpoint must be a mapping')
    state = checkpoint.get('model_state_dict')
    if not isinstance(state, Mapping) or not state:
        raise KeyError('checkpoint missing model_state_dict')
    return state

def _model_paths(config: Dict):
    model_file = config.get('model_file', 'Transolver2D.py')
    candidate_path = (CANDIDATE_DIR / model_file).resolve()
    candidate_path.relative_to(CANDIDATE_DIR)
    reference_path = (REFERENCE_DIR / model_file).resolve()
    reference_path.relative_to(REFERENCE_DIR)
    return candidate_path, reference_path

def _modules(config: Dict, tag: str):
    candidate_path, reference_path = _model_paths(config)
    generated_module, generated_error = _load_module_safely(f'transolver_generated_{tag}', candidate_path)
    reference_module, reference_error = _load_module_safely(f'transolver_reference_{tag}', reference_path)
    return generated_module, reference_module, generated_error, reference_error

def _runtime(config: Dict, tag: str):
    import torch
    generated_module, reference_module, generated_error, reference_error = _modules(config, tag)
    checkpoint_path = Path(config.get('checkpoint_file', str(CHECKPOINT_PATH))).expanduser().resolve()
    state = _load_checkpoint(torch, checkpoint_path)
    model_class_name = config.get('model_class', 'Transolver2D')
    model_args = config.get('model_args', MODEL_CONFIG)
    generated_class = getattr(generated_module, model_class_name)
    reference_class = getattr(reference_module, model_class_name)
    if not issubclass(generated_class, torch.nn.Module) or not issubclass(reference_class, torch.nn.Module):
        raise TypeError(f'{model_class_name} must inherit torch.nn.Module')
    strict = config.get('strict', True)
    generated_model = generated_class(**model_args).cpu()
    reference_model = reference_class(**model_args).cpu()
    generated_model.load_state_dict(state, strict=strict)
    reference_model.load_state_dict(state, strict=strict)
    generator = torch.Generator(device='cpu').manual_seed(config.get('seed', SEED))
    model_input = SimpleNamespace(
        x=torch.randn(POINT_COUNT, 7, generator=generator),
        pos=torch.stack((torch.linspace(-2.0, 4.0, POINT_COUNT), torch.linspace(-1.5, 1.5, POINT_COUNT)), dim=-1),
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    outputs = []
    for model in (generated_model, reference_model):
        model.to(device).eval()
        data = SimpleNamespace(x=model_input.x.to(device), pos=model_input.pos.to(device))
        with torch.no_grad():
            outputs.append(model(data).detach().cpu())
    return torch, generated_model, reference_model, outputs[0], outputs[1], device

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['required:Transolver2D.py', 'syntax:Transolver2D.py', 'required:AirfRANSDataSet.py', 'syntax:AirfRANSDataSet.py']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        for name in REQUIRED_FILES:
            path = (CANDIDATE_DIR / name).resolve()
            path.relative_to(CANDIDATE_DIR)
            update_check(current_index, path.is_file(), str(path))
            current_index += 1
            try:
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                ok, detail = True, 'syntax is valid'
            except Exception as exc:
                ok, detail = False, f'{type(exc).__name__}: {exc}'
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['generated_module', 'reference_module']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        files = tuple(config.get('files', ('Transolver2D.py',)))
        local_config = dict(config)
        local_config['model_file'] = files[0] if files else 'Transolver2D.py'
        generated_module, reference_module, generated_error, reference_error = _modules(local_config, 'interface')
        update_check(current_index, generated_module is not None, str((CANDIDATE_DIR / local_config['model_file']).resolve()) if generated_error is None else generated_error)
        current_index += 1
        update_check(current_index, reference_module is not None, str((REFERENCE_DIR / local_config['model_file']).resolve()) if reference_error is None else reference_error)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['checkpoint', 'reference_strict_load', 'generated_strict_load']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        generated_module, reference_module, generated_error, reference_error = _modules(config, 'checkpoint')
        checkpoint_path = Path(config.get('checkpoint_file', str(CHECKPOINT_PATH))).expanduser().resolve()
        state = _load_checkpoint(torch, checkpoint_path)
        update_check(current_index, True, f'model_state_dict parameters={len(state)}')
        current_index += 1
        model_args = config.get('model_args', MODEL_CONFIG)
        model_class_name = config.get('model_class', 'Transolver2D')
        strict = config.get('strict', True)
        for module, label in ((reference_module, 'reference'), (generated_module, 'generated')):
            try:
                model_class = getattr(module, model_class_name)
                if not issubclass(model_class, torch.nn.Module):
                    raise TypeError(f'{model_class_name} must inherit torch.nn.Module')
                model = model_class(**model_args).cpu()
                model.load_state_dict(state, strict=strict)
                update_check(current_index, True, f'parameters={len(model.state_dict())}')
            except Exception as exc:
                update_check(current_index, False, f'{type(exc).__name__}: {exc}')
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['forward_execution_contract', 'generated_output_shape']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        torch, generated_model, reference_model, generated_output, reference_output, device = _runtime(config, 'forward')
        update_check(current_index, generated_output is not None and reference_output is not None, 'both forward outputs retained')
        current_index += 1
        expected_shape = tuple(config.get('out_shape', (POINT_COUNT, 4)))
        update_check(current_index, tuple(generated_output.shape) == expected_shape, f'actual={tuple(generated_output.shape)}, expected={expected_shape}')
        current_index += 1
        generated_model.to('cpu')
        reference_model.to('cpu')
        if getattr(device, 'type', 'cpu') == 'cuda':
            torch.cuda.empty_cache()
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_output_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['generated_output_finite']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        torch, generated_model, reference_model, generated_output, reference_output, device = _runtime(config, 'schema')
        update_check(current_index, bool(torch.isfinite(generated_output).all()), 'generated output must be finite')
        current_index += 1
        generated_model.to('cpu')
        reference_model.to('cpu')
        if getattr(device, 'type', 'cpu') == 'cuda':
            torch.cuda.empty_cache()
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_metrics(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['metric:mse', 'metric:mae', 'metric:max_abs_error']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        torch, generated_model, reference_model, generated_output, reference_output, device = _runtime(config, 'metrics')
        if generated_output.shape != reference_output.shape:
            raise ValueError(f'output shape mismatch: generated={tuple(generated_output.shape)}, reference={tuple(reference_output.shape)}')
        difference = generated_output.double() - reference_output.double()
        metrics = {'mse': float(difference.square().mean()), 'mae': float(difference.abs().mean()), 'max_abs_error': float(difference.abs().max())}
        update_check(current_index, True, str(metrics['mse']))
        current_index += 1
        update_check(current_index, True, str(metrics['mae']))
        current_index += 1
        update_check(current_index, True, str(metrics['max_abs_error']))
        current_index += 1
        generated_model.to('cpu')
        reference_model.to('cpu')
        if getattr(device, 'type', 'cpu') == 'cuda':
            torch.cuda.empty_cache()
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['metric_forward_results_available', 'metric_shape_match', 'metric_forward_equivalence']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        torch, generated_model, reference_model, generated_output, reference_output, device = _runtime(config, 'oracle')
        available = generated_output is not None and reference_output is not None
        update_check(current_index, available, 'forward results available' if available else 'missing forward results')
        current_index += 1
        if available:
            shape_match = generated_output.shape == reference_output.shape
            update_check(current_index, shape_match, f'generated={tuple(generated_output.shape)}, reference={tuple(reference_output.shape)}')
            current_index += 1
            if shape_match:
                atol = config.get('atol', ATOL)
                rtol = config.get('rtol', RTOL)
                equivalent = bool(torch.allclose(generated_output, reference_output, atol=atol, rtol=rtol))
                difference = generated_output.double() - reference_output.double()
                metrics = {'mse': float(difference.square().mean()), 'mae': float(difference.abs().mean()), 'max_abs_error': float(difference.abs().max())}
                update_check(current_index, equivalent, str({**metrics, 'atol': atol, 'rtol': rtol}))
                current_index += 1
        generated_model.to('cpu')
        reference_model.to('cpu')
        if getattr(device, 'type', 'cpu') == 'cuda':
            torch.cuda.empty_cache()
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks
