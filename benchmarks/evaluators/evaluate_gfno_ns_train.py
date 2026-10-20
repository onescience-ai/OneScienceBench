'''V2 adapter for the legacy GFNO Navier--Stokes evaluator.'''

import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Iterable


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


REQUIRED_FILES = ('model.py', 'dataset.py', 'utils.py', 'train.py')
MODEL_DEFAULTS = {
    'num_channels': 1,
    'modes': 12,
    'width': 10,
    'initial_step': 10,
}
TRAIN_DEFAULTS = {
    'epochs': 100,
    'batch_size': 20,
    'learning_rate': 1.0e-3,
    'weight_decay': 1.0e-4,
}
SEED = 20260820
SPATIAL_SIZE = 24
INPUT_SHAPE = (1, SPATIAL_SIZE, SPATIAL_SIZE, 10, 1)
OUTPUT_SHAPE = (1, SPATIAL_SIZE, SPATIAL_SIZE, 1, 1)
ROLLOUT_TEST_STEPS = 3
FORWARD_ATOL = 1.0e-5
FORWARD_RTOL = 1.0e-5
EQUIVARIANCE_RTOL = 2.0e-4

def _read(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _project_source(directory: Path, names: Iterable[str]) -> str:
    return '\n'.join(
        _read(directory / name) for name in names if (directory / name).is_file()
    )


def _compact(source: str) -> str:
    return ''.join(source.lower().split())


def _load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
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
    return module


def _build_model(module: Any) -> Any:
    model_class = getattr(module, 'GFNO', None)
    if not inspect.isclass(model_class):
        raise TypeError('model.py must define class GFNO')
    signature = inspect.signature(model_class)
    aliases = {
        'num_channels': 1,
        'in_channels': 1,
        'out_channels': 1,
        'modes': 12,
        'modes1': 12,
        'modes2': 12,
        'width': 10,
        'initial_step': 10,
        'history_steps': 10,
        't_in': 10,
        'T_in': 10,
        'n_blocks': 4,
    }
    kwargs = {
        name: value
        for name, value in aliases.items()
        if name in signature.parameters
    }
    return model_class(**kwargs)


def _constructor_defaults(model_class: type) -> dict[str, Any]:
    signature = inspect.signature(model_class)
    aliases = {
        'num_channels': ('num_channels', 'in_channels'),
        'modes': ('modes', 'modes1'),
        'width': ('width',),
        'initial_step': ('initial_step', 'history_steps', 't_in', 'T_in'),
    }
    values: dict[str, Any] = {}
    for key, names in aliases.items():
        for name in names:
            parameter = signature.parameters.get(name)
            if parameter is not None and parameter.default is not inspect.Parameter.empty:
                values[key] = parameter.default
                break
    return values


def _input(torch: Any) -> Any:
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    return torch.randn(INPUT_SHAPE, generator=generator)


def _relative_error(torch: Any, actual: Any, expected: Any) -> float:
    numerator = torch.linalg.vector_norm((actual - expected).reshape(-1))
    denominator = torch.linalg.vector_norm(expected.reshape(-1)).clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'required:model.py',
        'syntax:model.py',
        'required:dataset.py',
        'syntax:dataset.py',
        'required:utils.py',
        'syntax:utils.py',
        'required:train.py',
        'syntax:train.py',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        for name in REQUIRED_FILES:
            path = CANDIDATE_DIR / name
            file_exists = path.is_file()
            update_check(current_index, file_exists, str(path))
            current_index += 1
            if file_exists:
                try:
                    source = _read(path)
                    ast.parse(source, filename=str(path))
                    update_check(
                        current_index,
                        bool(source.strip()),
                        f'lines={len(source.splitlines())}',
                    )
                except Exception as exc:
                    update_check(
                        current_index,
                        False,
                        f'{type(exc).__name__}: {exc}',
                    )
            else:
                update_check(current_index, False, f'not a regular file: {path}')
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_data_normalization(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'data:dataset_file',
        'data:mat_reader',
        'data:float32_five_dimensional',
        'data:official_split',
        'data:rollout_training_samples',
        'data:rollout_samples',
        'data:normalization_default_none',
        'data:train_only_statistics',
        'data:normalization_statistics',
        'data:dataloaders',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = CANDIDATE_DIR / 'dataset.py'
        if not path.is_file():
            update_check(current_index, False, str(path))
            current_index += 1
        else:
            update_check(current_index, True, str(path))
            current_index += 1
            source = _read(path)
            project_source = _project_source(CANDIDATE_DIR, ('dataset.py', 'train.py'))
            lower = source.lower()
            compact = _compact(source)
            train_slice = any(
                token in compact
                for token in ('data[:ntrain]', 'range(0,1000)', 'range(1000)')
            )
            valid_slice = any(
                token in compact
                for token in (
                    'data[-(ntest+nvalid):-ntest]',
                    'data[-(nvalid+ntest):-ntest]',
                    'data[-200:-100]',
                    'range(num_samples-200,num_samples-100)',
                )
            )
            test_slice = any(
                token in compact
                for token in (
                    'data[-ntest:]',
                    'data[-100:]',
                    'range(num_samples-100,num_samples)',
                )
            )
            ok = (
                ('loadmat' in lower or 'h5py' in lower)
                and ("'u'" in lower or '"u"' in lower)
            )
            update_check(current_index, ok, 'MAT/HDF5 reader exposes the vorticity variable u')
            current_index += 1
            ok = 'float32' in lower and any(
                token in lower for token in ('[..., none]', 'unsqueeze', 'ndim != 5')
            )
            update_check(current_index, ok, 'logical trajectory layout [N,H,W,T,1]')
            current_index += 1
            update_check(
                current_index,
                train_slice and valid_slice and test_slice,
                'train [:1000], validation [-200:-100], test [-100:]',
            )
            current_index += 1
            ok = (
                'rollout_steps' in lower
                and ('future' in lower or 'target' in lower or 'pde' in lower)
                and ('t_out' in lower or '20' in lower)
            )
            update_check(
                current_index,
                ok,
                'training targets retain an explicit multi-step future dimension',
            )
            current_index += 1
            ok = 'rollout' in lower and ('history' in lower or 't_in' in lower)
            update_check(current_index, ok, 'initial 10-frame history and 20-frame future target')
            current_index += 1
            ok = 'normalization' in lower and '"none"' in lower
            update_check(current_index, ok, "normalization='none' is supported")
            current_index += 1
            ok = any(
                token in compact
                for token in ('fit(train', 'fit(cls,train', 'splits.train')
            ) or 'train_indices' in lower
            update_check(current_index, ok, 'normalizer statistics originate from the training split')
            current_index += 1
            ok = 'mean' in lower and 'std' in lower
            update_check(current_index, ok, 'normalizer computes mean and standard deviation')
            current_index += 1
            ok = (
                'dataloader' in project_source.lower()
                and 'shuffle=true' in _compact(project_source)
                and 'shuffle=false' in _compact(project_source)
            )
            update_check(
                current_index,
                ok,
                'shuffled train and deterministic validation/test loaders',
            )
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'train:script_file',
        'train:gfno_usage',
        'train:defaults',
        'train:adam_scheduler',
        'train:rollout_relative_l2',
        'train:differentiable_feedback',
        'train:optimization_step',
        'train:validation_rollout_each_epoch',
        'train:best_and_last_checkpoints',
        'train:reload_best_for_test',
        'train:rotation_metric',
        'train:result_files',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = CANDIDATE_DIR / 'train.py'
        if not path.is_file():
            update_check(current_index, False, str(path))
            current_index += 1
        else:
            update_check(current_index, True, str(path))
            current_index += 1
            train_source = _read(path)
            project_source = _project_source(CANDIDATE_DIR, ('train.py', 'utils.py'))
            lower = project_source.lower()
            compact = _compact(project_source)
            rollout_call = (
                'autoregressive_rollout(' in compact
                or 'rollout_relative_l2_loss(' in compact
                or ('for' in lower and 'torch.cat' in lower and 'predictions' in lower)
            )
            relative_l2 = any(
                token in lower for token in ('lploss', 'relative_l2', 'relative l2')
            )
            rollout_default = 'rollout_steps' in lower and '20' in lower
            update_check(
                current_index,
                'gfno(' in train_source.lower(),
                'train.py instantiates GFNO',
            )
            current_index += 1
            ok = (
                '100' in lower
                and '20' in lower
                and ('1e-3' in lower or '0.001' in lower)
                and ('1e-4' in lower or '0.0001' in lower)
                and ('epochs' in lower and 'batch_size' in lower)
            )
            update_check(current_index, ok, f'expected={TRAIN_DEFAULTS}')
            current_index += 1
            ok = 'torch.optim.adam' in lower and (
                'cosineannealinglr' in lower or 'scheduler' in lower
            )
            update_check(current_index, ok, 'Adam and a learning-rate scheduler')
            current_index += 1
            update_check(
                current_index,
                rollout_call and relative_l2 and rollout_default,
                'multi-step autoregressive Relative L2 rollout loss (default 20 steps)',
            )
            current_index += 1
            ok = 'backward' in lower and (
                'detach_rollout_history' not in lower or 'false' in lower
            )
            update_check(current_index, ok, 'rollout feedback keeps gradients by default')
            current_index += 1
            ok = (
                'zero_grad' in lower
                and 'backward' in lower
                and 'optimizer.step' in lower
            )
            update_check(current_index, ok, 'zero-grad/backward/optimizer-step loop')
            current_index += 1
            ok = (
                ('evaluate(' in lower or 'evaluate_rollout(' in lower)
                and ('t_out' in lower or 'steps' in lower)
                and 'for epoch' in lower
            )
            update_check(
                current_index,
                ok,
                '20-step autoregressive validation inside the epoch loop',
            )
            current_index += 1
            ok = 'best.pt' in lower and 'last.pt' in lower and 'save_checkpoint' in lower
            update_check(current_index, ok, 'best and final training-state checkpoints')
            current_index += 1
            ok = 'load_checkpoint' in lower and 'test' in lower and 'best' in lower
            update_check(
                current_index,
                ok,
                'best checkpoint is reloaded before final test evaluation',
            )
            current_index += 1
            update_check(
                current_index,
                'rotation_equivariance' in lower,
                'rotation equivariance error is included in final evaluation',
            )
            current_index += 1
            ok = all(name in lower for name in ('history.json', 'summary.json', 'train.log'))
            update_check(current_index, ok, 'history.json, summary.json, and train.log')
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'eval:optional_absent',
        'eval:checkpoint',
        'eval:inference_mode',
        'eval:rollout_and_rotation',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = next(
            (
                CANDIDATE_DIR / name
                for name in ('evaluate.py', 'eval.py', 'test.py')
                if (CANDIDATE_DIR / name).is_file()
            ),
            None,
        )
        if path is None:
            update_check(
                current_index,
                True,
                'evaluation is integrated into train.py',
            )
            current_index += 1
            update_check(
                current_index,
                True,
                'not applicable because no standalone evaluation script is present',
            )
            current_index += 1
            update_check(
                current_index,
                True,
                'not applicable because no standalone evaluation script is present',
            )
            current_index += 1
            update_check(
                current_index,
                True,
                'not applicable because no standalone evaluation script is present',
            )
            current_index += 1
        else:
            update_check(current_index, True, f'standalone evaluation script present: {path}')
            current_index += 1
            lower = _read(path).lower()
            update_check(
                current_index,
                'load' in lower and 'checkpoint' in lower,
                str(path),
            )
            current_index += 1
            update_check(
                current_index,
                '.eval(' in lower and ('no_grad' in lower or 'inference_mode' in lower),
                str(path),
            )
            current_index += 1
            update_check(
                current_index,
                'rollout' in lower and 'rotation' in lower,
                str(path),
            )
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'model:gfno_class',
        'model:defaults',
        'model:four_blocks',
        'model:p4_group_convolution',
        'model:group_spectral',
        'model:group_mlp',
        'model:residual_1x1',
        'model:lifting_projection',
        'model:trainable',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch.nn as nn

        generated_module = _load_module(
            'gfno_ns_train_generated_model', CANDIDATE_DIR / 'model.py'
        )
        reference_module = _load_module(
            'gfno_ns_train_reference_model', REFERENCE_DIR / 'model.py'
        )
        del reference_module
        model_class = getattr(generated_module, 'GFNO', None)
        model = _build_model(generated_module).cpu()
        defaults = _constructor_defaults(model_class)
        expected_defaults = {
            key: value
            for key, value in MODEL_DEFAULTS.items()
            if key in defaults or key != 'num_channels'
        }
        defaults_ok = all(
            defaults.get(key) == value for key, value in expected_defaults.items()
        )
        modules = list(model.modules())
        names = [type(module).__name__.lower() for module in modules]
        source = inspect.getsource(generated_module).lower()
        spectral_count = sum('spectral' in name and name != 'gfno' for name in names)
        block_count = max(
            (
                len(value)
                for name in ('blocks', 'spectral', 'spectral_layers', 'fourier_blocks')
                if isinstance((value := getattr(model, name, None)), nn.ModuleList)
            ),
            default=0,
        )
        if block_count == 0 and spectral_count == 4:
            block_count = 4
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        update_check(
            current_index,
            inspect.isclass(model_class) and isinstance(model, nn.Module),
            type(model).__name__,
        )
        current_index += 1
        update_check(
            current_index,
            defaults_ok,
            f'actual={defaults} expected={expected_defaults}',
        )
        current_index += 1
        update_check(
            current_index,
            block_count == 4 or spectral_count == 4,
            f'blocks={block_count} spectral_modules={spectral_count}',
        )
        current_index += 1
        ok = 'p4' in source and 'rot90' in source and ('roll' in source or 'group' in source)
        update_check(current_index, ok, 'rotated shared kernels and cyclic group action')
        current_index += 1
        ok = 'spectral' in source and 'rfft2' in source and 'irfft2' in source
        update_check(current_index, ok, 'group-equivariant Fourier convolution')
        current_index += 1
        ok = any('mlp' in name for name in names) and 'gelu' in source
        update_check(current_index, ok, 'G-MLP branch')
        current_index += 1
        ok = (
            ('skip' in source or 'residual' in source)
            and 'kernel_size=1' in source.replace(' ', '')
        )
        update_check(current_index, ok, '1x1 group residual branch')
        current_index += 1
        ok = (
            ('lifting' in source or 'self.p' in source)
            and ('projection' in source or 'self.q' in source)
        )
        update_check(current_index, ok, 'planar/group lift and projection')
        current_index += 1
        update_check(current_index, parameter_count > 0, f'parameter_elements={parameter_count}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'forward:return_tensor',
        'forward:shape',
        'forward:finite',
        'forward:p4_equivariance',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        generated_module = _load_module(
            'gfno_ns_train_generated_model', CANDIDATE_DIR / 'model.py'
        )
        reference_module = _load_module(
            'gfno_ns_train_reference_model', REFERENCE_DIR / 'model.py'
        )
        del reference_module
        torch.set_num_threads(1)
        model = _build_model(generated_module).cpu().eval()
        x = _input(torch)
        with torch.no_grad():
            output = model(x)
            rotated_output = model(torch.rot90(x, 1, dims=(1, 2)))
            expected_rotation = torch.rot90(output, 1, dims=(1, 2))
        tensor_output = isinstance(output, torch.Tensor)
        shape = tuple(output.shape) if tensor_output else None
        rotation_error = (
            _relative_error(torch, rotated_output, expected_rotation)
            if tensor_output and tuple(rotated_output.shape) == OUTPUT_SHAPE
            else float('inf')
        )
        update_check(current_index, tensor_output, type(output).__name__)
        current_index += 1
        update_check(current_index, shape == OUTPUT_SHAPE, f'actual={shape} expected={OUTPUT_SHAPE}')
        current_index += 1
        update_check(
            current_index,
            tensor_output and bool(torch.isfinite(output).all()),
            'all values finite',
        )
        current_index += 1
        update_check(
            current_index,
            rotation_error <= EQUIVARIANCE_RTOL,
            f'relative_l2={rotation_error:.3e} tolerance={EQUIVARIANCE_RTOL:.1e}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'equivalence:state_schema',
        'equivalence:forward_values',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        generated_module = _load_module(
            'gfno_ns_train_generated_model', CANDIDATE_DIR / 'model.py'
        )
        reference_module = _load_module(
            'gfno_ns_train_reference_model', REFERENCE_DIR / 'model.py'
        )
        torch.set_num_threads(1)
        generated = _build_model(generated_module).cpu().eval()
        reference = _build_model(reference_module).cpu().eval()
        generated_state = generated.state_dict()
        reference_state = reference.state_dict()
        same_schema = generated_state.keys() == reference_state.keys() and all(
            generated_state[name].shape == value.shape
            for name, value in reference_state.items()
        )
        if not same_schema:
            update_check(
                current_index,
                False,
                f'generated_tensors={len(generated_state)} reference_tensors={len(reference_state)}',
            )
            current_index += 1
            update_check(
                current_index,
                False,
                'strict reference state loading is unavailable for a different architecture',
            )
            current_index += 1
        else:
            update_check(
                current_index,
                True,
                f'tensors={len(reference_state)} strict=True',
            )
            current_index += 1
            generated.load_state_dict(reference_state, strict=True)
            x = _input(torch)
            with torch.no_grad():
                actual = generated(x.clone())
                expected = reference(x.clone())
            same_shape = (
                isinstance(actual, torch.Tensor)
                and isinstance(expected, torch.Tensor)
                and tuple(actual.shape) == tuple(expected.shape) == OUTPUT_SHAPE
            )
            max_difference = (
                float((actual - expected).abs().max())
                if same_shape
                else float('inf')
            )
            equivalent = same_shape and bool(
                torch.allclose(actual, expected, atol=FORWARD_ATOL, rtol=FORWARD_RTOL)
            )
            update_check(
                current_index,
                equivalent,
                f'max_abs_diff={max_difference:.3e} atol={FORWARD_ATOL} rtol={FORWARD_RTOL}',
            )
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'training:parameters',
        'training:rollout_shape',
        'training:rollout_relative_l2',
        'training:finite_gradients',
        'training:parameter_updates',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        generated_module = _load_module(
            'gfno_ns_train_generated_model', CANDIDATE_DIR / 'model.py'
        )
        reference_module = _load_module(
            'gfno_ns_train_reference_model', REFERENCE_DIR / 'model.py'
        )
        del reference_module
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        model = _build_model(generated_module).cpu().train()
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not parameters:
            update_check(current_index, False, 'no trainable parameters')
            current_index += 1
        else:
            update_check(current_index, True, f'trainable_parameters={len(parameters)}')
            current_index += 1
            before = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
            optimizer = torch.optim.Adam(
                parameters.values(), lr=1.0e-3, weight_decay=1.0e-4
            )
            history = _input(torch)
            generator = torch.Generator(device='cpu').manual_seed(SEED + 1)
            target = torch.randn(
                (INPUT_SHAPE[0], SPATIAL_SIZE, SPATIAL_SIZE, ROLLOUT_TEST_STEPS, 1),
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = []
            current = history
            for _ in range(ROLLOUT_TEST_STEPS):
                next_field = model(current)
                predictions.append(next_field)
                current = torch.cat((current[..., 1:, :], next_field), dim=-2)
            prediction = torch.cat(predictions, dim=-2)
            difference = torch.linalg.vector_norm(
                (prediction - target).reshape(len(target), -1), dim=1
            )
            scale = torch.linalg.vector_norm(
                target.reshape(len(target), -1), dim=1
            ).clamp_min(1.0e-12)
            loss = (difference / scale).mean()
            loss.backward()
            finite_gradients = {
                name
                for name, parameter in parameters.items()
                if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            }
            optimizer.step()
            changed = {
                name
                for name, parameter in parameters.items()
                if not torch.equal(before[name], parameter.detach())
            }
            gradient_fraction = len(finite_gradients) / len(parameters)
            update_fraction = len(changed) / len(parameters)
            update_check(
                current_index,
                tuple(prediction.shape) == tuple(target.shape),
                f'actual={tuple(prediction.shape)} expected={tuple(target.shape)}',
            )
            current_index += 1
            update_check(
                current_index,
                bool(torch.isfinite(loss)),
                str(float(loss.detach())),
            )
            current_index += 1
            update_check(
                current_index,
                gradient_fraction >= 0.9,
                f'{len(finite_gradients)}/{len(parameters)}',
            )
            current_index += 1
            update_check(
                current_index,
                update_fraction >= 0.9,
                f'{len(changed)}/{len(parameters)}',
            )
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'artifacts:checkpoint_declarations',
        'artifacts:result_declarations',
        'artifacts:materialized_optional',
        'artifacts:materialized_if_present',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        source = _project_source(CANDIDATE_DIR, ('train.py', 'utils.py')).lower()
        output_dirs = [CANDIDATE_DIR / 'results' / 'ns_gfno2d_p4_seed1']
        output_dirs.extend(
            path for path in CANDIDATE_DIR.glob('results/*') if path.is_dir()
        )
        materialized = next((path for path in output_dirs if path.is_dir()), None)
        update_check(
            current_index,
            'best.pt' in source and 'last.pt' in source,
            'best.pt and last.pt',
        )
        current_index += 1
        ok = all(name in source for name in ('history.json', 'summary.json', 'train.log'))
        update_check(current_index, ok, 'history, summary, and log')
        current_index += 1
        if materialized is None:
            update_check(
                current_index,
                True,
                'full training artifacts are optional during code evaluation',
            )
            current_index += 1
            update_check(
                current_index,
                True,
                'not applicable because no results directory is materialized',
            )
            current_index += 1
        else:
            update_check(
                current_index,
                True,
                'not applicable because a results directory is materialized',
            )
            current_index += 1
            try:
                expected = [
                    materialized / name
                    for name in (
                        'best.pt',
                        'last.pt',
                        'history.json',
                        'summary.json',
                        'train.log',
                    )
                ]
                missing = [path.name for path in expected if not path.is_file()]
                for name in ('history.json', 'summary.json'):
                    json.loads(_read(materialized / name))
                empty = [path.name for path in expected if path.stat().st_size == 0]
                ok = not missing and not empty
                detail = f'directory={materialized} missing={missing} empty={empty}'
            except Exception as exc:
                ok = False
                detail = f'{type(exc).__name__}: {exc}'
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
