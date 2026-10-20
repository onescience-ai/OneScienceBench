'''V2 adapter for the legacy MMPINN 2D evaluator.'''
from pathlib import Path
import sys
import ast
import importlib.util
import inspect
import os
import traceback
import types
from importlib.machinery import ModuleSpec
from typing import Any, Dict, Sequence

# base_evaluator exposes shared checkpoint helpers but imports optional data
# readers that this evaluator does not use.
try:
    import h5py
except ModuleNotFoundError as exc:
    if exc.name != 'h5py':
        raise

    _h5py_stub = types.ModuleType('h5py')
    _h5py_stub.__spec__ = ModuleSpec('h5py', loader=None)

    def _missing_h5py(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'h5py'")

    _h5py_stub.File = _missing_h5py
    sys.modules['h5py'] = _h5py_stub

try:
    import pandas
except ModuleNotFoundError as exc:
    if exc.name != 'pandas':
        raise
    _pandas_stub = types.ModuleType('pandas')
    _pandas_stub.__spec__ = ModuleSpec('pandas', loader=None)
    sys.modules['pandas'] = _pandas_stub


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
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:model.py', 'file:train.py', 'file:config.yaml']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        REQUIRED_FILES = tuple(config.get('required_files', ('model.py', 'train.py', 'config.yaml')))
        for name in REQUIRED_FILES:
            candidate_path = (CANDIDATE_DIR / name).resolve()
            candidate_path.relative_to(CANDIDATE_DIR)
            update_check(current_index, candidate_path.is_file(), str(candidate_path))
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'data:no_external_data_reader',
        'data:script_generated_sampling',
        'data:exact_solution',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def _attribute_name(node: ast.AST) -> str:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                prefix = _attribute_name(node.value)
                return f'{prefix}.{node.attr}' if prefix else node.attr
            return ''

        def _contains_any(source: str, terms: Sequence[str]) -> bool:
            lowered = source.lower()
            return any(term.lower() in lowered for term in terms)

        source_file = config.get('source_file', 'train.py')
        source_path = (CANDIDATE_DIR / source_file).resolve()
        source_path.relative_to(CANDIDATE_DIR)
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(source_path))
        train_calls = frozenset(
            _attribute_name(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        )
        compact = ''.join(source.lower().split())
        forbidden_readers = {
            'np.load',
            'numpy.load',
            'sio.loadmat',
            'scipy.io.loadmat',
            'pd.read_csv',
            'pandas.read_csv',
            'h5py.File',
        }
        used_readers = sorted(train_calls.intersection(forbidden_readers))
        exact_formula = (
            'sin(' in compact
            and '20' in compact
            and 'pi' in compact
            and '1+9' in compact
            and ('x**2' in compact or 'x*x' in compact)
        )
        source_contract = {
            'no_external_data_reader': not used_readers,
            'script_generated_sampling': _contains_any(
                source, ('latin', 'lhs(', 'qmc', 'random.permutation', 'torch.rand')
            ),
            'exact_solution': exact_formula,
        }
        ok = source_contract['no_external_data_reader']
        update_check(current_index, ok, f'external_readers={used_readers}')
        current_index += 1
        ok = source_contract['script_generated_sampling']
        update_check(current_index, ok, 'required local sampling marker was not found')
        current_index += 1
        ok = source_contract['exact_solution']
        update_check(current_index, ok, f'exact_formula={exact_formula}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'train:entrypoint',
        'train:pde_autodiff',
        'train:multi_magnitude_loss',
        'train:adam_then_lbfgs',
        'train:lbfgs_segments',
        'train:config_driven',
        'train:live_loss_and_l2',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def _attribute_name(node: ast.AST) -> str:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                prefix = _attribute_name(node.value)
                return f'{prefix}.{node.attr}' if prefix else node.attr
            return ''

        def _contains_any(source: str, terms: Sequence[str]) -> bool:
            lowered = source.lower()
            return any(term.lower() in lowered for term in terms)

        def _as_number(value: Any) -> float | None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            text = str(value).strip()
            try:
                if '/' in text:
                    numerator, denominator = text.split('/', 1)
                    return float(numerator) / float(denominator)
                return float(text)
            except (TypeError, ValueError, ZeroDivisionError):
                return None

        def _numeric_values(value: Any) -> tuple[float, ...]:
            if isinstance(value, dict):
                children = value.values()
            elif isinstance(value, (list, tuple)):
                children = value
            else:
                number = _as_number(value)
                return () if number is None else (number,)
            return tuple(number for child in children for number in _numeric_values(child))

        import yaml

        EXPECTED_LBFGS_EXPONENTS = (1.0 / 3.0, 1.0 / 2.0, 1.0)
        source_file = config.get('source_file', 'train.py')
        config_file = config.get('config_file', 'config.yaml')
        source_path = (CANDIDATE_DIR / source_file).resolve()
        config_path = (CANDIDATE_DIR / config_file).resolve()
        source_path.relative_to(CANDIDATE_DIR)
        config_path.relative_to(CANDIDATE_DIR)
        source = source_path.read_text(encoding='utf-8')
        generated_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if generated_config is None:
            raise ValueError(f'YAML config must not be empty: {config_path}')
        tree = ast.parse(source, filename=str(source_path))
        train_functions = frozenset(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        lowered = source.lower()
        config_numbers = _numeric_values(generated_config)
        exponent_matches = tuple(
            any(abs(actual - expected) <= 1.0e-8 for actual in config_numbers)
            for expected in EXPECTED_LBFGS_EXPONENTS
        )
        one_third_in_source = _contains_any(source, ('1.0 / 3', '1 / 3', '1/3'))
        one_half_in_source = _contains_any(source, ('1.0 / 2', '1 / 2', '1/2'))
        source_contract = {
            'entrypoint': 'main' in train_functions and '__main__' in source,
            'pde_autodiff': 'autograd.grad' in lowered
            and 'u_t' in source
            and 'u_xx' in source
            and 'source' in lowered,
            'multi_magnitude_loss': _contains_any(source, ('torch.pow', '** p', '**p'))
            and (exponent_matches[0] or one_third_in_source)
            and all(term in lowered for term in ('ic', 'bc', 'res')),
            'adam_then_lbfgs': 'torch.optim.adam' in lowered
            and 'torch.optim.lbfgs' in lowered
            and lowered.find('torch.optim.adam') < lowered.rfind('torch.optim.lbfgs'),
            'lbfgs_segments': all(exponent_matches)
            or (one_third_in_source and one_half_in_source),
            'config_driven': 'config.yaml' in lowered
            and _contains_any(source, ('yaml', 'omegaconf', 'hydra')),
            'live_loss_and_l2': 'print(' in source and 'loss' in lowered and 'l2' in lowered,
        }
        message = (
            f'config_type={type(generated_config).__name__} '
            f'functions={sorted(train_functions)} exponent_matches={exponent_matches}'
        )
        ok = source_contract['entrypoint']
        update_check(current_index, ok, message)
        current_index += 1
        ok = source_contract['pde_autodiff']
        update_check(current_index, ok, message)
        current_index += 1
        ok = source_contract['multi_magnitude_loss']
        update_check(current_index, ok, message)
        current_index += 1
        ok = source_contract['adam_then_lbfgs']
        update_check(current_index, ok, message)
        current_index += 1
        ok = source_contract['lbfgs_segments']
        update_check(current_index, ok, message)
        current_index += 1
        ok = source_contract['config_driven']
        update_check(current_index, ok, message)
        current_index += 1
        ok = source_contract['live_loss_and_l2']
        update_check(current_index, ok, message)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'eval:checkpoint_restore',
        'eval:relative_l2',
        'eval:inference_mode',
        'eval:reports_or_saves',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def _contains_any(source: str, terms: Sequence[str]) -> bool:
            lowered = source.lower()
            return any(term.lower() in lowered for term in terms)

        EVAL_FILE_CANDIDATES = tuple(
            config.get('eval_file_candidates', ('evaluate.py', 'eval.py', 'inference.py'))
        )
        paths = [(CANDIDATE_DIR / name).resolve() for name in EVAL_FILE_CANDIDATES]
        for candidate_path in paths:
            candidate_path.relative_to(CANDIDATE_DIR)
        path = next((candidate_path for candidate_path in paths if candidate_path.is_file()), None)
        if path is None:
            message = 'no separate eval script required'
            update_check(current_index, True, message)
            current_index += 1
            update_check(current_index, True, message)
            current_index += 1
            update_check(current_index, True, message)
            current_index += 1
            update_check(current_index, True, message)
            current_index += 1
        else:
            source = path.read_text(encoding='utf-8')
            lowered = source.lower()
            requirements = {
                'checkpoint_restore': 'torch.load' in lowered and 'load_state_dict' in lowered,
                'relative_l2': 'l2' in lowered
                and _contains_any(source, ('linalg.norm', 'relative', 'rel_error')),
                'inference_mode': '.eval(' in source
                and _contains_any(source, ('no_grad', 'inference_mode')),
                'reports_or_saves': 'print(' in source
                or _contains_any(source, ('savemat', 'json.dump')),
            }
            message = f'script={path.name} requirements={requirements}'
            ok = requirements['checkpoint_restore']
            update_check(current_index, ok, message)
            current_index += 1
            ok = requirements['relative_l2']
            update_check(current_index, ok, message)
            current_index += 1
            ok = requirements['inference_mode']
            update_check(current_index, ok, message)
            current_index += 1
            ok = requirements['reports_or_saves']
            update_check(current_index, ok, message)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['model:interface', 'model:state_dict_contract', 'model:components']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        import torch.nn as nn

        def _load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f'Could not create import spec for {path}')
            directory = str(path.parent)
            while directory in sys.path:
                sys.path.remove(directory)
            sys.path.insert(0, directory)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            return module

        def _model_class(module: Any):
            model_class = getattr(module, MODEL_CLASS_NAME, None)
            if not inspect.isclass(model_class):
                raise TypeError(f'{MODEL_CLASS_NAME} is not a class in {module.__file__}')
            return model_class

        def _build_model(module: Any, layers: Sequence[int] | None = None):
            effective_layers = LAYERS if layers is None else layers
            return _model_class(module)(layers=list(effective_layers), lb=LB, ub=UB, sigma=SIGMA)

        def _model_pair(layers: Sequence[int] | None = None, align_state: bool = False):
            generated_model = _build_model(generated_module, layers).cpu()
            reference_model = _build_model(reference_module, layers).cpu()
            if align_state:
                generated_model.load_state_dict(reference_model.state_dict(), strict=True)
            return generated_model, reference_model

        MODEL_FILE = config.get('model_file', 'model.py')
        MODEL_CLASS_NAME = config.get('model_class', 'PINN2D')
        LAYERS = config.get('layers', [300, 300, 300, 300, 1])
        LB = config.get('lb', [-1.0, 0.0])
        UB = config.get('ub', [1.0, 1.0])
        SIGMA = config.get('sigma', 10.0)
        candidate_model_path = (CANDIDATE_DIR / MODEL_FILE).resolve()
        reference_model_path = (REFERENCE_DIR / MODEL_FILE).resolve()
        candidate_model_path.relative_to(CANDIDATE_DIR)
        reference_model_path.relative_to(REFERENCE_DIR)
        generated_module = _load_module_from_path('mmpinn_generated_model', candidate_model_path)
        reference_module = _load_module_from_path('mmpinn_reference_model', reference_model_path)
        generated_class = _model_class(generated_module)
        generated_model, reference_model = _model_pair()
        signature = inspect.signature(generated_class.__init__).parameters
        generated_state = generated_model.state_dict()
        reference_state = reference_model.state_dict()
        missing = sorted(set(reference_state) - set(generated_state))
        extra = sorted(set(generated_state) - set(reference_state))
        shape_mismatches = {
            name: {
                'expected': tuple(reference_state[name].shape),
                'actual': tuple(generated_state[name].shape),
            }
            for name in set(reference_state).intersection(generated_state)
            if generated_state[name].shape != reference_state[name].shape
        }
        buffers = dict(generated_model.named_buffers())
        buffer_shapes = {name: tuple(value.shape) for name, value in buffers.items()}
        parameters = dict(generated_model.named_parameters())
        lb = buffers.get('lb')
        ub = buffers.get('ub')
        domain_ok = (
            lb is not None
            and ub is not None
            and torch.allclose(lb.flatten().cpu(), torch.tensor(LB))
            and torch.allclose(ub.flatten().cpu(), torch.tensor(UB))
        )
        interface_ok = issubclass(generated_class, nn.Module) and {
            'layers', 'lb', 'ub', 'sigma'
        } <= set(signature)
        components_ok = (
            isinstance(getattr(generated_model, 'weights', None), nn.ParameterList)
            and isinstance(getattr(generated_model, 'biases', None), nn.ParameterList)
            and all(name.startswith(('weights.', 'biases.')) for name in parameters)
            and {'lb', 'ub', 'W'} <= set(buffers)
            and domain_ok
        )
        ok = interface_ok
        update_check(current_index, ok, f'class={generated_class} parameters={list(signature)}')
        current_index += 1
        ok = not missing and not extra and not shape_mismatches
        update_check(current_index, ok, f'missing={missing} extra={extra} shape_mismatches={shape_mismatches}')
        current_index += 1
        ok = components_ok
        update_check(
            current_index,
            ok,
            f'parameters={list(parameters)} buffers={buffer_shapes} '
            f'lb={lb.flatten().tolist() if lb is not None else None} '
            f'ub={ub.flatten().tolist() if ub is not None else None}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['forward:contract']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        def _load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f'Could not create import spec for {path}')
            directory = str(path.parent)
            while directory in sys.path:
                sys.path.remove(directory)
            sys.path.insert(0, directory)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            return module

        MODEL_FILE = config.get('model_file', 'model.py')
        MODEL_CLASS_NAME = config.get('model_class', 'PINN2D')
        LAYERS = config.get('layers', [300, 300, 300, 300, 1])
        LB = config.get('lb', [-1.0, 0.0])
        UB = config.get('ub', [1.0, 1.0])
        SIGMA = config.get('sigma', 10.0)
        CHECK_NODES = config.get('check_nodes', 64)
        candidate_model_path = (CANDIDATE_DIR / MODEL_FILE).resolve()
        candidate_model_path.relative_to(CANDIDATE_DIR)
        generated_module = _load_module_from_path('mmpinn_generated_model', candidate_model_path)
        model_class = getattr(generated_module, MODEL_CLASS_NAME, None)
        if not inspect.isclass(model_class):
            raise TypeError(f'{MODEL_CLASS_NAME} is not a class in {candidate_model_path.parent}')
        model = model_class(layers=list(LAYERS), lb=LB, ub=UB, sigma=SIGMA).cpu().eval()
        x = torch.linspace(LB[0], UB[0], CHECK_NODES)
        t = torch.linspace(UB[1], LB[1], CHECK_NODES)
        inputs = torch.stack((x, t), dim=1)
        with torch.no_grad():
            output = model(inputs)
        is_tensor = isinstance(output, torch.Tensor)
        shape = tuple(output.shape) if is_tensor else None
        finite = bool(torch.isfinite(output).all()) if is_tensor else False
        ok = is_tensor and shape == (CHECK_NODES, 1) and finite
        update_check(current_index, ok, f'type={type(output).__name__} shape={shape} finite={finite}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['equivalence:values']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        def _load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f'Could not create import spec for {path}')
            directory = str(path.parent)
            while directory in sys.path:
                sys.path.remove(directory)
            sys.path.insert(0, directory)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            return module

        MODEL_FILE = config.get('model_file', 'model.py')
        MODEL_CLASS_NAME = config.get('model_class', 'PINN2D')
        LAYERS = config.get('layers', [300, 300, 300, 300, 1])
        LB = config.get('lb', [-1.0, 0.0])
        UB = config.get('ub', [1.0, 1.0])
        SIGMA = config.get('sigma', 10.0)
        SEED = config.get('seed', 20260817)
        CHECK_NODES = config.get('check_nodes', 64)
        FORWARD_ATOL = config.get('forward_atol', 1.0e-6)
        FORWARD_RTOL = config.get('forward_rtol', 1.0e-6)
        candidate_model_path = (CANDIDATE_DIR / MODEL_FILE).resolve()
        reference_model_path = (REFERENCE_DIR / MODEL_FILE).resolve()
        candidate_model_path.relative_to(CANDIDATE_DIR)
        reference_model_path.relative_to(REFERENCE_DIR)
        generated_module = _load_module_from_path('mmpinn_generated_model', candidate_model_path)
        reference_module = _load_module_from_path('mmpinn_reference_model', reference_model_path)
        generated_class = getattr(generated_module, MODEL_CLASS_NAME, None)
        reference_class = getattr(reference_module, MODEL_CLASS_NAME, None)
        if not inspect.isclass(generated_class):
            raise TypeError(f'{MODEL_CLASS_NAME} is not a class in {candidate_model_path.parent}')
        if not inspect.isclass(reference_class):
            raise TypeError(f'{MODEL_CLASS_NAME} is not a class in {reference_model_path.parent}')
        torch.manual_seed(SEED)
        generated_model = generated_class(layers=list(LAYERS), lb=LB, ub=UB, sigma=SIGMA).cpu()
        reference_model = reference_class(layers=list(LAYERS), lb=LB, ub=UB, sigma=SIGMA).cpu()
        generated_model.load_state_dict(reference_model.state_dict(), strict=True)
        generated_model.eval()
        reference_model.eval()
        x = torch.linspace(LB[0], UB[0], CHECK_NODES)
        t = torch.linspace(UB[1], LB[1], CHECK_NODES)
        inputs = torch.stack((x, t), dim=1)
        with torch.no_grad():
            reference_output = reference_model(inputs)
            generated_output = generated_model(inputs)
        same_shape = (
            isinstance(reference_output, torch.Tensor)
            and isinstance(generated_output, torch.Tensor)
            and reference_output.shape == generated_output.shape
        )
        max_abs_diff = (
            float((generated_output - reference_output).abs().max())
            if same_shape
            else float('inf')
        )
        equivalent = same_shape and torch.allclose(
            generated_output,
            reference_output,
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
        )
        update_check(
            current_index,
            equivalent,
            f'same_shape={same_shape} max_abs_diff={max_abs_diff:.3e} '
            f'atol={FORWARD_ATOL:.0e} rtol={FORWARD_RTOL:.0e}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['training:optimization_step', 'training:reference_equivalence']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        def _load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f'Could not create import spec for {path}')
            directory = str(path.parent)
            while directory in sys.path:
                sys.path.remove(directory)
            sys.path.insert(0, directory)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            return module

        def _comparison_inputs():
            x = torch.linspace(LB[0], UB[0], CHECK_NODES)
            t = torch.linspace(UB[1], LB[1], CHECK_NODES)
            return torch.stack((x, t), dim=1)

        def _trainable_parameters(model: Any) -> dict[str, Any]:
            return {
                name: parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

        def _finite_gradients(parameters: dict[str, Any]) -> dict[str, Any]:
            return {
                name: parameter.grad.detach().clone()
                for name, parameter in parameters.items()
                if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            }

        def _physics_loss(model: Any):
            coordinates = _comparison_inputs()[:8].clone().requires_grad_(True)
            output = model(coordinates)
            first = torch.autograd.grad(
                output,
                coordinates,
                grad_outputs=torch.ones_like(output),
                create_graph=True,
            )[0]
            u_x = first[:, 0:1]
            u_t = first[:, 1:2]
            u_xx = torch.autograd.grad(
                u_x,
                coordinates,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
            )[0][:, 0:1]
            residual = torch.mean((u_t - u_xx) ** 2)
            x = coordinates[:, 0:1].detach()
            zeros = torch.zeros_like(x)
            initial = torch.mean(model(torch.cat((x, zeros), dim=1)) ** 2)
            t = coordinates[:, 1:2].detach()
            target = torch.sin(2.0 * torch.pi * t)
            lower = model(torch.cat((torch.full_like(t, LB[0]), t), dim=1))
            upper = model(torch.cat((torch.full_like(t, UB[0]), t), dim=1))
            boundary = torch.mean((lower - target) ** 2) + torch.mean((upper - target) ** 2)
            epsilon = torch.finfo(output.dtype).eps
            return torch.pow(initial + boundary + epsilon, 1.0 / 3.0) + torch.pow(
                residual + epsilon, 1.0 / 3.0
            )

        MODEL_FILE = config.get('model_file', 'model.py')
        MODEL_CLASS_NAME = config.get('model_class', 'PINN2D')
        LB = config.get('lb', [-1.0, 0.0])
        UB = config.get('ub', [1.0, 1.0])
        SIGMA = config.get('sigma', 10.0)
        SEED = config.get('seed', 20260817)
        CHECK_NODES = config.get('check_nodes', 64)
        FORWARD_ATOL = config.get('forward_atol', 1.0e-6)
        small_layers = config.get('small_layers', [32, 32, 1])
        candidate_model_path = (CANDIDATE_DIR / MODEL_FILE).resolve()
        reference_model_path = (REFERENCE_DIR / MODEL_FILE).resolve()
        candidate_model_path.relative_to(CANDIDATE_DIR)
        reference_model_path.relative_to(REFERENCE_DIR)
        generated_module = _load_module_from_path('mmpinn_generated_model', candidate_model_path)
        reference_module = _load_module_from_path('mmpinn_reference_model', reference_model_path)
        generated_class = getattr(generated_module, MODEL_CLASS_NAME, None)
        reference_class = getattr(reference_module, MODEL_CLASS_NAME, None)
        if not inspect.isclass(generated_class):
            raise TypeError(f'{MODEL_CLASS_NAME} is not a class in {candidate_model_path.parent}')
        if not inspect.isclass(reference_class):
            raise TypeError(f'{MODEL_CLASS_NAME} is not a class in {reference_model_path.parent}')
        torch.manual_seed(SEED)
        generated_model = generated_class(layers=list(small_layers), lb=LB, ub=UB, sigma=SIGMA).cpu()
        reference_model = reference_class(layers=list(small_layers), lb=LB, ub=UB, sigma=SIGMA).cpu()
        generated_model.load_state_dict(reference_model.state_dict(), strict=True)
        generated_model.train()
        reference_model.train()
        optimizer = torch.optim.Adam(generated_model.parameters(), lr=1.0e-3)
        trainable = _trainable_parameters(generated_model)
        reference_trainable = _trainable_parameters(reference_model)
        before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
        optimizer.zero_grad()
        generated_loss = _physics_loss(generated_model)
        generated_loss.backward()
        gradients = _finite_gradients(trainable)
        reference_model.zero_grad()
        reference_loss = _physics_loss(reference_model)
        reference_loss.backward()
        reference_gradients = _finite_gradients(reference_trainable)
        common_gradients = set(gradients).intersection(reference_gradients)
        max_gradient_diff = max(
            (
                float((gradients[name] - reference_gradients[name]).abs().max())
                for name in common_gradients
            ),
            default=float('inf'),
        )
        optimizer.step()
        changed = {
            name
            for name, parameter in trainable.items()
            if not torch.equal(before[name], parameter.detach())
        }
        loss_diff = float((generated_loss.detach() - reference_loss.detach()).abs())
        missing_gradients = sorted(set(trainable) - set(gradients))
        unchanged = sorted(set(trainable) - changed)
        ok = bool(torch.isfinite(generated_loss)) and not missing_gradients and not unchanged
        update_check(
            current_index,
            ok,
            f'loss={float(generated_loss.detach()):.6g} '
            f'missing_or_nonfinite={missing_gradients} unchanged={unchanged}',
        )
        current_index += 1
        ok = (
            loss_diff <= FORWARD_ATOL
            and set(gradients) == set(reference_gradients)
            and max_gradient_diff <= FORWARD_ATOL
        )
        update_check(
            current_index,
            ok,
            f'loss_diff={loss_diff:.3e} max_gradient_diff={max_gradient_diff:.3e}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'artifact:checkpoint',
        'artifact:mat_result',
        'artifact:output_directory',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def _attribute_name(node: ast.AST) -> str:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                prefix = _attribute_name(node.value)
                return f'{prefix}.{node.attr}' if prefix else node.attr
            return ''

        def _contains_any(source: str, terms: Sequence[str]) -> bool:
            lowered = source.lower()
            return any(term.lower() in lowered for term in terms)

        source_file = config.get('source_file', 'train.py')
        source_path = (CANDIDATE_DIR / source_file).resolve()
        source_path.relative_to(CANDIDATE_DIR)
        source = source_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(source_path))
        train_calls = frozenset(
            _attribute_name(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        )
        directory_calls = sorted(
            name
            for name in train_calls
            if name in {'os.makedirs', 'Path.mkdir'} or name.endswith('.mkdir')
        )
        requirements = {
            'checkpoint': 'model_2d.pth' in source
            and _contains_any(source, ('torch.save', 'save_state_dict')),
            'mat_result': '1.mat' in source
            and _contains_any(source, ('savemat', 'scipy.io.savemat'))
            and _contains_any(source, ("{'u'", '"u"', "'u'")),
            'output_directory': bool(directory_calls),
        }
        ok = requirements['checkpoint']
        update_check(current_index, ok, f'directory_calls={directory_calls}')
        current_index += 1
        ok = requirements['mat_result']
        update_check(current_index, ok, f'directory_calls={directory_calls}')
        current_index += 1
        ok = requirements['output_directory']
        update_check(current_index, ok, f'directory_calls={directory_calls}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
