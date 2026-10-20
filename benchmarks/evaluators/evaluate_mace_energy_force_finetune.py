'''V2 adapter for the legacy MACE energy-force fine-tuning evaluator.'''
from pathlib import Path
import sys
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import tempfile
import traceback
import types
from typing import Any, Dict

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
    CHECKPOINT_NAMES = [
        'file:mace_energy_force_finetune.py',
        'file:train_mace_finetune.py',
        'file:eval_mace_finetune.py',
        'file:mace_finetune.yaml',
        'file:portable_mace/__init__.py',
        'static:no_forbidden_imports',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        REQUIRED_FILES = (
            'mace_energy_force_finetune.py',
            'train_mace_finetune.py',
            'eval_mace_finetune.py',
            'mace_finetune.yaml',
            'portable_mace/__init__.py',
        )
        submission_dir = CANDIDATE_DIR
        for rel in REQUIRED_FILES:
            path = submission_dir / rel
            update_check(current_index, path.is_file(), str(path))
            current_index += 1

        invalid: list[str] = []
        for path in sorted(submission_dir.rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            imports = []
            importlib_aliases = {'importlib'}
            import_module_aliases: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                    for alias in node.names:
                        if alias.name == 'importlib':
                            importlib_aliases.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    imports.append(node.module)
                    if node.module == 'importlib':
                        import_module_aliases.update(
                            alias.asname or alias.name
                            for alias in node.names
                            if alias.name == 'import_module'
                        )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                dynamic_name = None
                if isinstance(node.func, ast.Name):
                    if node.func.id == '__import__':
                        dynamic_name = '__import__'
                    elif node.func.id in import_module_aliases:
                        dynamic_name = node.func.id
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'import_module'
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in importlib_aliases
                ):
                    dynamic_name = f'{node.func.value.id}.import_module'
                if dynamic_name is not None:
                    invalid.append(
                        f'{path.relative_to(submission_dir)}:dynamic_import:{dynamic_name}'
                    )
            forbidden = (
                'onescience',
                'mace',
                'evaluation.eval_datasets.codegen.standard',
                'evaluation.eval_datasets.codegen.evaluators',
            )
            for name in imports:
                if any(name == item or name.startswith(item + '.') for item in forbidden):
                    invalid.append(f'{path.relative_to(submission_dir)}:{name}')
        update_check(current_index, not invalid, f'imports={invalid}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_file(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'foundation:paths_resolved',
        'foundation:config_path',
        'foundation:weights_path',
        'foundation:config_exists',
        'foundation:weights_exists',
        'foundation:format',
        'foundation:source_fingerprint',
        'foundation:weights_fingerprint',
        'foundation:state_entry_count',
        'foundation:tensors_only',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        REQUIRED_MODEL_CLASS = 'ScaleShiftMACE'
        REQUIRED_TUNER_CLASS = 'MACEEnergyForceFineTuner'
        DEFAULT_PORTABLE_CONFIG_PATH = Path('/public/share/sugonhpcapp01/onestore/onemodels/mace/mace-mp-0-medium-finetune-portable.json')
        DEFAULT_PORTABLE_WEIGHTS_PATH = Path('/public/share/sugonhpcapp01/onestore/onemodels/mace/mace-mp-0-medium-finetune-portable.pt')
        EXPECTED_SOURCE_SHA256 = '81b87e94b1a2e5faf6a683af75bedaafa168a69de7099d2f54f616fc8d08af26'
        EXPECTED_WEIGHTS_SHA256 = 'dfb8e7cf1c4de2b8faaeef97bcb50514bb9950d61970c75af556cdb964a9d1a9'
        EXPECTED_STATE_ENTRY_COUNT = 79
        SEED = 20260804

        def portable_config_path() -> Path:
            return Path(os.environ.get('MACE_MP0_PORTABLE_CONFIG_PATH', str(DEFAULT_PORTABLE_CONFIG_PATH))).expanduser().resolve()

        def portable_weights_path() -> Path:
            return Path(os.environ.get('MACE_MP0_PORTABLE_WEIGHTS_PATH', str(DEFAULT_PORTABLE_WEIGHTS_PATH))).expanduser().resolve()

        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            return digest.hexdigest()

        def load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f'Could not create import spec for {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def purge_task_modules() -> None:
            roots = ('portable_mace', 'mace_energy_force_finetune', 'train_mace_finetune', 'eval_mace_finetune')
            for name in list(sys.modules):
                if any(name == root or name.startswith(root + '.') for root in roots):
                    del sys.modules[name]

        def load_task_modules(directory: Path, prefix: str):
            purge_task_modules()
            module_dir = str(directory)
            if module_dir in sys.path:
                sys.path.remove(module_dir)
            sys.path.insert(0, module_dir)
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + '.'):
                    del sys.modules[name]
            package = types.ModuleType(prefix)
            package.__file__ = str(directory / '__init__.py')
            package.__package__ = prefix
            package.__path__ = [module_dir]
            sys.modules[prefix] = package
            model_module = load_module_from_path(f'{prefix}.mace_energy_force_finetune', directory / 'mace_energy_force_finetune.py')
            sys.modules['mace_energy_force_finetune'] = model_module
            train_module = load_module_from_path(f'{prefix}.train_mace_finetune', directory / 'train_mace_finetune.py')
            sys.modules['train_mace_finetune'] = train_module
            eval_module = load_module_from_path(f'{prefix}.eval_mace_finetune', directory / 'eval_mace_finetune.py')
            return model_module, train_module, eval_module

        import torch
        torch.manual_seed(SEED)
        candidate_model_module, candidate_train, candidate_eval = load_task_modules(CANDIDATE_DIR, 'mace_real_finetune_submission')
        model_cls = getattr(candidate_model_module, REQUIRED_MODEL_CLASS, None)
        tuner_cls = getattr(candidate_model_module, REQUIRED_TUNER_CLASS, None)
        if not inspect.isclass(model_cls):
            raise TypeError(f'{REQUIRED_MODEL_CLASS} is not a class')
        if not inspect.isclass(tuner_cls):
            raise TypeError(f'{REQUIRED_TUNER_CLASS} is not a class')
        candidate_config = candidate_train.load_config(CANDIDATE_DIR / 'mace_finetune.yaml')
        if not isinstance(candidate_config, dict):
            raise TypeError('Submission load_config must return a dict-like config.')

        resolver = getattr(candidate_train, 'foundation_paths', None)
        if not callable(resolver):
            update_check(current_index, False, 'foundation_paths is not callable')
            current_index += 1
        else:
            try:
                resolved = resolver(candidate_config)
                if not isinstance(resolved, (tuple, list)) or len(resolved) != 2:
                    raise TypeError('foundation_paths must return (config_path, weights_path)')
                config_path = Path(resolved[0]).expanduser().resolve()
                weights_path = Path(resolved[1]).expanduser().resolve()
            except Exception as exc:
                update_check(current_index, False, f'{type(exc).__name__}: {exc}')
                current_index += 1
            else:
                expected_config = portable_config_path()
                expected_weights = portable_weights_path()
                update_check(current_index, True, 'foundation_paths(config)')
                current_index += 1
                update_check(current_index, config_path == expected_config, f'configured={config_path} expected={expected_config}')
                current_index += 1
                update_check(current_index, weights_path == expected_weights, f'configured={weights_path} expected={expected_weights}')
                current_index += 1
                update_check(current_index, config_path.is_file(), str(config_path))
                current_index += 1
                update_check(current_index, weights_path.is_file(), str(weights_path))
                current_index += 1
                if config_path.is_file() and weights_path.is_file():
                    document = json.loads(config_path.read_text(encoding='utf-8'))
                    payload = torch.load(weights_path, map_location='cpu', weights_only=True)
                    update_check(current_index, document.get('format') == 'mace-portable-state-dict-v1' and payload.get('format') == 'mace-portable-state-dict-v1', f"config={document.get('format')} weights={payload.get('format')}")
                    current_index += 1
                    update_check(current_index, document.get('source_sha256') == EXPECTED_SOURCE_SHA256 and payload.get('source_sha256') == EXPECTED_SOURCE_SHA256, f"source_sha256={payload.get('source_sha256')}")
                    current_index += 1
                    update_check(current_index, sha256(weights_path) == EXPECTED_WEIGHTS_SHA256, f'sha256={sha256(weights_path)}')
                    current_index += 1
                    update_check(current_index, len(payload['state_dict']) == EXPECTED_STATE_ENTRY_COUNT, f"count={len(payload['state_dict'])}")
                    current_index += 1
                    update_check(current_index, all(isinstance(value, torch.Tensor) for value in payload['state_dict'].values()), 'weights_only=True payload')
                    current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'data:batch_size',
        'data:positions_shape',
        'data:node_attrs_shape',
        'data:edge_index_shape',
        'data:finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        INPUT_BATCH_SIZE = 10
        SEED = 20260804

        class ReferenceBatch:
            def __init__(self, **values):
                self._values = values
                for name, value in values.items():
                    setattr(self, name, value)

            def __getitem__(self, name: str):
                return self._values[name]

            def to(self, device):
                return ReferenceBatch(**{name: value.to(device) for name, value in self._values.items()})

            def to_dict(self):
                return dict(self._values)

            def clone(self):
                return ReferenceBatch(**{name: value.detach().clone() for name, value in self._values.items()})

            def replace(self, **updates):
                values = dict(self._values)
                values.update(updates)
                return ReferenceBatch(**values)

        import torch
        torch.manual_seed(SEED)
        dtype = torch.float64
        base_positions = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]], dtype=dtype)
        positions = base_positions.repeat(INPUT_BATCH_SIZE, 1)
        node_attrs = torch.zeros(INPUT_BATCH_SIZE * 3, 89, dtype=dtype)
        element_indices = torch.tensor([7, 0, 0], dtype=torch.long).repeat(INPUT_BATCH_SIZE)
        node_attrs[torch.arange(INPUT_BATCH_SIZE * 3), element_indices] = 1.0
        edges = []
        for graph in range(INPUT_BATCH_SIZE):
            offset = graph * 3
            edges.extend((offset + sender, offset + receiver) for sender in range(3) for receiver in range(3) if sender != receiver)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        batch = torch.arange(INPUT_BATCH_SIZE, dtype=torch.long).repeat_interleave(3)
        ptr = torch.arange(0, 3 * (INPUT_BATCH_SIZE + 1), 3, dtype=torch.long)
        reference_batch = ReferenceBatch(positions=positions, node_attrs=node_attrs, edge_index=edge_index, shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), unit_shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), batch=batch, ptr=ptr, cell=torch.eye(3, dtype=dtype).repeat(INPUT_BATCH_SIZE, 1, 1) * 20.0, head=torch.zeros(INPUT_BATCH_SIZE, dtype=torch.long), energy=torch.zeros(INPUT_BATCH_SIZE, dtype=dtype), forces=torch.zeros(INPUT_BATCH_SIZE * 3, 3, dtype=dtype), weight=torch.linspace(0.8, 1.2, INPUT_BATCH_SIZE, dtype=dtype), energy_weight=torch.linspace(1.0, 1.4, INPUT_BATCH_SIZE, dtype=dtype), forces_weight=torch.linspace(0.7, 1.3, INPUT_BATCH_SIZE, dtype=dtype))
        data = reference_batch.to_dict()
        floating_values = [value for value in data.values() if isinstance(value, torch.Tensor) and value.is_floating_point()]
        update_check(current_index, data['ptr'].numel() - 1 == INPUT_BATCH_SIZE, f"graphs={data['ptr'].numel() - 1}")
        current_index += 1
        update_check(current_index, data['positions'].shape == (30, 3), f"shape={tuple(data['positions'].shape)}")
        current_index += 1
        update_check(current_index, data['node_attrs'].shape == (30, 89), f"shape={tuple(data['node_attrs'].shape)}")
        current_index += 1
        update_check(current_index, data['edge_index'].shape == (2, 60), f"shape={tuple(data['edge_index'].shape)}")
        current_index += 1
        update_check(current_index, all(bool(torch.isfinite(value).all()) for value in floating_values), 'all floating tensors finite')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'train:has_load_config',
        'train:has_foundation_paths',
        'train:has_build_model',
        'train:has_build_trainer',
        'train:has_train',
        'train:train_signature',
        'train:uses_adamw',
        'train:build_model_returns_module',
        'eval:has_checkpoint_path',
        'eval:checkpoint_path_signature',
        'eval:has_load_model_state',
        'eval:load_model_state_signature',
        'eval:has_evaluate',
        'eval:evaluate_signature',
        'eval:strict_state_loading',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f'Could not create import spec for {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def purge_task_modules() -> None:
            roots = ('portable_mace', 'mace_energy_force_finetune', 'train_mace_finetune', 'eval_mace_finetune')
            for name in list(sys.modules):
                if any(name == root or name.startswith(root + '.') for root in roots):
                    del sys.modules[name]

        def load_task_modules(directory: Path, prefix: str):
            purge_task_modules()
            module_dir = str(directory)
            if module_dir in sys.path:
                sys.path.remove(module_dir)
            sys.path.insert(0, module_dir)
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + '.'):
                    del sys.modules[name]
            package = types.ModuleType(prefix)
            package.__file__ = str(directory / '__init__.py')
            package.__package__ = prefix
            package.__path__ = [module_dir]
            sys.modules[prefix] = package
            model_module = load_module_from_path(f'{prefix}.mace_energy_force_finetune', directory / 'mace_energy_force_finetune.py')
            sys.modules['mace_energy_force_finetune'] = model_module
            train_module = load_module_from_path(f'{prefix}.train_mace_finetune', directory / 'train_mace_finetune.py')
            sys.modules['train_mace_finetune'] = train_module
            eval_module = load_module_from_path(f'{prefix}.eval_mace_finetune', directory / 'eval_mace_finetune.py')
            return model_module, train_module, eval_module

        import torch.nn as nn
        candidate_model_module, candidate_train, candidate_eval = load_task_modules(CANDIDATE_DIR, 'mace_real_finetune_submission')
        candidate_config = candidate_train.load_config(CANDIDATE_DIR / 'mace_finetune.yaml')
        if not isinstance(candidate_config, dict):
            raise TypeError('Submission load_config must return a dict-like config.')
        model = candidate_train.build_model(candidate_config)
        train_source = Path(candidate_train.__file__).read_text(encoding='utf-8')
        train_function = getattr(candidate_train, 'train', None)
        train_signature = inspect.signature(train_function) if callable(train_function) else None
        implementation_source = '\n'.join(path.read_text(encoding='utf-8') for path in sorted(CANDIDATE_DIR.rglob('*.py')))
        update_check(current_index, callable(getattr(candidate_train, 'load_config', None)), 'load_config')
        current_index += 1
        update_check(current_index, callable(getattr(candidate_train, 'foundation_paths', None)), 'foundation_paths')
        current_index += 1
        update_check(current_index, callable(getattr(candidate_train, 'build_model', None)), 'build_model')
        current_index += 1
        update_check(current_index, callable(getattr(candidate_train, 'build_trainer', None)), 'build_trainer')
        current_index += 1
        update_check(current_index, callable(getattr(candidate_train, 'train', None)), 'train')
        current_index += 1
        update_check(current_index, train_signature is not None and all(name in train_signature.parameters for name in ('config', 'train_batches', 'output_dir')), str(train_signature))
        current_index += 1
        update_check(current_index, 'AdamW' in implementation_source and 'MACEEnergyForceFineTuner' in train_source, 'AdamW via tuner')
        current_index += 1
        update_check(current_index, isinstance(model, nn.Module), str(type(model)))
        current_index += 1

        source = Path(candidate_eval.__file__).read_text(encoding='utf-8')
        checkpoint_path = getattr(candidate_eval, 'checkpoint_path_from_args', None)
        load_model_state = getattr(candidate_eval, 'load_model_state', None)
        evaluate_function = getattr(candidate_eval, 'evaluate', None)
        checkpoint_signature = inspect.signature(checkpoint_path) if callable(checkpoint_path) else None
        load_signature = inspect.signature(load_model_state) if callable(load_model_state) else None
        evaluate_signature = inspect.signature(evaluate_function) if callable(evaluate_function) else None
        update_check(current_index, callable(checkpoint_path), 'checkpoint_path_from_args')
        current_index += 1
        update_check(current_index, checkpoint_signature is not None and all(name in checkpoint_signature.parameters for name in ('config', 'override')), str(checkpoint_signature))
        current_index += 1
        update_check(current_index, callable(load_model_state), 'load_model_state')
        current_index += 1
        update_check(current_index, load_signature is not None and all(name in load_signature.parameters for name in ('config', 'checkpoint_path')), str(load_signature))
        current_index += 1
        update_check(current_index, callable(evaluate_function), 'evaluate')
        current_index += 1
        update_check(current_index, evaluate_signature is not None and all(name in evaluate_signature.parameters for name in ('model', 'batches')), str(evaluate_signature))
        current_index += 1
        update_check(current_index, 'strict=True' in source, 'strict=True')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'model:is_nn_module',
        'model:is_scale_shift_mace',
        'model:parameter_count',
        'model:state_entry_count',
        'model:float64_weights',
        'model:element_count',
        'model:r_max',
        'model:interaction_count',
        'model:readout_count',
        'train:tuner_method_contract',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        REQUIRED_MODEL_CLASS = 'ScaleShiftMACE'
        REQUIRED_TUNER_CLASS = 'MACEEnergyForceFineTuner'
        EXPECTED_PARAMETER_COUNT = 4_688_656
        EXPECTED_STATE_ENTRY_COUNT = 79
        SEED = 20260804

        def load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f'Could not create import spec for {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def purge_task_modules() -> None:
            roots = ('portable_mace', 'mace_energy_force_finetune', 'train_mace_finetune', 'eval_mace_finetune')
            for name in list(sys.modules):
                if any(name == root or name.startswith(root + '.') for root in roots):
                    del sys.modules[name]

        def load_task_modules(directory: Path, prefix: str):
            purge_task_modules()
            module_dir = str(directory)
            if module_dir in sys.path:
                sys.path.remove(module_dir)
            sys.path.insert(0, module_dir)
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + '.'):
                    del sys.modules[name]
            package = types.ModuleType(prefix)
            package.__file__ = str(directory / '__init__.py')
            package.__package__ = prefix
            package.__path__ = [module_dir]
            sys.modules[prefix] = package
            model_module = load_module_from_path(f'{prefix}.mace_energy_force_finetune', directory / 'mace_energy_force_finetune.py')
            sys.modules['mace_energy_force_finetune'] = model_module
            train_module = load_module_from_path(f'{prefix}.train_mace_finetune', directory / 'train_mace_finetune.py')
            sys.modules['train_mace_finetune'] = train_module
            eval_module = load_module_from_path(f'{prefix}.eval_mace_finetune', directory / 'eval_mace_finetune.py')
            return model_module, train_module, eval_module

        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        candidate_model_module, candidate_train, candidate_eval = load_task_modules(CANDIDATE_DIR, 'mace_real_finetune_submission')
        model_cls = getattr(candidate_model_module, REQUIRED_MODEL_CLASS, None)
        tuner_cls = getattr(candidate_model_module, REQUIRED_TUNER_CLASS, None)
        if not inspect.isclass(model_cls):
            raise TypeError(f'{REQUIRED_MODEL_CLASS} is not a class')
        if not inspect.isclass(tuner_cls):
            raise TypeError(f'{REQUIRED_TUNER_CLASS} is not a class')
        candidate_config = candidate_train.load_config(CANDIDATE_DIR / 'mace_finetune.yaml')
        if not isinstance(candidate_config, dict):
            raise TypeError('Submission load_config must return a dict-like config.')
        model = candidate_train.build_model(candidate_config)
        update_check(current_index, isinstance(model, nn.Module), str(type(model)))
        current_index += 1
        update_check(current_index, isinstance(model, model_cls), str(type(model)))
        current_index += 1
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        update_check(current_index, parameter_count == EXPECTED_PARAMETER_COUNT, f'count={parameter_count}')
        current_index += 1
        state_entry_count = len(model.state_dict())
        update_check(current_index, state_entry_count == EXPECTED_STATE_ENTRY_COUNT, f'count={state_entry_count}')
        current_index += 1
        update_check(current_index, all(parameter.dtype == torch.float64 for parameter in model.parameters() if parameter.is_floating_point()), 'all floating parameters are float64')
        current_index += 1
        update_check(current_index, int(model.atomic_numbers.numel()) == 89, f'count={model.atomic_numbers.numel()}')
        current_index += 1
        update_check(current_index, math.isclose(float(model.r_max), 6.0, rel_tol=0.0, abs_tol=1e-12), f'r_max={float(model.r_max)}')
        current_index += 1
        update_check(current_index, len(model.interactions) == 2 and len(model.products) == 2, f'interactions={len(model.interactions)} products={len(model.products)}')
        current_index += 1
        update_check(current_index, len(model.readouts) == 2, f'readouts={len(model.readouts)}')
        current_index += 1
        update_check(current_index, all(callable(getattr(tuner_cls, name, None)) for name in ('train_step', 'fit', 'save_checkpoint')), 'train_step, fit, save_checkpoint')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'forward:returns_mapping',
        'forward:energy_shape', 'forward:energy_finite',
        'forward:node_energy_shape', 'forward:node_energy_finite',
        'forward:interaction_energy_shape', 'forward:interaction_energy_finite',
        'forward:forces_shape', 'forward:forces_finite',
        'forward:node_feats_shape', 'forward:node_feats_finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        INPUT_BATCH_SIZE = 10
        SEED = 20260804

        class ReferenceBatch:
            def __init__(self, **values):
                self._values = values
                for name, value in values.items():
                    setattr(self, name, value)
            def to_dict(self): return dict(self._values)
            def clone(self): return ReferenceBatch(**{name: value.detach().clone() for name, value in self._values.items()})

        def load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None: raise ImportError(f'Could not create import spec for {path}')
            module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
        def purge_task_modules() -> None:
            roots = ('portable_mace', 'mace_energy_force_finetune', 'train_mace_finetune', 'eval_mace_finetune')
            for name in list(sys.modules):
                if any(name == root or name.startswith(root + '.') for root in roots): del sys.modules[name]
        def load_task_modules(directory: Path, prefix: str):
            purge_task_modules(); module_dir = str(directory)
            if module_dir in sys.path: sys.path.remove(module_dir)
            sys.path.insert(0, module_dir)
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + '.'): del sys.modules[name]
            package = types.ModuleType(prefix); package.__file__ = str(directory / '__init__.py'); package.__package__ = prefix; package.__path__ = [module_dir]; sys.modules[prefix] = package
            model_module = load_module_from_path(f'{prefix}.mace_energy_force_finetune', directory / 'mace_energy_force_finetune.py'); sys.modules['mace_energy_force_finetune'] = model_module
            train_module = load_module_from_path(f'{prefix}.train_mace_finetune', directory / 'train_mace_finetune.py'); sys.modules['train_mace_finetune'] = train_module
            eval_module = load_module_from_path(f'{prefix}.eval_mace_finetune', directory / 'eval_mace_finetune.py')
            return model_module, train_module, eval_module

        import torch
        torch.manual_seed(SEED)
        candidate_model_module, candidate_train, candidate_eval = load_task_modules(CANDIDATE_DIR, 'mace_real_finetune_submission')
        candidate_config = candidate_train.load_config(CANDIDATE_DIR / 'mace_finetune.yaml')
        if not isinstance(candidate_config, dict): raise TypeError('Submission load_config must return a dict-like config.')
        model = candidate_train.build_model(candidate_config)
        dtype = torch.float64
        base_positions = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]], dtype=dtype)
        positions = base_positions.repeat(INPUT_BATCH_SIZE, 1); node_attrs = torch.zeros(INPUT_BATCH_SIZE * 3, 89, dtype=dtype)
        element_indices = torch.tensor([7, 0, 0], dtype=torch.long).repeat(INPUT_BATCH_SIZE); node_attrs[torch.arange(INPUT_BATCH_SIZE * 3), element_indices] = 1.0
        edges = []
        for graph in range(INPUT_BATCH_SIZE):
            offset = graph * 3; edges.extend((offset + sender, offset + receiver) for sender in range(3) for receiver in range(3) if sender != receiver)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous(); batch = torch.arange(INPUT_BATCH_SIZE, dtype=torch.long).repeat_interleave(3); ptr = torch.arange(0, 3 * (INPUT_BATCH_SIZE + 1), 3, dtype=torch.long)
        reference_batch = ReferenceBatch(positions=positions, node_attrs=node_attrs, edge_index=edge_index, shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), unit_shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), batch=batch, ptr=ptr, cell=torch.eye(3, dtype=dtype).repeat(INPUT_BATCH_SIZE, 1, 1) * 20.0, head=torch.zeros(INPUT_BATCH_SIZE, dtype=torch.long), energy=torch.zeros(INPUT_BATCH_SIZE, dtype=dtype), forces=torch.zeros(INPUT_BATCH_SIZE * 3, 3, dtype=dtype), weight=torch.linspace(0.8, 1.2, INPUT_BATCH_SIZE, dtype=dtype), energy_weight=torch.linspace(1.0, 1.4, INPUT_BATCH_SIZE, dtype=dtype), forces_weight=torch.linspace(0.7, 1.3, INPUT_BATCH_SIZE, dtype=dtype))
        model.eval()
        output = model(reference_batch.clone().to_dict(), training=False, compute_force=True, compute_virials=False, compute_stress=False, compute_displacement=False)
        if not isinstance(output, dict): raise TypeError('model forward must return a dictionary')
        update_check(current_index, isinstance(output, dict), str(type(output)))
        current_index += 1
        expected = {'energy': (INPUT_BATCH_SIZE,), 'node_energy': (30,), 'interaction_energy': (INPUT_BATCH_SIZE,), 'forces': (30, 3), 'node_feats': (30, 640)}
        for key, shape in expected.items():
            value = output.get(key)
            update_check(current_index, isinstance(value, torch.Tensor) and tuple(value.shape) == shape, f'shape={tuple(value.shape) if isinstance(value, torch.Tensor) else None}')
            current_index += 1
            update_check(current_index, isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), key)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'compare:energy_standard_valid', 'compare:energy_generated_valid', 'compare:energy_shape', 'compare:energy_forward',
        'compare:node_energy_standard_valid', 'compare:node_energy_generated_valid', 'compare:node_energy_shape', 'compare:node_energy_forward',
        'compare:interaction_energy_standard_valid', 'compare:interaction_energy_generated_valid', 'compare:interaction_energy_shape', 'compare:interaction_energy_forward',
        'compare:forces_standard_valid', 'compare:forces_generated_valid', 'compare:forces_shape', 'compare:forces_forward',
        'compare:node_feats_standard_valid', 'compare:node_feats_generated_valid', 'compare:node_feats_shape', 'compare:node_feats_forward',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        INPUT_BATCH_SIZE = 10
        SEED = 20260804
        FORWARD_COMPARE_ATOL = 1e-8

        class ReferenceBatch:
            def __init__(self, **values): self._values = values; [setattr(self, name, value) for name, value in values.items()]
            def to_dict(self): return dict(self._values)
            def clone(self): return ReferenceBatch(**{name: value.detach().clone() for name, value in self._values.items()})

        def load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None: raise ImportError(f'Could not create import spec for {path}')
            module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
        def purge_task_modules() -> None:
            roots = ('portable_mace', 'mace_energy_force_finetune', 'train_mace_finetune', 'eval_mace_finetune')
            for name in list(sys.modules):
                if any(name == root or name.startswith(root + '.') for root in roots): del sys.modules[name]
        def load_task_modules(directory: Path, prefix: str):
            purge_task_modules(); module_dir = str(directory)
            if module_dir in sys.path: sys.path.remove(module_dir)
            sys.path.insert(0, module_dir)
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + '.'): del sys.modules[name]
            package = types.ModuleType(prefix); package.__file__ = str(directory / '__init__.py'); package.__package__ = prefix; package.__path__ = [module_dir]; sys.modules[prefix] = package
            model_module = load_module_from_path(f'{prefix}.mace_energy_force_finetune', directory / 'mace_energy_force_finetune.py'); sys.modules['mace_energy_force_finetune'] = model_module
            train_module = load_module_from_path(f'{prefix}.train_mace_finetune', directory / 'train_mace_finetune.py'); sys.modules['train_mace_finetune'] = train_module
            eval_module = load_module_from_path(f'{prefix}.eval_mace_finetune', directory / 'eval_mace_finetune.py')
            return model_module, train_module, eval_module

        import torch
        torch.manual_seed(SEED)
        candidate_model_module, candidate_train, candidate_eval = load_task_modules(CANDIDATE_DIR, 'mace_real_finetune_submission')
        candidate_config = candidate_train.load_config(CANDIDATE_DIR / 'mace_finetune.yaml')
        if not isinstance(candidate_config, dict): raise TypeError('Submission load_config must return a dict-like config.')
        generated_model = candidate_train.build_model(candidate_config)
        standard_model_module, standard_train, standard_eval = load_task_modules(REFERENCE_DIR, 'mace_real_finetune_standard')
        standard_config = standard_train.load_config(REFERENCE_DIR / 'mace_finetune.yaml')
        if not isinstance(standard_config, dict): raise TypeError('Standard load_config must return a dict-like config.')
        standard_model = standard_train.build_model(standard_config)
        dtype = torch.float64
        base_positions = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]], dtype=dtype)
        positions = base_positions.repeat(INPUT_BATCH_SIZE, 1); node_attrs = torch.zeros(INPUT_BATCH_SIZE * 3, 89, dtype=dtype)
        element_indices = torch.tensor([7, 0, 0], dtype=torch.long).repeat(INPUT_BATCH_SIZE); node_attrs[torch.arange(INPUT_BATCH_SIZE * 3), element_indices] = 1.0
        edges = []
        for graph in range(INPUT_BATCH_SIZE):
            offset = graph * 3; edges.extend((offset + sender, offset + receiver) for sender in range(3) for receiver in range(3) if sender != receiver)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous(); batch = torch.arange(INPUT_BATCH_SIZE, dtype=torch.long).repeat_interleave(3); ptr = torch.arange(0, 3 * (INPUT_BATCH_SIZE + 1), 3, dtype=torch.long)
        reference_batch = ReferenceBatch(positions=positions, node_attrs=node_attrs, edge_index=edge_index, shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), unit_shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), batch=batch, ptr=ptr, cell=torch.eye(3, dtype=dtype).repeat(INPUT_BATCH_SIZE, 1, 1) * 20.0, head=torch.zeros(INPUT_BATCH_SIZE, dtype=torch.long), energy=torch.zeros(INPUT_BATCH_SIZE, dtype=dtype), forces=torch.zeros(INPUT_BATCH_SIZE * 3, 3, dtype=dtype), weight=torch.linspace(0.8, 1.2, INPUT_BATCH_SIZE, dtype=dtype), energy_weight=torch.linspace(1.0, 1.4, INPUT_BATCH_SIZE, dtype=dtype), forces_weight=torch.linspace(0.7, 1.3, INPUT_BATCH_SIZE, dtype=dtype))
        generated_model.eval(); standard_model.eval()
        standard_out = standard_model(reference_batch.clone().to_dict(), training=False, compute_force=True, compute_virials=False, compute_stress=False, compute_displacement=False)
        if not isinstance(standard_out, dict): raise TypeError('model forward must return a dictionary')
        generated_out = generated_model(reference_batch.clone().to_dict(), training=False, compute_force=True, compute_virials=False, compute_stress=False, compute_displacement=False)
        if not isinstance(generated_out, dict): raise TypeError('model forward must return a dictionary')
        for key in ('energy', 'node_energy', 'interaction_energy', 'forces', 'node_feats'):
            standard_value = standard_out.get(key); generated_value = generated_out.get(key)
            same_shape = isinstance(standard_value, torch.Tensor) and isinstance(generated_value, torch.Tensor) and standard_value.shape == generated_value.shape
            standard_valid = isinstance(standard_value, torch.Tensor) and bool(torch.isfinite(standard_value).all()) and bool(torch.count_nonzero(standard_value).item())
            generated_valid = isinstance(generated_value, torch.Tensor) and bool(torch.isfinite(generated_value).all()) and bool(torch.count_nonzero(generated_value).item())
            max_abs_diff = float((standard_value - generated_value).abs().max()) if same_shape and standard_valid and generated_valid else float('inf')
            update_check(current_index, standard_valid, 'finite and not all zero')
            current_index += 1
            update_check(current_index, generated_valid, 'finite and not all zero')
            current_index += 1
            update_check(current_index, same_shape, f"standard={getattr(standard_out.get(key), 'shape', None)} generated={getattr(generated_out.get(key), 'shape', None)}")
            current_index += 1
            update_check(current_index, same_shape and max_abs_diff <= FORWARD_COMPARE_ATOL, f'max_abs_diff={max_abs_diff}, atol={FORWARD_COMPARE_ATOL}')
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'train:metrics_finite',
        'train_compare:train_loss',
        'train_compare:energy_loss',
        'train_compare:forces_loss',
        'train_compare:grad_norm',
        'train_compare:parameters',
        'train:all_trainable_parameters_updated',
        'train:frozen_parameters_unchanged',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        INPUT_BATCH_SIZE = 10
        SEED = 20260804
        FORWARD_COMPARE_ATOL = 1e-8

        class ReferenceBatch:
            def __init__(self, **values): self._values = values; [setattr(self, name, value) for name, value in values.items()]
            def to_dict(self): return dict(self._values)
            def clone(self): return ReferenceBatch(**{name: value.detach().clone() for name, value in self._values.items()})
            def replace(self, **updates): values = dict(self._values); values.update(updates); return ReferenceBatch(**values)

        def load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None: raise ImportError(f'Could not create import spec for {path}')
            module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
        def purge_task_modules() -> None:
            roots = ('portable_mace', 'mace_energy_force_finetune', 'train_mace_finetune', 'eval_mace_finetune')
            for name in list(sys.modules):
                if any(name == root or name.startswith(root + '.') for root in roots): del sys.modules[name]
        def load_task_modules(directory: Path, prefix: str):
            purge_task_modules(); module_dir = str(directory)
            if module_dir in sys.path: sys.path.remove(module_dir)
            sys.path.insert(0, module_dir)
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + '.'): del sys.modules[name]
            package = types.ModuleType(prefix); package.__file__ = str(directory / '__init__.py'); package.__package__ = prefix; package.__path__ = [module_dir]; sys.modules[prefix] = package
            model_module = load_module_from_path(f'{prefix}.mace_energy_force_finetune', directory / 'mace_energy_force_finetune.py'); sys.modules['mace_energy_force_finetune'] = model_module
            train_module = load_module_from_path(f'{prefix}.train_mace_finetune', directory / 'train_mace_finetune.py'); sys.modules['train_mace_finetune'] = train_module
            eval_module = load_module_from_path(f'{prefix}.eval_mace_finetune', directory / 'eval_mace_finetune.py')
            return model_module, train_module, eval_module
        def finite_training_metrics(record: Any) -> bool:
            required = {'train_loss', 'energy_loss', 'forces_loss', 'grad_norm'}
            if not isinstance(record, dict) or not required.issubset(record): return False
            try:
                return all(math.isfinite(float(record[name])) for name in required) and all(math.isfinite(float(value)) for value in record.values() if isinstance(value, (bool, int, float)))
            except (TypeError, ValueError): return False

        import torch
        torch.manual_seed(SEED)
        candidate_model_module, candidate_train, candidate_eval = load_task_modules(CANDIDATE_DIR, 'mace_real_finetune_submission')
        candidate_config = candidate_train.load_config(CANDIDATE_DIR / 'mace_finetune.yaml')
        if not isinstance(candidate_config, dict): raise TypeError('Submission load_config must return a dict-like config.')
        standard_model_module, standard_train, standard_eval = load_task_modules(REFERENCE_DIR, 'mace_real_finetune_standard')
        standard_config = standard_train.load_config(REFERENCE_DIR / 'mace_finetune.yaml')
        if not isinstance(standard_config, dict): raise TypeError('Standard load_config must return a dict-like config.')
        standard_model = standard_train.build_model(standard_config)
        dtype = torch.float64; base_positions = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]], dtype=dtype); positions = base_positions.repeat(INPUT_BATCH_SIZE, 1); node_attrs = torch.zeros(INPUT_BATCH_SIZE * 3, 89, dtype=dtype)
        element_indices = torch.tensor([7, 0, 0], dtype=torch.long).repeat(INPUT_BATCH_SIZE); node_attrs[torch.arange(INPUT_BATCH_SIZE * 3), element_indices] = 1.0; edges = []
        for graph in range(INPUT_BATCH_SIZE):
            offset = graph * 3; edges.extend((offset + sender, offset + receiver) for sender in range(3) for receiver in range(3) if sender != receiver)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous(); batch = torch.arange(INPUT_BATCH_SIZE, dtype=torch.long).repeat_interleave(3); ptr = torch.arange(0, 3 * (INPUT_BATCH_SIZE + 1), 3, dtype=torch.long)
        reference_batch = ReferenceBatch(positions=positions, node_attrs=node_attrs, edge_index=edge_index, shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), unit_shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), batch=batch, ptr=ptr, cell=torch.eye(3, dtype=dtype).repeat(INPUT_BATCH_SIZE, 1, 1) * 20.0, head=torch.zeros(INPUT_BATCH_SIZE, dtype=torch.long), energy=torch.zeros(INPUT_BATCH_SIZE, dtype=dtype), forces=torch.zeros(INPUT_BATCH_SIZE * 3, 3, dtype=dtype), weight=torch.linspace(0.8, 1.2, INPUT_BATCH_SIZE, dtype=dtype), energy_weight=torch.linspace(1.0, 1.4, INPUT_BATCH_SIZE, dtype=dtype), forces_weight=torch.linspace(0.7, 1.3, INPUT_BATCH_SIZE, dtype=dtype))
        standard_model.eval(); standard_output = standard_model(reference_batch.clone().to_dict(), training=False, compute_force=True, compute_virials=False, compute_stress=False, compute_displacement=False)
        if not isinstance(standard_output, dict): raise TypeError('model forward must return a dictionary')
        energy_delta = torch.linspace(0.01, 0.02, INPUT_BATCH_SIZE, dtype=torch.float64); force_delta = torch.linspace(-0.005, 0.005, 30 * 3, dtype=torch.float64).reshape(30, 3)
        training_batch = reference_batch.replace(energy=standard_output['energy'].detach() + energy_delta, forces=standard_output['forces'].detach() + force_delta)
        candidate_train_config = dict(candidate_config); candidate_train_config['training'] = dict(standard_config['training']); candidate_train_config['training']['max_steps'] = 1
        standard_train_config = dict(standard_config); standard_train_config['training'] = dict(standard_config['training']); standard_train_config['training']['max_steps'] = 1
        initial_model = candidate_train.build_model(candidate_train_config)
        initial_parameters = {name: parameter.detach().cpu().clone() for name, parameter in initial_model.named_parameters()}
        training_batch_dict = training_batch.clone().to_dict()
        with tempfile.TemporaryDirectory(prefix='mace-real-finetune-eval-') as temporary_directory:
            work_dir = Path(temporary_directory)
            candidate_result = candidate_train.train(candidate_train_config, (training_batch_dict,), work_dir / 'candidate')
            standard_result = standard_train.train(standard_train_config, (training_batch_dict,), work_dir / 'standard')
            if not isinstance(candidate_result, dict): raise TypeError('train must return a mapping of training artifacts')
            if not isinstance(standard_result, dict): raise TypeError('standard train must return a mapping of training artifacts')
            required_result_keys = {'model', 'history', 'checkpoint_path', 'metrics_path'}
            if not required_result_keys.issubset(candidate_result): raise KeyError(f'train result is missing artifact keys: {sorted(required_result_keys - set(candidate_result))}')
            candidate_model = candidate_result['model']; standard_model = standard_result['model']; candidate_history = candidate_result['history']; standard_history = standard_result['history']
            if not isinstance(candidate_history, list) or len(candidate_history) != 1: raise ValueError('train history must contain exactly one training step')
            if not isinstance(standard_history, list) or len(standard_history) != 1: raise ValueError('standard train history must contain exactly one training step')
            candidate_metrics = candidate_history[0]; standard_metrics = standard_history[0]
            update_check(current_index, finite_training_metrics(candidate_metrics), f'metrics={candidate_metrics}')
            current_index += 1
            for key in ('train_loss', 'energy_loss', 'forces_loss', 'grad_norm'):
                actual = float(candidate_metrics.get(key, float('inf'))); expected = float(standard_metrics[key])
                update_check(current_index, math.isclose(actual, expected, rel_tol=0.0, abs_tol=FORWARD_COMPARE_ATOL), f'standard={expected} generated={actual}')
                current_index += 1
            candidate_parameters = dict(candidate_model.named_parameters()); standard_parameters = dict(standard_model.named_parameters()); parameter_names_match = set(candidate_parameters) == set(standard_parameters); max_trained_diff = 0.0
            if parameter_names_match:
                for name, actual in candidate_parameters.items():
                    expected = standard_parameters[name]
                    if actual.numel() > 0: max_trained_diff = max(max_trained_diff, float((actual.detach().cpu() - expected.detach().cpu()).abs().max()))
            update_check(current_index, parameter_names_match and max_trained_diff <= FORWARD_COMPARE_ATOL, f'max_abs_diff={max_trained_diff}, atol={FORWARD_COMPARE_ATOL}')
            current_index += 1
            trainable_updates = {name: float((parameter.detach().cpu() - initial_parameters[name]).abs().max()) for name, parameter in candidate_parameters.items() if parameter.requires_grad and parameter.numel() > 0}
            frozen_unchanged = all(torch.equal(parameter.detach().cpu(), initial_parameters[name]) for name, parameter in candidate_parameters.items() if not parameter.requires_grad)
            update_check(current_index, bool(trainable_updates) and all(value > 0.0 for value in trainable_updates.values()), f'not_updated={sorted(name for name, value in trainable_updates.items() if value == 0.0)}')
            current_index += 1
            update_check(current_index, frozen_unchanged, 'all non-trainable parameters unchanged')
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'artifact:checkpoint_path',
        'artifact:metrics_path',
        'artifact:checkpoint_exists',
        'artifact:metrics_exists',
        'artifact:checkpoint_schema',
        'artifact:metrics_schema',
        'artifact:checkpoint_matches_trained_model',
        'artifact:checkpoint_strict_reload',
        'eval:metrics_finite',
        'eval_compare:energy_mae',
        'eval_compare:forces_mae',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        INPUT_BATCH_SIZE = 10
        SEED = 20260804
        FORWARD_COMPARE_ATOL = 1e-8
        TRAINING_CHECKPOINT_NAME = 'mace_finetuned.pt'
        TRAINING_METRICS_NAME = 'metrics.jsonl'
        TRAINING_ARTIFACT_FORMAT = 'mace-mp0-energy-force-finetune-v1'

        class ReferenceBatch:
            def __init__(self, **values): self._values = values; [setattr(self, name, value) for name, value in values.items()]
            def to_dict(self): return dict(self._values)
            def clone(self): return ReferenceBatch(**{name: value.detach().clone() for name, value in self._values.items()})
            def replace(self, **updates): values = dict(self._values); values.update(updates); return ReferenceBatch(**values)

        def load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None: raise ImportError(f'Could not create import spec for {path}')
            module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
        def purge_task_modules() -> None:
            roots = ('portable_mace', 'mace_energy_force_finetune', 'train_mace_finetune', 'eval_mace_finetune')
            for name in list(sys.modules):
                if any(name == root or name.startswith(root + '.') for root in roots): del sys.modules[name]
        def load_task_modules(directory: Path, prefix: str):
            purge_task_modules(); module_dir = str(directory)
            if module_dir in sys.path: sys.path.remove(module_dir)
            sys.path.insert(0, module_dir)
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + '.'): del sys.modules[name]
            package = types.ModuleType(prefix); package.__file__ = str(directory / '__init__.py'); package.__package__ = prefix; package.__path__ = [module_dir]; sys.modules[prefix] = package
            model_module = load_module_from_path(f'{prefix}.mace_energy_force_finetune', directory / 'mace_energy_force_finetune.py'); sys.modules['mace_energy_force_finetune'] = model_module
            train_module = load_module_from_path(f'{prefix}.train_mace_finetune', directory / 'train_mace_finetune.py'); sys.modules['train_mace_finetune'] = train_module
            eval_module = load_module_from_path(f'{prefix}.eval_mace_finetune', directory / 'eval_mace_finetune.py')
            return model_module, train_module, eval_module
        def finite_training_metrics(record: Any) -> bool:
            required = {'train_loss', 'energy_loss', 'forces_loss', 'grad_norm'}
            if not isinstance(record, dict) or not required.issubset(record): return False
            try:
                return all(math.isfinite(float(record[name])) for name in required) and all(math.isfinite(float(value)) for value in record.values() if isinstance(value, (bool, int, float)))
            except (TypeError, ValueError): return False

        import torch
        torch.manual_seed(SEED)
        candidate_model_module, candidate_train, candidate_eval = load_task_modules(CANDIDATE_DIR, 'mace_real_finetune_submission')
        candidate_config = candidate_train.load_config(CANDIDATE_DIR / 'mace_finetune.yaml')
        if not isinstance(candidate_config, dict): raise TypeError('Submission load_config must return a dict-like config.')
        standard_model_module, standard_train, standard_eval = load_task_modules(REFERENCE_DIR, 'mace_real_finetune_standard')
        standard_config = standard_train.load_config(REFERENCE_DIR / 'mace_finetune.yaml')
        if not isinstance(standard_config, dict): raise TypeError('Standard load_config must return a dict-like config.')
        standard_model = standard_train.build_model(standard_config)
        dtype = torch.float64; base_positions = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]], dtype=dtype); positions = base_positions.repeat(INPUT_BATCH_SIZE, 1); node_attrs = torch.zeros(INPUT_BATCH_SIZE * 3, 89, dtype=dtype)
        element_indices = torch.tensor([7, 0, 0], dtype=torch.long).repeat(INPUT_BATCH_SIZE); node_attrs[torch.arange(INPUT_BATCH_SIZE * 3), element_indices] = 1.0; edges = []
        for graph in range(INPUT_BATCH_SIZE):
            offset = graph * 3; edges.extend((offset + sender, offset + receiver) for sender in range(3) for receiver in range(3) if sender != receiver)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous(); batch = torch.arange(INPUT_BATCH_SIZE, dtype=torch.long).repeat_interleave(3); ptr = torch.arange(0, 3 * (INPUT_BATCH_SIZE + 1), 3, dtype=torch.long)
        reference_batch = ReferenceBatch(positions=positions, node_attrs=node_attrs, edge_index=edge_index, shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), unit_shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype), batch=batch, ptr=ptr, cell=torch.eye(3, dtype=dtype).repeat(INPUT_BATCH_SIZE, 1, 1) * 20.0, head=torch.zeros(INPUT_BATCH_SIZE, dtype=torch.long), energy=torch.zeros(INPUT_BATCH_SIZE, dtype=dtype), forces=torch.zeros(INPUT_BATCH_SIZE * 3, 3, dtype=dtype), weight=torch.linspace(0.8, 1.2, INPUT_BATCH_SIZE, dtype=dtype), energy_weight=torch.linspace(1.0, 1.4, INPUT_BATCH_SIZE, dtype=dtype), forces_weight=torch.linspace(0.7, 1.3, INPUT_BATCH_SIZE, dtype=dtype))
        standard_model.eval(); standard_output = standard_model(reference_batch.clone().to_dict(), training=False, compute_force=True, compute_virials=False, compute_stress=False, compute_displacement=False)
        if not isinstance(standard_output, dict): raise TypeError('model forward must return a dictionary')
        energy_delta = torch.linspace(0.01, 0.02, INPUT_BATCH_SIZE, dtype=torch.float64); force_delta = torch.linspace(-0.005, 0.005, 30 * 3, dtype=torch.float64).reshape(30, 3)
        training_batch = reference_batch.replace(energy=standard_output['energy'].detach() + energy_delta, forces=standard_output['forces'].detach() + force_delta)
        candidate_train_config = dict(candidate_config); candidate_train_config['training'] = dict(standard_config['training']); candidate_train_config['training']['max_steps'] = 1
        standard_train_config = dict(standard_config); standard_train_config['training'] = dict(standard_config['training']); standard_train_config['training']['max_steps'] = 1
        training_batch_dict = training_batch.clone().to_dict()
        with tempfile.TemporaryDirectory(prefix='mace-real-finetune-eval-') as temporary_directory:
            work_dir = Path(temporary_directory)
            candidate_result = candidate_train.train(candidate_train_config, (training_batch_dict,), work_dir / 'candidate')
            standard_result = standard_train.train(standard_train_config, (training_batch_dict,), work_dir / 'standard')
            if not isinstance(candidate_result, dict): raise TypeError('train must return a mapping of training artifacts')
            if not isinstance(standard_result, dict): raise TypeError('standard train must return a mapping of training artifacts')
            required_result_keys = {'model', 'history', 'checkpoint_path', 'metrics_path'}
            if not required_result_keys.issubset(candidate_result): raise KeyError(f'train result is missing artifact keys: {sorted(required_result_keys - set(candidate_result))}')
            candidate_history = candidate_result['history']; standard_history = standard_result['history']
            if not isinstance(candidate_history, list) or len(candidate_history) != 1: raise ValueError('train history must contain exactly one training step')
            if not isinstance(standard_history, list) or len(standard_history) != 1: raise ValueError('standard train history must contain exactly one training step')
            output_dir = (work_dir / 'candidate').resolve(); checkpoint_path = Path(candidate_result['checkpoint_path']).resolve(); metrics_path = Path(candidate_result['metrics_path']).resolve()
            update_check(current_index, checkpoint_path == output_dir / TRAINING_CHECKPOINT_NAME, str(checkpoint_path))
            current_index += 1
            update_check(current_index, metrics_path == output_dir / TRAINING_METRICS_NAME, str(metrics_path))
            current_index += 1
            update_check(current_index, checkpoint_path.is_file(), str(checkpoint_path))
            current_index += 1
            update_check(current_index, metrics_path.is_file(), str(metrics_path))
            current_index += 1
            if checkpoint_path.is_file() and metrics_path.is_file():
                payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                required_checkpoint_keys = {'format', 'model_state_dict', 'optimizer_state_dict', 'trainable_parameter_names', 'completed_steps', 'energy_weight', 'forces_weight'}
                model = candidate_result['model']
                expected_trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                checkpoint_schema_ok = isinstance(payload, dict) and required_checkpoint_keys.issubset(payload) and payload.get('format') == TRAINING_ARTIFACT_FORMAT and isinstance(payload.get('model_state_dict'), dict) and isinstance(payload.get('optimizer_state_dict'), dict) and set(payload.get('trainable_parameter_names', ())) == expected_trainable_names and int(payload.get('completed_steps', -1)) == 1 and math.isclose(float(payload.get('energy_weight', float('nan'))), float(standard_train_config['training']['energy_weight']), rel_tol=0.0, abs_tol=0.0) and math.isclose(float(payload.get('forces_weight', float('nan'))), float(standard_train_config['training']['forces_weight']), rel_tol=0.0, abs_tol=0.0)
                update_check(current_index, checkpoint_schema_ok, f"keys={sorted(payload) if isinstance(payload, dict) else None}")
                current_index += 1
                metrics_records = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                metrics_ok = metrics_records == candidate_result['history'] and len(metrics_records) == 1 and finite_training_metrics(metrics_records[0])
                update_check(current_index, metrics_ok, f'metrics={metrics_records}')
                current_index += 1
                if checkpoint_schema_ok:
                    runtime_state = model.state_dict(); saved_state = payload['model_state_dict']
                    state_matches = set(saved_state) == set(runtime_state) and all(torch.equal(saved_state[name].detach().cpu(), runtime_state[name].detach().cpu()) for name in runtime_state)
                    update_check(current_index, state_matches, f'saved={len(saved_state)} runtime={len(runtime_state)}')
                    current_index += 1
                    loaded = candidate_eval.load_model_state(candidate_train_config, checkpoint_path); loaded_state = loaded.state_dict()
                    strict_reload_ok = set(loaded_state) == set(saved_state) and all(torch.equal(loaded_state[name].detach().cpu(), saved_state[name].detach().cpu()) for name in saved_state)
                    update_check(current_index, strict_reload_ok, f'loaded={len(loaded_state)} saved={len(saved_state)}')
                    current_index += 1
                    standard_loaded = standard_eval.load_model_state(standard_train_config, standard_result['checkpoint_path'])
                    evaluation_batch = training_batch.clone().to_dict(); candidate_metrics = candidate_eval.evaluate(loaded, (evaluation_batch,)); standard_metrics = standard_eval.evaluate(standard_loaded, (evaluation_batch,))
                    metrics_valid = set(candidate_metrics) == {'energy_mae', 'forces_mae'} and all(math.isfinite(float(value)) for value in candidate_metrics.values())
                    update_check(current_index, metrics_valid, f'metrics={candidate_metrics}')
                    current_index += 1
                    for key in ('energy_mae', 'forces_mae'):
                        actual = float(candidate_metrics.get(key, float('inf'))); expected = float(standard_metrics[key])
                        update_check(current_index, math.isclose(actual, expected, rel_tol=0.0, abs_tol=FORWARD_COMPARE_ATOL), f'standard={expected} generated={actual}')
                        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
