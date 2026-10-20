'''V2 evaluator adapter for the legacy MACE training evaluator.'''
from pathlib import Path
import sys
import ast
import importlib.util
import inspect
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


def check_datasset_pipeline(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = [
        'dataset:candidate_module_load',
        'dataset:reference_module_load',
        'dataset:MACEDataset_class',
        'dataset:MACEDataset_init_parameters',
        'dataset:collate_function',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_path = (CANDIDATE_DIR / 'mace_train.py').resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(f'missing generated file: {candidate_path}')
        candidate_spec = importlib.util.spec_from_file_location('mace_train_submission', candidate_path)
        if candidate_spec is None or candidate_spec.loader is None:
            raise ImportError('could not load generated MACE launcher')
        module = importlib.util.module_from_spec(candidate_spec)
        sys.modules[candidate_spec.name] = module
        candidate_spec.loader.exec_module(module)
        update_check(current_index, True, 'candidate module loaded')
        current_index += 1
        reference_path = (REFERENCE_DIR / 'mace_train.py').resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f'missing reference file: {reference_path}')
        reference_spec = importlib.util.spec_from_file_location('mace_train_reference', reference_path)
        if reference_spec is None or reference_spec.loader is None:
            raise ImportError(f'could not load reference model: {reference_path}')
        reference = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(reference)
        update_check(current_index, True, 'reference module loaded')
        current_index += 1
        dataset_cls = getattr(module, 'MACEDataset', None)
        assert dataset_cls is not None, 'MACEDataset class is missing'
        update_check(current_index, True, 'MACEDataset class is available')
        current_index += 1
        required = {'data_path', 'data_format', 'energy_key', 'forces_key', 'atomic_numbers', 'statistics_file', 'r_max'}
        missing = required - set(inspect.signature(dataset_cls.__init__).parameters)
        assert not missing, f'MACEDataset.__init__ missing parameters: {sorted(missing)}'
        update_check(current_index, True, 'MACEDataset constructor parameters are present')
        current_index += 1
        collate = getattr(module, 'mace_collate_fn', None)
        if collate is None:
            collate = getattr(dataset_cls, 'collate_fn', None)
        assert callable(collate), 'mace_collate_fn or MACEDataset.collate_fn is missing'
        update_check(current_index, True, 'dataset collate function is available')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = [
        'interface:candidate_file',
        'interface:candidate_module_load',
        'interface:MACEModel',
        'interface:MACEDataset',
        'interface:MACETrainLauncher',
        'interface:launcher_run',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_path = (CANDIDATE_DIR / 'mace_train.py').resolve()
        assert candidate_path.is_file(), f'missing generated file: {candidate_path}'
        update_check(current_index, True, str(candidate_path))
        current_index += 1
        spec = importlib.util.spec_from_file_location('mace_train_submission', candidate_path)
        if spec is None or spec.loader is None:
            raise ImportError('could not load generated MACE launcher')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        update_check(current_index, True, 'candidate module loaded')
        current_index += 1
        assert callable(getattr(module, 'MACEModel', None)), 'MACEModel class is missing'
        update_check(current_index, True, 'MACEModel is available')
        current_index += 1
        assert callable(getattr(module, 'MACEDataset', None)), 'MACEDataset class is missing'
        update_check(current_index, True, 'MACEDataset is available')
        current_index += 1
        launcher_cls = getattr(module, 'MACETrainLauncher', None)
        assert callable(launcher_cls), 'MACETrainLauncher class is missing'
        update_check(current_index, True, 'MACETrainLauncher is available')
        current_index += 1
        assert callable(getattr(launcher_cls, 'run', None)), 'MACETrainLauncher.run() is missing'
        update_check(current_index, True, 'MACETrainLauncher.run is available')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = [
        'forward:candidate_module_load',
        'forward:model_instance',
        'forward:trainable_parameters',
        'forward:energy_shape',
        'forward:energy_finite',
        'forward:energy_requires_grad',
        'forward:forces_shape',
        'forward:forces_finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_path = (CANDIDATE_DIR / 'mace_train.py').resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(f'missing generated file: {candidate_path}')
        spec = importlib.util.spec_from_file_location('mace_train_submission', candidate_path)
        if spec is None or spec.loader is None:
            raise ImportError('could not load generated MACE launcher')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        update_check(current_index, True, 'candidate module loaded')
        current_index += 1
        torch.manual_seed(17)
        model_cls = getattr(module, 'MACEModel')
        model = model_cls()
        model.eval()
        update_check(current_index, True, 'MACEModel instantiated')
        current_index += 1
        assert sum(parameter.numel() for parameter in model.parameters()) > 0, 'MACEModel has no trainable parameters'
        update_check(current_index, True, 'MACEModel has trainable parameters')
        current_index += 1
        numbers = torch.tensor([[8, 1, 1]], dtype=torch.long)
        positions = torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]], dtype=torch.float32, requires_grad=True)
        atom_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
        energy = model(numbers, positions, atom_mask)
        assert energy.shape == (1,), 'energy shape must be (1,)'
        update_check(current_index, True, 'energy shape is correct')
        current_index += 1
        assert torch.isfinite(energy).all(), 'energy contains non-finite values'
        update_check(current_index, True, 'energy is finite')
        current_index += 1
        assert energy.requires_grad, 'energy must require grad for force computation'
        update_check(current_index, True, 'energy requires gradients')
        current_index += 1
        forces = -torch.autograd.grad(energy.sum(), positions, create_graph=False)[0]
        assert forces.shape == (1, 3, 3), 'forces shape must be (1, N, 3)'
        update_check(current_index, True, 'forces shape is correct')
        current_index += 1
        assert torch.isfinite(forces).all(), 'forces contain non-finite values'
        update_check(current_index, True, 'forces are finite')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = [
        'structure:candidate_module_load',
        'structure:launcher_class',
        'structure:launcher_required_init_parameters',
        'structure:launcher_run_method',
        'structure:model_class',
        'structure:model_inherits_torch_module',
        'structure:model_defaults',
        'structure:optimizer_reference',
        'structure:backward_reference',
        'structure:dataloader_reference',
        'structure:no_forbidden_onescience_mace_import',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_path = (CANDIDATE_DIR / 'mace_train.py').resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(f'missing generated file: {candidate_path}')
        spec = importlib.util.spec_from_file_location('mace_train_submission', candidate_path)
        if spec is None or spec.loader is None:
            raise ImportError('could not load generated MACE launcher')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        update_check(current_index, True, 'candidate module loaded')
        current_index += 1
        launcher_cls = getattr(module, 'MACETrainLauncher', None)
        assert callable(launcher_cls), 'MACETrainLauncher class is missing'
        update_check(current_index, True, 'MACETrainLauncher class is present')
        current_index += 1
        required = {'train_file', 'valid_file', 'test_file', 'output_dir', 'epochs', 'device', 'seed', 'energy_key', 'forces_key', 'e0s', 'data_format', 'statistics_file', 'atomic_numbers', 'r_max', 'batch_size', 'valid_batch_size', 'num_channels', 'max_l', 'num_interactions'}
        missing = required - set(inspect.signature(launcher_cls.__init__).parameters)
        assert not missing, f'MACETrainLauncher.__init__ missing parameters: {sorted(missing)}'
        update_check(current_index, True, 'launcher constructor parameters are present')
        current_index += 1
        assert callable(getattr(launcher_cls, 'run', None)), 'MACETrainLauncher.run() is missing'
        update_check(current_index, True, 'launcher run method is available')
        current_index += 1
        model_cls = getattr(module, 'MACEModel', None)
        assert model_cls is not None, 'MACEModel class is missing'
        update_check(current_index, True, 'MACEModel class is present')
        current_index += 1
        assert issubclass(model_cls, torch.nn.Module), 'MACEModel must inherit from torch.nn.Module'
        update_check(current_index, True, 'MACEModel inherits torch.nn.Module')
        current_index += 1
        model_sig = inspect.signature(model_cls.__init__)
        defaults = {'atomic_numbers': (1, 8), 'hidden_dim': 16, 'num_radial': 8, 'num_interactions': 2, 'r_max': 4.0}
        for name, expected in defaults.items():
            parameter = model_sig.parameters.get(name)
            assert parameter is not None and parameter.default is not inspect.Parameter.empty, f'missing default: {name}'
            value = tuple(parameter.default) if isinstance(expected, tuple) else parameter.default
            assert value == expected, f'{name} default mismatch: {value!r} != {expected!r}'
        update_check(current_index, True, 'model defaults match the contract')
        current_index += 1
        source = candidate_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert 'Adam' in names or 'Adam' in attrs, 'training script does not reference Adam optimizer'
        update_check(current_index, True, 'training script references Adam')
        current_index += 1
        assert 'backward' in names or 'backward' in attrs, 'training script does not reference backward()'
        update_check(current_index, True, 'training script references backward')
        current_index += 1
        assert 'DataLoader' in names or 'DataLoader' in attrs, 'training script does not reference DataLoader'
        update_check(current_index, True, 'training script references DataLoader')
        current_index += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert 'onescience' not in node.module or 'mace' not in node.module, f'forbidden import of existing MACE implementation: {node.module}'
        update_check(current_index, True, 'training script has no forbidden OneScience MACE import')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    CHECKPOINT_NAMES = [
        'training:candidate_module_load',
        'training:launcher_class',
        'training:launcher_run_method',
        'training:artifact_contract',
        'training:metrics_contract',
        'training:checkpoint_contract',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_path = (CANDIDATE_DIR / 'mace_train.py').resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(f'missing generated file: {candidate_path}')
        spec = importlib.util.spec_from_file_location('mace_train_submission', candidate_path)
        if spec is None or spec.loader is None:
            raise ImportError('could not load generated MACE launcher')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        update_check(current_index, True, 'candidate module loaded')
        current_index += 1
        launcher_cls = getattr(module, 'MACETrainLauncher', None)
        assert callable(launcher_cls), 'MACETrainLauncher class is missing'
        update_check(current_index, True, 'MACETrainLauncher is available')
        current_index += 1
        assert callable(getattr(launcher_cls, 'run', None)), 'MACETrainLauncher.run() is missing'
        update_check(current_index, True, 'MACETrainLauncher.run is available')
        current_index += 1
        source = candidate_path.read_text(encoding='utf-8')
        assert all(name in source for name in ('train.log', 'metrics.jsonl', 'mace_train.model')), 'training artifact names are missing'
        update_check(current_index, True, 'training artifact contract is present')
        current_index += 1
        assert 'Training complete' in source and 'epoch' in source and 'loss' in source, 'metrics or completion contract is missing'
        update_check(current_index, True, 'metrics contract is present')
        current_index += 1
        assert 'model_state_dict' in source and 'model_config' in source, 'checkpoint contract is missing'
        update_check(current_index, True, 'checkpoint contract is present')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks
