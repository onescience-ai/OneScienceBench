"""V2 evaluator adapter for the MACE element affine adapter."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
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

REQUIRED_FILES = (
    'mace_element_affine_adapter.py',
    'train_mace_adapter.py',
    'eval_mace_adapter.py',
    'mace_adapter.yaml',
    'portable_mace/__init__.py',
)
EXPECTED_PARAMETER_COUNT = 4688656
EXPECTED_STATE_ENTRY_COUNT = 79
EXPECTED_ADAPTER_PARAMETER_COUNT = 90
INPUT_BATCH_SIZE = 10
FORWARD_COMPARE_ATOL = 1e-8
TRAINING_CHECKPOINT_NAME = 'mace_element_affine_adapter.pt'
TRAINING_METRICS_NAME = 'metrics.json'
TRAINING_ARTIFACT_FORMAT = 'mace-element-affine-adapter-v1'

class _Batch:
    def __init__(self, **values):
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)
    def __getitem__(self, key):
        return self._values[key]
    def to_dict(self):
        return dict(self._values)
    def clone(self):
        return _Batch(**{key: value.detach().clone() for key, value in self._values.items()})
    def to(self, device):
        return _Batch(**{key: value.to(device) for key, value in self._values.items()})
    def replace(self, **updates):
        values = dict(self._values)
        values.update(updates)
        return _Batch(**values)

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not create import spec for {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module

def _load_task(directory: Path, prefix: str):
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + '.') or name in ('portable_mace', 'mace_element_affine_adapter', 'train_mace_adapter', 'eval_mace_adapter'):
            del sys.modules[name]
    model = _load_module(prefix + '.mace_element_affine_adapter', directory / 'mace_element_affine_adapter.py')
    sys.modules['mace_element_affine_adapter'] = model
    train = _load_module(prefix + '.train_mace_adapter', directory / 'train_mace_adapter.py')
    sys.modules['train_mace_adapter'] = train
    evaluation = _load_module(prefix + '.eval_mace_adapter', directory / 'eval_mace_adapter.py')
    return model, train, evaluation

def _batch(torch):
    dtype = torch.float64
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]], dtype=dtype).repeat(10, 1)
    node_attrs = torch.zeros(30, 89, dtype=dtype)
    elements = torch.tensor([7, 0, 0], dtype=torch.long).repeat(10)
    node_attrs[torch.arange(30), elements] = 1.0
    edges = []
    for graph in range(10):
        offset = graph * 3
        edges.extend((offset + sender, offset + receiver) for sender in range(3) for receiver in range(3) if sender != receiver)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return _Batch(
        positions=positions,
        node_attrs=node_attrs,
        edge_index=edge_index,
        shifts=torch.zeros(60, 3, dtype=dtype),
        unit_shifts=torch.zeros(60, 3, dtype=dtype),
        batch=torch.arange(10, dtype=torch.long).repeat_interleave(3),
        ptr=torch.arange(0, 33, 3, dtype=torch.long),
        cell=torch.eye(3, dtype=dtype).repeat(10, 1, 1) * 20.0,
        head=torch.zeros(10, dtype=torch.long),
        energy=torch.zeros(10, dtype=dtype),
        forces=torch.zeros(30, 3, dtype=dtype),
    )

def _forward(model, batch, training=False, compute_force=True):
    output = model(batch.clone().to_dict(), training=training, compute_force=compute_force, compute_virials=False, compute_stress=False, compute_displacement=False)
    if not isinstance(output, dict):
        raise TypeError('model forward must return a dictionary')
    return output

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _forbidden(directory: Path):
    invalid = []
    for path in sorted(directory.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.append(node.module)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ('__import__', 'import_module'):
                    invalid.append(f'{path.relative_to(directory)}:dynamic_import')
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module':
                    invalid.append(f'{path.relative_to(directory)}:dynamic_import')
        forbidden = ('onescience', 'mace', 'evaluation.eval_datasets.codegen.standard', 'evaluation.eval_datasets.codegen.evaluators')
        invalid.extend(f'{path.relative_to(directory)}:{name}' for name in imports if any(name == item or name.startswith(item + '.') for item in forbidden))
    return invalid

def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['foundation:paths_resolved', 'foundation:config_path', 'foundation:weights_path', 'foundation:config_exists', 'foundation:weights_exists', 'foundation:format', 'foundation:source_fingerprint', 'foundation:weights_fingerprint', 'foundation:state_entry_count', 'foundation:tensors_only']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        train = _load_module('mace_adapter_checkpoint_train', CANDIDATE_DIR / 'train_mace_adapter.py')
        resolver = getattr(train, 'foundation_paths', None)
        ok = callable(resolver)
        update_check(current_index, ok, 'foundation_paths')
        current_index += 1
        if not ok:
            raise TypeError('foundation_paths is not callable')
        load_config = getattr(train, 'load_config', None)
        if not callable(load_config):
            raise TypeError('load_config is not callable')
        candidate_config = load_config(CANDIDATE_DIR / 'mace_adapter.yaml')
        resolved = resolver(candidate_config)
        if not isinstance(resolved, (tuple, list)) or len(resolved) != 2:
            raise TypeError('foundation_paths must return (config_path, weights_path)')
        config_path = Path(resolved[0]).expanduser().resolve()
        weights_path = Path(resolved[1]).expanduser().resolve()
        expected_config = Path(os.environ.get('MACE_MP0_PORTABLE_CONFIG_PATH', '/public/share/sugonhpcapp01/onestore/onemodels/mace/mace-mp-0-medium-finetune-portable.json')).expanduser().resolve()
        expected_weights = Path(os.environ.get('MACE_MP0_PORTABLE_WEIGHTS_PATH', '/public/share/sugonhpcapp01/onestore/onemodels/mace/mace-mp-0-medium-finetune-portable.pt')).expanduser().resolve()
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
            update_check(current_index, document.get('format') == 'mace-portable-state-dict-v1' and payload.get('format') == 'mace-portable-state-dict-v1', 'portable format')
            current_index += 1
            update_check(current_index, document.get('source_sha256') == '81b87e94b1a2e5faf6a683af75bedaafa168a69de7099d2f54f616fc8d08af26' and payload.get('source_sha256') == '81b87e94b1a2e5faf6a683af75bedaafa168a69de7099d2f54f616fc8d08af26', 'source fingerprint')
            current_index += 1
            update_check(current_index, _sha256(weights_path) == 'dfb8e7cf1c4de2b8faaeef97bcb50514bb9950d61970c75af556cdb964a9d1a9', 'weights fingerprint')
            current_index += 1
            update_check(current_index, len(payload['state_dict']) == 79, f"count={len(payload['state_dict'])}")
            current_index += 1
            update_check(current_index, all(isinstance(value, torch.Tensor) for value in payload['state_dict'].values()), 'weights_only=True payload')
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['data:batch_size', 'data:positions_shape', 'data:node_attrs_shape', 'data:edge_index_shape', 'data:finite']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        batch = _batch(torch)
        values = batch.to_dict()
        floating = [value for value in values.values() if isinstance(value, torch.Tensor) and value.is_floating_point()]
        results = [(values['ptr'].numel() - 1 == 10, 'graphs=10'), (tuple(values['positions'].shape) == (30, 3), str(tuple(values['positions'].shape))), (tuple(values['node_attrs'].shape) == (30, 89), str(tuple(values['node_attrs'].shape))), (tuple(values['edge_index'].shape) == (2, 60), str(tuple(values['edge_index'].shape))), (all(bool(torch.isfinite(value).all()) for value in floating), 'all floating tensors finite')]
        for passed, message in results:
            update_check(current_index, passed, message)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['file:mace_element_affine_adapter.py', 'file:train_mace_adapter.py', 'file:eval_mace_adapter.py', 'file:mace_adapter.yaml', 'file:portable_mace/__init__.py']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        for relative in REQUIRED_FILES:
            path = CANDIDATE_DIR / relative
            update_check(current_index, path.is_file(), str(path))
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['train:has_load_config', 'train:has_foundation_paths', 'train:has_build_model', 'train:has_train_step', 'train:has_train', 'train:train_accepts_output_dir', 'train:train_signature', 'train:uses_adamw', 'train:build_model_returns_module', 'eval:has_load_model_state', 'eval:load_model_state_signature', 'eval:has_evaluate', 'eval:evaluate_signature', 'eval:strict_state_loading']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch.nn as nn
        train = _load_module('mace_adapter_interface_train', CANDIDATE_DIR / 'train_mace_adapter.py')
        evaluation = _load_module('mace_adapter_interface_eval', CANDIDATE_DIR / 'eval_mace_adapter.py')
        model = train.build_model(train.load_config(CANDIDATE_DIR / 'mace_adapter.yaml')) if callable(getattr(train, 'build_model', None)) and callable(getattr(train, 'load_config', None)) else None
        train_results = [(callable(getattr(train, 'load_config', None)), 'load_config'), (callable(getattr(train, 'foundation_paths', None)), 'foundation_paths'), (callable(getattr(train, 'build_model', None)), 'build_model'), (callable(getattr(train, 'train_step', None)), 'train_step'), (callable(getattr(train, 'train', None)), 'train')]
        for passed, message in train_results:
            update_check(current_index, passed, message)
            current_index += 1
        train_function = getattr(train, 'train', None)
        signature = inspect.signature(train_function) if callable(train_function) else None
        update_check(current_index, signature is not None and 'output_dir' in signature.parameters, str(signature))
        current_index += 1
        update_check(current_index, signature is not None and all(name in signature.parameters for name in ('config', 'batches', 'output_dir')), str(signature))
        current_index += 1
        update_check(current_index, 'AdamW' in Path(train.__file__).read_text(encoding='utf-8'), 'AdamW')
        current_index += 1
        update_check(current_index, isinstance(model, nn.Module), str(type(model)))
        current_index += 1
        loader = getattr(evaluation, 'load_model_state', None)
        evaluator = getattr(evaluation, 'evaluate', None)
        loader_signature = inspect.signature(loader) if callable(loader) else None
        evaluator_signature = inspect.signature(evaluator) if callable(evaluator) else None
        update_check(current_index, callable(loader), 'load_model_state')
        current_index += 1
        update_check(current_index, loader_signature is not None and all(name in loader_signature.parameters for name in ('config', 'state_dict')), str(loader_signature))
        current_index += 1
        update_check(current_index, callable(evaluator), 'evaluate')
        current_index += 1
        update_check(current_index, evaluator_signature is not None and all(name in evaluator_signature.parameters for name in ('model', 'batches')), str(evaluator_signature))
        current_index += 1
        update_check(current_index, 'strict=True' in Path(evaluation.__file__).read_text(encoding='utf-8'), 'strict=True')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['static:no_forbidden_imports']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        invalid = _forbidden(CANDIDATE_DIR)
        update_check(current_index, invalid == [], f'imports={invalid}')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['model:is_nn_module', 'model:has_real_base', 'model:base_parameter_count', 'model:base_state_entry_count', 'model:float64_weights', 'model:element_count', 'model:r_max', 'model:interaction_count', 'model:readout_count', 'adapter:init_contract', 'adapter:energy_scale_parameter', 'adapter:element_shift_parameter', 'adapter:trainable_parameter_count', 'adapter:base_frozen']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        import torch.nn as nn
        model_module, train, _ = _load_task(CANDIDATE_DIR, 'mace_adapter_structure')
        model_cls = getattr(model_module, 'ScaleShiftMACE')
        adapter_cls = getattr(model_module, 'MACEElementAffineAdapter')
        model = train.build_model(train.load_config(CANDIDATE_DIR / 'mace_adapter.yaml'))
        base = getattr(model, 'model', None)
        signature = inspect.signature(adapter_cls.__init__)
        base_count = sum(parameter.numel() for parameter in base.parameters()) if isinstance(base, nn.Module) else -1
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        results = [(isinstance(model, nn.Module), str(type(model))), (isinstance(base, model_cls), str(type(base))), (base_count == EXPECTED_PARAMETER_COUNT, f'count={base_count}'), (isinstance(base, nn.Module) and len(base.state_dict()) == EXPECTED_STATE_ENTRY_COUNT, 'state entries'), (isinstance(base, nn.Module) and all(parameter.dtype == torch.float64 for parameter in base.parameters() if parameter.is_floating_point()), 'float64'), (isinstance(base, nn.Module) and int(base.atomic_numbers.numel()) == 89, 'elements'), (isinstance(base, nn.Module) and math.isclose(float(base.r_max), 6.0, rel_tol=0.0, abs_tol=1e-12), 'r_max'), (isinstance(base, nn.Module) and len(base.interactions) == 2 and len(base.products) == 2, 'interactions/products'), (isinstance(base, nn.Module) and len(base.readouts) == 2, 'readouts'), (all(name in signature.parameters for name in ('model', 'atomic_numbers', 'trainable', 'freeze_base', 'energy_scale_init', 'element_shift_init')), str(signature)), (isinstance(getattr(model, 'energy_scale', None), nn.Parameter), 'energy_scale'), (isinstance(getattr(model, 'element_shift', None), nn.Parameter) and model.element_shift.shape == (89,), 'element_shift'), (trainable_count == EXPECTED_ADAPTER_PARAMETER_COUNT, f'count={trainable_count}'), (isinstance(base, nn.Module) and not any(parameter.requires_grad for parameter in base.parameters()), 'freeze_base=true')]
        for passed, message in results:
            update_check(current_index, passed, message)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['forward:returns_mapping', 'forward:energy_shape', 'forward:energy_finite', 'forward:node_energy_shape', 'forward:node_energy_finite', 'forward:interaction_energy_shape', 'forward:interaction_energy_finite', 'forward:forces_shape', 'forward:forces_finite', 'forward:node_feats_shape', 'forward:node_feats_finite']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        _, train, _ = _load_task(CANDIDATE_DIR, 'mace_adapter_forward')
        model = train.build_model(train.load_config(CANDIDATE_DIR / 'mace_adapter.yaml')).eval()
        output = _forward(model, _batch(torch))
        update_check(current_index, isinstance(output, dict), str(type(output)))
        current_index += 1
        for key, shape in (('energy', (10,)), ('node_energy', (30,)), ('interaction_energy', (10,)), ('forces', (30, 3)), ('node_feats', (30, 640))):
            value = output.get(key)
            update_check(current_index, isinstance(value, torch.Tensor) and tuple(value.shape) == shape, f'shape={getattr(value, "shape", None)}')
            current_index += 1
            update_check(current_index, isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), key)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['compare:energy_standard_valid', 'compare:energy_generated_valid', 'compare:energy_shape', 'compare:energy_forward', 'compare:node_energy_standard_valid', 'compare:node_energy_generated_valid', 'compare:node_energy_shape', 'compare:node_energy_forward', 'compare:interaction_energy_standard_valid', 'compare:interaction_energy_generated_valid', 'compare:interaction_energy_shape', 'compare:interaction_energy_forward', 'compare:forces_standard_valid', 'compare:forces_generated_valid', 'compare:forces_shape', 'compare:forces_forward', 'compare:node_feats_standard_valid', 'compare:node_feats_generated_valid', 'compare:node_feats_shape', 'compare:node_feats_forward']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        _, candidate_train, _ = _load_task(CANDIDATE_DIR, 'mace_adapter_candidate')
        _, standard_train, _ = _load_task(REFERENCE_DIR, 'mace_adapter_reference')
        candidate = candidate_train.build_model(candidate_train.load_config(CANDIDATE_DIR / 'mace_adapter.yaml')).eval()
        standard = standard_train.build_model(standard_train.load_config(REFERENCE_DIR / 'mace_adapter.yaml')).eval()
        batch = _batch(torch)
        candidate_output = _forward(candidate, batch)
        standard_output = _forward(standard, batch)
        for key in ('energy', 'node_energy', 'interaction_energy', 'forces', 'node_feats'):
            standard_value = standard_output.get(key)
            candidate_value = candidate_output.get(key)
            same_shape = isinstance(standard_value, torch.Tensor) and isinstance(candidate_value, torch.Tensor) and standard_value.shape == candidate_value.shape
            standard_valid = isinstance(standard_value, torch.Tensor) and bool(torch.isfinite(standard_value).all()) and bool(torch.count_nonzero(standard_value).item())
            candidate_valid = isinstance(candidate_value, torch.Tensor) and bool(torch.isfinite(candidate_value).all()) and bool(torch.count_nonzero(candidate_value).item())
            difference = float((standard_value - candidate_value).abs().max()) if same_shape and standard_valid and candidate_valid else float('inf')
            for passed, message in ((standard_valid, 'finite and not all zero'), (candidate_valid, 'finite and not all zero'), (same_shape, 'same shape'), (same_shape and standard_valid and candidate_valid and difference <= FORWARD_COMPARE_ATOL, f'max_abs_diff={difference}')):
                update_check(current_index, passed, message)
                current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['adapter:energy_semantics', 'adapter:node_energy_semantics', 'adapter:interaction_energy_semantics', 'adapter:forces_semantics', 'adapter:node_feats_semantics', 'adapter:energy_scale_gradient', 'adapter:element_shift_gradient', 'adapter:frozen_base_has_no_gradient']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        _, train, _ = _load_task(CANDIDATE_DIR, 'mace_adapter_semantics')
        model = train.build_model(train.load_config(CANDIDATE_DIR / 'mace_adapter.yaml')).eval()
        batch = _batch(torch)
        with torch.no_grad():
            model.energy_scale.fill_(1.4)
            model.element_shift.zero_()
            model.element_shift[0] = 0.2
            model.element_shift[7] = -0.3
        base = _forward(model.model, batch)
        adapted = _forward(model, batch)
        species = torch.argmax(batch.node_attrs, dim=-1)
        node_shift = model.element_shift.detach()[species]
        graph_shift = torch.zeros_like(base['energy']).index_add(0, batch.batch, node_shift)
        expected = {'energy': 1.4 * base['energy'] + graph_shift, 'node_energy': 1.4 * base['node_energy'] + node_shift, 'interaction_energy': 1.4 * base['interaction_energy'], 'forces': 1.4 * base['forces'], 'node_feats': base['node_feats']}
        for key in ('energy', 'node_energy', 'interaction_energy', 'forces', 'node_feats'):
            difference = float((adapted[key] - expected[key]).abs().max())
            update_check(current_index, difference <= FORWARD_COMPARE_ATOL, f'max_abs_diff={difference}')
            current_index += 1
        model.zero_grad(set_to_none=True)
        _forward(model, batch, training=True, compute_force=False)['energy'].sum().backward()
        for name in ('energy_scale', 'element_shift'):
            gradient = getattr(model, name).grad
            update_check(current_index, gradient is not None and bool(torch.isfinite(gradient).all()) and bool(torch.any(gradient != 0)), f'gradient={gradient is not None}')
            current_index += 1
        update_check(current_index, all(parameter.grad is None for parameter in model.model.parameters()), 'base gradients are None')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['train:loss_finite', 'train_compare:loss', 'train_compare:energy_scale', 'train:energy_scale_updated', 'train_compare:element_shift', 'train:element_shift_updated', 'train:frozen_base_unchanged', 'train:only_adapter_trainable']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        _, candidate_train, _ = _load_task(CANDIDATE_DIR, 'mace_adapter_training_candidate')
        _, standard_train, _ = _load_task(REFERENCE_DIR, 'mace_adapter_training_reference')
        candidate_config = candidate_train.load_config(CANDIDATE_DIR / 'mace_adapter.yaml')
        standard_config = standard_train.load_config(REFERENCE_DIR / 'mace_adapter.yaml')
        batch = _batch(torch)
        candidate_initial = candidate_train.build_model(candidate_config)
        initial_base = {name: value.detach().cpu().clone() for name, value in candidate_initial.model.state_dict().items()}
        initial_adapter = {name: getattr(candidate_initial, name).detach().cpu().clone() for name in ('energy_scale', 'element_shift')}
        standard_model = standard_train.build_model(standard_config)
        standard_output = _forward(standard_model, batch)
        training_batch = batch.replace(energy=standard_output['energy'].detach() + torch.linspace(0.01, 0.02, 10, dtype=torch.float64), forces=standard_output['forces'].detach() + torch.linspace(-0.005, 0.005, 90, dtype=torch.float64).reshape(30, 3)).to_dict()
        candidate_config = dict(candidate_config)
        candidate_config['epochs'] = 1
        candidate_config['training'] = dict(candidate_config.get('training', {}))
        with tempfile.TemporaryDirectory(prefix='mace-adapter-training-') as directory:
            candidate_result = candidate_train.train(candidate_config, (training_batch,), Path(directory) / 'candidate')
            standard_result = standard_train.train(standard_config, (training_batch,), Path(directory) / 'standard')
        if not isinstance(candidate_result, dict) or not isinstance(standard_result, dict):
            raise TypeError('train must return a mapping of training artifacts')
        if not {'model', 'history', 'checkpoint_path', 'metrics_path'}.issubset(candidate_result):
            raise KeyError('train result is missing artifact keys')
        candidate_history = candidate_result['history']
        standard_history = standard_result['history']
        if not isinstance(candidate_history, list) or len(candidate_history) != 1 or not isinstance(standard_history, list) or len(standard_history) != 1:
            raise ValueError('train history must contain exactly one batch loss')
        candidate_loss = float(candidate_history[0]['loss'] if isinstance(candidate_history[0], dict) else candidate_history[0])
        standard_loss = float(standard_history[0]['loss'] if isinstance(standard_history[0], dict) else standard_history[0])
        update_check(current_index, math.isfinite(candidate_loss), f'loss={candidate_loss}')
        current_index += 1
        update_check(current_index, math.isclose(candidate_loss, standard_loss, rel_tol=0.0, abs_tol=FORWARD_COMPARE_ATOL), f'standard={standard_loss} generated={candidate_loss}')
        current_index += 1
        trained = candidate_result['model']
        reference_trained = standard_result['model']
        for name in ('energy_scale', 'element_shift'):
            candidate_value = getattr(trained, name).detach()
            standard_value = getattr(reference_trained, name).detach()
            difference = float((candidate_value - standard_value).abs().max())
            update_check(current_index, difference <= FORWARD_COMPARE_ATOL, f'max_abs_diff={difference}')
            current_index += 1
            update = float((candidate_value.cpu() - initial_adapter[name]).abs().max())
            update_check(current_index, update > 0.0, f'max_update={update}')
            current_index += 1
        unchanged = all(torch.equal(value.detach().cpu(), initial_base[name]) for name, value in trained.model.state_dict().items())
        update_check(current_index, unchanged, 'base state unchanged')
        current_index += 1
        names = {name for name, parameter in trained.named_parameters() if parameter.requires_grad}
        update_check(current_index, names == {'energy_scale', 'element_shift'}, f'trainable={sorted(names)}')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_metrics(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['artifact:checkpoint_path', 'artifact:metrics_path', 'artifact:checkpoint_exists', 'artifact:metrics_exists', 'artifact:checkpoint_schema', 'artifact:metrics_schema', 'artifact:checkpoint_matches_trained_model', 'artifact:checkpoint_strict_reload', 'eval:metrics_finite', 'eval_compare:energy_mae', 'eval_compare:forces_mae']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch

        # The training check creates its artifacts in a temporary directory and
        # removes them before this independent rule invocation runs. Recreate
        # that same caller-owned output flow here instead of inspecting the
        # static reference directory for generated files.
        _, candidate_train, candidate_eval = _load_task(
            CANDIDATE_DIR, 'mace_adapter_infer_candidate'
        )
        _, standard_train, standard_eval = _load_task(
            REFERENCE_DIR, 'mace_adapter_infer_reference'
        )
        candidate_config = candidate_train.load_config(
            CANDIDATE_DIR / 'mace_adapter.yaml'
        )
        standard_config = standard_train.load_config(
            REFERENCE_DIR / 'mace_adapter.yaml'
        )
        batch = _batch(torch)
        standard_model = standard_train.build_model(standard_config)
        standard_output = _forward(standard_model, batch)
        training_batch = batch.replace(
            energy=standard_output['energy'].detach()
            + torch.linspace(0.01, 0.02, 10, dtype=torch.float64),
            forces=standard_output['forces'].detach()
            + torch.linspace(-0.005, 0.005, 90, dtype=torch.float64).reshape(30, 3),
        ).to_dict()
        candidate_train_config = dict(candidate_config)
        candidate_train_config['epochs'] = 1
        candidate_train_config['training'] = dict(
            candidate_config.get('training', {})
        )

        with tempfile.TemporaryDirectory(prefix='mace-adapter-infer-') as directory:
            output_dir = Path(directory) / 'candidate'
            candidate_result = candidate_train.train(
                candidate_train_config, (training_batch,), output_dir
            )
            standard_result = standard_train.train(
                standard_config, (training_batch,), Path(directory) / 'standard'
            )
            if not isinstance(candidate_result, dict) or not isinstance(standard_result, dict):
                raise TypeError('train must return a mapping of training artifacts')
            required_result_keys = {
                'model', 'history', 'checkpoint_path', 'metrics_path'
            }
            if not required_result_keys.issubset(candidate_result):
                raise KeyError('train result is missing artifact keys')

            checkpoint_path = Path(candidate_result['checkpoint_path']).expanduser().resolve()
            metrics_path = Path(candidate_result['metrics_path']).expanduser().resolve()
            output_dir = output_dir.resolve()
            results = [
                (checkpoint_path == output_dir / TRAINING_CHECKPOINT_NAME, str(checkpoint_path)),
                (metrics_path == output_dir / TRAINING_METRICS_NAME, str(metrics_path)),
                (checkpoint_path.is_file(), str(checkpoint_path)),
                (metrics_path.is_file(), str(metrics_path)),
            ]
            for passed, message in results:
                update_check(current_index, passed, message)
                current_index += 1
            if not checkpoint_path.is_file() or not metrics_path.is_file():
                return checks

            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            candidate_history = candidate_result['history']
            required_checkpoint_keys = {
                'format', 'model_state_dict', 'optimizer_state_dict',
                'trainable_parameter_names', 'history',
            }
            checkpoint_schema_ok = (
                isinstance(payload, dict)
                and required_checkpoint_keys.issubset(payload)
                and payload.get('format') == TRAINING_ARTIFACT_FORMAT
                and isinstance(payload.get('model_state_dict'), dict)
                and isinstance(payload.get('optimizer_state_dict'), dict)
                and set(payload.get('trainable_parameter_names', ()))
                == {'energy_scale', 'element_shift'}
                and payload.get('history') == candidate_history
            )
            update_check(current_index, checkpoint_schema_ok, 'checkpoint schema')
            current_index += 1
            metrics_history = metrics.get('history') if isinstance(metrics, dict) else None
            metrics_schema_ok = (
                isinstance(metrics_history, list)
                and metrics_history == candidate_history
                and len(metrics_history) == 1
                and all(
                    math.isfinite(
                        float(value['loss'] if isinstance(value, dict) and 'loss' in value else value)
                    )
                    for value in metrics_history
                )
            )
            update_check(current_index, metrics_schema_ok, f'metrics={metrics}')
            current_index += 1
            if not checkpoint_schema_ok:
                return checks

            saved_state = payload['model_state_dict']
            runtime_state = candidate_result['model'].state_dict()
            state_matches = (
                set(saved_state) == set(runtime_state)
                and all(
                    torch.equal(
                        saved_state[name].detach().cpu(),
                        runtime_state[name].detach().cpu(),
                    )
                    for name in runtime_state
                )
            )
            update_check(
                current_index,
                state_matches,
                f'saved={len(saved_state)} runtime={len(runtime_state)}',
            )
            current_index += 1

            candidate_loaded = candidate_eval.load_model_state(
                candidate_config, saved_state
            )
            standard_loaded = standard_eval.load_model_state(
                standard_config, standard_result['model'].state_dict()
            )
            loaded_state = candidate_loaded.state_dict()
            strict_reload_ok = (
                set(loaded_state) == set(saved_state)
                and all(
                    torch.equal(
                        loaded_state[name].detach().cpu(),
                        saved_state[name].detach().cpu(),
                    )
                    for name in saved_state
                )
            )
            update_check(
                current_index,
                strict_reload_ok,
                f'loaded={len(loaded_state)} saved={len(saved_state)}',
            )
            current_index += 1
            candidate_metrics = candidate_eval.evaluate(
                candidate_loaded, (training_batch,)
            )
            standard_metrics = standard_eval.evaluate(
                standard_loaded, (training_batch,)
            )
            metrics_valid = (
                set(candidate_metrics) == {'energy_mae', 'forces_mae'}
                and all(math.isfinite(float(value)) for value in candidate_metrics.values())
            )
            update_check(current_index, metrics_valid, f'metrics={candidate_metrics}')
            current_index += 1
            for name in ('energy_mae', 'forces_mae'):
                actual = float(candidate_metrics.get(name, float('inf')))
                expected = float(standard_metrics[name])
                update_check(
                    current_index,
                    math.isclose(actual, expected, rel_tol=0.0, abs_tol=FORWARD_COMPARE_ATOL),
                    f'standard={expected} generated={actual}',
                )
                current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks
