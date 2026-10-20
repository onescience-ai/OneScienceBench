'''V2 adapter for MatterGen property fine-tuning evaluation.'''
from __future__ import annotations

import ast
import gc
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
    'mattergen_property_finetune.py',
    'train_mattergen_finetune.py',
    'mattergen_property_finetune.yaml',
    'portable_mattergen/__init__.py',
    'portable_mattergen/gemnet/gemnet-dT.json',
)
FORBIDDEN_IMPORT_ROOTS = {
    'onescience', 'mattergen', 'torch_geometric', 'torch_scatter', 'torch_sparse',
}
OUTPUT_KEYS = ('pos', 'cell', 'atomic_numbers')
CHECKPOINT_PATH = Path('/public/share/sugonhpcapp01/onestore/onemodels/mattergen/mattergen_base/checkpoints/last.ckpt')
EXPECTED_CHECKPOINT_SHA256 = '81668ee12afc1ee1b037f362420730de3460bfd2d36e547585fdcb911a3dfdef'
EXPECTED_BASE_STATE_ENTRY_COUNT = 281
EXPECTED_BASE_STATE_NUMEL = 53_737_595
EXPECTED_STATE_ENTRY_COUNT = 302
EXPECTED_STATE_NUMEL = 57_936_251
EXPECTED_ADAPTER_PARAMETER_COUNT = 4_198_400
EXPECTED_ADAPTER_PARAMETER_TENSORS = 20
FORWARD_ATOL = 1.0e-6
SEED = 20260819

def _shared_context(config: Dict, caller_globals, caller_locals) -> dict[str, Any]:
    for scope in (caller_globals, caller_locals, config):
        if isinstance(scope, dict):
            context = scope.get('_mattergen_property_finetune_context')
            if not isinstance(context, dict):
                context = {}
                scope['_mattergen_property_finetune_context'] = context
            return context
    return {}

def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not create import spec for {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _purge_loaded_modules() -> None:
    for name in list(sys.modules):
        if name in {'mattergen_property_finetune', 'train_mattergen_finetune'} or name.startswith('portable_mattergen') or name.startswith('mattergen_property_finetune_eval_loaded_'):
            sys.modules.pop(name, None)

def _load_implementation(root: Path, role: str):
    _purge_loaded_modules()
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    model_module = _load_module('mattergen_property_finetune', root / 'mattergen_property_finetune.py')
    train_module = _load_module(f'mattergen_property_finetune_eval_loaded_{role}', root / 'train_mattergen_finetune.py')
    return model_module, train_module

def _candidate_modules(context: dict[str, Any]):
    if 'candidate_model_module' not in context:
        model_module, train_module = _load_implementation(CANDIDATE_DIR, 'candidate')
        candidate_config = model_module.load_config(CANDIDATE_DIR / 'mattergen_property_finetune.yaml')
        context.update({'candidate_model_module': model_module, 'candidate_train_module': train_module, 'candidate_config': candidate_config})
    return context['candidate_model_module'], context['candidate_train_module'], context['candidate_config']

def _tensor_equal(left: Any, right: Any) -> bool:
    import torch
    return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and bool(torch.equal(left.detach().cpu(), right.detach().cpu()))

def _fixture_to_device(fixture: dict[str, Any], device: Any) -> dict[str, Any]:
    return {'batch': fixture['batch'].to(device), 'timesteps': fixture['timesteps'].to(device), 'targets': {key: value.to(device) for key, value in fixture['targets'].items()}, 'atom_loss_mask': fixture['atom_loss_mask'].to(device)}

def _select_device(torch) -> Any:
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def initialize_parameters_with_constants(model, weight_value: float = 0.002, bias_value: float = 0.0001, embedding_value: float = 0.001, norm_value: float = 1.0, attention_value: float = 0.001, temperature_value: float = 1.0, other_value: float = 0.001):
    import torch
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not (name.startswith('gemnet.cond_adapt_layers.') or name.startswith('gemnet.cond_mixin_layers.')):
                continue
            name_lower = name.lower()
            if 'bias' in name_lower:
                parameter.fill_(bias_value)
            elif any(token in name_lower for token in ('embedding', 'embed', 'token', 'position', 'pos_embed')):
                parameter.fill_(embedding_value)
            elif any(token in name_lower for token in ('norm', 'ln', 'layernorm')):
                parameter.fill_(norm_value)
            elif 'temperature' in name_lower:
                parameter.fill_(temperature_value)
            elif parameter.ndim >= 2 and 'weight' in name_lower:
                pattern = torch.arange(parameter.numel(), device=parameter.device, dtype=parameter.dtype).reshape(parameter.shape)
                pattern = pattern - pattern.mean()
                scale = pattern.abs().max().clamp_min(1.0)
                parameter.copy_(weight_value * pattern / scale)
            elif any(token in name_lower for token in ('attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value')):
                parameter.fill_(attention_value)
            else:
                parameter.fill_(other_value if parameter.ndim == 0 else embedding_value)
    return model

def _reset_forward_rng(torch) -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def _scan_sources(submission_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    has_torch_load = False
    has_strict_load = False
    python_files = sorted(submission_dir.rglob('*.py'))
    for path in python_files:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.', 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                        errors.append(f'{path.relative_to(submission_dir)}: forbidden import {alias.name}')
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module
            if imported and imported.split('.', 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                errors.append(f'{path.relative_to(submission_dir)}: forbidden import {imported}')
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                normalized = node.value.replace(chr(92), '/').lower()
                if 'benchmarks/standard' in normalized:
                    errors.append(f'{path.relative_to(submission_dir)}: standard source access is forbidden')
                if 'benchmarks/evaluators' in normalized:
                    errors.append(f'{path.relative_to(submission_dir)}: evaluator source access is forbidden')
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == '__import__':
                    errors.append(f'{path.relative_to(submission_dir)}: dynamic __import__ is forbidden')
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'import_module':
                        errors.append(f'{path.relative_to(submission_dir)}: dynamic import_module is forbidden')
                    if node.func.attr == 'load' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch':
                        has_torch_load = True
                    if node.func.attr == 'load_state_dict':
                        has_strict_load |= any(keyword.arg == 'strict' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords)
    return {'python_file_count': len(python_files), 'errors': errors, 'has_torch_load': has_torch_load, 'has_strict_load': has_strict_load}

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file_mattergen_property_finetune_py', 'file_train_mattergen_finetune_py', 'file_mattergen_property_finetune_yaml', 'file_portable_mattergen_init', 'file_gemnet_config', 'eval_script_optional_not_required']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        context = _shared_context(config, caller_globals, caller_locals)
        for relative in REQUIRED_FILES:
            ok = (CANDIDATE_DIR / relative).is_file()
            update_check(current_index, ok, 'required file is present' if ok else f'missing required file: {relative}')
            current_index += 1
        optional_paths = [path.name for path in CANDIDATE_DIR.glob('eval*.py') if path.name != 'evaluate_mattergen_property_finetune.py']
        update_check(current_index, True, f'not required; optional eval files: {sorted(optional_paths)}')
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['fixture_keys', 'batch_keys', 'target_keys', 'batch_pos_shape', 'batch_cell_shape', 'batch_atomic_numbers_shape', 'batch_num_atoms_shape', 'batch_dft_mag_density_shape', 'batch_unconditional_mask_shape', 'target_pos_shape', 'target_cell_shape', 'target_atomic_numbers_shape', 'batch_pos_dtype', 'batch_cell_dtype', 'batch_atomic_numbers_dtype', 'batch_num_atoms_dtype', 'batch_dft_mag_density_dtype', 'batch_unconditional_mask_dtype', 'timesteps_shape', 'timesteps_dtype', 'atom_loss_mask_shape', 'atom_loss_mask_dtype', 'num_atoms_values', 'num_atoms_sum', 'atom_loss_mask_values', 'masked_atomic_numbers', 'dft_mag_density_values', 'conditional_property_embeddings', 'finite_pos_and_cell', 'positive_cell_determinants', 'fixture_determinism']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = _shared_context(config, caller_globals, caller_locals)
        _model_module, train_module, _candidate_config = _candidate_modules(context)
        first = train_module.build_finetune_batch(seed=SEED)
        second = train_module.build_finetune_batch(seed=SEED)
        required = {'batch', 'timesteps', 'targets', 'atom_loss_mask'}
        ok = isinstance(first, dict) and set(first) == required
        update_check(current_index, ok, 'fixture keys are exact' if ok else f'fixture keys must be {sorted(required)}')
        current_index += 1
        if ok:
            batch = first['batch']
            targets = first['targets']
            expected_batch_shapes = {'pos': (7, 3), 'cell': (2, 3, 3), 'atomic_numbers': (7,), 'num_atoms': (2,), 'dft_mag_density': (2,), 'use_unconditional_dft_mag_density': (2, 1)}
            expected_target_shapes = {'pos': (7, 3), 'cell': (2, 3, 3), 'atomic_numbers': (7,)}
            ok = set(batch) == set(expected_batch_shapes)
            update_check(current_index, ok, 'batch keys are exact' if ok else 'batch must contain the four crystal fields and two property-conditioning fields')
            current_index += 1
            ok = set(targets) == set(expected_target_shapes)
            update_check(current_index, ok, 'target keys are exact' if ok else 'targets must contain exactly three denoising fields')
            current_index += 1
            for key, shape in expected_batch_shapes.items():
                value = batch.get(key)
                ok = isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                update_check(current_index, ok, f'batch[{key}] shape is {shape}' if ok else f'batch[{key}] shape must be {shape}')
                current_index += 1
            for key, shape in expected_target_shapes.items():
                value = targets.get(key)
                ok = isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                update_check(current_index, ok, f'targets[{key}] shape is {shape}' if ok else f'targets[{key}] shape must be {shape}')
                current_index += 1
            for key, dtype, message in [('pos', torch.float32, 'pos must be float32'), ('cell', torch.float32, 'cell must be float32'), ('atomic_numbers', torch.long, 'atomic_numbers must be int64'), ('num_atoms', torch.long, 'num_atoms must be int64'), ('dft_mag_density', torch.float32, 'dft_mag_density must be float32'), ('use_unconditional_dft_mag_density', torch.bool, 'property conditioning mask must be bool')]:
                ok = batch[key].dtype == dtype
                update_check(current_index, ok, 'dtype is correct' if ok else message)
                current_index += 1
            ok = tuple(first['timesteps'].shape) == (2,)
            update_check(current_index, ok, 'timesteps shape is correct' if ok else 'timesteps must have shape [2]')
            current_index += 1
            ok = first['timesteps'].dtype == torch.float32
            update_check(current_index, ok, 'timesteps dtype is correct' if ok else 'timesteps must be float32')
            current_index += 1
            ok = tuple(first['atom_loss_mask'].shape) == (7,)
            update_check(current_index, ok, 'atom loss mask shape is correct' if ok else 'atom_loss_mask must have shape [7]')
            current_index += 1
            ok = first['atom_loss_mask'].dtype == torch.bool
            update_check(current_index, ok, 'atom loss mask dtype is correct' if ok else 'atom_loss_mask must be bool')
            current_index += 1
            ok = batch['num_atoms'].tolist() == [4, 3]
            update_check(current_index, ok, 'num_atoms values are correct' if ok else 'num_atoms must be [4, 3]')
            current_index += 1
            ok = int(batch['num_atoms'].sum()) == 7
            update_check(current_index, ok, 'num_atoms sum is correct' if ok else 'num_atoms must sum to seven')
            current_index += 1
            ok = first['atom_loss_mask'].tolist() == [False, True, False, False, False, True, False]
            update_check(current_index, ok, 'fixed atom mask is correct' if ok else 'fixed atom mask is incorrect')
            current_index += 1
            ok = batch['atomic_numbers'][first['atom_loss_mask']].tolist() == [101, 101]
            update_check(current_index, ok, 'masked atom types are correct' if ok else 'masked atom types must use class 101')
            current_index += 1
            ok = bool(torch.allclose(batch['dft_mag_density'], torch.tensor([0.12, 0.68])))
            update_check(current_index, ok, 'fixed dft_mag_density values are correct' if ok else 'fixed dft_mag_density values are incorrect')
            current_index += 1
            ok = not bool(batch['use_unconditional_dft_mag_density'].any())
            update_check(current_index, ok, 'batch uses conditional property embeddings' if ok else 'the fixed batch must use conditional property embeddings')
            current_index += 1
            ok = bool(torch.isfinite(batch['pos']).all() and torch.isfinite(batch['cell']).all())
            update_check(current_index, ok, 'batch values are finite' if ok else 'batch contains non-finite values')
            current_index += 1
            ok = bool((torch.linalg.det(batch['cell']) > 0).all())
            update_check(current_index, ok, 'cell determinants are positive' if ok else 'cells must be invertible with positive determinant')
            current_index += 1
            deterministic = all(_tensor_equal(first['batch'][key], second['batch'][key]) for key in expected_batch_shapes) and all(_tensor_equal(first['targets'][key], second['targets'][key]) for key in expected_target_shapes) and _tensor_equal(first['timesteps'], second['timesteps']) and _tensor_equal(first['atom_loss_mask'], second['atom_loss_mask'])
            update_check(current_index, deterministic, 'fixture is deterministic' if deterministic else 'build_finetune_batch must be deterministic for a fixed seed')
            current_index += 1
            context['fixture'] = first
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['required_functions', 'source_restrictions', 'torch_load_call', 'strict_load_state_dict_call', 'cli_flags', 'loss_backward_call', 'optimizer_step_call', 'adam_construction', 'model_hidden_dim', 'model_denoise_atom_types', 'model_atom_type_diffusion', 'gemnet_num_targets', 'gemnet_cutoff', 'gemnet_emb_size_atom', 'gemnet_emb_size_edge', 'gemnet_max_neighbors', 'gemnet_max_cell_images_per_dim', 'gemnet_num_blocks', 'gemnet_regress_stress', 'gemnet_otf_graph', 'finetuning_property_name', 'finetuning_checkpoint_path', 'finetuning_full_finetuning', 'finetuning_seed', 'finetuning_optimizer', 'finetuning_steps', 'finetuning_learning_rate', 'finetuning_gradient_clip_val', 'finetuning_loss_weights']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        context = _shared_context(config, caller_globals, caller_locals)
        model_module, train_module, candidate_config = _candidate_modules(context)
        model_functions = ('load_config', 'build_model', 'load_checkpoint_payload', 'extract_base_model_state', 'load_pretrained_model', 'is_adapter_parameter')
        train_functions = ('build_finetune_batch', 'compute_materials_loss', 'train_step', 'train', 'main')
        missing = [name for name in model_functions if not callable(getattr(model_module, name, None))] + [name for name in train_functions if not callable(getattr(train_module, name, None))]
        ok = not missing
        update_check(current_index, ok, 'all required functions are present' if ok else f'missing required functions: {missing}')
        current_index += 1
        source_scan = _scan_sources(CANDIDATE_DIR)
        ok = not source_scan['errors']
        update_check(current_index, ok, 'source restrictions are satisfied' if ok else '; '.join(source_scan['errors']))
        current_index += 1
        ok = source_scan['has_torch_load']
        update_check(current_index, ok, 'torch.load call is present' if ok else 'submission must load the fixed checkpoint with torch.load')
        current_index += 1
        ok = source_scan['has_strict_load']
        update_check(current_index, ok, 'strict load_state_dict call is present' if ok else 'submission must call load_state_dict(..., strict=True)')
        current_index += 1
        source = (CANDIDATE_DIR / 'train_mattergen_finetune.py').read_text(encoding='utf-8')
        tree = ast.parse(source, filename='train_mattergen_finetune.py')
        cli_flags: set[str] = set()
        called_attributes: set[str] = set()
        called_names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            if isinstance(node.func, ast.Attribute):
                called_attributes.add(node.func.attr)
                if node.func.attr == 'add_argument' and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    cli_flags.add(node.args[0].value)
        required_flags = {'--config', '--output-dir', '--device'}
        ok = required_flags <= cli_flags
        update_check(current_index, ok, 'required CLI flags are present' if ok else f'CLI flags must include {sorted(required_flags)}')
        current_index += 1
        ok = 'backward' in called_attributes
        update_check(current_index, ok, 'loss.backward call is present' if ok else 'training script must call loss.backward()')
        current_index += 1
        ok = 'step' in called_attributes
        update_check(current_index, ok, 'optimizer.step call is present' if ok else 'training script must call optimizer.step()')
        current_index += 1
        ok = 'Adam' in called_attributes or 'Adam' in called_names
        update_check(current_index, ok, 'Adam construction is present' if ok else 'training script must construct Adam')
        current_index += 1
        expected_model = {'hidden_dim': 512, 'denoise_atom_types': True, 'atom_type_diffusion': 'mask'}
        expected_gemnet = {'num_targets': 1, 'cutoff': 7.0, 'emb_size_atom': 512, 'emb_size_edge': 512, 'max_neighbors': 50, 'max_cell_images_per_dim': 5, 'num_blocks': 4, 'regress_stress': True, 'otf_graph': True}
        model_config = candidate_config.get('model', {})
        gemnet_config = model_config.get('gemnet', {})
        for key, expected in expected_model.items():
            ok = model_config.get(key) == expected
            update_check(current_index, ok, f'model.{key} is correct' if ok else f'model.{key} must be {expected!r}')
            current_index += 1
        for key, expected in expected_gemnet.items():
            ok = gemnet_config.get(key) == expected
            update_check(current_index, ok, f'model.gemnet.{key} is correct' if ok else f'model.gemnet.{key} must be {expected!r}')
            current_index += 1
        finetuning = candidate_config.get('finetuning', {})
        conditions = [
            (finetuning.get('property_name') == 'dft_mag_density', 'finetuning.property_name must be dft_mag_density'),
            (Path(str(finetuning.get('checkpoint_path', ''))) == CHECKPOINT_PATH, f'checkpoint_path must be {CHECKPOINT_PATH}'),
            (finetuning.get('full_finetuning') is False, 'full_finetuning must be false for adapter-only tuning'),
            (finetuning.get('seed') == SEED, f'finetuning.seed must be {SEED}'),
            (finetuning.get('optimizer') == 'Adam', 'finetuning.optimizer must be Adam'),
            (finetuning.get('steps') == 2, 'finetuning.steps must be 2'),
            (math.isclose(float(finetuning.get('learning_rate', -1)), 5.0e-6), 'finetuning.learning_rate must be 5e-6'),
            (math.isclose(float(finetuning.get('gradient_clip_val', -1)), 0.5), 'finetuning.gradient_clip_val must be 0.5'),
            (finetuning.get('loss_weights') == {'pos': 0.1, 'cell': 1.0, 'atomic_numbers': 1.0}, "loss_weights must be {'pos': 0.1, 'cell': 1.0, 'atomic_numbers': 1.0}"),
        ]
        for ok, message in conditions:
            update_check(current_index, ok, 'configuration value is correct' if ok else message)
            current_index += 1
        if not missing:
            signatures = {name: str(inspect.signature(getattr(train_module, name))) for name in train_functions[:-1]}
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['checkpoint_file', 'checkpoint_sha256', 'model_is_module', 'model_class', 'model_class_local', 'forward_signature', 'top_level_order', 'hidden_dim', 'property_embeddings_type', 'property_adapters_type', 'property_adapter_keys', 'gemnet_components', 'interaction_graph_builder', 'gemnet_block_counts', 'adapt_layers_type', 'mixin_layers_type', 'adapter_block_counts', 'shape_probe_execution', 'atom_embedding_shape', 'time_embedding_shape', 'atom_head_shape', 'property_embedding_shape', 'state_entry_count', 'state_numel', 'trainable_tensor_count', 'trainable_numel', 'trainable_adapter_names', 'frozen_pretrained_parameters', 'zero_initialized_mixin', 'base_state_entry_count', 'base_state_numel', 'base_state_restored_exactly']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = _shared_context(config, caller_globals, caller_locals)
        model_module, _train_module, candidate_config = _candidate_modules(context)
        torch.manual_seed(SEED)
        model = model_module.load_pretrained_model(candidate_config, CHECKPOINT_PATH, device='cpu')
        ok = CHECKPOINT_PATH.is_file()
        update_check(current_index, ok, 'fixed checkpoint is present' if ok else f'fixed checkpoint is missing: {CHECKPOINT_PATH}')
        current_index += 1
        ok = _sha256(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_SHA256 if CHECKPOINT_PATH.is_file() else False
        update_check(current_index, ok, 'fixed checkpoint SHA256 is correct' if ok else 'fixed checkpoint SHA256 differs')
        current_index += 1
        ok = isinstance(model, torch.nn.Module)
        update_check(current_index, ok, 'model is a torch.nn.Module' if ok else 'build_model must return torch.nn.Module')
        current_index += 1
        ok = type(model).__name__ == 'MatterGenPropertyAdapter'
        update_check(current_index, ok, 'model class is MatterGenPropertyAdapter' if ok else 'model class must be MatterGenPropertyAdapter')
        current_index += 1
        try:
            source_path = Path(inspect.getfile(type(model))).resolve()
            local_class = source_path.is_relative_to(CANDIDATE_DIR.resolve())
        except Exception:
            source_path = Path('<unknown>')
            local_class = False
        update_check(current_index, local_class, 'model class is local to the submission' if local_class else 'MatterGenPropertyAdapter must be implemented inside the submission')
        current_index += 1
        forward_parameters = list(inspect.signature(model.forward).parameters)
        ok = forward_parameters[:2] == ['batch', 'timesteps']
        update_check(current_index, ok, 'forward signature starts with batch and timesteps' if ok else 'forward signature must start with (batch, timesteps)')
        current_index += 1
        top_children = [name for name, _module in model.named_children()]
        required_order = ['gemnet', 'noise_level_encoding', 'property_embeddings', 'fc_atom', 'property_embeddings_adapt']
        ok = top_children == required_order
        update_check(current_index, ok, 'top-level module order is correct' if ok else f'top-level module order must be {required_order}')
        current_index += 1
        ok = int(getattr(model, 'hidden_dim', -1)) == 512
        update_check(current_index, ok, 'hidden_dim is correct' if ok else 'hidden_dim must be 512')
        current_index += 1
        ok = isinstance(getattr(model, 'property_embeddings', None), torch.nn.ModuleDict)
        update_check(current_index, ok, 'property_embeddings is a ModuleDict' if ok else 'property_embeddings must be ModuleDict')
        current_index += 1
        property_adapters = getattr(model, 'property_embeddings_adapt', None)
        ok = isinstance(property_adapters, torch.nn.ModuleDict)
        update_check(current_index, ok, 'property_embeddings_adapt is a ModuleDict' if ok else 'property_embeddings_adapt must be ModuleDict')
        current_index += 1
        ok = list(property_adapters) == ['dft_mag_density'] if isinstance(property_adapters, torch.nn.ModuleDict) else False
        update_check(current_index, ok, 'property adapter keys are correct' if ok else 'exactly the dft_mag_density property adapter is required')
        current_index += 1
        backbone = getattr(model, 'gemnet', None)
        component_names = ('radial_basis', 'cbf_basis3', 'mlp_rbf_lattice', 'mlp_rbf3', 'mlp_cbf3', 'mlp_rbf_h', 'mlp_rbf_out', 'atom_emb', 'atom_latent_emb', 'edge_emb', 'int_blocks', 'out_blocks', 'lattice_out_blocks')
        missing_components = [name for name in component_names if not hasattr(backbone, name)]
        ok = not missing_components
        update_check(current_index, ok, 'required GemNet-T components are present' if ok else f'GemNet-T components missing: {missing_components}')
        current_index += 1
        ok = callable(getattr(backbone, 'generate_interaction_graph', None))
        update_check(current_index, ok, 'GemNet-T graph builder is present' if ok else 'GemNet-T graph builder is missing')
        current_index += 1
        block_counts = {'interaction': len(getattr(backbone, 'int_blocks', [])), 'output': len(getattr(backbone, 'out_blocks', [])), 'lattice': len(getattr(backbone, 'lattice_out_blocks', []))}
        ok = block_counts == {'interaction': 4, 'output': 5, 'lattice': 5}
        update_check(current_index, ok, 'GemNet-T block counts are correct' if ok else f'incorrect GemNet-T block counts: {block_counts}')
        current_index += 1
        adapt_layers = getattr(backbone, 'cond_adapt_layers', None)
        mixin_layers = getattr(backbone, 'cond_mixin_layers', None)
        ok = isinstance(adapt_layers, torch.nn.ModuleDict)
        update_check(current_index, ok, 'cond_adapt_layers is a ModuleDict' if ok else 'cond_adapt_layers must be ModuleDict')
        current_index += 1
        ok = isinstance(mixin_layers, torch.nn.ModuleDict)
        update_check(current_index, ok, 'cond_mixin_layers is a ModuleDict' if ok else 'cond_mixin_layers must be ModuleDict')
        current_index += 1
        adapter_block_counts = {'adapt': len(adapt_layers['dft_mag_density']) if isinstance(adapt_layers, torch.nn.ModuleDict) and 'dft_mag_density' in adapt_layers else 0, 'mixin': len(mixin_layers['dft_mag_density']) if isinstance(mixin_layers, torch.nn.ModuleDict) and 'dft_mag_density' in mixin_layers else 0}
        ok = adapter_block_counts == {'adapt': 4, 'mixin': 4}
        update_check(current_index, ok, 'property-adapter block counts are correct' if ok else f'incorrect property-adapter block counts: {adapter_block_counts}')
        current_index += 1
        embedding_shapes: dict[str, Any] = {}
        probe_error = ''
        try:
            embedding_shapes['atom'] = list(backbone.atom_emb(torch.tensor([1, 6, 14, 101], dtype=torch.long)).shape)
            embedding_shapes['time'] = list(model.noise_level_encoding(torch.tensor([0.125, 0.875])).shape)
            embedding_shapes['atom_head'] = list(model.fc_atom(torch.zeros((3, 512), dtype=torch.float32)).shape)
            embedding_shapes['property'] = list(property_adapters['dft_mag_density'](torch.tensor([0.12, 0.68]), torch.zeros(2, 1, dtype=torch.bool)).shape)
            probe_ok = True
        except Exception as exc:
            probe_ok = False
            probe_error = f'module shape probes failed: {type(exc).__name__}: {exc}'
        update_check(current_index, probe_ok, 'module shape probes completed' if probe_ok else probe_error)
        current_index += 1
        for key, expected, message in [('atom', [4, 512], 'atom embedding shape must be [4, 512]'), ('time', [2, 512], 'time embedding shape must be [2, 512]'), ('atom_head', [3, 101], 'atom head shape must be [3, 101]'), ('property', [2, 512], 'property embedding shape must be [2, 512]')]:
            ok = embedding_shapes.get(key) == expected
            update_check(current_index, ok, 'module shape is correct' if ok else message)
            current_index += 1
        state = model.state_dict()
        state_numel = sum(value.numel() for value in state.values())
        ok = len(state) == EXPECTED_STATE_ENTRY_COUNT
        update_check(current_index, ok, 'state entry count is correct' if ok else f'state_dict must contain {EXPECTED_STATE_ENTRY_COUNT} entries')
        current_index += 1
        ok = state_numel == EXPECTED_STATE_NUMEL
        update_check(current_index, ok, 'state element count is correct' if ok else f'state_dict must contain {EXPECTED_STATE_NUMEL} elements')
        current_index += 1
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        frozen = {name: parameter for name, parameter in model.named_parameters() if not parameter.requires_grad}
        trainable_numel = sum(parameter.numel() for parameter in trainable.values())
        ok = len(trainable) == EXPECTED_ADAPTER_PARAMETER_TENSORS
        update_check(current_index, ok, 'trainable tensor count is correct' if ok else f'adapter must expose {EXPECTED_ADAPTER_PARAMETER_TENSORS} trainable tensors')
        current_index += 1
        ok = trainable_numel == EXPECTED_ADAPTER_PARAMETER_COUNT
        update_check(current_index, ok, 'trainable element count is correct' if ok else f'adapter must expose {EXPECTED_ADAPTER_PARAMETER_COUNT} trainable elements')
        current_index += 1
        ok = all(model_module.is_adapter_parameter(name) for name in trainable)
        update_check(current_index, ok, 'only adapter parameters are trainable' if ok else 'only adapter parameters may be trainable')
        current_index += 1
        ok = bool(frozen)
        update_check(current_index, ok, 'pretrained parameters are frozen' if ok else 'pretrained MatterGen parameters must be frozen')
        current_index += 1
        zero_mixin = all(bool(torch.count_nonzero(parameter.detach()) == 0) for name, parameter in trainable.items() if name.startswith('gemnet.cond_mixin_layers.'))
        update_check(current_index, zero_mixin, 'mixin weights are zero initialized' if zero_mixin else 'all mixin weights must be zero initialized')
        current_index += 1
        payload = model_module.load_checkpoint_payload(CHECKPOINT_PATH)
        base_state = model_module.extract_base_model_state(payload)
        ok = len(base_state) == EXPECTED_BASE_STATE_ENTRY_COUNT
        update_check(current_index, ok, 'base state entry count is correct' if ok else f'base checkpoint must contain {EXPECTED_BASE_STATE_ENTRY_COUNT} entries')
        current_index += 1
        ok = sum(value.numel() for value in base_state.values()) == EXPECTED_BASE_STATE_NUMEL
        update_check(current_index, ok, 'base state element count is correct' if ok else f'base checkpoint must contain {EXPECTED_BASE_STATE_NUMEL} elements')
        current_index += 1
        loaded_base_matches = set(base_state) <= set(state) and all(torch.equal(base_state[name].detach().cpu(), state[name].detach().cpu()) for name in base_state)
        update_check(current_index, loaded_base_matches, 'base checkpoint tensors restored exactly' if loaded_base_matches else 'local model did not restore every base checkpoint tensor exactly')
        current_index += 1
        del model, state
        gc.collect()
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['pos_tensor', 'pos_shape', 'pos_floating', 'pos_finite', 'pos_requires_grad', 'cell_tensor', 'cell_shape', 'cell_floating', 'cell_finite', 'cell_requires_grad', 'atomic_numbers_tensor', 'atomic_numbers_shape', 'atomic_numbers_floating', 'atomic_numbers_finite', 'atomic_numbers_requires_grad', 'property_sensitivity']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = _shared_context(config, caller_globals, caller_locals)
        model_module, train_module, candidate_config = _candidate_modules(context)
        fixture = context.get('fixture') or train_module.build_finetune_batch(seed=SEED)
        device = _select_device(torch)
        torch.manual_seed(SEED)
        model = model_module.load_pretrained_model(candidate_config, CHECKPOINT_PATH, device=device).train()
        initialize_parameters_with_constants(model)
        moved = _fixture_to_device(fixture, device)
        output = model(moved['batch'], moved['timesteps'])
        shapes = {'pos': (7, 3), 'cell': (2, 3, 3), 'atomic_numbers': (7, 101)}
        for key, expected_shape in shapes.items():
            value = output.get(key) if hasattr(output, 'get') else None
            ok = isinstance(value, torch.Tensor)
            update_check(current_index, ok, f'output[{key}] is a tensor' if ok else f'output[{key}] must be a tensor')
            current_index += 1
            ok = isinstance(value, torch.Tensor) and tuple(value.shape) == expected_shape
            update_check(current_index, ok, f'output[{key}] shape is correct' if ok else f'output[{key}] shape must be {expected_shape}')
            current_index += 1
            ok = isinstance(value, torch.Tensor) and value.is_floating_point()
            update_check(current_index, ok, f'output[{key}] is floating point' if ok else f'output[{key}] must be floating point')
            current_index += 1
            ok = isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
            update_check(current_index, ok, f'output[{key}] is finite' if ok else f'output[{key}] contains NaN or Inf')
            current_index += 1
            ok = isinstance(value, torch.Tensor) and value.requires_grad
            update_check(current_index, ok, f'output[{key}] retains gradients' if ok else f'output[{key}] must retain gradients in training mode')
            current_index += 1
        altered_batch = moved['batch'].replace(dft_mag_density=moved['batch']['dft_mag_density'] + 0.25)
        altered_output = model(altered_batch, moved['timesteps'])
        property_sensitivity = {key: float((output[key] - altered_output[key]).abs().max().detach().cpu()) for key in OUTPUT_KEYS}
        ok = any(value > 0.0 for value in property_sensitivity.values())
        update_check(current_index, ok, 'conditioned forward depends on dft_mag_density' if ok else 'conditioned forward must depend on dft_mag_density')
        current_index += 1
        context['device'] = str(device)
        del model, moved, output, altered_output
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['state_layout', 'initialized_state_values', 'forward_pos', 'forward_cell', 'forward_atomic_numbers', 'weighted_materials_loss', 'pos_loss_component', 'cell_loss_component', 'atomic_numbers_loss_component']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = _shared_context(config, caller_globals, caller_locals)
        candidate_model_module, candidate_train_module, candidate_config = _candidate_modules(context)
        fixture = context.get('fixture') or candidate_train_module.build_finetune_batch(seed=SEED)
        reference_model_module, reference_train_module = _load_implementation(REFERENCE_DIR, 'standard')
        reference_config = reference_model_module.load_config(REFERENCE_DIR / 'mattergen_property_finetune.yaml')
        torch.manual_seed(SEED)
        reference_model = reference_model_module.load_pretrained_model(reference_config, CHECKPOINT_PATH, device='cpu').train()
        torch.manual_seed(SEED)
        candidate_model = candidate_model_module.load_pretrained_model(candidate_config, CHECKPOINT_PATH, device='cpu').train()
        initialize_parameters_with_constants(reference_model)
        initialize_parameters_with_constants(candidate_model)
        reference_state = reference_model.state_dict()
        candidate_state = candidate_model.state_dict()
        same_layout = set(reference_state) == set(candidate_state) and all(tuple(reference_state[name].shape) == tuple(candidate_state[name].shape) for name in reference_state.keys() & candidate_state.keys())
        update_check(current_index, same_layout, 'candidate and standard state layouts match' if same_layout else 'candidate and standard state layouts differ')
        current_index += 1
        identical_initialized_state = same_layout and all(torch.equal(reference_state[name], candidate_state[name]) for name in reference_state)
        update_check(current_index, identical_initialized_state, 'shared initialization produced identical states' if identical_initialized_state else 'shared deterministic initialization did not produce identical states')
        current_index += 1
        if same_layout and identical_initialized_state:
            device = torch.device('cpu')
            moved = _fixture_to_device(fixture, device)
            candidate_model = candidate_model.to(device)
            _reset_forward_rng(torch)
            candidate_output_device = candidate_model(moved['batch'], moved['timesteps'])
            candidate_output = {key: candidate_output_device[key].detach().cpu() for key in OUTPUT_KEYS}
            candidate_model.cpu()
            del candidate_output_device
            reference_model = reference_model.to(device)
            _reset_forward_rng(torch)
            reference_output_device = reference_model(moved['batch'], moved['timesteps'])
            reference_output = {key: reference_output_device[key].detach().cpu() for key in OUTPUT_KEYS}
            for key in OUTPUT_KEYS:
                actual = candidate_output[key]
                expected = reference_output[key]
                max_abs = float((actual - expected).abs().max().item())
                close = bool(torch.allclose(actual, expected, rtol=1.0e-6, atol=FORWARD_ATOL))
                update_check(current_index, close, f'{key} matches standard' if close else f'{key} differs from standard (max_abs_diff={max_abs})')
                current_index += 1
            cpu_fixture = _fixture_to_device(fixture, torch.device('cpu'))
            candidate_loss, candidate_components = candidate_train_module.compute_materials_loss(candidate_output, cpu_fixture['targets'], cpu_fixture['atom_loss_mask'], candidate_config['finetuning']['loss_weights'])
            reference_loss, reference_components = reference_train_module.compute_materials_loss(reference_output, cpu_fixture['targets'], cpu_fixture['atom_loss_mask'], reference_config['finetuning']['loss_weights'])
            loss_diff = abs(float(candidate_loss) - float(reference_loss))
            ok = math.isclose(float(candidate_loss), float(reference_loss), rel_tol=1.0e-6, abs_tol=1.0e-6)
            update_check(current_index, ok, 'weighted MaterialsLoss matches standard' if ok else f'weighted MaterialsLoss differs from standard by {loss_diff}')
            current_index += 1
            for key in ('pos', 'cell', 'atomic_numbers'):
                actual_component = float(candidate_components[key])
                expected_component = float(reference_components[key])
                ok = math.isclose(actual_component, expected_component, rel_tol=1.0e-6, abs_tol=1.0e-6)
                update_check(current_index, ok, f'{key} loss matches standard' if ok else f'{key} loss differs from standard')
                current_index += 1
            del candidate_model, reference_model, moved, reference_output_device
            gc.collect()
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['training_checkpoint_created', 'train_return_mapping', 'checkpoint_payload_mapping', 'checkpoint_state_layout', 'checkpoint_format', 'checkpoint_completed_steps', 'optimizer_state', 'trainable_parameter_names', 'all_adapter_tensors_changed', 'frozen_parameters_unchanged', 'strict_checkpoint_reload', 'history_record_count', 'history_1_keys', 'history_1_finite_values', 'history_1_step', 'history_1_positive_loss', 'history_1_positive_gradient_norm', 'history_1_adapter_gradients', 'history_2_keys', 'history_2_finite_values', 'history_2_step', 'history_2_positive_loss', 'history_2_positive_gradient_norm', 'history_2_adapter_gradients', 'return_history_matches_checkpoint']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = _shared_context(config, caller_globals, caller_locals)
        model_module, train_module, candidate_config = _candidate_modules(context)
        finetuning = candidate_config['finetuning']
        torch.manual_seed(int(finetuning['seed']))
        baseline_model = model_module.load_pretrained_model(candidate_config, CHECKPOINT_PATH, device='cpu')
        initial_adapter_state = {name: parameter.detach().cpu().clone() for name, parameter in baseline_model.named_parameters() if parameter.requires_grad}
        frozen_parameter_names = [name for name, parameter in baseline_model.named_parameters() if not parameter.requires_grad]
        baseline_keys = set(baseline_model.state_dict())
        del baseline_model
        gc.collect()
        temporary = tempfile.TemporaryDirectory(prefix='mattergen-property-finetune-eval-')
        output_dir = Path(temporary.name)
        context['training_temporary'] = temporary
        context['training_output_dir'] = output_dir
        train_result = train_module.train(CANDIDATE_DIR / 'mattergen_property_finetune.yaml', output_dir, device='auto')
        checkpoint_path = output_dir / 'mattergen_property_finetuned.pt'
        ok = checkpoint_path.is_file()
        update_check(current_index, ok, 'training checkpoint was created' if ok else f'training did not create {checkpoint_path}')
        current_index += 1
        if not ok:
            raise FileNotFoundError(f'training did not create {checkpoint_path}')
        try:
            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        context['checkpoint_payload'] = payload
        context['train_result'] = train_result
        ok = isinstance(train_result, dict)
        update_check(current_index, ok, 'train returned a dictionary' if ok else 'train must return a dictionary')
        current_index += 1
        ok = isinstance(payload, dict)
        update_check(current_index, ok, 'checkpoint is a dictionary' if ok else 'checkpoint must be a dictionary')
        current_index += 1
        trained_state = payload.get('model_state_dict', {}) if isinstance(payload, dict) else {}
        optimizer_state = payload.get('optimizer_state_dict', {}) if isinstance(payload, dict) else {}
        history = payload.get('history', []) if isinstance(payload, dict) else []
        trainable_names = payload.get('trainable_parameter_names', []) if isinstance(payload, dict) else []
        ok = set(trained_state) == baseline_keys
        update_check(current_index, ok, 'checkpoint state layout is exact' if ok else 'checkpoint model_state_dict layout is not exact')
        current_index += 1
        ok = payload.get('format') == 'mattergen-dft-mag-density-adapter-v1'
        update_check(current_index, ok, 'checkpoint format is correct' if ok else 'checkpoint format is incorrect')
        current_index += 1
        ok = payload.get('completed_steps') == 2
        update_check(current_index, ok, 'checkpoint completed steps are correct' if ok else 'checkpoint completed_steps must be 2')
        current_index += 1
        ok = isinstance(optimizer_state, dict) and bool(optimizer_state.get('state'))
        update_check(current_index, ok, 'Adam optimizer state is non-empty' if ok else 'Adam optimizer state must be non-empty')
        current_index += 1
        expected_trainable = sorted(initial_adapter_state)
        ok = sorted(trainable_names) == expected_trainable
        update_check(current_index, ok, 'checkpoint trainable names identify the adapter' if ok else 'checkpoint trainable_parameter_names must exactly identify the adapter')
        current_index += 1
        changed_adapter = [name for name in expected_trainable if name in trained_state and not torch.equal(initial_adapter_state[name], trained_state[name].detach().cpu())]
        ok = len(changed_adapter) == len(expected_trainable)
        update_check(current_index, ok, 'every adapter tensor changed' if ok else 'every trainable adapter tensor must change after two steps')
        current_index += 1
        pretrained_state = model_module.extract_base_model_state(model_module.load_checkpoint_payload(CHECKPOINT_PATH))
        changed_frozen = [name for name in frozen_parameter_names if name not in pretrained_state or not torch.equal(pretrained_state[name].detach().cpu(), trained_state[name].detach().cpu())]
        ok = not changed_frozen
        update_check(current_index, ok, 'frozen pretrained parameters are unchanged' if ok else f'frozen pretrained parameters changed: {changed_frozen[:8]}')
        current_index += 1
        reload_model = model_module.load_pretrained_model(candidate_config, CHECKPOINT_PATH, device='cpu')
        reload_result = reload_model.load_state_dict(trained_state, strict=True)
        ok = not reload_result.missing_keys and not reload_result.unexpected_keys
        update_check(current_index, ok, 'fine-tuned checkpoint strict-reloads' if ok else 'fine-tuned checkpoint did not strict-reload')
        current_index += 1
        del reload_model
        required_metrics = {'step', 'loss', 'pos_loss', 'cell_loss', 'atomic_numbers_loss', 'gradient_norm', 'parameters_with_grad'}
        ok = isinstance(history, list) and len(history) == 2
        update_check(current_index, ok, 'checkpoint history has two records' if ok else 'checkpoint history must contain exactly two records')
        current_index += 1
        for index in (1, 2):
            metrics = history[index - 1] if isinstance(history, list) and len(history) >= index else None
            ok = isinstance(metrics, dict) and required_metrics <= set(metrics)
            update_check(current_index, ok, f'history record {index} has required keys' if ok else f'history record {index} must contain {sorted(required_metrics)}')
            current_index += 1
            numeric_values = [metrics.get(key) for key in required_metrics - {'step'}] if isinstance(metrics, dict) else []
            ok = isinstance(metrics, dict) and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in numeric_values)
            update_check(current_index, ok, f'history record {index} values are finite' if ok else f'history record {index} must contain finite numbers')
            current_index += 1
            ok = isinstance(metrics, dict) and metrics.get('step') == index
            update_check(current_index, ok, f'history record {index} step is correct' if ok else f'history record {index} has the wrong step')
            current_index += 1
            ok = isinstance(metrics, dict) and float(metrics.get('loss', 0.0)) > 0.0
            update_check(current_index, ok, f'history record {index} loss is positive' if ok else f'history record {index} loss must be positive')
            current_index += 1
            ok = isinstance(metrics, dict) and float(metrics.get('gradient_norm', 0.0)) > 0.0
            update_check(current_index, ok, f'history record {index} gradient norm is positive' if ok else f'history record {index} gradient norm must be positive')
            current_index += 1
            ok = isinstance(metrics, dict) and int(metrics.get('parameters_with_grad', 0)) > 0
            update_check(current_index, ok, f'history record {index} observes adapter gradients' if ok else f'history record {index} must observe adapter gradients')
            current_index += 1
        ok = isinstance(train_result, dict) and train_result.get('history') == history
        update_check(current_index, ok, 'train return history matches checkpoint history' if ok else 'train return history must match checkpoint history')
        current_index += 1
        context['training_history'] = history
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['checkpoint_artifact', 'metrics_artifact', 'log_artifact', 'metrics_jsonl_parse', 'log_jsonl_parse', 'metrics_record_count', 'log_record_count', 'metrics_history_match', 'log_history_match', 'artifact_state_entry_count', 'artifact_state_numel', 'artifact_config_mapping', 'artifact_optimizer_state', 'artifact_trainable_names']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        context = _shared_context(config, caller_globals, caller_locals)
        output_dir: Path = context['training_output_dir']
        payload = context['checkpoint_payload']
        expected_paths = {'checkpoint': output_dir / 'mattergen_property_finetuned.pt', 'metrics': output_dir / 'metrics.jsonl', 'log': output_dir / 'train.log'}
        for label, path in expected_paths.items():
            ok = path.is_file() and path.stat().st_size > 0
            update_check(current_index, ok, f'{label} artifact is non-empty' if ok else f'missing or empty {label} artifact: {path.name}')
            current_index += 1
        metric_records: list[dict[str, Any]] = []
        if expected_paths['metrics'].is_file():
            metric_records = [json.loads(line) for line in expected_paths['metrics'].read_text(encoding='utf-8').splitlines() if line.strip()]
        update_check(current_index, True, 'metrics.jsonl records parsed')
        current_index += 1
        log_records: list[dict[str, Any]] = []
        if expected_paths['log'].is_file():
            log_records = [json.loads(line) for line in expected_paths['log'].read_text(encoding='utf-8').splitlines() if line.strip()]
        update_check(current_index, True, 'train.log records parsed')
        current_index += 1
        checkpoint_history = payload.get('history', []) if isinstance(payload, dict) else []
        ok = len(metric_records) == 2
        update_check(current_index, ok, 'metrics.jsonl has two records' if ok else 'metrics.jsonl must contain exactly two records')
        current_index += 1
        ok = len(log_records) == 2
        update_check(current_index, ok, 'train.log has two records' if ok else 'train.log must contain exactly two JSON records')
        current_index += 1
        ok = metric_records == checkpoint_history
        update_check(current_index, ok, 'metrics.jsonl matches checkpoint history' if ok else 'metrics.jsonl must match checkpoint history')
        current_index += 1
        ok = log_records == checkpoint_history
        update_check(current_index, ok, 'train.log matches checkpoint history' if ok else 'train.log must match checkpoint history')
        current_index += 1
        state = payload.get('model_state_dict', {}) if isinstance(payload, dict) else {}
        ok = len(state) == EXPECTED_STATE_ENTRY_COUNT
        update_check(current_index, ok, 'checkpoint state entry count is correct' if ok else f'checkpoint must contain {EXPECTED_STATE_ENTRY_COUNT} model entries')
        current_index += 1
        ok = sum(value.numel() for value in state.values()) == EXPECTED_STATE_NUMEL
        update_check(current_index, ok, 'checkpoint state element count is correct' if ok else f'checkpoint must contain {EXPECTED_STATE_NUMEL} model elements')
        current_index += 1
        ok = isinstance(payload.get('config'), dict)
        update_check(current_index, ok, 'checkpoint preserves parsed config' if ok else 'checkpoint must preserve parsed config')
        current_index += 1
        ok = isinstance(payload.get('optimizer_state_dict'), dict) and bool(payload['optimizer_state_dict'].get('state'))
        update_check(current_index, ok, 'checkpoint preserves non-empty Adam state' if ok else 'checkpoint must preserve non-empty Adam state')
        current_index += 1
        ok = len(payload.get('trainable_parameter_names', [])) == EXPECTED_ADAPTER_PARAMETER_TENSORS
        update_check(current_index, ok, 'checkpoint preserves adapter names' if ok else 'checkpoint must preserve exact trainable adapter names')
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks
