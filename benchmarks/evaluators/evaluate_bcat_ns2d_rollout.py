'''V2 adapter for the BCAT NS2D rollout legacy evaluator.'''
from __future__ import annotations

import contextlib
import importlib.util
import io
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


CASE_ID = 'bcat_ns2d_rollout'
REQUIRED_FILES = ('model.py', 'data.py', 'train.py', 'inference.py', 'result.py', 'config.yaml')
REQUIRED_CLASS_NAME = 'BCATLite'
EXPECTED_GRID_SIZE = 16
EXPECTED_TOTAL_LEN = 5
EXPECTED_CHANNELS = 3
EXPECTED_PATCH_NUM = 4
EXPECTED_N_HEAD = 4
EXPECTED_N_LAYER = 2
FORWARD_COMPARE_ATOL = 1.0e-6
SEED = 2026


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:model.py', 'file:data.py', 'file:train.py', 'file:inference.py', 'file:result.py', 'file:config.yaml']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        for rel_path in REQUIRED_FILES:
            ok = (CANDIDATE_DIR / rel_path).is_file()
            msg = str(CANDIDATE_DIR / rel_path)
            update_check(current_index, ok, msg)
            current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['model_structure:class_exported']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def import_module_from_path(path: Path, module_name: str):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot import module from {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        for name in ('model', 'data', 'train', 'inference', 'result'):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(CANDIDATE_DIR))
        model_module = import_module_from_path(CANDIDATE_DIR / 'model.py', 'bcat_candidate_model')
        data_module = import_module_from_path(CANDIDATE_DIR / 'data.py', 'bcat_candidate_data')
        train_module = import_module_from_path(CANDIDATE_DIR / 'train.py', 'bcat_candidate_train')
        inference_module = import_module_from_path(CANDIDATE_DIR / 'inference.py', 'bcat_candidate_inference')
        result_module = import_module_from_path(CANDIDATE_DIR / 'result.py', 'bcat_candidate_result')
        import yaml
        with (CANDIDATE_DIR / 'config.yaml').open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        model_cls = getattr(model_module, REQUIRED_CLASS_NAME, None)
        ok = model_cls is not None
        msg = REQUIRED_CLASS_NAME
        update_check(current_index, ok, msg)
        current_index += 1
        if model_cls is None:
            raise AttributeError(f'{REQUIRED_CLASS_NAME} is not exported by model.py')
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['data:grid_size', 'data:total_len', 'data:channels', 'data:channel_names', 'data:has_make_grid', 'data:has_vortex_sequence', 'data:has_batch_builder', 'data:grid_shape', 'data:sequence_shape', 'data:times_shape', 'data:target_shape', 'data:data_mask_shape', 'data:finite_batch']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def import_module_from_path(path: Path, module_name: str):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot import module from {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        sys.path.insert(0, str(CANDIDATE_DIR))
        data_module = import_module_from_path(CANDIDATE_DIR / 'data.py', 'bcat_candidate_data')
        import torch
        import yaml
        with (CANDIDATE_DIR / 'config.yaml').open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        data_cfg = config.get('data', {})
        ok = int(data_cfg.get('grid_size', 0)) == EXPECTED_GRID_SIZE
        msg = str(data_cfg.get('grid_size'))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = int(data_cfg.get('total_len', 0)) == EXPECTED_TOTAL_LEN
        msg = str(data_cfg.get('total_len'))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = int(data_cfg.get('channels', 0)) == EXPECTED_CHANNELS
        msg = str(data_cfg.get('channels'))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = data_cfg.get('channel_names') == ['u', 'v', 'omega']
        msg = str(data_cfg.get('channel_names'))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = hasattr(data_module, 'make_grid')
        msg = 'make_grid'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = hasattr(data_module, 'periodic_vortex_sequence')
        msg = 'periodic_vortex_sequence'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = hasattr(data_module, 'build_ns2d_batch')
        msg = 'build_ns2d_batch'
        update_check(current_index, ok, msg)
        current_index += 1
        grid = data_module.make_grid(EXPECTED_GRID_SIZE)
        ok = isinstance(grid, torch.Tensor) and tuple(grid.shape) == (16, 16, 2)
        msg = f'shape={getattr(grid, "shape", None)}'
        update_check(current_index, ok, msg)
        current_index += 1
        batch = data_module.build_ns2d_batch(batch_size=2, total_len=EXPECTED_TOTAL_LEN, grid_size=EXPECTED_GRID_SIZE, viscosity=1.0e-3, seed=SEED)
        expected_shapes = {'sequence': (2, 5, 16, 16, 3), 'times': (2, 5, 1), 'target': (2, 4, 16, 16, 3), 'data_mask': (1, 1, 1, 1, 3)}
        for key, shape in expected_shapes.items():
            value = batch.get(key)
            ok = isinstance(value, torch.Tensor) and tuple(value.shape) == shape
            msg = f'{key}={getattr(value, "shape", None)}'
            update_check(current_index, ok, msg)
            current_index += 1
        finite = all(torch.isfinite(value).all().item() for value in batch.values() if isinstance(value, torch.Tensor))
        ok = finite
        msg = 'all tensors finite'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['train_script:has_load_config', 'train_script:has_build_model', 'train_script:has_mse_rollout_loss', 'train_script:has_train_one_step', 'train_script:has_main', 'train_script:uses_adamw', 'train_script:prints_loss', 'train_script:saves_checkpoint', 'train_script:checkpoint_path_contract', 'train_script:results_dir_contract', 'train_script:learning_rate_contract']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def import_module_from_path(path: Path, module_name: str):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot import module from {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        sys.path.insert(0, str(CANDIDATE_DIR))
        train_module = import_module_from_path(CANDIDATE_DIR / 'train.py', 'bcat_candidate_train')
        import yaml
        with (CANDIDATE_DIR / 'config.yaml').open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        source = (CANDIDATE_DIR / 'train.py').read_text(encoding='utf-8')
        for name in ('load_config', 'build_model', 'mse_rollout_loss', 'train_one_step', 'main'):
            ok = hasattr(train_module, name)
            msg = name
            update_check(current_index, ok, msg)
            current_index += 1
        ok = 'AdamW' in source
        msg = 'AdamW optimizer'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = 'train_loss' in source and 'print' in source
        msg = 'train_loss print'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = 'torch.save' in source
        msg = 'checkpoint save'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = config.get('experiment', {}).get('checkpoint_path') == 'weight/best_model.pt'
        msg = str(config.get('experiment', {}).get('checkpoint_path'))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = config.get('experiment', {}).get('results_dir') == 'results'
        msg = str(config.get('experiment', {}).get('results_dir'))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = abs(float(config.get('training', {}).get('learning_rate', 0.0)) - 1.0e-3) < 1.0e-12
        msg = str(config.get('training', {}).get('learning_rate'))
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['eval_script:inference_has_main', 'eval_script:result_has_main', 'eval_script:has_relative_l2', 'eval_script:writes_metrics', 'eval_script:writes_predictions', 'eval_script:summarizes_results', 'eval_script:metric_contract']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def import_module_from_path(path: Path, module_name: str):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot import module from {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        sys.path.insert(0, str(CANDIDATE_DIR))
        inference_module = import_module_from_path(CANDIDATE_DIR / 'inference.py', 'bcat_candidate_inference')
        result_module = import_module_from_path(CANDIDATE_DIR / 'result.py', 'bcat_candidate_result')
        import yaml
        with (CANDIDATE_DIR / 'config.yaml').open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        inference_source = (CANDIDATE_DIR / 'inference.py').read_text(encoding='utf-8')
        result_source = (CANDIDATE_DIR / 'result.py').read_text(encoding='utf-8')
        ok = hasattr(inference_module, 'main')
        msg = 'inference.main'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = hasattr(result_module, 'main')
        msg = 'result.main'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = hasattr(inference_module, 'relative_l2')
        msg = 'relative_l2'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = 'metrics.json' in inference_source and 'json' in inference_source
        msg = 'metrics.json'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = 'predictions.npz' in inference_source and 'np.savez' in inference_source
        msg = 'predictions.npz'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = 'summary.json' in result_source
        msg = 'summary.json'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = config.get('inference', {}).get('metric') == 'relative_l2'
        msg = str(config.get('inference', {}).get('metric'))
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['model_structure:class_name', 'model_structure:is_torch_module', 'model_structure:has_patch_embedder', 'model_structure:has_transformer_encoder', 'model_structure:has_block_mask_function', 'model_structure:seq_len_per_step', 'model_structure:patch_num_contract', 'model_structure:attention_heads', 'model_structure:parameter_count_range', 'model_structure:uses_transformer', 'model_structure:mask_shape', 'model_structure:mask_blocks_future']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def import_module_from_path(path: Path, module_name: str):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot import module from {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def instantiate_model(model_cls, config: dict[str, Any]):
            data_cfg = config.get('data', {})
            model_cfg = config.get('model', {})
            return model_cls(grid_size=int(data_cfg.get('grid_size', EXPECTED_GRID_SIZE)), channels=int(data_cfg.get('channels', EXPECTED_CHANNELS)), patch_num=int(model_cfg.get('patch_num', EXPECTED_PATCH_NUM)), dim_emb=int(model_cfg.get('dim_emb', 32)), n_head=int(model_cfg.get('n_head', EXPECTED_N_HEAD)), n_layer=int(model_cfg.get('n_layer', EXPECTED_N_LAYER)), dim_ffn=int(model_cfg.get('dim_ffn', 64)), dropout=float(model_cfg.get('dropout', 0.0)), max_time_len=max(20, int(data_cfg.get('total_len', EXPECTED_TOTAL_LEN))))

        sys.path.insert(0, str(CANDIDATE_DIR))
        model_module = import_module_from_path(CANDIDATE_DIR / 'model.py', 'bcat_candidate_model')
        import torch
        import yaml
        with (CANDIDATE_DIR / 'config.yaml').open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        model_cls = getattr(model_module, REQUIRED_CLASS_NAME)
        source = (CANDIDATE_DIR / 'model.py').read_text(encoding='utf-8')
        model = instantiate_model(model_cls, config)
        attention_modules = [module for module in model.modules() if isinstance(module, torch.nn.MultiheadAttention)]
        param_count = sum(param.numel() for param in model.parameters())
        ok = model_cls.__name__ == REQUIRED_CLASS_NAME
        msg = model_cls.__name__
        update_check(current_index, ok, msg)
        current_index += 1
        ok = isinstance(model, torch.nn.Module)
        msg = str(type(model))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = hasattr(model, 'embedder')
        msg = 'embedder'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = hasattr(model, 'transformer_encoder')
        msg = 'transformer_encoder'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = hasattr(model_module, 'block_lower_triangular_mask')
        msg = 'block_lower_triangular_mask'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = getattr(model, 'seq_len_per_step', None) == EXPECTED_PATCH_NUM**2
        msg = str(getattr(model, 'seq_len_per_step', None))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = getattr(model, 'patch_num', None) == EXPECTED_PATCH_NUM
        msg = str(getattr(model, 'patch_num', None))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = len(attention_modules) >= EXPECTED_N_LAYER and all(module.num_heads == EXPECTED_N_HEAD for module in attention_modules)
        msg = f'attention_count={len(attention_modules)}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = 10000 <= param_count <= 200000
        msg = f'params={param_count}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = 'TransformerEncoder' in source and 'TransformerEncoderLayer' in source
        msg = 'TransformerEncoder'
        update_check(current_index, ok, msg)
        current_index += 1
        mask = model_module.block_lower_triangular_mask(block_size=4, block_num=3, use_float=False)
        ok = tuple(mask.shape) == (12, 12)
        msg = f'shape={tuple(mask.shape)}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = bool(mask[0, 8].item()) and not bool(mask[8, 0].item()) and not bool(mask[0, 3].item())
        msg = 'block causal mask behavior'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['forward_executable:returns_tensor', 'forward_executable:shape', 'forward_executable:finite']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def import_module_from_path(path: Path, module_name: str):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot import module from {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def instantiate_model(model_cls, config: dict[str, Any]):
            data_cfg = config.get('data', {})
            model_cfg = config.get('model', {})
            return model_cls(grid_size=int(data_cfg.get('grid_size', EXPECTED_GRID_SIZE)), channels=int(data_cfg.get('channels', EXPECTED_CHANNELS)), patch_num=int(model_cfg.get('patch_num', EXPECTED_PATCH_NUM)), dim_emb=int(model_cfg.get('dim_emb', 32)), n_head=int(model_cfg.get('n_head', EXPECTED_N_HEAD)), n_layer=int(model_cfg.get('n_layer', EXPECTED_N_LAYER)), dim_ffn=int(model_cfg.get('dim_ffn', 64)), dropout=float(model_cfg.get('dropout', 0.0)), max_time_len=max(20, int(data_cfg.get('total_len', EXPECTED_TOTAL_LEN))))

        sys.path.insert(0, str(CANDIDATE_DIR))
        model_module = import_module_from_path(CANDIDATE_DIR / 'model.py', 'bcat_candidate_model')
        data_module = import_module_from_path(CANDIDATE_DIR / 'data.py', 'bcat_candidate_data')
        import torch
        import yaml
        with (CANDIDATE_DIR / 'config.yaml').open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        model_cls = getattr(model_module, REQUIRED_CLASS_NAME)
        torch.manual_seed(SEED)
        model = instantiate_model(model_cls, config).eval()
        batch = data_module.build_ns2d_batch(batch_size=1, total_len=EXPECTED_TOTAL_LEN, grid_size=EXPECTED_GRID_SIZE, viscosity=1.0e-3, seed=SEED + 5)
        with torch.no_grad():
            pred = model(batch['sequence'], batch['times'], input_len=int(config['data']['input_len']))
        ok = isinstance(pred, torch.Tensor)
        msg = str(type(pred))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = tuple(pred.shape) == (1, 4, 16, 16, 3)
        msg = f'shape={tuple(pred.shape)}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = bool(torch.isfinite(pred).all().item())
        msg = 'all finite'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['forward_equivalence:state_dict_keys', 'forward_equivalence:state_dict_shapes', 'forward_equivalence:same_shape', 'forward_equivalence:same_forward']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def import_module_from_path(path: Path, module_name: str):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot import module from {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def instantiate_model(model_cls, config: dict[str, Any]):
            data_cfg = config.get('data', {})
            model_cfg = config.get('model', {})
            return model_cls(grid_size=int(data_cfg.get('grid_size', EXPECTED_GRID_SIZE)), channels=int(data_cfg.get('channels', EXPECTED_CHANNELS)), patch_num=int(model_cfg.get('patch_num', EXPECTED_PATCH_NUM)), dim_emb=int(model_cfg.get('dim_emb', 32)), n_head=int(model_cfg.get('n_head', EXPECTED_N_HEAD)), n_layer=int(model_cfg.get('n_layer', EXPECTED_N_LAYER)), dim_ffn=int(model_cfg.get('dim_ffn', 64)), dropout=float(model_cfg.get('dropout', 0.0)), max_time_len=max(20, int(data_cfg.get('total_len', EXPECTED_TOTAL_LEN))))

        sys.path.insert(0, str(CANDIDATE_DIR))
        model_module = import_module_from_path(CANDIDATE_DIR / 'model.py', 'bcat_candidate_model')
        data_module = import_module_from_path(CANDIDATE_DIR / 'data.py', 'bcat_candidate_data')
        import torch
        import yaml
        with (CANDIDATE_DIR / 'config.yaml').open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        model_cls = getattr(model_module, REQUIRED_CLASS_NAME)
        standard_model_module = import_module_from_path(REFERENCE_DIR / 'model.py', 'bcat_standard_model')
        standard_cls = getattr(standard_model_module, REQUIRED_CLASS_NAME)
        torch.manual_seed(SEED)
        standard_model = instantiate_model(standard_cls, config).eval()
        torch.manual_seed(SEED)
        candidate_model = instantiate_model(model_cls, config).eval()
        standard_state = standard_model.state_dict()
        candidate_state = candidate_model.state_dict()
        same_keys = list(standard_state.keys()) == list(candidate_state.keys())
        same_shapes = same_keys and all(standard_state[key].shape == candidate_state[key].shape for key in standard_state)
        ok = same_keys
        msg = f'keys_match={same_keys}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = same_shapes
        msg = f'shapes_match={same_shapes}'
        update_check(current_index, ok, msg)
        current_index += 1
        if not same_shapes:
            return checks
        candidate_model.load_state_dict(standard_state, strict=True)
        batch = data_module.build_ns2d_batch(batch_size=1, total_len=EXPECTED_TOTAL_LEN, grid_size=EXPECTED_GRID_SIZE, viscosity=1.0e-3, seed=SEED + 5)
        with torch.no_grad():
            standard_out = standard_model(batch['sequence'], batch['times'], input_len=int(config['data']['input_len']))
            candidate_out = candidate_model(batch['sequence'], batch['times'], input_len=int(config['data']['input_len']))
        same_shape = tuple(standard_out.shape) == tuple(candidate_out.shape)
        max_abs_diff = float((standard_out - candidate_out).abs().max().item()) if same_shape else float('inf')
        ok = same_shape
        msg = f'standard={tuple(standard_out.shape)}, candidate={tuple(candidate_out.shape)}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = max_abs_diff <= FORWARD_COMPARE_ATOL
        msg = f'max_abs_diff={max_abs_diff}, atol={FORWARD_COMPARE_ATOL}'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['traning:single_step_loss_finite', 'traning:checkpoint_created', 'traning:history_created', 'traning:prints_loss']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def import_module_from_path(path: Path, module_name: str):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot import module from {path}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        sys.path.insert(0, str(CANDIDATE_DIR))
        train_module = import_module_from_path(CANDIDATE_DIR / 'train.py', 'bcat_candidate_train')
        import yaml
        with (CANDIDATE_DIR / 'config.yaml').open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        with tempfile.TemporaryDirectory(prefix='bcat_ns2d_eval_') as tmp:
            captured_stdout = io.StringIO()
            with contextlib.redirect_stdout(captured_stdout):
                history = train_module.train_one_step(config, Path(tmp))
            train_loss = float(history.get('train_loss', float('nan')))
            checkpoint = Path(tmp) / config['experiment']['checkpoint_path']
            history_path = Path(tmp) / config['experiment']['results_dir'] / 'train_history.json'
            ok = math.isfinite(train_loss)
            msg = f'train_loss={train_loss}'
            update_check(current_index, ok, msg)
            current_index += 1
            ok = checkpoint.is_file()
            msg = str(checkpoint)
            update_check(current_index, ok, msg)
            current_index += 1
            ok = history_path.is_file()
            msg = str(history_path)
            update_check(current_index, ok, msg)
            current_index += 1
            ok = 'train_loss' in captured_stdout.getvalue()
            msg = 'train_one_step printed loss'
            update_check(current_index, ok, msg)
            current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['artifacts:checkpoint_path', 'artifacts:results_dir']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import yaml
        with (CANDIDATE_DIR / 'config.yaml').open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        ok = config.get('experiment', {}).get('checkpoint_path') == 'weight/best_model.pt'
        msg = str(config.get('experiment', {}).get('checkpoint_path'))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = config.get('experiment', {}).get('results_dir') == 'results'
        msg = str(config.get('experiment', {}).get('results_dir'))
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks
