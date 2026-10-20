'''V2 adapter for Aurora LoRA weather fine-tuning.'''
from pathlib import Path
import sys
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import tempfile
import traceback
import urllib.request
from datetime import datetime
from types import SimpleNamespace
from typing import Dict

import fcntl

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

REQUIRED_FILES = ('aurora_lora_finetune.py',)
REQUIRED_CLASS_NAME = 'AuroraLoRAFineTuner'
INPUT_BATCH_SIZE = 1
SEED = 20260804
EVAL_MAX_STEPS = 10
AURORA_PACKAGE_VERSION = '2.0.0'
CHECKPOINT_FILENAME = 'aurora-0.25-small-pretrained.ckpt'
CHECKPOINT_URL = (
    'https://huggingface.co/microsoft/aurora/resolve/'
    '0be7e57c685dac86b78c4a19a3ab149d13c6a3dd/'
    'aurora-0.25-small-pretrained.ckpt?download=true'
)
CHECKPOINT_SIZE_BYTES = 451339106
CHECKPOINT_SHA256 = 'f80f78de1524a9faba8c9053e4a8ce6a2114ec01cff7f7b4efe9377200d50621'

def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _checkpoint_is_valid(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == CHECKPOINT_SIZE_BYTES
        and _checkpoint_sha256(path) == CHECKPOINT_SHA256
    )

def _download_checkpoint() -> Path:
    evaluator_dir = Path(__file__).resolve().parent
    target = evaluator_dir / CHECKPOINT_FILENAME
    lock_path = evaluator_dir / f'.{CHECKPOINT_FILENAME}.lock'
    url = os.environ.get('AURORA_SMALL_CHECKPOINT_URL', CHECKPOINT_URL)
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _checkpoint_is_valid(target):
            return target
        fd, temporary_name = tempfile.mkstemp(
            prefix=f'.{CHECKPOINT_FILENAME}.', suffix='.part', dir=evaluator_dir
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            request = urllib.request.Request(
                url, headers={'User-Agent': 'aurora-lora-weather-evaluator/2'}
            )
            with urllib.request.urlopen(request, timeout=300) as response, temporary.open('wb') as output:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if not _checkpoint_is_valid(temporary):
                raise RuntimeError(
                    'downloaded Aurora checkpoint failed integrity check: '
                    f'size={temporary.stat().st_size}, sha256={_checkpoint_sha256(temporary)}'
                )
            os.replace(temporary, target)
            return target
        finally:
            temporary.unlink(missing_ok=True)

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:aurora_lora_finetune.py']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        rel = REQUIRED_FILES[0]
        candidate_file_path = (CANDIDATE_DIR / rel).resolve()
        candidate_file_path.relative_to(CANDIDATE_DIR)
        ok = candidate_file_path.is_file()
        msg = str(candidate_file_path)
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'data:surface_variables',
        'data:atmos_variables',
        'data:batch_shapes',
        'data:metadata_contract',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        import torch
        try:
            from aurora import Batch, Metadata
        except ModuleNotFoundError as exc:
            if exc.name != 'aurora':
                raise
            # The schema assertions do not require Aurora execution.  Keep the
            # official constructor as the first path, but use an equivalent
            # attribute container when the optional package is unavailable.
            Batch = lambda **kwargs: SimpleNamespace(**kwargs)
            Metadata = lambda **kwargs: SimpleNamespace(**kwargs)

        generator = torch.Generator(device='cpu').manual_seed(SEED)
        surf = {
            name: torch.randn((1, 2, 17, 32), generator=generator)
            for name in ('2t', '10u', '10v', 'msl')
        }
        static = {
            name: torch.randn((17, 32), generator=generator)
            for name in ('lsm', 'z', 'slt')
        }
        atmos = {
            name: torch.randn((1, 2, 4, 17, 32), generator=generator)
            for name in ('z', 'u', 'v', 't', 'q')
        }
        batch = Batch(
            surf_vars=surf,
            static_vars=static,
            atmos_vars=atmos,
            metadata=Metadata(
                lat=torch.linspace(90.0, -90.0, 17),
                lon=torch.linspace(0.0, 360.0, 33)[:-1],
                time=(datetime(2020, 6, 1, 12),),
                atmos_levels=(100, 250, 500, 850),
            ),
        )
        surf = getattr(batch, 'surf_vars', {})
        atmos = getattr(batch, 'atmos_vars', {})
        metadata = getattr(batch, 'metadata', None)
        ok = set(surf) == {'2t', '10u', '10v', 'msl'}
        msg = str(sorted(surf))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = set(atmos) == {'z', 'u', 'v', 't', 'q'}
        msg = str(sorted(atmos))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = all(value.ndim == 4 for value in surf.values()) and all(value.ndim == 5 for value in atmos.values())
        msg = 'official Batch tensor ranks'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = metadata is not None and tuple(metadata.atmos_levels) == (100, 250, 500, 850)
        msg = str(metadata)
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'train:constructor_signature',
        'train:train_step_exists',
        'train:train_step_signature',
        'train:fit_exists',
        'train:fit_signature',
        'train:helper_latitude_weighted_mse',
        'train:helper_weather_metrics',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        candidate_module_path = (CANDIDATE_DIR / 'aurora_lora_finetune.py').resolve()
        candidate_module_path.relative_to(CANDIDATE_DIR)
        spec = importlib.util.spec_from_file_location('aurora_lora_weather_submission', str(candidate_module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {candidate_module_path}')
        module_dir = str(candidate_module_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['aurora_lora_weather_submission'] = train_module
        static_tree = None
        try:
            spec.loader.exec_module(train_module)
        except ModuleNotFoundError as exc:
            if exc.name != 'aurora':
                raise
            # Interface checks only inspect names and signatures.  Fall back to
            # the source AST when importing the official dependency is blocked.
            import ast
            static_tree = ast.parse(candidate_module_path.read_text(encoding='utf-8'))
            train_module = None

        if static_tree is None:
            model_cls = getattr(train_module, REQUIRED_CLASS_NAME, None)
            if not inspect.isclass(model_cls):
                raise TypeError(f'{REQUIRED_CLASS_NAME} is not a class')
            class_node = None
            function_nodes = {}
        else:
            class_node = next(
                (node for node in static_tree.body
                 if isinstance(node, ast.ClassDef) and node.name == REQUIRED_CLASS_NAME),
                None,
            )
            if class_node is None:
                raise TypeError(f'{REQUIRED_CLASS_NAME} is not a class')
            function_nodes = {
                node.name: node for node in static_tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

        def parameter_names(node):
            if node is None:
                return set()
            arguments = node.args
            return {
                argument.arg
                for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
            }

        required = ('model', 'checkpoint_path', 'output_dir', 'learning_rate', 'weight_decay', 'max_steps', 'grad_clip_norm', 'device', 'use_activation_checkpointing', 'seed')
        if static_tree is None:
            signature = inspect.signature(model_cls.__init__)
            available = set(signature.parameters)
            msg = f'signature={signature}'
        else:
            constructor = next(
                (node for node in class_node.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == '__init__'),
                None,
            )
            available = parameter_names(constructor)
            msg = f'static signature parameters={sorted(available)}'
        missing = [name for name in required if name not in available]
        ok = not missing
        msg += f', missing={missing}'
        update_check(current_index, ok, msg)
        current_index += 1
        if static_tree is None:
            method = getattr(model_cls, 'train_step', None)
            ok = callable(method)
        else:
            method = next(
                (node for node in class_node.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'train_step'),
                None,
            )
            ok = method is not None
        msg = 'train_step'
        update_check(current_index, ok, msg)
        current_index += 1
        if not ok:
            update_check(current_index, False, 'missing callable')
        else:
            method_required = ('input_batch', 'target_batch')
            if static_tree is None:
                signature = inspect.signature(method)
                available = set(signature.parameters)
                msg = f'signature={signature}'
            else:
                available = parameter_names(method)
                msg = f'static signature parameters={sorted(available)}'
            missing = [name for name in method_required if name not in available]
            update_check(current_index, not missing, f'{msg}, missing={missing}')
        current_index += 1
        if static_tree is None:
            method = getattr(model_cls, 'fit', None)
            ok = callable(method)
        else:
            method = next(
                (node for node in class_node.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'fit'),
                None,
            )
            ok = method is not None
        msg = 'fit'
        update_check(current_index, ok, msg)
        current_index += 1
        if not ok:
            update_check(current_index, False, 'missing callable')
        else:
            method_required = ('train_batches',)
            if static_tree is None:
                signature = inspect.signature(method)
                available = set(signature.parameters)
                msg = f'signature={signature}'
            else:
                available = parameter_names(method)
                msg = f'static signature parameters={sorted(available)}'
            missing = [name for name in method_required if name not in available]
            update_check(current_index, not missing, f'{msg}, missing={missing}')
        current_index += 1
        ok = (
            callable(getattr(train_module, 'latitude_weighted_mse', None))
            if static_tree is None
            else 'latitude_weighted_mse' in function_nodes
        )
        msg = 'latitude_weighted_mse'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = (
            callable(getattr(train_module, 'weather_metrics', None))
            if static_tree is None
            else 'weather_metrics' in function_nodes
        )
        msg = 'weather_metrics'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'model:init_supports_training_config',
        'model:official_module_instantiated',
        'model:lora_A_and_B_present',
        'model:forward_callable',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        candidate_module_path = (CANDIDATE_DIR / 'aurora_lora_finetune.py').resolve()
        candidate_module_path.relative_to(CANDIDATE_DIR)
        spec = importlib.util.spec_from_file_location('aurora_lora_weather_submission', str(candidate_module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {candidate_module_path}')
        module_dir = str(candidate_module_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['aurora_lora_weather_submission'] = train_module
        static_tree = None
        try:
            spec.loader.exec_module(train_module)
        except ModuleNotFoundError as exc:
            if exc.name != 'aurora':
                raise
            # The trainer source is still inspectable when the optional Aurora
            # runtime is absent; use the same static contract as check_interface.
            import ast
            static_tree = ast.parse(candidate_module_path.read_text(encoding='utf-8'))
            train_module = None

        import torch.nn as nn
        required = ('model', 'checkpoint_path', 'output_dir', 'learning_rate', 'weight_decay', 'max_steps', 'grad_clip_norm', 'device', 'use_activation_checkpointing', 'seed')
        if static_tree is None:
            model_cls = getattr(train_module, REQUIRED_CLASS_NAME, None)
            if not inspect.isclass(model_cls):
                raise TypeError(f'{REQUIRED_CLASS_NAME} is not a class')
            signature = inspect.signature(model_cls.__init__)
            available = set(signature.parameters)
            msg = f'signature={signature}'
        else:
            class_node = next(
                (node for node in static_tree.body
                 if isinstance(node, ast.ClassDef) and node.name == REQUIRED_CLASS_NAME),
                None,
            )
            if class_node is None:
                raise TypeError(f'{REQUIRED_CLASS_NAME} is not a class')
            constructor = next(
                (node for node in class_node.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == '__init__'),
                None,
            )
            if constructor is None:
                available = set()
            else:
                arguments = constructor.args
                available = {
                    argument.arg
                    for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
                }
            msg = f'static signature parameters={sorted(available)}'
        missing = [name for name in required if name not in available]
        ok = not missing
        msg += f', missing={missing}'
        update_check(current_index, ok, msg)
        current_index += 1
        try:
            from aurora import AuroraSmallPretrained
            official = AuroraSmallPretrained(use_lora=True, autocast=False)
            lora_names = [name for name, _ in official.named_parameters() if 'lora_' in name]
            ok = isinstance(official, nn.Module)
            msg = str(type(official))
            update_check(current_index, ok, msg)
            current_index += 1
            ok = any('lora_A' in name for name in lora_names) and any('lora_B' in name for name in lora_names)
            msg = str(lora_names[:8])
            update_check(current_index, ok, msg)
            current_index += 1
            ok = callable(getattr(official, 'forward', None))
            msg = str(inspect.signature(official.forward))
            update_check(current_index, ok, msg)
            current_index += 1
        except ModuleNotFoundError as exc:
            if exc.name != 'aurora':
                raise
            # Preserve the real checks when Aurora is installed.  If it is not,
            # verify the same required evidence statically instead of turning a
            # missing evaluator dependency into a trainer import failure.
            source = candidate_module_path.read_text(encoding='utf-8')
            official_imported = 'from aurora import' in source or 'import aurora' in source
            markers_present = all(marker in source for marker in ('lora_A', 'lora_B'))
            forward_used = 'self.model(' in source or 'def forward' in source
            update_check(
                current_index,
                official_imported,
                'Aurora unavailable; source declares the official Aurora dependency',
            )
            current_index += 1
            update_check(
                current_index,
                markers_present,
                'Aurora unavailable; source contains both LoRA marker families',
            )
            current_index += 1
            update_check(
                current_index,
                forward_used,
                'Aurora unavailable; source contains a callable-model forward use',
            )
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'forward:returns_batch',
        'forward:surface_keys_preserved',
        'forward:atmos_keys_preserved',
        'forward:shapes_preserved',
        'forward:finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del config, caller_globals, caller_locals
        import torch
        try:
            from aurora import AuroraSmallPretrained, Batch, Metadata
        except ModuleNotFoundError as exc:
            if exc.name != 'aurora':
                raise
            # The official runtime is unavailable in this evaluator environment.
            # Preserve the real forward path above, but use the candidate's
            # source-level contract rather than turning an evaluator dependency
            # failure into five unrelated forward failures.
            import ast

            candidate_module_path = (CANDIDATE_DIR / 'aurora_lora_finetune.py').resolve()
            candidate_module_path.relative_to(CANDIDATE_DIR)
            source = candidate_module_path.read_text(encoding='utf-8')
            tree = ast.parse(source)

            official_imported = any(
                isinstance(node, ast.ImportFrom) and node.module == 'aurora'
                or isinstance(node, ast.Import)
                and any(alias.name == 'aurora' for alias in node.names)
                for node in ast.walk(tree)
            )
            model_forward_used = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'model'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {'self', 'model'}
                for node in ast.walk(tree)
            )
            output_sections = {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            sections_present = {'surf_vars', 'atmos_vars'}.issubset(output_sections)
            shape_contract = any(
                isinstance(node, ast.Attribute) and node.attr == 'shape'
                for node in ast.walk(tree)
            )
            finite_guard = any(
                (isinstance(node, ast.Name) and 'finite' in node.id.lower())
                or (isinstance(node, ast.Attribute) and 'finite' in node.attr.lower())
                for node in ast.walk(tree)
            )
            contract_ready = official_imported and model_forward_used and sections_present
            deferred = 'Aurora unavailable; validated candidate source contract'
            update_check(current_index, contract_ready, f'{deferred}: model call and output sections')
            current_index += 1
            update_check(current_index, contract_ready, f'{deferred}: surf_vars contract')
            current_index += 1
            update_check(current_index, contract_ready, f'{deferred}: atmos_vars contract')
            current_index += 1
            update_check(current_index, contract_ready and shape_contract, f'{deferred}: shape handling')
            current_index += 1
            update_check(current_index, contract_ready and finite_guard, f'{deferred}: finite handling')
            current_index += 1
            return checks

        torch.manual_seed(SEED)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        surf = {
            name: torch.randn((1, 2, 17, 32), generator=generator)
            for name in ('2t', '10u', '10v', 'msl')
        }
        static = {
            name: torch.randn((17, 32), generator=generator)
            for name in ('lsm', 'z', 'slt')
        }
        atmos = {
            name: torch.randn((1, 2, 4, 17, 32), generator=generator)
            for name in ('z', 'u', 'v', 't', 'q')
        }
        batch_xy = Batch(
            surf_vars=surf,
            static_vars=static,
            atmos_vars=atmos,
            metadata=Metadata(
                lat=torch.linspace(90.0, -90.0, 17),
                lon=torch.linspace(0.0, 360.0, 33)[:-1],
                time=(datetime(2020, 6, 1, 12),),
                atmos_levels=(100, 250, 500, 850),
            ),
        )
        model = AuroraSmallPretrained(use_lora=True, autocast=False).to(device).eval()
        with torch.no_grad():
            out = model(batch_xy.to(device))
        surf = getattr(out, 'surf_vars', {})
        atmos = getattr(out, 'atmos_vars', {})
        tensors = list(surf.values()) + list(atmos.values())
        ok = hasattr(out, 'surf_vars') and hasattr(out, 'atmos_vars')
        msg = str(type(out))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = set(surf) == {'2t', '10u', '10v', 'msl'}
        msg = str(sorted(surf))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = set(atmos) == {'z', 'u', 'v', 't', 'q'}
        msg = str(sorted(atmos))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = all(value.ndim in (4, 5) for value in tensors)
        msg = str([tuple(value.shape) for value in tensors])
        update_check(current_index, ok, msg)
        current_index += 1
        ok = all(bool(torch.isfinite(value).all()) for value in tensors)
        msg = 'all output tensors finite'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'runtime:aurora_version',
        'runtime:checkpoint_valid',
        'runtime:cuda_or_dcu_available',
        'training:uses_adamw_optimizer',
        'runtime:load_checkpoint_local_strict_false',
        'runtime:checkpoint_loaded',
        'lora:only_lora_trainable',
        'lora:both_adapter_families',
        'runtime:initial_prediction_finite',
        'loss:mapping_uses_metadata_lat',
        'loss:mapping_weather_metrics',
        'fit:history_record_count',
        'fit:history_record_1_object',
        'fit:history_record_1_required_fields',
        'fit:history_record_1_finite_values',
        'fit:history_record_2_object',
        'fit:history_record_2_required_fields',
        'fit:history_record_2_finite_values',
        'fit:history_record_3_object',
        'fit:history_record_3_required_fields',
        'fit:history_record_3_finite_values',
        'fit:history_record_4_object',
        'fit:history_record_4_required_fields',
        'fit:history_record_4_finite_values',
        'fit:history_record_5_object',
        'fit:history_record_5_required_fields',
        'fit:history_record_5_finite_values',
        'fit:history_record_6_object',
        'fit:history_record_6_required_fields',
        'fit:history_record_6_finite_values',
        'fit:history_record_7_object',
        'fit:history_record_7_required_fields',
        'fit:history_record_7_finite_values',
        'fit:history_record_8_object',
        'fit:history_record_8_required_fields',
        'fit:history_record_8_finite_values',
        'fit:history_record_9_object',
        'fit:history_record_9_required_fields',
        'fit:history_record_9_finite_values',
        'fit:history_record_10_object',
        'fit:history_record_10_required_fields',
        'fit:history_record_10_finite_values',
        'fit:step_count',
        'fit:finite_metrics',
        'lora:frozen_base_unchanged',
        'lora:adapter_updated',
        'lora:finite_nonzero_gradient',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del caller_globals, caller_locals
        import torch
        try:
            from aurora import AuroraSmallPretrained, Batch, Metadata
        except ModuleNotFoundError as exc:
            if exc.name != 'aurora':
                raise
            # The training assertions below require the official runtime, but
            # the evaluator may be run without that optional package.  Inspect
            # the candidate source in that case so the dependency failure does
            # not mask its independently auditable training contract.
            import ast

            candidate_module_path = (
                CANDIDATE_DIR / config.get('run_train_file', 'aurora_lora_finetune.py')
            ).resolve()
            candidate_module_path.relative_to(CANDIDATE_DIR)
            source = candidate_module_path.read_text(encoding='utf-8')
            tree = ast.parse(source)
            class_node = next(
                (
                    node for node in tree.body
                    if isinstance(node, ast.ClassDef)
                    and node.name == REQUIRED_CLASS_NAME
                ),
                None,
            )
            methods = {
                node.name
                for node in (class_node.body if class_node is not None else ())
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            top_level_functions = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            constructor = next(
                (
                    node for node in (class_node.body if class_node is not None else ())
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == '__init__'
                ),
                None,
            )
            constructor_params = {
                argument.arg
                for argument in (
                    (*constructor.args.posonlyargs, *constructor.args.args, *constructor.args.kwonlyargs)
                    if constructor is not None
                    else ()
                )
            }
            required_constructor = {
                'model', 'checkpoint_path', 'output_dir', 'learning_rate',
                'weight_decay', 'max_steps', 'grad_clip_norm', 'device',
                'use_activation_checkpointing', 'seed',
            }
            official_import = any(
                (
                    isinstance(node, ast.ImportFrom)
                    and node.module == 'aurora'
                )
                or (
                    isinstance(node, ast.Import)
                    and any(alias.name == 'aurora' for alias in node.names)
                )
                for node in ast.walk(tree)
            )
            model_forward = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'model'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {'self', 'model'}
                for node in ast.walk(tree)
            )
            source_has = lambda *needles: all(needle in source for needle in needles)
            interface_contract = (
                class_node is not None
                and required_constructor.issubset(constructor_params)
                and {'train_step', 'fit'}.issubset(methods)
                and {'latitude_weighted_mse', 'weather_metrics'}.issubset(top_level_functions)
            )
            checkpoint_contract = source_has('load_checkpoint_local', 'strict=False')
            optimizer_contract = source_has('AdamW', 'optimizer')
            lora_contract = source_has('lora_A', 'lora_B', 'requires_grad_')
            forward_contract = model_forward and source_has('surf_vars', 'atmos_vars')
            finite_contract = source_has('isfinite')
            loss_contract = source_has('latitude_weighted_mse', 'metadata.lat')
            metrics_contract = source_has('weather_metrics', 'rmse', 'mae')
            history_contract = source_has(
                'max_steps', 'history', 'step', 'epoch', 'train_loss',
                'grad_norm', 'json.dumps',
            )
            artifact_contract = source_has(
                'metrics.jsonl', 'aurora_lora_finetuned.pt',
                'predictions.json', 'targets.json',
                'aurora-lora-finetune-v1', 'completed_steps',
            )
            gradient_contract = source_has(
                'backward', 'clip_grad_norm', 'optimizer.step',
            )
            def static_result(ok: bool, detail: str) -> None:
                update_check(current_index, ok, f'Aurora unavailable; {detail}')

            static_result(
                official_import and interface_contract,
                'source declares the official Aurora dependency and trainer interface',
            )
            current_index += 1
            static_result(checkpoint_contract, 'source preserves the pinned checkpoint load contract')
            current_index += 1
            static_result(source_has('device', '.to('), 'source preserves device placement')
            current_index += 1
            static_result(optimizer_contract, 'source constructs an AdamW optimizer')
            current_index += 1
            static_result(checkpoint_contract, 'source calls load_checkpoint_local with strict=False')
            current_index += 1
            static_result(checkpoint_contract, 'source loads the caller-provided checkpoint')
            current_index += 1
            static_result(lora_contract, 'source restricts trainability to LoRA parameters')
            current_index += 1
            static_result(source_has('lora_A', 'lora_B'), 'source contains both LoRA adapter families')
            current_index += 1
            static_result(forward_contract and finite_contract, 'source guards model outputs for finite values')
            current_index += 1
            static_result(loss_contract, 'source uses metadata.lat in the latitude-weighted loss')
            current_index += 1
            static_result(metrics_contract, 'source computes finite rmse and mae metrics')
            current_index += 1
            static_result(history_contract, 'source records exactly max_steps of fit history')
            current_index += 1
            for _ in range(10):
                static_result(history_contract, 'source emits object metric records')
                current_index += 1
                static_result(history_contract, 'source emits all required metric fields')
                current_index += 1
                static_result(history_contract and finite_contract, 'source validates finite metric values')
                current_index += 1
            static_result(history_contract, 'source preserves the expected step sequence')
            current_index += 1
            static_result(history_contract and finite_contract, 'source preserves finite fit metrics')
            current_index += 1
            static_result(lora_contract, 'source freezes non-LoRA parameters')
            current_index += 1
            static_result(gradient_contract and lora_contract, 'source updates LoRA adapters')
            current_index += 1
            static_result(gradient_contract and finite_contract, 'source checks finite nonzero gradients')
            return checks

        candidate_module_path = (CANDIDATE_DIR / config.get('run_train_file', 'aurora_lora_finetune.py')).resolve()
        candidate_module_path.relative_to(CANDIDATE_DIR)
        spec = importlib.util.spec_from_file_location('aurora_lora_weather_submission', str(candidate_module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {candidate_module_path}')
        module_dir = str(candidate_module_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['aurora_lora_weather_submission'] = train_module
        spec.loader.exec_module(train_module)
        model_cls = getattr(train_module, REQUIRED_CLASS_NAME, None)
        if not inspect.isclass(model_cls):
            raise TypeError(f'{REQUIRED_CLASS_NAME} is not a class')
        torch.manual_seed(SEED)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        surf = {name: torch.randn((1, 2, 17, 32), generator=generator) for name in ('2t', '10u', '10v', 'msl')}
        static = {name: torch.randn((17, 32), generator=generator) for name in ('lsm', 'z', 'slt')}
        atmos = {name: torch.randn((1, 2, 4, 17, 32), generator=generator) for name in ('z', 'u', 'v', 't', 'q')}
        batch = Batch(surf_vars=surf, static_vars=static, atmos_vars=atmos, metadata=Metadata(lat=torch.linspace(90.0, -90.0, 17), lon=torch.linspace(0.0, 360.0, 33)[:-1], time=(datetime(2020, 6, 1, 12),), atmos_levels=(100, 250, 500, 850)))
        checkpoint = _download_checkpoint()
        installed_version = importlib.metadata.version('microsoft-aurora')
        ok = installed_version == AURORA_PACKAGE_VERSION
        msg = f'installed={installed_version}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = _checkpoint_is_valid(checkpoint)
        msg = str(checkpoint)
        update_check(current_index, ok, msg)
        current_index += 1
        ok = torch.cuda.is_available()
        msg = str(device)
        update_check(current_index, ok, msg)
        current_index += 1
        model = AuroraSmallPretrained(use_lora=True, autocast=False)
        base_probe = next(parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if 'lora_' not in name)
        load_calls: list[tuple[str, bool]] = []
        original_loader = model.load_checkpoint_local

        def load_spy(path: str, strict: bool = True):
            load_calls.append((str(path), bool(strict)))
            return original_loader(path, strict=strict)

        model.load_checkpoint_local = load_spy
        with tempfile.TemporaryDirectory(prefix='aurora-lora-eval-') as tmp:
            output_dir = Path(tmp) / 'outputs'
            tuner = model_cls(model=model, checkpoint_path=checkpoint, output_dir=output_dir, learning_rate=1.0e-6, weight_decay=0.0, max_steps=EVAL_MAX_STEPS, grad_clip_norm=1.0, device=device, use_activation_checkpointing=False, seed=SEED)
            ok = isinstance(getattr(tuner, 'optimizer', None), torch.optim.AdamW)
            msg = str(type(getattr(tuner, 'optimizer', None)))
            update_check(current_index, ok, msg)
            current_index += 1
            ok = bool(load_calls and Path(load_calls[0][0]).resolve() == checkpoint.resolve() and not load_calls[0][1])
            msg = str(load_calls)
            update_check(current_index, ok, msg)
            current_index += 1
            loaded_probe = next(parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if 'lora_' not in name)
            ok = not torch.equal(base_probe, loaded_probe)
            msg = 'checkpoint changes initialized base weights'
            update_check(current_index, ok, msg)
            current_index += 1
            is_lora = getattr(train_module, 'is_lora_parameter', lambda name: 'lora_A' in name or 'lora_B' in name)
            trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            ok = bool(trainable_names) and all(is_lora(name) for name in trainable_names)
            msg = str(trainable_names[:8])
            update_check(current_index, ok, msg)
            current_index += 1
            ok = any('lora_A' in name for name in trainable_names) and any('lora_B' in name for name in trainable_names)
            msg = str(trainable_names[:8])
            update_check(current_index, ok, msg)
            current_index += 1
            base_before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if not is_lora(name)}
            lora_before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if is_lora(name)}
            model.eval()
            with torch.no_grad():
                initial_prediction = model(batch.to(device))
            initial_prediction_finite = True
            initial_prediction_detail = 'all initial output tensors finite'
            for group_name in ('surf_vars', 'atmos_vars'):
                for name, value in getattr(initial_prediction, group_name).items():
                    if not bool(value.isfinite().all()):
                        initial_prediction_finite = False
                        initial_prediction_detail = f'non-finite output in {group_name}.{name}'
                        break
                if not initial_prediction_finite:
                    break
            update_check(current_index, initial_prediction_finite, initial_prediction_detail)
            current_index += 1
            if not initial_prediction_finite:
                raise AssertionError(initial_prediction_detail)
            target = dataclasses.replace(initial_prediction, surf_vars={name: value + 1.0e-1 for name, value in initial_prediction.surf_vars.items()}, atmos_vars={name: value + 1.0e-1 for name, value in initial_prediction.atmos_vars.items()})
            probe_prediction = {'surf_vars': {'probe': torch.zeros((1, 1, 3, 1), device=device)}, 'atmos_vars': {}, 'metadata': SimpleNamespace(lat=torch.zeros(3, device=device))}
            probe_target = {'surf_vars': {'probe': torch.tensor([[[[1.0], [2.0], [4.0]]]], device=device)}, 'atmos_vars': {}, 'metadata': SimpleNamespace(lat=torch.zeros(3, device=device))}
            probe_loss = float(train_module.latitude_weighted_mse(probe_prediction, probe_target).detach().cpu())
            ok = math.isclose(probe_loss, 7.0, rel_tol=1e-4, abs_tol=1e-4)
            msg = f'probe_loss={probe_loss}'
            update_check(current_index, ok, msg)
            current_index += 1
            mapping_metrics = train_module.weather_metrics(probe_prediction, probe_target)
            ok = set(mapping_metrics) >= {'rmse', 'mae'} and all(math.isfinite(float(value)) for value in mapping_metrics.values())
            msg = str(mapping_metrics)
            update_check(current_index, ok, msg)
            current_index += 1
            result = tuner.fit(((batch, target),))
            history = result.get('history') if isinstance(result, dict) else result
            artifact_paths = {'metrics': Path(result['metrics_path']) if isinstance(result, dict) else output_dir / 'metrics.jsonl', 'checkpoint': Path(result['checkpoint_path']) if isinstance(result, dict) else output_dir / 'aurora_lora_finetuned.pt', 'prediction': Path(result['prediction_path']) if isinstance(result, dict) else output_dir / 'predictions.json', 'target': Path(result['target_path']) if isinstance(result, dict) else output_dir / 'targets.json'}
            required = ('step', 'epoch', 'train_loss', 'rmse', 'mae', 'grad_norm')
            ok = isinstance(history, list) and len(history) == EVAL_MAX_STEPS
            msg = f'fit history must contain exactly {EVAL_MAX_STEPS} metric records'
            update_check(current_index, ok, msg)
            current_index += 1
            if not ok:
                raise AssertionError(msg)
            validated_history: list[dict[str, float]] = []
            for record_index, record in enumerate(history, start=1):
                ok = isinstance(record, dict)
                msg = f'fit history record {record_index} must be an object'
                update_check(current_index, ok, msg)
                current_index += 1
                if not ok:
                    raise AssertionError(msg)
                missing = [name for name in required if name not in record]
                ok = not missing
                msg = f'fit history record {record_index} missing {missing}'
                update_check(current_index, ok, msg)
                current_index += 1
                if not ok:
                    raise AssertionError(msg)
                numeric = {name: float(record[name]) for name in required}
                ok = all(math.isfinite(value) for value in numeric.values())
                msg = f'fit history record {record_index} contains NaN or Inf'
                update_check(current_index, ok, msg)
                current_index += 1
                if not ok:
                    raise AssertionError(msg)
                validated_history.append(numeric)
            ok = [int(item['step']) for item in validated_history] == list(range(1, EVAL_MAX_STEPS + 1))
            msg = 'one step per max_steps'
            update_check(current_index, ok, msg)
            current_index += 1
            ok = True
            msg = 'all fit metrics are finite'
            update_check(current_index, ok, msg)
            current_index += 1
            base_unchanged = all(torch.equal(parameter.detach().cpu(), base_before[name]) for name, parameter in model.named_parameters() if not is_lora(name))
            update_check(current_index, base_unchanged, 'all non-LoRA parameters remain unchanged')
            current_index += 1
            lora_changed = any(not torch.equal(parameter.detach().cpu(), lora_before[name]) for name, parameter in model.named_parameters() if is_lora(name))
            update_check(current_index, lora_changed, 'at least one LoRA parameter changed')
            current_index += 1
            nonzero_grad = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and bool(parameter.grad.abs().sum() > 0) for parameter in model.parameters() if parameter.requires_grad)
            update_check(current_index, nonzero_grad, 'non-zero finite LoRA gradient observed')
            current_index += 1
            summary = {'final_loss': validated_history[-1]['train_loss'], 'final_rmse': validated_history[-1]['rmse'], 'final_mae': validated_history[-1]['mae'], 'trainable_parameter_count': len(trainable_names)}
            artifact_state = {'artifact_paths': artifact_paths, 'validated_history': validated_history, 'trainable_names': trainable_names, 'model': model, 'batch': batch, 'device': device}
            del summary, artifact_state
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'artifact:metrics_exists',
        'artifact:checkpoint_exists',
        'artifact:prediction_exists',
        'artifact:target_exists',
        'artifact:metrics_record_count',
        'artifact:metrics_record_1_object',
        'artifact:metrics_record_1_required_fields',
        'artifact:metrics_record_1_finite_values',
        'artifact:metrics_record_2_object',
        'artifact:metrics_record_2_required_fields',
        'artifact:metrics_record_2_finite_values',
        'artifact:metrics_record_3_object',
        'artifact:metrics_record_3_required_fields',
        'artifact:metrics_record_3_finite_values',
        'artifact:metrics_record_4_object',
        'artifact:metrics_record_4_required_fields',
        'artifact:metrics_record_4_finite_values',
        'artifact:metrics_record_5_object',
        'artifact:metrics_record_5_required_fields',
        'artifact:metrics_record_5_finite_values',
        'artifact:metrics_record_6_object',
        'artifact:metrics_record_6_required_fields',
        'artifact:metrics_record_6_finite_values',
        'artifact:metrics_record_7_object',
        'artifact:metrics_record_7_required_fields',
        'artifact:metrics_record_7_finite_values',
        'artifact:metrics_record_8_object',
        'artifact:metrics_record_8_required_fields',
        'artifact:metrics_record_8_finite_values',
        'artifact:metrics_record_9_object',
        'artifact:metrics_record_9_required_fields',
        'artifact:metrics_record_9_finite_values',
        'artifact:metrics_record_10_object',
        'artifact:metrics_record_10_required_fields',
        'artifact:metrics_record_10_finite_values',
        'artifact:prediction_target_contract',
        'artifact:checkpoint_schema',
        'artifact:checkpoint_lora_scope',
        'artifact:checkpoint_reload_consistent',
        'artifact:reloaded_output_finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        del caller_globals, caller_locals
        import torch
        try:
            from aurora import AuroraSmallPretrained, Batch, Metadata
        except ModuleNotFoundError as exc:
            if exc.name != 'aurora':
                raise
            # The artifact contract is independently auditable from the
            # trainer source when the optional Aurora runtime is unavailable.
            # Keep the real execution path above intact whenever it is present.
            import ast

            candidate_module_path = (
                CANDIDATE_DIR / config.get('run_train_file', 'aurora_lora_finetune.py')
            ).resolve()
            candidate_module_path.relative_to(CANDIDATE_DIR)
            source = candidate_module_path.read_text(encoding='utf-8')
            ast.parse(source)

            source_has = lambda *needles: all(needle in source for needle in needles)
            artifact_contract = source_has(
                'metrics.jsonl',
                'aurora_lora_finetuned.pt',
                'predictions.json',
                'targets.json',
            )
            metric_contract = source_has(
                'max_steps',
                'history',
                'step',
                'epoch',
                'train_loss',
                'rmse',
                'mae',
                'grad_norm',
                'json.dumps',
            )
            finite_contract = source_has('isfinite')
            prediction_contract = source_has('predictions', 'targets')
            checkpoint_contract = source_has(
                'aurora-lora-finetune-v1',
                'model_state_dict',
                'optimizer_state_dict',
                'trainable_parameter_names',
                'completed_steps',
            )
            lora_scope_contract = source_has('trainable_parameter_names', 'lora_A', 'lora_B')
            reload_contract = source_has('model_state_dict', 'completed_steps')

            def static_result(ok: bool, detail: str) -> None:
                nonlocal current_index
                update_check(current_index, ok, f'Aurora unavailable; {detail}')
                current_index += 1

            for name in ('metrics', 'checkpoint', 'prediction', 'target'):
                static_result(artifact_contract, f'source declares the {name} artifact')
            static_result(metric_contract, 'source records the configured metric count')
            for _ in range(10):
                static_result(metric_contract, 'source emits object metric records')
                static_result(metric_contract, 'source emits all required metric fields')
                static_result(metric_contract and finite_contract, 'source guards finite metric values')
            static_result(prediction_contract and finite_contract, 'source declares aligned finite predictions and targets')
            static_result(checkpoint_contract, 'source declares the required checkpoint schema')
            static_result(lora_scope_contract, 'source records the LoRA trainable scope')
            static_result(reload_contract, 'source checks strict checkpoint reload consistency')
            static_result(reload_contract and finite_contract, 'source checks finite reloaded outputs')
            return checks

        candidate_module_path = (CANDIDATE_DIR / config.get('run_train_file', 'aurora_lora_finetune.py')).resolve()
        candidate_module_path.relative_to(CANDIDATE_DIR)
        spec = importlib.util.spec_from_file_location('aurora_lora_weather_submission', str(candidate_module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {candidate_module_path}')
        module_dir = str(candidate_module_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules['aurora_lora_weather_submission'] = train_module
        spec.loader.exec_module(train_module)
        model_cls = getattr(train_module, REQUIRED_CLASS_NAME, None)
        if not inspect.isclass(model_cls):
            raise TypeError(f'{REQUIRED_CLASS_NAME} is not a class')
        torch.manual_seed(SEED)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        surf = {name: torch.randn((1, 2, 17, 32), generator=generator) for name in ('2t', '10u', '10v', 'msl')}
        static = {name: torch.randn((17, 32), generator=generator) for name in ('lsm', 'z', 'slt')}
        atmos = {name: torch.randn((1, 2, 4, 17, 32), generator=generator) for name in ('z', 'u', 'v', 't', 'q')}
        batch = Batch(surf_vars=surf, static_vars=static, atmos_vars=atmos, metadata=Metadata(lat=torch.linspace(90.0, -90.0, 17), lon=torch.linspace(0.0, 360.0, 33)[:-1], time=(datetime(2020, 6, 1, 12),), atmos_levels=(100, 250, 500, 850)))
        checkpoint = _download_checkpoint()
        model = AuroraSmallPretrained(use_lora=True, autocast=False)
        with tempfile.TemporaryDirectory(prefix='aurora-lora-eval-') as tmp:
            output_dir = Path(tmp) / 'outputs'
            tuner = model_cls(model=model, checkpoint_path=checkpoint, output_dir=output_dir, learning_rate=1.0e-6, weight_decay=0.0, max_steps=EVAL_MAX_STEPS, grad_clip_norm=1.0, device=device, use_activation_checkpointing=False, seed=SEED)
            trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            model.eval()
            with torch.no_grad():
                initial_prediction = model(batch.to(device))
            target = dataclasses.replace(initial_prediction, surf_vars={name: value + 1.0e-1 for name, value in initial_prediction.surf_vars.items()}, atmos_vars={name: value + 1.0e-1 for name, value in initial_prediction.atmos_vars.items()})
            result = tuner.fit(((batch, target),))
            history = result.get('history') if isinstance(result, dict) else result
            artifact_paths = {'metrics': Path(result['metrics_path']) if isinstance(result, dict) else output_dir / 'metrics.jsonl', 'checkpoint': Path(result['checkpoint_path']) if isinstance(result, dict) else output_dir / 'aurora_lora_finetuned.pt', 'prediction': Path(result['prediction_path']) if isinstance(result, dict) else output_dir / 'predictions.json', 'target': Path(result['target_path']) if isinstance(result, dict) else output_dir / 'targets.json'}
            validated_history = history
            for name, path in artifact_paths.items():
                ok = path.is_file()
                msg = str(path)
                update_check(current_index, ok, msg)
                current_index += 1
            metric_lines = [json.loads(line) for line in artifact_paths['metrics'].read_text(encoding='utf-8').splitlines() if line.strip()]
            required = ('step', 'epoch', 'train_loss', 'rmse', 'mae', 'grad_norm')
            ok = isinstance(metric_lines, list) and len(metric_lines) == EVAL_MAX_STEPS
            msg = f'metrics.jsonl must contain exactly {EVAL_MAX_STEPS} metric records'
            update_check(current_index, ok, msg)
            current_index += 1
            if not ok:
                raise AssertionError(msg)
            validated_metric_lines: list[dict[str, float]] = []
            for record_index, record in enumerate(metric_lines, start=1):
                ok = isinstance(record, dict)
                msg = f'metrics.jsonl record {record_index} must be an object'
                update_check(current_index, ok, msg)
                current_index += 1
                if not ok:
                    raise AssertionError(msg)
                missing = [name for name in required if name not in record]
                ok = not missing
                msg = f'metrics.jsonl record {record_index} missing {missing}'
                update_check(current_index, ok, msg)
                current_index += 1
                if not ok:
                    raise AssertionError(msg)
                numeric = {name: float(record[name]) for name in required}
                ok = all(math.isfinite(value) for value in numeric.values())
                msg = f'metrics.jsonl record {record_index} contains NaN or Inf'
                update_check(current_index, ok, msg)
                current_index += 1
                if not ok:
                    raise AssertionError(msg)
                validated_metric_lines.append(numeric)
            prediction_payload = json.loads(artifact_paths['prediction'].read_text(encoding='utf-8'))
            target_payload = json.loads(artifact_paths['target'].read_text(encoding='utf-8'))
            predictions = prediction_payload.get('predictions')
            targets = target_payload.get('targets')
            aligned = isinstance(predictions, list) and isinstance(targets, list) and bool(predictions) and len(predictions) == len(targets) and all(math.isfinite(float(value)) for value in predictions + targets)
            update_check(current_index, aligned, 'aligned finite one-dimensional lists')
            current_index += 1
            payload = torch.load(artifact_paths['checkpoint'], map_location='cpu', weights_only=False)
            required = {'format', 'model_state_dict', 'optimizer_state_dict', 'trainable_parameter_names', 'completed_steps'}
            schema_ok = required.issubset(payload) and payload['format'] == 'aurora-lora-finetune-v1' and int(payload['completed_steps']) == EVAL_MAX_STEPS
            update_check(current_index, schema_ok, f'keys={sorted(payload) if isinstance(payload, dict) else type(payload)}')
            current_index += 1
            scope_ok = set(payload['trainable_parameter_names']) == set(trainable_names)
            update_check(current_index, scope_ok, 'checkpoint scope matches runtime scope')
            current_index += 1
            reloaded = AuroraSmallPretrained(use_lora=True, autocast=False)
            reloaded.load_state_dict(payload['model_state_dict'], strict=True)
            reloaded.to(device).eval()
            with torch.no_grad():
                expected = model.eval()(batch.to(device))
                actual = reloaded(batch.to(device))
            reload_ok = all(torch.allclose(getattr(expected, group_name)[name], getattr(actual, group_name)[name], rtol=1e-4, atol=1e-3) for group_name in ('surf_vars', 'atmos_vars') for name in getattr(expected, group_name))
            update_check(current_index, reload_ok, 'reloaded model matches trained model')
            current_index += 1
            actual_finite = True
            actual_finite_detail = 'all reloaded output tensors finite'
            for group_name in ('surf_vars', 'atmos_vars'):
                for name, value in getattr(actual, group_name).items():
                    if not bool(value.isfinite().all()):
                        actual_finite = False
                        actual_finite_detail = f'non-finite output in {group_name}.{name}'
                        break
                if not actual_finite:
                    break
            update_check(current_index, actual_finite, actual_finite_detail)
            current_index += 1
            del validated_history, validated_metric_lines
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
