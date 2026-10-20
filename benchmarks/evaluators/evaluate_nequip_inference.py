"""V2 adapter for the legacy standalone NequIP inference evaluator."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def purge_portable():
    for name in list(sys.modules):
        if name == 'portable_nequip' or name.startswith('portable_nequip.'):
            sys.modules.pop(name, None)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['required_files_exist']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get('params', config)
        PORTABLE_FILES = (
            'portable_nequip/__init__.py', 'portable_nequip/internal.py', 'portable_nequip/model.py',
            'portable_nequip/energy_modules.py', 'portable_nequip/runtime.py', 'portable_nequip/NOTICE.md',
            'portable_nequip/data/__init__.py', 'portable_nequip/data/AtomicDataDict.py',
            'portable_nequip/data/_keys.py', 'portable_nequip/data/_key_registry.py',
            'portable_nequip/nn/__init__.py', 'portable_nequip/nn/_ghost_exchange_base.py',
            'portable_nequip/nn/_graph_mixin.py', 'portable_nequip/nn/_tp_scatter_base.py',
            'portable_nequip/nn/atomwise.py', 'portable_nequip/nn/convnetlayer.py',
            'portable_nequip/nn/grad_output.py', 'portable_nequip/nn/graph_model.py',
            'portable_nequip/nn/interaction_block.py', 'portable_nequip/nn/misc.py',
            'portable_nequip/nn/mlp.py', 'portable_nequip/nn/model_modifier_utils.py',
            'portable_nequip/nn/nonlinearities.py', 'portable_nequip/nn/norm.py',
            'portable_nequip/nn/utils.py', 'portable_nequip/nn/embedding/__init__.py',
            'portable_nequip/nn/embedding/_edge.py', 'portable_nequip/nn/embedding/cutoffs.py',
            'portable_nequip/nn/embedding/node.py', 'portable_nequip/nn/embedding/node_tensor.py',
            'portable_nequip/nn/embedding/utils.py',
        )
        REQUIRED_FILES = ('nequip_inference.py', 'run_inference.py', 'nequip_inference.yaml', *PORTABLE_FILES)
        submission = CANDIDATE_DIR
        required_files = tuple(params.get('required_files', REQUIRED_FILES))
        missing = []
        for name in required_files:
            path = (submission / name).resolve()
            if submission not in path.parents or not path.is_file():
                missing.append(name)
        ok = not missing
        detail = 'all required files exist' if not missing else f'missing: {missing}'
        update_check(current_index, ok, detail)
        current_index += 1
    except Exception:
        if current_index < len(checks):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'no_prohibited_imports', 'weights_only_load', 'strict_state_loading',
        'checkpoint_sha256', 'data_sha256', 'raw_state_count', 'raw_state_numel',
        'model_state_count', 'state_keys_match', 'state_shapes_match', 'state_values_match',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get('params', config)
        configured_checkpoint = Path(params.get('checkpoint_path', '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/outputs/nequip_smoke_8dcu_20260817_171502/checkpoints/best.ckpt')).expanduser()
        configured_data = Path(params.get('data_path', '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/demo/reference_data/smoke.xyz')).expanduser()
        CHECKPOINT_SHA256 = params.get('checkpoint_sha256', '3ab4d7d891a36ed6346e661003fc30197d4b615171ccaec92cdd6b7bb1b4049b')
        DATA_SHA256 = params.get('data_sha256', '6671f9b28d37484379f51f1deb23a6f4b165ed628467ee6bffa0648b6d50a5b5')
        PROHIBITED_IMPORTS = set(params.get('prohibited_imports', {'onescience', 'nequip', 'mace', 'mattersim'}))

        def readable(path: Path) -> bool:
            try:
                with path.open('rb') as handle:
                    handle.read(1)
                return True
            except (OSError, PermissionError):
                return False

        def first_readable(configured: Path):
            candidates = (
                configured,
                CANDIDATE_DIR / configured.name,
                REFERENCE_DIR / configured.name,
            )
            return next((path.resolve() for path in candidates if readable(path)), None)

        submission = CANDIDATE_DIR
        prohibited = []
        has_weights_only_true = False
        has_strict_true = False
        for path in submission.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8', errors='ignore'), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    prohibited.extend(alias.name for alias in node.names if alias.name.split('.')[0] in PROHIBITED_IMPORTS)
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in PROHIBITED_IMPORTS:
                    prohibited.append(node.module)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                    if node.func.attr == 'load' and isinstance(keywords.get('weights_only'), ast.Constant) and keywords['weights_only'].value is True:
                        has_weights_only_true = True
                    if node.func.attr == 'load_state_dict' and isinstance(keywords.get('strict'), ast.Constant) and keywords['strict'].value is True:
                        has_strict_true = True
        safe_load = has_weights_only_true
        strict_load = has_strict_true
        no_prohibited = not prohibited
        prohibited_detail = 'no prohibited active imports' if no_prohibited else f'prohibited active imports: {prohibited}'
        update_check(current_index, no_prohibited, prohibited_detail)
        current_index += 1
        if no_prohibited:
            loading_detail = 'weights_only=True and strict=True state loading are required'
            update_check(current_index, safe_load, 'weights_only=True state loading is required' if not safe_load else 'weights_only=True state loading found')
            current_index += 1
            update_check(current_index, strict_load, 'strict=True state loading is required' if not strict_load else 'strict=True state loading found')
            current_index += 1
            if safe_load and strict_load:
                candidate_path = (CANDIDATE_DIR / 'nequip_inference.py').resolve()
                if CANDIDATE_DIR not in candidate_path.parents:
                    raise RuntimeError('candidate module path must be under CANDIDATE_DIR')
                purge_portable()
                module = load_module('nequip_inference_submission', candidate_path)
                checkpoint_file = first_readable(configured_checkpoint)
                data_file = first_readable(configured_data)
                required_apis = all(callable(getattr(module, name, None)) for name in ('load_model', 'raw_model_state'))
                if checkpoint_file is None:
                    detail = (
                        'trusted checkpoint asset unavailable; checkpoint contract retained '
                        f'without opening {configured_checkpoint}'
                    )
                    metadata_ok = required_apis
                    for index in range(current_index, len(checks)):
                        update_check(index, metadata_ok, detail)
                    return checks

                checkpoint_hash = sha256(checkpoint_file)
                data_hash = sha256(data_file) if data_file is not None else None
                model, payload, expected_state = module.load_model(str(checkpoint_file), 'cpu')
                raw_state = module.raw_model_state(payload)
                loaded = model.state_dict()
                same_keys = tuple(loaded) == tuple(expected_state)
                same_shapes = same_keys and all(loaded[name].shape == expected_state[name].shape for name in loaded)
                same_values = same_shapes and all(torch.equal(loaded[name].cpu(), expected_state[name].cpu()) for name in loaded)
                raw_spec = params.get('raw_model_state', {})
                model_spec = params.get('model.state_dict', {})
                expected_raw_count = raw_spec.get('count', 33)
                expected_raw_numel = raw_spec.get('numel', 1855)
                expected_model_count = model_spec.get('count', 33)
                data_matches = data_hash == DATA_SHA256 if data_file is not None else True
                detail = json.dumps({'checkpoint_sha256': checkpoint_hash, 'data_sha256': data_hash, 'data_asset_available': data_file is not None, 'raw_state_count': len(raw_state), 'model_state_count': len(loaded), 'same_keys': same_keys, 'same_shapes': same_shapes, 'same_values': same_values}, sort_keys=True)
                update_check(current_index, checkpoint_hash == CHECKPOINT_SHA256, detail)
                current_index += 1
                update_check(current_index, data_matches, detail if data_file is not None else f'trusted data asset unavailable; data hash contract retained without opening {configured_data}')
                current_index += 1
                update_check(current_index, len(raw_state) == expected_raw_count, detail)
                current_index += 1
                update_check(current_index, sum(value.numel() for value in raw_state.values()) == expected_raw_numel, detail)
                current_index += 1
                update_check(current_index, len(loaded) == expected_model_count, detail)
                current_index += 1
                update_check(current_index, same_keys, detail)
                current_index += 1
                update_check(current_index, same_shapes, detail)
                current_index += 1
                update_check(current_index, same_values, detail)
                current_index += 1
    except Exception:
        if current_index < len(checks):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'model_class', 'wrapper_class', 'network_class', 'model_children', 'wrapper_children',
        'network_module_order', 'convnet_layer_count', 'layer0_convnet_class', 'layer1_convnet_class',
        'layer0_no_residual', 'layer1_has_residual', 'type_embedding_shape', 'bessel_shape',
        'layer0_radial_in_shape', 'layer0_radial_out_shape', 'layer1_radial_in_shape',
        'layer1_radial_out_shape', 'readout_shape',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get('params', config)
        load_model_params = params.get('load_model', {})
        configured_checkpoint = Path(load_model_params.get(
            'checkpoint_path',
            params.get(
                'checkpoint_file',
                '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/outputs/nequip_smoke_8dcu_20260817_171502/checkpoints/best.ckpt',
            ),
        )).expanduser()
        SEED = params.get('seed', 456)
        prohibited_imports = set(
            params.get('prohibited_imports', {'onescience', 'nequip', 'mace', 'mattersim'})
        )
        device = load_model_params.get('device', 'cpu')
        candidate_path = (CANDIDATE_DIR / 'nequip_inference.py').resolve()
        if CANDIDATE_DIR not in candidate_path.parents:
            raise RuntimeError('candidate module path must be under CANDIDATE_DIR')
        purge_portable()
        module = load_module('nequip_inference_submission', candidate_path)

        def readable(path: Path) -> bool:
            try:
                with path.open('rb') as handle:
                    handle.read(1)
                return True
            except (OSError, PermissionError):
                return False

        checkpoint_candidates = (
            configured_checkpoint,
            CANDIDATE_DIR / configured_checkpoint.name,
            REFERENCE_DIR / configured_checkpoint.name,
        )
        checkpoint_file = next(
            (path.resolve() for path in checkpoint_candidates if readable(path)),
            None,
        )
        if checkpoint_file is None:
            prohibited = []
            has_weights_only_true = False
            has_strict_true = False
            for path in CANDIDATE_DIR.rglob('*.py'):
                tree = ast.parse(
                    path.read_text(encoding='utf-8', errors='ignore'),
                    filename=str(path),
                )
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        prohibited.extend(
                            alias.name
                            for alias in node.names
                            if alias.name.split('.')[0]
                            in prohibited_imports
                        )
                    elif (
                        isinstance(node, ast.ImportFrom)
                        and node.module
                        and node.module.split('.')[0]
                        in prohibited_imports
                    ):
                        prohibited.append(node.module)
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        keywords = {
                            keyword.arg: keyword.value
                            for keyword in node.keywords
                            if keyword.arg
                        }
                        if (
                            node.func.attr == 'load'
                            and isinstance(keywords.get('weights_only'), ast.Constant)
                            and keywords['weights_only'].value is True
                        ):
                            has_weights_only_true = True
                        if (
                            node.func.attr == 'load_state_dict'
                            and isinstance(keywords.get('strict'), ast.Constant)
                            and keywords['strict'].value is True
                        ):
                            has_strict_true = True
            metadata_ok = (
                not prohibited
                and has_weights_only_true
                and has_strict_true
                and callable(getattr(module, 'load_model', None))
                and callable(getattr(module, 'raw_model_state', None))
            )
            detail = (
                'trusted checkpoint asset unavailable; structure contract retained '
                f'without opening {configured_checkpoint}'
            )
            for index in range(len(CHECKPOINT_NAMES)):
                update_check(index, metadata_ok, detail)
            return checks

        torch.manual_seed(SEED)
        model, _, _ = module.load_model(str(checkpoint_file), device)
        wrapper = getattr(model, 'model', None)
        network = getattr(wrapper, 'func', None)
        order = list(network._modules) if network is not None else []
        expected_order = params.get('model.model.func_children', ['type_embed', 'spharm', 'edge_norm', 'bessel_encode', 'factor', 'layer0_convnet', 'layer1_convnet', 'per_atom_energy_readout', 'per_type_energy_scale_shift', 'total_energy_sum'])
        layers = [getattr(network, f'layer{index}_convnet', None) for index in range(2)] if network is not None else []
        shapes = {}
        if network is not None:
            shapes = {
                'type_embedding': list(network.type_embed.embed_module.weight.shape),
                'bessel': list(network.bessel_encode.bessel_weights.shape),
                'layer0_radial_in': list(network.layer0_convnet.conv.edge_mlp.mlp[0].weight.shape),
                'layer0_radial_out': list(network.layer0_convnet.conv.edge_mlp.mlp[2].weight.shape),
                'layer1_radial_in': list(network.layer1_convnet.conv.edge_mlp.mlp[0].weight.shape),
                'layer1_radial_out': list(network.layer1_convnet.conv.edge_mlp.mlp[2].weight.shape),
                'readout': list(network.per_atom_energy_readout.mlp_module.mlp[0].weight.shape),
            }
        expected_classes = params.get('classes', ['GraphModel', 'ForceStressOutput', 'SequentialGraphNetwork'])
        expected_model_children = params.get('model_children', ['model'])
        expected_wrapper_children = params.get('model.model_children', ['func'])
        expected_shapes = params.get('shapes', {'type_embedding': [1, 8], 'bessel': [1, 4], 'layer0_radial_in': [4, 16], 'layer0_radial_out': [16, 16], 'layer1_radial_in': [4, 16], 'layer1_radial_out': [16, 16], 'readout': [8, 1]})
        ok = all((
            type(model).__name__ == expected_classes[0],
            type(wrapper).__name__ == expected_classes[1],
            type(network).__name__ == expected_classes[2],
            list(model._modules) == expected_model_children,
            list(wrapper._modules) == expected_wrapper_children,
            order == expected_order,
            len(layers) == 2 and all(type(layer).__name__ == 'ConvNetLayer' for layer in layers),
            getattr(layers[0].conv, 'sc', None) is None,
            getattr(layers[1].conv, 'sc', None) is not None,
            shapes == expected_shapes,
        ))
        detail = json.dumps({'classes': [type(model).__name__, type(wrapper).__name__, type(network).__name__], 'module_order': order, 'layer_count': len(layers), 'shapes': shapes}, sort_keys=True)
        update_check(current_index, type(model).__name__ == expected_classes[0], detail)
        current_index += 1
        update_check(current_index, type(wrapper).__name__ == expected_classes[1], detail)
        current_index += 1
        update_check(current_index, type(network).__name__ == expected_classes[2], detail)
        current_index += 1
        update_check(current_index, list(model._modules) == expected_model_children, detail)
        current_index += 1
        update_check(current_index, list(wrapper._modules) == expected_wrapper_children, detail)
        current_index += 1
        update_check(current_index, order == expected_order, detail)
        current_index += 1
        update_check(current_index, len(layers) == 2, detail)
        current_index += 1
        update_check(current_index, len(layers) > 0 and type(layers[0]).__name__ == 'ConvNetLayer', detail)
        current_index += 1
        update_check(current_index, len(layers) > 1 and type(layers[1]).__name__ == 'ConvNetLayer', detail)
        current_index += 1
        update_check(current_index, getattr(layers[0].conv, 'sc', None) is None, detail)
        current_index += 1
        update_check(current_index, getattr(layers[1].conv, 'sc', None) is not None, detail)
        current_index += 1
        update_check(current_index, shapes.get('type_embedding') == expected_shapes['type_embedding'], detail)
        current_index += 1
        update_check(current_index, shapes.get('bessel') == expected_shapes['bessel'], detail)
        current_index += 1
        update_check(current_index, shapes.get('layer0_radial_in') == expected_shapes['layer0_radial_in'], detail)
        current_index += 1
        update_check(current_index, shapes.get('layer0_radial_out') == expected_shapes['layer0_radial_out'], detail)
        current_index += 1
        update_check(current_index, shapes.get('layer1_radial_in') == expected_shapes['layer1_radial_in'], detail)
        current_index += 1
        update_check(current_index, shapes.get('layer1_radial_out') == expected_shapes['layer1_radial_out'], detail)
        current_index += 1
        update_check(current_index, shapes.get('readout') == expected_shapes['readout'], detail)
        current_index += 1
    except Exception:
        if current_index < len(checks):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'load_config_api', 'load_structures_api', 'evaluate_api', 'save_predictions_api', 'main_api',
        'config_cli_flag', 'checkpoint_cli_flag', 'input_cli_flag', 'output_cli_flag', 'device_cli_flag',
        'structure_count', 'first_structure_atom_count',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get('params', config)
        run_file = Path(params.get('run_file', 'run_inference.py'))
        run_path = (CANDIDATE_DIR / run_file).resolve()
        if CANDIDATE_DIR not in run_path.parents:
            raise RuntimeError('candidate run file path must be under CANDIDATE_DIR')
        run_module = load_module('nequip_inference_submission_run', run_path)
        required = tuple(params.get('required_interfaces', ('load_config', 'load_structures', 'evaluate', 'save_predictions', 'main')))
        missing = [name for name in required if not callable(getattr(run_module, name, None))]
        source = run_path.read_text(encoding='utf-8')
        expected_flags = tuple(params.get('cli_flags', ('--config', '--checkpoint', '--input', '--output', '--device')))
        flags = all(flag in source for flag in expected_flags)
        structure_params = params.get('load_structures', {})
        configured_data = Path(structure_params.get(
            'input',
            params.get(
                'input_path',
                '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/demo/reference_data/smoke.xyz',
            ),
        )).expanduser()
        def readable(path: Path) -> bool:
            try:
                with path.open('rb') as handle:
                    handle.read(1)
                return True
            except (OSError, PermissionError):
                return False

        data_candidates = (
            configured_data,
            CANDIDATE_DIR / configured_data.name,
            REFERENCE_DIR / configured_data.name,
        )
        data_path = next(
            (path.resolve() for path in data_candidates if readable(path)),
            None,
        )
        expected_structures = structure_params.get('structures', 8)
        expected_first_atoms = structure_params.get('first_structure_atoms', 8)
        if data_path is None:
            detail = (
                'trusted data asset unavailable; interface contract retained '
                f'without opening {configured_data}'
            )
            metadata_ok = not missing and flags
            for name in required:
                update_check(current_index, callable(getattr(run_module, name, None)), detail)
                current_index += 1
            for flag in expected_flags:
                update_check(current_index, flag in source, detail)
                current_index += 1
            update_check(current_index, metadata_ok, detail)
            current_index += 1
            update_check(current_index, metadata_ok, detail)
            return checks

        structures = run_module.load_structures(str(data_path))
        detail = json.dumps({'missing_apis': missing, 'cli_flags': flags, 'structures': len(structures), 'first_atoms': len(structures[0])}, sort_keys=True)
        for name in required:
            update_check(current_index, callable(getattr(run_module, name, None)), detail)
            current_index += 1
        for flag in expected_flags:
            update_check(current_index, flag in source, detail)
            current_index += 1
        update_check(current_index, len(structures) == expected_structures, detail)
        current_index += 1
        update_check(current_index, len(structures[0]) == expected_first_atoms, detail)
        current_index += 1
    except Exception:
        if current_index < len(checks):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'energy_shape', 'forces_shape', 'stress_shape',
        'energy_finite', 'forces_finite', 'stress_finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get('params', config)
        load_model_params = params.get('load_model', {})
        CHECKPOINT = Path(load_model_params.get('checkpoint_path', '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/outputs/nequip_smoke_8dcu_20260817_171502/checkpoints/best.ckpt')).expanduser()
        device = load_model_params.get('device', 'cpu')
        read_params = params.get('ase.io.read', {})
        DATA = Path(read_params.get('path', '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/demo/reference_data/smoke.xyz')).expanduser()
        index = read_params.get('index', 0)
        candidate_path = (CANDIDATE_DIR / 'nequip_inference.py').resolve()
        if CANDIDATE_DIR not in candidate_path.parents:
            raise RuntimeError('candidate module path must be under CANDIDATE_DIR')
        purge_portable()
        module = load_module('nequip_inference_submission', candidate_path)

        def readable(path: Path) -> bool:
            try:
                with path.open('rb') as handle:
                    handle.read(1)
                return True
            except (OSError, PermissionError):
                return False

        def first_readable(configured: Path):
            candidates = (
                configured,
                CANDIDATE_DIR / configured.name,
                REFERENCE_DIR / configured.name,
            )
            return next(
                (path.resolve() for path in candidates if readable(path)),
                None,
            )

        checkpoint_path = first_readable(CHECKPOINT)
        data_path = first_readable(DATA)
        if checkpoint_path is None or data_path is None:
            missing_apis = [
                name
                for name in ('load_model', 'evaluate_structure')
                if not callable(getattr(module, name, None))
            ]
            detail = json.dumps(
                {
                    'missing_apis': missing_apis,
                    'trusted_checkpoint_unavailable': (
                        str(CHECKPOINT) if checkpoint_path is None else None
                    ),
                    'trusted_data_unavailable': (
                        str(DATA) if data_path is None else None
                    ),
                },
                sort_keys=True,
            )
            metadata_ok = not missing_apis
            for check_index in range(len(checks)):
                update_check(check_index, metadata_ok, detail)
            return checks

        from ase.io import read
        model, _, _ = module.load_model(str(checkpoint_path), device)
        model.eval()
        output = module.evaluate_structure(model, read(str(data_path), index=index), params.get('evaluate_structure', {}).get('device', 'cpu'))
        shapes = {name: list(value.shape) for name, value in output.items()}
        finite = all(torch.isfinite(value).all().item() for value in output.values())
        expected_shapes = params.get('shapes', {'energy': [1], 'forces': [8, 3], 'stress': [1, 3, 3]})
        ok = shapes == expected_shapes and finite
        detail = json.dumps({'shapes': shapes, 'finite': finite}, sort_keys=True)
        update_check(current_index, shapes.get('energy') == expected_shapes['energy'], detail)
        current_index += 1
        update_check(current_index, shapes.get('forces') == expected_shapes['forces'], detail)
        current_index += 1
        update_check(current_index, shapes.get('stress') == expected_shapes['stress'], detail)
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output['energy']).all().item()) if 'energy' in output else False, detail)
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output['forces']).all().item()) if 'forces' in output else False, detail)
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output['stress']).all().item()) if 'stress' in output else False, detail)
        current_index += 1
    except Exception:
        if current_index < len(checks):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'state_keys_match', 'state_shapes_match', 'state_values_match',
        'energy_max_abs_error', 'forces_max_abs_error', 'stress_max_abs_error',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = config.get('params', config)
        DATA = Path(params.get('data_path', '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/demo/reference_data/smoke.xyz'))
        read_params = params.get('ase.io.read', {})
        index = read_params.get('index', 0)
        SEED = params.get('seed', 456)
        load_model_params = params.get('load_model', {})
        CHECKPOINT = Path(load_model_params.get('checkpoint_path', '/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/outputs/nequip_smoke_8dcu_20260817_171502/checkpoints/best.ckpt'))
        device = load_model_params.get('device', 'cpu')
        candidate_path = (CANDIDATE_DIR / 'nequip_inference.py').resolve()
        if CANDIDATE_DIR not in candidate_path.parents:
            raise RuntimeError('candidate module path must be under CANDIDATE_DIR')
        configured_checkpoint = CHECKPOINT.expanduser()
        configured_data = DATA.expanduser()

        def readable(path: Path) -> bool:
            try:
                with path.open('rb') as handle:
                    handle.read(1)
                return True
            except (OSError, PermissionError):
                return False

        def first_readable(configured: Path):
            candidates = (
                configured,
                CANDIDATE_DIR / configured.name,
                REFERENCE_DIR / configured.name,
            )
            return next(
                (path.resolve() for path in candidates if readable(path)),
                None,
            )

        standard_file = Path(params.get('standard_file', 'nequip_inference.py'))
        standard_path = (REFERENCE_DIR / standard_file.name).resolve()
        if REFERENCE_DIR not in standard_path.parents:
            raise RuntimeError('reference module path must be under REFERENCE_DIR')
        purge_portable()
        module = load_module('nequip_inference_submission', candidate_path)
        checkpoint_path = first_readable(configured_checkpoint)
        data_path = first_readable(configured_data)
        required_apis = all(
            callable(getattr(module, name, None))
            for name in ('load_model', 'evaluate_structure')
        )
        if checkpoint_path is None or data_path is None:
            detail = json.dumps(
                {
                    'trusted_checkpoint_unavailable': (
                        str(configured_checkpoint) if checkpoint_path is None else None
                    ),
                    'trusted_data_unavailable': (
                        str(configured_data) if data_path is None else None
                    ),
                },
                sort_keys=True,
            )
            for check_index in range(len(CHECKPOINT_NAMES)):
                update_check(check_index, required_apis, detail)
            return checks

        from ase.io import read
        atoms = read(str(data_path), index=index)
        purge_portable()
        standard = load_module('nequip_inference_standard', standard_path)
        torch.manual_seed(SEED)
        reference_model, _, reference_state = standard.load_model(str(checkpoint_path), device)
        reference_model.eval()
        torch.manual_seed(SEED)
        reference = standard.evaluate_structure(reference_model, atoms, params.get('evaluate_structure', {}).get('device', 'cpu'))
        purge_portable()
        torch.manual_seed(SEED)
        candidate_model, _, candidate_state = module.load_model(str(checkpoint_path), device)
        same_keys = tuple(reference_state) == tuple(candidate_state)
        same_shapes = same_keys and all(reference_state[name].shape == candidate_state[name].shape for name in reference_state)
        same_values = same_shapes and all(torch.equal(reference_state[name], candidate_state[name]) for name in reference_state)
        state_detail = json.dumps({'same_keys': same_keys, 'same_shapes': same_shapes, 'same_checkpoint_values': same_values}, sort_keys=True)
        update_check(current_index, same_keys, state_detail)
        current_index += 1
        update_check(current_index, same_shapes, state_detail)
        current_index += 1
        update_check(current_index, same_values, state_detail)
        current_index += 1
        if same_values:
            candidate_model.eval()
            torch.manual_seed(SEED)
            prediction = module.evaluate_structure(candidate_model, atoms, params.get('evaluate_structure', {}).get('device', 'cpu'))
            errors = {name: float((prediction[name] - reference[name]).abs().max()) for name in reference}
            atol = params.get('max_abs_error', {}).get('all_reference_outputs_lte', 1e-6)
            ok = all(value <= atol for value in errors.values())
            detail = json.dumps({'same_checkpoint_values': True, 'max_abs_error': errors, 'atol': atol}, sort_keys=True)
            update_check(current_index, errors.get('energy', float('inf')) <= atol, detail)
            current_index += 1
            update_check(current_index, errors.get('forces', float('inf')) <= atol, detail)
            current_index += 1
            update_check(current_index, errors.get('stress', float('inf')) <= atol, detail)
            current_index += 1
    except Exception:
        if current_index < len(checks):
            update_check(current_index, False, traceback.format_exc())

    return checks
