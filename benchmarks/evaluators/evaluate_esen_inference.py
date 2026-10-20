'''V2 adapter for the standalone eSEN inference evaluator.'''
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
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


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'esen_inference.py',
        'run_inference.py',
        'esen_inference.yaml',
        'portable_esen/__init__.py',
        'portable_esen/model.py',
        'portable_esen/esen_dens.py',
        'portable_esen/graph.py',
        'portable_esen/layer/esen/esen_block.py',
        'portable_esen/layer/esen/common/rotation.py',
        'portable_esen/layer/esen/common/so3.py',
        'portable_esen/layer/esen/nn/embedding.py',
        'portable_esen/layer/esen/nn/layer_norm.py',
        'portable_esen/layer/esen/nn/radial.py',
        'portable_esen/layer/esen/nn/so2_layers.py',
        'portable_esen/layer/esen/nn/so3_layers.py',
        'portable_esen/layer/esen/nn/activation.py',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        REQUIRED_FILES = (
            'esen_inference.py',
            'run_inference.py',
            'esen_inference.yaml',
            'portable_esen/__init__.py',
            'portable_esen/model.py',
            'portable_esen/esen_dens.py',
            'portable_esen/graph.py',
            'portable_esen/layer/esen/esen_block.py',
            'portable_esen/layer/esen/common/rotation.py',
            'portable_esen/layer/esen/common/so3.py',
            'portable_esen/layer/esen/nn/embedding.py',
            'portable_esen/layer/esen/nn/layer_norm.py',
            'portable_esen/layer/esen/nn/radial.py',
            'portable_esen/layer/esen/nn/so2_layers.py',
            'portable_esen/layer/esen/nn/so3_layers.py',
            'portable_esen/layer/esen/nn/activation.py',
        )
        missing = [item for item in REQUIRED_FILES if not (CANDIDATE_DIR / item).is_file()]
        for item in REQUIRED_FILES:
            ok = item not in missing
            detail = 'all required files exist' if ok else f'missing: {missing}'
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'prohibited_active_imports',
        'weights_only_true_loading',
        'strict_true_state_loading',
        'payload_state_entry_count',
        'payload_state_parameter_numel',
        'checkpoint_sha256',
        'rotation_basis_sha256',
        'restored_model_state_entry_count',
        'restored_state_ordered_keys',
        'restored_state_shapes',
        'restored_state_tensor_values',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        CHECKPOINT = Path('/public/share/sugonhpcapp01/onestore/onemodels/eSEN/esen_30m_mptrj.pt')
        JD_PATH = Path('/public/share/sugonhpcapp01/onestore/onemodels/UMA/checkpoint/Jd.pt')
        CHECKPOINT_SHA256 = 'f44aecdfbdb74575f8861c08ee6aa86939cce4571e73ac508666190cb67f064b'
        JD_SHA256 = 'b4059c45be246dcb6c49c545670b65c56550eb0c2e7a9c92b4b50a92d370dbe2'
        STATE_COUNT = 394
        STATE_NUMEL = 30087404
        PROHIBITED = {'onescience', 'fairchem', 'mattersim', 'mace'}
        for name in list(sys.modules):
            if name == 'portable_esen' or name.startswith('portable_esen.'):
                sys.modules.pop(name, None)
        module_path = CANDIDATE_DIR / 'esen_inference.py'
        spec = importlib.util.spec_from_file_location('esen_inference_submission', module_path)
        if spec is None or spec.loader is None:
            raise ImportError(module_path)
        if str(module_path.parent) not in sys.path:
            sys.path.insert(0, str(module_path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules['esen_inference_submission'] = module
        spec.loader.exec_module(module)
        prohibited = []
        has_weights_only_true = False
        has_strict_true = False
        for path in CANDIDATE_DIR.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8', errors='ignore'), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    prohibited += [alias.name for alias in node.names if alias.name.split('.')[0] in PROHIBITED]
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in PROHIBITED:
                    prohibited.append(node.module)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                    if node.func.attr == 'load' and isinstance(keywords.get('weights_only'), ast.Constant) and keywords['weights_only'].value is True:
                        has_weights_only_true = True
                    if node.func.attr == 'load_state_dict' and isinstance(keywords.get('strict'), ast.Constant) and keywords['strict'].value is True:
                        has_strict_true = True
        ok = not prohibited
        update_check(current_index, ok, 'no prohibited active imports' if ok else f'prohibited active imports: {prohibited}')
        current_index += 1
        if not prohibited:
            ok = has_weights_only_true
            update_check(current_index, ok, 'weights_only=True state loading is present' if ok else 'candidate must use weights_only=True and strict=True state loading')
            current_index += 1
            ok = has_strict_true
            update_check(current_index, ok, 'strict=True state loading is present' if ok else 'candidate must use weights_only=True and strict=True state loading')
            current_index += 1
            if has_weights_only_true and has_strict_true:
                payload = module.load_checkpoint_payload(str(CHECKPOINT))
                state = payload['state_dict']
                state_numel = sum(value.numel() for value in state.values())
                state_count_ok = len(state) == STATE_COUNT
                state_numel_ok = state_numel == STATE_NUMEL
                state_detail = f'unexpected state envelope count={len(state)} numel={state_numel}'
                update_check(current_index, state_count_ok, 'checkpoint state entry count is correct' if state_count_ok else state_detail)
                current_index += 1
                update_check(current_index, state_numel_ok, 'checkpoint state parameter numel is correct' if state_numel_ok else state_detail)
                current_index += 1
                if state_count_ok and state_numel_ok:
                    checkpoint_digest = hashlib.sha256()
                    with CHECKPOINT.open('rb') as handle:
                        for chunk in iter(lambda: handle.read(1 << 20), b''):
                            checkpoint_digest.update(chunk)
                    checkpoint_hash = checkpoint_digest.hexdigest()
                    jd_digest = hashlib.sha256()
                    with JD_PATH.open('rb') as handle:
                        for chunk in iter(lambda: handle.read(1 << 20), b''):
                            jd_digest.update(chunk)
                    jd_hash = jd_digest.hexdigest()
                    hash_detail = f'fixed resource SHA256 mismatch checkpoint={checkpoint_hash} jd={jd_hash}'
                    ok = checkpoint_hash == CHECKPOINT_SHA256
                    update_check(current_index, ok, 'fixed checkpoint SHA256 matches' if ok else hash_detail)
                    current_index += 1
                    ok = jd_hash == JD_SHA256
                    update_check(current_index, ok, 'fixed rotation basis SHA256 matches' if ok else hash_detail)
                    current_index += 1
                    if checkpoint_hash == CHECKPOINT_SHA256 and jd_hash == JD_SHA256:
                        loaded = module.load_model(str(CHECKPOINT), device='cpu', jd_path=str(JD_PATH))
                        model = loaded[0] if isinstance(loaded, tuple) else loaded
                        expected_state = {str(key)[len('module.'):]: value for key, value in state.items()}
                        model_state = model.state_dict()
                        same_keys = tuple(expected_state) == tuple(model_state)
                        same_shapes = same_keys and all(expected_state[name].shape == model_state[name].shape for name in expected_state)
                        same_values = same_shapes and all(torch.equal(expected_state[name], model_state[name].cpu()) for name in expected_state)
                        ok = len(model_state) == STATE_COUNT and same_values
                        detail = json.dumps({'checkpoint_sha256': checkpoint_hash, 'jd_sha256': jd_hash, 'state_count': len(model_state), 'same_keys': same_keys, 'same_shapes': same_shapes, 'same_values': same_values}, sort_keys=True)
                        update_check(current_index, len(model_state) == STATE_COUNT, detail)
                        current_index += 1
                        update_check(current_index, same_keys, detail)
                        current_index += 1
                        update_check(current_index, same_shapes, detail)
                        current_index += 1
                        update_check(current_index, same_values, detail)
                        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'backbone_class',
        'mptrj_head_class',
        'backbone_block_count',
        'backbone_lmax',
        'backbone_mmax',
        'backbone_sphere_channels',
        'backbone_cutoff',
        'backbone_num_distance_basis',
        'backbone_direct_forces',
        'model_module_order',
        'backbone_module_order',
        'block_module_order',
        'head_module_order',
        'sphere_embedding_shape',
        'distance_expansion_shape',
        'energy_head_shape',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        CHECKPOINT = Path('/public/share/sugonhpcapp01/onestore/onemodels/eSEN/esen_30m_mptrj.pt')
        JD_PATH = Path('/public/share/sugonhpcapp01/onestore/onemodels/UMA/checkpoint/Jd.pt')
        SEED = 20260820
        module_path = CANDIDATE_DIR / 'esen_inference.py'
        spec = importlib.util.spec_from_file_location('esen_inference_submission', module_path)
        if spec is None or spec.loader is None:
            raise ImportError(module_path)
        if str(module_path.parent) not in sys.path:
            sys.path.insert(0, str(module_path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules['esen_inference_submission'] = module
        spec.loader.exec_module(module)
        torch.manual_seed(SEED)
        loaded = module.load_model(str(CHECKPOINT), device='cpu', jd_path=str(JD_PATH))
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        backbone = getattr(model, 'backbone', None)
        heads = getattr(model, 'output_heads', None)
        head = heads['mptrj'] if heads is not None and 'mptrj' in heads else None
        model_order = list(model._modules)
        backbone_order = list(backbone._modules) if backbone is not None else []
        required_backbone_order = ('SO3_grid', 'sphere_embedding', 'distance_expansion', 'source_embedding', 'target_embedding', 'edge_degree_embedding', 'blocks', 'norm', 'force_embedding')
        positions = [backbone_order.index(name) for name in required_backbone_order if name in backbone_order]
        backbone_order_ok = len(positions) == len(required_backbone_order) and positions == sorted(positions)
        blocks = list(getattr(backbone, 'blocks', []))
        block_order_ok = len(blocks) == 10 and all(tuple(block._modules) == ('norm_1', 'edge_wise', 'norm_2', 'atom_wise') for block in blocks)
        head_order = list(head._modules) if head is not None else []
        with torch.no_grad():
            sphere_shape = list(backbone.sphere_embedding(torch.tensor([1, 8])).shape) if backbone is not None else []
            distance_shape = list(backbone.distance_expansion(torch.tensor([0.5, 3.0, 5.5])).shape) if backbone is not None else []
            energy_shape = list(head.energy_block(torch.ones(2, 128)).shape) if head is not None else []
        details = {'backbone': type(backbone).__name__, 'head': type(head).__name__ if head else None, 'blocks': len(blocks), 'lmax': getattr(backbone, 'lmax', None), 'mmax': getattr(backbone, 'mmax', None), 'sphere_channels': getattr(backbone, 'sphere_channels', None), 'cutoff': getattr(backbone, 'cutoff', None), 'num_distance_basis': getattr(backbone, 'num_distance_basis', None), 'direct_forces': getattr(backbone, 'direct_forces', None), 'model_order': model_order, 'backbone_order_ok': backbone_order_ok, 'block_order_ok': block_order_ok, 'head_order': head_order, 'sphere_embedding_shape': sphere_shape, 'distance_basis_shape': distance_shape, 'energy_head_shape': energy_shape}
        ok = all((type(backbone).__name__ == 'eSEN_DeNS_Backbone', type(head).__name__ == 'MLP_EFS_Head', len(blocks) == 10, getattr(backbone, 'lmax', None) == 3, getattr(backbone, 'mmax', None) == 2, getattr(backbone, 'sphere_channels', None) == 128, getattr(backbone, 'cutoff', None) == 6.0, getattr(backbone, 'num_distance_basis', None) == 10, getattr(backbone, 'direct_forces', None) is False, model_order == ['backbone', 'output_heads'], backbone_order_ok, block_order_ok, head_order == ['energy_block', 'denoising_linear'], sphere_shape == [2, 128], distance_shape == [3, 10], energy_shape == [2, 1]))
        detail = json.dumps(details, sort_keys=True)
        update_check(current_index, type(backbone).__name__ == 'eSEN_DeNS_Backbone', detail)
        current_index += 1
        update_check(current_index, type(head).__name__ == 'MLP_EFS_Head', detail)
        current_index += 1
        update_check(current_index, len(blocks) == 10, detail)
        current_index += 1
        update_check(current_index, getattr(backbone, 'lmax', None) == 3, detail)
        current_index += 1
        update_check(current_index, getattr(backbone, 'mmax', None) == 2, detail)
        current_index += 1
        update_check(current_index, getattr(backbone, 'sphere_channels', None) == 128, detail)
        current_index += 1
        update_check(current_index, getattr(backbone, 'cutoff', None) == 6.0, detail)
        current_index += 1
        update_check(current_index, getattr(backbone, 'num_distance_basis', None) == 10, detail)
        current_index += 1
        update_check(current_index, getattr(backbone, 'direct_forces', None) is False, detail)
        current_index += 1
        update_check(current_index, model_order == ['backbone', 'output_heads'], detail)
        current_index += 1
        update_check(current_index, backbone_order_ok, detail)
        current_index += 1
        update_check(current_index, block_order_ok, detail)
        current_index += 1
        update_check(current_index, head_order == ['energy_block', 'denoising_linear'], detail)
        current_index += 1
        update_check(current_index, sphere_shape == [2, 128], detail)
        current_index += 1
        update_check(current_index, distance_shape == [3, 10], detail)
        current_index += 1
        update_check(current_index, energy_shape == [2, 1], detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'load_config_callable',
        'load_structures_callable',
        'evaluate_callable',
        'save_predictions_callable',
        'main_callable',
        'load_structures_ase_input',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        module_path = CANDIDATE_DIR / 'run_inference.py'
        spec = importlib.util.spec_from_file_location('esen_inference_submission_run', module_path)
        if spec is None or spec.loader is None:
            raise ImportError(module_path)
        if str(module_path.parent) not in sys.path:
            sys.path.insert(0, str(module_path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules['esen_inference_submission_run'] = module
        spec.loader.exec_module(module)
        required = ('load_config', 'load_structures', 'evaluate', 'save_predictions', 'main')
        missing = [name for name in required if not callable(getattr(module, name, None))]
        for name in required:
            ok = name not in missing
            update_check(current_index, ok, f'missing run_inference APIs: {missing}' if not ok else 'required run_inference API is present')
            current_index += 1
        if not missing:
            from ase.build import bulk
            structures = module.load_structures(bulk('Si', 'diamond', a=5.43))
            ok = len(structures) == 1
            update_check(current_index, ok, 'ASE input, evaluate/save APIs and required CLI entry points are present' if ok else 'load_structures must accept one ASE Atoms')
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'exact_output_keys',
        'energy_shape',
        'forces_shape',
        'stress_shape',
        'energy_finite',
        'forces_finite',
        'stress_finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        CHECKPOINT = Path('/public/share/sugonhpcapp01/onestore/onemodels/eSEN/esen_30m_mptrj.pt')
        JD_PATH = Path('/public/share/sugonhpcapp01/onestore/onemodels/UMA/checkpoint/Jd.pt')
        SEED = 20260820
        module_path = CANDIDATE_DIR / 'esen_inference.py'
        spec = importlib.util.spec_from_file_location('esen_inference_submission', module_path)
        if spec is None or spec.loader is None:
            raise ImportError(module_path)
        if str(module_path.parent) not in sys.path:
            sys.path.insert(0, str(module_path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules['esen_inference_submission'] = module
        spec.loader.exec_module(module)
        from ase.build import bulk
        torch.manual_seed(SEED)
        payload = module.load_checkpoint_payload(str(CHECKPOINT))
        loaded = module.load_model(str(CHECKPOINT), device='cpu', jd_path=str(JD_PATH))
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model.eval()
        torch.manual_seed(SEED)
        output = module.evaluate_structure(model, bulk('Si', 'diamond', a=5.43), payload, 'cpu')
        shapes = {key: list(value.shape) for key, value in output.items()}
        finite = all(torch.isfinite(value).all().item() for value in output.values())
        ok = shapes == {'energy': [1], 'forces': [2, 3], 'stress': [1, 3, 3]} and finite
        detail = json.dumps({'shapes': shapes, 'finite': finite}, sort_keys=True)
        output_keys_ok = set(shapes) == {'energy', 'forces', 'stress'}
        energy_shape_ok = shapes.get('energy') == [1]
        forces_shape_ok = shapes.get('forces') == [2, 3]
        stress_shape_ok = shapes.get('stress') == [1, 3, 3]
        energy_finite_ok = 'energy' in output and torch.isfinite(output['energy']).all().item()
        forces_finite_ok = 'forces' in output and torch.isfinite(output['forces']).all().item()
        stress_finite_ok = 'stress' in output and torch.isfinite(output['stress']).all().item()
        update_check(current_index, output_keys_ok, detail)
        current_index += 1
        update_check(current_index, energy_shape_ok, detail)
        current_index += 1
        update_check(current_index, forces_shape_ok, detail)
        current_index += 1
        update_check(current_index, stress_shape_ok, detail)
        current_index += 1
        update_check(current_index, energy_finite_ok, detail)
        current_index += 1
        update_check(current_index, forces_finite_ok, detail)
        current_index += 1
        update_check(current_index, stress_finite_ok, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'model_state_ordered_keys',
        'model_state_shapes',
        'model_state_tensor_values',
        'energy_max_absolute_error',
        'forces_max_absolute_error',
        'stress_max_absolute_error',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        CHECKPOINT = Path('/public/share/sugonhpcapp01/onestore/onemodels/eSEN/esen_30m_mptrj.pt')
        JD_PATH = Path('/public/share/sugonhpcapp01/onestore/onemodels/UMA/checkpoint/Jd.pt')
        SEED = 20260820
        for name in list(sys.modules):
            if name == 'portable_esen' or name.startswith('portable_esen.'):
                sys.modules.pop(name, None)
        standard_root = REFERENCE_DIR
        standard_path = standard_root / 'esen_inference.py'
        spec = importlib.util.spec_from_file_location('esen_inference_standard', standard_path)
        if spec is None or spec.loader is None:
            raise ImportError(standard_path)
        if str(standard_path.parent) not in sys.path:
            sys.path.insert(0, str(standard_path.parent))
        standard = importlib.util.module_from_spec(spec)
        sys.modules['esen_inference_standard'] = standard
        spec.loader.exec_module(standard)
        from ase.build import bulk
        atoms = bulk('Si', 'diamond', a=5.43)
        torch.manual_seed(SEED)
        ref_payload = standard.load_checkpoint_payload(str(CHECKPOINT))
        ref_model = standard.load_model(str(CHECKPOINT), device='cpu', jd_path=str(JD_PATH))
        ref_model = ref_model[0] if isinstance(ref_model, tuple) else ref_model
        torch.manual_seed(SEED)
        ref = standard.evaluate_structure(ref_model, atoms, ref_payload, 'cpu')
        for name in list(sys.modules):
            if name == 'portable_esen' or name.startswith('portable_esen.'):
                sys.modules.pop(name, None)
        module_path = CANDIDATE_DIR / 'esen_inference.py'
        spec = importlib.util.spec_from_file_location('esen_inference_submission', module_path)
        if spec is None or spec.loader is None:
            raise ImportError(module_path)
        if str(module_path.parent) not in sys.path:
            sys.path.insert(0, str(module_path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules['esen_inference_submission'] = module
        spec.loader.exec_module(module)
        torch.manual_seed(SEED)
        candidate_payload = module.load_checkpoint_payload(str(CHECKPOINT))
        candidate_model = module.load_model(str(CHECKPOINT), device='cpu', jd_path=str(JD_PATH))
        candidate_model = candidate_model[0] if isinstance(candidate_model, tuple) else candidate_model
        reference_state = ref_model.state_dict()
        candidate_state = candidate_model.state_dict()
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
            torch.manual_seed(SEED)
            pred = module.evaluate_structure(candidate_model, atoms, candidate_payload, 'cpu')
            errors = {key: float((pred[key] - ref[key]).abs().max()) for key in ref}
            ok = all(value <= 1e-4 for value in errors.values())
            detail = json.dumps({'same_checkpoint_values': True, 'max_abs_error': errors, 'atol': 1e-4}, sort_keys=True)
            update_check(current_index, errors.get('energy', float('inf')) <= 1e-4, detail)
            current_index += 1
            update_check(current_index, errors.get('forces', float('inf')) <= 1e-4, detail)
            current_index += 1
            update_check(current_index, errors.get('stress', float('inf')) <= 1e-4, detail)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
