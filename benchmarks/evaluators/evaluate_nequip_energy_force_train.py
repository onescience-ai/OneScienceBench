"""V2 evaluator adapter for deterministic NequIP energy-force training."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
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

DATA = Path('/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/demo/reference_data/smoke.xyz')
DATA_SHA256 = '6671f9b28d37484379f51f1deb23a6f4b165ed628467ee6bffa0648b6d50a5b5'
SEED = 456
PORTABLE_FILES = ('portable_nequip/__init__.py', 'portable_nequip/internal.py', 'portable_nequip/model.py', 'portable_nequip/energy_modules.py', 'portable_nequip/runtime.py', 'portable_nequip/NOTICE.md', 'portable_nequip/data/__init__.py', 'portable_nequip/data/AtomicDataDict.py', 'portable_nequip/data/_keys.py', 'portable_nequip/data/_key_registry.py', 'portable_nequip/nn/__init__.py', 'portable_nequip/nn/_ghost_exchange_base.py', 'portable_nequip/nn/_graph_mixin.py', 'portable_nequip/nn/_tp_scatter_base.py', 'portable_nequip/nn/atomwise.py', 'portable_nequip/nn/convnetlayer.py', 'portable_nequip/nn/grad_output.py', 'portable_nequip/nn/graph_model.py', 'portable_nequip/nn/interaction_block.py', 'portable_nequip/nn/misc.py', 'portable_nequip/nn/mlp.py', 'portable_nequip/nn/model_modifier_utils.py', 'portable_nequip/nn/nonlinearities.py', 'portable_nequip/nn/norm.py', 'portable_nequip/nn/utils.py', 'portable_nequip/nn/embedding/__init__.py', 'portable_nequip/nn/embedding/_edge.py', 'portable_nequip/nn/embedding/cutoffs.py', 'portable_nequip/nn/embedding/node.py', 'portable_nequip/nn/embedding/node_tensor.py', 'portable_nequip/nn/embedding/utils.py')
REQUIRED_FILES = ('nequip_energy_force_train.py', 'train_nequip_from_scratch.py', 'nequip_train.yaml', *PORTABLE_FILES)
PROHIBITED_IMPORTS = {'onescience', 'nequip', 'mace', 'mattersim'}

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

def source_imports_ok(submission):
    prohibited = []
    for path in submission.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8', errors='ignore'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                prohibited.extend(alias.name for alias in node.names if alias.name.split('.')[0] in PROHIBITED_IMPORTS)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in PROHIBITED_IMPORTS:
                prohibited.append(node.module)
    return prohibited

def tensor_dict_equal(left, right):
    return tuple(left) == tuple(right) and all(left[name].shape == right[name].shape and torch.equal(left[name].cpu(), right[name].cpu()) for name in left)

def structure_details(model):
    wrapper = getattr(model, 'model', None)
    network = getattr(wrapper, 'func', None)
    order = list(network._modules) if network is not None else []
    expected = ['type_embed', 'spharm', 'edge_norm', 'bessel_encode', 'factor', 'layer0_convnet', 'layer1_convnet', 'per_atom_energy_readout', 'per_type_energy_scale_shift', 'total_energy_sum']
    layers = [getattr(network, f'layer{index}_convnet', None) for index in range(2)] if network is not None else []
    shapes = {}
    if network is not None:
        shapes = {'type_embedding': list(network.type_embed.embed_module.weight.shape), 'bessel': list(network.bessel_encode.bessel_weights.shape), 'layer0_radial_in': list(network.layer0_convnet.conv.edge_mlp.mlp[0].weight.shape), 'layer0_radial_out': list(network.layer0_convnet.conv.edge_mlp.mlp[2].weight.shape), 'layer1_radial_in': list(network.layer1_convnet.conv.edge_mlp.mlp[0].weight.shape), 'layer1_radial_out': list(network.layer1_convnet.conv.edge_mlp.mlp[2].weight.shape), 'readout': list(network.per_atom_energy_readout.mlp_module.mlp[0].weight.shape)}
    ok = all((type(model).__name__ == 'GraphModel', type(wrapper).__name__ == 'ForceStressOutput', type(network).__name__ == 'SequentialGraphNetwork', list(model._modules) == ['model'], list(wrapper._modules) == ['func'], order == expected, len(layers) == 2 and all(type(layer).__name__ == 'ConvNetLayer' for layer in layers), getattr(layers[0].conv, 'sc', None) is None, getattr(layers[1].conv, 'sc', None) is not None, shapes == {'type_embedding': [1, 8], 'bessel': [1, 4], 'layer0_radial_in': [4, 16], 'layer0_radial_out': [16, 16], 'layer1_radial_in': [4, 16], 'layer1_radial_out': [16, 16], 'readout': [8, 1]}))
    return ok, {'classes': [type(model).__name__, type(wrapper).__name__, type(network).__name__], 'module_order': order, 'layer_count': len(layers), 'shapes': shapes}

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['files:required', 'files:forbidden_imports']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        params = config.get('params', config)
        required = tuple(params.get('required_files', REQUIRED_FILES))
        missing = []
        for name in required:
            path = (CANDIDATE_DIR / Path(name)).resolve(); path.relative_to(CANDIDATE_DIR)
            if not path.is_file():
                missing.append(name)
        update_check(current_index, not missing, json.dumps({'missing': missing}, sort_keys=True)); current_index += 1
        prohibited = source_imports_ok(CANDIDATE_DIR) if not missing else []
        update_check(current_index, not prohibited, json.dumps({'prohibited_active_imports': prohibited}, sort_keys=True)); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['data:sha256', 'data:structure_count', 'data:first_structure_atoms', 'data:positions_shape', 'data:edge_index_shape', 'data:cell_shift_shape', 'data:cell_shape', 'data:energy_shape', 'data:forces_shape', 'data:energy_value', 'data:first_force']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        params = config.get('params', config); candidate = params.get('candidate', {})
        path = (CANDIDATE_DIR / Path(candidate.get('file', 'nequip_energy_force_train.py'))).resolve(); path.relative_to(CANDIDATE_DIR)
        purge_portable(); module = load_module('nequip_energy_force_train_submission_data', path)
        structures = module.load_training_structures(str(DATA)); batch = module.build_training_batch(str(DATA), 0, 'cpu'); data = batch['data']
        expected_energy = -27.622615177843876; expected_force = torch.tensor([-0.01459779, 0.12819022, 0.10740306], dtype=torch.float64)
        shapes = {'positions': list(data['pos'].shape), 'edge_index': list(data['edge_index'].shape), 'cell_shift': list(data['edge_cell_shift'].shape), 'cell': list(data['cell'].shape), 'energy': list(batch['energy'].shape), 'forces': list(batch['forces'].shape)}
        detail = json.dumps({'data_sha256': sha256(DATA), 'structure_count': len(structures), 'shapes': shapes, 'first_energy': float(batch['energy'].item()), 'first_force': batch['forces'][0].tolist()}, sort_keys=True)
        results = [(sha256(DATA) == DATA_SHA256, detail), (len(structures) == 8, detail), (len(structures[0]) == 8, detail), (shapes['positions'] == [8, 3], detail), (shapes['edge_index'] == [2, 144], detail), (shapes['cell_shift'] == [144, 3], detail), (shapes['cell'] == [1, 3, 3], detail), (shapes['energy'] == [1, 1], detail), (shapes['forces'] == [8, 3], detail), (abs(float(batch['energy'].item()) - expected_energy) <= 1e-12, detail), (torch.allclose(batch['forces'][0], expected_force, atol=1e-8, rtol=0.0), detail)]
        for passed, message in results:
            update_check(current_index, passed, message); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['train:public_apis', 'train:required_config_keys', 'train:optimizer', 'train:lr', 'train:max_steps', 'train:seed']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        params = config.get('params', config); files = params.get('files', ['train_nequip_from_scratch.py']); path = (CANDIDATE_DIR / Path(files[0])).resolve(); path.relative_to(CANDIDATE_DIR)
        train_module = load_module('nequip_energy_force_train_submission_train_interface', path)
        required = tuple(params.get('required_interfaces', ('load_config', 'build_model', 'build_trainer', 'train_step', 'train', 'main'))); missing = [name for name in required if not callable(getattr(train_module, name, None))]
        config_path = (CANDIDATE_DIR / Path(params.get('config_name', 'nequip_train.yaml'))).resolve(); config_path.relative_to(CANDIDATE_DIR); loaded = train_module.load_config(str(config_path)); training = loaded.get('training', {})
        required_keys = set(params.get('required_training_keys', ('optimizer', 'lr', 'eps', 'energy_weight', 'force_weight', 'gradient_clip_norm', 'max_steps', 'structure_index', 'seed'))); missing_keys = sorted(required_keys - set(training))
        detail = json.dumps({'missing_apis': missing, 'missing_config_keys': missing_keys, 'optimizer': training.get('optimizer'), 'lr': training.get('lr'), 'max_steps': training.get('max_steps'), 'seed': training.get('seed')}, sort_keys=True)
        update_check(current_index, not missing, detail); current_index += 1; update_check(current_index, not missing_keys, detail); current_index += 1
        if not missing_keys:
            for passed in (training['optimizer'] == 'Adam', float(training['lr']) == 0.01, int(training['max_steps']) == 1, int(training['seed']) == SEED):
                update_check(current_index, passed, detail); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['structure:model_class', 'structure:wrapper_class', 'structure:network_class', 'structure:model_children', 'structure:wrapper_children', 'structure:module_order', 'structure:layer_count', 'structure:layer0_class', 'structure:layer1_class', 'structure:layer0_skip_connection', 'structure:layer1_skip_connection', 'structure:type_embedding_shape', 'structure:bessel_shape', 'structure:layer0_radial_in_shape', 'structure:layer0_radial_out_shape', 'structure:layer1_radial_in_shape', 'structure:layer1_radial_out_shape', 'structure:readout_shape', 'structure:state_count', 'structure:state_numel']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        params = config.get('params', config); path = (CANDIDATE_DIR / Path(params.get('model_file', 'nequip_energy_force_train.py'))).resolve(); path.relative_to(CANDIDATE_DIR); purge_portable(); module = load_module('nequip_energy_force_train_submission_structure', path)
        seed = params.get('seed', SEED); torch.manual_seed(seed); model = module.build_model(seed, params.get('device', 'cpu')); _, details = structure_details(model); state = model.state_dict(); details.update({'state_count': len(state), 'state_numel': sum(value.numel() for value in state.values())})
        wrapper = getattr(model, 'model', None); network = getattr(wrapper, 'func', None); order = list(network._modules) if network is not None else []; expected = ['type_embed', 'spharm', 'edge_norm', 'bessel_encode', 'factor', 'layer0_convnet', 'layer1_convnet', 'per_atom_energy_readout', 'per_type_energy_scale_shift', 'total_energy_sum']; layers = [getattr(network, f'layer{index}_convnet', None) for index in range(2)] if network is not None else []; shapes = details['shapes']; detail = json.dumps(details, sort_keys=True)
        results = [(type(model).__name__ == 'GraphModel', detail), (type(wrapper).__name__ == 'ForceStressOutput', detail), (type(network).__name__ == 'SequentialGraphNetwork', detail), (list(model._modules) == ['model'], detail), (list(wrapper._modules) == ['func'], detail), (order == expected, detail), (len(layers) == 2, detail), (type(layers[0]).__name__ == 'ConvNetLayer', detail), (type(layers[1]).__name__ == 'ConvNetLayer', detail), (getattr(layers[0].conv, 'sc', None) is None, detail), (getattr(layers[1].conv, 'sc', None) is not None, detail), (shapes['type_embedding'] == [1, 8], detail), (shapes['bessel'] == [1, 4], detail), (shapes['layer0_radial_in'] == [4, 16], detail), (shapes['layer0_radial_out'] == [16, 16], detail), (shapes['layer1_radial_in'] == [4, 16], detail), (shapes['layer1_radial_out'] == [16, 16], detail), (shapes['readout'] == [8, 1], detail), (len(state) == 33, detail), (sum(value.numel() for value in state.values()) == 1855, detail)]
        for passed, message in results:
            update_check(current_index, passed, message); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['forward:energy_shape', 'forward:forces_shape', 'forward:stress_shape', 'forward:energy_finite', 'forward:forces_finite', 'forward:stress_finite']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        params = config.get('params', config); path = (CANDIDATE_DIR / Path(params.get('model_file', 'nequip_energy_force_train.py'))).resolve(); path.relative_to(CANDIDATE_DIR); purge_portable(); module = load_module('nequip_energy_force_train_submission_forward', path); seed = params.get('torch_manual_seed_before_model', SEED); torch.manual_seed(seed); model = module.build_model(seed, params.get('device', 'cpu')).eval(); batch = module.build_training_batch(str(DATA), 0, params.get('device', 'cpu')); torch.manual_seed(params.get('torch_manual_seed_before_forward', SEED)); output = module.predict_energy_forces(model, batch); shapes = {name: list(value.shape) for name, value in output.items()}; detail = json.dumps({'shapes': shapes, 'finite': all(torch.isfinite(value).all().item() for value in output.values())}, sort_keys=True)
        results = [(shapes.get('energy') == [1, 1], detail), (shapes.get('forces') == [8, 3], detail), (shapes.get('stress') == [1, 3, 3], detail), ('energy' in output and torch.isfinite(output['energy']).all().item(), detail), ('forces' in output and torch.isfinite(output['forces']).all().item(), detail), ('stress' in output and torch.isfinite(output['stress']).all().item(), detail)]
        for passed, message in results:
            update_check(current_index, passed, message); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['state:ordered_keys', 'state:shapes', 'state:values', 'forward:energy_reference_equivalence', 'forward:forces_reference_equivalence', 'forward:stress_reference_equivalence']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        params = config.get('params', config); candidate_path = (CANDIDATE_DIR / Path(params.get('model_file', 'nequip_energy_force_train.py'))).resolve(); candidate_path.relative_to(CANDIDATE_DIR); standard_path = (REFERENCE_DIR / Path(params.get('standard_module', 'nequip_energy_force_train.py'))).resolve(); standard_path.relative_to(REFERENCE_DIR); seed = params.get('seed', SEED)
        purge_portable(); standard = load_module('nequip_energy_force_train_standard', standard_path); torch.manual_seed(seed); reference_model = standard.build_model(seed, 'cpu').eval(); reference_batch = standard.build_training_batch(str(DATA), 0, 'cpu'); torch.manual_seed(seed); reference = standard.predict_energy_forces(reference_model, reference_batch); reference_state = reference_model.state_dict(); purge_portable(); module = load_module('nequip_energy_force_train_submission_equivalence', candidate_path); torch.manual_seed(seed); candidate_model = module.build_model(seed, 'cpu').eval(); candidate_state = candidate_model.state_dict()
        same_keys = tuple(reference_state) == tuple(candidate_state); same_shapes = same_keys and all(reference_state[name].shape == candidate_state[name].shape for name in reference_state); same_values = same_shapes and all(torch.equal(reference_state[name], candidate_state[name]) for name in reference_state); detail = json.dumps({'same_keys': same_keys, 'same_shapes': same_shapes, 'same_initial_values': same_values}, sort_keys=True)
        update_check(current_index, same_keys, detail); current_index += 1; update_check(current_index, same_shapes, detail); current_index += 1; update_check(current_index, same_values, detail); current_index += 1
        if same_values:
            torch.manual_seed(seed); prediction = module.predict_energy_forces(candidate_model, reference_batch); errors = {name: float((prediction[name] - reference[name]).abs().max()) for name in reference}; atol = params.get('compare_atol', 1e-6); detail = json.dumps({'same_initial_values': True, 'max_abs_error': errors, 'atol': atol}, sort_keys=True)
            for name in ('energy', 'forces', 'stress'):
                update_check(current_index, errors[name] <= atol, detail); current_index += 1
        else:
            current_index = len(CHECKPOINT_NAMES)
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['training:deterministic_initialization', 'training:optimizer_parameter_identity', 'training:trainable_parameter_count', 'training:one_step_metric_finiteness', 'training:one_step_parameter_update']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        params = config.get('params', config); model_path = (CANDIDATE_DIR / Path(params.get('model_file', 'nequip_energy_force_train.py'))).resolve(); model_path.relative_to(CANDIDATE_DIR); train_path = (CANDIDATE_DIR / Path(params.get('candidate_train_module', 'train_nequip_from_scratch.py'))).resolve(); train_path.relative_to(CANDIDATE_DIR); config_path = (CANDIDATE_DIR / Path(params.get('config_name', 'nequip_train.yaml'))).resolve(); config_path.relative_to(CANDIDATE_DIR); purge_portable(); module = load_module('nequip_energy_force_train_submission_training_data', model_path); train_module = load_module('nequip_energy_force_train_submission_training', train_path); loaded = train_module.load_config(str(config_path)); torch.manual_seed(SEED); model, optimizer = train_module.build_trainer(loaded, 'cpu'); torch.manual_seed(SEED); second_model = train_module.build_model(loaded, 'cpu'); deterministic_init = tensor_dict_equal(model.state_dict(), second_model.state_dict()); trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]; trainable_ids = {id(parameter) for _, parameter in trainable}; optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}; before = {name: parameter.detach().clone() for name, parameter in trainable}; batch = module.build_training_batch(str(DATA), int(loaded['training']['structure_index']), 'cpu'); record = train_module.train_step(model, optimizer, batch, loaded); changed = [name for name, parameter in trainable if not torch.equal(before[name], parameter.detach())]; finite = all(torch.isfinite(torch.tensor(float(value))).item() for value in record.values()); detail = json.dumps({'record': record, 'deterministic_initial_state': deterministic_init, 'trainable': len(trainable), 'changed': changed, 'optimizer_matches_trainable': optimizer_ids == trainable_ids}, sort_keys=True)
        for passed in (deterministic_init, optimizer_ids == trainable_ids, len(trainable) == 11, finite, len(changed) >= 10):
            update_check(current_index, passed, detail); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['artifacts:nequip_trained_pt', 'artifacts:metrics_jsonl', 'artifacts:train_log', 'artifacts:checkpoint_keys', 'artifacts:format', 'artifacts:completed_steps', 'artifacts:optimizer_nonempty', 'artifacts:history_length', 'artifacts:strict_reload']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        params = config.get('params', config); train_path = (CANDIDATE_DIR / Path(params.get('candidate_train_module', 'train_nequip_from_scratch.py'))).resolve(); train_path.relative_to(CANDIDATE_DIR); config_path = (CANDIDATE_DIR / Path(params.get('config_name', 'nequip_train.yaml'))).resolve(); config_path.relative_to(CANDIDATE_DIR); train_module = load_module('nequip_energy_force_train_submission_artifacts', train_path)
        with tempfile.TemporaryDirectory(prefix='nequip_train_eval_') as directory:
            train_module.train(str(config_path), directory, 'cpu'); output = Path(directory); required = list(params.get('output_files', ('nequip_trained.pt', 'metrics.jsonl', 'train.log'))); missing = [name for name in required if not (output / name).is_file()]
            for name in required:
                update_check(current_index, (output / name).is_file(), str(output / name)); current_index += 1
            if not missing:
                payload = torch.load(output / required[0], map_location='cpu', weights_only=True); expected = set(params.get('required_checkpoint_keys', ('format', 'model_name', 'model', 'optimizer', 'completed_steps', 'config', 'history'))); detail = json.dumps({'keys': sorted(payload), 'completed_steps': payload.get('completed_steps'), 'optimizer_nonempty': bool(payload.get('optimizer')), 'strict_reload': True}, sort_keys=True)
                update_check(current_index, expected.issubset(payload), detail); current_index += 1; update_check(current_index, payload['format'] == 'nequip-energy-force-train-v1', detail); current_index += 1; update_check(current_index, payload['completed_steps'] == 1, detail); current_index += 1; update_check(current_index, bool(payload['optimizer']), detail); current_index += 1; update_check(current_index, len(payload['history']) == 1, detail); current_index += 1; fresh = train_module.build_model(payload['config'], 'cpu'); fresh.load_state_dict(payload['model'], strict=True); update_check(current_index, True, detail); current_index += 1
            else:
                current_index = len(CHECKPOINT_NAMES)
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks
