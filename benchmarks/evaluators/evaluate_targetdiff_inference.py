'''TargetDiff inference V2 evaluator adapter.'''
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Dict

import torch
from torch import nn

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
    '''Verify the candidate script and trusted TargetDiff assets.'''
    CHECKPOINT_NAMES = [
        'file:targetdiff_inference.py',
        'file:standard:inference_config.json',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_file = config.get('candidate_file', config.get('model_file', config.get('script', 'targetdiff_inference.py')))
        submission_path = (CANDIDATE_DIR / candidate_file).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        if not submission_path.is_file():
            raise FileNotFoundError(f'TargetDiff submission not found: {submission_path}')
        if not submission_path.read_text(encoding='utf-8').strip():
            raise ValueError(f'TargetDiff submission is empty: {submission_path}')
        update_check(current_index, True, str(submission_path))
        current_index += 1

        reference_path = (REFERENCE_DIR / 'targetdiff_inference.py').resolve()
        reference_path.relative_to(REFERENCE_DIR)
        if not reference_path.is_file():
            raise FileNotFoundError(f'Reference implementation not found: {reference_path}')
        asset_dir = reference_path.parent
        config_path = (REFERENCE_DIR / 'inference_config.json').resolve()
        config_path.relative_to(REFERENCE_DIR)
        if not config_path.is_file():
            raise FileNotFoundError('TargetDiff reference assets missing: [inference_config.json]')
        reference_config = json.loads(config_path.read_text(encoding='utf-8'))
        if not isinstance(reference_config, dict) or reference_config.get('model_class') != 'TargetDiff':
            raise ValueError('TargetDiff inference_config.json must identify the TargetDiff model')
        if int(reference_config.get('hidden_size', 0)) != 32 or int(reference_config.get('ligand_vocab', 0)) != 8:
            raise ValueError('TargetDiff config dimensions do not match the reference model')
        assets = reference_config.get('external_assets')
        if (
            not isinstance(assets, dict)
            or assets.get('mount_required') is not True
            or not isinstance(assets.get('weights'), str)
            or not assets['weights'].startswith('/public/share/')
            or not isinstance(assets.get('dataset'), str)
            or not assets['dataset'].startswith('/public/share/')
        ):
            raise ValueError('TargetDiff config must declare externally mounted weights and dataset')
        update_check(current_index, True, str(asset_dir / 'inference_config.json'))
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Verify the standalone source and CLI contract.'''
    CHECKPOINT_NAMES = ['evaluation_interface:source_and_cli_contract']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(script_path: Path, module_name: str):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

        candidate_file = config.get('candidate_file', config.get('model_file', 'targetdiff_inference.py'))
        submission_path = (CANDIDATE_DIR / candidate_file).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        source = submission_path.read_text(encoding='utf-8')
        syntax_tree = ast.parse(source, filename=str(submission_path))
        candidate_module = load_module(submission_path, 'targetdiff_candidate_structure')
        required_classes = {'EGNNRefineLayer', 'TargetDiff'}
        required_functions = {'parse_pdb_coordinates', 'parse_sdf_coordinates', 'write_sdf'}
        required_flags = {
            '--protein-path', '--ligand-path', '--checkpoint', '--out-dir',
            '--num-steps', '--seed', '--device',
        }
        imported_roots = set()
        called_attributes = set()
        parser_flags = set()
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split('.')[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called_attributes.add(node.func.attr)
                if node.func.attr == 'add_argument' and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        parser_flags.add(argument.value)
        if 'onescience' in imported_roots:
            raise AssertionError('The generated implementation must not import onescience')
        if 'eval' not in called_attributes:
            raise AssertionError('The inference CLI must switch the model to eval mode')
        if not {'no_grad', 'inference_mode'} & called_attributes:
            raise AssertionError('The inference CLI must use torch.no_grad or torch.inference_mode')
        missing_flags = sorted(required_flags - parser_flags)
        if missing_flags:
            raise AssertionError(f'Missing required CLI flags: {missing_flags}')
        if 'if __name__ == "__main__"' not in source:
            raise AssertionError('The script must provide a standalone entrypoint guard')
        missing_classes = sorted(name for name in required_classes if not hasattr(candidate_module, name))
        if missing_classes:
            raise AssertionError(f'Missing required model classes: {missing_classes}')
        for class_name in required_classes:
            model_class = getattr(candidate_module, class_name)
            if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
                raise AssertionError(f'{class_name} must inherit torch.nn.Module')
        missing_functions = sorted(name for name in required_functions if not callable(getattr(candidate_module, name, None)))
        if missing_functions:
            raise AssertionError(f'Missing required helper functions: {missing_functions}')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Verify strict checkpoint compatibility and checkpoint variants.'''
    CHECKPOINT_NAMES = [
        'checkpoint_loading:strict_reference_state_layout',
        'checkpoint_loading:raw_and_wrapped_state_dict',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(script_path: Path, module_name: str):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

        candidate_file = config.get('candidate_model_file', config.get('model_file', 'targetdiff_inference.py'))
        submission_path = (CANDIDATE_DIR / candidate_file).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / 'targetdiff_inference.py').resolve()
        reference_path.relative_to(REFERENCE_DIR)
        candidate_module = load_module(submission_path, 'targetdiff_candidate_checkpoint')
        reference_module = load_module(reference_path, 'targetdiff_reference_checkpoint')
        torch.manual_seed(20260820)
        reference_model = reference_module.TargetDiff()
        candidate_model = candidate_module.TargetDiff()
        reference_state = reference_model.state_dict()
        candidate_state = candidate_model.state_dict()
        if set(candidate_state) != set(reference_state):
            missing = sorted(set(reference_state) - set(candidate_state))
            unexpected = sorted(set(candidate_state) - set(reference_state))
            raise AssertionError(f'TargetDiff state layout mismatch; missing={missing}, unexpected={unexpected}')
        for name, reference_tensor in reference_state.items():
            candidate_tensor = candidate_state[name]
            if tuple(candidate_tensor.shape) != tuple(reference_tensor.shape):
                raise AssertionError(f'State tensor {name} shape mismatch: {tuple(candidate_tensor.shape)} != {tuple(reference_tensor.shape)}')
            if not torch.isfinite(candidate_tensor).all():
                raise AssertionError(f'State tensor {name} contains NaN or Inf')
        candidate_model.load_state_dict(reference_state, strict=True)
        reference_model.eval()
        candidate_model.eval()
        update_check(current_index, True, 'reference state dict loaded with strict=True')
        current_index += 1

        with tempfile.TemporaryDirectory(prefix='targetdiff_eval_') as temporary:
            work_root = Path(temporary)
            raw_path = work_root / 'reference_weights.pt'
            wrapped_path = work_root / 'wrapped_reference_weights.pt'
            invalid_path = work_root / 'invalid_reference_weights.pt'
            state = reference_model.state_dict()
            torch.save(state, raw_path)
            torch.save({'model_state_dict': state, 'step': 1}, wrapped_path)
            invalid_state = dict(state)
            invalid_state.pop(next(iter(invalid_state)))
            torch.save(invalid_state, invalid_path)
            if not raw_path.is_file() or not wrapped_path.is_file() or not invalid_path.is_file():
                raise AssertionError('raw, wrapped, and invalid checkpoint fixtures were not prepared')
        update_check(current_index, True, 'raw, wrapped, and invalid checkpoint fixtures prepared')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Verify the TargetDiff module layout.'''
    CHECKPOINT_NAMES = ['model_structure:targetdiff_modules']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(script_path: Path, module_name: str):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

        candidate_file = config.get('model_file', 'targetdiff_inference.py')
        submission_path = (CANDIDATE_DIR / candidate_file).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / 'targetdiff_inference.py').resolve()
        reference_path.relative_to(REFERENCE_DIR)
        candidate_module = load_module(submission_path, 'targetdiff_candidate_model_structure')
        reference_module = load_module(reference_path, 'targetdiff_reference_model_structure')
        torch.manual_seed(20260820)
        reference_model = reference_module.TargetDiff()
        candidate_model = candidate_module.TargetDiff()
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        reference_model.eval()
        candidate_model.eval()
        required_modules = (
            'protein_atom_emb', 'ligand_atom_emb', 'time_embedding', 'refine_net',
            'position_head', 'atom_type_head',
        )
        for name in required_modules:
            if not hasattr(candidate_model, name):
                raise AssertionError(f'TargetDiff is missing module {name}')
            candidate_type = type(getattr(candidate_model, name))
            reference_type = type(getattr(reference_model, name))
            if candidate_type.__name__ != reference_type.__name__:
                raise AssertionError(f'TargetDiff module type mismatch for {name}')
        if not isinstance(candidate_model.refine_net, nn.ModuleList):
            raise AssertionError('refine_net must be torch.nn.ModuleList')
        if len(candidate_model.refine_net) != len(reference_model.refine_net):
            raise AssertionError('Unexpected TargetDiff refinement depth')
        if candidate_model.hidden_size != reference_model.hidden_size:
            raise AssertionError('TargetDiff hidden_size differs from reference')
        if candidate_model.ligand_vocab != reference_model.ligand_vocab:
            raise AssertionError('TargetDiff ligand vocabulary differs from reference')
        if candidate_model.position_head.out_features != 3:
            raise AssertionError('position_head must emit xyz coordinates')
        if candidate_model.atom_type_head.out_features != 8:
            raise AssertionError('atom_type_head must emit eight atom-type logits')
        for index, layer in enumerate(candidate_model.refine_net):
            for name in ('edge_mlp', 'node_mlp', 'coordinate_gate'):
                if not hasattr(layer, name):
                    raise AssertionError(f'Refinement layer {index} is missing {name}')
            if layer.coordinate_gate.out_features != 1:
                raise AssertionError(f'Refinement layer {index} coordinate gate must be scalar')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Compare refinement, forward, activation, and atom-count behavior.'''
    CHECKPOINT_NAMES = [
        'model_structure:egnn_refinement_layers',
        'forward_executable:multi_time_forward',
        'forward_executable:intermediate_activations',
        'forward_executable:atom_count_sweep',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(script_path: Path, module_name: str):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

        def make_tensor_fixture():
            protein_positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float32)
            ligand_positions = torch.tensor([[0.4, 0.4, 0.2], [0.9, 0.4, 0.2], [0.4, 0.9, 0.2]], dtype=torch.float32)
            return protein_positions, protein_positions.clone(), ligand_positions, torch.tensor([0, 1, 2], dtype=torch.long)

        def assert_tensor_close(actual, expected, description: str) -> None:
            if not isinstance(actual, torch.Tensor):
                raise AssertionError(f'{description} must be a torch.Tensor')
            if tuple(actual.shape) != tuple(expected.shape):
                raise AssertionError(f'{description} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}')
            if actual.dtype != expected.dtype:
                raise AssertionError(f'{description} dtype mismatch: {actual.dtype} != {expected.dtype}')
            if not torch.isfinite(actual).all():
                raise AssertionError(f'{description} contains NaN or Inf')
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7, msg=lambda message: f'{description}: {message}')

        def assert_nested_close(actual, expected, description: str) -> None:
            if isinstance(expected, torch.Tensor):
                assert_tensor_close(actual, expected, description)
                return
            if isinstance(expected, (tuple, list)):
                if not isinstance(actual, type(expected)) or len(actual) != len(expected):
                    raise AssertionError(f'{description}: nested output structure mismatch')
                for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                    assert_nested_close(actual_item, expected_item, f'{description}[{index}]')
                return
            raise AssertionError(f'{description}: unsupported captured output type')

        def assert_output_close(actual, expected, description: str) -> None:
            output_keys = {'pred_ligand_pos', 'pred_ligand_v', 'final_h', 'final_ligand_h'}
            if not isinstance(actual, dict):
                raise AssertionError(f'{description}: forward output must be a dictionary')
            if set(actual) != output_keys:
                raise AssertionError(f'{description}: output keys mismatch; expected={sorted(output_keys)}, actual={sorted(actual)}')
            if set(expected) != output_keys:
                raise AssertionError('Reference output does not satisfy evaluator contract')
            for name in sorted(output_keys):
                assert_tensor_close(actual[name], expected[name], f'{description}, output={name}')

        candidate_file = config.get('model_file', 'targetdiff_inference.py')
        submission_path = (CANDIDATE_DIR / candidate_file).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / 'targetdiff_inference.py').resolve()
        reference_path.relative_to(REFERENCE_DIR)
        candidate_module = load_module(submission_path, 'targetdiff_candidate_forward')
        reference_module = load_module(reference_path, 'targetdiff_reference_forward')
        torch.manual_seed(20260820)
        reference_model = reference_module.TargetDiff()
        candidate_model = candidate_module.TargetDiff()
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        reference_model.eval()
        candidate_model.eval()

        torch.manual_seed(20260821)
        reference_layer = reference_module.EGNNRefineLayer(16).eval()
        candidate_layer = candidate_module.EGNNRefineLayer(16).eval()
        candidate_layer.load_state_dict(reference_layer.state_dict(), strict=True)
        hidden = torch.linspace(-1.0, 1.0, steps=4 * 16).reshape(4, 16)
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with torch.no_grad():
            reference_output = reference_layer(hidden, positions)
            candidate_output = candidate_layer(hidden, positions)
        assert_nested_close(candidate_output, reference_output, 'EGNNRefineLayer output')
        if tuple(candidate_output[0].shape) != (4, 16):
            raise AssertionError('EGNNRefineLayer hidden output has an invalid shape')
        if tuple(candidate_output[1].shape) != (4, 3):
            raise AssertionError('EGNNRefineLayer coordinate output has an invalid shape')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1

        protein, protein_features, ligand, ligand_types = make_tensor_fixture()
        test_cases = [
            (protein, protein_features, ligand, ligand_types, 0.0),
            (protein, protein_features, ligand, ligand_types, 0.25),
            (protein, protein_features, ligand, ligand_types, 0.5),
            (protein, protein_features, ligand, ligand_types, 1.0),
            (protein + 0.5, protein_features + 0.5, ligand - 0.25, torch.tensor([2, 1, 0]), 0.75),
        ]
        for case_index, arguments in enumerate(test_cases):
            with torch.no_grad():
                reference_output = reference_model(*arguments)
                candidate_output = candidate_model(*arguments)
            assert_output_close(candidate_output, reference_output, f'TargetDiff forward case {case_index}')
            expected_shapes = {
                'pred_ligand_pos': (arguments[2].shape[0], 3),
                'pred_ligand_v': (arguments[2].shape[0], 8),
                'final_h': (arguments[0].shape[0] + arguments[2].shape[0], 32),
                'final_ligand_h': (arguments[2].shape[0], 32),
            }
            for name, expected_shape in expected_shapes.items():
                if tuple(candidate_output[name].shape) != expected_shape:
                    raise AssertionError(f'Forward case {case_index} output {name} has shape {tuple(candidate_output[name].shape)}, expected {expected_shape}')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1

        reference_modules = {
            'protein_atom_emb': reference_model.protein_atom_emb,
            'ligand_atom_emb': reference_model.ligand_atom_emb,
            'time_embedding': reference_model.time_embedding,
            'refine_net.0': reference_model.refine_net[0],
            'refine_net.1': reference_model.refine_net[1],
            'position_head': reference_model.position_head,
            'atom_type_head': reference_model.atom_type_head,
        }
        candidate_modules = {
            'protein_atom_emb': candidate_model.protein_atom_emb,
            'ligand_atom_emb': candidate_model.ligand_atom_emb,
            'time_embedding': candidate_model.time_embedding,
            'refine_net.0': candidate_model.refine_net[0],
            'refine_net.1': candidate_model.refine_net[1],
            'position_head': candidate_model.position_head,
            'atom_type_head': candidate_model.atom_type_head,
        }
        def capture(module_map, model):
            activations = {}
            handles = []
            for stage_name, stage_module in module_map.items():
                handles.append(stage_module.register_forward_hook(lambda _module, _inputs, output, name=stage_name: activations.__setitem__(name, output)))
            try:
                with torch.no_grad():
                    model(*make_tensor_fixture(), time_step=0.5)
            finally:
                for handle in handles:
                    handle.remove()
            return activations
        reference_activations = capture(reference_modules, reference_model)
        candidate_activations = capture(candidate_modules, candidate_model)
        if set(candidate_activations) != set(reference_activations):
            raise AssertionError('Captured TargetDiff stages differ from reference')
        for name, reference_activation in reference_activations.items():
            assert_nested_close(candidate_activations[name], reference_activation, f'TargetDiff activation {name}')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1

        protein, protein_features, ligand, ligand_types = make_tensor_fixture()
        ligand_cases = [
            (ligand[:1], ligand_types[:1]),
            (ligand[:2], ligand_types[:2]),
            (ligand, ligand_types),
            (torch.cat([ligand, ligand[:2] + 0.1], dim=0), torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)),
        ]
        for ligand_positions, atom_types in ligand_cases:
            with torch.no_grad():
                reference_output = reference_model(protein, protein_features, ligand_positions, atom_types, time_step=0.5)
                candidate_output = candidate_model(protein, protein_features, ligand_positions, atom_types, time_step=0.5)
            assert_output_close(candidate_output, reference_output, f'ligand atom count {ligand_positions.shape[0]}')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Compare diffusion loss and coordinate parser behavior.'''
    CHECKPOINT_NAMES = [
        'forward_executable:diffusion_loss_gradients',
        'evaluation_interface:input_parsers',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(script_path: Path, module_name: str):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

        def make_tensor_fixture():
            protein_positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float32)
            ligand_positions = torch.tensor([[0.4, 0.4, 0.2], [0.9, 0.4, 0.2], [0.4, 0.9, 0.2]], dtype=torch.float32)
            return protein_positions, protein_positions.clone(), ligand_positions, torch.tensor([0, 1, 2], dtype=torch.long)

        def assert_tensor_close(actual, expected, description: str) -> None:
            if not isinstance(actual, torch.Tensor):
                raise AssertionError(f'{description} must be a torch.Tensor')
            if tuple(actual.shape) != tuple(expected.shape):
                raise AssertionError(f'{description} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}')
            if actual.dtype != expected.dtype:
                raise AssertionError(f'{description} dtype mismatch: {actual.dtype} != {expected.dtype}')
            if not torch.isfinite(actual).all():
                raise AssertionError(f'{description} contains NaN or Inf')
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7, msg=lambda message: f'{description}: {message}')

        def require_operation_failure(operation, description: str) -> None:
            try:
                operation()
            except Exception:
                return
            raise AssertionError(f'{description} unexpectedly succeeded')

        candidate_file = config.get('candidate_file', config.get('model_file', 'targetdiff_inference.py'))
        submission_path = (CANDIDATE_DIR / candidate_file).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / 'targetdiff_inference.py').resolve()
        reference_path.relative_to(REFERENCE_DIR)
        candidate_module = load_module(submission_path, 'targetdiff_candidate_methods')
        reference_module = load_module(reference_path, 'targetdiff_reference_methods')
        torch.manual_seed(20260820)
        reference_model = reference_module.TargetDiff()
        candidate_model = candidate_module.TargetDiff()
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        reference_model.eval()
        candidate_model.eval()

        protein, protein_features, ligand, ligand_types = make_tensor_fixture()
        loss_output_keys = {'loss_pos', 'loss_v', 'loss'}
        for time_step in (0.1, 0.5, 0.9):
            reference_loss = reference_model.get_diffusion_loss(protein, protein_features, ligand, ligand_types, time_step=time_step)
            candidate_loss = candidate_model.get_diffusion_loss(protein, protein_features, ligand, ligand_types, time_step=time_step)
            if set(candidate_loss) != loss_output_keys:
                raise AssertionError(f'Unexpected diffusion loss keys: {sorted(candidate_loss)}')
            for name in sorted(loss_output_keys):
                assert_tensor_close(candidate_loss[name], reference_loss[name], f'diffusion loss {name} at t={time_step}')
                if candidate_loss[name].ndim != 0:
                    raise AssertionError(f'Diffusion loss {name} must be scalar')
                if float(candidate_loss[name]) < 0.0:
                    raise AssertionError(f'Diffusion loss {name} must be non-negative')
            torch.testing.assert_close(candidate_loss['loss'], candidate_loss['loss_pos'] + candidate_loss['loss_v'], rtol=1e-6, atol=1e-7)
        candidate_model.zero_grad(set_to_none=True)
        train_loss = candidate_model.get_diffusion_loss(protein, protein_features, ligand, ligand_types, time_step=0.5)['loss']
        if not train_loss.requires_grad:
            raise AssertionError('Diffusion loss must retain a gradient graph')
        train_loss.backward()
        gradients = [parameter.grad for parameter in candidate_model.parameters() if parameter.requires_grad and parameter.grad is not None]
        if not gradients:
            raise AssertionError('Diffusion loss produced no parameter gradients')
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError('Diffusion loss produced non-finite gradients')
        candidate_model.zero_grad(set_to_none=True)
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1

        with tempfile.TemporaryDirectory(prefix='targetdiff_eval_') as temporary:
            work_root = Path(temporary)
            protein_positions, _, ligand_positions, _ = make_tensor_fixture()
            protein_path = work_root / 'protein.pdb'
            protein_lines = []
            for index, (x, y, z) in enumerate(protein_positions.tolist(), 1):
                protein_lines.append(f'ATOM  {index:5d}  CA  ALA A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C')
            protein_path.write_text('\n'.join(protein_lines) + '\n', encoding='utf-8')
            ligand_path = work_root / 'ligand.sdf'
            ligand_lines = ['ligand', 'OneScience evaluation', '', '  3  0  0  0  0  0  0  0  0999 V2000']
            ligand_lines.extend(f'{x:10.4f}{y:10.4f}{z:10.4f} C' for x, y, z in ligand_positions.tolist())
            ligand_lines.extend(['M  END', '$$$$'])
            ligand_path.write_text('\n'.join(ligand_lines) + '\n', encoding='utf-8')
            reference_protein = reference_module.parse_pdb_coordinates(protein_path)
            candidate_protein = candidate_module.parse_pdb_coordinates(protein_path)
            assert_tensor_close(candidate_protein, reference_protein, 'PDB coordinates')
            reference_ligand = reference_module.parse_sdf_coordinates(ligand_path)
            candidate_ligand = candidate_module.parse_sdf_coordinates(ligand_path)
            assert_tensor_close(candidate_ligand, reference_ligand, 'SDF coordinates')
            if tuple(candidate_protein.shape) != (4, 3):
                raise AssertionError(f'Unexpected parsed protein shape: {tuple(candidate_protein.shape)}')
            if tuple(candidate_ligand.shape) != (3, 3):
                raise AssertionError(f'Unexpected parsed ligand shape: {tuple(candidate_ligand.shape)}')
            malformed_pdb = work_root / 'malformed.pdb'
            malformed_pdb.write_text('ATOM invalid coordinates\n', encoding='utf-8')
            require_operation_failure(lambda: candidate_module.parse_pdb_coordinates(malformed_pdb), 'malformed PDB parsing')
            malformed_sdf = work_root / 'malformed.sdf'
            malformed_sdf.write_text('not an sdf\n', encoding='utf-8')
            require_operation_failure(lambda: candidate_module.parse_sdf_coordinates(malformed_sdf), 'malformed SDF parsing')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Compare seeded diffusion sampling against the reference implementation.'''
    CHECKPOINT_NAMES = ['metric:multi_seed_sampling']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(script_path: Path, module_name: str):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

        def make_tensor_fixture():
            protein_positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float32)
            ligand_positions = torch.tensor([[0.4, 0.4, 0.2], [0.9, 0.4, 0.2], [0.4, 0.9, 0.2]], dtype=torch.float32)
            return protein_positions, protein_positions.clone(), ligand_positions, torch.tensor([0, 1, 2], dtype=torch.long)

        def assert_tensor_close(actual, expected, description: str) -> None:
            if not isinstance(actual, torch.Tensor):
                raise AssertionError(f'{description} must be a torch.Tensor')
            if tuple(actual.shape) != tuple(expected.shape):
                raise AssertionError(f'{description} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}')
            if actual.dtype != expected.dtype:
                raise AssertionError(f'{description} dtype mismatch: {actual.dtype} != {expected.dtype}')
            if not torch.isfinite(actual).all():
                raise AssertionError(f'{description} contains NaN or Inf')
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7, msg=lambda message: f'{description}: {message}')

        candidate_file = config.get('model_file', 'targetdiff_inference.py')
        submission_path = (CANDIDATE_DIR / candidate_file).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / 'targetdiff_inference.py').resolve()
        reference_path.relative_to(REFERENCE_DIR)
        candidate_module = load_module(submission_path, 'targetdiff_candidate_sampling')
        reference_module = load_module(reference_path, 'targetdiff_reference_sampling')
        torch.manual_seed(20260820)
        reference_model = reference_module.TargetDiff()
        candidate_model = candidate_module.TargetDiff()
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        reference_model.eval()
        candidate_model.eval()
        protein, protein_features, ligand, ligand_types = make_tensor_fixture()
        for num_steps in (1, 2, 4, 6):
            for seed in (0, 7, 31):
                reference_sample = reference_model.sample_diffusion(protein, protein_features, ligand, ligand_types, num_steps=num_steps, seed=seed)
                candidate_sample = candidate_model.sample_diffusion(protein, protein_features, ligand, ligand_types, num_steps=num_steps, seed=seed)
                if set(candidate_sample) != {'pos', 'v', 'pos_traj'}:
                    raise AssertionError(f'Unexpected sampling keys: {sorted(candidate_sample)}')
                for name in ('pos', 'v', 'pos_traj'):
                    assert_tensor_close(candidate_sample[name], reference_sample[name], f'sample steps={num_steps}, seed={seed}, output={name}')
                expected_trajectory_shape = (num_steps, ligand.shape[0], 3)
                if tuple(candidate_sample['pos_traj'].shape) != expected_trajectory_shape:
                    raise AssertionError(f'Unexpected trajectory shape: {tuple(candidate_sample["pos_traj"].shape)}')
                if not torch.all((candidate_sample['v'] >= 0) & (candidate_sample['v'] < candidate_model.ligand_vocab)):
                    raise AssertionError('Sampled ligand atom type is out of range')
        first = candidate_model.sample_diffusion(protein, protein_features, ligand, ligand_types, num_steps=4, seed=9)
        repeated = candidate_model.sample_diffusion(protein, protein_features, ligand, ligand_types, num_steps=4, seed=9)
        different = candidate_model.sample_diffusion(protein, protein_features, ligand, ligand_types, num_steps=4, seed=10)
        torch.testing.assert_close(first['pos'], repeated['pos'], rtol=0.0, atol=0.0)
        torch.testing.assert_close(first['v'], repeated['v'], rtol=0.0, atol=0.0)
        if torch.equal(first['pos'], different['pos']):
            raise AssertionError('Different sampling seeds produced identical positions')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Verify invalid-input rejection and inference-state integrity.'''
    CHECKPOINT_NAMES = [
        'evaluation_interface:invalid_inputs',
        'evaluation_interface:state_integrity',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(script_path: Path, module_name: str):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

        def make_tensor_fixture():
            protein_positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float32)
            ligand_positions = torch.tensor([[0.4, 0.4, 0.2], [0.9, 0.4, 0.2], [0.4, 0.9, 0.2]], dtype=torch.float32)
            return protein_positions, protein_positions.clone(), ligand_positions, torch.tensor([0, 1, 2], dtype=torch.long)

        def require_operation_failure(operation, description: str) -> None:
            try:
                operation()
            except Exception:
                return
            raise AssertionError(f'{description} unexpectedly succeeded')

        def assert_tensor_close(actual, expected, description: str) -> None:
            if not isinstance(actual, torch.Tensor):
                raise AssertionError(f'{description} must be a torch.Tensor')
            if tuple(actual.shape) != tuple(expected.shape):
                raise AssertionError(f'{description} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}')
            if actual.dtype != expected.dtype:
                raise AssertionError(f'{description} dtype mismatch: {actual.dtype} != {expected.dtype}')
            if not torch.isfinite(actual).all():
                raise AssertionError(f'{description} contains NaN or Inf')
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7, msg=lambda message: f'{description}: {message}')

        def assert_output_close(actual, expected, description: str) -> None:
            output_keys = {'pred_ligand_pos', 'pred_ligand_v', 'final_h', 'final_ligand_h'}
            if not isinstance(actual, dict):
                raise AssertionError(f'{description}: forward output must be a dictionary')
            if set(actual) != output_keys:
                raise AssertionError(f'{description}: output keys mismatch; expected={sorted(output_keys)}, actual={sorted(actual)}')
            if set(expected) != output_keys:
                raise AssertionError('Reference output does not satisfy evaluator contract')
            for name in sorted(output_keys):
                assert_tensor_close(actual[name], expected[name], f'{description}, output={name}')

        candidate_file = config.get('model_file', 'targetdiff_inference.py')
        submission_path = (CANDIDATE_DIR / candidate_file).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / 'targetdiff_inference.py').resolve()
        reference_path.relative_to(REFERENCE_DIR)
        candidate_module = load_module(submission_path, 'targetdiff_candidate_forward_exec')
        reference_module = load_module(reference_path, 'targetdiff_reference_forward_exec')
        torch.manual_seed(20260820)
        reference_model = reference_module.TargetDiff()
        candidate_model = candidate_module.TargetDiff()
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        reference_model.eval()
        candidate_model.eval()
        protein, protein_features, ligand, ligand_types = make_tensor_fixture()
        invalid_calls = {
            'invalid protein coordinate width': lambda: candidate_model(protein[:, :2], protein_features, ligand, ligand_types),
            'invalid ligand coordinate width': lambda: candidate_model(protein, protein_features, ligand[:, :2], ligand_types),
            'invalid ligand type length': lambda: candidate_model(protein, protein_features, ligand, ligand_types[:2]),
            'invalid ligand type rank': lambda: candidate_model(protein, protein_features, ligand, ligand_types[:, None]),
            'out-of-range ligand type': lambda: candidate_model(protein, protein_features, ligand, torch.full_like(ligand_types, candidate_model.ligand_vocab)),
        }
        for description, operation in invalid_calls.items():
            require_operation_failure(operation, description)
        require_operation_failure(lambda: candidate_model.sample_diffusion(protein, protein_features, ligand, ligand_types, num_steps=0, seed=0), 'zero-step diffusion sampling')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1

        if any(module.training for module in candidate_model.modules()):
            raise AssertionError('TargetDiff or a child module remained in training mode')
        state_before = {name: value.detach().clone() for name, value in candidate_model.state_dict().items()}
        arguments = (*make_tensor_fixture(),)
        repeated_outputs = []
        for _ in range(3):
            with torch.inference_mode():
                repeated_outputs.append(candidate_model(*arguments, time_step=0.5))
        for repeat_index in range(1, len(repeated_outputs)):
            assert_output_close(repeated_outputs[repeat_index], repeated_outputs[0], f'deterministic forward repeat {repeat_index}')
        for name, expected in state_before.items():
            torch.testing.assert_close(candidate_model.state_dict()[name], expected, rtol=0.0, atol=0.0, msg=lambda message, parameter=name: f'Inference mutated parameter {parameter}: {message}')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_execution(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Execute the inference CLI against prepared checkpoint fixtures.'''
    CHECKPOINT_NAMES = ['metric:cli_outputs']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        def load_module(script_path: Path, module_name: str):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

        def make_tensor_fixture():
            protein_positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float32)
            ligand_positions = torch.tensor([[0.4, 0.4, 0.2], [0.9, 0.4, 0.2], [0.4, 0.9, 0.2]], dtype=torch.float32)
            return protein_positions, protein_positions.clone(), ligand_positions, torch.tensor([0, 1, 2], dtype=torch.long)

        def write_input_fixture(work_root: Path):
            protein_positions, _, ligand_positions, _ = make_tensor_fixture()
            protein_path = work_root / 'protein.pdb'
            protein_lines = []
            for index, (x, y, z) in enumerate(protein_positions.tolist(), 1):
                protein_lines.append(f'ATOM  {index:5d}  CA  ALA A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C')
            protein_path.write_text('\n'.join(protein_lines) + '\n', encoding='utf-8')
            ligand_path = work_root / 'ligand.sdf'
            ligand_lines = ['ligand', 'OneScience evaluation', '', '  3  0  0  0  0  0  0  0  0999 V2000']
            ligand_lines.extend(f'{x:10.4f}{y:10.4f}{z:10.4f} C' for x, y, z in ligand_positions.tolist())
            ligand_lines.extend(['M  END', '$$$$'])
            ligand_path.write_text('\n'.join(ligand_lines) + '\n', encoding='utf-8')
            return protein_path, ligand_path

        def execute_cli(script_path: Path, arguments: list[str]):
            command = [sys.executable, str(script_path), *arguments]
            completed = subprocess.run(command, cwd=script_path.parent, capture_output=True, text=True, timeout=180, check=False)
            return command, completed

        def require_cli_failure(script_path: Path, arguments: list[str], description: str) -> None:
            command, completed = execute_cli(script_path, arguments)
            if completed.returncode == 0:
                raise AssertionError(f'{description} unexpectedly succeeded: {" ".join(command)}')
            if not completed.stdout.strip() and not completed.stderr.strip():
                raise AssertionError(f'{description} failed without an error message')

        def run_cli(script_path: Path, protein_path: Path, ligand_path: Path, checkpoint_path: Path, output_dir: Path, num_steps: int, seed: int) -> None:
            arguments = [
                '--protein-path', str(protein_path), '--ligand-path', str(ligand_path),
                '--checkpoint', str(checkpoint_path), '--out-dir', str(output_dir),
                '--num-steps', str(num_steps), '--seed', str(seed), '--device', 'cpu',
            ]
            command, completed = execute_cli(script_path, arguments)
            if completed.returncode != 0:
                raise AssertionError(f'TargetDiff CLI failed\ncommand: {" ".join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}')

        candidate_file = config.get('script', 'targetdiff_inference.py')
        submission_path = (CANDIDATE_DIR / candidate_file).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / 'targetdiff_inference.py').resolve()
        reference_path.relative_to(REFERENCE_DIR)
        candidate_module = load_module(submission_path, 'targetdiff_candidate_cli')
        reference_module = load_module(reference_path, 'targetdiff_reference_cli')
        torch.manual_seed(20260820)
        reference_model = reference_module.TargetDiff()
        candidate_model = candidate_module.TargetDiff()
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        reference_model.eval()
        candidate_model.eval()
        with tempfile.TemporaryDirectory(prefix='targetdiff_eval_') as temporary:
            work_root = Path(temporary)
            raw_checkpoint_path = work_root / 'reference_weights.pt'
            wrapped_checkpoint_path = work_root / 'wrapped_reference_weights.pt'
            invalid_checkpoint_path = work_root / 'invalid_reference_weights.pt'
            state = reference_model.state_dict()
            torch.save(state, raw_checkpoint_path)
            torch.save({'model_state_dict': state, 'step': 1}, wrapped_checkpoint_path)
            invalid_state = dict(state)
            invalid_state.pop(next(iter(invalid_state)))
            torch.save(invalid_state, invalid_checkpoint_path)
            protein_path, ligand_path = write_input_fixture(work_root)
            configurations = [('raw', raw_checkpoint_path, 2, 3), ('wrapped', wrapped_checkpoint_path, 4, 9)]
            for name, checkpoint_path, num_steps, seed in configurations:
                output_dir = work_root / f'{name}_outputs'
                run_cli(submission_path, protein_path, ligand_path, checkpoint_path, output_dir, num_steps, seed)
                actual_files = {path.name for path in output_dir.iterdir()}
                expected_files = {'ligand_sample.sdf', 'scores.json'}
                if actual_files != expected_files:
                    raise AssertionError(f'Unexpected CLI files for {name}: expected={sorted(expected_files)}, actual={sorted(actual_files)}')
                coordinates = candidate_module.parse_sdf_coordinates(output_dir / 'ligand_sample.sdf')
                if tuple(coordinates.shape) != (3, 3):
                    raise AssertionError(f'CLI SDF has unexpected coordinate shape: {tuple(coordinates.shape)}')
                if not torch.isfinite(coordinates).all():
                    raise AssertionError('CLI SDF contains NaN or Inf coordinates')
                scores = json.loads((output_dir / 'scores.json').read_text(encoding='utf-8'))
                expected_scores = {'num_steps': num_steps, 'seed': seed, 'final_atom_count': 3}
                if scores != expected_scores:
                    raise AssertionError(f'Unexpected CLI score metadata: {scores} != {expected_scores}')
                protein, protein_features, ligand, ligand_types = make_tensor_fixture()
                expected = reference_model.sample_diffusion(protein, protein_features, ligand, torch.zeros_like(ligand_types), num_steps=num_steps, seed=seed)
                torch.testing.assert_close(coordinates, expected['pos'], rtol=0.0, atol=5e-5)
            common_arguments = [
                '--protein-path', str(protein_path), '--ligand-path', str(ligand_path),
                '--checkpoint', str(raw_checkpoint_path), '--out-dir', str(work_root / 'invalid_outputs'),
                '--seed', '1', '--device', 'cpu',
            ]
            require_cli_failure(submission_path, [*common_arguments, '--num-steps', '0'], 'zero-step CLI validation')
            require_cli_failure(submission_path, [*common_arguments[:5], str(invalid_checkpoint_path), *common_arguments[6:], '--num-steps', '2'], 'strict invalid-checkpoint validation')
            malformed_pdb = work_root / 'bad_cli.pdb'
            malformed_pdb.write_text('ATOM invalid coordinates\n', encoding='utf-8')
            require_cli_failure(submission_path, [
                '--protein-path', str(malformed_pdb), '--ligand-path', str(ligand_path),
                '--checkpoint', str(raw_checkpoint_path), '--out-dir', str(work_root / 'bad_pdb_outputs'),
                '--num-steps', '2', '--seed', '1', '--device', 'cpu',
            ], 'malformed PDB CLI validation')
        update_check(current_index, True, 'named TargetDiff comparison completed')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks
