'''V2 OpenFold inference evaluator adapter.'''
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
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'file:openfold_inference.py',
        'file:standard:inference_config.json',
        'reference_config:model.feature_dim',
        'reference_config:external_assets',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        required_files = config.get('required_files', ['openfold_inference.py'])
        submission_path = (CANDIDATE_DIR / required_files[0]).resolve()
        if CANDIDATE_DIR not in submission_path.parents:
            raise RuntimeError('Candidate file must resolve under CANDIDATE_DIR')
        if not submission_path.is_file():
            raise FileNotFoundError(f'openfold_inference.py was not found: {submission_path}')
        update_check(current_index, True, str(submission_path))
        current_index += 1

        reference_path = (REFERENCE_DIR / 'openfold_inference.py').resolve()
        config_path = (REFERENCE_DIR / 'inference_config.json').resolve()
        if REFERENCE_DIR not in reference_path.parents or REFERENCE_DIR not in config_path.parents:
            raise RuntimeError('Reference assets must resolve under REFERENCE_DIR')
        if not reference_path.is_file():
            raise FileNotFoundError(f'Reference implementation not found: {reference_path}')
        if not config_path.is_file():
            raise FileNotFoundError(f'OpenFold reference assets missing: {[config_path.name]}')
        update_check(current_index, True, str(config_path))
        current_index += 1

        reference_config = json.loads(config_path.read_text(encoding='utf-8'))
        model_config = reference_config.get('model') if isinstance(reference_config, dict) else None
        if not isinstance(model_config, dict) or model_config.get('feature_dim') != 22:
            raise ValueError('OpenFold config must declare feature_dim=22')
        update_check(current_index, True, 'feature_dim=22')
        current_index += 1

        assets = reference_config.get('external_assets') if isinstance(reference_config, dict) else None
        if (not isinstance(assets, dict) or assets.get('mount_required') is not True
                or not isinstance(assets.get('weights'), str)
                or not assets['weights'].startswith('/public/share/')
                or not isinstance(assets.get('dataset'), str)
                or not assets['dataset'].startswith('/public/share/')):
            raise ValueError('OpenFold config must declare externally mounted weights and dataset')
        update_check(current_index, True, 'mounted weights and dataset declared')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'source:no_onescience_import', 'source:model_eval_called',
        'source:gradient_context', 'source:required_cli_flags',
        'source:main_guard', 'source:required_module_classes',
        'source:module_base_classes', 'source:required_callable_helpers',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        script_path = (CANDIDATE_DIR / config.get('model_file', 'openfold_inference.py')).resolve()
        if CANDIDATE_DIR not in script_path.parents:
            raise RuntimeError('Candidate module must resolve under CANDIDATE_DIR')
        specification = importlib.util.spec_from_file_location('openfold_candidate_structure', script_path)
        if specification is None or specification.loader is None:
            raise ImportError(f'Cannot load Python module from {script_path}')
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        source = script_path.read_text(encoding='utf-8')
        syntax_tree = ast.parse(source, filename=str(script_path))
        imported_roots, called_attributes, parser_flags = set(), set(), set()
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split('.')[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called_attributes.add(node.func.attr)
                if node.func.attr == 'add_argument' and node.args:
                    first_argument = node.args[0]
                    if isinstance(first_argument, ast.Constant) and isinstance(first_argument.value, str):
                        parser_flags.add(first_argument.value)

        if 'onescience' in imported_roots:
            raise AssertionError('The generated implementation must not import onescience')
        update_check(current_index, True, 'onescience is not imported')
        current_index += 1
        if 'eval' not in called_attributes:
            raise AssertionError('The inference CLI must switch the model to eval mode')
        update_check(current_index, True, 'eval is called')
        current_index += 1
        if not {'no_grad', 'inference_mode'} & called_attributes:
            raise AssertionError('The inference CLI must use torch.no_grad or torch.inference_mode')
        update_check(current_index, True, 'gradient-disabled inference context is used')
        current_index += 1
        required_flags = {'--fasta', '--checkpoint', '--output-pdb', '--device'}
        missing_flags = sorted(required_flags - parser_flags)
        if missing_flags:
            raise AssertionError(f'Missing required CLI flags: {missing_flags}')
        update_check(current_index, True, 'all required CLI flags are present')
        current_index += 1
        if 'if __name__ == "__main__"' not in source:
            raise AssertionError('The script must provide a standalone entrypoint guard')
        update_check(current_index, True, 'standalone entrypoint guard is present')
        current_index += 1

        required_classes = {'InputEmbedder', 'RecyclingEmbedder', 'EvoformerBlock', 'StructureModule', 'OpenFold'}
        missing_classes = sorted(name for name in required_classes if not hasattr(module, name))
        if missing_classes:
            raise AssertionError(f'Missing required model classes: {missing_classes}')
        update_check(current_index, True, 'all required model classes are exported')
        current_index += 1
        for class_name in required_classes:
            model_class = getattr(module, class_name)
            if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
                raise AssertionError(f'{class_name} must inherit torch.nn.Module')
        update_check(current_index, True, 'all required classes inherit torch.nn.Module')
        current_index += 1
        required_functions = {'encode_sequence', 'make_features', 'write_pdb'}
        missing_functions = sorted(name for name in required_functions if not callable(getattr(module, name, None)))
        if missing_functions:
            raise AssertionError(f'Missing required helper functions: {missing_functions}')
        update_check(current_index, True, 'all required helper functions are callable')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'checkpoint_loading:state_layout',
        'checkpoint_loading:strict_reference_state_layout',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_path = (CANDIDATE_DIR / config.get('model_file', 'openfold_inference.py')).resolve()
        reference_path = (REFERENCE_DIR / 'openfold_inference.py').resolve()
        if CANDIDATE_DIR not in candidate_path.parents or REFERENCE_DIR not in reference_path.parents:
            raise RuntimeError('Model modules must resolve under their configured directories')
        def load_module(script_path, module_name):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module
        candidate_module = load_module(candidate_path, 'openfold_candidate_checkpoint')
        reference_module = load_module(reference_path, 'openfold_reference_checkpoint')
        model_args = config.get('model_args', {'num_recycles': 2, 'num_evoformer_blocks': 2})
        seed = config.get('seed', 20260820)
        torch.manual_seed(seed)
        reference_model = reference_module.OpenFold(**model_args)
        candidate_model = candidate_module.OpenFold(**model_args)
        reference_state = reference_model.state_dict()
        candidate_state = candidate_model.state_dict()
        if set(candidate_state) != set(reference_state):
            missing = sorted(set(reference_state) - set(candidate_state))
            unexpected = sorted(set(candidate_state) - set(reference_state))
            raise AssertionError(f'OpenFold state layout mismatch; missing={missing}, unexpected={unexpected}')
        for name, reference_value in reference_state.items():
            candidate_value = candidate_state[name]
            if tuple(candidate_value.shape) != tuple(reference_value.shape):
                raise AssertionError(f'State tensor {name} shape mismatch: {tuple(candidate_value.shape)} != {tuple(reference_value.shape)}')
            if not torch.isfinite(candidate_value).all():
                raise AssertionError(f'State tensor {name} contains NaN or Inf')
        update_check(current_index, True, 'reference-compatible finite state-dict layout')
        current_index += 1
        candidate_model.load_state_dict(reference_state, strict=True)
        reference_model.eval()
        candidate_model.eval()
        update_check(current_index, True, 'reference state dict loaded with strict=True')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'model:input_embedder_type', 'model:recycling_embedder_type',
        'model:evoformer_type', 'model:structure_module_type',
        'model:num_recycles', 'model:evoformer_modulelist',
        'model:evoformer_depth', 'model:input_projections',
        'model:recycling_norms', 'model:evoformer_block_attributes',
        'model:structure_coordinate_head',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_path = (CANDIDATE_DIR / config.get('model_file', 'openfold_inference.py')).resolve()
        reference_path = (REFERENCE_DIR / 'openfold_inference.py').resolve()
        if CANDIDATE_DIR not in candidate_path.parents or REFERENCE_DIR not in reference_path.parents:
            raise RuntimeError('Model modules must resolve under their configured directories')
        def load_module(script_path, module_name):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module
        reference_module = load_module(reference_path, 'openfold_reference_structure')
        candidate_module = load_module(candidate_path, 'openfold_candidate_model_structure')
        model_args = config.get('model_args', {'num_recycles': 2, 'num_evoformer_blocks': 2})
        torch.manual_seed(config.get('seed', 20260820))
        reference_model = reference_module.OpenFold(**model_args)
        candidate_model = candidate_module.OpenFold(**model_args)
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        reference_model.eval()
        candidate_model.eval()
        for name in ('input_embedder', 'recycling_embedder', 'evoformer', 'structure_module'):
            if not hasattr(candidate_model, name):
                raise AssertionError(f'OpenFold is missing module {name}')
            if type(getattr(candidate_model, name)).__name__ != type(getattr(reference_model, name)).__name__:
                raise AssertionError(f'OpenFold module type mismatch for {name}')
            update_check(current_index, True, f'{name} matches the reference type')
            current_index += 1
        if candidate_model.num_recycles != reference_model.num_recycles:
            raise AssertionError('OpenFold num_recycles differs from reference')
        update_check(current_index, True, 'num_recycles matches reference')
        current_index += 1
        if not isinstance(candidate_model.evoformer, nn.ModuleList):
            raise AssertionError('evoformer must be torch.nn.ModuleList')
        update_check(current_index, True, 'evoformer is torch.nn.ModuleList')
        current_index += 1
        if len(candidate_model.evoformer) != len(reference_model.evoformer):
            raise AssertionError('OpenFold Evoformer depth differs from reference')
        update_check(current_index, True, 'evoformer depth matches reference')
        current_index += 1
        input_embedder = candidate_model.input_embedder
        if input_embedder.single_projection.in_features != 22:
            raise AssertionError('Single input projection must consume 22 channels')
        if input_embedder.single_projection.out_features != 32:
            raise AssertionError('Single input projection must emit 32 channels')
        if input_embedder.pair_projection.in_features != 45:
            raise AssertionError('Pair input projection must consume 45 channels')
        if input_embedder.pair_projection.out_features != 32:
            raise AssertionError('Pair input projection must emit 32 channels')
        update_check(current_index, True, 'input projection dimensions are correct')
        current_index += 1
        recycling = candidate_model.recycling_embedder
        if not isinstance(recycling.single_norm, nn.LayerNorm):
            raise AssertionError('Recycling single branch must use LayerNorm')
        if not isinstance(recycling.pair_norm, nn.LayerNorm):
            raise AssertionError('Recycling pair branch must use LayerNorm')
        update_check(current_index, True, 'recycling branches use LayerNorm')
        current_index += 1
        for index, block in enumerate(candidate_model.evoformer):
            for name in ('single_norm', 'single_update', 'pair_norm', 'pair_update'):
                if not hasattr(block, name):
                    raise AssertionError(f'Evoformer block {index} is missing {name}')
        update_check(current_index, True, 'all Evoformer blocks expose required attributes')
        current_index += 1
        if candidate_model.structure_module.coordinate_head.out_features != 37 * 3:
            raise AssertionError('StructureModule must emit atom37 xyz coordinates')
        update_check(current_index, True, 'coordinate head emits atom37 xyz coordinates')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'method:encode_sequence', 'method:make_features',
        'component:InputEmbedder', 'component:RecyclingEmbedder',
        'component:EvoformerBlock', 'component:StructureModule',
        'method:write_pdb',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        RTOL, ATOL, SEED = 1e-6, 1e-7, config.get('seed_base', 20260820)
        candidate_path = (CANDIDATE_DIR / 'openfold_inference.py').resolve()
        reference_path = (REFERENCE_DIR / 'openfold_inference.py').resolve()
        if CANDIDATE_DIR not in candidate_path.parents or REFERENCE_DIR not in reference_path.parents:
            raise RuntimeError('Implementation modules must resolve under their configured directories')
        def load_module(script_path, module_name):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module
        def tensor_close(actual, expected, description):
            if not isinstance(actual, torch.Tensor):
                raise AssertionError(f'{description} must be a torch.Tensor')
            if tuple(actual.shape) != tuple(expected.shape):
                raise AssertionError(f'{description} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}')
            if actual.dtype != expected.dtype:
                raise AssertionError(f'{description} dtype mismatch: {actual.dtype} != {expected.dtype}')
            if not torch.isfinite(actual).all():
                raise AssertionError(f'{description} contains NaN or Inf')
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL, msg=lambda message: f'{description}: {message}')
        def nested_close(actual, expected, description):
            if isinstance(expected, torch.Tensor):
                tensor_close(actual, expected, description)
                return
            if isinstance(expected, (tuple, list)):
                if not isinstance(actual, type(expected)) or len(actual) != len(expected):
                    raise AssertionError(f'{description}: nested output structure mismatch')
                for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                    nested_close(actual_item, expected_item, f'{description}[{index}]')
                return
            raise AssertionError(f'{description}: unsupported captured output type')
        def require_failure(operation, description):
            try:
                operation()
            except Exception:
                return
            raise AssertionError(f'{description} unexpectedly succeeded')
        reference_module = load_module(reference_path, 'openfold_reference_methods')
        candidate_module = load_module(candidate_path, 'openfold_candidate_methods')

        for sequence in ['A', 'ACDE', ' acdefghik ', '\nMNPQRSTVWY\n']:
            reference_tokens = reference_module.encode_sequence(sequence)
            candidate_tokens = candidate_module.encode_sequence(sequence)
            tensor_close(candidate_tokens, reference_tokens, f'encoded sequence {sequence!r}')
            if candidate_tokens.dtype != torch.long:
                raise AssertionError('encode_sequence must return torch.long tokens')
            if candidate_tokens.ndim != 1:
                raise AssertionError('encode_sequence must return a rank-one tensor')
        for description, sequence in {'empty sequence': '', 'whitespace-only sequence': '  \n', 'non-standard residue': 'ACD1', 'ambiguous residue': 'ACDX', 'gap character': 'AC-D'}.items():
            require_failure(lambda value=sequence: candidate_module.encode_sequence(value), description)
        update_check(current_index, True, 'valid encodings match and invalid sequences are rejected')
        current_index += 1

        for sequence in ('A', 'ACDE', 'ACDEFGHIK'):
            for num_recycles in (1, 2, 3):
                reference_features = reference_module.make_features(sequence, num_recycles)
                candidate_features = candidate_module.make_features(sequence, num_recycles)
                if set(candidate_features) != {'aatype', 'target_feat', 'msa_feat', 'residue_index'}:
                    raise AssertionError(f'Unexpected OpenFold feature keys: {sorted(candidate_features)}')
                for name, reference_value in reference_features.items():
                    tensor_close(candidate_features[name], reference_value, f'feature={name}, L={len(sequence)}, R={num_recycles}')
                length = len(sequence)
                expected_shapes = {'aatype': (1, length, num_recycles), 'target_feat': (1, length, 22, num_recycles), 'msa_feat': (1, 2, length, 22, num_recycles), 'residue_index': (1, length, num_recycles)}
                for name, expected_shape in expected_shapes.items():
                    if tuple(candidate_features[name].shape) != expected_shape:
                        raise AssertionError(f'Feature {name} has shape {tuple(candidate_features[name].shape)}, expected {expected_shape}')
                if not torch.equal(candidate_features['aatype'][..., 0], candidate_features['aatype'][..., -1]):
                    raise AssertionError('aatype must be stable across recycle slices')
        update_check(current_index, True, 'feature values, shapes, and recycle stability match')
        current_index += 1

        torch.manual_seed(SEED + 1)
        reference_layer = reference_module.InputEmbedder(22, 16, 12).eval()
        candidate_layer = candidate_module.InputEmbedder(22, 16, 12).eval()
        candidate_layer.load_state_dict(reference_layer.state_dict(), strict=True)
        target = torch.linspace(-1.0, 1.0, steps=2 * 5 * 22).reshape(2, 5, 22)
        residue_index = torch.arange(5, dtype=torch.float32).repeat(2, 1)
        with torch.no_grad():
            reference_output = reference_layer(target, residue_index)
            candidate_output = candidate_layer(target, residue_index)
        nested_close(candidate_output, reference_output, 'InputEmbedder output')
        if tuple(candidate_output[0].shape) != (2, 5, 16) or tuple(candidate_output[1].shape) != (2, 5, 5, 12):
            raise AssertionError('InputEmbedder output shapes are invalid')
        update_check(current_index, True, 'InputEmbedder oracle and shapes match')
        current_index += 1

        torch.manual_seed(SEED + 2)
        reference_layer = reference_module.RecyclingEmbedder(16, 12).eval()
        candidate_layer = candidate_module.RecyclingEmbedder(16, 12).eval()
        candidate_layer.load_state_dict(reference_layer.state_dict(), strict=True)
        single = torch.linspace(-0.5, 0.5, steps=2 * 5 * 16).reshape(2, 5, 16)
        pair = torch.linspace(-0.75, 0.75, steps=2 * 5 * 5 * 12).reshape(2, 5, 5, 12)
        for index, arguments in enumerate([(single, pair, None, None), (single, pair, single.flip(-1), pair.flip(-1))]):
            with torch.no_grad():
                reference_output = reference_layer(*arguments)
                candidate_output = candidate_layer(*arguments)
            nested_close(candidate_output, reference_output, f'RecyclingEmbedder case {index}')
        update_check(current_index, True, 'RecyclingEmbedder fixture cases match')
        current_index += 1

        torch.manual_seed(SEED + 3)
        reference_block = reference_module.EvoformerBlock(16, 12).eval()
        candidate_block = candidate_module.EvoformerBlock(16, 12).eval()
        candidate_block.load_state_dict(reference_block.state_dict(), strict=True)
        single = torch.linspace(-1.0, 1.0, steps=2 * 5 * 16).reshape(2, 5, 16)
        pair = torch.linspace(-1.0, 1.0, steps=2 * 5 * 5 * 12).reshape(2, 5, 5, 12)
        with torch.no_grad():
            reference_output = reference_block(single, pair)
            candidate_output = candidate_block(single, pair)
        nested_close(candidate_output, reference_output, 'EvoformerBlock output')
        if tuple(candidate_output[0].shape) != tuple(single.shape) or tuple(candidate_output[1].shape) != tuple(pair.shape):
            raise AssertionError('EvoformerBlock must preserve representation shapes')
        update_check(current_index, True, 'EvoformerBlock oracle and shapes match')
        current_index += 1

        torch.manual_seed(SEED + 4)
        reference_head = reference_module.StructureModule(16).eval()
        candidate_head = candidate_module.StructureModule(16).eval()
        candidate_head.load_state_dict(reference_head.state_dict(), strict=True)
        single = torch.linspace(-1.0, 1.0, steps=2 * 5 * 16).reshape(2, 5, 16)
        with torch.no_grad():
            reference_output = reference_head(single)
            candidate_output = candidate_head(single)
        expected_keys = {'final_atom_positions', 'final_atom_mask', 'final_affine_tensor'}
        if set(candidate_output) != expected_keys:
            raise AssertionError(f'Unexpected StructureModule keys: {sorted(candidate_output)}')
        for name in sorted(expected_keys):
            tensor_close(candidate_output[name], reference_output[name], f'StructureModule output {name}')
        if tuple(candidate_output['final_atom_positions'].shape) != (2, 5, 37, 3):
            raise AssertionError('StructureModule atom37 output shape is invalid')
        if not torch.all(candidate_output['final_atom_mask'] == 1):
            raise AssertionError('StructureModule fixture atom mask must be all valid')
        update_check(current_index, True, 'StructureModule oracle, keys, shape, and mask match')
        current_index += 1

        torch.manual_seed(SEED)
        reference_model = reference_module.OpenFold(num_recycles=2, num_evoformer_blocks=2)
        candidate_model = candidate_module.OpenFold(num_recycles=2, num_evoformer_blocks=2)
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        reference_model.eval()
        candidate_model.eval()
        with tempfile.TemporaryDirectory(prefix='openfold_eval_') as temporary:
            output_path = Path(temporary) / 'writer_output.pdb'
            coordinates = candidate_model(candidate_module.make_features('ACDEFG', 2))['final_atom_positions']
            candidate_module.write_pdb(coordinates, output_path)
            if not output_path.is_file():
                raise AssertionError('write_pdb did not create the output file')
            lines = output_path.read_text(encoding='utf-8').splitlines()
            atom_lines = [line for line in lines if line.startswith('ATOM')]
            if len(atom_lines) != coordinates.shape[1]:
                raise AssertionError(f'PDB atom count mismatch: {len(atom_lines)} != {coordinates.shape[1]}')
            if not lines or lines[-1] != 'END':
                raise AssertionError('PDB output must terminate with END')
            for index, line in enumerate(atom_lines, 1):
                if int(line[6:11]) != index:
                    raise AssertionError('PDB atom serial numbers must be contiguous')
                values = torch.tensor([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                if not torch.isfinite(values).all():
                    raise AssertionError('PDB coordinate field contains NaN or Inf')
        update_check(current_index, True, 'PDB writer output is finite and structurally valid')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'forward:reference_equivalence', 'forward:intermediate_activations',
        'forward:recycle_configurations', 'forward:batch_isolation',
        'forward:invalid_inputs', 'forward:gradient_contract',
        'forward:inference_state_integrity',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        RTOL, ATOL, SEED = 1e-6, 1e-7, config.get('seed', 20260820)
        candidate_path = (CANDIDATE_DIR / 'openfold_inference.py').resolve()
        reference_path = (REFERENCE_DIR / 'openfold_inference.py').resolve()
        if CANDIDATE_DIR not in candidate_path.parents or REFERENCE_DIR not in reference_path.parents:
            raise RuntimeError('Implementation modules must resolve under their configured directories')
        def load_module(script_path, module_name):
            specification = importlib.util.spec_from_file_location(module_name, script_path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {script_path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module
        def tensor_close(actual, expected, description):
            if not isinstance(actual, torch.Tensor):
                raise AssertionError(f'{description} must be a torch.Tensor')
            if tuple(actual.shape) != tuple(expected.shape):
                raise AssertionError(f'{description} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}')
            if actual.dtype != expected.dtype:
                raise AssertionError(f'{description} dtype mismatch: {actual.dtype} != {expected.dtype}')
            if not torch.isfinite(actual).all():
                raise AssertionError(f'{description} contains NaN or Inf')
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL, msg=lambda message: f'{description}: {message}')
        def output_close(actual, expected, description):
            expected_keys = {'final_atom_positions', 'final_atom_mask', 'final_affine_tensor', 'single', 'pair', 'num_recycles'}
            if not isinstance(actual, dict):
                raise AssertionError(f'{description}: forward output must be a dictionary')
            if set(actual) != expected_keys:
                raise AssertionError(f'{description}: output keys mismatch; expected={sorted(expected_keys)}, actual={sorted(actual)}')
            for name in sorted(expected_keys):
                tensor_close(actual[name], expected[name], f'{description}, output={name}')
        def nested_close(actual, expected, description):
            if isinstance(expected, torch.Tensor):
                tensor_close(actual, expected, description)
                return
            if isinstance(expected, (tuple, list)):
                if not isinstance(actual, type(expected)) or len(actual) != len(expected):
                    raise AssertionError(f'{description}: nested output structure mismatch')
                for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                    nested_close(actual_item, expected_item, f'{description}[{index}]')
                return
            raise AssertionError(f'{description}: unsupported captured output type')
        def require_failure(operation, description):
            try:
                operation()
            except Exception:
                return
            raise AssertionError(f'{description} unexpectedly succeeded')
        reference_module = load_module(reference_path, 'openfold_reference_forward')
        candidate_module = load_module(candidate_path, 'openfold_candidate_forward')
        model_args = config.get('model_args', {'num_recycles': 2, 'num_evoformer_blocks': 2})
        torch.manual_seed(SEED)
        reference_model = reference_module.OpenFold(**model_args)
        candidate_model = candidate_module.OpenFold(**model_args)
        reference_state = reference_model.state_dict()
        candidate_state = candidate_model.state_dict()
        if set(candidate_state) != set(reference_state):
            raise AssertionError('OpenFold state layout mismatch')
        for name, reference_value in reference_state.items():
            candidate_value = candidate_state[name]
            if tuple(candidate_value.shape) != tuple(reference_value.shape):
                raise AssertionError(f'State tensor {name} shape mismatch')
            if not torch.isfinite(candidate_value).all():
                raise AssertionError(f'State tensor {name} contains NaN or Inf')
        candidate_model.load_state_dict(reference_state, strict=True)
        reference_model.eval()
        candidate_model.eval()

        for sequence in ('A', 'ACDE', 'ACDEFG', 'MNPQRSTVWY'):
            reference_features = reference_module.make_features(sequence, 2)
            candidate_features = candidate_module.make_features(sequence, 2)
            with torch.no_grad():
                reference_output = reference_model(reference_features)
                candidate_output = candidate_model(candidate_features)
            output_close(candidate_output, reference_output, f'OpenFold forward sequence={sequence}')
            length = len(sequence)
            expected_shapes = {'final_atom_positions': (1, length, 37, 3), 'final_atom_mask': (1, length, 37), 'final_affine_tensor': (1, length, 3), 'single': (1, length, 32), 'pair': (1, length, length, 32), 'num_recycles': ()}
            for name, expected_shape in expected_shapes.items():
                if tuple(candidate_output[name].shape) != expected_shape:
                    raise AssertionError(f'Forward output {name} has shape {tuple(candidate_output[name].shape)}, expected {expected_shape}')
            if int(candidate_output['num_recycles']) != 2:
                raise AssertionError('Forward output reports an invalid recycle count')
        update_check(current_index, True, 'forward values, keys, dtypes, finiteness, and shapes match')
        current_index += 1

        reference_modules = {'input_embedder': reference_model.input_embedder, 'recycling_embedder': reference_model.recycling_embedder, 'evoformer.0': reference_model.evoformer[0], 'evoformer.1': reference_model.evoformer[1], 'structure_module': reference_model.structure_module}
        candidate_modules = {'input_embedder': candidate_model.input_embedder, 'recycling_embedder': candidate_model.recycling_embedder, 'evoformer.0': candidate_model.evoformer[0], 'evoformer.1': candidate_model.evoformer[1], 'structure_module': candidate_model.structure_module}
        def capture(module_map, model, features):
            activations, handles = {}, []
            for stage_name, stage_module in module_map.items():
                handles.append(stage_module.register_forward_hook(lambda _module, _inputs, output, name=stage_name: activations.__setitem__(name, output)))
            try:
                with torch.no_grad():
                    model(features)
            finally:
                for handle in handles:
                    handle.remove()
            return activations
        reference_activations = capture(reference_modules, reference_model, reference_module.make_features('ACDEFG', 2))
        candidate_activations = capture(candidate_modules, candidate_model, candidate_module.make_features('ACDEFG', 2))
        if set(candidate_activations) != set(reference_activations):
            raise AssertionError('Captured OpenFold stages differ from reference')
        for name, reference_activation in reference_activations.items():
            if isinstance(reference_activation, dict):
                for output_name, reference_value in reference_activation.items():
                    tensor_close(candidate_activations[name][output_name], reference_value, f'OpenFold activation {name}.{output_name}')
            else:
                nested_close(candidate_activations[name], reference_activation, f'OpenFold activation {name}')
        update_check(current_index, True, 'intermediate activations match')
        current_index += 1

        for num_recycles in (1, 2, 3):
            torch.manual_seed(SEED)
            recycle_reference = reference_module.OpenFold(num_recycles=num_recycles, num_evoformer_blocks=2)
            recycle_candidate = candidate_module.OpenFold(num_recycles=num_recycles, num_evoformer_blocks=2)
            recycle_candidate.load_state_dict(recycle_reference.state_dict(), strict=True)
            recycle_reference.eval()
            recycle_candidate.eval()
            for sequence in ('ACDE', 'ACDEFGH'):
                with torch.no_grad():
                    reference_output = recycle_reference(reference_module.make_features(sequence, num_recycles))
                    candidate_output = recycle_candidate(candidate_module.make_features(sequence, num_recycles))
                output_close(candidate_output, reference_output, f'OpenFold recycles={num_recycles}, sequence={sequence}')
                if int(candidate_output['num_recycles']) != num_recycles:
                    raise AssertionError(f'Expected {num_recycles} recycles, got {int(candidate_output["num_recycles"])}')
        update_check(current_index, True, 'one-, two-, and three-cycle configurations match')
        current_index += 1

        sequence = 'ACDEFG'
        reference_features = reference_module.make_features(sequence, 2)
        candidate_features = candidate_module.make_features(sequence, 2)
        reference_batch = {name: value.repeat(3, *([1] * (value.ndim - 1))) for name, value in reference_features.items()}
        candidate_batch = {name: value.repeat(3, *([1] * (value.ndim - 1))) for name, value in candidate_features.items()}
        with torch.no_grad():
            reference_output = reference_model(reference_batch)
            candidate_output = candidate_model(candidate_batch)
            single_output = candidate_model(candidate_features)
        output_close(candidate_output, reference_output, 'OpenFold batch output')
        for name in ('final_atom_positions', 'final_atom_mask', 'final_affine_tensor', 'single', 'pair'):
            for batch_index in range(3):
                torch.testing.assert_close(candidate_output[name][batch_index], single_output[name][0], rtol=RTOL, atol=1e-6, msg=lambda message, output_name=name, index=batch_index: f'Batch isolation failed for {output_name}[{index}]: {message}')
        update_check(current_index, True, 'batched examples match isolated execution')
        current_index += 1

        features = candidate_module.make_features('ACDEFG', 2)
        invalid_batches = {}
        wrong_recycle = dict(features); wrong_recycle['target_feat'] = features['target_feat'][..., :1]; invalid_batches['target recycle mismatch'] = wrong_recycle
        missing_index = dict(features); missing_index.pop('residue_index'); invalid_batches['missing residue_index'] = missing_index
        invalid_width = dict(features); invalid_width['target_feat'] = features['target_feat'][:, :, :21, :]; invalid_batches['target feature width mismatch'] = invalid_width
        invalid_rank = dict(features); invalid_rank['residue_index'] = features['residue_index'][0]; invalid_batches['residue index rank mismatch'] = invalid_rank
        for description, batch in invalid_batches.items():
            require_failure(lambda value=batch: candidate_model(value), description)
        update_check(current_index, True, 'malformed feature dictionaries are rejected')
        current_index += 1

        candidate_model.zero_grad(set_to_none=True)
        output = candidate_model(features)
        objective = output['final_atom_positions'].square().mean() + output['single'].square().mean() + output['pair'].square().mean()
        if not objective.requires_grad:
            raise AssertionError('OpenFold forward outputs must retain gradients')
        objective.backward()
        gradients = [parameter.grad for parameter in candidate_model.parameters() if parameter.requires_grad and parameter.grad is not None]
        if not gradients:
            raise AssertionError('OpenFold forward produced no parameter gradients')
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError('OpenFold forward produced non-finite gradients')
        candidate_model.zero_grad(set_to_none=True)
        update_check(current_index, True, 'finite parameter gradients are retained')
        current_index += 1

        if any(module.training for module in candidate_model.modules()):
            raise AssertionError('OpenFold or a child module remained in training mode')
        state_before = {name: value.detach().clone() for name, value in candidate_model.state_dict().items()}
        repeated_outputs = []
        features = candidate_module.make_features('ACDEFGHIK', 2)
        for _ in range(3):
            with torch.inference_mode():
                repeated_outputs.append(candidate_model(features))
        for repeat_index in range(1, len(repeated_outputs)):
            output_close(repeated_outputs[repeat_index], repeated_outputs[0], f'OpenFold deterministic repeat {repeat_index}')
        for name, expected in state_before.items():
            torch.testing.assert_close(candidate_model.state_dict()[name], expected, rtol=0.0, atol=0.0, msg=lambda message, parameter=name: f'Inference mutated parameter {parameter}: {message}')
        update_check(current_index, True, 'inference is deterministic and state preserving')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_execution(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'cli:checkpoint_fixtures', 'cli:raw_checkpoint',
        'cli:wrapped_checkpoint', 'cli:invalid_fasta_rejected',
        'cli:invalid_checkpoint_rejected', 'cli:missing_fasta_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        script_path = (CANDIDATE_DIR / config.get('run_file', 'openfold_inference.py')).resolve()
        reference_path = (REFERENCE_DIR / 'openfold_inference.py').resolve()
        if CANDIDATE_DIR not in script_path.parents or REFERENCE_DIR not in reference_path.parents:
            raise RuntimeError('CLI and reference modules must resolve under their configured directories')
        def load_module(path, name):
            specification = importlib.util.spec_from_file_location(name, path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module
        def execute_cli(arguments):
            command = [sys.executable, str(script_path), *arguments]
            completed = subprocess.run(command, cwd=script_path.parent, capture_output=True, text=True, timeout=180, check=False)
            return command, completed
        def run_cli(fasta_path, checkpoint_path, output_path):
            command, completed = execute_cli(['--fasta', str(fasta_path), '--checkpoint', str(checkpoint_path), '--output-pdb', str(output_path), '--device', 'cpu'])
            if completed.returncode != 0:
                raise AssertionError(f'OpenFold CLI failed\ncommand: {" ".join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}')
        def require_cli_failure(arguments, description):
            command, completed = execute_cli(arguments)
            if completed.returncode == 0:
                raise AssertionError(f'{description} unexpectedly succeeded: {" ".join(command)}')
            if not completed.stdout.strip() and not completed.stderr.strip():
                raise AssertionError(f'{description} failed without an error message')
        def read_coordinates(path):
            coordinates = []
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.startswith('ATOM'):
                    coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if not coordinates:
                raise AssertionError(f'PDB contains no ATOM records: {path}')
            return torch.tensor(coordinates, dtype=torch.float32)
        reference_module = load_module(reference_path, 'openfold_reference_cli')
        torch.manual_seed(config.get('seed', 20260820))
        reference_model = reference_module.OpenFold(num_recycles=2, num_evoformer_blocks=2).eval()
        with tempfile.TemporaryDirectory(prefix='openfold_eval_') as temporary:
            work_root = Path(temporary)
            raw_checkpoint = work_root / 'reference_weights.pt'
            wrapped_checkpoint = work_root / 'wrapped_reference_weights.pt'
            invalid_checkpoint = work_root / 'invalid_reference_weights.pt'
            state = reference_model.state_dict()
            torch.save(state, raw_checkpoint)
            torch.save({'model_state_dict': state, 'epoch': 1}, wrapped_checkpoint)
            invalid_state = dict(state)
            invalid_state.pop(next(iter(invalid_state)))
            torch.save(invalid_state, invalid_checkpoint)
            update_check(current_index, True, 'raw, wrapped, and invalid checkpoint fixtures prepared')
            current_index += 1
            fasta_path = work_root / 'input.fasta'
            fasta_path.write_text('>openfold_fixture\nACDEFGHIK\n', encoding='utf-8')
            for name, checkpoint_path in [('raw', raw_checkpoint), ('wrapped', wrapped_checkpoint)]:
                output_path = work_root / name / 'structure.pdb'
                run_cli(fasta_path, checkpoint_path, output_path)
                if not output_path.is_file():
                    raise AssertionError(f'OpenFold CLI did not create {output_path}')
                actual_coordinates = read_coordinates(output_path)
                if tuple(actual_coordinates.shape) != (9, 3):
                    raise AssertionError(f'Unexpected CLI PDB shape: {tuple(actual_coordinates.shape)}')
                if not torch.isfinite(actual_coordinates).all():
                    raise AssertionError('OpenFold CLI PDB contains NaN or Inf')
                lines = output_path.read_text(encoding='utf-8').splitlines()
                if lines[-1] != 'END':
                    raise AssertionError('OpenFold CLI PDB must terminate with END')
                update_check(current_index, True, f'{name} checkpoint produced valid PDB output')
                current_index += 1
            invalid_fasta = work_root / 'invalid.fasta'
            invalid_fasta.write_text('>openfold_fixture\nACD1\n', encoding='utf-8')
            common_arguments = ['--checkpoint', str(raw_checkpoint), '--output-pdb', str(work_root / 'invalid.pdb'), '--device', 'cpu']
            require_cli_failure(['--fasta', str(invalid_fasta), *common_arguments], 'invalid FASTA CLI validation')
            update_check(current_index, True, 'invalid FASTA is rejected with an error')
            current_index += 1
            require_cli_failure(['--fasta', str(fasta_path), '--checkpoint', str(invalid_checkpoint), '--output-pdb', str(work_root / 'invalid_checkpoint.pdb'), '--device', 'cpu'], 'strict invalid-checkpoint validation')
            update_check(current_index, True, 'invalid checkpoint is rejected with an error')
            current_index += 1
            require_cli_failure(['--fasta', str(work_root / 'missing.fasta'), *common_arguments], 'missing FASTA CLI validation')
            update_check(current_index, True, 'missing FASTA is rejected with an error')
            current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['oracle:raw_checkpoint_coordinates', 'oracle:wrapped_checkpoint_coordinates']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        script_path = (CANDIDATE_DIR / config.get('run_file', 'openfold_inference.py')).resolve()
        reference_path = (REFERENCE_DIR / 'openfold_inference.py').resolve()
        if CANDIDATE_DIR not in script_path.parents or REFERENCE_DIR not in reference_path.parents:
            raise RuntimeError('CLI and reference modules must resolve under their configured directories')
        def load_module(path, name):
            specification = importlib.util.spec_from_file_location(name, path)
            if specification is None or specification.loader is None:
                raise ImportError(f'Cannot load Python module from {path}')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module
        def read_coordinates(path):
            coordinates = []
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.startswith('ATOM'):
                    coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if not coordinates:
                raise AssertionError(f'PDB contains no ATOM records: {path}')
            return torch.tensor(coordinates, dtype=torch.float32)
        reference_module = load_module(reference_path, 'openfold_reference_oracle')
        torch.manual_seed(config.get('seed', 20260820))
        reference_model = reference_module.OpenFold(num_recycles=2, num_evoformer_blocks=2).eval()
        with tempfile.TemporaryDirectory(prefix='openfold_eval_') as temporary:
            work_root = Path(temporary)
            state = reference_model.state_dict()
            raw_checkpoint = work_root / 'reference_weights.pt'
            wrapped_checkpoint = work_root / 'wrapped_reference_weights.pt'
            torch.save(state, raw_checkpoint)
            torch.save({'model_state_dict': state, 'epoch': 1}, wrapped_checkpoint)
            fasta_path = work_root / 'input.fasta'
            sequence = 'ACDEFGHIK'
            fasta_path.write_text(f'>openfold_fixture\n{sequence}\n', encoding='utf-8')
            for name, checkpoint_path in [('raw', raw_checkpoint), ('wrapped', wrapped_checkpoint)]:
                output_path = work_root / name / 'structure.pdb'
                command = [sys.executable, str(script_path), '--fasta', str(fasta_path), '--checkpoint', str(checkpoint_path), '--output-pdb', str(output_path), '--device', 'cpu']
                completed = subprocess.run(command, cwd=script_path.parent, capture_output=True, text=True, timeout=180, check=False)
                if completed.returncode != 0:
                    raise AssertionError(f'OpenFold CLI failed\ncommand: {" ".join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}')
                if not output_path.is_file():
                    raise AssertionError(f'OpenFold CLI did not create {output_path}')
                actual_coordinates = read_coordinates(output_path)
                if tuple(actual_coordinates.shape) != (len(sequence), 3):
                    raise AssertionError(f'Unexpected CLI PDB shape: {tuple(actual_coordinates.shape)}')
                if not torch.isfinite(actual_coordinates).all():
                    raise AssertionError('OpenFold CLI PDB contains NaN or Inf')
                with torch.no_grad():
                    expected = reference_model(reference_module.make_features(sequence, 2))['final_atom_positions'][0, :, 1]
                torch.testing.assert_close(actual_coordinates, expected, rtol=0.0, atol=5e-4)
                lines = output_path.read_text(encoding='utf-8').splitlines()
                if lines[-1] != 'END':
                    raise AssertionError('OpenFold CLI PDB must terminate with END')
                update_check(current_index, True, f'{name} CLI CA coordinates match reference')
                current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks
