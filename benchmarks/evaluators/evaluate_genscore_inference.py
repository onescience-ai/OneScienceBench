'''V2 adapter for the legacy GenScore inference evaluator.'''
from pathlib import Path
import sys
import ast
import importlib.util
import inspect
import json
import os
import subprocess
import tempfile
import traceback
from typing import Dict

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
    '''
    Checkpoint implementation.
    '''
    CHECKPOINT_NAMES = [
        'file:genscore_inference.py',
        'file:standard:inference_config.json',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        FILENAME = config.get('submission_filename', 'genscore_inference.py')
        STANDARD = REFERENCE_DIR / config.get('reference_filename', 'genscore_inference.py')

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(path)
            sys.path.insert(0, str(path.parent))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        path = CANDIDATE_DIR / FILENAME
        candidate = load_module(config.get('candidate_module_name', 'genscore_candidate'), path)
        reference = load_module(config.get('reference_module_name', 'genscore_reference'), STANDARD)
        assets = json.loads(
            (REFERENCE_DIR / config.get('standard_config', 'inference_config.json')).read_text(encoding='utf-8')
        )['external_assets']
        ok = (
            assets.get('mount_required') is True
            and Path(assets['weights']).exists()
            and Path(assets['dataset']).exists()
        )
        update_check(current_index, True, str(path))
        current_index += 1
        update_check(current_index, ok, str(assets))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''
    Checkpoint implementation.
    '''
    CHECKPOINT_NAMES = [
        'checkpoint_loading:strict_state_dict',
        'checkpoint_loading:uses_torch_load',
        'checkpoint_loading:uses_load_state_dict',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        FILENAME = config.get('submission_filename', 'genscore_inference.py')
        STANDARD = REFERENCE_DIR / config.get('reference_filename', 'genscore_inference.py')
        CTOR = config.get(
            'constructor_args',
            {
                'ligand_features': 8,
                'pocket_features': 10,
                'hidden_size': 32,
                'radial_bins': 8,
                'num_layers': 2,
            },
        )

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(path)
            sys.path.insert(0, str(path.parent))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        def source_info(path):
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source)
            imports = any(
                (
                    isinstance(node, ast.Import)
                    and any(alias.name.split('.')[0] == 'onescience' for alias in node.names)
                )
                or (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.split('.')[0] == 'onescience'
                )
                for node in ast.walk(tree)
            )
            return (
                source,
                imports,
                {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)},
            )

        path = CANDIDATE_DIR / FILENAME
        candidate = load_module('genscore_candidate', path)
        reference = load_module('genscore_reference', STANDARD)
        model = candidate.GenScoreModel(**CTOR)
        reference_model = reference.GenScoreModel(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        manager = tempfile.TemporaryDirectory(prefix='genscore_infer_')
        checkpoint = Path(manager.name) / config.get('checkpoint_fixture', {}).get('filename', 'reference.pt')
        torch.save({'model_state_dict': reference_model.state_dict(), 'config': CTOR}, checkpoint)
        model = model.eval()
        reference_model = reference_model.eval()
        source, _imports, names = source_info(path)
        update_check(
            current_index,
            not result.missing_keys and not result.unexpected_keys,
            str(result),
        )
        current_index += 1
        update_check(current_index, 'torch.load' in source, 'torch.load')
        current_index += 1
        update_check(current_index, 'load_state_dict' in names, str(sorted(names)))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''
    Checkpoint implementation.
    '''
    CHECKPOINT_NAMES = [
        'model_structure:nn_module',
        'model_structure:required_components',
        'model_structure:state_dict_contract',
        'model_structure:module_order',
        'model_structure:constructor_contract',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        FILENAME = config.get('submission_filename', 'genscore_inference.py')
        STANDARD = REFERENCE_DIR / config.get('reference_filename', 'genscore_inference.py')
        CTOR = config.get(
            'constructor_args',
            {
                'ligand_features': 8,
                'pocket_features': 10,
                'hidden_size': 32,
                'radial_bins': 8,
                'num_layers': 2,
            },
        )
        ATTRS = tuple(
            config.get(
                'required_component_paths',
                (
                    'ligand_encoder',
                    'pocket_encoder',
                    'distance_expansion',
                    'ligand_layers',
                    'ligand_layers.0.edge_gate',
                    'pocket_layers',
                    'cross_projection',
                    'score_head',
                ),
            )
        )

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(path)
            sys.path.insert(0, str(path.parent))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        def dotted(value, name):
            for part in name.split('.'):
                value = getattr(value, part)
            return value

        path = CANDIDATE_DIR / FILENAME
        candidate = load_module('genscore_candidate', path)
        reference = load_module('genscore_reference', STANDARD)
        model = candidate.GenScoreModel(**CTOR)
        reference_model = reference.GenScoreModel(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        model = model.eval()
        reference_model = reference_model.eval()
        missing, mismatched = [], []
        for name in ATTRS:
            try:
                if (
                    type(dotted(model, name)).__name__
                    != type(dotted(reference_model, name)).__name__
                ):
                    mismatched.append(name)
            except AttributeError:
                missing.append(name)
        state, ref = model.state_dict(), reference_model.state_dict()
        children = [(n, type(m).__name__) for n, m in model.named_children()]
        ref_children = [(n, type(m).__name__) for n, m in reference_model.named_children()]
        signature = inspect.signature(candidate.GenScoreModel.__init__)
        update_check(current_index, isinstance(model, torch.nn.Module), type(model).__name__)
        current_index += 1
        update_check(
            current_index,
            not missing and not mismatched,
            f'missing={missing}, mismatched={mismatched}',
        )
        current_index += 1
        update_check(
            current_index,
            list(state) == list(ref) and all(state[name].shape == ref[name].shape for name in state),
            f'keys={len(state)}',
        )
        current_index += 1
        update_check(
            current_index,
            children == ref_children,
            f'candidate={children}, reference={ref_children}',
        )
        current_index += 1
        update_check(
            current_index,
            all(name in signature.parameters for name in CTOR),
            str(signature),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''
    Checkpoint implementation.
    '''
    CHECKPOINT_NAMES = [
        'forward_executable:tensor',
        'forward_executable:shape',
        'forward_executable:finite',
        'forward_executable:deterministic',
        'forward_executable:translation_invariant',
        'forward_executable:invalid_coordinates_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        FILENAME = config.get('submission_filename', 'genscore_inference.py')
        STANDARD = REFERENCE_DIR / config.get('reference_filename', 'genscore_inference.py')
        CTOR = config.get(
            'constructor_args',
            {
                'ligand_features': 8,
                'pocket_features': 10,
                'hidden_size': 32,
                'radial_bins': 8,
                'num_layers': 2,
            },
        )

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(path)
            sys.path.insert(0, str(path.parent))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        def fixture(torch):
            generator = torch.Generator().manual_seed(71)
            return (
                torch.randn(2, 5, 8, generator=generator),
                torch.randn(2, 5, 3, generator=generator),
                torch.randn(2, 7, 10, generator=generator),
                torch.randn(2, 7, 3, generator=generator),
                torch.tensor([[True, True, True, True, False], [True] * 5]),
                torch.tensor([[True] * 6 + [False], [True] * 7]),
            )

        path = CANDIDATE_DIR / FILENAME
        candidate = load_module('genscore_candidate', path)
        reference = load_module('genscore_reference', STANDARD)
        model = candidate.GenScoreModel(**CTOR)
        reference_model = reference.GenScoreModel(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        model = model.eval()
        reference_model = reference_model.eval()
        torch, model, args = torch, model, fixture(torch)
        with torch.inference_mode():
            output, repeated = model(*args), model(*args)
            shift = torch.tensor([3.0, -2.0, 1.0])
            shifted = model(
                args[0], args[1] + shift, args[2], args[3] + shift, args[4], args[5]
            )
        rejected = False
        try:
            model(args[0], args[1][..., :2], args[2], args[3], args[4], args[5])
        except ValueError:
            rejected = True
        update_check(current_index, isinstance(output, torch.Tensor), str(type(output)))
        current_index += 1
        update_check(current_index, tuple(output.shape) == (2,), str(tuple(output.shape)))
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output).all()), 'finite')
        current_index += 1
        update_check(current_index, torch.equal(output, repeated), 'repeat equality')
        current_index += 1
        update_check(
            current_index,
            torch.allclose(output, shifted, atol=1e-6),
            f'max_diff={float((output - shifted).abs().max())}',
        )
        current_index += 1
        update_check(current_index, rejected, 'invalid coordinate shape fails')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''
    Checkpoint implementation.
    '''
    CHECKPOINT_NAMES = [
        'evaluation_interface:independent_from_onescience',
        'evaluation_interface:cli_flags',
        'evaluation_interface:no_grad',
        'evaluation_interface:cli_smoke',
        'evaluation_interface:artifact',
        'evaluation_interface:reject_partial_checkpoint',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import numpy as np
        import torch

        FILENAME = config.get('script_path', 'genscore_inference.py')
        STANDARD = REFERENCE_DIR / config.get('reference_filename', 'genscore_inference.py')
        CTOR = config.get(
            'constructor_args',
            {
                'ligand_features': 8,
                'pocket_features': 10,
                'hidden_size': 32,
                'radial_bins': 8,
                'num_layers': 2,
            },
        )

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(path)
            sys.path.insert(0, str(path.parent))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        def source_info(path):
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source)
            imports = any(
                (
                    isinstance(node, ast.Import)
                    and any(alias.name.split('.')[0] == 'onescience' for alias in node.names)
                )
                or (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.split('.')[0] == 'onescience'
                )
                for node in ast.walk(tree)
            )
            return (
                source,
                imports,
                {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)},
            )

        def fixture(torch):
            generator = torch.Generator().manual_seed(71)
            return (
                torch.randn(2, 5, 8, generator=generator),
                torch.randn(2, 5, 3, generator=generator),
                torch.randn(2, 7, 10, generator=generator),
                torch.randn(2, 7, 3, generator=generator),
                torch.tensor([[True, True, True, True, False], [True] * 5]),
                torch.tensor([[True] * 6 + [False], [True] * 7]),
            )

        path = CANDIDATE_DIR / FILENAME
        candidate = load_module('genscore_candidate', path)
        reference = load_module('genscore_reference', STANDARD)
        model = candidate.GenScoreModel(**CTOR)
        reference_model = reference.GenScoreModel(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        manager = tempfile.TemporaryDirectory(prefix='genscore_infer_')
        checkpoint = Path(manager.name) / config.get('checkpoint', {}).get('filename', 'reference.pt')
        torch.save({'model_state_dict': reference_model.state_dict(), 'config': CTOR}, checkpoint)
        model = model.eval()
        reference_model = reference_model.eval()
        source, imports, names = source_info(path)
        flags = tuple(config.get('required_cli_flags', ('--complex-npz', '--checkpoint', '--output', '--device')))
        root = Path(manager.name)
        source_file, artifact = (
            root / config.get('npz_fixture', {}).get('filename', 'complex.npz'),
            root / config.get('output', {}).get('filename', 'scores.json'),
        )
        args = fixture(torch)
        keys = (
            'ligand_features',
            'ligand_coordinates',
            'pocket_features',
            'pocket_coordinates',
            'ligand_mask',
            'pocket_mask',
        )
        np.savez_compressed(
            source_file, **{key: value.numpy() for key, value in zip(keys, args)}
        )
        with torch.inference_mode():
            expected = reference_model(*args).numpy()
        command = [
            sys.executable,
            str(path),
            '--complex-npz',
            str(source_file),
            '--checkpoint',
            str(checkpoint),
            '--output',
            str(artifact),
            '--device',
            'cpu',
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.get('command', {}).get('timeout_seconds', 180),
        )
        state = reference_model.state_dict()
        partial = root / config.get('partial_checkpoint', {}).get('filename', 'partial.pt')
        torch.save({'model_state_dict': dict(list(state.items())[1:])}, partial)
        invalid_command = command.copy()
        invalid_command[invalid_command.index('--checkpoint') + 1] = str(partial)
        invalid_command[invalid_command.index('--output') + 1] = str(root / 'invalid.json')
        rejected = subprocess.run(
            invalid_command,
            capture_output=True,
            text=True,
            timeout=config.get('command', {}).get('timeout_seconds', 180),
        )
        update_check(current_index, not imports, 'no onescience import')
        current_index += 1
        update_check(current_index, all(flag in source for flag in flags), str(flags))
        current_index += 1
        update_check(current_index, bool({'inference_mode', 'no_grad'} & names), str(sorted(names)))
        current_index += 1
        update_check(current_index, process.returncode == 0, process.stderr[-1000:])
        current_index += 1
        update_check(current_index, artifact.is_file(), str(artifact))
        current_index += 1
        update_check(
            current_index,
            rejected.returncode != 0,
            f'returncode={rejected.returncode}, stderr={rejected.stderr[-500:]}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''
    Checkpoint implementation.
    '''
    CHECKPOINT_NAMES = [
        'metric:forward_equivalence',
        'metric:artifact_oracle',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import numpy as np
        import torch

        FILENAME = config.get('script_path', 'genscore_inference.py')
        STANDARD = REFERENCE_DIR / config.get('reference_filename', 'genscore_inference.py')
        CTOR = config.get(
            'constructor_args',
            {
                'ligand_features': 8,
                'pocket_features': 10,
                'hidden_size': 32,
                'radial_bins': 8,
                'num_layers': 2,
            },
        )

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(path)
            sys.path.insert(0, str(path.parent))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        def source_info(path):
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source)
            imports = any(
                (
                    isinstance(node, ast.Import)
                    and any(alias.name.split('.')[0] == 'onescience' for alias in node.names)
                )
                or (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.split('.')[0] == 'onescience'
                )
                for node in ast.walk(tree)
            )
            return (
                source,
                imports,
                {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)},
            )

        def fixture(torch):
            generator = torch.Generator().manual_seed(71)
            return (
                torch.randn(2, 5, 8, generator=generator),
                torch.randn(2, 5, 3, generator=generator),
                torch.randn(2, 7, 10, generator=generator),
                torch.randn(2, 7, 3, generator=generator),
                torch.tensor([[True, True, True, True, False], [True] * 5]),
                torch.tensor([[True] * 6 + [False], [True] * 7]),
            )

        path = CANDIDATE_DIR / FILENAME
        candidate_module = load_module('genscore_candidate', path)
        reference_module = load_module('genscore_reference', STANDARD)
        model = candidate_module.GenScoreModel(**CTOR)
        reference_model = reference_module.GenScoreModel(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        manager = tempfile.TemporaryDirectory(prefix='genscore_infer_')
        checkpoint = Path(manager.name) / config.get('checkpoint', {}).get('filename', 'reference.pt')
        torch.save({'model_state_dict': reference_model.state_dict(), 'config': CTOR}, checkpoint)
        model = model.eval()
        reference_model = reference_model.eval()
        source, imports, names = source_info(path)
        flags = ('--complex-npz', '--checkpoint', '--output', '--device')
        root = Path(manager.name)
        source_file, artifact = root / 'complex.npz', root / 'scores.json'
        args = fixture(torch)
        keys = (
            'ligand_features',
            'ligand_coordinates',
            'pocket_features',
            'pocket_coordinates',
            'ligand_mask',
            'pocket_mask',
        )
        np.savez_compressed(
            source_file, **{key: value.numpy() for key, value in zip(keys, args)}
        )
        with torch.inference_mode():
            expected = reference_model(*args).numpy()
        command = [
            sys.executable,
            str(path),
            '--complex-npz',
            str(source_file),
            '--checkpoint',
            str(checkpoint),
            '--output',
            str(artifact),
            '--device',
            'cpu',
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=180)
        state = reference_model.state_dict()
        partial = root / 'partial.pt'
        torch.save({'model_state_dict': dict(list(state.items())[1:])}, partial)
        invalid_command = command.copy()
        invalid_command[invalid_command.index('--checkpoint') + 1] = str(partial)
        invalid_command[invalid_command.index('--output') + 1] = str(root / 'invalid.json')
        rejected = subprocess.run(
            invalid_command, capture_output=True, text=True, timeout=180
        )
        with torch.inference_mode():
            candidate, reference = model(*args), reference_model(*args)
        actual, expected = (
            json.loads(artifact.read_text(encoding='utf-8'))['scores'],
            expected,
        )
        update_check(
            current_index,
            bool(torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6)),
            f'max_diff={float((candidate - reference).abs().max())}',
        )
        current_index += 1
        update_check(
            current_index,
            bool(
                np.asarray(actual).shape == expected.shape
                and np.allclose(actual, expected, atol=1e-6, rtol=1e-6)
            ),
            f'actual={actual}, expected={expected.tolist()}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
