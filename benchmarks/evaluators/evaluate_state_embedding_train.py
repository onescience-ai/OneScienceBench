"""V2 simple-mode evaluator adapter for state embedding training."""
from pathlib import Path
import sys
import ast
import importlib.util
import inspect
import json
import math
import os
import subprocess
import tempfile
import traceback
from typing import Any, Callable, Dict

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

FILENAME = 'state_embedding_train.py'
CTOR = {'num_genes': 64, 'hidden_size': 32, 'embedding_size': 16, 'num_layers': 3}
ATTRS = (
    'gene_embedding',
    'value_projection',
    'blocks',
    'blocks.0.up_projection',
    'blocks.0.gate',
    'pool_gate',
    'embedding_projection',
    'decoder',
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
    return (torch.rand(3, 64, generator=torch.Generator().manual_seed(97)) * 4.0,)


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'file:state_embedding_train.py',
        'file:standard:train_config.json',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        candidate = load_module('state_train_candidate', path)
        reference = load_module('state_train_reference', REFERENCE_DIR / FILENAME)
        assets = json.loads(
            (REFERENCE_DIR / 'train_config.json').read_text(encoding='utf-8')
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


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'data:expression_shape',
        'data:nonnegative_expression',
        'data:invalid_expression_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        module = load_module('state_train_candidate', path)
        expression = module.mock_expression()
        negative, shape = False, False
        try:
            module.StateEmbeddingModel()(torch.full((1, 64), -1.0))
        except ValueError:
            negative = True
        try:
            module.StateEmbeddingModel()(torch.zeros(2, 63))
        except ValueError:
            shape = True
        ok = tuple(expression.shape) == (16, 64)
        update_check(current_index, ok, str(tuple(expression.shape)))
        current_index += 1
        ok = bool((expression >= 0).all())
        update_check(current_index, ok, f'min={float(expression.min())}')
        current_index += 1
        ok = negative and shape
        update_check(current_index, ok, f'negative={negative}, shape={shape}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'train_script:independent_from_onescience',
        'train_script:cli_flags',
        'train_script:adamw_backward',
        'train_script:gradient_clipping',
        'train_script:structured_artifacts',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = CANDIDATE_DIR / FILENAME
        source, imports, names = source_info(path)
        flags = (
            '--dataset-npz',
            '--mock-data',
            '--output-dir',
            '--max-steps',
            '--mask-ratio',
            '--seed',
        )
        ok = not imports
        update_check(current_index, ok, 'no onescience import')
        current_index += 1
        ok = all(flag in source for flag in flags)
        update_check(current_index, ok, str(flags))
        current_index += 1
        ok = 'AdamW' in names and 'backward' in names
        update_check(current_index, ok, str(sorted(names)))
        current_index += 1
        ok = 'clip_grad_norm_' in names
        update_check(current_index, ok, str(sorted(names)))
        current_index += 1
        ok = 'metrics.jsonl' in source and 'model_state_dict' in source
        update_check(current_index, ok, 'structured outputs')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
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

        path = CANDIDATE_DIR / FILENAME
        candidate = load_module('state_train_candidate', path)
        reference_module = load_module('state_train_reference', REFERENCE_DIR / FILENAME)
        model = candidate.StateEmbeddingModel(**CTOR)
        reference = reference_module.StateEmbeddingModel(**CTOR)
        result = model.load_state_dict(reference.state_dict(), strict=True)
        model.eval()
        reference.eval()
        missing, mismatched = [], []
        for name in ATTRS:
            try:
                if (
                    type(dotted(model, name)).__name__
                    != type(dotted(reference, name)).__name__
                ):
                    mismatched.append(name)
            except AttributeError:
                missing.append(name)
        state, ref = model.state_dict(), reference.state_dict()
        children = [(n, type(m).__name__) for n, m in model.named_children()]
        ref_children = [(n, type(m).__name__) for n, m in reference.named_children()]
        signature = inspect.signature(candidate.StateEmbeddingModel.__init__)
        ok = isinstance(model, torch.nn.Module)
        update_check(current_index, ok, type(model).__name__)
        current_index += 1
        ok = not missing and not mismatched
        update_check(current_index, ok, f'missing={missing}, mismatched={mismatched}')
        current_index += 1
        ok = list(state) == list(ref) and not result.missing_keys and not result.unexpected_keys
        update_check(current_index, ok, f'keys={len(state)}')
        current_index += 1
        ok = children == ref_children
        update_check(current_index, ok, f'candidate={children}, reference={ref_children}')
        current_index += 1
        ok = all(name in signature.parameters for name in CTOR)
        update_check(current_index, ok, str(signature))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'forward_executable:tensor',
        'forward_executable:shape',
        'forward_executable:finite',
        'forward_executable:deterministic',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        candidate = load_module('state_train_candidate', path)
        reference_module = load_module('state_train_reference', REFERENCE_DIR / FILENAME)
        model = candidate.StateEmbeddingModel(**CTOR)
        reference = reference_module.StateEmbeddingModel(**CTOR)
        result = model.load_state_dict(reference.state_dict(), strict=True)
        model.eval()
        reference.eval()
        args = fixture(torch)
        with torch.inference_mode():
            output, repeated = model(*args), model(*args)
        ok = isinstance(output, torch.Tensor)
        update_check(current_index, ok, str(type(output)))
        current_index += 1
        ok = tuple(output.shape) == (3, 64)
        update_check(current_index, ok, str(tuple(output.shape)))
        current_index += 1
        ok = bool(torch.isfinite(output).all())
        update_check(current_index, ok, 'finite')
        current_index += 1
        ok = torch.equal(output, repeated)
        update_check(current_index, ok, 'repeat equality')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['forward_equivalence:forward_equivalence']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        candidate_module = load_module('state_train_candidate', path)
        reference_module = load_module('state_train_reference', REFERENCE_DIR / FILENAME)
        model = candidate_module.StateEmbeddingModel(**CTOR)
        reference_model = reference_module.StateEmbeddingModel(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        model.eval()
        reference_model.eval()
        args = fixture(torch)
        with torch.inference_mode():
            candidate, reference = model(*args), reference_model(*args)
        ok = bool(torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6))
        update_check(current_index, ok, f'max_diff={float((candidate - reference).abs().max())}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'traning:short_fit',
        'traning:parameter_update',
        'traning:npz_fit',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        candidate = load_module('state_train_candidate', path)
        reference = load_module('state_train_reference', REFERENCE_DIR / FILENAME)
        manager = tempfile.TemporaryDirectory(prefix='state_train_')
        output = Path(manager.name) / 'output'
        process = subprocess.run(
            [
                sys.executable,
                str(path),
                '--mock-data',
                '--output-dir',
                str(output),
                '--max-steps',
                '5',
                '--batch-size',
                '4',
                '--mask-ratio',
                '0.2',
                '--seed',
                '53',
                '--device',
                'cpu',
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        checkpoint = output / 'state_embedding_train.pt'
        updated = False
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location='cpu')
            torch.manual_seed(53)
            reference.mock_expression()
            initial = reference.StateEmbeddingModel(**CTOR).state_dict()
            state = payload.get('model_state_dict', {})
            updated = set(state) == set(initial) and any(
                not torch.equal(initial[name], state[name]) for name in initial
            )
        import numpy as np

        root = Path(manager.name)
        dataset = root / 'expression.npz'
        np.savez_compressed(
            dataset,
            expression=reference.mock_expression().numpy(),
        )
        npz_output = root / 'npz_output'
        npz_process = subprocess.run(
            [
                sys.executable,
                str(path),
                '--dataset-npz',
                str(dataset),
                '--output-dir',
                str(npz_output),
                '--max-steps',
                '1',
                '--batch-size',
                '4',
                '--mask-ratio',
                '0.2',
                '--seed',
                '53',
                '--device',
                'cpu',
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        ok = process.returncode == 0
        update_check(current_index, ok, f'returncode={process.returncode}, stderr={process.stderr[-1000:]}')
        current_index += 1
        update_check(current_index, updated, str(checkpoint))
        current_index += 1
        ok = (
            npz_process.returncode == 0
            and (npz_output / 'metrics.jsonl').is_file()
            and (npz_output / 'state_embedding_train.pt').is_file()
        )
        update_check(current_index, ok, f'returncode={npz_process.returncode}, stderr={npz_process.stderr[-1000:]}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['artifacts:checkpoint', 'artifacts:metrics']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        manager = tempfile.TemporaryDirectory(prefix='state_train_')
        output = Path(manager.name) / 'output'
        process = subprocess.run(
            [
                sys.executable,
                str(path),
                '--mock-data',
                '--output-dir',
                str(output),
                '--max-steps',
                '5',
                '--batch-size',
                '4',
                '--mask-ratio',
                '0.2',
                '--seed',
                '53',
                '--device',
                'cpu',
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        checkpoint = output / 'state_embedding_train.pt'
        payload = {}
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location='cpu')
        metrics = output / 'metrics.jsonl'
        rows = (
            [
                json.loads(line)
                for line in metrics.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            if metrics.is_file()
            else []
        )
        required = {'model_state_dict', 'optimizer_state_dict', 'step', 'config'}
        ok = (
            isinstance(payload, dict)
            and required.issubset(payload)
            and payload['step'] == 5
        )
        update_check(current_index, ok, f'keys={sorted(payload)}')
        current_index += 1
        ok = len(rows) == 5 and all(
            row.get('step') == index
            and math.isfinite(float(row.get('train_loss', math.nan)))
            for index, row in enumerate(rows, 1)
        )
        update_check(current_index, ok, str(rows))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
