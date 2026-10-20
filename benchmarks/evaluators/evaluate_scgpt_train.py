'''V2 adapter for the scGPT training evaluator.'''
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

FILENAME = 'scgpt_train.py'
STANDARD = REFERENCE_DIR / FILENAME
CTOR = {
    'num_genes': 128,
    'hidden_size': 32,
    'num_heads': 4,
    'num_layers': 2,
    'num_labels': 3,
}
ATTRS = (
    'gene_encoder',
    'value_encoder',
    'value_encoder.projection',
    'blocks',
    'blocks.0.attention',
    'classifier',
)


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:scgpt_train.py', 'file:standard:train_config.json']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        spec = importlib.util.spec_from_file_location('scgpt_train_candidate', path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.path.insert(0, str(path.parent))
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_candidate'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('scgpt_train_reference', STANDARD)
        if spec is None or spec.loader is None:
            raise ImportError(STANDARD)
        sys.path.insert(0, str(STANDARD.parent))
        reference = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_reference'] = reference
        spec.loader.exec_module(reference)
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
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'data:mock_shapes',
        'data:nonnegative_expression',
        'data:negative_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        spec = importlib.util.spec_from_file_location('scgpt_train_candidate', path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.path.insert(0, str(path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_candidate'] = module
        spec.loader.exec_module(module)
        genes, values, labels = module.mock_dataset()
        rejected = False
        try:
            module.ContinuousValueEncoder(8)(torch.tensor([[1.0, -1.0]]))
        except ValueError:
            rejected = True
        ok = (
            tuple(genes.shape) == (12, 10)
            and values.shape == genes.shape
            and labels.shape == (12,)
        )
        update_check(
            current_index,
            ok,
            f'genes={tuple(genes.shape)}, labels={tuple(labels.shape)}',
        )
        current_index += 1
        update_check(current_index, bool((values >= 0).all()), f'min={float(values.min())}')
        current_index += 1
        update_check(current_index, rejected, 'negative values fail')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
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
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        flags = ('--dataset-npz', '--mock-data', '--output-dir', '--max-steps', '--seed')
        update_check(current_index, not imports, 'no onescience import')
        current_index += 1
        update_check(current_index, all(flag in source for flag in flags), str(flags))
        current_index += 1
        update_check(
            current_index,
            'AdamW' in names and 'backward' in names,
            str(sorted(names)),
        )
        current_index += 1
        update_check(current_index, 'clip_grad_norm_' in names, str(sorted(names)))
        current_index += 1
        update_check(
            current_index,
            'metrics.jsonl' in source and 'model_state_dict' in source,
            'structured outputs',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
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
        spec = importlib.util.spec_from_file_location('scgpt_train_candidate', path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.path.insert(0, str(path.parent))
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_candidate'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('scgpt_train_reference', STANDARD)
        if spec is None or spec.loader is None:
            raise ImportError(STANDARD)
        sys.path.insert(0, str(STANDARD.parent))
        reference = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_reference'] = reference
        spec.loader.exec_module(reference)
        model = candidate.ScGPTClassifier(**CTOR)
        reference = reference.ScGPTClassifier(**CTOR)
        result = model.load_state_dict(reference.state_dict(), strict=True)
        model.eval()
        reference.eval()
        missing, mismatched = [], []
        for name in ATTRS:
            try:
                value = model
                reference_value = reference
                for part in name.split('.'):
                    value = getattr(value, part)
                    reference_value = getattr(reference_value, part)
                if type(value).__name__ != type(reference_value).__name__:
                    mismatched.append(name)
            except AttributeError:
                missing.append(name)
        state, ref = model.state_dict(), reference.state_dict()
        children = [(n, type(m).__name__) for n, m in model.named_children()]
        ref_children = [(n, type(m).__name__) for n, m in reference.named_children()]
        signature = inspect.signature(candidate.ScGPTClassifier.__init__)
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
            list(state) == list(ref)
            and not result.missing_keys
            and not result.unexpected_keys,
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
    '''Checkpoint implementation.'''
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
        spec = importlib.util.spec_from_file_location('scgpt_train_candidate', path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.path.insert(0, str(path.parent))
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_candidate'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('scgpt_train_reference', STANDARD)
        if spec is None or spec.loader is None:
            raise ImportError(STANDARD)
        sys.path.insert(0, str(STANDARD.parent))
        reference = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_reference'] = reference
        spec.loader.exec_module(reference)
        model = candidate.ScGPTClassifier(**CTOR)
        reference = reference.ScGPTClassifier(**CTOR)
        model.load_state_dict(reference.state_dict(), strict=True)
        model.eval()
        reference.eval()
        genes = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.long)
        values = torch.tensor([[1.0, 0.5, 2.0, 0.0], [0.2, 1.2, 0.0, 3.0]])
        args = (genes, values, genes.eq(0))
        with torch.inference_mode():
            output, repeated = model(*args), model(*args)
        update_check(current_index, isinstance(output, torch.Tensor), str(type(output)))
        current_index += 1
        update_check(current_index, tuple(output.shape) == (2, 3), str(tuple(output.shape)))
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output).all()), 'finite')
        current_index += 1
        update_check(current_index, torch.equal(output, repeated), 'repeat equality')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['forward_equivalence:forward_equivalence']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        spec = importlib.util.spec_from_file_location('scgpt_train_candidate', path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.path.insert(0, str(path.parent))
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_candidate'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('scgpt_train_reference', STANDARD)
        if spec is None or spec.loader is None:
            raise ImportError(STANDARD)
        sys.path.insert(0, str(STANDARD.parent))
        reference = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_reference'] = reference
        spec.loader.exec_module(reference)
        model = candidate.ScGPTClassifier(**CTOR)
        reference = reference.ScGPTClassifier(**CTOR)
        model.load_state_dict(reference.state_dict(), strict=True)
        model.eval()
        reference.eval()
        genes = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.long)
        values = torch.tensor([[1.0, 0.5, 2.0, 0.0], [0.2, 1.2, 0.0, 3.0]])
        args = (genes, values, genes.eq(0))
        with torch.inference_mode():
            candidate, reference = model(*args), reference(*args)
        update_check(
            current_index,
            bool(torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6)),
            f'max_diff={float((candidate - reference).abs().max())}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
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
        spec = importlib.util.spec_from_file_location('scgpt_train_reference', STANDARD)
        if spec is None or spec.loader is None:
            raise ImportError(STANDARD)
        sys.path.insert(0, str(STANDARD.parent))
        reference = importlib.util.module_from_spec(spec)
        sys.modules['scgpt_train_reference'] = reference
        spec.loader.exec_module(reference)
        manager = tempfile.TemporaryDirectory(prefix='scgpt_train_')
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
                '--seed',
                '29',
                '--device',
                'cpu',
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        checkpoint = output / 'scgpt_train.pt'
        updated = False
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location='cpu')
            torch.manual_seed(29)
            initial = reference.ScGPTClassifier(**CTOR).state_dict()
            state = payload.get('model_state_dict', {})
            updated = set(state) == set(initial) and any(
                not torch.equal(initial[name], state[name]) for name in initial
            )
        import numpy as np

        root = Path(manager.name)
        dataset = root / 'cells.npz'
        np.savez_compressed(
            dataset,
            gene_ids=np.asarray([[1, 2, 3, 0], [4, 5, 6, 7], [2, 3, 4, 0], [6, 7, 8, 9]]),
            values=np.asarray(
                [
                    [1.0, 0.2, 2.0, 0.0],
                    [0.3, 1.0, 0.0, 2.5],
                    [1.1, 2.0, 1.0, 0.0],
                    [0.5, 0.2, 0.8, 1.2],
                ],
                dtype=np.float32,
            ),
            labels=np.asarray([0, 1, 2, 1]),
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
                '2',
                '--seed',
                '29',
                '--device',
                'cpu',
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        update_check(
            current_index,
            process.returncode == 0,
            f'returncode={process.returncode}, stderr={process.stderr[-1000:]}',
        )
        current_index += 1
        update_check(current_index, updated, str(checkpoint))
        current_index += 1
        update_check(
            current_index,
            npz_process.returncode == 0
            and (npz_output / 'metrics.jsonl').is_file()
            and (npz_output / 'scgpt_train.pt').is_file(),
            f'returncode={npz_process.returncode}, stderr={npz_process.stderr[-1000:]}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['artifacts:checkpoint', 'artifacts:metrics']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        manager = tempfile.TemporaryDirectory(prefix='scgpt_train_')
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
                '--seed',
                '29',
                '--device',
                'cpu',
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        checkpoint = output / 'scgpt_train.pt'
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
        update_check(
            current_index,
            isinstance(payload, dict)
            and required.issubset(payload)
            and payload['step'] == 5,
            f'keys={sorted(payload)}',
        )
        current_index += 1
        update_check(
            current_index,
            len(rows) == 5
            and all(
                row.get('step') == index
                and math.isfinite(float(row.get('train_loss', math.nan)))
                for index, row in enumerate(rows, 1)
            ),
            str(rows),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
