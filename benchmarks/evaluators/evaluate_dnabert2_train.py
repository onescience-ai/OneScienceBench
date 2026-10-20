"""V2 adapter for the legacy DNABERT-2 training evaluator."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
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

FILENAME, ENV_NAME = 'dnabert2_train.py', 'DNABERT2_TRAIN_SUBMISSION_DIR'
CTOR = {
    'vocab_size': 7,
    'hidden_size': 32,
    'num_heads': 4,
    'num_layers': 2,
    'num_labels': 2,
}
ATTRS = (
    'token_embedding',
    'blocks',
    'blocks.0.attention',
    'blocks.0.attention.qkv',
    'classifier',
)
EXPECTED = {
    'file': ('file:dnabert2_train.py', 'file:standard:train_config.json'),
    'data': (
        'data:sequence_encoding',
        'data:invalid_sequence_rejected',
        'data:batch_contract',
    ),
    'train_script': (
        'train_script:independent_from_onescience',
        'train_script:cli_flags',
        'train_script:adamw_backward',
        'train_script:gradient_clipping',
        'train_script:structured_artifacts',
    ),
    'model_structure': (
        'model_structure:nn_module',
        'model_structure:required_components',
        'model_structure:state_dict_contract',
        'model_structure:module_order',
        'model_structure:constructor_contract',
    ),
    'forward_executable': (
        'forward_executable:tensor',
        'forward_executable:shape',
        'forward_executable:finite',
        'forward_executable:deterministic',
    ),
    'forward_equivalence': ('forward_equivalence:forward_equivalence',),
    'traning': ('traning:short_fit', 'traning:parameter_update', 'traning:csv_fit'),
    'artifacts': ('artifacts:checkpoint', 'artifacts:metrics'),
}
_ADAPTER_STATE: Dict[str, Any] = {}

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def dotted(value: Any, name: str):
    for part in name.split('.'):
        value = getattr(value, part)
    return value

def source_info(path: Path):
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
    return source, imports, names

def fixture(torch):
    tokens = torch.tensor([[6, 1, 2, 3, 4], [6, 4, 3, 2, 1]], dtype=torch.long)
    return tokens, tokens.ne(0)

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['file:dnabert2_train.py', 'file:standard:train_config.json']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = (CANDIDATE_DIR / Path(config.get('submission_filename', FILENAME)).name).resolve()
        candidate = load_module('dnabert2_train_candidate', path)
        reference_path = (REFERENCE_DIR / FILENAME).resolve()
        reference = load_module('dnabert2_train_reference', reference_path)
        standard_config_path = (
            REFERENCE_DIR / Path(config.get('standard_config_path', 'train_config.json')).name
        ).resolve()
        assets = json.loads(standard_config_path.read_text(encoding='utf-8'))['external_assets']
        _ADAPTER_STATE.update(
            torch=torch, path=path, candidate=candidate, reference=reference
        )
        ok, msg = True, str(path)
        update_check(current_index, ok, msg)
        current_index += 1
        ok = (
            assets.get('mount_required') is True
            and Path(assets['weights']).exists()
            and Path(assets['dataset']).exists()
        )
        msg = str(assets)
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['data:sequence_encoding', 'data:invalid_sequence_rejected', 'data:batch_contract']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        candidate_path = (
            CANDIDATE_DIR / Path(config.get('candidate', {}).get('file', FILENAME)).name
        ).resolve()
        module = load_module('dnabert2_train_candidate_data', candidate_path)
        encoded = module.encode_dna('acgtn')
        ok = torch.equal(encoded, torch.tensor([6, 1, 2, 3, 4, 5]))
        msg = str(encoded.tolist())
        update_check(current_index, ok, msg)
        current_index += 1
        invalid = False
        try:
            module.encode_dna('ACGX')
        except ValueError:
            invalid = True
        ok, msg = invalid, 'invalid base raises ValueError'
        update_check(current_index, ok, msg)
        current_index += 1
        tokens, mask, labels = module.collate([('ACGT', 0), ('TTGG', 1)])
        ok = (
            tuple(tokens.shape) == (2, 5)
            and mask.dtype == torch.bool
            and labels.tolist() == [0, 1]
        )
        msg = f'shape={tuple(tokens.shape)}, labels={labels.tolist()}'
        update_check(current_index, ok, msg)
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
        files = config.get('files', [FILENAME])
        path = (CANDIDATE_DIR / Path(files[0]).name).resolve()
        source, imports, names = source_info(path)
        flags = ('--dataset-dir', '--mock-data', '--output-dir', '--max-steps', '--seed')
        ok, msg = not imports, 'no onescience import'
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = all(flag in source for flag in flags), str(flags)
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = 'AdamW' in names and 'backward' in names, str(sorted(names))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = 'clip_grad_norm_' in names, str(sorted(names))
        update_check(current_index, ok, msg)
        current_index += 1
        ok = 'metrics.jsonl' in source and 'model_state_dict' in source
        msg = 'metrics and state dict'
        update_check(current_index, ok, msg)
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

        candidate_file = (CANDIDATE_DIR / Path(config.get('candidate_file', FILENAME)).name).resolve()
        reference_file = (REFERENCE_DIR / Path(config.get('reference_file', FILENAME)).name).resolve()
        candidate = load_module('dnabert2_train_candidate_structure', candidate_file)
        reference = load_module('dnabert2_train_reference_structure', reference_file)
        model_class = config.get('model_class', 'DNABert2Classifier')
        constructor_args = config.get('constructor_args', CTOR)
        model = getattr(candidate, model_class)(**constructor_args)
        reference_model = getattr(reference, model_class)(**constructor_args)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        model.eval()
        reference_model.eval()
        _ADAPTER_STATE.update(
            torch=torch, candidate=candidate, reference=reference, model=model,
            reference_model=reference_model
        )
        missing, mismatched = [], []
        for name in ATTRS:
            try:
                if type(dotted(model, name)).__name__ != type(dotted(reference_model, name)).__name__:
                    mismatched.append(name)
            except AttributeError:
                missing.append(name)
        state, ref_state = model.state_dict(), reference_model.state_dict()
        state_ok = (
            list(state) == list(ref_state)
            and not result.missing_keys
            and not result.unexpected_keys
        )
        children = [(name, type(module).__name__) for name, module in model.named_children()]
        ref_children = [
            (name, type(module).__name__) for name, module in reference_model.named_children()
        ]
        signature = inspect.signature(getattr(candidate, model_class).__init__)
        ok, msg = isinstance(model, torch.nn.Module), type(model).__name__
        update_check(current_index, ok, msg)
        current_index += 1
        ok = not missing and not mismatched
        msg = f'missing={missing}, mismatched={mismatched}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = state_ok, f'keys={len(state)}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok = children == ref_children
        msg = f'candidate={children}, reference={ref_children}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = all(name in signature.parameters for name in CTOR), str(signature)
        update_check(current_index, ok, msg)
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

        model_file = (CANDIDATE_DIR / Path(config.get('model_file', FILENAME)).name).resolve()
        module = load_module('dnabert2_train_candidate_forward', model_file)
        model = getattr(module, config.get('model_class', 'DNABert2Classifier'))(**CTOR)
        model.eval()
        args = fixture(torch)
        with torch.inference_mode():
            output, repeated = model(*args), model(*args)
        ok, msg = isinstance(output, torch.Tensor), str(type(output))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = tuple(output.shape) == (2, 2), str(tuple(output.shape))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = bool(torch.isfinite(output).all()), 'finite'
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = torch.equal(output, repeated), 'repeat equality'
        update_check(current_index, ok, msg)
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

        candidate_file = (
            CANDIDATE_DIR / Path(config.get('candidate', {}).get('model_file', FILENAME)).name
        ).resolve()
        reference_file = (
            REFERENCE_DIR / Path(config.get('reference', {}).get('model_file', FILENAME)).name
        ).resolve()
        candidate_module = load_module('dnabert2_train_candidate_equivalence', candidate_file)
        reference_module = load_module('dnabert2_train_reference_equivalence', reference_file)
        model_class = config.get('candidate', {}).get('model_class', 'DNABert2Classifier')
        reference_class = config.get('reference', {}).get('model_class', 'DNABert2Classifier')
        model = getattr(candidate_module, model_class)(**CTOR)
        reference_model = getattr(reference_module, reference_class)(**CTOR)
        model.load_state_dict(reference_model.state_dict(), strict=True)
        model.eval()
        reference_model.eval()
        args = fixture(torch)
        with torch.inference_mode():
            candidate, reference = model(*args), reference_model(*args)
        ok = torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6)
        msg = f'max_diff={float((candidate - reference).abs().max())}'
        update_check(current_index, bool(ok), msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_smoke_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['traning:short_fit', 'traning:parameter_update', 'traning:csv_fit']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        run_file = (CANDIDATE_DIR / Path(config.get('run_file', FILENAME)).name).resolve()
        mock_run = config.get('mock_run', {})
        csv_run = config.get('csv_run', {})
        manager = tempfile.TemporaryDirectory(prefix='dnabert2_train_')
        output = Path(manager.name) / mock_run.get('output_subdir', 'output')
        _ADAPTER_STATE.update(manager=manager, output=output, torch=torch)
        process = subprocess.run(
            [
                sys.executable,
                str(run_file),
                '--mock-data',
                '--output-dir',
                str(output),
                '--max-steps',
                str(mock_run.get('max_steps', 5)),
                '--batch-size',
                str(mock_run.get('batch_size', 4)),
                '--seed',
                str(mock_run.get('seed', 17)),
                '--device',
                str(mock_run.get('device', 'cpu')),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        checkpoint = output / config.get('checkpoint', 'dnabert2_train.pt')
        _ADAPTER_STATE.update(checkpoint=checkpoint)
        updated = False
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location='cpu')
            torch.manual_seed(mock_run.get('seed', 17))
            reference_file = (REFERENCE_DIR / FILENAME).resolve()
            reference_module = load_module('dnabert2_train_reference_smoke', reference_file)
            initial = reference_module.DNABert2Classifier(**CTOR).state_dict()
            state = payload.get('model_state_dict', {})
            updated = set(state) == set(initial) and any(
                not torch.equal(initial[name], state[name]) for name in initial
            )
        dataset_dir = Path(manager.name) / 'dataset'
        dataset_dir.mkdir()
        (dataset_dir / csv_run.get('dataset_file', 'train.csv')).write_text(
            csv_run.get(
                'content',
                'sequence,label\nACGTACGT,0\nTTGGCCAA,1\nACACACAC,0\nTGTGTGTG,1\n',
            ),
            encoding='utf-8',
        )
        csv_output = Path(manager.name) / csv_run.get('output_subdir', 'csv_output')
        csv_process = subprocess.run(
            [
                sys.executable,
                str(run_file),
                '--dataset-dir',
                str(dataset_dir),
                '--output-dir',
                str(csv_output),
                '--max-steps',
                str(csv_run.get('max_steps', 1)),
                '--batch-size',
                str(csv_run.get('batch_size', 2)),
                '--seed',
                str(csv_run.get('seed', 17)),
                '--device',
                str(csv_run.get('device', 'cpu')),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        ok = process.returncode == 0
        msg = f'returncode={process.returncode}, stderr={process.stderr[-1000:]}'
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = updated, str(checkpoint)
        update_check(current_index, ok, msg)
        current_index += 1
        ok = (
            csv_process.returncode == 0
            and (csv_output / 'metrics.jsonl').is_file()
            and (csv_output / 'dnabert2_train.pt').is_file()
        )
        msg = f'returncode={csv_process.returncode}, stderr={csv_process.stderr[-1000:]}'
        update_check(current_index, ok, msg)
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

        checkpoint = _ADAPTER_STATE['checkpoint']
        output = _ADAPTER_STATE['output']
        payload = {}
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location='cpu')
        required = {'model_state_dict', 'optimizer_state_dict', 'step', 'config'}
        checkpoint_ok = (
            isinstance(payload, dict)
            and required.issubset(payload)
            and payload['step'] == 5
        )
        ok, msg = checkpoint_ok, f'keys={sorted(payload)}'
        update_check(current_index, ok, msg)
        current_index += 1
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
        metrics_ok = len(rows) == 5 and all(
            row.get('step') == index
            and math.isfinite(float(row.get('train_loss', math.nan)))
            for index, row in enumerate(rows, 1)
        )
        ok, msg = metrics_ok, str(rows)
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
