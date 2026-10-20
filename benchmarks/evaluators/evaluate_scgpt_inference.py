"""V2 adapter for the legacy scGPT inference evaluator."""
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

FILENAME = 'scgpt_inference.py'
CTOR = {
    'num_genes': 128,
    'hidden_size': 32,
    'num_heads': 4,
    'num_layers': 2,
    'embedding_size': 16,
}
ATTRS = (
    'gene_encoder',
    'value_encoder',
    'value_encoder.projection',
    'blocks',
    'blocks.0.attention',
    'cell_projection',
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
    genes = torch.tensor([[1, 2, 3, 4, 0], [4, 3, 2, 1, 5]], dtype=torch.long)
    values = torch.tensor([[1.0, 0.5, 2.0, 0.0, 0.0], [0.2, 1.2, 0.0, 3.0, 0.7]])
    return genes, values, genes.eq(0)


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        'file:scgpt_inference.py',
        'file:standard:inference_config.json',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        reference_path = REFERENCE_DIR / FILENAME
        candidate = load_module('scgpt_candidate', path)
        reference = load_module('scgpt_reference', reference_path)
        assets = json.loads(
            (REFERENCE_DIR / 'inference_config.json').read_text(encoding='utf-8')
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
    """
    Checkpoint implementation.
    """
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

        path = CANDIDATE_DIR / FILENAME
        reference_path = REFERENCE_DIR / FILENAME
        candidate = load_module('scgpt_candidate', path)
        reference = load_module('scgpt_reference', reference_path)
        model = candidate.ScGPTEncoder(**CTOR)
        reference_model = reference.ScGPTEncoder(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        manager = tempfile.TemporaryDirectory(prefix='scgpt_infer_')
        checkpoint = Path(manager.name) / 'reference.pt'
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
    """
    Checkpoint implementation.
    """
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
        reference_path = REFERENCE_DIR / FILENAME
        candidate = load_module('scgpt_candidate', path)
        reference = load_module('scgpt_reference', reference_path)
        model = candidate.ScGPTEncoder(**CTOR)
        reference_model = reference.ScGPTEncoder(**CTOR)
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
        state_ok = list(state) == list(ref) and all(
            state[name].shape == ref[name].shape for name in state
        )
        children = [
            (name, type(module).__name__) for name, module in model.named_children()
        ]
        ref_children = [
            (name, type(module).__name__) for name, module in reference_model.named_children()
        ]
        signature = inspect.signature(candidate.ScGPTEncoder.__init__)
        update_check(current_index, isinstance(model, torch.nn.Module), type(model).__name__)
        current_index += 1
        update_check(
            current_index,
            not missing and not mismatched,
            f'missing={missing}, mismatched={mismatched}',
        )
        current_index += 1
        update_check(current_index, state_ok, f'keys={len(state)}')
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
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        'forward_executable:tensor',
        'forward_executable:shape',
        'forward_executable:finite',
        'forward_executable:deterministic',
        'forward_executable:l2_normalized',
        'forward_executable:cells_distinct',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = CANDIDATE_DIR / FILENAME
        reference_path = REFERENCE_DIR / FILENAME
        candidate = load_module('scgpt_candidate', path)
        reference = load_module('scgpt_reference', reference_path)
        model = candidate.ScGPTEncoder(**CTOR)
        reference_model = reference.ScGPTEncoder(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        model = model.eval()
        reference_model = reference_model.eval()
        args = fixture(torch)
        with torch.inference_mode():
            output, repeated = model(*args), model(*args)
        norms = output.norm(dim=-1)
        update_check(current_index, isinstance(output, torch.Tensor), str(type(output)))
        current_index += 1
        update_check(current_index, tuple(output.shape) == (2, 16), str(tuple(output.shape)))
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output).all()), 'finite')
        current_index += 1
        update_check(current_index, torch.equal(output, repeated), 'repeat equality')
        current_index += 1
        update_check(
            current_index,
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
            str(norms.tolist()),
        )
        current_index += 1
        update_check(current_index, not torch.equal(output[0], output[1]), 'distinct profiles')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
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

        path = CANDIDATE_DIR / FILENAME
        reference_path = REFERENCE_DIR / FILENAME
        candidate = load_module('scgpt_candidate', path)
        reference = load_module('scgpt_reference', reference_path)
        model = candidate.ScGPTEncoder(**CTOR)
        reference_model = reference.ScGPTEncoder(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        manager = tempfile.TemporaryDirectory(prefix='scgpt_infer_')
        checkpoint = Path(manager.name) / 'reference.pt'
        torch.save({'model_state_dict': reference_model.state_dict(), 'config': CTOR}, checkpoint)
        model = model.eval()
        reference_model = reference_model.eval()
        source, imports, names = source_info(path)
        flags = ('--input-npz', '--checkpoint', '--output', '--batch-size', '--device')
        root = Path(manager.name)
        genes = np.asarray([[1, 2, 3, 4, 0], [5, 6, 7, 8, 9]], dtype=np.int64)
        values = np.asarray(
            [[1.0, 2.0, 0.5, 0.0, 0.0], [0.3, 0.2, 1.1, 2.2, 0.7]], dtype=np.float32
        )
        source_file, artifact = root / 'cells.npz', root / 'embeddings.npz'
        np.savez_compressed(source_file, gene_ids=genes, values=values)
        with torch.inference_mode():
            expected = reference_model(
                torch.as_tensor(genes),
                torch.as_tensor(values),
                torch.as_tensor(genes).eq(0),
            ).numpy()
        command = [
            sys.executable,
            str(path),
            '--input-npz',
            str(source_file),
            '--checkpoint',
            str(checkpoint),
            '--output',
            str(artifact),
            '--batch-size',
            '1',
            '--device',
            'cpu',
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
        )
        state = reference_model.state_dict()
        partial = root / 'partial.pt'
        torch.save({'model_state_dict': dict(list(state.items())[1:])}, partial)
        invalid_command = command.copy()
        invalid_command[invalid_command.index('--checkpoint') + 1] = str(partial)
        invalid_command[invalid_command.index('--output') + 1] = str(root / 'invalid.npz')
        rejected = subprocess.run(
            invalid_command, capture_output=True, text=True, timeout=180
        )
        update_check(current_index, not imports, 'no onescience import')
        current_index += 1
        update_check(current_index, all(flag in source for flag in flags), str(flags))
        current_index += 1
        update_check(
            current_index,
            bool({'inference_mode', 'no_grad'} & names),
            str(sorted(names)),
        )
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
    """
    Checkpoint implementation.
    """
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

        path = CANDIDATE_DIR / FILENAME
        reference_path = REFERENCE_DIR / FILENAME
        candidate_module = load_module('scgpt_candidate', path)
        reference_module = load_module('scgpt_reference', reference_path)
        model = candidate_module.ScGPTEncoder(**CTOR)
        reference_model = reference_module.ScGPTEncoder(**CTOR)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        manager = tempfile.TemporaryDirectory(prefix='scgpt_infer_')
        checkpoint = Path(manager.name) / 'reference.pt'
        torch.save({'model_state_dict': reference_model.state_dict(), 'config': CTOR}, checkpoint)
        model = model.eval()
        reference_model = reference_model.eval()
        root = Path(manager.name)
        genes = np.asarray([[1, 2, 3, 4, 0], [5, 6, 7, 8, 9]], dtype=np.int64)
        values = np.asarray(
            [[1.0, 2.0, 0.5, 0.0, 0.0], [0.3, 0.2, 1.1, 2.2, 0.7]], dtype=np.float32
        )
        source_file, artifact = root / 'cells.npz', root / 'embeddings.npz'
        np.savez_compressed(source_file, gene_ids=genes, values=values)
        with torch.inference_mode():
            expected = reference_model(
                torch.as_tensor(genes),
                torch.as_tensor(values),
                torch.as_tensor(genes).eq(0),
            ).numpy()
        command = [
            sys.executable,
            str(path),
            '--input-npz',
            str(source_file),
            '--checkpoint',
            str(checkpoint),
            '--output',
            str(artifact),
            '--batch-size',
            '1',
            '--device',
            'cpu',
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
        )
        state = reference_model.state_dict()
        partial = root / 'partial.pt'
        torch.save({'model_state_dict': dict(list(state.items())[1:])}, partial)
        invalid_command = command.copy()
        invalid_command[invalid_command.index('--checkpoint') + 1] = str(partial)
        invalid_command[invalid_command.index('--output') + 1] = str(root / 'invalid.npz')
        rejected = subprocess.run(
            invalid_command, capture_output=True, text=True, timeout=180
        )
        args = fixture(torch)
        with torch.inference_mode():
            candidate, reference = model(*args), reference_model(*args)
        actual = np.load(artifact)['embeddings']
        update_check(
            current_index,
            bool(torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6)),
            f'max_diff={float((candidate - reference).abs().max())}',
        )
        current_index += 1
        update_check(
            current_index,
            bool(
                actual.shape == expected.shape
                and np.allclose(actual, expected, atol=1e-6, rtol=1e-6)
            ),
            f'actual={actual.shape}, expected={expected.shape}',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
