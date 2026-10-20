"""V2 adapter for the legacy SimpleFold training evaluator."""
from pathlib import Path
import sys
import ast
import importlib.util
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

STANDARD_PATH = REFERENCE_DIR / 'simplefold_train.py'
SUBMISSION_FILENAME = 'simplefold_train.py'

SEED = 20260819
FORWARD_COMPARE_ATOL = 1e-6
CLI_TIMEOUT_SECONDS = 180

BATCH_SIZE = 2
ATOM_COUNT = 8
TOKEN_COUNT = 4
ESM_DIM = 16
HIDDEN_SIZE = 32
REQUIRED_MODULES = (
    'coordinate_embedding',
    'reference_embedding',
    'self_condition_embedding',
    'time_embedder',
    'atom_encoder',
    'esm_projection',
    'trunk',
    'atom_decoder',
    'velocity_head',
)
REQUIRED_CLI_FLAGS = ('--output-dir', '--max-steps', '--seed', '--mock-data')
REQUIRED_METRIC_KEYS = ('train_loss', 'rmse', 'mae', 'grad_norm')
STANDARD_ASSETS = ('train_config.json',)


def _load_torch():
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise RuntimeError(f'torch is required but unavailable: {exc}') from exc
    return torch, nn


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not create import spec for {path}')
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_source(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding='utf-8')
    return source, ast.parse(source, filename=str(path))


def _imports_onescience(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split('.')[0] == 'onescience' for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] == 'onescience':
                return True
    return False


def _has_attribute_call(tree: ast.Module, attribute: str) -> bool:
    return any(
        isinstance(node, ast.Attribute) and node.attr == attribute for node in ast.walk(tree)
    )


def run_cli(script: Path, arguments: list[str]) -> subprocess.CompletedProcess:
    command = [sys.executable, str(script), *arguments]
    return subprocess.run(
        command,
        cwd=str(script.parent),
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
        check=False,
    )


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['file:simplefold_train.py', 'file:standard:train_config.json']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        candidate = load_module_from_path('simplefold_train_submission', path)
        reference = load_module_from_path('simplefold_train_reference', STANDARD_PATH)
        ok, msg = True, str(path)
        update_check(current_index, ok, msg)
        current_index += 1

        for name in STANDARD_ASSETS:
            path = STANDARD_PATH.parent / name
            ok = path.is_file()
            detail = str(path)
            if ok and path.suffix == '.json':
                try:
                    payload = json.loads(path.read_text(encoding='utf-8'))
                    ok = isinstance(payload, dict)
                    if ok:
                        assets = payload.get('external_assets')
                        if not isinstance(assets, dict) or assets.get('mount_required') is not True:
                            ok, detail = False, 'train_config.json must declare mount_required=true external_assets'
                        else:
                            missing = [
                                name
                                for name in ('weights', 'dataset')
                                if not isinstance(assets.get(name), str)
                                or not assets[name].startswith('/public/share/')
                            ]
                            if missing:
                                ok, detail = False, f'external assets must use onestore paths for: {missing}'
                            else:
                                ok, detail = True, 'complete weights and dataset are declared as externally mounted assets'
                    else:
                        detail = f'{path} (JSON object required)'
                except (OSError, json.JSONDecodeError) as error:
                    ok = False
                    detail = f'{path}: {type(error).__name__}: {error}'
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'data:simplefold_contract',
        'data:invalid_shape_rejected',
        'data:missing_feature_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        candidate = load_module_from_path('simplefold_train_submission', path)
        batch = candidate.make_synthetic_batch(
            batch_size=BATCH_SIZE, atom_count=ATOM_COUNT, token_count=TOKEN_COUNT, esm_dim=ESM_DIM, seed=23
        )
        candidate.validate_batch(batch)
        feats = batch['feats']
        aligned = (
            tuple(batch['positions'].shape) == (BATCH_SIZE, ATOM_COUNT, 3)
            and tuple(feats['ref_pos'].shape) == (BATCH_SIZE, ATOM_COUNT, 3)
            and tuple(feats['atom_to_token'].shape) == (BATCH_SIZE, ATOM_COUNT, TOKEN_COUNT)
            and tuple(feats['atom_pad_mask'].shape) == (BATCH_SIZE, ATOM_COUNT)
            and tuple(feats['esm_s'].shape) == (BATCH_SIZE, TOKEN_COUNT, ESM_DIM)
        )
        ok = aligned
        msg = 'positions, ref_pos, atom_to_token, atom_pad_mask and esm_s align on batch/atom/token dims'
        update_check(current_index, ok, msg)
        current_index += 1

        invalid = {'positions': batch['positions'], 'feats': dict(feats)}
        invalid['feats']['atom_pad_mask'] = torch.ones(BATCH_SIZE, ATOM_COUNT - 1, dtype=torch.bool)
        try:
            candidate.validate_batch(invalid)
            shape_ok, shape_detail = False, 'misaligned atom mask accepted'
        except ValueError:
            shape_ok, shape_detail = True, 'misaligned atom mask rejected with ValueError'
        except Exception as error:
            shape_ok, shape_detail = False, f'{type(error).__name__}: {error}'
        update_check(current_index, shape_ok, shape_detail)
        current_index += 1

        missing = {
            'positions': batch['positions'],
            'feats': {key: value for key, value in feats.items() if key != 'ref_pos'},
        }
        try:
            candidate.validate_batch(missing)
            missing_ok, missing_detail = False, 'missing ref_pos accepted'
        except ValueError:
            missing_ok, missing_detail = True, 'missing ref_pos rejected with ValueError'
        except Exception as error:
            missing_ok, missing_detail = False, f'{type(error).__name__}: {error}'
        update_check(current_index, missing_ok, missing_detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'train_script:independent_training',
        'train_script:grad_clipping',
        'train_script:mock_data_cli',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        source, tree = _parse_source(path)
        independent = (
            not _imports_onescience(tree)
            and _has_attribute_call(tree, 'AdamW')
            and _has_attribute_call(tree, 'backward')
        )
        ok = independent
        msg = 'no onescience import; AdamW optimizer and backward pass are present'
        update_check(current_index, ok, msg)
        current_index += 1

        grad_clip = _has_attribute_call(tree, 'clip_grad_norm_') or _has_attribute_call(tree, 'clip_grad_norm')
        ok = grad_clip
        msg = 'gradient clipping is applied during training'
        update_check(current_index, ok, msg)
        current_index += 1

        cli_ok = all(flag in source for flag in REQUIRED_CLI_FLAGS)
        ok = cli_ok
        msg = f'required CLI flags present: {", ".join(REQUIRED_CLI_FLAGS)}'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'model_structure:folding_dit_modules',
        'model_structure:atom_token_dimensions',
        'model_structure:module_order',
        'model_structure:module_io_shapes',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, nn = _load_torch()
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        candidate = load_module_from_path('simplefold_train_submission', path)
        model = candidate.FoldingDiT()

        modules_ok = all(hasattr(model, name) for name in REQUIRED_MODULES)
        ok = modules_ok
        msg = f'required={REQUIRED_MODULES}'
        update_check(current_index, ok, msg)
        current_index += 1

        dimensions_ok = (
            getattr(model, 'hidden_size', None) == HIDDEN_SIZE
            and getattr(model, 'esm_dim', None) == ESM_DIM
            and isinstance(model.velocity_head, nn.Linear)
            and model.velocity_head.in_features == HIDDEN_SIZE
            and model.velocity_head.out_features == 3
            and isinstance(model.esm_projection, nn.Linear)
            and model.esm_projection.in_features == ESM_DIM
            and model.esm_projection.out_features == HIDDEN_SIZE
        )
        ok = dimensions_ok
        msg = (
            f'hidden={getattr(model, "hidden_size", None)}, '
            f'esm={getattr(model, "esm_dim", None)}, '
            f'velocity out={getattr(model.velocity_head, "out_features", None)}'
        )
        update_check(current_index, ok, msg)
        current_index += 1

        actual_order = [name for name, _child in model.named_children()]
        dataflow = ['coordinate_embedding', 'atom_encoder', 'trunk', 'atom_decoder', 'velocity_head']
        positions = [actual_order.index(name) for name in dataflow if name in actual_order]
        order_ok = len(positions) == len(dataflow) and positions == sorted(positions)
        ok = order_ok
        msg = f'dataflow={dataflow}, actual={actual_order}'
        update_check(current_index, ok, msg)
        current_index += 1

        probes_ok = True
        probe_details = []
        with torch.no_grad():
            for name, raw_input, expected_out in (
                ('coordinate_embedding', torch.randn(BATCH_SIZE, ATOM_COUNT, 3), (BATCH_SIZE, ATOM_COUNT, HIDDEN_SIZE)),
                ('reference_embedding', torch.randn(BATCH_SIZE, ATOM_COUNT, 3), (BATCH_SIZE, ATOM_COUNT, HIDDEN_SIZE)),
                ('self_condition_embedding', torch.randn(BATCH_SIZE, ATOM_COUNT, 3), (BATCH_SIZE, ATOM_COUNT, HIDDEN_SIZE)),
                ('time_embedder', torch.randn(BATCH_SIZE, 1), (BATCH_SIZE, HIDDEN_SIZE)),
                ('esm_projection', torch.randn(BATCH_SIZE, TOKEN_COUNT, ESM_DIM), (BATCH_SIZE, TOKEN_COUNT, HIDDEN_SIZE)),
                ('atom_encoder', torch.randn(BATCH_SIZE, ATOM_COUNT, HIDDEN_SIZE), (BATCH_SIZE, ATOM_COUNT, HIDDEN_SIZE)),
                ('trunk', torch.randn(BATCH_SIZE, TOKEN_COUNT, HIDDEN_SIZE), (BATCH_SIZE, TOKEN_COUNT, HIDDEN_SIZE)),
                ('atom_decoder', torch.randn(BATCH_SIZE, ATOM_COUNT, HIDDEN_SIZE), (BATCH_SIZE, ATOM_COUNT, HIDDEN_SIZE)),
                ('velocity_head', torch.randn(BATCH_SIZE, ATOM_COUNT, HIDDEN_SIZE), (BATCH_SIZE, ATOM_COUNT, 3)),
            ):
                module = getattr(model, name, None)
                try:
                    output = module(raw_input)
                    shape_ok = isinstance(output, torch.Tensor) and tuple(output.shape) == expected_out
                except Exception as error:
                    shape_ok = False
                    output = None
                    probe_details.append(f'{name}:{type(error).__name__}')
                if not shape_ok:
                    probe_details.append(
                        f'{name}->{tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__}!={expected_out}'
                    )
                    probes_ok = False
        ok = probes_ok
        msg = '; '.join(probe_details) if probe_details else 'atom/token embeddings, time embedder, trunk, decoder and velocity head match the I/O contract'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'forward_executable:output_keys',
        'forward_executable:output_shapes',
        'forward_executable:finite',
        'forward_executable:masked_atoms_zeroed',
        'forward_executable:time_sensitivity',
        'forward_executable:self_conditioning',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        candidate = load_module_from_path('simplefold_train_submission', path)
        model = candidate.FoldingDiT().eval()
        batch = candidate.make_synthetic_batch(
            batch_size=BATCH_SIZE, atom_count=ATOM_COUNT, token_count=TOKEN_COUNT, esm_dim=ESM_DIM, seed=23
        )
        noised_pos = batch['positions'] * 0.25
        t = torch.tensor([0.25, 0.75])
        with torch.inference_mode():
            output = model(noised_pos, t, batch['feats'])

        keys_ok = isinstance(output, dict) and set(output) == {'predict_velocity', 'latent'}
        ok = keys_ok
        msg = str(sorted(output) if isinstance(output, dict) else type(output).__name__)
        update_check(current_index, ok, msg)
        current_index += 1

        velocity = output.get('predict_velocity') if isinstance(output, dict) else None
        latent = output.get('latent') if isinstance(output, dict) else None
        shapes_ok = (
            isinstance(velocity, torch.Tensor)
            and tuple(velocity.shape) == (BATCH_SIZE, ATOM_COUNT, 3)
            and isinstance(latent, torch.Tensor)
            and tuple(latent.shape) == (BATCH_SIZE, TOKEN_COUNT, HIDDEN_SIZE)
        )
        ok = shapes_ok
        msg = f'velocity={tuple(velocity.shape) if isinstance(velocity, torch.Tensor) else None}, latent={tuple(latent.shape) if isinstance(latent, torch.Tensor) else None}'
        update_check(current_index, ok, msg)
        current_index += 1

        finite = all(
            isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
            for value in (velocity, latent)
        )
        ok = finite
        msg = 'velocity and latent are finite'
        update_check(current_index, ok, msg)
        current_index += 1

        masked = {key: (value.clone() if isinstance(value, torch.Tensor) else value) for key, value in batch['feats'].items()}
        masked['atom_pad_mask'] = batch['feats']['atom_pad_mask'].clone()
        masked['atom_pad_mask'][:, -2:] = False
        with torch.inference_mode():
            masked_output = model(noised_pos, t, masked)
        masked_velocity = masked_output['predict_velocity'] if isinstance(masked_output, dict) else None
        masked_ok = (
            isinstance(masked_velocity, torch.Tensor)
            and bool((masked_velocity[:, -2:] == 0).all())
            and bool(torch.isfinite(masked_velocity).all())
        )
        ok = masked_ok
        msg = 'velocity rows of padded atoms are zeroed by the atom mask'
        update_check(current_index, ok, msg)
        current_index += 1

        with torch.inference_mode():
            swapped = model(noised_pos, torch.tensor([0.75, 0.25]), batch['feats'])
        time_ok = (
            isinstance(swapped, dict)
            and not torch.allclose(swapped['predict_velocity'], velocity, atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
        )
        ok = time_ok
        msg = 'swapping the diffusion time changes the predicted velocity'
        update_check(current_index, ok, msg)
        current_index += 1

        with torch.inference_mode():
            conditioned = model(noised_pos, t, batch['feats'], self_cond=torch.zeros_like(noised_pos))
        self_cond_ok = (
            isinstance(conditioned, dict)
            and not torch.allclose(conditioned['predict_velocity'], velocity, atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
        )
        ok = self_cond_ok
        msg = 'providing self-conditioning coordinates changes the predicted velocity'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['forward_equivalence:velocity_and_latent']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        candidate_path = CANDIDATE_DIR / SUBMISSION_FILENAME
        candidate = load_module_from_path('simplefold_train_submission', candidate_path)
        reference = load_module_from_path('simplefold_train_reference', STANDARD_PATH)
        reference_model = reference.FoldingDiT().eval()
        candidate_model = candidate.FoldingDiT().eval()
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)

        batch = candidate.make_synthetic_batch(
            batch_size=BATCH_SIZE, atom_count=ATOM_COUNT, token_count=TOKEN_COUNT, esm_dim=ESM_DIM, seed=23
        )
        noised_pos = batch['positions'] * 0.25
        t = torch.tensor([0.25, 0.75])
        with torch.inference_mode():
            expected = reference_model(noised_pos, t, batch['feats'])
            generated = candidate_model(noised_pos, t, batch['feats'])
        equivalent = all(
            torch.allclose(generated[name], expected[name], atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
            for name in expected
        )
        max_diff = max(
            float((expected[name] - generated[name]).abs().max()) for name in expected
        ) if equivalent else float('inf')
        ok = equivalent
        msg = f'fixed-weight outputs match the standard implementation; max_abs_diff={max_diff:.3e}, atol={FORWARD_COMPARE_ATOL}'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['traning:flow_matching_fit', 'traning:cli_mock_fit']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        candidate = load_module_from_path('simplefold_train_submission', path)

        manager = tempfile.TemporaryDirectory()
        output_dir = Path(manager.name)

        model = candidate.FoldingDiT()
        parameter_before = next(model.parameters()).detach().clone()
        trainer = candidate.SimpleFoldTrainer(model, output_dir, learning_rate=1e-3, max_steps=2, seed=29)
        history = trainer.fit([candidate.make_synthetic_batch(seed=29)])

        finite = (
            isinstance(history, list)
            and len(history) == 2
            and all(
                isinstance(row, dict) and all(math.isfinite(float(row.get(name, float('nan')))) for name in REQUIRED_METRIC_KEYS)
                for row in history
            )
        )
        updated = not torch.equal(parameter_before, next(model.parameters()).detach())
        ok = finite and updated
        msg = f'{len(history) if isinstance(history, list) else 0} finite optimizer steps; parameters changed={updated}'
        update_check(current_index, ok, msg)
        current_index += 1

        cli = path
        cli_run = run_cli(
            cli,
            ['--output-dir', str(output_dir / 'cli_run'), '--max-steps', '2', '--seed', '29', '--mock-data'],
        )
        cli_ok = (
            cli_run.returncode == 0
            and (output_dir / 'cli_run' / 'metrics.jsonl').is_file()
            and (output_dir / 'cli_run' / 'simplefold_train.pt').is_file()
        )
        ok = cli_ok
        msg = f'returncode={cli_run.returncode}; stderr={cli_run.stderr.strip()[:200]}'
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['artifacts:training_outputs', 'artifacts:metrics_rows']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        candidate = load_module_from_path('simplefold_train_submission', path)
        manager = tempfile.TemporaryDirectory()
        output_dir = Path(manager.name)
        model = candidate.FoldingDiT()
        trainer = candidate.SimpleFoldTrainer(model, output_dir, learning_rate=1e-3, max_steps=2, seed=29)
        history = trainer.fit([candidate.make_synthetic_batch(seed=29)])

        required = ('metrics.jsonl', 'simplefold_train.pt', 'predictions.json', 'targets.json')
        artifact_ok = all((output_dir / name).is_file() for name in required)
        detail = f'files present: {[name for name in required if (output_dir / name).is_file()]}'
        if artifact_ok:
            checkpoint = torch.load(output_dir / 'simplefold_train.pt', map_location='cpu', weights_only=False)
            artifact_ok = (
                isinstance(checkpoint, dict)
                and checkpoint.get('format') == 'simplefold-flow-train-v1'
                and {'model_state_dict', 'optimizer_state_dict', 'completed_steps'} <= set(checkpoint)
            )
            detail = f'checkpoint fields={sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__}'
        if artifact_ok:
            predictions = json.loads((output_dir / 'predictions.json').read_text(encoding='utf-8'))['predictions']
            targets = json.loads((output_dir / 'targets.json').read_text(encoding='utf-8'))['targets']
            artifact_ok = (
                isinstance(predictions, list)
                and isinstance(targets, list)
                and len(predictions) == len(targets) > 0
            )
            detail = f'predictions={len(predictions) if isinstance(predictions, list) else "?"}, targets={len(targets) if isinstance(targets, list) else "?"}'
        ok = artifact_ok
        msg = detail
        update_check(current_index, ok, msg)
        current_index += 1

        rows_ok = False
        rows_detail = 'metrics.jsonl missing'
        metrics_path = output_dir / 'metrics.jsonl'
        if metrics_path.is_file():
            rows = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            rows_ok = len(rows) == 2 and all(
                {'step', *REQUIRED_METRIC_KEYS} <= set(row)
                and all(math.isfinite(float(row[name])) for name in REQUIRED_METRIC_KEYS)
                for row in rows
            )
            rows_detail = f'{len(rows)} rows with finite {REQUIRED_METRIC_KEYS}'
        update_check(current_index, rows_ok, rows_detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
