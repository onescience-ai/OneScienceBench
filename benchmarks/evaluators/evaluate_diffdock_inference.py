'''V2 adapter for the legacy DiffDock inference evaluator.'''
from pathlib import Path
import sys

import ast
import csv
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

SUBMISSION_FILENAME = 'diffdock_inference.py'
SEED = 20260819
FORWARD_COMPARE_ATOL = 1e-6
CLI_TIMEOUT_SECONDS = 180
DEFAULT_HIDDEN = 32
DEFAULT_DISTANCE_BINS = 16
SAMPLE_COUNT = 3
REQUIRED_CLI_FLAGS = ('--protein-path', '--ligand-path', '--checkpoint', '--out-dir', '--samples', '--seed')
REQUIRED_MODULES = (
    'ligand_encoder',
    'receptor_encoder',
    'cross_edge_encoder',
    'cross_message',
    'translation_head',
    'rotation_head',
    'torsion_head',
)
STANDARD_ASSETS = ('inference_config.json',)


def _external_assets_contract(payload: dict[str, Any]) -> tuple[bool, str]:
    assets = payload.get('external_assets')
    if not isinstance(assets, dict) or assets.get('mount_required') is not True:
        return False, 'inference_config.json must declare mount_required=true external_assets'
    missing = [name for name in ('weights', 'dataset') if not isinstance(assets.get(name), str) or not assets[name].startswith('/public/share/')]
    if missing:
        return False, f'external assets must use onestore paths for: {missing}'
    return True, 'complete weights and dataset are declared as externally mounted assets'


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


def _fixture_protein() -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]]


def _fixture_ligand() -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.8, 0.0]]


def _write_protein_pdb(path: Path, coordinates: list[list[float]]) -> None:
    lines = []
    for index, (x, y, z) in enumerate(coordinates, 1):
        lines.append(
            f'ATOM  {index:5d}  C   ALA A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C'
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_ligand_sdf(path: Path, coordinates: list[list[float]]) -> None:
    lines = ['ligand', 'OneScience', '', f'{len(coordinates):3d}  0  0  0  0  0  0  0  0999 V2000']
    lines.extend(f'{x:10.4f}{y:10.4f}{z:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0' for x, y, z in coordinates)
    lines.extend(['M  END', '$$$$'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:diffdock_inference.py', 'file:standard:inference_config.json']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        context: dict[str, Any] = {'device': 'cpu'}
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        context['submission_path'] = path
        context['candidate'] = load_module_from_path('diffdock_submission', path)
        context['reference'] = load_module_from_path('diffdock_reference', REFERENCE_DIR / SUBMISSION_FILENAME)
        context['temp_manager'] = tempfile.TemporaryDirectory()
        context['work_dir'] = Path(context['temp_manager'].name)
        ok = True
        msg = str(path)
        update_check(current_index, ok, msg)
        current_index += 1
        for name in STANDARD_ASSETS:
            path = REFERENCE_DIR / name
            ok = path.is_file()
            detail = str(path)
            if ok and path.suffix == '.json':
                try:
                    payload = json.loads(path.read_text(encoding='utf-8'))
                    ok = isinstance(payload, dict)
                    if ok:
                        ok, detail = _external_assets_contract(payload)
                    else:
                        detail = f'{path} (JSON object required)'
                except (OSError, json.JSONDecodeError) as error:
                    ok = False
                    detail = f'{path}: {type(error).__name__}: {error}'
            elif ok and path.suffix in {'.pdb', '.sdf'}:
                try:
                    ok = bool(path.read_text(encoding='utf-8').strip())
                except OSError as error:
                    ok = False
                    detail = f'{path}: {type(error).__name__}: {error}'
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'checkpoint_loading:strict_state_dict',
        'checkpoint_loading:checkpoint_file_load',
        'checkpoint_loading:state_key_contract',
        'checkpoint_loading:uses_torch_load',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        context: dict[str, Any] = {'device': 'cpu'}
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        context['submission_path'] = path
        context['candidate'] = load_module_from_path('diffdock_submission', path)
        context['reference'] = load_module_from_path('diffdock_reference', REFERENCE_DIR / SUBMISSION_FILENAME)
        context['temp_manager'] = tempfile.TemporaryDirectory()
        context['work_dir'] = Path(context['temp_manager'].name)
        torch, nn = _load_torch()
        candidate = context['candidate']
        reference_model = context['reference'].DiffDock().eval()
        context['reference_model'] = reference_model
        state_dict = reference_model.state_dict()

        model = candidate.DiffDock()
        model.load_state_dict(state_dict, strict=True)
        context['model'] = model.eval()
        ok = True
        msg = 'reference state_dict loaded with strict=True'
        update_check(current_index, ok, msg)
        current_index += 1

        work_dir = context['work_dir']
        checkpoint_path = work_dir / 'reference_checkpoint.pt'
        torch.save({'model_state_dict': state_dict}, checkpoint_path)
        protein_path = work_dir / 'protein.pdb'
        ligand_path = work_dir / 'ligand.sdf'
        _write_protein_pdb(protein_path, _fixture_protein())
        _write_ligand_sdf(ligand_path, _fixture_ligand())
        context['protein_path'] = protein_path
        context['ligand_path'] = ligand_path
        context['checkpoint_path'] = checkpoint_path

        out_dir = work_dir / 'cli_checkpoint_load'
        completed = run_cli(
            context['submission_path'],
            [
                '--protein-path', str(protein_path),
                '--ligand-path', str(ligand_path),
                '--checkpoint', str(checkpoint_path),
                '--out-dir', str(out_dir),
                '--samples', '1',
                '--seed', '5',
            ],
        )
        load_ok = completed.returncode == 0 and (out_dir / 'scores.csv').is_file()
        update_check(current_index, load_ok, f'CLI loads the saved reference weights; returncode={completed.returncode}; stderr={completed.stderr.strip()[:200]}')
        current_index += 1
        keys = set(reference_model.state_dict())
        update_check(current_index, bool(keys) and all(isinstance(key, str) for key in keys), f'state_dict keys={len(keys)}')
        current_index += 1

        source, _tree = _parse_source(context['submission_path'])
        update_check(current_index, 'torch.load' in source, 'checkpoint loading uses torch.load')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'model_structure:cross_graph_score_model',
        'model_structure:head_shapes',
        'model_structure:rbf_cross_edges',
        'model_structure:module_order',
        'model_structure:torsion_head_contract',
        'model_structure:module_io_shapes',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        context: dict[str, Any] = {'device': 'cpu'}
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        context['submission_path'] = path
        context['candidate'] = load_module_from_path('diffdock_submission', path)
        context['reference'] = load_module_from_path('diffdock_reference', REFERENCE_DIR / SUBMISSION_FILENAME)
        context['temp_manager'] = tempfile.TemporaryDirectory()
        context['work_dir'] = Path(context['temp_manager'].name)
        reference_model = context['reference'].DiffDock().eval()
        context['reference_model'] = reference_model
        state_dict = reference_model.state_dict()
        model = context['candidate'].DiffDock()
        model.load_state_dict(state_dict, strict=True)
        context['model'] = model.eval()

        torch, nn = _load_torch()
        model = context['model']
        modules_ok = all(hasattr(model, name) for name in REQUIRED_MODULES)
        update_check(current_index, modules_ok, f'required={REQUIRED_MODULES}')
        current_index += 1

        head_ok = (
            isinstance(model.translation_head, nn.Linear)
            and model.translation_head.out_features == 3
            and isinstance(model.rotation_head, nn.Linear)
            and model.rotation_head.out_features == 3
        )
        update_check(current_index, head_ok, f'translation_head={type(model.translation_head).__name__}({getattr(model.translation_head, "out_features", "?")}), rotation_head={type(model.rotation_head).__name__}({getattr(model.rotation_head, "out_features", "?")})')
        current_index += 1
        distance_bins = getattr(model, 'distance_bins', None)
        rbf_ok = (
            distance_bins == DEFAULT_DISTANCE_BINS
            and isinstance(model.cross_edge_encoder, nn.Sequential)
            and model.cross_edge_encoder[0].in_features == DEFAULT_DISTANCE_BINS
        )
        update_check(current_index, rbf_ok, f'distance_bins={distance_bins}, cross_edge_encoder in_features={getattr(model.cross_edge_encoder[0], "in_features", "?")}')
        current_index += 1

        order = [name for name, _ in model.named_children()]
        order_ok = order[:7] == ['ligand_encoder', 'receptor_encoder', 'cross_edge_encoder', 'cross_message', 'translation_head', 'rotation_head', 'torsion_head']
        update_check(current_index, order_ok, f'children={order}')
        current_index += 1
        torsion_ok = isinstance(model.torsion_head, nn.Sequential) and model.torsion_head[-1].out_features == 1
        update_check(current_index, torsion_ok, f'torsion_out={getattr(model.torsion_head[-1], "out_features", None)}')
        current_index += 1

        probes_ok = True
        probe_details = []
        with torch.no_grad():
            for name, raw_input, expected_out in (
                ('ligand_encoder', torch.randn(3, 3), (3, DEFAULT_HIDDEN)),
                ('receptor_encoder', torch.randn(4, 3), (4, DEFAULT_HIDDEN)),
                ('cross_edge_encoder', torch.randn(3, 4, DEFAULT_DISTANCE_BINS), (3, 4, DEFAULT_HIDDEN)),
                ('cross_message', torch.randn(3, 4, DEFAULT_HIDDEN * 3), (3, 4, DEFAULT_HIDDEN)),
                ('translation_head', torch.randn(DEFAULT_HIDDEN), (3,)),
                ('rotation_head', torch.randn(2, DEFAULT_HIDDEN), (2, 3)),
                ('torsion_head', torch.randn(2, DEFAULT_HIDDEN * 2), (2, 1)),
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
                    probe_details.append(f'{name}->{tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__}!={expected_out}')
                    probes_ok = False
        if not (
            isinstance(model.cross_message, nn.Sequential)
            and model.cross_message[0].in_features == DEFAULT_HIDDEN * 3
        ):
            probes_ok = False
            probe_details.append('cross_message must fuse ligand+receptor+edge features (in_features=3*hidden)')
        update_check(current_index, probes_ok, '; '.join(probe_details) if probe_details else 'encoders, RBF cross edges, cross message fusion and tr/rot/tor heads match the I/O contract')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'forward_executable:score_keys',
        'forward_executable:score_shapes',
        'forward_executable:finite',
        'forward_executable:translation_invariance',
        'forward_executable:deterministic',
        'forward_executable:invalid_shape_rejected',
        'forward_executable:time_sensitivity',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        context: dict[str, Any] = {'device': 'cpu'}
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        context['submission_path'] = path
        context['candidate'] = load_module_from_path('diffdock_submission', path)
        context['reference'] = load_module_from_path('diffdock_reference', REFERENCE_DIR / SUBMISSION_FILENAME)
        reference_model = context['reference'].DiffDock().eval()
        context['reference_model'] = reference_model
        state_dict = reference_model.state_dict()
        model = context['candidate'].DiffDock()
        model.load_state_dict(state_dict, strict=True)
        context['model'] = model.eval()

        torch, _nn = _load_torch()
        model = context['model']
        protein = torch.tensor(_fixture_protein())
        ligand = torch.tensor(_fixture_ligand())
        with torch.inference_mode():
            output = model(protein, ligand, diffusion_time=0.5)
        context['protein'] = protein
        context['ligand'] = ligand
        context['output'] = output
        keys_ok = isinstance(output, dict) and set(output) == {'tr_pred', 'rot_pred', 'tor_pred'}
        update_check(current_index, keys_ok, str(sorted(output) if isinstance(output, dict) else type(output).__name__))
        current_index += 1

        tr = output.get('tr_pred') if isinstance(output, dict) else None
        rot = output.get('rot_pred') if isinstance(output, dict) else None
        tor = output.get('tor_pred') if isinstance(output, dict) else None
        shapes_ok = (
            isinstance(tr, torch.Tensor)
            and tuple(tr.shape) == (1, 3)
            and isinstance(rot, torch.Tensor)
            and tuple(rot.shape) == (1, 3)
            and isinstance(tor, torch.Tensor)
            and tuple(tor.shape) == (ligand.shape[0] - 1,)
        )
        update_check(current_index, shapes_ok, f'tr={tuple(tr.shape) if isinstance(tr, torch.Tensor) else None}, rot={tuple(rot.shape) if isinstance(rot, torch.Tensor) else None}, tor={tuple(tor.shape) if isinstance(tor, torch.Tensor) else None}')
        current_index += 1
        finite = all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in (tr, rot, tor))
        update_check(current_index, finite, 'all score tensors finite')
        current_index += 1

        offset = torch.tensor([3.0, -2.0, 1.5])
        with torch.inference_mode():
            shifted = model(protein + offset, ligand + offset, diffusion_time=0.5)
        invariance_ok = keys_ok and all(torch.allclose(output[name], shifted[name], atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL) for name in ('tr_pred', 'rot_pred', 'tor_pred'))
        update_check(current_index, invariance_ok, 'rigidly translating protein and ligand leaves the score predictions unchanged')
        current_index += 1

        with torch.inference_mode():
            repeat = model(protein, ligand, diffusion_time=0.5)
        deterministic_ok = keys_ok and all(torch.allclose(output[name], repeat[name], atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL) for name in ('tr_pred', 'rot_pred', 'tor_pred'))
        update_check(current_index, deterministic_ok, 'repeated forward passes return identical scores')
        current_index += 1

        invalid_ok = False
        try:
            model(protein[:, :2], ligand, diffusion_time=0.5)
        except (ValueError, RuntimeError):
            invalid_ok = True
        update_check(current_index, invalid_ok, 'malformed coordinate shape rejected')
        current_index += 1
        with torch.inference_mode():
            later = model(protein, ligand, diffusion_time=0.9)
        time_ok = not torch.allclose(output['tr_pred'], later['tr_pred'], atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
        update_check(current_index, time_ok, 'diffusion time changes translation score')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_execution(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'evaluation_interface:independent_cli',
        'evaluation_interface:eval_no_grad',
        'evaluation_interface:input_parsers',
        'evaluation_interface:cli_smoke',
        'evaluation_interface:raw_checkpoint_cli',
        'evaluation_interface:invalid_samples_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        context: dict[str, Any] = {'device': 'cpu'}
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        context['submission_path'] = path
        context['candidate'] = load_module_from_path('diffdock_submission', path)
        context['reference'] = load_module_from_path('diffdock_reference', REFERENCE_DIR / SUBMISSION_FILENAME)
        context['temp_manager'] = tempfile.TemporaryDirectory()
        context['work_dir'] = Path(context['temp_manager'].name)
        reference_model = context['reference'].DiffDock().eval()
        context['reference_model'] = reference_model
        state_dict = reference_model.state_dict()
        model = context['candidate'].DiffDock()
        model.load_state_dict(state_dict, strict=True)
        context['model'] = model.eval()
        work_dir = context['work_dir']
        checkpoint_path = work_dir / 'reference_checkpoint.pt'
        torch.save({'model_state_dict': state_dict}, checkpoint_path)
        protein_path = work_dir / 'protein.pdb'
        ligand_path = work_dir / 'ligand.sdf'
        _write_protein_pdb(protein_path, _fixture_protein())
        _write_ligand_sdf(ligand_path, _fixture_ligand())
        context['protein_path'] = protein_path
        context['ligand_path'] = ligand_path
        context['checkpoint_path'] = checkpoint_path

        source, tree = _parse_source(context['submission_path'])
        flags_ok = all(flag in source for flag in REQUIRED_CLI_FLAGS)
        independent = not _imports_onescience(tree) and flags_ok
        update_check(current_index, independent, f'required CLI flags and no onescience import; flags_ok={flags_ok}')
        current_index += 1

        no_grad_ok = '.eval()' in source and ('inference_mode' in source or 'no_grad' in source)
        update_check(current_index, no_grad_ok, 'entry point switches to eval mode and uses an inference context')
        current_index += 1

        candidate = context['candidate']
        torch, _nn = _load_torch()
        work_dir = context['work_dir']
        protein_path = work_dir / 'parser_protein.pdb'
        ligand_path = work_dir / 'parser_ligand.sdf'
        _write_protein_pdb(protein_path, _fixture_protein())
        _write_ligand_sdf(ligand_path, _fixture_ligand())
        parser_ok = False
        parser_detail = 'parsers failed'
        try:
            parsed_protein = candidate.parse_pdb_coordinates(protein_path)
            parsed_ligand = candidate.parse_sdf_coordinates(ligand_path)
            parser_ok = (
                torch.is_tensor(parsed_protein)
                and tuple(parsed_protein.shape) == (4, 3)
                and torch.is_tensor(parsed_ligand)
                and tuple(parsed_ligand.shape) == (3, 3)
            )
            parser_detail = f'protein={tuple(parsed_protein.shape) if torch.is_tensor(parsed_protein) else type(parsed_protein).__name__}, ligand={tuple(parsed_ligand.shape) if torch.is_tensor(parsed_ligand) else type(parsed_ligand).__name__}'
        except Exception as error:
            parser_detail = f'{type(error).__name__}: {str(error)[:240]}'
        update_check(current_index, parser_ok, parser_detail)
        current_index += 1

        out_dir = work_dir / 'cli_smoke'
        completed = run_cli(
            context['submission_path'],
            [
                '--protein-path', str(context['protein_path']),
                '--ligand-path', str(context['ligand_path']),
                '--checkpoint', str(context['checkpoint_path']),
                '--out-dir', str(out_dir),
                '--samples', str(SAMPLE_COUNT),
                '--seed', '5',
            ],
        )
        smoke_ok = False
        smoke_detail = f'returncode={completed.returncode}; stderr={completed.stderr.strip()[:240]}'
        if completed.returncode == 0:
            scores_path = out_dir / 'scores.csv'
            if not scores_path.is_file():
                smoke_detail = 'scores.csv was not produced'
            else:
                with scores_path.open('r', encoding='utf-8', newline='') as handle:
                    rows = list(csv.DictReader(handle))
                pose_files = sorted(out_dir.glob('*.sdf'))
                scores = [float(row['score']) for row in rows]
                ranks_ok = [row['rank'] for row in rows] == [str(index) for index in range(1, SAMPLE_COUNT + 1)]
                columns_ok = rows and {'rank', 'score', 'pose_file'} <= set(rows[0])
                sorted_ok = scores == sorted(scores, reverse=True)
                files_ok = len(pose_files) == SAMPLE_COUNT
                atoms_ok = True
                for pose_path in pose_files:
                    lines = pose_path.read_text(encoding='utf-8').splitlines()
                    try:
                        atoms_ok = atoms_ok and int(lines[3][:3]) == 3
                    except (ValueError, IndexError):
                        atoms_ok = False
                smoke_ok = columns_ok and ranks_ok and sorted_ok and files_ok and atoms_ok
                smoke_detail = f'rows={len(rows)}, columns_ok={columns_ok}, ranks_ok={ranks_ok}, sorted_desc={sorted_ok}, pose_files={len(pose_files)}, sdf_atoms_ok={atoms_ok}'
        update_check(current_index, smoke_ok, smoke_detail)
        current_index += 1

        raw_dir = work_dir / 'cli_raw_checkpoint'
        raw_checkpoint = work_dir / 'raw_state.pt'
        torch.save(context['reference_model'].state_dict(), raw_checkpoint)
        raw_run = run_cli(context['submission_path'], ['--protein-path', str(context['protein_path']), '--ligand-path', str(context['ligand_path']), '--checkpoint', str(raw_checkpoint), '--out-dir', str(raw_dir), '--samples', '1', '--seed', '5'])
        update_check(current_index, raw_run.returncode == 0 and (raw_dir / 'scores.csv').is_file(), f'returncode={raw_run.returncode}')
        current_index += 1
        invalid_run = run_cli(context['submission_path'], ['--protein-path', str(context['protein_path']), '--ligand-path', str(context['ligand_path']), '--checkpoint', str(context['checkpoint_path']), '--out-dir', str(work_dir / 'invalid_samples'), '--samples', '0', '--seed', '5'])
        update_check(current_index, invalid_run.returncode != 0, f'returncode={invalid_run.returncode}')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = [
        'metric:score_equivalence',
        'metric:pose_sampling',
        'metric:score_pose_rule',
        'metric:seed_sensitivity',
        'metric:score_finiteness',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        context: dict[str, Any] = {'device': 'cpu'}
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        path = CANDIDATE_DIR / SUBMISSION_FILENAME
        context['submission_path'] = path
        context['candidate'] = load_module_from_path('diffdock_submission', path)
        context['reference'] = load_module_from_path('diffdock_reference', REFERENCE_DIR / SUBMISSION_FILENAME)
        reference_model = context['reference'].DiffDock().eval()
        context['reference_model'] = reference_model
        state_dict = reference_model.state_dict()
        model = context['candidate'].DiffDock()
        model.load_state_dict(state_dict, strict=True)
        context['model'] = model.eval()
        protein = torch.tensor(_fixture_protein())
        ligand = torch.tensor(_fixture_ligand())
        with torch.inference_mode():
            output = context['model'](protein, ligand, diffusion_time=0.5)
        context['protein'] = protein
        context['ligand'] = ligand
        context['output'] = output

        torch, _nn = _load_torch()
        reference_model = context['reference_model']
        model = context['model']
        protein = context['protein']
        ligand = context['ligand']
        output = context['output']
        metrics: dict[str, float] = {}
        with torch.inference_mode():
            expected = reference_model(protein, ligand, diffusion_time=0.5)
        equivalent = True
        for name in ('tr_pred', 'rot_pred', 'tor_pred'):
            diff = float((expected[name] - output[name]).abs().max())
            metrics[f'{name.replace("_pred", "")}_mae'] = float((expected[name] - output[name]).abs().mean())
            equivalent = equivalent and torch.allclose(output[name], expected[name], atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
        update_check(current_index, equivalent, f'fixed-weight tr/rot/tor scores match the standard implementation; atol={FORWARD_COMPARE_ATOL}')
        current_index += 1

        poses = model.sample(protein, ligand, samples=SAMPLE_COUNT, seed=5)
        poses_ok = (
            isinstance(poses, list)
            and len(poses) == SAMPLE_COUNT
            and all(
                isinstance(pose, dict)
                and tuple(pose['translation'].shape) == (3,)
                and tuple(pose['rotation_score'].shape) == (3,)
                and tuple(pose['torsion_score'].shape) == (ligand.shape[0] - 1,)
                and isinstance(pose['score'], float)
                for pose in poses
            )
        )
        update_check(current_index, poses_ok, f'{len(poses) if isinstance(poses, list) else 0} poses carry translation, rotation, torsion and confidence score')
        current_index += 1

        center = protein.mean(dim=0) - ligand.mean(dim=0)
        with torch.inference_mode():
            near_score = model.score_pose(protein, ligand, center)
            far_score = model.score_pose(protein, ligand, center + torch.tensor([50.0, 0.0, 0.0]))
        rule_ok = bool(near_score > far_score)
        metrics['pose_score_gap'] = float(near_score - far_score)
        update_check(current_index, rule_ok, f'near={float(near_score):.4f} > far={float(far_score):.4f}: contact score decreases as the ligand moves away')
        current_index += 1

        other_poses = model.sample(protein, ligand, samples=SAMPLE_COUNT, seed=6)
        seed_sensitive = any(abs(float(a['score']) - float(b['score'])) > 1e-8 for a, b in zip(poses, other_poses))
        score_finite = all(math.isfinite(float(pose['score'])) for pose in poses)
        update_check(current_index, seed_sensitive, 'different seeds change sampled pose scores')
        current_index += 1
        update_check(current_index, score_finite, 'all pose confidence scores are finite')
        current_index += 1
        context['metrics'] = metrics
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
