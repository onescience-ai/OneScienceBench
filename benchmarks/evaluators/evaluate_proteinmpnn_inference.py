'''V2 adapter for the legacy ProteinMPNN inference evaluator.'''
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

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

SUBMISSION_FILENAME = 'proteinmpnn_inference.py'
SEED = 20260819
FORWARD_COMPARE_ATOL = 1e-6
CLI_TIMEOUT_SECONDS = 180
DEFAULT_HIDDEN = 32
DEFAULT_K = 4
DEFAULT_EDGE_FEATURES = 19
DEFAULT_ENCODER_LAYERS = 2
DEFAULT_DECODER_LAYERS = 2
DEFAULT_VOCAB = 21
SAMPLE_SEQUENCES = 3
REQUIRED_CLI_FLAGS = (
    '--pdb-path', '--checkpoint', '--out-folder', '--num-seq-per-target',
    '--temperature', '--seed', '--device',
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
    return subprocess.run(command, cwd=str(script.parent), capture_output=True, text=True, timeout=CLI_TIMEOUT_SECONDS, check=False)

def _fixture_coordinates() -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.2, 0.0], [4.5, 0.3, 0.1], [5.7, 0.6, 0.2]]

def _write_pdb(path: Path, coordinates: list[list[float]]) -> None:
    lines = []
    for index, (x, y, z) in enumerate(coordinates, 1):
        lines.append(f'ATOM  {index:5d}  CA  ALA A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if header is not None:
                records.append((header, ''.join(chunks)))
            header = line[1:]
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, ''.join(chunks)))
    return records

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['file:proteinmpnn_inference.py', 'file:standard:inference_config.json']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        path = (CANDIDATE_DIR / SUBMISSION_FILENAME).resolve()
        candidate = load_module_from_path('proteinmpnn_submission', path)
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        reference = load_module_from_path('proteinmpnn_reference', reference_path)
        temp_manager = tempfile.TemporaryDirectory()
        work_dir = Path(temp_manager.name)
        update_check(current_index, True, str(path))
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
            elif ok and path.suffix == '.pdb':
                try:
                    ok = bool(path.read_text(encoding='utf-8').strip())
                except OSError as error:
                    ok = False
                    detail = f'{path}: {type(error).__name__}: {error}'
            update_check(current_index, ok, detail)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['checkpoint_loading:strict_state_dict', 'checkpoint_loading:checkpoint_file_load', 'checkpoint_loading:state_key_contract', 'checkpoint_loading:uses_torch_load']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        submission_path = (CANDIDATE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        candidate = load_module_from_path('proteinmpnn_submission', submission_path)
        reference = load_module_from_path('proteinmpnn_reference', reference_path)
        reference_model = reference.ProteinMPNN().eval()
        state_dict = reference_model.state_dict()
        model = candidate.ProteinMPNN()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        update_check(current_index, True, 'reference state_dict loaded with strict=True')
        current_index += 1
        temp_manager = tempfile.TemporaryDirectory()
        work_dir = Path(temp_manager.name)
        checkpoint_path = work_dir / 'reference_checkpoint.pt'
        torch.save({'model_state_dict': state_dict}, checkpoint_path)
        pdb_path = work_dir / 'input.pdb'
        _write_pdb(pdb_path, _fixture_coordinates())
        out_folder = work_dir / 'cli_checkpoint_load'
        completed = run_cli(submission_path, ['--pdb-path', str(pdb_path), '--checkpoint', str(checkpoint_path), '--out-folder', str(out_folder), '--num-seq-per-target', '1', '--temperature', '0.5', '--seed', '9', '--device', 'cpu'])
        load_ok = completed.returncode == 0 and (out_folder / 'seqs' / 'designed.fasta').is_file()
        update_check(current_index, load_ok, f'CLI loads the saved reference weights; returncode={completed.returncode}; stderr={completed.stderr.strip()[:200]}')
        current_index += 1
        keys = set(reference_model.state_dict())
        update_check(current_index, bool(keys) and all(isinstance(key, str) for key in keys), f'state_dict keys={len(keys)}')
        current_index += 1
        source, _tree = _parse_source(submission_path)
        update_check(current_index, 'torch.load' in source, 'checkpoint loading uses torch.load')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['model_structure:encoder_decoder_graph', 'model_structure:message_passing_layers', 'model_structure:embedding_dimensions', 'model_structure:knn_and_edge_features', 'model_structure:module_io_shapes', 'model_structure:module_order']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, nn = _load_torch()
        torch.manual_seed(SEED)
        submission_path = (CANDIDATE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        candidate = load_module_from_path('proteinmpnn_submission', submission_path)
        reference = load_module_from_path('proteinmpnn_reference', reference_path)
        reference_model = reference.ProteinMPNN().eval()
        state_dict = reference_model.state_dict()
        model = candidate.ProteinMPNN()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        graph_ok = (isinstance(model.node_embedding, nn.Linear) and model.node_embedding.in_features == 3 and model.node_embedding.out_features == DEFAULT_HIDDEN and isinstance(model.edge_embedding, nn.Linear) and isinstance(model.encoder_layers, nn.ModuleList) and len(model.encoder_layers) == DEFAULT_ENCODER_LAYERS and isinstance(model.sequence_embedding, nn.Embedding) and model.sequence_embedding.num_embeddings == DEFAULT_VOCAB + 1 and isinstance(model.decoder_layers, nn.ModuleList) and len(model.decoder_layers) == DEFAULT_DECODER_LAYERS and isinstance(model.output_projection, nn.Linear) and model.output_projection.in_features == DEFAULT_HIDDEN and model.output_projection.out_features == DEFAULT_VOCAB)
        update_check(current_index, graph_ok, str(model))
        current_index += 1
        message_ok = all(hasattr(layer, 'message_mlp') and hasattr(layer, 'update_mlp') and hasattr(layer, 'norm') for layer in [*model.encoder_layers, *model.decoder_layers])
        update_check(current_index, message_ok, 'two encoder and two decoder message layers with message_mlp/update_mlp/norm')
        current_index += 1
        embedding_ok = model.sequence_embedding.embedding_dim == DEFAULT_HIDDEN and model.sequence_embedding.num_embeddings == DEFAULT_VOCAB + 1 and model.output_projection.out_features == DEFAULT_VOCAB
        update_check(current_index, embedding_ok, f'sequence_vocab={model.sequence_embedding.num_embeddings}, output_vocab={model.output_projection.out_features}')
        current_index += 1
        knn_ok = getattr(model, 'k_neighbors', None) == DEFAULT_K and model.edge_embedding.in_features == DEFAULT_EDGE_FEATURES
        update_check(current_index, knn_ok, f'k={getattr(model, "k_neighbors", None)}, edge_features={getattr(model.edge_embedding, "in_features", "?")} (16 RBF + 3 unit direction)')
        current_index += 1
        coordinates = torch.tensor(_fixture_coordinates())
        probes_ok = True
        probe_details = []
        with torch.no_grad():
            neighbors, edges = model.build_geometry_graph(coordinates)
            for name, module, raw_input, expected_out in (('node_embedding', model.node_embedding, coordinates, (len(coordinates), DEFAULT_HIDDEN)), ('edge_embedding', model.edge_embedding, torch.randn(len(coordinates), DEFAULT_K, DEFAULT_EDGE_FEATURES), (len(coordinates), DEFAULT_K, DEFAULT_HIDDEN)), ('output_projection', model.output_projection, torch.randn(len(coordinates), DEFAULT_HIDDEN), (len(coordinates), DEFAULT_VOCAB))):
                try:
                    output = module(raw_input)
                    shape_ok = torch.is_tensor(output) and tuple(output.shape) == expected_out
                except Exception as error:
                    shape_ok = False
                    output = None
                    probe_details.append(f'{name}:{type(error).__name__}')
                if not shape_ok:
                    probe_details.append(f'{name}->{tuple(output.shape) if torch.is_tensor(output) else type(output).__name__}!={expected_out}')
                    probes_ok = False
            nodes = model.node_embedding(coordinates)
            for layer_index, layer in enumerate([*model.encoder_layers, *model.decoder_layers]):
                try:
                    output = layer(nodes, edges, neighbors)
                    layer_ok = torch.is_tensor(output) and tuple(output.shape) == tuple(nodes.shape)
                except Exception as error:
                    layer_ok = False
                    output = None
                    probe_details.append(f'layer{layer_index}:{type(error).__name__}')
                if not layer_ok:
                    probe_details.append(f'layer{layer_index}->{tuple(output.shape) if torch.is_tensor(output) else type(output).__name__}!={tuple(nodes.shape)}')
                    probes_ok = False
        update_check(current_index, probes_ok, '; '.join(probe_details) if probe_details else 'embeddings, message layers and the output head match the I/O contract')
        current_index += 1
        actual_order = [name for name, _child in model.named_children()]
        dataflow = ['node_embedding', 'encoder_layers', 'sequence_embedding', 'decoder_layers', 'output_projection']
        positions = [actual_order.index(name) for name in dataflow if name in actual_order]
        order_ok = len(positions) == len(dataflow) and positions == sorted(positions)
        update_check(current_index, order_ok, f'dataflow={dataflow}, actual={actual_order}')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['forward_executable:knn_graph', 'forward_executable:no_self_neighbors', 'forward_executable:logit_contract', 'forward_executable:sequence_conditioning', 'forward_executable:translation_invariance', 'forward_executable:invalid_shape_rejected']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        submission_path = (CANDIDATE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        candidate = load_module_from_path('proteinmpnn_submission', submission_path)
        reference = load_module_from_path('proteinmpnn_reference', reference_path)
        reference_model = reference.ProteinMPNN().eval()
        state_dict = reference_model.state_dict()
        model = candidate.ProteinMPNN()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        coordinates = torch.tensor(_fixture_coordinates())
        with torch.inference_mode():
            neighbors, edges = model.build_geometry_graph(coordinates)
            output = model(coordinates)
        graph_ok = torch.is_tensor(neighbors) and tuple(neighbors.shape) == (len(coordinates), DEFAULT_K) and torch.is_tensor(edges) and tuple(edges.shape) == (len(coordinates), DEFAULT_K, DEFAULT_HIDDEN)
        update_check(current_index, graph_ok, f'neighbors={tuple(neighbors.shape) if torch.is_tensor(neighbors) else None}, edges={tuple(edges.shape) if torch.is_tensor(edges) else None}')
        current_index += 1
        self_ok = all(index not in neighbors[index].tolist() for index in range(len(coordinates)))
        update_check(current_index, self_ok, 'the kNN graph never links a residue to itself')
        current_index += 1
        logits_ok = torch.is_tensor(output) and tuple(output.shape) == (len(coordinates), DEFAULT_VOCAB) and bool(torch.isfinite(output).all())
        update_check(current_index, logits_ok, f'shape={tuple(output.shape) if torch.is_tensor(output) else type(output).__name__}, all finite')
        current_index += 1
        with torch.inference_mode():
            conditioned = model(coordinates, sequence_ids=torch.zeros(len(coordinates), dtype=torch.long))
        conditioning_ok = not torch.allclose(conditioned, output, atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
        update_check(current_index, conditioning_ok, 'conditioning on sequence ids changes the decoder logits')
        current_index += 1
        shifted = coordinates + torch.tensor([1.25, -0.5, 0.75])
        with torch.inference_mode():
            shifted_output = model(shifted)
        translation_ok = torch.allclose(shifted_output, output, atol=1e-5, rtol=1e-5)
        update_check(current_index, translation_ok, 'centering makes logits invariant to global translation')
        current_index += 1
        invalid_ok = False
        try:
            model(coordinates[:, :2])
        except (ValueError, RuntimeError):
            invalid_ok = True
        update_check(current_index, invalid_ok, 'coordinate shape validation rejects malformed input')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['evaluation_interface:independent_cli', 'evaluation_interface:eval_no_grad', 'evaluation_interface:pdb_parser', 'evaluation_interface:cli_smoke', 'evaluation_interface:cli_deterministic', 'evaluation_interface:raw_checkpoint_cli', 'evaluation_interface:invalid_temperature_rejected']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        submission_path = (CANDIDATE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        candidate = load_module_from_path('proteinmpnn_submission', submission_path)
        reference = load_module_from_path('proteinmpnn_reference', reference_path)
        reference_model = reference.ProteinMPNN().eval()
        state_dict = reference_model.state_dict()
        model = candidate.ProteinMPNN()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        temp_manager = tempfile.TemporaryDirectory()
        work_dir = Path(temp_manager.name)
        checkpoint_path = work_dir / 'reference_checkpoint.pt'
        torch.save({'model_state_dict': state_dict}, checkpoint_path)
        pdb_path = work_dir / 'input.pdb'
        _write_pdb(pdb_path, _fixture_coordinates())
        source, tree = _parse_source(submission_path)
        flags_ok = all(flag in source for flag in REQUIRED_CLI_FLAGS)
        independent = not _imports_onescience(tree) and flags_ok
        update_check(current_index, independent, f'required CLI flags and no onescience import; flags_ok={flags_ok}')
        current_index += 1
        no_grad_ok = '.eval()' in source and ('inference_mode' in source or 'no_grad' in source)
        update_check(current_index, no_grad_ok, 'entry point switches to eval mode and uses an inference context')
        current_index += 1
        parser_ok = False
        parser_detail = 'PDB parser failed'
        try:
            parsed = candidate.parse_ca_coordinates(pdb_path)
            parser_ok = torch.is_tensor(parsed) and tuple(parsed.shape) == (5, 3)
            parser_detail = f'shape={tuple(parsed.shape) if torch.is_tensor(parsed) else type(parsed).__name__}'
        except Exception as error:
            parser_detail = f'{type(error).__name__}: {str(error)[:240]}'
        update_check(current_index, parser_ok, parser_detail)
        current_index += 1
        out_folder = work_dir / 'cli_smoke'
        completed = run_cli(submission_path, ['--pdb-path', str(pdb_path), '--checkpoint', str(checkpoint_path), '--out-folder', str(out_folder), '--num-seq-per-target', str(SAMPLE_SEQUENCES), '--temperature', '0.5', '--seed', '9', '--device', 'cpu'])
        fasta_path = out_folder / 'seqs' / 'designed.fasta'
        smoke_ok = False
        smoke_detail = f'returncode={completed.returncode}; stderr={completed.stderr.strip()[:240]}'
        if completed.returncode == 0 and fasta_path.is_file():
            records = _read_fasta(fasta_path)
            amino_acids = getattr(candidate, 'AMINO_ACIDS', None)
            if not isinstance(amino_acids, str):
                amino_acids = reference.AMINO_ACIDS
            smoke_ok = len(records) == SAMPLE_SEQUENCES and all(header.strip() != '' and len(sequence) == 5 and set(sequence) <= set(amino_acids) for header, sequence in records)
            smoke_detail = f'records={len(records)}, sequences={[sequence for _header, sequence in records]}'
        elif completed.returncode == 0:
            smoke_detail = 'out-folder/seqs/designed.fasta was not produced'
        update_check(current_index, smoke_ok, smoke_detail)
        current_index += 1
        deterministic_ok = False
        deterministic_detail = 'second CLI run failed'
        if smoke_ok:
            second_folder = work_dir / 'cli_smoke_second'
            second_run = run_cli(submission_path, ['--pdb-path', str(pdb_path), '--checkpoint', str(checkpoint_path), '--out-folder', str(second_folder), '--num-seq-per-target', str(SAMPLE_SEQUENCES), '--temperature', '0.5', '--seed', '9', '--device', 'cpu'])
            second_fasta = second_folder / 'seqs' / 'designed.fasta'
            if second_run.returncode == 0 and second_fasta.is_file():
                deterministic_ok = second_fasta.read_text(encoding='utf-8') == fasta_path.read_text(encoding='utf-8')
                deterministic_detail = f'rerun identical={deterministic_ok}'
            else:
                deterministic_detail = f'returncode={second_run.returncode}; stderr={second_run.stderr.strip()[:200]}'
        update_check(current_index, deterministic_ok, deterministic_detail)
        current_index += 1
        raw_folder = work_dir / 'cli_raw_checkpoint'
        raw_checkpoint = work_dir / 'raw_state.pt'
        torch.save(reference_model.state_dict(), raw_checkpoint)
        raw_run = run_cli(submission_path, ['--pdb-path', str(pdb_path), '--checkpoint', str(raw_checkpoint), '--out-folder', str(raw_folder), '--num-seq-per-target', '1', '--temperature', '0.5', '--seed', '9', '--device', 'cpu'])
        update_check(current_index, raw_run.returncode == 0 and (raw_folder / 'seqs' / 'designed.fasta').is_file(), f'returncode={raw_run.returncode}')
        current_index += 1
        invalid_run = run_cli(submission_path, ['--pdb-path', str(pdb_path), '--checkpoint', str(checkpoint_path), '--out-folder', str(work_dir / 'invalid_temp'), '--num-seq-per-target', '1', '--temperature', '0', '--seed', '9', '--device', 'cpu'])
        update_check(current_index, invalid_run.returncode != 0, f'returncode={invalid_run.returncode}')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    '''Checkpoint implementation.'''
    CHECKPOINT_NAMES = ['metric:forward_equivalence', 'metric:autoregressive_sequences', 'metric:seed_sensitivity', 'metric:sequence_token_range', 'metric:sampling_determinism']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        torch, _nn = _load_torch()
        torch.manual_seed(SEED)
        submission_path = (CANDIDATE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        candidate = load_module_from_path('proteinmpnn_submission', submission_path)
        reference = load_module_from_path('proteinmpnn_reference', reference_path)
        reference_model = reference.ProteinMPNN().eval()
        state_dict = reference_model.state_dict()
        model = candidate.ProteinMPNN()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        coordinates = torch.tensor(_fixture_coordinates())
        with torch.inference_mode():
            neighbors, edges = model.build_geometry_graph(coordinates)
            output = model(coordinates)
        metrics: dict[str, float] = {}
        with torch.inference_mode():
            expected = reference_model(coordinates)
        logits_mae = float((expected - output).abs().mean())
        logits_ok = torch.allclose(output, expected, atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
        metrics['logits_mae'] = logits_mae
        update_check(current_index, logits_ok, f'fixed-weight geometry logits match the standard implementation; mae={logits_mae:.3e}, atol={FORWARD_COMPARE_ATOL}')
        current_index += 1
        amino_acids = getattr(candidate, 'AMINO_ACIDS', None)
        if not isinstance(amino_acids, str):
            amino_acids = reference.AMINO_ACIDS
        with torch.inference_mode():
            sequences = model.sample(coordinates, num_sequences=SAMPLE_SEQUENCES, temperature=0.5, seed=9)
        valid = isinstance(sequences, list) and len(sequences) == SAMPLE_SEQUENCES and all(isinstance(item, str) and len(item) == len(coordinates) and set(item) <= set(amino_acids) for item in sequences)
        update_check(current_index, valid, str(sequences))
        current_index += 1
        seed_other = model.sample(coordinates, num_sequences=SAMPLE_SEQUENCES, temperature=0.5, seed=10)
        update_check(current_index, sequences != seed_other, 'different seeds change sampled sequences')
        current_index += 1
        update_check(current_index, all(set(item) <= set(amino_acids[:20]) for item in sequences), 'sampled FASTA sequences use standard amino-acid tokens only')
        current_index += 1
        with torch.inference_mode():
            repeated = model.sample(coordinates, num_sequences=SAMPLE_SEQUENCES, temperature=0.5, seed=9)
        determinism_ok = isinstance(sequences, list) and sequences == repeated
        update_check(current_index, determinism_ok, 'sampling twice with the same seed returns identical sequences')
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks
