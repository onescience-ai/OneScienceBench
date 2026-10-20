"""V2 adapter for the legacy Evo2 inference evaluator."""
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

SUBMISSION_FILENAME = 'evo2_inference.py'
SEED = 20260819
FORWARD_COMPARE_ATOL = 1e-6
CLI_TIMEOUT_SECONDS = 180
EXPECTED_LAYERS = 2
DEFAULT_HIDDEN = 32
EXPECTED_VOCAB_SIZE = 47
GENERATION_PROMPT = 'ACGT|'
GENERATION_TOKENS = 4
REQUIRED_BLOCK_ATTRS = (
    'norm',
    'input_projection',
    'depthwise_conv',
    'ssm_projection',
    'delta_projection',
    'A_log',
    'D',
    'output_projection',
)
REQUIRED_CLI_FLAGS = (
    '--prompt',
    '--ckpt-dir',
    '--max-new-tokens',
    '--temperature',
    '--top-k',
    '--top-p',
    '--seed',
    '--output-file',
    '--device',
)

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['file:evo2_inference.py', 'file:standard:inference_config.json']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        torch.manual_seed(SEED)
        candidate_filename = config.get('candidate_filename', SUBMISSION_FILENAME)
        path = (CANDIDATE_DIR / candidate_filename).resolve()
        path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path.relative_to(REFERENCE_DIR)
        spec = importlib.util.spec_from_file_location('evo2_inference_submission', str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {path}')
        module_dir = str(path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_submission'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('evo2_inference_reference', str(reference_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {reference_path}')
        module_dir = str(reference_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        reference = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_reference'] = reference
        spec.loader.exec_module(reference)
        temp_manager = tempfile.TemporaryDirectory()
        work_dir = Path(temp_manager.name)
        update_check(current_index, True, str(path))
        current_index += 1
        path = REFERENCE_DIR / 'inference_config.json'
        ok = path.is_file()
        detail = str(path)
        if ok and path.suffix == '.json':
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                ok = isinstance(payload, dict)
                if ok:
                    assets = payload.get('external_assets')
                    if not isinstance(assets, dict) or assets.get('mount_required') is not True:
                        ok = False
                        detail = 'inference_config.json must declare mount_required=true external_assets'
                    else:
                        missing = [
                            name
                            for name in ('weights', 'dataset')
                            if not isinstance(assets.get(name), str) or not assets[name].startswith('/public/share/')
                        ]
                        if missing:
                            ok = False
                            detail = f'external assets must use onestore paths for: {missing}'
                        else:
                            detail = 'complete weights and dataset are declared as externally mounted assets'
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

def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'checkpoint_loading:strict_state_dict',
        'checkpoint_loading:file_formats',
        'checkpoint_loading:uses_torch_load',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        torch.manual_seed(SEED)
        candidate_filename = config.get('candidate_filename', SUBMISSION_FILENAME)
        submission_path = (CANDIDATE_DIR / candidate_filename).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path.relative_to(REFERENCE_DIR)
        spec = importlib.util.spec_from_file_location('evo2_inference_submission', str(submission_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {submission_path}')
        module_dir = str(submission_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_submission'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('evo2_inference_reference', str(reference_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {reference_path}')
        module_dir = str(reference_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        reference = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_reference'] = reference
        spec.loader.exec_module(reference)
        reference_model = reference.Evo2InferenceModel().eval()
        state_dict = reference_model.state_dict()
        model = candidate.Evo2InferenceModel()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        update_check(current_index, True, 'reference state_dict loaded with strict=True')
        current_index += 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_dir = Path(temporary_directory)
            raw_path = work_dir / 'reference_raw.pt'
            wrapped_path = work_dir / 'reference_wrapped.pt'
            torch.save(state_dict, raw_path)
            torch.save({'model_state_dict': state_dict}, wrapped_path)
            loader = getattr(candidate, 'load_checkpoint', None)
            formats_ok = True
            details: list[str] = []
            if callable(loader):
                for label, path in (('raw', raw_path), ('wrapped', wrapped_path)):
                    probe = candidate.Evo2InferenceModel()
                    try:
                        loader(probe, path)
                        weight_ok = all(
                            torch.equal(probe_state, reference_state)
                            for probe_state, reference_state in zip(probe.state_dict().values(), state_dict.values())
                        )
                    except Exception as error:
                        weight_ok = False
                        details.append(f'{label}:{type(error).__name__}: {error}')
                    if not weight_ok:
                        formats_ok = False
                        details.append(f'{label}: weights mismatch after load')
                if formats_ok:
                    details = ['load_checkpoint restores both raw state_dict and model_state_dict envelopes']
            else:
                for label, path in (('raw', raw_path), ('wrapped', wrapped_path)):
                    output_file = work_dir / f'cli_load_{label}.json'
                    try:
                        command = [
                            sys.executable, str(submission_path), '--prompt', GENERATION_PROMPT,
                            '--ckpt-dir', str(path), '--max-new-tokens', str(GENERATION_TOKENS),
                            '--temperature', '1.0', '--seed', '7', '--output-file', str(output_file),
                            '--device', 'cpu',
                        ]
                        completed = subprocess.run(
                            command, cwd=str(submission_path.parent), capture_output=True,
                            text=True, timeout=CLI_TIMEOUT_SECONDS, check=False,
                        )
                        if completed.returncode != 0:
                            raise RuntimeError(
                                f'inference CLI failed with returncode={completed.returncode}: '
                                f'{completed.stderr.strip()[:300]}'
                            )
                        payload = json.loads(output_file.read_text(encoding='utf-8'))
                        exit_ok = set(payload) >= {'prompt', 'generated_text'}
                    except Exception as error:
                        exit_ok = False
                        details.append(f'{label}:{type(error).__name__}: {str(error)[:160]}')
                    if not exit_ok:
                        formats_ok = False
                        details.append(f'{label}: CLI could not load the checkpoint file')
                if formats_ok:
                    details = ['CLI loads both raw state_dict and model_state_dict checkpoint files']
            update_check(
                current_index,
                formats_ok,
                '; '.join(details) if details else 'checkpoint file loading verified',
            )
            current_index += 1
        source = submission_path.read_text(encoding='utf-8')
        _tree = ast.parse(source, filename=str(submission_path))
        update_check(current_index, 'torch.load' in source, 'checkpoint loading uses torch.load')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'model_structure:mamba_stack',
        'model_structure:selective_ssm_components',
        'model_structure:block_projection_shapes',
        'model_structure:module_order',
        'model_structure:no_rnn_substitute',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn
        torch.manual_seed(SEED)
        candidate_filename = config.get('candidate_filename', SUBMISSION_FILENAME)
        submission_path = (CANDIDATE_DIR / candidate_filename).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path.relative_to(REFERENCE_DIR)
        spec = importlib.util.spec_from_file_location('evo2_inference_submission', str(submission_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {submission_path}')
        module_dir = str(submission_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_submission'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('evo2_inference_reference', str(reference_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {reference_path}')
        module_dir = str(reference_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        reference = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_reference'] = reference
        spec.loader.exec_module(reference)
        reference_model = reference.Evo2InferenceModel().eval()
        state_dict = reference_model.state_dict()
        model = candidate.Evo2InferenceModel()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        layer_cls = getattr(candidate, 'Evo2MambaBlock', None)
        stack_ok = (
            isinstance(model.token_embedding, nn.Embedding)
            and model.token_embedding.num_embeddings == EXPECTED_VOCAB_SIZE
            and model.token_embedding.embedding_dim == DEFAULT_HIDDEN
            and isinstance(model.mamba_layers, nn.ModuleList)
            and len(model.mamba_layers) == EXPECTED_LAYERS
            and inspect.isclass(layer_cls)
            and all(isinstance(layer, layer_cls) for layer in model.mamba_layers)
            and isinstance(model.final_norm, nn.LayerNorm)
            and model.final_norm.normalized_shape == (DEFAULT_HIDDEN,)
            and isinstance(model.lm_head, nn.Linear)
            and model.lm_head.in_features == DEFAULT_HIDDEN
            and model.lm_head.out_features == EXPECTED_VOCAB_SIZE
        )
        update_check(current_index, stack_ok, str(model))
        current_index += 1
        layer = model.mamba_layers[0]
        components_ok = (
            all(hasattr(layer, name) for name in REQUIRED_BLOCK_ATTRS)
            and isinstance(layer.depthwise_conv, nn.Conv1d)
            and layer.depthwise_conv.groups == layer.depthwise_conv.in_channels == DEFAULT_HIDDEN
        )
        update_check(
            current_index,
            components_ok,
            f'required={REQUIRED_BLOCK_ATTRS}; depthwise conv groups={getattr(layer.depthwise_conv, "groups", None)}',
        )
        current_index += 1
        probes_ok = True
        probe_details = []
        with torch.no_grad():
            for name, raw_input, expected_out in (
                ('input_projection', torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN * 2)),
                ('depthwise_conv', torch.randn(2, DEFAULT_HIDDEN, 7), (2, DEFAULT_HIDDEN, 4)),
                ('ssm_projection', torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN * 3)),
                ('delta_projection', torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN)),
                ('output_projection', torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN)),
                ('norm', torch.randn(2, 5, DEFAULT_HIDDEN), (2, 5, DEFAULT_HIDDEN)),
            ):
                module = getattr(layer, name, None)
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
        if not (hasattr(layer, 'A_log') and tuple(layer.A_log.shape) == (DEFAULT_HIDDEN,)):
            probes_ok = False
            probe_details.append('A_log shape mismatch')
        if not (hasattr(layer, 'D') and tuple(layer.D.shape) == (DEFAULT_HIDDEN,)):
            probes_ok = False
            probe_details.append('D shape mismatch')
        update_check(
            current_index,
            probes_ok,
            '; '.join(probe_details) if probe_details else 'gated input, 3-way ssm projection, delta and output projections match the contract',
        )
        current_index += 1
        expected_order = ['token_embedding', 'mamba_layers', 'final_norm', 'lm_head']
        actual_order = [name for name, _child in model.named_children()]
        order_ok = actual_order[:len(expected_order)] == expected_order
        update_check(current_index, order_ok, f'expected prefix={expected_order}, actual={actual_order}')
        current_index += 1
        rnn_ok = not any(isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)) for module in model.modules())
        update_check(
            current_index,
            rnn_ok,
            'the causal stack must use selective state-space blocks, not GRU/LSTM/RNN',
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'forward_executable:logit_shape',
        'forward_executable:vocab_consistency',
        'forward_executable:finite',
        'forward_executable:causal_prefix',
        'forward_executable:batch_consistency',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        torch.manual_seed(SEED)
        candidate_filename = config.get('candidate_filename', SUBMISSION_FILENAME)
        submission_path = (CANDIDATE_DIR / candidate_filename).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path.relative_to(REFERENCE_DIR)
        spec = importlib.util.spec_from_file_location('evo2_inference_submission', str(submission_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {submission_path}')
        module_dir = str(submission_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_submission'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('evo2_inference_reference', str(reference_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {reference_path}')
        module_dir = str(reference_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        reference = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_reference'] = reference
        spec.loader.exec_module(reference)
        reference_model = reference.Evo2InferenceModel().eval()
        state_dict = reference_model.state_dict()
        model = candidate.Evo2InferenceModel()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        with torch.inference_mode():
            output = model(input_ids)
        shape_ok = isinstance(output, torch.Tensor) and tuple(output.shape) == (1, 5, EXPECTED_VOCAB_SIZE)
        update_check(
            current_index,
            shape_ok,
            f'shape={tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__}',
        )
        current_index += 1
        vocab_ok = model.token_embedding.num_embeddings == model.lm_head.out_features == EXPECTED_VOCAB_SIZE
        update_check(
            current_index,
            vocab_ok,
            f'embedding={model.token_embedding.num_embeddings}, lm_head={model.lm_head.out_features}',
        )
        current_index += 1
        finite_ok = isinstance(output, torch.Tensor) and bool(torch.isfinite(output).all())
        update_check(current_index, finite_ok, 'all logits finite')
        current_index += 1
        with torch.inference_mode():
            prefix_a = model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))[:, :3]
            prefix_b = model(torch.tensor([[0, 1, 2, 4]], dtype=torch.long))[:, :3]
        causal_ok = torch.allclose(prefix_a, prefix_b, atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
        update_check(current_index, causal_ok, 'changing a future token leaves prefix logits unchanged')
        current_index += 1
        batch = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
        with torch.inference_mode():
            batched = model(batch)
        batch_ok = (
            tuple(batched.shape) == (2, 4, EXPECTED_VOCAB_SIZE)
            and torch.allclose(batched[0], batched[1], atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL)
        )
        update_check(current_index, batch_ok, 'identical sequences in one batch produce identical logits')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_infer_execution(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'evaluation_interface:independent_cli',
        'evaluation_interface:generation_and_scoring',
        'evaluation_interface:eval_no_grad',
        'evaluation_interface:cli_smoke',
        'evaluation_interface:cli_deterministic',
        'evaluation_interface:invalid_generation_args_rejected',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        torch.manual_seed(SEED)
        candidate_filename = config.get('candidate_filename', SUBMISSION_FILENAME)
        submission_path = (CANDIDATE_DIR / candidate_filename).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path.relative_to(REFERENCE_DIR)
        spec = importlib.util.spec_from_file_location('evo2_inference_submission', str(submission_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {submission_path}')
        module_dir = str(submission_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_submission'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('evo2_inference_reference', str(reference_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {reference_path}')
        module_dir = str(reference_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        reference = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_reference'] = reference
        spec.loader.exec_module(reference)
        reference_model = reference.Evo2InferenceModel().eval()
        state_dict = reference_model.state_dict()
        model = candidate.Evo2InferenceModel()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        source = submission_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(submission_path))
        flags_ok = all(flag in source for flag in REQUIRED_CLI_FLAGS)
        imports_onescience = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name.split('.')[0] == 'onescience' for alias in node.names):
                    imports_onescience = True
                    break
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] == 'onescience':
                    imports_onescience = True
                    break
        independent = not imports_onescience and flags_ok
        update_check(
            current_index,
            independent,
            f'required CLI flags and no onescience import; flags_ok={flags_ok}',
        )
        current_index += 1
        generation_ok = callable(getattr(model, 'generate', None)) and callable(getattr(model, 'score_sequence', None))
        update_check(current_index, generation_ok, 'generate and score_sequence are callable')
        current_index += 1
        no_grad_ok = '.eval()' in source and ('inference_mode' in source or 'no_grad' in source)
        update_check(current_index, no_grad_ok, 'entry point switches to eval mode and uses an inference context')
        current_index += 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_dir = Path(temporary_directory)
            wrapped_path = work_dir / 'reference_wrapped.pt'
            output_file = work_dir / 'cli_generation.json'
            torch.save({'model_state_dict': state_dict}, wrapped_path)
            smoke_ok = False
            smoke_detail = 'CLI run failed'
            try:
                command = [
                    sys.executable, str(submission_path), '--prompt', GENERATION_PROMPT,
                    '--ckpt-dir', str(wrapped_path), '--max-new-tokens', str(GENERATION_TOKENS),
                    '--temperature', '1.0', '--seed', '7', '--output-file', str(output_file),
                    '--device', 'cpu',
                ]
                completed = subprocess.run(
                    command, cwd=str(submission_path.parent), capture_output=True,
                    text=True, timeout=CLI_TIMEOUT_SECONDS, check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f'inference CLI failed with returncode={completed.returncode}: '
                        f'{completed.stderr.strip()[:300]}'
                    )
                payload = json.loads(output_file.read_text(encoding='utf-8'))
                generated = payload.get('generated_text')
                score = payload.get('mean_log_probability')
                smoke_ok = (
                    payload.get('prompt') == GENERATION_PROMPT
                    and isinstance(generated, str)
                    and generated.startswith(GENERATION_PROMPT)
                    and len(generated) == len(GENERATION_PROMPT) + GENERATION_TOKENS
                    and isinstance(score, float)
                    and math.isfinite(score)
                )
                smoke_detail = f'generated={generated!r}, score={score}'
            except Exception as error:
                smoke_detail = f'{type(error).__name__}: {str(error)[:240]}'
            update_check(current_index, smoke_ok, smoke_detail)
            current_index += 1
            deterministic_ok = False
            deterministic_detail = 'second CLI run failed'
            if smoke_ok:
                try:
                    second_output = work_dir / 'cli_generation_second.json'
                    command = [
                        sys.executable, str(submission_path), '--prompt', GENERATION_PROMPT,
                        '--ckpt-dir', str(wrapped_path), '--max-new-tokens', str(GENERATION_TOKENS),
                        '--temperature', '1.0', '--seed', '7', '--output-file', str(second_output),
                        '--device', 'cpu',
                    ]
                    completed = subprocess.run(
                        command, cwd=str(submission_path.parent), capture_output=True,
                        text=True, timeout=CLI_TIMEOUT_SECONDS, check=False,
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f'inference CLI failed with returncode={completed.returncode}: '
                            f'{completed.stderr.strip()[:300]}'
                        )
                    second = json.loads(second_output.read_text(encoding='utf-8'))
                    deterministic_ok = second == payload
                    deterministic_detail = f'rerun identical={deterministic_ok}'
                except Exception as error:
                    deterministic_detail = f'{type(error).__name__}: {str(error)[:240]}'
            update_check(current_index, deterministic_ok, deterministic_detail)
            current_index += 1
            command = [
                sys.executable, str(submission_path), '--prompt', GENERATION_PROMPT,
                '--ckpt-dir', str(wrapped_path), '--max-new-tokens', '-1', '--temperature', '1.0',
                '--seed', '7', '--output-file', str(work_dir / 'invalid.json'), '--device', 'cpu',
            ]
            invalid_run = subprocess.run(
                command, cwd=str(submission_path.parent), capture_output=True,
                text=True, timeout=CLI_TIMEOUT_SECONDS, check=False,
            )
            update_check(
                current_index,
                invalid_run.returncode != 0,
                f'returncode={invalid_run.returncode}',
            )
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        'metric:forward_equivalence',
        'metric:score_sequence_equivalence',
        'metric:score_finiteness',
        'metric:generation_oracle',
        'metric:seed_sensitivity',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        torch.manual_seed(SEED)
        candidate_filename = config.get('candidate_filename', SUBMISSION_FILENAME)
        submission_path = (CANDIDATE_DIR / candidate_filename).resolve()
        submission_path.relative_to(CANDIDATE_DIR)
        reference_path = (REFERENCE_DIR / SUBMISSION_FILENAME).resolve()
        reference_path.relative_to(REFERENCE_DIR)
        spec = importlib.util.spec_from_file_location('evo2_inference_submission', str(submission_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {submission_path}')
        module_dir = str(submission_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_submission'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('evo2_inference_reference', str(reference_path))
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not create import spec for {reference_path}')
        module_dir = str(reference_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        reference = importlib.util.module_from_spec(spec)
        sys.modules['evo2_inference_reference'] = reference
        spec.loader.exec_module(reference)
        reference_model = reference.Evo2InferenceModel().eval()
        state_dict = reference_model.state_dict()
        model = candidate.Evo2InferenceModel()
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        metrics: dict[str, float] = {}
        input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        with torch.inference_mode():
            expected = reference_model(input_ids)
            generated_output = model(input_ids)
        logits_mae = float((expected - generated_output).abs().mean())
        logits_ok = torch.allclose(
            generated_output, expected, atol=FORWARD_COMPARE_ATOL, rtol=FORWARD_COMPARE_ATOL
        )
        metrics['logits_mae'] = logits_mae
        update_check(
            current_index,
            logits_ok,
            f'fixed-weight logits match the standard implementation; mae={logits_mae:.3e}, atol={FORWARD_COMPARE_ATOL}',
        )
        current_index += 1
        sequence = 'ACGTACGTAC'
        score_ok = False
        score_detail = 'score_sequence failed'
        score_mae = float('inf')
        try:
            with torch.inference_mode():
                expected_score = reference_model.score_sequence(sequence)
                candidate_score = model.score_sequence(sequence)
            score_mae = abs(float(expected_score) - float(candidate_score))
            score_ok = score_mae <= FORWARD_COMPARE_ATOL
            score_detail = f'expected={expected_score:.6f}, candidate={candidate_score:.6f}, mae={score_mae:.3e}'
        except Exception as error:
            score_detail = f'{type(error).__name__}: {str(error)[:240]}'
        metrics['log_probability_mae'] = score_mae if math.isfinite(score_mae) else float('nan')
        update_check(current_index, score_ok, score_detail)
        current_index += 1
        update_check(
            current_index,
            math.isfinite(float(candidate_score)) if 'candidate_score' in locals() else False,
            'sequence score is finite',
        )
        current_index += 1
        generation_ok = False
        generation_detail = 'generate failed'
        try:
            with torch.inference_mode():
                generated = model.generate(
                    GENERATION_PROMPT,
                    max_new_tokens=GENERATION_TOKENS,
                    temperature=1.0,
                    seed=7,
                )
                score = model.score_sequence(generated)
            generation_ok = (
                isinstance(generated, str)
                and len(generated) == len(GENERATION_PROMPT) + GENERATION_TOKENS
                and generated.startswith(GENERATION_PROMPT)
                and math.isfinite(float(score))
            )
            generation_detail = f'generated={generated!r}, score={score:.6f}'
        except Exception as error:
            generation_detail = f'{type(error).__name__}: {str(error)[:240]}'
        update_check(current_index, generation_ok, generation_detail)
        current_index += 1
        try:
            with torch.inference_mode():
                other_generation = model.generate(
                    GENERATION_PROMPT,
                    max_new_tokens=GENERATION_TOKENS,
                    temperature=1.0,
                    seed=8,
                )
            seed_sensitive = other_generation != generated
        except Exception:
            seed_sensitive = False
        update_check(current_index, seed_sensitive, 'different seeds can change sampled generation')
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
