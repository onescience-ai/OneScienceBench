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
    """Check the candidate files named by the case."""
    params = config.get('params', config)
    phase = params['phase']
    required_files = params.get('required_files', [])
    CHECKPOINT_NAMES = ['file:dnabert2_train.py', 'file:standard:train_config.json']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    try:
        if not required_files:
            raise ValueError('required_files is empty')
        path = (CANDIDATE_DIR / Path(required_files[0])).resolve()
        exists = path.is_file()
        checks[0] = (CHECKPOINT_NAMES[0], exists, f'{phase}: {path}')
        nonempty = exists and path.stat().st_size > 0
        checks[1] = (CHECKPOINT_NAMES[1], nonempty, f'{phase}: non-empty={nonempty}')
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, traceback.format_exc())
    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = ['data:sequence_encoding', 'data:invalid_sequence_rejected', 'data:batch_contract']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def normalize(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    try:
        candidate = params.get('candidate') or {}
        candidate_path = (CANDIDATE_DIR / Path(candidate['file'])).resolve()
        module = load_module('dnabert2_train_candidate_data', candidate_path)
        calls = candidate.get('calls') or []
        if len(calls) != len(CHECKPOINT_NAMES):
            raise ValueError('candidate.calls must define exactly one entry per data checkpoint')
        for index, call in enumerate(calls):
            expected = call.get('expected') or {}
            try:
                function = getattr(module, call['function'])
                if not callable(function):
                    raise TypeError('candidate target is not callable')
                result = function(**(call.get('args') or {}))
                if 'raises' in expected:
                    checks[index] = (CHECKPOINT_NAMES[index], False, 'expected exception was not raised')
                    continue
                outputs = list(result) if isinstance(result, (list, tuple)) else [result]
                descriptors = expected.get('outputs')
                if descriptors is None and any(key in expected for key in ('value', 'shape', 'dtype')):
                    descriptors = [expected]
                passed, messages = True, []
                for position, descriptor in enumerate(descriptors or []):
                    output = outputs[position] if position < len(outputs) else None
                    if 'shape' in descriptor and list(getattr(output, 'shape', [])) != list(descriptor['shape']):
                        passed = False
                        messages.append(f'output[{position}] shape={list(getattr(output, "shape", []))}')
                    if 'dtype' in descriptor and str(getattr(output, 'dtype', '')) != descriptor['dtype']:
                        passed = False
                        messages.append(f'output[{position}] dtype={getattr(output, "dtype", None)}')
                    if 'value' in descriptor and normalize(output) != descriptor['value']:
                        passed = False
                        messages.append(f'output[{position}] value={normalize(output)}')
                checks[index] = (CHECKPOINT_NAMES[index], passed, '; '.join(messages) or 'call output matched expected')
            except Exception as error:
                if expected.get('raises') and type(error).__name__ == expected['raises']:
                    checks[index] = (CHECKPOINT_NAMES[index], True, f"raised {expected['raises']}")
                else:
                    checks[index] = (CHECKPOINT_NAMES[index], False, traceback.format_exc())
    except Exception:
        checks[0] = (CHECKPOINT_NAMES[0], False, traceback.format_exc())
    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = [
        'train_script:independent_from_onescience',
        'train_script:cli_flags',
        'train_script:adamw_backward',
        'train_script:gradient_clipping',
        'train_script:structured_artifacts',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    try:
        files = params.get('files', [])
        rules = params.get('check_config') or {}
        sources, trees = [], []
        for relative in files:
            source = (CANDIDATE_DIR / Path(relative)).read_text(encoding='utf-8')
            sources.append(source)
            trees.append(ast.parse(source))
        source = '\n'.join(sources)
        imports = {
            alias.name.split('.')[0]
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module.split('.')[0]
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        names = {
            node.id for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        names.update(
            node.attr for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        )
        forbidden = set(rules.get('forbidden_imports') or [])
        checks[0] = (CHECKPOINT_NAMES[0], not (imports & forbidden), f'imported={sorted(imports)}')
        flags = rules.get('required_cli_flags') or []
        checks[1] = (CHECKPOINT_NAMES[1], all(flag in source for flag in flags), str(flags))
        symbols = rules.get('required_symbols') or []
        checks[2] = (CHECKPOINT_NAMES[2], all(symbol in names for symbol in symbols), str(symbols))
        clip_name = rules['gradient_clip_symbol']
        checks[3] = (CHECKPOINT_NAMES[3], clip_name in names, clip_name)
        artifacts = rules.get('required_artifacts') or []
        checks[4] = (CHECKPOINT_NAMES[4], all(item in source for item in artifacts), str(artifacts))
    except Exception:
        checks[0] = (CHECKPOINT_NAMES[0], False, traceback.format_exc())

    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = [
        'model_structure:nn_module',
        'model_structure:required_components',
        'model_structure:state_dict_contract',
        'model_structure:module_order',
        'model_structure:constructor_contract',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    try:
        import torch
        model_file = (CANDIDATE_DIR / Path(params['model_file'])).resolve()
        module = load_module('dnabert2_train_candidate_structure', model_file)
        model_class_name = params['model_class']
        model_class = getattr(module, model_class_name)
        model_args = params.get('model_args') or {}
        model = model_class(**model_args)
        rules = params.get('rules') or []
        rule_params = [rule.get('params') or {} for rule in rules]
        checks[0] = (CHECKPOINT_NAMES[0], isinstance(model, torch.nn.Module), type(model).__name__)
        required = rule_params[1].get('names', []) if len(rule_params) > 1 else []
        missing = [name for name in required if not any(path == name for path, _ in model.named_modules())]
        checks[1] = (CHECKPOINT_NAMES[1], not missing, f'missing={missing}')
        shapes = rule_params[2].get('modules', {}) if len(rule_params) > 2 else {}
        shape_ok, shape_messages = True, []
        for path, expected in shapes.items():
            target = dotted(model, path)
            for attribute, value in expected.items():
                actual = getattr(target, attribute, None)
                if actual != value:
                    shape_ok = False
                    shape_messages.append(f'{path}.{attribute}={actual}, expected={value}')
        checks[2] = (CHECKPOINT_NAMES[2], shape_ok, '; '.join(shape_messages) or 'parameter shapes match')
        expected_order = rule_params[3].get('names', []) if len(rule_params) > 3 else []
        actual_order = [name for name, _ in model.named_children()]
        checks[3] = (CHECKPOINT_NAMES[3], actual_order == expected_order, f'actual={actual_order}')
        defaults = rule_params[4].get('defaults', {}) if len(rule_params) > 4 else {}
        signature = inspect.signature(model_class.__init__)
        default_ok = all(
            name in signature.parameters and signature.parameters[name].default == value
            for name, value in defaults.items()
        )
        checks[4] = (CHECKPOINT_NAMES[4], default_ok, str(signature))
    except Exception:
        checks[0] = (CHECKPOINT_NAMES[0], False, traceback.format_exc())

    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = [
        'forward_executable:tensor',
        'forward_executable:shape',
        'forward_executable:finite',
        'forward_executable:deterministic',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    try:
        import torch
        model_file = (CANDIDATE_DIR / Path(params['model_file'])).resolve()
        module = load_module('dnabert2_train_candidate_forward', model_file)
        model_class = getattr(module, params['model_class'])
        model = model_class(**(params.get('model_args') or {}))
        model.eval()
        context = {'config': {}}
        for hook in params.get('pre_forward_hooks') or []:
            function = getattr(module, hook['function'])
            hook_args = hook.get('args') or {}
            result = function(**hook_args)
            if hook.get('save_as'):
                context[hook['save_as']] = result
        arg_types = params.get('forward_arg_types') or {}
        args = {}
        for name, value in (params.get('forward_args') or {}).items():
            args[name] = torch.tensor(value, dtype=getattr(torch, arg_types[name])) if name in arg_types else value
        signature = inspect.signature(model.forward)
        parameters = [parameter for parameter in signature.parameters.values() if parameter.name != 'self']
        parameter_names = [parameter.name for parameter in parameters]
        if all(name in parameter_names for name in args):
            kwargs, positional = args, []
        else:
            kwargs, positional = {}, list(args.values())
        with torch.inference_mode():
            output = model(*positional, **kwargs)
        tensor_ok = isinstance(output, torch.Tensor)
        checks[0] = (CHECKPOINT_NAMES[0], tensor_ok, str(type(output)))
        expected_shape = params.get('out_shape')
        shape_ok = tensor_ok and (expected_shape is None or list(output.shape) == list(expected_shape))
        checks[1] = (CHECKPOINT_NAMES[1], shape_ok, str(tuple(output.shape)) if tensor_ok else 'not a tensor')
        finite_ok = tensor_ok and bool(torch.isfinite(output).all())
        checks[2] = (CHECKPOINT_NAMES[2], finite_ok, 'finite logits' if finite_ok else 'non-finite logits')
        checks[3] = (CHECKPOINT_NAMES[3], True, 'repeatability is not an additional prompt contract')
    except Exception:
        checks[0] = (CHECKPOINT_NAMES[0], False, traceback.format_exc())

    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = ['forward_equivalence:forward_equivalence']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    try:
        reference = params.get('reference')
        candidate = params.get('candidate') or {}
        if not candidate or not reference:
            raise ValueError('candidate and reference model specs are required')
        atol = params.get('compare_atol', 1e-5)
        import torch
        init_seed = params.get('init_seed')
        arg_types = params.get('forward_arg_types') or {}
        def build(spec, prefix):
            module = load_module(prefix, (REFERENCE_DIR if spec is reference else CANDIDATE_DIR) / Path(spec['model_file']))
            if init_seed is not None:
                torch.manual_seed(init_seed)
            model = getattr(module, spec['model_class'])(**(spec.get('model_args') or {})).eval()
            args = {
                key: torch.tensor(value, dtype=getattr(torch, arg_types[key])) if key in arg_types else value
                for key, value in (spec.get('forward_args') or {}).items()
            }
            parameter_names = [
                parameter.name
                for parameter in inspect.signature(model.forward).parameters.values()
                if parameter.name != 'self'
            ]
            if all(name in parameter_names for name in args):
                return model, args
            return model, tuple(args.values())
        candidate_model, candidate_args = build(candidate, 'dnabert2_train_candidate_equivalence')
        reference_model, reference_args = build(reference, 'dnabert2_train_reference_equivalence')
        with torch.inference_mode():
            candidate_output = (
                candidate_model(**candidate_args) if isinstance(candidate_args, dict) else candidate_model(*candidate_args)
            )
            reference_output = (
                reference_model(**reference_args) if isinstance(reference_args, dict) else reference_model(*reference_args)
            )
        same = (
            isinstance(candidate_output, torch.Tensor)
            and isinstance(reference_output, torch.Tensor)
            and candidate_output.shape == reference_output.shape
            and torch.allclose(candidate_output, reference_output, atol=atol)
        )
        checks[0] = (CHECKPOINT_NAMES[0], bool(same), 'forward outputs compared')
    except Exception:
        checks[0] = (CHECKPOINT_NAMES[0], False, traceback.format_exc())

    return checks

def check_smoke_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = ['traning:short_fit', 'traning:parameter_update', 'traning:csv_fit']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    try:
        smoke = params.get('smoke_config') or {}
        mock = smoke['mock']
        csv_spec = smoke['csv']
        checkpoint_file = smoke['checkpoint_file']
        metrics_file = smoke['metrics_file']
        run_file = (CANDIDATE_DIR / Path(params.get('run_file', FILENAME))).resolve()
        manager = tempfile.TemporaryDirectory(prefix='dnabert2_train_')
        _ADAPTER_STATE.update(manager=manager)
        def run(spec, output, dataset_dir=None):
            command = [sys.executable, str(run_file)]
            mode = spec.get('mode')
            if mode:
                command.append(mode)
            if dataset_dir is not None:
                command.extend(['--dataset-dir', str(dataset_dir)])
            command.extend(['--output-dir', str(output)])
            for key in ('max_steps', 'batch_size', 'seed', 'device'):
                if key in spec:
                    command.extend([f'--{key.replace("_", "-")}', str(spec[key])])
            return subprocess.run(command, capture_output=True, text=True, timeout=180)
        output = Path(manager.name) / mock['output_subdir']
        process = run(mock, output)
        _ADAPTER_STATE.update(
            output=output,
            checkpoint=output / checkpoint_file,
            metrics=output / metrics_file,
        )
        checks[0] = (CHECKPOINT_NAMES[0], process.returncode == 0, f'returncode={process.returncode}')
        checks[1] = (
            CHECKPOINT_NAMES[1],
            _ADAPTER_STATE['checkpoint'].is_file() and _ADAPTER_STATE['checkpoint'].stat().st_size > 0,
            str(_ADAPTER_STATE['checkpoint']),
        )
        dataset_dir = Path(manager.name) / 'dataset'
        dataset_dir.mkdir()
        rows = ['sequence,label'] + [f'{sequence},{label}' for sequence, label in csv_spec['samples']]
        (dataset_dir / csv_spec['dataset_file']).write_text('\n'.join(rows) + '\n', encoding='utf-8')
        csv_output = Path(manager.name) / csv_spec['output_subdir']
        csv_process = run(csv_spec, csv_output, dataset_dir)
        checks[2] = (
            CHECKPOINT_NAMES[2],
            csv_process.returncode == 0
            and (csv_output / metrics_file).is_file()
            and (csv_output / checkpoint_file).is_file(),
            f'returncode={csv_process.returncode}',
        )
    except Exception:
        checks[0] = (CHECKPOINT_NAMES[0], False, traceback.format_exc())

    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = ['artifacts:checkpoint', 'artifacts:metrics']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    try:
        output = _ADAPTER_STATE.get('output')
        if output is None:
            raise RuntimeError('smoke training did not produce an output directory')
        checkpoint_path = output / params['checkpoint_file']
        metrics_path = output / params['metrics_file']
        payload = {}
        if checkpoint_path.is_file():
            import torch
            try:
                payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint_path, map_location='cpu')
        required_keys = set(params['required_checkpoint_keys'])
        checkpoint_ok = (
            isinstance(payload, dict)
            and required_keys.issubset(payload)
            and payload.get('step') == params['expected_step']
        )
        checks[0] = (CHECKPOINT_NAMES[0], checkpoint_ok, f'keys={sorted(payload)}')
        rows = (
            [
                json.loads(line)
                for line in metrics_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            if metrics_path.is_file()
            else []
        )
        loss_field = params['loss_field']
        metrics_ok = len(rows) == params['expected_metric_rows'] and all(
            row.get('step') == expected_step
            and math.isfinite(float(row.get(loss_field, math.nan)))
            for expected_step, row in zip(params['expected_step_values'], rows)
        )
        checks[1] = (CHECKPOINT_NAMES[1], metrics_ok, str(rows))
    except Exception:
        checks[0] = (CHECKPOINT_NAMES[0], False, traceback.format_exc())

    return checks
