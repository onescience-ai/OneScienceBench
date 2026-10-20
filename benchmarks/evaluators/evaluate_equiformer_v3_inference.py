'''V2 adapter for EquiformerV3 fixed-checkpoint inference checks.'''
from pathlib import Path
import sys
import ast
import hashlib
import importlib.util
import json
import os
import traceback
from typing import Dict

import torch

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
    params = config.get('params', config)
    CHECKPOINT_NAMES = [
        'required_files_present',
        'prohibited_active_imports_absent',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        phase = params['phase']
        required_files = tuple(params['required_files'])
        prohibited_imports = set(params['prohibited_import_roots'])
        submission = CANDIDATE_DIR
        required_paths = []
        for file_name in required_files:
            direct_path = submission / file_name
            matches = [path for path in submission.rglob(file_name) if path.is_file()]
            required_paths.append(direct_path if direct_path.is_file() else (matches[0] if matches else direct_path))
        missing = [
            name
            for name, path in zip(required_files, required_paths)
            if not path.is_file()
        ]
        prohibited = []
        if not missing:
            for path in submission.rglob('*.py'):
                tree = ast.parse(path.read_text(encoding='utf-8', errors='ignore'), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        prohibited.extend(
                            alias.name
                            for alias in node.names
                            if alias.name.split('.')[0] in prohibited_imports
                        )
                    elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in prohibited_imports:
                        prohibited.append(node.module)
        detail = json.dumps(
            {'phase': phase, 'missing': missing, 'prohibited_active_imports': prohibited},
            sort_keys=True,
        )
        update_check(current_index, not missing, detail)
        current_index += 1
        update_check(current_index, not prohibited, detail)
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = [
        'prohibited_active_imports_absent',
        'strict_state_dict_loading',
        'checkpoint_sha256_matches',
        'state_entry_count_matches',
        'state_tensor_numel_matches',
        'restored_state_keys_match',
        'restored_state_values_match',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        model_path = Path(params['checkpoint_file'])
        model_file = params['model_file']
        strict_required = params['strict']
        expected_state_count = params['state_entry_count']
        expected_state_numel = params['state_tensor_numel']
        prohibited_imports = set(params['prohibited_import_roots'])
        submission = CANDIDATE_DIR
        name = 'equiformer_v3_inference_submission'
        direct_module_path = Path(model_file)
        matches = [candidate for candidate in submission.rglob(model_file) if candidate.is_file()]
        path = ((submission / direct_module_path) if (submission / direct_module_path).is_file()
                else matches[0]).resolve()
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(str(path))
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        for variable in ('ONESCIENCE_UMA_JD_PATH', 'ONESCIENCE_EQUIFORMER_V3_JD_PATH'):
            os.environ.setdefault(variable, params['wigner_path'])
        loader = getattr(module, params['model_loader'])
        loaded_result = loader(str(model_path), params['device'])
        pending = [loaded_result]
        seen = set()
        model = None
        while pending and model is None:
            value = pending.pop()
            marker = id(value)
            if marker in seen:
                continue
            seen.add(marker)
            if isinstance(value, torch.nn.Module):
                model = value
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple, set)):
                pending.extend(value)
            elif hasattr(value, '__dict__'):
                pending.extend(vars(value).values())
        if model is None:
            raise TypeError('model loader result does not contain a torch module')
        payload = getattr(module, params['payload_function'])(str(model_path))
        state = getattr(module, params['state_function'])(payload)
        if not hasattr(state, 'items'):
            raise TypeError('raw model state must be a mapping')
        prohibited = []
        strict_load = False
        for source_path in submission.rglob('*.py'):
            tree = ast.parse(source_path.read_text(encoding='utf-8', errors='ignore'), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    prohibited.extend(
                        alias.name
                        for alias in node.names
                        if alias.name.split('.')[0] in prohibited_imports
                    )
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in prohibited_imports:
                    prohibited.append(node.module)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                    if (
                        node.func.attr == params['state_load_function']
                        and isinstance(keywords.get('strict'), ast.Constant)
                        and keywords['strict'].value is strict_required
                    ):
                        strict_load = True
        digest = hashlib.sha256()
        with Path(model_path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        model_hash = digest.hexdigest()
        state_keys = list(state)
        state_numel = sum(v.numel() for v in state.values() if torch.is_tensor(v))
        expected_state = {
            str(key).removeprefix('_orig_mod.').removeprefix('module.'): value
            for key, value in state.items()
        }
        loaded_state = {str(key).removeprefix('_orig_mod.').removeprefix('module.'): value
                        for key, value in model.state_dict().items()}
        same_keys = set(loaded_state) == set(expected_state)
        same_values = same_keys and all(
            torch.is_tensor(loaded_state[key])
            and torch.is_tensor(expected_state[key])
            and torch.equal(loaded_state[key].detach().cpu(), expected_state[key].detach().cpu())
            for key in loaded_state
        )
        detail = json.dumps(
            {
                'model_sha256': model_hash,
                'strict_load': strict_load,
                'state_count': len(state_keys),
                'state_numel': state_numel,
                'same_keys': same_keys,
                'same_values': same_values,
            },
            sort_keys=True,
        )
        update_check(current_index, not prohibited, detail)
        current_index += 1
        update_check(current_index, strict_load, detail)
        current_index += 1
        update_check(
            current_index,
            model_hash == '429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98',
            detail,
        )
        current_index += 1
        update_check(current_index, len(state_keys) == expected_state_count, detail)
        current_index += 1
        update_check(current_index, state_numel == expected_state_numel, detail)
        current_index += 1
        update_check(current_index, same_keys, detail)
        current_index += 1
        update_check(current_index, same_values, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = [
        'model_class_matches',
        'top_level_module_order_matches',
        'parameter_tensor_count_matches',
        'max_neighbors_matches',
        'cutoff_matches',
        'num_layers_matches',
        'num_channels_matches',
        'lmax_matches',
        'regress_forces_enabled',
        'regress_stress_enabled',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        model_path = Path(params['checkpoint_file'])
        model_file = params['model_file']
        expected_model_class = params['model_class']
        casts = {
            'int': int,
            'float': float,
            'bool': bool,
        }
        rules = {rule['id']: rule for rule in params['rules']}
        name = 'equiformer_v3_inference_submission'
        direct_module_path = Path(model_file)
        matches = [candidate for candidate in CANDIDATE_DIR.rglob(model_file) if candidate.is_file()]
        path = ((CANDIDATE_DIR / direct_module_path) if (CANDIDATE_DIR / direct_module_path).is_file()
                else matches[0]).resolve()
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(str(path))
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        for variable in ('ONESCIENCE_UMA_JD_PATH', 'ONESCIENCE_EQUIFORMER_V3_JD_PATH'):
            os.environ.setdefault(variable, params['wigner_path'])
        loader = getattr(module, params['model_loader'])
        loaded_result = loader(str(model_path), params['device'])
        pending = [loaded_result]
        seen = set()
        model = None
        while pending and model is None:
            value = pending.pop()
            marker = id(value)
            if marker in seen:
                continue
            seen.add(marker)
            if isinstance(value, torch.nn.Module):
                model = value
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple, set)):
                pending.extend(value)
            elif hasattr(value, '__dict__'):
                pending.extend(vars(value).values())
        if model is None:
            raise TypeError('model loader result does not contain a torch module')
        detail = json.dumps({'model': type(model).__name__}, sort_keys=True)
        update_check(
            current_index,
            type(model).__name__ == expected_model_class,
            detail,
        )
        for index, checkpoint_name in enumerate(CHECKPOINT_NAMES[1:], start=1):
            current_index = index
            try:
                rule = rules[checkpoint_name]
                rule_type = rule['type']
                expected = rule['expected']
                if rule_type == 'attribute':
                    actual = getattr(model, rule['attribute'])
                    cast_name = rule.get('cast')
                    if cast_name is not None:
                        actual = casts[cast_name](actual)
                    passed = (
                        abs(actual - expected) < rule['absolute_tolerance_exclusive']
                        if 'absolute_tolerance_exclusive' in rule
                        else actual == expected
                    )
                elif rule_type == 'module_order':
                    actual = list(model._modules)
                    passed = actual == expected
                elif rule_type == 'parameter_tensor_count':
                    actual = len(list(model.parameters()))
                    passed = actual == expected
                else:
                    raise ValueError(f'Unsupported structure rule type: {rule_type}')
                update_check(index, passed, f'actual={actual!r}, expected={expected!r}')
            except Exception as error:
                update_check(index, False, str(error))
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = [
        'required_public_apis_present',
        'evaluate_structure_signature_matches',
        'required_output_keys_present',
        'energy_shape_matches',
        'forces_shape_matches',
        'stress_shape_matches',
        'energy_is_finite',
        'forces_are_finite',
        'stress_is_finite',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        files = params['files']
        phase = params['phase']
        required_interfaces = params['required_interfaces']
        checkpoint_path = Path(params['checkpoint_file'])
        fixture = params['fixture']
        evaluation = params['evaluation']
        output_contract = params['output_contract']
        require_finite = output_contract['finite']
        fixture_factory = fixture['factory']
        name = 'equiformer_v3_inference_submission'
        model_file = files[0]
        direct_module_path = Path(model_file)
        matches = [candidate for candidate in CANDIDATE_DIR.rglob(model_file) if candidate.is_file()]
        path = ((CANDIDATE_DIR / direct_module_path) if (CANDIDATE_DIR / direct_module_path).is_file()
                else matches[0]).resolve()
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(str(path))
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        device = evaluation['device']
        for variable in ('ONESCIENCE_UMA_JD_PATH', 'ONESCIENCE_EQUIFORMER_V3_JD_PATH'):
            os.environ.setdefault(variable, params['wigner_path'])
        loader = getattr(module, params['model_loader'])
        loaded_result = loader(str(checkpoint_path), device)
        pending = [loaded_result]
        seen = set()
        model = None
        while pending and model is None:
            value = pending.pop()
            marker = id(value)
            if marker in seen:
                continue
            seen.add(marker)
            if isinstance(value, torch.nn.Module):
                model = value
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple, set)):
                pending.extend(value)
            elif hasattr(value, '__dict__'):
                pending.extend(vars(value).values())
        if model is None:
            raise TypeError('model loader result does not contain a torch module')
        factory_module, factory_name = fixture_factory.rsplit('.', 1)
        factory = getattr(__import__(factory_module, fromlist=[factory_name]), factory_name)
        atoms = factory(*fixture['args'], **fixture['kwargs'])
        atoms = atoms.repeat(tuple(fixture['repeat']))
        atoms.rattle(**fixture['rattle'])
        missing = [
            interface_name
            for interface_name in required_interfaces
            if not callable(getattr(module, interface_name, None))
        ]
        detail = json.dumps(
            {'phase': phase, 'fixture_factory': fixture_factory, 'missing_apis': missing},
            sort_keys=True,
        )
        update_check(current_index, not missing, detail)
        current_index = 1
        evaluate_function = getattr(module, evaluation['function'])
        sig = inspect.signature(evaluate_function)
        signature_params = set(sig.parameters)
        argument_values = {'context.model': model, 'fixture': atoms, 'cpu': device}
        call_arguments = evaluation['arguments']
        resolved_arguments = [argument_values.get(argument, argument) for argument in call_arguments]
        out = evaluate_function(*resolved_arguments)
        keys = set(out)
        detail = json.dumps(
            {
                'missing_apis': missing,
                'output_keys': sorted(keys),
                'signature': sorted(signature_params),
            },
            sort_keys=True,
        )
        required_parameters = set(params['signature_parameters'])
        update_check(current_index, required_parameters <= signature_params, detail)
        current_index = 2
        required_keys = output_contract['required_keys']
        if not set(required_keys) <= keys:
            update_check(current_index, False, detail)
        else:
            update_check(current_index, True, detail)
            current_index = 3
            shapes = output_contract['shapes']
            for index, output_name in enumerate(required_keys, start=3):
                current_index = index
                update_check(index, list(out[output_name].shape) == shapes[output_name], detail)
            current_index = 6
            for index, output_name in enumerate(required_keys, start=6):
                current_index = index
                finite = (not require_finite) or bool(torch.isfinite(out[output_name]).all())
                update_check(index, finite, detail)
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = [
        'energy_is_finite',
        'forces_are_finite',
        'stress_is_finite',
        'energy_shape_matches',
        'forces_shape_matches',
        'stress_shape_matches',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        model_file = params['model_file']
        checkpoint_path = Path(params['checkpoint_file'])
        forward_args = params['forward_args']
        fixture = params['fixture']
        output_shapes = params['out_shape']
        require_finite = params['finite_outputs']
        fixture_factory = fixture['factory']
        name = 'equiformer_v3_inference_submission'
        direct_module_path = Path(model_file)
        matches = [candidate for candidate in CANDIDATE_DIR.rglob(model_file) if candidate.is_file()]
        path = ((CANDIDATE_DIR / direct_module_path) if (CANDIDATE_DIR / direct_module_path).is_file()
                else matches[0]).resolve()
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(str(path))
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        device = forward_args['device']
        for variable in ('ONESCIENCE_UMA_JD_PATH', 'ONESCIENCE_EQUIFORMER_V3_JD_PATH'):
            os.environ.setdefault(variable, params['wigner_path'])
        loader = getattr(module, params['model_loader'])
        loaded_result = loader(str(checkpoint_path), device)
        pending = [loaded_result]
        seen = set()
        model = None
        while pending and model is None:
            value = pending.pop()
            marker = id(value)
            if marker in seen:
                continue
            seen.add(marker)
            if isinstance(value, torch.nn.Module):
                model = value
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple, set)):
                pending.extend(value)
            elif hasattr(value, '__dict__'):
                pending.extend(vars(value).values())
        if model is None:
            raise TypeError('model loader result does not contain a torch module')
        factory_module, factory_name = fixture_factory.rsplit('.', 1)
        factory = getattr(__import__(factory_module, fromlist=[factory_name]), factory_name)
        atoms = factory(*fixture['args'], **fixture['kwargs'])
        atoms = atoms.repeat(tuple(fixture['repeat']))
        atoms.rattle(**fixture['rattle'])
        argument_values = {'context.model': model, 'fixture': atoms, 'cpu': device}
        call_arguments = forward_args['arguments']
        resolved_arguments = [argument_values.get(argument, argument) for argument in call_arguments]
        out = getattr(module, forward_args['function'])(*resolved_arguments)
        shapes = {
            output_name: list(out[output_name].shape)
            for output_name in output_shapes
        }
        finite = {
            output_name: bool(torch.isfinite(out[output_name]).all())
            for output_name in output_shapes
        }
        detail = json.dumps(
            {'fixture_factory': fixture_factory, 'shapes': shapes, 'finite': finite},
            sort_keys=True,
        )
        update_check(current_index, (not require_finite) or finite['energy'], detail)
        current_index += 1
        update_check(current_index, (not require_finite) or finite['forces'], detail)
        current_index += 1
        update_check(current_index, (not require_finite) or finite['stress'], detail)
        current_index += 1
        update_check(current_index, shapes['energy'] == output_shapes['energy'], detail)
        current_index += 1
        update_check(current_index, shapes['forces'] == output_shapes['forces'], detail)
        current_index += 1
        update_check(current_index, shapes['stress'] == output_shapes['stress'], detail)
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks



def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get('params', config)
    CHECKPOINT_NAMES = [
        'energy_oracle_matches',
        'forces_oracle_matches',
        'stress_oracle_matches',
    ]
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate = params['candidate']
        reference = params['reference']
        infer_args = candidate['infer_args']
        model_path = Path(infer_args['checkpoint_file'])
        model_file = candidate['run_file']
        name = 'equiformer_v3_inference_submission'
        direct_module_path = Path(model_file)
        matches = [source for source in CANDIDATE_DIR.rglob(model_file) if source.is_file()]
        path = ((CANDIDATE_DIR / direct_module_path) if (CANDIDATE_DIR / direct_module_path).is_file()
                else matches[0]).resolve()
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(str(path))
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        device = infer_args['device']
        for variable in ('ONESCIENCE_UMA_JD_PATH', 'ONESCIENCE_EQUIFORMER_V3_JD_PATH'):
            os.environ.setdefault(variable, params['wigner_path'])
        loader = getattr(module, candidate['model_loader'])
        loaded_result = loader(str(model_path), device)
        pending = [loaded_result]
        seen = set()
        model = None
        while pending and model is None:
            value = pending.pop()
            marker = id(value)
            if marker in seen:
                continue
            seen.add(marker)
            if isinstance(value, torch.nn.Module):
                model = value
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple, set)):
                pending.extend(value)
            elif hasattr(value, '__dict__'):
                pending.extend(vars(value).values())
        if model is None:
            raise TypeError('model loader result does not contain a torch module')
        fixture = params['fixture']
        fixture_factory = fixture['factory']
        factory_module, factory_name = fixture_factory.rsplit('.', 1)
        factory = getattr(__import__(factory_module, fromlist=[factory_name]), factory_name)
        atoms = factory(*fixture['args'], **fixture['kwargs'])
        atoms = atoms.repeat(tuple(fixture['repeat']))
        atoms.rattle(**fixture['rattle'])
        from ase.stress import full_3x3_to_voigt_6_stress
        argument_values = {'context.model': model, 'fixture': atoms, 'cpu': device}
        call_arguments = candidate['call_args']
        resolved_arguments = [argument_values.get(argument, argument) for argument in call_arguments]
        out = getattr(module, candidate['infer_function'])(*resolved_arguments)

        energy_reference = reference['energy']
        forces_reference = reference['forces']
        stress_reference = reference['stress']
        energy_error = abs(float(out['energy'].reshape(-1)[0]) - energy_reference['reference'])
        forces = out['forces'].detach().cpu().float()
        reference_forces = torch.tensor(forces_reference['reference'], dtype=torch.float32)
        force_error = float((forces - reference_forces).abs().max())
        stress = out['stress'].detach().cpu().float()
        stress_voigt = torch.tensor(
            full_3x3_to_voigt_6_stress(stress.reshape(3, 3).numpy()),
            dtype=torch.float32,
        )
        reference_stress = torch.tensor(stress_reference['reference'], dtype=torch.float32)
        stress_error = float((stress_voigt - reference_stress).abs().max())
        tolerances = {
            'energy': energy_reference['maximum_absolute_error'],
            'forces': forces_reference['maximum_absolute_error'],
            'stress': stress_reference['maximum_absolute_error'],
        }
        detail = json.dumps(
            {
                'fixture_factory': fixture_factory,
                'energy_abs_error': energy_error,
                'force_abs_error': force_error,
                'stress_abs_error': stress_error,
                'atol': tolerances,
            },
            sort_keys=True,
        )
        update_check(current_index, energy_error <= tolerances['energy'], detail)
        current_index += 1
        update_check(current_index, force_error <= tolerances['forces'], detail)
        current_index += 1
        update_check(current_index, stress_error <= tolerances['stress'], detail)
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
