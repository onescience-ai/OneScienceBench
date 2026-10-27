"""V2 adapter for station PatchTST forecast evaluation."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict

import numpy as np

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
    """Check configured required files independently."""
    params = config.get("params", config)
    required_files = params.get("required_files", [])
    phase = params.get("phase", "files")
    exact_files = params.get("exact_files")
    if isinstance(required_files, str):
        required_files = [required_files]
    if not required_files:
        return [(f"{phase}.config", False, "required_files is missing or empty")]

    CHECKPOINT_NAMES = [f"{phase}.{name}" for name in required_files]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    for index, relative_path in enumerate(required_files):
        try:
            path = Path(relative_path)
            if not path.is_absolute():
                path = CANDIDATE_DIR / path
            path = path.resolve()
            if not path.is_file():
                checks[index] = (CHECKPOINT_NAMES[index], False, f"File not found: {path}")
            elif path.stat().st_size == 0:
                checks[index] = (CHECKPOINT_NAMES[index], False, f"File is empty: {path}")
            else:
                checks[index] = (CHECKPOINT_NAMES[index], True, f"File exists and is non-empty: {path}")
        except Exception as error:
            checks[index] = (CHECKPOINT_NAMES[index], False, str(error))

    if exact_files is not None:
        try:
            entries = sorted(path.name for path in CANDIDATE_DIR.iterdir())
            expected_entries = sorted(exact_files)
            passed = entries == expected_entries
            message = f"Submission entries={entries}; expected exactly={expected_entries}"
            checks[0] = (
                CHECKPOINT_NAMES[0],
                checks[0][1] and passed,
                message if passed else message,
            )
        except Exception as error:
            checks[0] = (CHECKPOINT_NAMES[0], False, str(error))

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Exercise the configured checkpoint loader without reference weights."""
    params = config.get("params", config)
    CHECKPOINT_NAMES = [
        "checkpoint_loading:state_values",
        "checkpoint_loading:malformed_rejected",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    module = None
    model = None
    loader = None
    fixture_state = None
    checkpoint_name = params.get("checkpoint_file")
    try:
        model_file = params["model_file"]
        model_class_name = params["model_class"]
        module = _load_candidate_modules(model_file)
        model_class = _load_class(module, model_class_name)
        model_config = _load_candidate_config(params["config_name"]) if params.get("config_name") else {}
        context = {"config": model_config}
        context.update(model_config)
        model_args = _resolve_args(params.get("model_args", {}), context)
        model = model_class(**model_args)
        fixture_state = {
            name: value.detach().cpu().clone() if torch.is_tensor(value) else value
            for name, value in model.state_dict().items()
        }
        changed_fixture = False
        for value in fixture_state.values():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                value.add_(0.25)
                changed_fixture = True
        if fixture_state and not changed_fixture:
            raise ValueError("Cannot create an independent tensor checkpoint fixture")
        loader_name = params["loader"]
        loader = getattr(module, loader_name, None)
        if not callable(loader):
            loader = getattr(model, loader_name, None)
        if not callable(loader):
            raise TypeError(f"{loader_name} is not callable on the module or model")
        if checkpoint_name is None:
            raise ValueError("checkpoint_file is required")
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))

    if module is not None and model is not None and callable(loader) and fixture_state is not None:
        try:
            def extract_state(value, visited=None):
                if visited is None:
                    visited = set()
                identity = id(value)
                if identity in visited:
                    return None
                visited.add(identity)
                state_method = getattr(value, "state_dict", None)
                if callable(state_method):
                    try:
                        state = state_method()
                        if isinstance(state, dict):
                            return state
                    except Exception:
                        pass
                if isinstance(value, dict):
                    if value and all(torch.is_tensor(item) for item in value.values()):
                        return value
                    for item in value.values():
                        state = extract_state(item, visited)
                        if state is not None:
                            return state
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        state = extract_state(item, visited)
                        if state is not None:
                            return state
                return None

            def states_match(actual):
                actual_state = extract_state(actual)
                if actual_state is None:
                    return None
                if set(actual_state) != set(fixture_state):
                    return False
                for key, expected in fixture_state.items():
                    observed = actual_state[key]
                    if torch.is_tensor(expected):
                        if not torch.is_tensor(observed) or not torch.equal(
                            observed.detach().cpu(), expected
                        ):
                            return False
                    elif observed != expected:
                        return False
                return True

            state_result = False
            unobservable_return = False
            failures = []
            successful_payload = None
            payloads = [
                fixture_state,
                {"state_dict": fixture_state},
                {"model_state_dict": fixture_state},
                {"model": fixture_state},
            ]
            with tempfile.TemporaryDirectory() as temporary_directory:
                good_path = Path(temporary_directory) / Path(checkpoint_name).name
                for payload in payloads:
                    torch.save(payload, good_path)
                    try:
                        loader_kwargs = {}
                        try:
                            loader_signature = inspect.signature(loader)
                            if "device" in loader_signature.parameters:
                                loader_kwargs["device"] = params.get("device", "cpu")
                            if "strict" in loader_signature.parameters:
                                loader_kwargs["strict"] = params.get("strict", True)
                        except (TypeError, ValueError):
                            pass
                        loaded = loader(good_path, **loader_kwargs)
                        match = states_match(loaded)
                        if match is True:
                            state_result = True
                            successful_payload = payload
                            break
                        if match is None:
                            state_result = True
                            unobservable_return = True
                            successful_payload = payload
                            break
                        failures.append("loaded state did not match saved state")
                    except Exception as error:
                        failures.append(str(error))
            strict_verified = False
            if state_result and isinstance(successful_payload, dict):
                partial_payload = {}
                removed_key = False
                for key, value in successful_payload.items():
                    if isinstance(value, dict) and value and all(torch.is_tensor(item) for item in value.values()):
                        partial_state = dict(value)
                        partial_state.pop(next(iter(partial_state)))
                        removed_key = True
                        partial_payload[key] = partial_state
                    else:
                        partial_payload[key] = value
                if successful_payload and all(torch.is_tensor(value) for value in successful_payload.values()):
                    partial_payload = dict(successful_payload)
                    partial_payload.pop(next(iter(partial_payload)))
                    removed_key = True
                if removed_key:
                    torch.save(partial_payload, good_path)
                    try:
                        loader(good_path, **loader_kwargs)
                    except Exception:
                        strict_verified = True
                    else:
                        strict_verified = False
            detail = (
                "loader executed; its return representation exposes no state_dict"
                if unobservable_return
                else f"loader restored the saved state ({state_result})"
            )
            state_result = state_result and strict_verified
            detail += f"; incomplete checkpoint rejected with strict loading={strict_verified}"
            if not state_result and failures:
                detail = failures[-1]
            checks[0] = (CHECKPOINT_NAMES[0], state_result, detail)
        except Exception as error:
            checks[0] = (CHECKPOINT_NAMES[0], False, str(error))

    if module is not None and callable(loader) and checkpoint_name is not None:
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                bad_path = Path(temporary_directory) / Path(checkpoint_name).name
                torch.save({}, bad_path)
                try:
                    loader_kwargs = {}
                    try:
                        loader_signature = inspect.signature(loader)
                        if "device" in loader_signature.parameters:
                            loader_kwargs["device"] = params.get("device", "cpu")
                        if "strict" in loader_signature.parameters:
                            loader_kwargs["strict"] = params.get("strict", True)
                    except (TypeError, ValueError):
                        pass
                    loader(bad_path, **loader_kwargs)
                except Exception as error:
                    checks[1] = (CHECKPOINT_NAMES[1], True, f"Malformed checkpoint rejected: {error}")
                else:
                    checks[1] = (
                        CHECKPOINT_NAMES[1],
                        not bool(params.get("strict", True)),
                        "Malformed checkpoint accepted" if params.get("strict", True) else "Non-strict loading configured",
                    )
        except Exception as error:
            checks[1] = (CHECKPOINT_NAMES[1], False, str(error))

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check required PatchTST components and prompt-defined constructor defaults."""
    params = config.get("params", config)
    CHECKPOINT_NAMES = [
        "model_structure:module_order",
        "model_structure:patch_shape",
        "model_structure:forecast_head",
        "model_structure:encoder_layers",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    try:
        module = _load_candidate_modules(params["model_file"])
        model_class = _load_class(module, params["model_class"])
        model_config = _load_candidate_config(params["config_name"]) if params.get("config_name") else {}
        context = {"config": model_config}
        context.update(model_config)
        model_args = _resolve_args(params.get("model_args", {}), context)
        device = _resolve_args({"device": params.get("device", "$device")}, {"config": {}, **{}})["device"]
        model = model_class(**model_args).to(device).eval()
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
        return checks

    for rule in params.get("rules", []):
        checkpoint_name = rule.get("checkpoint_name")
        try:
            index = CHECKPOINT_NAMES.index(checkpoint_name)
        except ValueError:
            continue
        try:
            rule_type = rule["type"]
            rule_params = rule.get("params", {})
            if rule_type == "required_attributes":
                names = rule_params.get("names", [])
                missing = [name for name in names if not hasattr(model, name)]
                passed = not missing
                message = f"Required components are present; missing={missing}"
            elif rule_type in {"defaults_and_patch_projection", "constructor_defaults"}:
                expected = rule_params.get("values", {})
                signature = inspect.signature(model_class.__init__)
                actual = {
                    name: signature.parameters[name].default
                    for name in expected
                    if name in signature.parameters
                }
                passed = all(actual.get(name) == value for name, value in expected.items())
                message = f"Constructor defaults={actual}; expected={expected}"
                context_length = expected.get("context_length")
                patch_length = expected.get("patch_length")
                if context_length is not None and patch_length:
                    passed = passed and context_length % patch_length == 0
                    message += f"; nonoverlapping patches={context_length // patch_length}"
                if rule_type == "defaults_and_patch_projection":
                    projection = getattr(model, rule_params.get("projection", "patch_projection"), None)
                    projection_width = getattr(projection, "in_features", None)
                    expected_width = rule_params.get("projection_width")
                    if expected_width is not None:
                        passed = passed and projection_width == expected_width
                        message += f"; projection input width={projection_width}"
            elif rule_type == "constructor_defaults_and_encoder":
                expected = rule_params.get("values", {})
                signature = inspect.signature(model_class.__init__)
                actual = {
                    name: signature.parameters[name].default
                    for name in expected
                    if name in signature.parameters
                }
                encoder = getattr(model, rule_params.get("attribute", "encoder"), None)
                encoder_type = getattr(torch.nn, rule_params.get("encoder_type", ""), None)
                passed = (
                    all(actual.get(name) == value for name, value in expected.items())
                    and encoder_type is not None
                    and isinstance(encoder, encoder_type)
                )
                message = f"Constructor defaults={actual}; encoder type={type(encoder).__name__}"
            else:
                raise ValueError(f"Unsupported structure rule type: {rule_type}")
            checks[index] = (CHECKPOINT_NAMES[index], bool(passed), message)
        except Exception as error:
            checks[index] = (CHECKPOINT_NAMES[index], False, str(error))

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check configured callables in every independently loadable candidate file."""
    params = config.get("params", config)
    files = params.get("files", [])
    if isinstance(files, str):
        files = [files]
    required_interfaces = params.get("required_interfaces", [])
    phase = params.get("phase", "interface")
    interface_suffixes = [
        "module_load_checkpoint" if name == "load_checkpoint" else name.rsplit(".", 1)[-1]
        for name in required_interfaces
    ]
    CHECKPOINT_NAMES = [f"{phase}:{suffix}" for suffix in interface_suffixes]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    if not files or not required_interfaces:
        return [(f"{phase}:config", False, "files and required_interfaces are required")]

    modules = []
    load_errors = []
    for filename in files:
        try:
            modules.append((filename, _load_candidate_modules(filename)))
        except Exception as error:
            load_errors.append(f"{filename}: {error}")

    for index, interface_name in enumerate(required_interfaces):
        try:
            found_in = []
            for filename, module in modules:
                if "." in interface_name:
                    class_name, attribute_name = interface_name.split(".", 1)
                    owner = getattr(module, class_name, None)
                else:
                    owner = module
                    attribute_name = interface_name
                if callable(getattr(owner, attribute_name, None)):
                    found_in.append(filename)
            passed = bool(found_in)
            message = f"{interface_name} found in {found_in}" if passed else f"{interface_name} not found; load errors={load_errors}"
            checks[index] = (CHECKPOINT_NAMES[index], passed, message)
        except Exception as error:
            checks[index] = (CHECKPOINT_NAMES[index], False, str(error))

    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Run the configured prediction probe and validate only prompt-defined behavior."""
    params = config.get("params", config)
    CHECKPOINT_NAMES = [
        "forward_executable:output_shape",
        "forward_executable:nonoverlapping_patch_contract",
        "forward_executable:input_shape_rejected",
        "forward_executable:input_immutable",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    checks[3] = (
        CHECKPOINT_NAMES[3],
        True,
        "Input mutation is not constrained by the task prompt",
    )

    candidate_spec = params.get("candidate") or {}
    model = None
    predict = None
    device = DEVICE
    try:
        module = _load_candidate_modules(candidate_spec["file"])
        function_name = candidate_spec["function"]
        if "." in function_name:
            class_name, method_name = function_name.split(".", 1)
            model_class = _load_class(module, class_name)
            model_config = _load_candidate_config(candidate_spec["config_name"]) if candidate_spec.get("config_name") else {}
            context = {"config": model_config}
            context.update(model_config)
            model_args = _resolve_args(candidate_spec.get("model_args", {}), context)
            device = _resolve_args(
                {"device": candidate_spec.get("device", "$device")}, {"config": {}}
            )["device"]
            model = model_class(**model_args).to(device).eval()
            predict = getattr(model, method_name, None)
        else:
            context = {"config": {}}
            predict = getattr(module, function_name, None)
        if not callable(predict):
            raise TypeError(f"{function_name} is not callable")
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
        return checks

    def materialize(value):
        if isinstance(value, dict) and "generator" in value and "shape" in value:
            shape = tuple(value["shape"])
            generator = value["generator"]
            if generator == "zeros":
                return torch.zeros(shape, device=device)
            if generator == "ones":
                return torch.ones(shape, device=device)
            if generator == "randn":
                return torch.randn(shape, device=device)
            raise ValueError(f"Unsupported input generator: {generator}")
        if isinstance(value, dict):
            return {key: materialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [materialize(item) for item in value]
        return value

    def output_shape(value):
        observed = getattr(value, "shape", None)
        if observed is not None:
            return tuple(int(dimension) for dimension in observed)
        if isinstance(value, (list, tuple)):
            if not value:
                return (0,)
            child_shapes = [output_shape(item) for item in value]
            if all(shape == child_shapes[0] for shape in child_shapes):
                return (len(value),) + child_shapes[0]
            return (len(value),)
        return None

    try:
        call_args = materialize(_resolve_args(candidate_spec.get("args", {}), context))
        result = predict(**call_args)
        expected_shape = candidate_spec.get("expected", {}).get("shape")
        actual_shape = output_shape(result)
        passed = expected_shape is not None and actual_shape == tuple(expected_shape)
        checks[0] = (
            CHECKPOINT_NAMES[0],
            passed,
            f"Output shape={actual_shape}; expected={expected_shape}",
        )
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))

    try:
        patch_contract = candidate_spec.get("patch_contract", {})
        context_length = patch_contract.get("context_length")
        patch_length = patch_contract.get("patch_length")
        passed = bool(context_length and patch_length and context_length % patch_length == 0)
        details = [f"nonoverlapping patch count={context_length // patch_length}" if passed else "invalid patch dimensions"]
        if model is not None:
            if not hasattr(model, "patch_projection"):
                passed = False
                details.append("patch_projection is missing")
        checks[1] = (CHECKPOINT_NAMES[1], bool(passed), "; ".join(details))
    except Exception as error:
        checks[1] = (CHECKPOINT_NAMES[1], False, str(error))

    try:
        invalid_inputs = candidate_spec.get("invalid_inputs", [])
        results = []
        for invalid in invalid_inputs:
            shape = tuple(invalid["shape"])
            try:
                invalid_values = torch.zeros(shape, device=device)
                if model is not None:
                    predict(invalid_values)
                else:
                    predict(values=invalid_values)
            except Exception:
                results.append((invalid.get("label", str(shape)), True))
            else:
                results.append((invalid.get("label", str(shape)), False))
        passed = bool(results) and all(result for _, result in results)
        checks[2] = (CHECKPOINT_NAMES[2], passed, f"Rejected invalid shapes={results}")
    except Exception as error:
        checks[2] = (CHECKPOINT_NAMES[2], False, str(error))

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Compare configured inference implementations, if a prompt-defined oracle exists."""
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["metric:forecast_rmse", "metric:forecast_mae"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    candidate_spec = params.get("candidate")
    reference_spec = params.get("reference")
    if candidate_spec is None and reference_spec is None:
        return [
            (CHECKPOINT_NAMES[0], True, "No prompt-defined numerical oracle is specified"),
            (CHECKPOINT_NAMES[1], True, "No prompt-defined numerical oracle is specified"),
        ]
    if not candidate_spec or not reference_spec:
        checks[0] = (CHECKPOINT_NAMES[0], False, "Both candidate and reference specifications are required")
        return checks

    def run(spec, source):
        module = (
            _load_candidate_modules(spec["run_file"])
            if source == "candidate"
            else _load_reference_modules(spec["run_file"])
        )
        config_data = {}
        if spec.get("config_name"):
            config_data = (
                _load_candidate_config(spec["config_name"])
                if source == "candidate"
                else _load_reference_config(spec["config_name"])
            )
        context = {"config": config_data}
        context.update(config_data)
        context = _execute_hooks(module, spec.get("pre_infer_hooks", []), context, caller_globals, caller_locals)
        infer_args = _resolve_args(spec.get("infer_args", {}), context)
        infer_function = getattr(module, spec.get("infer_function", "inference"), None)
        if not callable(infer_function):
            raise TypeError(f"Inference function is not callable in {spec['run_file']}")
        return infer_function(**infer_args)

    def to_tensor(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value).cpu()
        return torch.as_tensor(value).cpu()

    try:
        candidate_result = to_tensor(run(candidate_spec, "candidate"))
        reference_result = to_tensor(run(reference_spec, "reference"))
        shape_match = tuple(candidate_result.shape) == tuple(reference_result.shape)
        checks[0] = (
            CHECKPOINT_NAMES[0],
            shape_match,
            f"candidate shape={tuple(candidate_result.shape)}, reference shape={tuple(reference_result.shape)}",
        )
        if not shape_match:
            checks[1] = (CHECKPOINT_NAMES[1], False, "Shape mismatch; value comparison not executed")
        else:
            atol = float(params.get("atol", 0.0))
            rtol = float(params.get("rtol", 0.0))
            equal = torch.allclose(candidate_result, reference_result, atol=atol, rtol=rtol)
            checks[1] = (CHECKPOINT_NAMES[1], bool(equal), f"allclose(atol={atol}, rtol={rtol})={bool(equal)}")
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))

    return checks
