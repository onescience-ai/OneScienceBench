"""V2 adapter for the legacy GenScore training evaluator."""
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

CTOR = {
    "ligand_features": 8,
    "pocket_features": 10,
    "hidden_size": 32,
    "radial_bins": 8,
    "num_layers": 2,
}
ATTRS = (
    "ligand_encoder", "pocket_encoder", "distance_expansion", "ligand_layers",
    "ligand_layers.0.edge_gate", "pocket_layers", "cross_projection", "score_head",
)

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

FILENAME, ENV_NAME = "genscore_train.py", "GENSCORE_TRAIN_SUBMISSION_DIR"
EXPECTED = {
    "file": ("file:genscore_train.py", "file:standard:train_config.json"),
    "data": (
        "data:graph_shapes",
        "data:mask_contract",
        "data:invalid_coordinates_rejected",
    ),
    "train_script": (
        "train_script:independent_from_onescience",
        "train_script:cli_flags",
        "train_script:adamw_backward",
        "train_script:gradient_clipping",
        "train_script:structured_artifacts",
    ),
    "model_structure": (
        "model_structure:nn_module",
        "model_structure:required_components",
        "model_structure:state_dict_contract",
        "model_structure:module_order",
        "model_structure:constructor_contract",
    ),
    "forward_executable": (
        "forward_executable:tensor",
        "forward_executable:shape",
        "forward_executable:finite",
        "forward_executable:deterministic",
    ),
    "forward_equivalence": ("forward_equivalence:forward_equivalence",),
    "traning": (
        "traning:short_fit",
        "traning:parameter_update",
        "traning:npz_fit",
    ),
    "artifacts": ("artifacts:checkpoint", "artifacts:metrics"),
}
_ADAPTER_STATE: Dict[str, Any] = {}

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
    for part in name.split("."):
        value = getattr(value, part)
    return value

def source_info(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = any(
        (
            isinstance(node, ast.Import)
            and any(
                alias.name.split(".")[0] == "onescience" for alias in node.names
            )
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "onescience"
        )
        for node in ast.walk(tree)
    )
    return (
        source,
        imports,
        {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)},
    )

def fixture(torch):
    generator = torch.Generator().manual_seed(73)
    return (
        torch.randn(2, 5, 8, generator=generator),
        torch.randn(2, 5, 3, generator=generator),
        torch.randn(2, 7, 10, generator=generator),
        torch.randn(2, 7, 3, generator=generator),
        torch.ones(2, 5, dtype=torch.bool),
        torch.ones(2, 7, dtype=torch.bool),
    )

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    required_files = params["required_files"]
    phase = params["phase"]
    CHECKPOINT_NAMES = [f"{phase}:{name}" for name in required_files]
    CHECKPOINT_NAMES.append(f"{phase}:{params['legacy_checkpoint']}")
    checks = [(name, False, "Not executed due to earlier failure")
              for name in CHECKPOINT_NAMES]
    for current_index, relative_path in enumerate(required_files):
        try:
            path = (CANDIDATE_DIR / relative_path).resolve()
            passed = path.is_file() and bool(path.read_text(encoding="utf-8").strip())
            checks[current_index] = (CHECKPOINT_NAMES[current_index], passed,
                                     str(path) if passed else "File is missing or empty")
        except Exception as error:
            checks[current_index] = (CHECKPOINT_NAMES[current_index], False, str(error))
    legacy_index = len(required_files)
    checks[legacy_index] = (CHECKPOINT_NAMES[legacy_index], True,
                            f"Not required by the task contract: {params['legacy_checkpoint']}")
    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["data:graph_shapes", "data:mask_contract",
                        "data:invalid_coordinates_rejected"]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in CHECKPOINT_NAMES]
    spec = params["candidate"]
    for checkpoint in spec.get("non_contract_checkpoints", []):
        if checkpoint in CHECKPOINT_NAMES:
            index = CHECKPOINT_NAMES.index(checkpoint)
            checks[index] = (CHECKPOINT_NAMES[index], True,
                             f"No task contract is configured for {checkpoint}")
    try:
        import torch
        from base_evaluator import _resolve_args, _execute_hooks

        module = load_module("candidate_method_check", CANDIDATE_DIR / spec["file"])
        context = {"config": params, "device": spec["device"]}
        context = _execute_hooks(module, spec.get("pre_hooks", []), context,
                                 caller_globals, caller_locals)
        constructor = dotted(module, spec["function"])
        model = constructor(**_resolve_args(spec["args"], context))
        if callable(getattr(model, "to", None)):
            model.to(spec["device"])
        if callable(getattr(model, "eval", None)):
            model.eval()
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
        model = None

    def make_fields(spec, device, context):
        resolved = _resolve_args(spec, context)
        generator = torch.Generator().manual_seed(resolved["seed"])
        fields = []
        for item in resolved["inputs"]:
            dtype = getattr(torch, item["dtype"])
            factory = getattr(torch, item["factory"])
            options = {"dtype": dtype}
            if item["factory"] in spec["generator_factories"]:
                options["generator"] = generator
            tensor = factory(*item["shape"], **options)
            if "offset" in item:
                tensor = tensor + item["offset"]
            fields += [{**item, "tensor": tensor.to(device)}]
        return fields

    def invoke(model, fields):
        # Bind semantic inputs by signature; also support a single batch mapping.
        callable_model = getattr(model, "forward", model)
        signature = inspect.signature(callable_model)
        parameters = list(signature.parameters.values())
        named = [p for p in parameters if p.kind != inspect.Parameter.VAR_POSITIONAL
                 and p.kind != inspect.Parameter.VAR_KEYWORD]
        aliases = {
            alias.lower(): index
            for index, field in enumerate(fields)
            for alias in field["names"]
        }
        class Bundle(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as error:
                    raise AttributeError(name) from error

        groups = {}
        for field in fields:
            if "path" not in field:
                continue
            group, member = field["path"]
            bundle = groups.setdefault(group, Bundle())
            bundle[member] = field["tensor"]
            # Short aliases support graph objects exposing x/pos/coords/mask.
            for alias in field["names"]:
                if alias.startswith(group + "_"):
                    bundle[alias[len(group) + 1:]] = field["tensor"]
        grouped = bool(named) and all(p.name.lower() in groups for p in named)
        if grouped:
            positional = [groups[p.name.lower()] for p in named
                          if p.kind == inspect.Parameter.POSITIONAL_ONLY]
            keywords = {p.name: groups[p.name.lower()] for p in named
                        if p.kind != inspect.Parameter.POSITIONAL_ONLY}
            return callable_model(*positional, **keywords)
        if len(named) == 1 and named[0].name.lower() not in aliases:
            batch = Bundle({field["names"][0]: field["tensor"] for field in fields})
            if named[0].kind == inspect.Parameter.KEYWORD_ONLY:
                return callable_model(**{named[0].name: batch})
            return callable_model(batch)
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
        assigned = {
            p.name: aliases[p.name.lower()]
            for p in named if p.name.lower() in aliases
        }
        remaining = [i for i in range(len(fields)) if i not in assigned.values()]
        positional, keywords = [], {}
        for parameter in named:
            index = assigned.get(parameter.name)
            if index is None and parameter.default is inspect.Parameter.empty and remaining:
                index = remaining.pop(0)
            if index is None:
                continue
            value = fields[index]["tensor"]
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY or (
                    has_varargs and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
                positional.append(value)
            else:
                keywords[parameter.name] = value
        if has_varargs:
            positional.extend(fields[i]["tensor"] for i in remaining)
        elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
            keywords.update({fields[i]["names"][0]: fields[i]["tensor"] for i in remaining})
        return callable_model(*positional, **keywords)

    if model is not None:
        try:
            fields = make_fields(spec["forward_args"], spec["device"], context)
            with torch.inference_mode():
                result = invoke(model, fields)
            output_ok = isinstance(result, torch.Tensor)
            output_ok = output_ok and list(result.shape) == spec["expected"]["shape"]
            graph_index = CHECKPOINT_NAMES.index("data:graph_shapes")
            checks[graph_index] = (CHECKPOINT_NAMES[graph_index], output_ok,
                                   "Output matches the configured batch shape")
            mask_fields = [field for field in fields
                           if ".".join(field.get("path", [])) in spec["mask_members"]]
            masks_ok = len(mask_fields) == len(spec["mask_members"])
            masks_ok = masks_ok and all(
                field["tensor"].dtype == getattr(torch, spec["mask_dtype"])
                for field in mask_fields
            )
            mask_index = CHECKPOINT_NAMES.index("data:mask_contract")
            checks[mask_index] = (CHECKPOINT_NAMES[mask_index], masks_ok,
                                  "Configured mask inputs have the required dtype")
        except Exception as error:
            for checkpoint in ("data:graph_shapes", "data:mask_contract"):
                index = CHECKPOINT_NAMES.index(checkpoint)
                checks[index] = (CHECKPOINT_NAMES[index], False, str(error))
    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = [
        "train_script:independent_from_onescience", "train_script:cli_flags",
        "train_script:adamw_backward", "train_script:gradient_clipping",
        "train_script:structured_artifacts",
    ]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in CHECKPOINT_NAMES]
    nodes, sources, errors = [], [], []
    for filename in params["files"]:
        try:
            source = (CANDIDATE_DIR / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            nodes += list(ast.walk(tree))
            sources += [source]
        except Exception as error:
            errors += [str(error)]
    rules = params["check_config"]
    aliases = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = (node.module or "") + "." + alias.name

    def call_name(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return call_name(node.value) + "." + node.attr
        return ""

    names = {call_name(node.func).split(".")[-1]
             for node in nodes if isinstance(node, ast.Call)}
    source = "\n".join(sources)
    imports = [alias.name for node in nodes if isinstance(node, ast.Import)
               for alias in node.names]
    imports += [node.module or "" for node in nodes if isinstance(node, ast.ImportFrom)]
    predicates = [
        lambda: not any(name.split(".")[0] in rules["excluded_import_roots"]
                        for name in imports),
        lambda: all(flag in source for flag in rules["cli_flags"])
                or bool(names.intersection(rules["cli_parser_names"])),
        lambda: bool(names.intersection(rules["optimizer_names"]))
                and bool(names.intersection(rules["backward_names"]))
                and bool(names.intersection(rules["loss_names"])),
        lambda: bool(names.intersection(rules["clipping_names"])),
        lambda: all(filename in source for filename in rules["artifact_files"])
                or bool(names.intersection(rules["artifact_writers"])),
    ]
    for index, predicate in enumerate(predicates):
        try:
            passed = bool(nodes) and not errors and bool(predicate())
            checks[index] = (CHECKPOINT_NAMES[index], passed,
                             str(errors) if errors else "Training AST contract")
        except Exception as error:
            checks[index] = (CHECKPOINT_NAMES[index], False, str(error))
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    rules = params["rules"]
    CHECKPOINT_NAMES = ["model_structure:nn_module"] + [
        "model_structure:" + rule["id"] for rule in rules
    ]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in CHECKPOINT_NAMES]
    model = None
    try:
        import torch
        from base_evaluator import _resolve_args

        module = load_module("candidate_structure", CANDIDATE_DIR / params["model_file"])
        configuration = {}
        if params.get("config_name"):
            configuration = yaml.safe_load(
                (CANDIDATE_DIR / params["config_name"]).read_text(encoding="utf-8"))
        context = {"config": configuration, **configuration}
        constructor = dotted(module, params["model_class"])
        model = constructor(**_resolve_args(params["model_args"], context))
        checks[0] = (CHECKPOINT_NAMES[0], model is not None,
                     "Configured public model constructed")
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))

    for index, rule in enumerate(rules, 1):
        try:
            kind = rule["type"]
            options = rule.get("params", {})
            if kind == "unconstrained":
                passed = True
                message = "The prompt defines no state-dict key or tensor-layout ABI"
            elif model is None:
                continue
            elif kind == "attributes":
                passed = all(hasattr(model, name) for name in options["names"])
                message = str(options["names"])
            elif kind == "attribute_order":
                names = options["names"]
                order = [name for name in model._modules if name in names]
                passed = order == names
                message = str(order)
            elif kind == "defaults":
                signature = inspect.signature(constructor)
                # **kwargs constructors are valid; named defaults, when exposed, must agree.
                passed = all(
                    name not in signature.parameters
                    or signature.parameters[name].default == expected
                    for name, expected in options["values"].items()
                )
                message = str(signature)
            else:
                raise ValueError("Unknown structural rule: " + kind)
            checks[index] = (CHECKPOINT_NAMES[index], bool(passed), message)
        except Exception as error:
            checks[index] = (CHECKPOINT_NAMES[index], False, str(error))
    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = [
        "forward_executable:tensor", "forward_executable:shape",
        "forward_executable:finite", "forward_executable:deterministic",
    ]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in CHECKPOINT_NAMES]
    outputs = []
    current_index = 0

    def make_fields(spec, device, context):
        resolved = _resolve_args(spec, context)
        generator = torch.Generator().manual_seed(resolved["seed"])
        fields = []
        for item in resolved["inputs"]:
            dtype = getattr(torch, item["dtype"])
            factory = getattr(torch, item["factory"])
            options = {"dtype": dtype}
            if item["factory"] in params["generator_factories"]:
                options["generator"] = generator
            tensor = factory(*item["shape"], **options)
            if "offset" in item:
                tensor = tensor + item["offset"]
            fields += [{**item, "tensor": tensor.to(device)}]
        return fields

    def invoke(model, fields):
        # Bind semantic inputs by signature; also support a single batch mapping.
        callable_model = getattr(model, "forward", model)
        signature = inspect.signature(callable_model)
        parameters = list(signature.parameters.values())
        named = [p for p in parameters if p.kind != inspect.Parameter.VAR_POSITIONAL
                 and p.kind != inspect.Parameter.VAR_KEYWORD]
        aliases = {
            alias.lower(): index
            for index, field in enumerate(fields)
            for alias in field["names"]
        }
        class Bundle(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as error:
                    raise AttributeError(name) from error

        groups = {}
        for field in fields:
            if "path" not in field:
                continue
            group, member = field["path"]
            bundle = groups.setdefault(group, Bundle())
            bundle[member] = field["tensor"]
            # Short aliases support graph objects exposing x/pos/coords/mask.
            for alias in field["names"]:
                if alias.startswith(group + "_"):
                    bundle[alias[len(group) + 1:]] = field["tensor"]
        grouped = bool(named) and all(p.name.lower() in groups for p in named)
        if grouped:
            positional = [groups[p.name.lower()] for p in named
                          if p.kind == inspect.Parameter.POSITIONAL_ONLY]
            keywords = {p.name: groups[p.name.lower()] for p in named
                        if p.kind != inspect.Parameter.POSITIONAL_ONLY}
            return callable_model(*positional, **keywords)
        if len(named) == 1 and named[0].name.lower() not in aliases:
            batch = Bundle({field["names"][0]: field["tensor"] for field in fields})
            if named[0].kind == inspect.Parameter.KEYWORD_ONLY:
                return callable_model(**{named[0].name: batch})
            return callable_model(batch)
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
        assigned = {
            p.name: aliases[p.name.lower()]
            for p in named if p.name.lower() in aliases
        }
        remaining = [i for i in range(len(fields)) if i not in assigned.values()]
        positional, keywords = [], {}
        for parameter in named:
            index = assigned.get(parameter.name)
            if index is None and parameter.default is inspect.Parameter.empty and remaining:
                index = remaining.pop(0)
            if index is None:
                continue
            value = fields[index]["tensor"]
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY or (
                    has_varargs and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
                positional.append(value)
            else:
                keywords[parameter.name] = value
        if has_varargs:
            positional.extend(fields[i]["tensor"] for i in remaining)
        elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
            keywords.update({fields[i]["names"][0]: fields[i]["tensor"] for i in remaining})
        return callable_model(*positional, **keywords)

    try:
        import torch
        from base_evaluator import _resolve_args, _execute_hooks

        module = load_module("candidate_forward", CANDIDATE_DIR / params["model_file"])
        configuration = {}
        if params.get("config_name"):
            configuration = yaml.safe_load(
                (CANDIDATE_DIR / params["config_name"]).read_text(encoding="utf-8"))
        device = params["device"]
        context = {"config": configuration, **configuration, "device": device}
        context = _execute_hooks(module, params.get("pre_forward_hooks", []), context,
                                 caller_globals, caller_locals)
        constructor = dotted(module, params["model_class"])
        model = constructor(**_resolve_args(params["model_args"], context))
        if callable(getattr(model, "to", None)):
            model.to(device)
        if callable(getattr(model, "eval", None)):
            model.eval()
        fields = make_fields(params["forward_args"], device, context)
        with torch.inference_mode():
            output = invoke(model, fields)
        outputs += [output]
        checks[0] = (CHECKPOINT_NAMES[0], isinstance(output, torch.Tensor), str(type(output)))
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))

    if outputs:
        output_is_tensor = isinstance(outputs[0], torch.Tensor)
        shape_ok = output_is_tensor and (
            params.get("out_shape") is None or list(outputs[0].shape) == params["out_shape"]
        )
        checks[1] = (CHECKPOINT_NAMES[1], shape_ok, "Configured output shape")
        finite_ok = output_is_tensor and (
            not params["require_finite"] or bool(torch.isfinite(outputs[0]).all())
        )
        checks[2] = (CHECKPOINT_NAMES[2], finite_ok, "Finite affinity output")
        checks[3] = (CHECKPOINT_NAMES[3], not params["require_deterministic"],
                     "Deterministic repeat behavior is not required")
    else:
        checks[1] = (CHECKPOINT_NAMES[1], False, "Forward call did not produce a result")
    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["forward_equivalence:forward_equivalence"]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in CHECKPOINT_NAMES]
    def make_fields(spec, device, context):
        resolved = _resolve_args(spec, context)
        generator = torch.Generator().manual_seed(resolved["seed"])
        fields = []
        for item in resolved["inputs"]:
            dtype = getattr(torch, item["dtype"])
            factory = getattr(torch, item["factory"])
            options = {"dtype": dtype}
            if item["factory"] in spec["generator_factories"]:
                options["generator"] = generator
            tensor = factory(*item["shape"], **options)
            if "offset" in item:
                tensor = tensor + item["offset"]
            fields += [{**item, "tensor": tensor.to(device)}]
        return fields

    def invoke(model, fields):
        # Bind semantic inputs by signature; also support a single batch mapping.
        callable_model = getattr(model, "forward", model)
        signature = inspect.signature(callable_model)
        parameters = list(signature.parameters.values())
        named = [p for p in parameters if p.kind not in
                 (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
        aliases = {
            alias.lower(): index
            for index, field in enumerate(fields)
            for alias in field["names"]
        }
        class Bundle(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as error:
                    raise AttributeError(name) from error

        groups = {}
        for field in fields:
            if "path" not in field:
                continue
            group, member = field["path"]
            bundle = groups.setdefault(group, Bundle())
            bundle[member] = field["tensor"]
            # Short aliases support graph objects exposing x/pos/coords/mask.
            for alias in field["names"]:
                if alias.startswith(group + "_"):
                    bundle[alias[len(group) + 1:]] = field["tensor"]
        grouped = bool(named) and all(p.name.lower() in groups for p in named)
        if grouped:
            positional = [groups[p.name.lower()] for p in named
                          if p.kind == inspect.Parameter.POSITIONAL_ONLY]
            keywords = {p.name: groups[p.name.lower()] for p in named
                        if p.kind != inspect.Parameter.POSITIONAL_ONLY}
            return callable_model(*positional, **keywords)
        if len(named) == 1 and named[0].name.lower() not in aliases:
            batch = Bundle({field["names"][0]: field["tensor"] for field in fields})
            if named[0].kind == inspect.Parameter.KEYWORD_ONLY:
                return callable_model(**{named[0].name: batch})
            return callable_model(batch)
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
        assigned = {
            p.name: aliases[p.name.lower()]
            for p in named if p.name.lower() in aliases
        }
        remaining = [i for i in range(len(fields)) if i not in assigned.values()]
        positional, keywords = [], {}
        for parameter in named:
            index = assigned.get(parameter.name)
            if index is None and parameter.default is inspect.Parameter.empty and remaining:
                index = remaining.pop(0)
            if index is None:
                continue
            value = fields[index]["tensor"]
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY or (
                    has_varargs and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
                positional.append(value)
            else:
                keywords[parameter.name] = value
        if has_varargs:
            positional.extend(fields[i]["tensor"] for i in remaining)
        elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
            keywords.update({fields[i]["names"][0]: fields[i]["tensor"] for i in remaining})
        return callable_model(*positional, **keywords)

    def run_side(spec, default_source, name):
        torch.manual_seed(spec["seed"])
        source = spec.get("source", default_source)
        root = CANDIDATE_DIR if source == "candidate" else REFERENCE_DIR
        module = load_module(name, root / spec["model_file"])
        configuration = {}
        if spec.get("config_name"):
            configuration = yaml.safe_load(
                (root / spec["config_name"]).read_text(encoding="utf-8"))
        device = spec["device"]
        context = {"config": configuration, **configuration, "device": device}
        context = _execute_hooks(module, spec.get("pre_forward_hooks", []), context,
                                 caller_globals, caller_locals)
        constructor = dotted(module, spec["model_class"])
        model = constructor(**_resolve_args(spec["model_args"], context))
        # Constant initialization avoids imposing the reference's parameter names or shapes.
        _initialize_parameters_with_constants(model, **spec["initialization"])
        if callable(getattr(model, "to", None)):
            model.to(device)
        if callable(getattr(model, "eval", None)):
            model.eval()
        fields = make_fields(spec["forward_args"], device, context)
        with torch.inference_mode():
            return invoke(model, fields)

    if params["oracle_required"]:
        try:
            import torch
            from base_evaluator import (
                _resolve_args, _execute_hooks, _initialize_parameters_with_constants,
            )

            candidate_output = run_side(params["candidate"], "candidate", "candidate_equivalence")
            reference_output = run_side(params["reference"], "reference", "reference_equivalence")
            shape_ok = candidate_output.shape == reference_output.shape
            passed = shape_ok and bool(torch.allclose(
                candidate_output, reference_output, atol=params["compare_atol"],
                rtol=params["compare_rtol"],
            ))
            checks[0] = (CHECKPOINT_NAMES[0], passed,
                         "Configured equivalent forwards agree in shape and value")
        except Exception as error:
            checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
    else:
        checks[0] = (CHECKPOINT_NAMES[0], True,
                     "The task defines no independent reference-output oracle")
    return checks


def check_smoke_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["traning:short_fit", "traning:parameter_update", "traning:npz_fit"]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in CHECKPOINT_NAMES]
    root = None
    try:
        from base_evaluator import _resolve_args

        manager = tempfile.TemporaryDirectory(prefix="training_smoke_")
        root = Path(manager.name)
        _ADAPTER_STATE["manager"] = manager
        context = {"work_dir": str(root), "config": params["smoke_config"]}
        smoke = _resolve_args(params["smoke_config"], context)
        context["output_dir"] = smoke["output_dir"]
        smoke = _resolve_args(params["smoke_config"], context)
        _ADAPTER_STATE.update(context)
        run_file = (CANDIDATE_DIR / params["run_file"]).resolve()
    except Exception as error:
        checks[0] = (CHECKPOINT_NAMES[0], False, str(error))
        root = None

    # Observe actual optimizer updates without requiring a checkpoint payload layout.
    observer_code = """
import json
import runpy
import sys
from pathlib import Path
import torch
spec = json.loads(sys.argv[1])
optimizer_class = getattr(torch.optim, spec["optimizer"])
original_step = optimizer_class.step
updated = False
def observed_step(self, *args, **kwargs):
    global updated
    parameters = [p for group in self.param_groups for p in group["params"]]
    before = [p.detach().clone() for p in parameters]
    result = original_step(self, *args, **kwargs)
    updated = updated or any(not torch.equal(old, new.detach())
                             for old, new in zip(before, parameters))
    return result
optimizer_class.step = observed_step
sys.path.insert(0, str(Path(spec["file"]).parent))
sys.argv = [spec["file"]] + spec["args"]
try:
    runpy.run_path(spec["file"], run_name="__main__")
finally:
    Path(spec["report"]).write_text(json.dumps({"updated": updated}), encoding="utf-8")
"""
    if root is not None:
        for run in smoke["runs"]:
            index = CHECKPOINT_NAMES.index(run["checkpoint"])
            try:
                if index == 2:
                    import numpy as np
                    import torch

                    fixture_spec = smoke["npz_fixture"]
                    generator = torch.Generator().manual_seed(fixture_spec["seed"])
                    data = {}
                    for field, spec in fixture_spec["fields"].items():
                        factory = getattr(torch, spec["factory"])
                        dtype = getattr(torch, spec["dtype"])
                        options = {"dtype": dtype}
                        if spec["factory"] in fixture_spec["generator_factories"]:
                            options["generator"] = generator
                        tensor = factory(*spec["shape"], **options)
                        data[field] = tensor.numpy()
                    np.savez_compressed(fixture_spec["path"], **data)
                arguments = run["args"]
                if index == 0:
                    observation = smoke["parameter_update"]
                    command = [
                        sys.executable, "-c", observer_code,
                        json.dumps({"file": str(run_file), "args": arguments,
                                    "optimizer": observation["optimizer"],
                                    "report": observation["report"]}),
                    ]
                else:
                    command = [sys.executable, str(run_file), *arguments]
                process = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                         timeout=smoke["timeout_seconds"])
                passed = process.returncode == 0
                passed = passed and all(Path(path).is_file() for path in run.get("artifacts", []))
                checks[index] = (CHECKPOINT_NAMES[index], passed,
                                 f"returncode={process.returncode}; {process.stderr[-1000:]}")
            except Exception as error:
                checks[index] = (CHECKPOINT_NAMES[index], False, str(error))
            if index == 0:
                try:
                    report = json.loads(Path(smoke["parameter_update"]["report"]).read_text(
                        encoding="utf-8"))
                    checks[1] = (CHECKPOINT_NAMES[1], report["updated"] is True,
                                 "Observed changes to candidate optimizer parameters")
                except Exception as error:
                    checks[1] = (CHECKPOINT_NAMES[1], False, str(error))
    return checks


def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    params = config.get("params", config)
    CHECKPOINT_NAMES = ["artifacts:checkpoint", "artifacts:metrics"]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in CHECKPOINT_NAMES]
    artifacts = params["artifacts"]
    groups = [artifacts.get("checkpoints", []), artifacts.get("files", [])]
    for index, paths in enumerate(groups):
        try:
            from base_evaluator import _resolve_args

            resolved = _resolve_args({"root": artifacts.get("root", ".")}, _ADAPTER_STATE)
            if resolved["root"] is None:
                raise FileNotFoundError("Smoke artifact directory has not been created")
            root = Path(resolved["root"])
            if not root.is_absolute():
                root = CANDIDATE_DIR / root
            passed = bool(paths) and all((root / path).is_file() for path in paths)
            checks[index] = (CHECKPOINT_NAMES[index], passed,
                             str([str(root / path) for path in paths]))
        except Exception as error:
            checks[index] = (CHECKPOINT_NAMES[index], False, str(error))
    return checks
