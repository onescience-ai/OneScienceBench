#!/usr/bin/env python
"""Stormer phase-level checks bounded by the unchanged Prompt.

Complete Base checks are imported directly where applicable. The named model
interface delegates to Base; adapted checks preserve the applicable Base
execution skeleton while removing requirements absent from the
Prompt, such as a mandatory config file. Candidate loading and execution use
the runner-bound submission directory exposed by BaseEvaluator.
No fixed train/evaluate name, returned dict/key, config or checkpoint is required.
Unadaptable behavior is explicitly UNRESOLVED, never a fabricated PASS.
"""
from __future__ import annotations

import ast
import copy
import inspect
import math
import numbers
import sys
import textwrap
import traceback
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = nn = DataLoader = TensorDataset = None

# ``rule_eval`` binds the candidate environment before importing this module.
# Export complete Base checks directly where their contract fits the Prompt.
_EVALUATOR_DIR = Path(__file__).resolve().parent
for _parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    _base_dir = _parent / 'benchmarks' / 'evaluators' / 'base'
    if _base_dir.is_dir() and str(_base_dir) not in sys.path:
        sys.path.insert(0, str(_base_dir))

from base import base_evaluator
from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _compute_mae,
    _compute_mse,
    _compute_rmse,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _move_tensors_to_device,
    _resolve_args,
    check_files,
    check_parameter_update,
)
from base.model_structure_checker import ModelStructureChecker

# Evaluator-owned legal inputs, NOT required candidate defaults.
DEFAULT_MODEL_ARGS = {
    "in_channels": 2, "out_channels": 2, "img_size": (8, 8),
    "patch_size": 2, "embed_dim": 16, "depth": 2, "num_heads": 4, "history": 2,
}
SEED = 20260819


class EvaluationUnresolved(RuntimeError):
    """Insufficient evaluator adaptation, not proof of an Agent defect."""


def _scalar(value):
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("not a scalar tensor")
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"not a numeric scalar: {type(value).__name__}")
    return float(value)

def _select_layout(model, inputs, target):
    failures = []
    for layout, value in (("5D", inputs), ("4D", inputs.flatten(1, 2))):
        try:
            with torch.no_grad():
                output = model(value)
            if tuple(output.shape) != tuple(target.shape):
                raise ValueError(f"output shape {tuple(output.shape)} != {tuple(target.shape)}")
            if not torch.isfinite(output).all():
                raise ValueError("non-finite forecast for finite inputs")
            return layout, value, output
        except Exception as exc:
            failures.append(f"{layout}: {type(exc).__name__}: {exc}")
    raise ValueError("neither Prompt-permitted layout works: " + "; ".join(failures))


def _discover_entry(module, path, training):
    """Discover a behavior-bearing public entry without imposing a fixed name."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def call_name(node):
        return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""

    def reachable(function, seen=None):
        seen = set() if seen is None else seen
        if function.name in seen:
            return []
        seen.add(function.name)
        nodes = list(ast.walk(function))
        for node in list(nodes):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                nodes.extend(reachable(functions[node.func.id], seen))
        return nodes

    def metric_name(value):
        text = "".join(character for character in str(value).lower() if character.isalnum())
        return {
            "mae": "mae", "meanabsoluteerror": "mae", "rmse": "rmse",
            "rootmeansquarederror": "rmse",
        }.get(text)

    def aliases(function):
        result = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                result.update({target.id: node.value for target in node.targets if isinstance(target, ast.Name)})
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                result[node.target.id] = node.value
        return result

    def metric_intent(expr, known, seen=None):
        def expanded(value, visited):
            nodes = list(ast.walk(value))
            for node in list(nodes):
                if isinstance(node, ast.Name) and node.id in known and node.id not in visited:
                    visited.add(node.id)
                    nodes.extend(expanded(known[node.id], visited))
            return nodes

        nodes = expanded(expr, set() if seen is None else seen)
        for node in nodes:
            if isinstance(node, ast.Call) and metric_name(call_name(node.func)):
                return metric_name(call_name(node.func))
        calls = {call_name(node.func).lower() for node in nodes if isinstance(node, ast.Call)}
        if "sqrt" in calls or any(
            isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Constant) and node.right.value == 0.5
            for node in nodes
        ):
            return "rmse"
        return "mae" if {"abs", "absolute"} & calls else None

    scores, direct_scores = {}, {}
    for name, function in functions.items():
        nodes = reachable(function)
        calls = {call_name(node.func) for node in nodes if isinstance(node, ast.Call)}
        if training:
            scores[name] = int("backward" in calls) + int("step" in calls)
            direct_calls = {
                call_name(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)
            }
            direct_scores[name] = int("backward" in direct_calls) + int("step" in direct_calls)
            continue
        known = aliases(function)
        labels = {
            metric_intent(node.value, known)
            for node in nodes if isinstance(node, ast.Return) and node.value
        }
        labels.update(metric_name(node.id) for node in nodes if isinstance(node, ast.Name))
        labels.update(metric_name(node.value) for node in nodes if isinstance(node, ast.Constant) and isinstance(node.value, str))
        labels.update(metric_name(call) for call in calls)
        scores[name] = int("mae" in labels or bool({"abs", "absolute"} & calls)) + int(
            "rmse" in labels or "sqrt" in calls
        )
        direct_nodes = list(ast.walk(function))
        direct_calls = {
            call_name(node.func) for node in direct_nodes if isinstance(node, ast.Call)
        }
        direct_labels = {
            metric_intent(node.value, known)
            for node in direct_nodes if isinstance(node, ast.Return) and node.value
        }
        direct_labels.update(metric_name(node.id) for node in direct_nodes if isinstance(node, ast.Name))
        direct_labels.update(
            metric_name(node.value) for node in direct_nodes
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        direct_labels.update(metric_name(call) for call in direct_calls)
        direct_scores[name] = int(
            "mae" in direct_labels or bool({"abs", "absolute"} & direct_calls)
        ) + int("rmse" in direct_labels or "sqrt" in direct_calls)
    choices = []
    for name, function in functions.items():
        value = getattr(module, name, None)
        if callable(value) and scores[name] == 2:
            direct = {
                node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
            }
            required = sum(p.default is inspect.Parameter.empty for p in inspect.signature(value).parameters.values()
                           if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
            choices.append((int("backward" in direct and "step" in direct) if training else
                            10 * int("model" in _argument_roles(function).values()) +
                            int(direct_scores[name] == 2),
                            -required, name, value, function))
    if not choices:
        raise EvaluationUnresolved("UNRESOLVED: no adaptably located behavior-bearing callable; class/CLI-only code needs an adapter, not fixed function names")
    choices.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(choices) > 1 and choices[0][:2] == choices[1][:2]:
        raise EvaluationUnresolved(f"UNRESOLVED: ambiguous entries {[c[2] for c in choices]}")
    return choices[0][3], choices[0][4]


def _argument_roles(function):
    def aliases():
        result = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                result.update({target.id: node.value for target in node.targets if isinstance(target, ast.Name)})
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                result[node.target.id] = node.value
        return result

    def expanded(expr, known, seen=None):
        seen = set() if seen is None else seen
        nodes = list(ast.walk(expr))
        for node in list(nodes):
            if isinstance(node, ast.Name) and node.id in known and node.id not in seen:
                seen.add(node.id)
                nodes.extend(expanded(known[node.id], known, seen))
        return nodes

    parameters = {a.arg: ast.unparse(a.annotation) if a.annotation else ""
                  for a in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]}
    known, roles = aliases(), {}
    for name, annotation in parameters.items():
        if "Module" in annotation or "Stormer" in annotation:
            roles[name] = "model"
        elif "Optimizer" in annotation:
            roles[name] = "optimizer"
        elif "DataLoader" in annotation:
            roles[name] = "loader"
        elif "device" in annotation.lower():
            roles[name] = "device"
        elif annotation in ("int", "builtins.int"):
            roles[name] = "integer_fixture"
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            name = node.func.value.id
            if name in parameters and node.func.attr in ("parameters", "train", "eval"):
                roles[name] = "model"
            elif name in parameters and node.func.attr in ("step", "zero_grad"):
                roles[name] = "optimizer"
            if node.func.attr == "to":
                for item in node.args:
                    if isinstance(item, ast.Name) and item.id in parameters:
                        roles[item.id] = "device"
        if isinstance(node, ast.For):
            for item in ast.walk(node.iter):
                if isinstance(item, ast.Name) and item.id in parameters:
                    roles[item.id] = "loader"
    model_names = {name for name, role in roles.items() if role == "model"}
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in parameters:
            roles[node.func.id] = "model"
            model_names.add(node.func.id)
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in model_names:
            for item in ast.walk(ast.Tuple(elts=node.args, ctx=ast.Load())):
                if isinstance(item, ast.Name) and item.id in parameters:
                    roles[item.id] = "inputs"
    for node in ast.walk(function):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            sides = node.left, node.right
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            sides = node.args[:2]
        else:
            continue
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in model_names
               for n in expanded(sides[0], known)):
            for item in ast.walk(sides[1]):
                if isinstance(item, ast.Name) and item.id in parameters and item.id not in roles:
                    roles[item.id] = "targets"
    return roles


def _bind_entry(entry, function, model, inputs, targets):
    roles = _argument_roles(function)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=inputs.shape[0], shuffle=False)
    values = {"model": model, "inputs": inputs, "targets": targets, "loader": loader,
              "device": DEVICE, "integer_fixture": 1}
    args, kwargs, injected = [], {}, False
    for name, p in inspect.signature(entry).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.default is not inspect.Parameter.empty:
            continue
        role = roles.get(name)
        if role is None:
            raise EvaluationUnresolved(f"UNRESOLVED: required parameter {name!r} has no source-proven input role")
        if role == "optimizer":
            value, injected = torch.optim.AdamW(model.parameters(), lr=1e-3), True
        else:
            value = values[role]
        if p.kind == p.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value
    inspect.signature(entry).bind(*args, **kwargs)
    return args, kwargs, injected


def check_interface(config: dict, caller_globals=None, caller_locals=None, artifact_path=None):
    """Use Base for the named model interface; adapt unnamed script entries."""
    files = config.get("files", [])
    if isinstance(files, str):
        files = [files]
    if config.get("phase") == "model_structure" and files == ["stormer.py"]:
        return base_evaluator.check_interface(config, caller_globals, caller_locals)
    training = files == ["train.py"]
    name = f"{'train_interface' if training else 'infer_interface'}:behavior_implementation"
    try:
        root, phase = CANDIDATE_DIR, config.get("phase")
        if phase not in ("train_script", "infer_script") or len(files) != 1:
            raise ValueError("script interface requires exactly one own-phase file")
        # checks2-update.xls assigns infer_interface to train_script.  The
        # Prompt's phase-specific file, not that inconsistent label, determines
        # which candidate behavior to discover.  No function name is imposed.
        if files[0] not in ("train.py", "evaluate.py"):
            raise ValueError("script interface file must be train.py or evaluate.py")
        training = files[0] == "train.py"
        name = f"{'train_interface' if training else 'infer_interface'}:behavior_implementation"
        module = _load_candidate_modules(files[0])
        entry, function = _discover_entry(module, root / files[0], training)
        return [(name, True, f"candidate-defined entry={function.name}, signature={inspect.signature(entry)}; no required name")]
    except Exception:
        return [(name, False, traceback.format_exc())]


def _components(model, args):
    def block_parts(block):
        children = list(block.children())
        norm_types = tuple(
            candidate for candidate in (nn.LayerNorm, getattr(nn, "RMSNorm", None))
            if candidate is not None
        )
        norms = [child for child in children if isinstance(child, norm_types)]
        attentions = []
        for child in children:
            if isinstance(child, nn.MultiheadAttention):
                attentions.append(child)
                continue
            try:
                source = textwrap.dedent(inspect.getsource(type(child).forward))
            except (OSError, TypeError):
                source = ""
            if "scaled_dot_product_attention" in source or (
                "softmax" in source and any(token in source for token in ("matmul", "@", "einsum"))
            ):
                attentions.append(child)
        mlps = [
            child for child in children
            if child not in attentions and not isinstance(child, norm_types)
            and sum(isinstance(item, nn.Linear) for item in child.modules()) >= 2
        ]
        if len(norms) == 2 and len(attentions) == len(mlps) == 1:
            return norms, attentions[0], mlps[0]
        return None

    patches, blocks, positions = [], [], []
    p, d = args["patch_size"], args["embed_dim"]
    n = math.prod(size // p for size in args["img_size"])
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (p, p) and module.stride == (p, p) and module.out_channels == d:
            patches.append((name, module))
        elif isinstance(module, nn.Linear) and module.out_features == d and module.in_features in (
                args["in_channels"] * p * p, args["in_channels"] * args["history"] * p * p):
            patches.append((name, module))
        parts = block_parts(module)
        if parts:
            blocks.append((name, module, parts))
    # A block's attention/MLP projections can coincidentally have patch-sized
    # dimensions, but they are not the patch embedding before the blocks.
    patches = [
        (name, module) for name, module in patches
        if not any(path and name.startswith(f"{path}.") for path, _, _ in blocks)
    ]
    for name, parameter in model.named_parameters():
        shape, owner_path = tuple(parameter.shape), name.rpartition(".")[0]
        owner = model.get_submodule(owner_path) if owner_path else model
        spatial_table = ((len(shape) in (2, 3) and shape[-1] == d and n in shape[:-1])
                         or (len(shape) == 4 and parameter.numel() == n * d))
        if spatial_table and (owner is model or isinstance(owner, nn.Embedding)) and parameter.requires_grad:
            positions.append((name, parameter))
    if not patches or len(blocks) < 2 or not positions:
        raise EvaluationUnresolved("UNRESOLVED: patch/block/spatial-table representation could not be identified; functional/alternative forms need evidence, not fixed attribute names")
    return patches, blocks, positions


def check_model_structure(config: dict, caller_globals=None, caller_locals=None):
    """Base structure flow with config-free construction and Stormer roles patched."""
    def structure_rules(model, args, patches, blocks):
        def validate(value):
            if isinstance(value, str) and value.startswith("$") and value[1:] not in args:
                raise ValueError(f"Unknown structure rule model argument: {value}")
            if isinstance(value, list):
                for item in value:
                    validate(item)
            elif isinstance(value, dict):
                for item in value.values():
                    validate(item)

        templates = config.get("rules")
        if not isinstance(templates, list) or not templates:
            raise ValueError("model_structure.params.rules must be a nonempty list")
        module_paths = {id(module): path for path, module in model.named_modules()}
        targets = {
            "patch_projection": patches,
            "block_norm": [],
            "block_mlp_input": [],
            "block_mlp_output": [],
        }
        for _path, _block, (norms, _attention, mlp) in blocks:
            targets["block_norm"].extend(
                (module_paths[id(norm)], norm) for norm in norms if isinstance(norm, nn.LayerNorm)
            )
            linears = [module for module in mlp.modules() if isinstance(module, nn.Linear)]
            targets["block_mlp_input"].append((module_paths[id(linears[0])], linears[0]))
            targets["block_mlp_output"].append((module_paths[id(linears[-1])], linears[-1]))

        configured_roles, rules = set(), []
        covered_targets = {role: set() for role in targets}
        module_types = {"Conv2d": nn.Conv2d, "Linear": nn.Linear}
        for template in templates:
            if not isinstance(template, dict) or not template.get("id") or not template.get("type"):
                raise ValueError("Each structure rule needs an id and Base rule type")
            params = template.get("params")
            if not isinstance(params, dict):
                raise ValueError(f"Structure rule {template['id']} needs params")
            role = params.get("target_role")
            if role not in targets:
                raise ValueError(f"Unknown structure rule target_role: {role!r}")
            configured_roles.add(role)
            when = params.get("when_module_type")
            if when is not None and when not in module_types:
                raise ValueError(f"Unknown structure rule when_module_type: {when!r}")
            for index, (path, module) in enumerate(targets[role]):
                if when is not None and not isinstance(module, module_types[when]):
                    continue
                rule = copy.deepcopy({key: value for key, value in template.items() if key != "params"})
                rule["id"] = f"{template['id']}:{index}"
                raw = {
                    key: value for key, value in params.items()
                    if key not in {"target_role", "when_module_type"}
                }
                validate(raw)
                rule["params"] = _resolve_args(raw, args)
                rule["params"]["target_path"] = path
                rules.append(rule)
                covered_targets[role].add(id(module))
        required = {"patch_projection", "block_norm", "block_mlp_input", "block_mlp_output"}
        if required - configured_roles or not rules:
            raise ValueError(f"Incomplete model structure rules: {sorted(required - configured_roles)}")
        for role, members in targets.items():
            if any(id(module) not in covered_targets[role] for _path, module in members):
                raise ValueError(f"No applicable model structure rule for target_role={role!r}")
        return rules

    templates = config.get("rules", [])
    checkpoint_names = ["model_structure:build_model_ok"]
    checkpoint_names += [f"model_structure:{rule['id']}" for rule in templates]
    checkpoint_names += [
        "stormer_structure:transformer_blocks",
        "stormer_structure:learned_spatial_position",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update(name: str, passed: bool, detail: object):
        index = checkpoint_names.index(name)
        checks[index] = (name, bool(passed), str(detail))

    try:
        args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        if isinstance(args.get("img_size"), list):
            args["img_size"] = tuple(args["img_size"])
        module = _load_candidate_modules(config.get("model_file", "stormer.py"))
        model_class = _load_class(module, config.get("model_class", "Stormer"))
        model = model_class(**_resolve_args(args, {}))
        if nn is None or not isinstance(model, nn.Module):
            raise EvaluationUnresolved(
                "UNRESOLVED: Torch-only dynamic adapter; Prompt does not require nn.Module"
            )
        model = model.to(DEVICE).eval()
        update(
            "model_structure:build_model_ok",
            True,
            f"Model built from {config.get('model_file', 'stormer.py')} with inline model_args",
        )

        patches, blocks, positions = _components(model, args)
        rules = structure_rules(model, args, patches, blocks)
        checker = ModelStructureChecker()
        checker.add_rules_from_rules(rules)
        rule_results = checker.check(model)
        for template in templates:
            prefix = f"{template['id']}:"
            matched = [result for result in rule_results if result.rule_id.startswith(prefix)]
            conditional = template.get("params", {}).get("when_module_type") is not None
            update(
                f"model_structure:{template['id']}",
                (conditional and not matched) or (bool(matched) and all(item.passed for item in matched)),
                "; ".join(f"{item.rule_id}={item.passed}: {item.message}" for item in matched)
                or "Conditional structure rule not applicable to the identified patch projection",
            )
        update(
            "stormer_structure:transformer_blocks",
            len(blocks) == args["depth"],
            f"expected depth={args['depth']}, identified={[item[0] for item in blocks]}",
        )
        update(
            "stormer_structure:learned_spatial_position",
            True,
            f"identified={[item[0] for item in positions]}",
        )
    except Exception:
        first_pending = next(
            (
                index for index, item in enumerate(checks)
                if item[2] == "Not executed due to earlier failure"
            ),
            len(checks) - 1,
        )
        checks[first_pending] = (
            checkpoint_names[first_pending], False, traceback.format_exc()
        )
    return checks


def check_model_forward_executable(config: dict, caller_globals=None, caller_locals=None):
    """Check Prompt-specific layouts/rejections that Base forward cannot express."""
    names = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
        "stormer_forward:reject_history_contract",
        "stormer_forward:reject_channel_contract",
        "stormer_forward:reject_grid",
        "stormer_forward:reject_same_patch_count_grid",
        "stormer_forward:patch_divisibility",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in names]
    try:
        args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        if isinstance(args.get("img_size"), list):
            args["img_size"] = tuple(args["img_size"])
        module = _load_candidate_modules(config.get("model_file", "stormer.py"))
        model_class = _load_class(module, config.get("model_class", "Stormer"))
        model = model_class(**_resolve_args(args, {}))
        if nn is None or not isinstance(model, nn.Module):
            raise EvaluationUnresolved(
                "UNRESOLVED: Torch-only dynamic adapter; Prompt does not require nn.Module"
            )
        model = model.to(DEVICE).eval()
        h, w = args["img_size"]
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        inputs, targets = _move_tensors_to_device((
            torch.randn(
                2, args["history"], args["in_channels"], h, w, generator=generator
            ),
            torch.randn(2, args["out_channels"], h, w, generator=generator),
        ))
        layout, value, output = _select_layout(model, inputs, targets)
        checks[0] = (names[0], True, f"Forward executed with Prompt-permitted {layout} layout")
        checks[1] = (
            names[1], True,
            f"shape={tuple(output.shape)}, expected={tuple(targets.shape)}, finite=True",
        )
        h, w = args["img_size"]
        if layout == "5D":
            invalid = {
                "history_contract": torch.randn(2, args["history"] + 1, args["in_channels"], h, w, device=value.device),
                "channel_contract": torch.randn(2, args["history"], args["in_channels"] + 1, h, w, device=value.device),
                "grid": torch.randn(2, args["history"], args["in_channels"], h, w + args["patch_size"], device=value.device),
                "same_patch_count_grid": torch.randn(2, args["history"], args["in_channels"], h // 2, w * 2, device=value.device),
            }
        else:
            channels = args["history"] * args["in_channels"]
            combined = torch.randn(2, channels + 1, h, w, device=value.device)
            invalid = {
                "history_contract": combined,
                "channel_contract": combined,
                "grid": torch.randn(2, channels, h, w + args["patch_size"], device=value.device),
                "same_patch_count_grid": torch.randn(2, channels, h // 2, w * 2, device=value.device),
            }
        for index, (label, bad) in enumerate(invalid.items(), start=2):
            rejected, message = False, "invalid input was accepted"
            try:
                with torch.no_grad():
                    model(bad)
            except (MemoryError, OSError, ImportError) as exc:
                raise EvaluationUnresolved(
                    f"UNRESOLVED: infrastructure failure during probe: {exc}"
                ) from exc
            except Exception as exc:
                rejected, message = True, f"{type(exc).__name__}: {exc}"
            checks[index] = (names[index], bool(rejected), message)

        rejected = False
        try:
            irregular = type(model)(**dict(args, img_size=(h - 1, w))).to(value.device).eval()
            shape = (
                (2, args["history"], args["in_channels"], h - 1, w)
                if layout == "5D"
                else (2, args["history"] * args["in_channels"], h - 1, w)
            )
            with torch.no_grad():
                irregular(torch.randn(*shape, device=value.device, dtype=value.dtype))
        except (MemoryError, OSError, ImportError) as exc:
            raise EvaluationUnresolved(
                f"UNRESOLVED: constructor infrastructure failure: {exc}"
            ) from exc
        except Exception:
            rejected = True
        checks[6] = (names[6], bool(rejected), "non-divisible grid rejected")
    except Exception:
        first_pending = next(
            (index for index, item in enumerate(checks) if item[2] == "Not executed due to earlier failure"),
            len(checks) - 1,
        )
        checks[first_pending] = (names[first_pending], False, traceback.format_exc())
    return checks


def _check_stormer_model_semantics(config: dict, artifact_path=None):
    """Check Stormer-only dataflow semantics as a fixed set of leaf results."""
    args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
    if isinstance(args.get("img_size"), list):
        args["img_size"] = tuple(args["img_size"])
    names = [
        "stormer_usage:patch_embedding",
        "stormer_usage:transformer_blocks",
        "stormer_usage:pre_norm_residual",
        "stormer_usage:position_embedding",
    ] + [f"stormer_usage:history_frame_{index}" for index in range(args["history"])]
    checks = [(name, False, "Not executed due to earlier failure") for name in names]

    def changed(left: torch.Tensor, right: torch.Tensor) -> bool:
        return left.shape == right.shape and not torch.allclose(
            left, right, atol=1e-7, rtol=1e-6
        )

    def perturbation_like(value: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
        pattern = torch.linspace(
            -scale,
            scale,
            value.numel(),
            device=value.device,
            dtype=value.dtype,
        )
        return pattern.reshape_as(value)

    def attention_output(attention, tokens):
        result = (
            attention(tokens, tokens, tokens, need_weights=False)
            if isinstance(attention, nn.MultiheadAttention)
            else attention(tokens)
        )
        result = result[0] if isinstance(result, tuple) else result
        if not isinstance(result, torch.Tensor):
            raise TypeError("attention output is not a tensor")
        return result

    def record(index, probe, detail):
        try:
            checks[index] = (names[index], bool(probe()), str(detail))
        except Exception as exc:
            checks[index] = (names[index], False, f"{type(exc).__name__}: {exc}")

    try:
        module = _load_candidate_modules(config.get("model_file", "stormer.py"))
        model_class = _load_class(module, config.get("model_class", "Stormer"))
        model = model_class(**_resolve_args(args, {}))
        if nn is None or not isinstance(model, nn.Module):
            raise EvaluationUnresolved(
                "UNRESOLVED: Torch-only dynamic adapter; Prompt does not require nn.Module"
            )
        model = model.to(DEVICE).eval()
        h, w = args["img_size"]
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        inputs, targets = _move_tensors_to_device((
            torch.randn(
                2, args["history"], args["in_channels"], h, w, generator=generator
            ),
            torch.randn(2, args["out_channels"], h, w, generator=generator),
        ))
        layout, value, baseline = _select_layout(model, inputs, targets)
        patches, blocks, positions = _components(model, args)
    except Exception:
        checks[0] = (names[0], False, traceback.format_exc())
        return checks

    def intervene(module):
        called = []
        def hook(_module, _arguments, output):
            called.append(True)
            if isinstance(output, tuple):
                return (output[0] + perturbation_like(output[0]), *output[1:])
            return output + perturbation_like(output)
        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                result = model(value)
        finally:
            handle.remove()
        if called and not changed(baseline, result):
            raise EvaluationUnresolved("UNRESOLVED: component ran but numerical intervention is inconclusive")
        return bool(called) and changed(baseline, result)

    record(
        0,
        lambda: bool(patches) and all(intervene(module) for _, module in patches),
        f"modules={[path for path, _ in patches]}",
    )
    record(
        1,
        lambda: len(blocks) == args["depth"] and all(
            intervene(block) for _, block, _ in blocks
        ),
        f"blocks={[path for path, _, _ in blocks]}",
    )

    def residual_equations():
        outcomes = []
        for path, block, (norms, attention, mlp) in blocks:
            tokens = torch.randn(
                2, 4, args["embed_dim"], device=value.device,
                dtype=next(block.parameters()).dtype,
            )
            with torch.no_grad():
                actual, equivalent = block(tokens), False
                for first, second in (norms, list(reversed(norms))):
                    residual = tokens + attention_output(attention, first(tokens))
                    equivalent |= torch.allclose(
                        actual, residual + mlp(second(residual)), atol=1e-6, rtol=1e-5
                    )
            outcomes.append((path, equivalent))
        return len(outcomes) == args["depth"] and all(item[1] for item in outcomes)

    record(
        2,
        residual_equations,
        "actual block equations, independent of registration order and attribute names",
    )

    def positions_used():
        outcomes = []
        for path, position in positions:
            saved = position.detach().clone()
            try:
                with torch.no_grad():
                    position.add_(perturbation_like(position))
                    varied_output = model(value)
            finally:
                with torch.no_grad():
                    position.copy_(saved)
            with torch.enable_grad():
                prediction = model(value)
                gradient = torch.autograd.grad(prediction, position,
                    grad_outputs=perturbation_like(prediction), allow_unused=True)[0] if prediction.requires_grad else None
            used = changed(baseline, varied_output)
            if not used and gradient is not None:
                raise EvaluationUnresolved(f"UNRESOLVED: connected learned table {path} intervention is inconclusive")
            outcomes.append((path, used and gradient is not None))
        return bool(outcomes) and all(item[1] for item in outcomes)

    record(3, positions_used, f"parameters={[path for path, _ in positions]}")

    for frame in range(inputs.shape[1]):
        def frame_usage(frame_index=frame):
            varied = inputs.clone()
            varied[:, frame_index] += perturbation_like(varied[:, frame_index], 0.5)
            selected = varied if layout == "5D" else varied.flatten(1, 2)
            with torch.no_grad():
                result = model(selected)
            if not changed(baseline, result):
                raise EvaluationUnresolved(
                    f"UNRESOLVED: history frame {frame_index} requires dataflow confirmation"
                )
            return True
        record(4 + frame, frame_usage, f"selected layout={layout}")
    return checks


def _check_stormer_script_structure(config: dict, artifact_path=None):
    """Verify one phase-local behavior path without imposing a fixed entry name."""
    phase = config.get("phase")
    leaf_names = {
        "train_script": "train_script:behavior_path_structure",
        "infer_script": "infer_script:behavior_path_structure",
    }
    name = leaf_names[phase] if phase in leaf_names else f"{phase}:behavior_path_structure"
    try:
        if phase not in {"train_script", "infer_script"}:
            raise ValueError("phase must be train_script or infer_script")
        files = config.get("files", [])
        if isinstance(files, str):
            files = [files]
        if len(files) != 1:
            raise ValueError("each script structure check requires exactly its own phase file")
        module = _load_candidate_modules(files[0])
        _entry, function = _discover_entry(
            module, CANDIDATE_DIR / files[0], phase == "train_script"
        )
        return [(
            name,
            True,
            f"source path={function.name}; dynamic behavior checked separately",
        )]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_train_structure(config: dict, artifact_path=None):
    """Adapt Base's train-structure checkpoint to a callable training step."""
    if config.get("phase") != "train_script":
        raise ValueError("train structure requires train_script phase")
    return _check_stormer_script_structure(config, artifact_path)


def check_infer_structure(config: dict, artifact_path=None):
    """Adapt Base's infer-structure checkpoint to in-memory evaluation."""
    if config.get("phase") != "infer_script":
        raise ValueError("infer structure requires infer_script phase")
    return _check_stormer_script_structure(config, artifact_path)


def _training_evidence(config):
    root = CANDIDATE_DIR
    module = _load_candidate_modules(config.get("train_file", "train.py"))
    entry, function = _discover_entry(module, root / config.get("train_file", "train.py"), True)
    model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
    if isinstance(model_args.get("img_size"), list):
        model_args["img_size"] = tuple(model_args["img_size"])
    model_module = _load_candidate_modules(config.get("model_file", "stormer.py"))
    model_class = _load_class(model_module, config.get("model_class", "Stormer"))
    model = model_class(**_resolve_args(model_args, {}))
    if nn is None or not isinstance(model, nn.Module):
        raise EvaluationUnresolved(
            "UNRESOLVED: Torch-only dynamic adapter; Prompt does not require nn.Module"
        )
    model = model.to(DEVICE).train()
    h, w = model_args["img_size"]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    inputs, targets = _move_tensors_to_device((
        torch.randn(
            2, model_args["history"], model_args["in_channels"], h, w,
            generator=generator,
        ),
        torch.randn(2, model_args["out_channels"], h, w, generator=generator),
    ))
    _, inputs, _ = _select_layout(model.eval(), inputs, targets)
    model.train()
    predictions, roots, gradients, steps = [], [], [], []
    parameters = [p for p in model.parameters() if p.requires_grad]
    original_backward, aliases = torch.autograd.backward, []
    old_classes = {name: getattr(torch.optim, name) for name in ("AdamW", "Adam", "SGD")}
    def forward_hook(_model, _inputs, output):
        predictions.append(output)
        if output.requires_grad:
            output.register_hook(lambda grad: gradients.append((output, grad.detach().clone())))
    def backward(tensors, *args, **kwargs):
        observed = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
        roots.extend((tensor, _scalar(tensor)) for tensor in observed if tensor.numel() == 1)
        return original_backward(tensors, *args, **kwargs)
    def tracked_class(name, original):
        class Tracking(original):
            def step(self, *args, **kwargs):
                ids = {id(p) for group in self.param_groups for p in group["params"]}
                result = super().step(*args, **kwargs)
                steps.append((name, bool(ids & {id(p) for p in parameters})))
                return result
        return Tracking
    handle = model.register_forward_hook(forward_hook)
    try:
        for name, original in old_classes.items():
            replacement = tracked_class(name, original)
            for key, value in list(vars(module).items()):
                if value is original:
                    aliases.append((key, value))
                    setattr(module, key, replacement)
            setattr(torch.optim, name, replacement)
        torch.autograd.backward = backward
        args, kwargs, injected = _bind_entry(entry, function, model, inputs, targets)
        entry(*args, **kwargs)
    finally:
        handle.remove()
        torch.autograd.backward = original_backward
        for name, original in old_classes.items():
            setattr(torch.optim, name, original)
        for key, value in aliases:
            setattr(module, key, value)
    expected = [_compute_mse(pred.detach(), targets)
                for pred in predictions if tuple(pred.shape) == tuple(targets.shape)]
    mse_value = bool(roots and expected) and all(any(math.isclose(value, oracle, rel_tol=1e-5, abs_tol=1e-6)
                    for oracle in expected) for _, value in roots)
    connected = bool(gradients) and any(torch.isfinite(grad).all() and bool(grad.abs().sum() > 0)
                                       for _, grad in gradients)
    # No inference from an arbitrary fitted positive gradient scale. Complex
    # autograd/scaled objectives remain an evidence limitation pending review.
    return {"entry": function.name, "roots": len(roots), "mse_value": mse_value,
            "connected_backward": connected, "steps": steps,
            "injected_optimizer": injected}


def _check_stormer_train_semantics(config: dict, artifact_path=None):
    name = "stormer_training:adamw_executed"
    try:
        evidence = _training_evidence(config)
        if evidence["injected_optimizer"]:
            raise EvaluationUnresolved(
                "UNRESOLVED: fixture supplies AdamW, proving use but not train.py's own optimizer selection"
            )
        passed = bool(evidence["steps"]) and all(
            optimizer == "AdamW" and live for optimizer, live in evidence["steps"]
        )
        return [(name, bool(passed), str(evidence))]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_method_correctness(config: dict, artifact_path=None):
    """Check dataflow or optimizer behavior; Base's named-call oracle cannot observe either."""
    rule = (config.get("phase"), config.get("task_rule"))
    if rule == ("model_structure", "component_usage"):
        return _check_stormer_model_semantics(config, artifact_path)
    if rule == ("train_script", "optimizer"):
        return _check_stormer_train_semantics(config, artifact_path)
    raise ValueError(f"unsupported Stormer method rule: {rule!r}")


def check_train_loss(config: dict, artifact_path=None):
    name = "stormer_training:mse_in_backward_path"
    try:
        evidence = _training_evidence(config)
        if not evidence["roots"]:
            raise EvaluationUnresolved(
                "UNRESOLVED: no observable backward loss root; alternative autograd requires adaptation"
            )
        passed = evidence["mse_value"] and evidence["connected_backward"]
        return [(name, bool(passed), str(evidence))]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_smoke_training(config: dict, caller_globals=None, caller_locals=None):
    """Base smoke-training flow with candidate-defined entry discovery patched."""
    checkpoint_names = ["training:smoke_test"]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in checkpoint_names]

    def update(index: int, passed: bool, detail: object):
        checks[index] = (checkpoint_names[index], bool(passed), str(detail))

    current_index = 0
    try:
        root = CANDIDATE_DIR
        train_file = config.get("run_file", config.get("train_file", "train.py"))
        train_module = _load_candidate_modules(train_file)
        entry, function = _discover_entry(train_module, root / train_file, True)

        work_dir = (root / "smoke-training").resolve()
        checkpoint_dir, results_dir = work_dir / "checkpoints", work_dir / "results"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        runtime_config = {}
        config_name = config.get("config_name")
        if config_name:
            runtime_config.update(_load_candidate_config(config_name))
        context = {"checkpoint_dir": checkpoint_dir, "results_dir": results_dir}
        runtime_config.update(_resolve_args(config.get("smoke_config", {}), context))
        context["config"] = runtime_config
        context.update(runtime_config)

        model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        if isinstance(model_args.get("img_size"), list):
            model_args["img_size"] = tuple(model_args["img_size"])
        model_module = _load_candidate_modules(config.get("model_file", "stormer.py"))
        model_class = _load_class(model_module, config.get("model_class", "Stormer"))
        model = model_class(**_resolve_args(model_args, context))
        if nn is None or not isinstance(model, nn.Module):
            raise EvaluationUnresolved(
                "UNRESOLVED: Torch-only dynamic adapter; Prompt does not require nn.Module"
            )
        model = model.to(DEVICE).train()
        h, w = model_args["img_size"]
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        inputs, targets = _move_tensors_to_device((
            torch.randn(
                2, model_args["history"], model_args["in_channels"], h, w,
                generator=generator,
            ),
            torch.randn(2, model_args["out_channels"], h, w, generator=generator),
        ))
        _, inputs, _ = _select_layout(model.eval(), inputs, targets)
        model.train()
        args, kwargs, _ = _bind_entry(entry, function, model, inputs, targets)
        entry(*args, **kwargs)
        update(current_index, True, f"Smoke test completed: {function.name}(...)")
    except Exception:
        update(current_index, False, traceback.format_exc())
    return checks


def _evaluation_evidence(config):
    def metric_name(value):
        text = "".join(character for character in str(value).lower() if character.isalnum())
        return {
            "mae": "mae", "meanabsoluteerror": "mae", "rmse": "rmse",
            "rootmeansquarederror": "rmse",
        }.get(text)

    def aliases(function):
        result = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                result.update({target.id: node.value for target in node.targets if isinstance(target, ast.Name)})
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                result[node.target.id] = node.value
        return result

    def metric_intent(expr, known):
        def expanded(value, visited):
            nodes = list(ast.walk(value))
            for node in list(nodes):
                if isinstance(node, ast.Name) and node.id in known and node.id not in visited:
                    visited.add(node.id)
                    nodes.extend(expanded(known[node.id], visited))
            return nodes

        nodes = expanded(expr, set())
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            if metric_name(name):
                return metric_name(name)
        calls = {
            (node.func.id if isinstance(node.func, ast.Name) else node.func.attr).lower()
            for node in nodes
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        if "sqrt" in calls or any(
            isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Constant) and node.right.value == 0.5
            for node in nodes
        ):
            return "rmse"
        return "mae" if {"abs", "absolute"} & calls else None

    def metric_values(expr, value, known):
        result = []
        if isinstance(value, dict):
            result.extend((label, item) for key, item in value.items() if (label := metric_name(key)))
        if isinstance(expr, ast.Name) and expr.id in known:
            expr = known[expr.id]
        if isinstance(expr, ast.Dict) and isinstance(value, dict):
            for key, expression in zip(expr.keys, expr.values):
                if isinstance(key, ast.Constant) and key.value in value:
                    label = metric_name(key.value) or metric_intent(expression, known)
                    if label:
                        result.append((label, value[key.value]))
        elif isinstance(expr, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
            for expression, item in zip(expr.elts, value):
                label = (
                    metric_name(expression.id) if isinstance(expression, ast.Name) else None
                ) or metric_intent(expression, known)
                if label:
                    result.append((label, item))
        else:
            label = metric_intent(expr, known)
            if label:
                result.append((label, value))
        return result

    root, filename = CANDIDATE_DIR, config.get("evaluate_file", "evaluate.py")
    module = _load_candidate_modules(filename)
    entry, function = _discover_entry(module, root / filename, False)
    model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
    if isinstance(model_args.get("img_size"), list):
        model_args["img_size"] = tuple(model_args["img_size"])
    model_module = _load_candidate_modules(config.get("model_file", "stormer.py"))
    model_class = _load_class(model_module, config.get("model_class", "Stormer"))
    model = model_class(**_resolve_args(model_args, {}))
    if nn is None or not isinstance(model, nn.Module):
        raise EvaluationUnresolved(
            "UNRESOLVED: Torch-only dynamic adapter; Prompt does not require nn.Module"
        )
    model = model.to(DEVICE).eval()
    h, w = model_args["img_size"]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    inputs, targets = _move_tensors_to_device((
        torch.randn(
            2, model_args["history"], model_args["in_channels"], h, w,
            generator=generator,
        ),
        torch.randn(2, model_args["out_channels"], h, w, generator=generator),
    ))
    _, inputs, _ = _select_layout(model, inputs, targets)
    args, kwargs, _ = _bind_entry(entry, function, model, inputs, targets)
    predictions, observed, source_cache = [], [], {}
    def hook(_model, _inputs, output):
        predictions.append(output.detach().clone())
    def trace(frame, event, value):
        filename = Path(frame.f_code.co_filename).resolve()
        if not filename.is_relative_to(root):
            return None
        if event == "return":
            if filename not in source_cache:
                tree = ast.parse(filename.read_text(encoding="utf-8"), filename=str(filename))
                source_cache[filename] = [n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            matches = [n for n in source_cache[filename] if n.name == frame.f_code.co_name
                       and n.lineno <= frame.f_lineno <= getattr(n, "end_lineno", frame.f_lineno)]
            for node in matches:
                known = aliases(node)
                for name, item in frame.f_locals.items():
                    if label := metric_name(name):
                        observed.append((label, item))
                for statement in ast.walk(node):
                    if isinstance(statement, ast.Return) and statement.value is not None and statement.lineno <= frame.f_lineno <= statement.end_lineno:
                        observed.extend(metric_values(statement.value, value, known))
        return trace
    handle, previous_trace = model.register_forward_hook(hook), sys.gettrace()
    try:
        sys.settrace(trace)
        with torch.no_grad():
            entry(*args, **kwargs)
    finally:
        sys.settrace(previous_trace)
        handle.remove()
    if len(predictions) != 1 or tuple(predictions[0].shape) != tuple(targets.shape):
        raise EvaluationUnresolved("UNRESOLVED: prediction/target correspondence is ambiguous")
    expected = {"mae": _compute_mae(predictions[0], targets),
                "rmse": _compute_rmse(predictions[0], targets)}
    metrics = {"mae": [], "rmse": []}
    for label, value in observed:
        try:
            metrics[label].append(_scalar(value))
        except TypeError:
            pass  # Per-channel arrays are not global scalar metrics.
    if any(not values for values in metrics.values()):
        raise EvaluationUnresolved("UNRESOLVED: actual metrics cannot be associated with source; no dict/key requirement")
    return {"entry": function.name, "metrics": metrics, "expected": expected}


def check_infer_execution(config: dict, artifact_path=None):
    name = "stormer_inference:execution"
    try:
        root, filename = CANDIDATE_DIR, config.get("evaluate_file", "evaluate.py")
        module = _load_candidate_modules(filename)
        entry, function = _discover_entry(module, root / filename, False)
        model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        if isinstance(model_args.get("img_size"), list):
            model_args["img_size"] = tuple(model_args["img_size"])
        model_module = _load_candidate_modules(config.get("model_file", "stormer.py"))
        model_class = _load_class(model_module, config.get("model_class", "Stormer"))
        model = model_class(**_resolve_args(model_args, {}))
        if nn is None or not isinstance(model, nn.Module):
            raise EvaluationUnresolved(
                "UNRESOLVED: Torch-only dynamic adapter; Prompt does not require nn.Module"
            )
        model = model.to(DEVICE).eval()
        h, w = model_args["img_size"]
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        inputs, targets = _move_tensors_to_device((
            torch.randn(
                2, model_args["history"], model_args["in_channels"], h, w,
                generator=generator,
            ),
            torch.randn(2, model_args["out_channels"], h, w, generator=generator),
        ))
        _, inputs, _ = _select_layout(model, inputs, targets)
        args, kwargs, _ = _bind_entry(entry, function, model, inputs, targets)
        with torch.no_grad():
            entry(*args, **kwargs)
        return [(name, True, f"candidate entry={function.name}; return schema unrestricted")]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_infer_output_validity(config: dict, artifact_path=None):
    names = ["stormer_evaluation:mae_validity", "stormer_evaluation:rmse_validity"]
    checks = [(name, False, "Not executed due to earlier failure") for name in names]
    try:
        evidence = _evaluation_evidence(config)
        return [
            (
                f"stormer_evaluation:{metric}_validity",
                bool(all(math.isfinite(value) and value >= 0 for value in values)),
                f"actual={values}; no key/dtype/file contract",
            )
            for metric, values in evidence["metrics"].items()
        ]
    except Exception:
        checks[0] = (names[0], False, traceback.format_exc())
        return checks


def check_infer_metrics(config: dict, artifact_path=None):
    names = ["stormer_evaluation:mae_formula", "stormer_evaluation:rmse_formula"]
    checks = [(name, False, "Not executed due to earlier failure") for name in names]
    try:
        evidence = _evaluation_evidence(config)
        return [
            (
                f"stormer_evaluation:{metric}_formula",
                bool(all(
                    math.isfinite(value) and math.isclose(
                        value, evidence["expected"][metric], rel_tol=1e-5, abs_tol=1e-6
                    )
                    for value in values
                )),
                f"actual={values}, oracle={evidence['expected'][metric]}; entry={evidence['entry']}",
            )
            for metric, values in evidence["metrics"].items()
        ]
    except Exception:
        checks[0] = (names[0], False, traceback.format_exc())
        return checks
