#!/usr/bin/env python
"""Standardized StormCast checks with Base-first reuse.

Complete generic checks are exported directly from BaseEvaluator. Adapted
checks retain the applicable Base skeleton but remove requirements absent from
the Prompt, such as a mandatory config file or file-based inference output.
StormCast-only behavior remains local: conditional/timestep use, iterative
sampling, AdamW selection, noise-MSE correctness, and MAE/RMSE formulas.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import pickle
import sys
import traceback
import types
from pathlib import Path
from typing import Any
from unittest import mock

import torch
from torch import nn

_EVALUATOR_DIR = Path(__file__).resolve().parent
for _parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    _base_dir = _parent / 'benchmarks' / 'evaluators' / 'base'
    if _base_dir.is_dir() and str(_base_dir) not in sys.path:
        sys.path.insert(0, str(_base_dir))

from base import base_evaluator
from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _execute_hooks,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _load_reference_modules,
    _move_tensors_to_device,
    _resolve_args,
    check_files,
    check_infer_output_schema,
    check_interface,
    check_parameter_update,
)
from base.model_structure_checker import ModelStructureChecker

TASK_NAME = STANDARD_NAME = "stormcast_paper_repro"
DEFAULT_MODEL_ARGS = {
    "in_channels": 1,
    "cond_channels": 2,
    "base_channels": 8,
    "num_steps": 20,
}
SEED = 20260819


def _components(model: nn.Module, args: dict[str, Any]):
    named = list(model.named_modules())
    input_convs = [
        (name, module) for name, module in named
        if isinstance(module, nn.Conv2d)
        and (module.in_channels, module.out_channels)
        == (args["in_channels"], args["base_channels"])
    ]
    condition_convs = [
        (name, module) for name, module in named
        if isinstance(module, nn.Conv2d)
        and (module.in_channels, module.out_channels)
        == (args["cond_channels"], args["base_channels"])
    ]
    output_convs = [
        (name, module) for name, module in named
        if isinstance(module, nn.Conv2d)
        and (module.in_channels, module.out_channels)
        == (args["base_channels"], args["in_channels"])
    ]
    time_candidates = []
    for name, module in named:
        if module is model:
            continue
        linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
        embeddings = [child for child in module.modules() if isinstance(child, nn.Embedding)]
        convolutions = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
        if (linears or embeddings) and not convolutions:
            time_candidates.append((name, module))
    candidate_names = {name for name, _ in time_candidates}
    time_candidates = [
        (name, module) for name, module in time_candidates
        if not any(
            name.startswith(parent + ".")
            for parent in candidate_names if parent != name
        )
    ]
    block_lists = []
    for name, module in named:
        if not isinstance(module, nn.ModuleList) or not len(module):
            continue
        valid = []
        for block in module:
            children = list(block.modules())
            valid.append(
                any(isinstance(child, nn.Conv2d) for child in children)
                and any(
                    isinstance(child, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d))
                    for child in children
                )
                and any(isinstance(child, (nn.GELU, nn.ReLU, nn.SiLU)) for child in children)
            )
        if all(valid):
            block_lists.append((name, module))
    return input_convs, condition_convs, time_candidates, block_lists, output_convs


def check_model_structure(config: dict, caller_globals=None, caller_locals=None):
    """Base structure check with config-free construction and StormCast roles."""
    def structure_rules(model, components):
        templates = config.get("rules")
        if not isinstance(templates, list) or not templates:
            raise ValueError("model_structure.params.rules must be a nonempty list")
        args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        input_convs, condition_convs, _, block_lists, output_convs = components
        paths = {id(module): path for path, module in model.named_modules()}
        residual_convs = [
            (paths[id(child)], child)
            for _, blocks in block_lists for block in blocks
            for child in block.modules() if isinstance(child, nn.Conv2d)
        ]
        targets = {
            "input_projection": input_convs,
            "conditioning_projection": condition_convs,
            "residual_convolution": residual_convs,
            "output_projection": output_convs,
        }
        rules, configured = [], set()
        for template in templates:
            params = template.get("params")
            if not isinstance(params, dict):
                raise ValueError("each structure rule needs params")
            role = params.get("target_role")
            if role not in targets:
                raise ValueError(f"unknown structure target_role: {role!r}")
            configured.add(role)
            for index, (path, _) in enumerate(targets[role]):
                rule = copy.deepcopy({key: value for key, value in template.items() if key != "params"})
                rule["id"] = f"{template['id']}:{index}"
                raw = {key: value for key, value in params.items() if key != "target_role"}
                rule["params"] = _resolve_args(raw, args)
                rule["params"]["target_path"] = path
                rules.append(rule)
        missing = set(targets) - configured
        empty = [role for role, values in targets.items() if not values]
        if missing or empty:
            raise ValueError(
                f"incomplete StormCast structure roles: missing={sorted(missing)}, empty={empty}"
            )
        return rules

    templates = config.get("rules", [])
    checkpoint_names = ["model_structure:build_model_ok"]
    checkpoint_names += [f"model_structure:{rule['id']}" for rule in templates]
    checkpoint_names += [
        "stormcast_structure:independent_projections",
        "stormcast_structure:residual_blocks",
        "stormcast_structure:timestep_embedding",
    ]
    results = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update(name: str, passed: bool, detail: object):
        index = checkpoint_names.index(name)
        results[index] = (name, bool(passed), str(detail))

    try:
        root = CANDIDATE_DIR
        args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        module = _load_candidate_modules(config.get("model_file", "stormcast.py"))
        model_class = _load_class(module, config.get("model_class", "StormCast"))
        if not issubclass(model_class, nn.Module):
            raise TypeError("StormCast must be a torch.nn.Module class")
        model = model_class(**_resolve_args(args, {})).to(DEVICE).eval()
        update(
            "model_structure:build_model_ok",
            True,
            f"Model built from {config.get('model_file', 'stormcast.py')} with inline model_args",
        )

        components = _components(model, args)
        rules = structure_rules(model, components)
        checker = ModelStructureChecker()
        checker.add_rules_from_rules(rules)
        rule_results = checker.check(model)
        for template in templates:
            prefix = f"{template['id']}:"
            matched = [item for item in rule_results if item.rule_id.startswith(prefix)]
            update(
                f"model_structure:{template['id']}",
                bool(matched) and all(item.passed for item in matched),
                "; ".join(f"{item.rule_id}={item.passed}: {item.message}" for item in matched)
                or "No module matched the configured StormCast role",
            )

        input_convs, condition_convs, time_candidates, block_lists, _ = components
        independent = bool(
            input_convs and condition_convs
            and input_convs[0][1] is not condition_convs[0][1]
            and {id(item) for item in input_convs[0][1].parameters()}.isdisjoint(
                id(item) for item in condition_convs[0][1].parameters()
            )
        )
        model_file = root / config.get("model_file", "stormcast.py")
        tree = ast.parse(model_file.read_text(encoding="utf-8"), filename=str(model_file))
        residual_add = any(
            isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Add)
            for forward in ast.walk(tree)
            if isinstance(forward, (ast.FunctionDef, ast.AsyncFunctionDef)) and forward.name == "forward"
            for node in ast.walk(forward)
        )
        update(
            "stormcast_structure:independent_projections",
            independent,
            f"input={[name for name, _ in input_convs]}, condition={[name for name, _ in condition_convs]}",
        )
        update(
            "stormcast_structure:residual_blocks",
            bool(block_lists) and residual_add,
            f"block_lists={[(name, len(value)) for name, value in block_lists]}",
        )
        update(
            "stormcast_structure:timestep_embedding",
            bool(time_candidates),
            f"candidates={[name for name, _ in time_candidates]}",
        )
    except Exception:
        first_pending = next(
            (
                index for index, item in enumerate(results)
                if item[2] == "Not executed due to earlier failure"
            ),
            len(results) - 1,
        )
        results[first_pending] = (
            checkpoint_names[first_pending], False, traceback.format_exc()
        )
    return results


def check_model_forward_executable(config: dict, caller_globals=None, caller_locals=None):
    """Check Prompt-specific conditioning/timestep validation and output shape."""
    checkpoint_names = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
        "stormcast_forward:input_sensitivity",
        "stormcast_forward:invalid_input_rejection",
    ]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in checkpoint_names]
    current_index = 0
    try:
        args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        module = _load_candidate_modules(config.get("model_file", "stormcast.py"))
        model_class = _load_class(module, config.get("model_class", "StormCast"))
        model = model_class(**_resolve_args(args, {})).to(DEVICE).eval()
        fixture = config.get("fixture", {})
        batch = int(fixture.get("batch_size", 2))
        height = int(fixture.get("height", 8))
        width = int(fixture.get("width", 8))
        generator = torch.Generator(device="cpu").manual_seed(int(fixture.get("seed", SEED)))
        noisy, condition, timestep = _move_tensors_to_device((
            torch.randn(batch, args["in_channels"], height, width, generator=generator),
            torch.randn(batch, args["cond_channels"], height, width, generator=generator),
            torch.tensor([3 + index for index in range(batch)]),
        ))

        def perturb(value: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
            return torch.linspace(
                -scale, scale, value.numel(), device=value.device, dtype=value.dtype
            ).reshape_as(value)

        def changed(left: torch.Tensor, right: torch.Tensor) -> bool:
            return left.shape == right.shape and not torch.allclose(
                left, right, atol=1e-7, rtol=1e-6
            )

        def rejects(callable_) -> bool:
            try:
                callable_()
            except (TypeError, ValueError):
                return True
            return False
        with torch.no_grad():
            output = model(noisy, condition, timestep)
        checks[current_index] = (checkpoint_names[current_index], bool(True), str("Forward pass executed successfully"))

        current_index += 1
        valid = bool(
            isinstance(output, torch.Tensor)
            and output.shape == noisy.shape
            and output.dtype == noisy.dtype
            and torch.isfinite(output).all()
        )
        checks[current_index] = (checkpoint_names[current_index], bool(valid), str(f"shape={tuple(output.shape)}, expected={tuple(noisy.shape)}, dtype={output.dtype}"))

        with torch.no_grad():
            changed_noisy = model(noisy + perturb(noisy), condition, timestep)
            changed_condition = model(noisy, condition + perturb(condition), timestep)
            changed_time = model(noisy, condition, timestep + 5)
        sensitivity = {
            "noisy": changed(output, changed_noisy),
            "conditioning": changed(output, changed_condition),
            "timestep": changed(output, changed_time),
        }
        batch, _, height, width = noisy.shape
        rejection = {
            "rank": rejects(lambda: model(noisy[0], condition, timestep)),
            "batch": rejects(lambda: model(noisy[:1], condition, timestep)),
            "noisy_channels": rejects(lambda: model(
                torch.randn(batch, args["in_channels"] + 1, height, width, device=noisy.device),
                condition, timestep,
            )),
            "conditioning_channels": rejects(lambda: model(
                noisy,
                torch.randn(batch, args["cond_channels"] + 1, height, width, device=noisy.device),
                timestep,
            )),
            "grid": rejects(lambda: model(noisy, condition[:, :, :-1], timestep)),
            "timestep_length": rejects(lambda: model(
                noisy, condition, torch.tensor([1, 2, 3], device=noisy.device)
            )),
        }
        current_index += 1
        checks[current_index] = (
            checkpoint_names[current_index], bool(all(sensitivity.values())), str(sensitivity)
        )
        current_index += 1
        checks[current_index] = (
            checkpoint_names[current_index], bool(all(rejection.values())), str(rejection)
        )
    except Exception:
        checks[current_index] = (checkpoint_names[current_index], bool(False), str(traceback.format_exc()))
    return checks


def _check_stormcast_component_usage(config: dict, artifact_path=None):
    names = [
        "stormcast_usage:noisy_projection",
        "stormcast_usage:conditioning_projection",
        "stormcast_usage:timestep_embedding",
        "stormcast_usage:residual_blocks",
        "stormcast_usage:output_projection",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in names]
    try:
        args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        module = _load_candidate_modules(config.get("model_file", "stormcast.py"))
        model_class = _load_class(module, config.get("model_class", "StormCast"))
        model = model_class(**_resolve_args(args, {})).to(DEVICE).eval()
        components = _components(model, args)
        input_convs, condition_convs, time_candidates, block_lists, output_convs = components
        fixture = config.get("fixture", {})
        batch, height, width = (
            int(fixture.get("batch_size", 2)),
            int(fixture.get("height", 8)),
            int(fixture.get("width", 8)),
        )
        generator = torch.Generator(device="cpu").manual_seed(int(fixture.get("seed", SEED)))
        noisy, condition, timestep = _move_tensors_to_device((
            torch.randn(batch, args["in_channels"], height, width, generator=generator),
            torch.randn(batch, args["cond_channels"], height, width, generator=generator),
            torch.tensor([3 + index for index in range(batch)]),
        ))
        with torch.no_grad():
            baseline = model(noisy, condition, timestep)

        def intervention(module: nn.Module) -> bool:
            def hook(_module, _inputs, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError("component output is not a tensor")
                return output + torch.linspace(
                    -0.25, 0.25, output.numel(), device=output.device, dtype=output.dtype
                ).reshape_as(output)

            handle = module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    changed = model(noisy, condition, timestep)
            finally:
                handle.remove()
            return baseline.shape == changed.shape and not torch.allclose(
                baseline, changed, atol=1e-7, rtol=1e-6
            )

        usage = {
            "noisy_projection": bool(input_convs) and intervention(input_convs[0][1]),
            "conditioning_projection": bool(condition_convs) and intervention(condition_convs[0][1]),
            "timestep_embedding": any(intervention(module) for _, module in time_candidates),
            "residual_blocks": bool(block_lists) and all(
                intervention(block) for block in block_lists[0][1]
            ),
            "output_projection": bool(output_convs) and intervention(output_convs[0][1]),
        }
        return [
            (f"stormcast_usage:{name}", bool(passed), str(usage))
            for name, passed in usage.items()
        ]
    except Exception:
        checks[0] = (names[0], False, traceback.format_exc())
        return checks


def _check_stormcast_sampling(config: dict, artifact_path=None):
    labels = [
        "shape_dtype_finite", "seed_reproducibility", "conditioning_sensitivity",
        "iterative_forward_usage", "invalid_steps_rejection",
    ]
    pending = [
        (f"stormcast_sample:{name}", False, "Not executed due to earlier failure")
        for name in labels
    ]
    try:
        args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        module = _load_candidate_modules(config.get("model_file", "stormcast.py"))
        model_class = _load_class(module, config.get("model_class", "StormCast"))
        model = model_class(**_resolve_args(args, {})).to(DEVICE).eval()
        fixture = config.get("fixture", {})
        batch, height, width = (
            int(fixture.get("batch_size", 2)),
            int(fixture.get("height", 8)),
            int(fixture.get("width", 8)),
        )
        generator = torch.Generator(device="cpu").manual_seed(int(fixture.get("seed", SEED)))
        condition = _move_tensors_to_device(torch.randn(
            batch, args["cond_channels"], height, width, generator=generator
        ))
        steps = int(config.get("sample_steps", 3))
        generator_device = condition.device.type

        def changed(left: torch.Tensor, right: torch.Tensor) -> bool:
            return left.shape == right.shape and not torch.allclose(
                left, right, atol=1e-7, rtol=1e-6
            )

        def generated(seed: int, current_condition: torch.Tensor):
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            with torch.no_grad():
                return model.sample(current_condition, steps=steps, generator=generator)

        first = generated(SEED, condition)
        second = generated(SEED, condition)
        third = generated(SEED + 1, condition)
        expected_shape = (
            condition.shape[0], args["in_channels"],
            condition.shape[-2], condition.shape[-1],
        )
        calls = []
        original_forward = model.forward

        def tracked_forward(self, noisy, cond, timestep):
            calls.append(timestep)
            return original_forward(noisy, cond, timestep)

        model.forward = types.MethodType(tracked_forward, model)
        try:
            generated(SEED, condition)
        finally:
            model.forward = original_forward
        outcomes = {
            "shape_dtype_finite": bool(
                isinstance(first, torch.Tensor)
                and first.shape == expected_shape
                and first.dtype == condition.dtype
                and torch.isfinite(first).all()
            ),
            "seed_reproducibility": torch.equal(first, second) and changed(first, third),
            "conditioning_sensitivity": changed(
                first,
                generated(
                    SEED,
                    condition + torch.linspace(
                        -0.5, 0.5, condition.numel(),
                        device=condition.device, dtype=condition.dtype,
                    ).reshape_as(condition),
                ),
            ),
            "iterative_forward_usage": len(calls) >= steps,
            "invalid_steps_rejection": False,
        }
        try:
            model.sample(
                condition, steps=0,
                generator=torch.Generator(device=generator_device).manual_seed(SEED),
            )
        except (TypeError, ValueError):
            outcomes["invalid_steps_rejection"] = True
        return [
            (f"stormcast_sample:{name}", bool(passed), str(outcomes))
            for name, passed in outcomes.items()
        ]
    except Exception:
        pending[0] = (pending[0][0], False, traceback.format_exc())
        return pending


def _training_evidence(config: dict):
    model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
    model_module = _load_candidate_modules(config.get("model_file", "stormcast.py"))
    train_module = _load_candidate_modules(config.get("train_file", "train.py"))
    cls = _load_class(model_module, config.get("model_class", "StormCast"))
    model = cls(**_resolve_args(model_args, {})).to(DEVICE).train()
    train_step = getattr(train_module, config.get("train_function", "train_step"), None)
    if not callable(train_step):
        raise AttributeError("train_step is missing or not callable")
    fixture = config.get("fixture", {})
    batch, height, width = (
        int(fixture.get("batch_size", 2)),
        int(fixture.get("height", 8)),
        int(fixture.get("width", 8)),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(fixture.get("seed", SEED)))
    noisy, condition, target = _move_tensors_to_device((
        torch.randn(batch, model_args["in_channels"], height, width, generator=generator),
        torch.randn(batch, model_args["cond_channels"], height, width, generator=generator),
        torch.randn(batch, model_args["in_channels"], height, width, generator=generator),
    ))
    original_adamw = torch.optim.AdamW
    aliases = []
    tracker = {"constructed": 0, "parameter_ids": set()}

    class TrackingAdamW(original_adamw):
        def __init__(self, *args, **kwargs):
            tracker["constructed"] += 1
            super().__init__(*args, **kwargs)
            tracker["parameter_ids"].update(
                id(parameter)
                for group in self.param_groups
                for parameter in group["params"]
            )

    captured = {}
    original_forward = model.forward

    def tracked_forward(self, candidate_noisy, candidate_cond, candidate_timestep):
        output = original_forward(candidate_noisy, candidate_cond, candidate_timestep)
        captured["output"] = output.detach().clone()
        return output

    try:
        torch.optim.AdamW = TrackingAdamW
        for name, value in list(vars(train_module).items()):
            if value is original_adamw:
                aliases.append((name, value))
                setattr(train_module, name, TrackingAdamW)
        model.forward = types.MethodType(tracked_forward, model)
        inspect.signature(train_step).bind(model, noisy, condition, target, optimizer=None)
        returned = train_step(model, noisy, condition, target, optimizer=None)
    finally:
        model.forward = original_forward
        torch.optim.AdamW = original_adamw
        for name, value in aliases:
            setattr(train_module, name, value)
    if "output" not in captured:
        raise RuntimeError("train_step did not execute model.forward")
    if isinstance(returned, torch.Tensor):
        if returned.numel() != 1:
            raise TypeError("train_step must return a scalar")
        returned = returned.detach().item()
    if isinstance(returned, bool) or not isinstance(returned, (int, float)):
        raise TypeError(f"train_step must return a numeric scalar, got {type(returned).__name__}")
    actual_loss = float(returned)
    expected_loss = float((captured["output"] - target).square().mean())
    candidate_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    return {
        "actual_loss": actual_loss,
        "expected_loss": expected_loss,
        "adamw_constructed": tracker["constructed"],
        "owns_model_parameters": tracker["parameter_ids"] == candidate_ids,
    }


def _check_stormcast_train_semantics(config: dict, artifact_path=None):
    name = "stormcast_training:adamw_selected"
    try:
        evidence = _training_evidence(config)
        passed = evidence["adamw_constructed"] >= 1 and evidence["owns_model_parameters"]
        return [(name, bool(passed), str(evidence))]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_method_correctness(config: dict, artifact_path=None):
    """Check dataflow, sampling, or optimizer behavior beyond Base's named-call oracle."""
    rule = (config.get("phase"), config.get("task_rule"))
    if rule == ("model_structure", "component_usage"):
        return _check_stormcast_component_usage(config, artifact_path)
    if rule == ("model_structure", "sampling"):
        return _check_stormcast_sampling(config, artifact_path)
    if rule == ("train_script", "optimizer"):
        return _check_stormcast_train_semantics(config, artifact_path)
    raise ValueError(f"unsupported StormCast method rule: {rule!r}")


def check_train_structure(config: dict, caller_globals=None, caller_locals=None):
    """Reuse Base AST checks while excluding CLI/DataLoader leaves absent from Prompt."""
    name = config.get("train_function", "train_step")
    base_config = {
        "files": [config.get("train_file", "train.py")],
        "check_config": {
            "entry_names": [name],
            "model_names": ["model"],
            "model_attributes": ["forward", "parameters", "train"],
            "optimizer_names": ["AdamW"],
            "loss_names": ["MSELoss", "mse_loss", "mean"],
        },
    }
    wanted = {
        "train_start:has_main", "train_start:has_model",
        "train_start:has_optimizer", "train_start:has_loss",
        "train_start:has_backward",
    }
    results = base_evaluator.check_train_structure(base_config)
    return [item for item in results if item[0] in wanted]


def check_train_loss(config: dict, artifact_path=None):
    name = "stormcast_training:noise_mse"
    try:
        evidence = _training_evidence(config)
        passed = math.isfinite(evidence["actual_loss"]) and math.isclose(
            evidence["actual_loss"], evidence["expected_loss"],
            rel_tol=1e-5, abs_tol=1e-6,
        )
        return [(name, bool(passed), str(evidence))]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_smoke_training(config: dict, caller_globals=None, caller_locals=None):
    """Base smoke-training flow with optional config and cross-file hooks."""
    checkpoint_names = ["training:smoke_test"]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in checkpoint_names]

    def update(index: int, passed: bool, detail: object):
        checks[index] = (checkpoint_names[index], bool(passed), str(detail))

    current_index = 0
    try:
        train_file = config.get("run_file")
        if not train_file:
            raise ValueError("run_file not provided in config")
        train_module = _load_candidate_modules(train_file)
        train_name = config.get("train_fn_name", "train")
        train_fn = getattr(train_module, train_name, None)
        if not callable(train_fn):
            raise AttributeError(f"{train_name} is missing or not callable")

        work_dir = (CANDIDATE_DIR / "smoke-training").resolve()
        checkpoint_dir, results_dir = work_dir / "checkpoints", work_dir / "results"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        runtime_config = {}
        config_name = config.get("config_name")
        if config_name:
            runtime_config.update(_load_candidate_config(config_name))
        context = {
            "checkpoint_dir": checkpoint_dir,
            "results_dir": results_dir,
            "dtype": torch.float32,
            "device": DEVICE,
        }
        runtime_config.update(_resolve_args(config.get("smoke_config", {}), context))
        context["config"] = runtime_config
        context.update(runtime_config)
        model_module = _load_candidate_modules(config.get("model_file", "stormcast.py"))
        model_class = _load_class(model_module, config.get("model_class", "StormCast"))
        model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        context["model"] = model_class(**_resolve_args(model_args, context)).to(DEVICE)
        context = _execute_hooks(
            train_module,
            config.get("pre_train_hooks", []),
            context,
            caller_globals={"torch.randn": torch.randn},
        )

        train_args = _resolve_args(config.get("train_args", {}), context)
        if config.get("args_as_positional", False):
            train_fn(*train_args.values())
        else:
            train_fn(**train_args) if train_args else train_fn(runtime_config)
        update(current_index, True, f"Smoke test completed: {train_name}(...)")
    except Exception:
        update(current_index, False, traceback.format_exc())
    return checks


def check_infer_structure(config: dict, caller_globals=None, caller_locals=None):
    """Reuse Base AST checks without requiring model loading or file output here."""
    name = config.get("evaluate_function", "evaluate")
    base_config = {
        "files": [config.get("evaluate_file", "evaluate.py")],
        "check_config": {
            "entry_function_names": [name],
            "input_parameter_names": [
                "model", "noisy", "cond", "conditioning", "target_noise"
            ],
            "infer_call_attributes": ["forward", "predict", "inference", "infer"],
        },
    }
    wanted = {
        "inference_start:has_entry", "inference_start:has_eval",
        "inference_start:has_input", "inference_start:has_inference",
        "inference_start:has_no_grad",
    }
    results = base_evaluator.check_infer_structure(base_config)
    return [item for item in results if item[0] in wanted]


def check_infer_execution(config: dict, caller_globals=None, caller_locals=None):
    """Base inference execution with optional config and selected-field output."""
    checkpoint_names = [
        "inference_execution:module_load",
        "inference_execution:pre_hooks",
        "inference_execution:infer_call",
        "inference_execution:output_save",
    ]
    checks = [(name, False, "Not executed due to earlier failure")
              for name in checkpoint_names]
    current_index = 0
    try:
        output_path = config.get("output_path")
        if output_path:
            previous = Path(output_path)
            if not previous.is_absolute():
                previous = CANDIDATE_DIR / previous
            previous.unlink(missing_ok=True)

        infer_file = config.get("run_file", config.get("evaluate_file"))
        if not infer_file:
            raise ValueError("config['run_file'] is missing or empty")
        evaluate_module = _load_candidate_modules(infer_file)
        checks[current_index] = (
            checkpoint_names[current_index], True, f"Module loaded from {infer_file}"
        )

        current_index += 1
        config_name = config.get("config_name")
        candidate_config = (
            _load_candidate_config(config_name) if config_name else {}
        )
        context = {
            "config": candidate_config,
            "dtype": torch.float32,
            "device": DEVICE,
        }
        context.update(candidate_config)
        model_module = _load_candidate_modules(config.get("model_file", "stormcast.py"))
        model_class = _load_class(model_module, config.get("model_class", "StormCast"))
        model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        context["model"] = model_class(**_resolve_args(model_args, context)).to(DEVICE)
        context = _execute_hooks(
            evaluate_module,
            config.get("pre_infer_hooks", []),
            context,
            caller_globals={"torch.randn": torch.randn},
        )
        checks[current_index] = (
            checkpoint_names[current_index], True,
            f"All {len(config.get('pre_infer_hooks', []))} pre-hooks executed",
        )

        current_index += 1
        evaluate_fn = getattr(
            evaluate_module,
            config.get("infer_function", config.get("evaluate_function", "evaluate")),
            None,
        )
        if not callable(evaluate_fn):
            raise AttributeError("evaluate is missing or not callable")
        infer_args = _resolve_args(config.get("infer_args", {}), context)
        result = (
            evaluate_fn(*infer_args.values())
            if config.get("args_as_positional", False)
            else evaluate_fn(**infer_args)
        )
        checks[current_index] = (
            checkpoint_names[current_index], True, "Inference executed successfully"
        )

        current_index += 1
        if output_path:
            path = Path(output_path)
            if not path.is_absolute():
                path = CANDIDATE_DIR / path
            path.parent.mkdir(parents=True, exist_ok=True)
            fields = config.get("output_fields")
            if fields:
                if not isinstance(result, dict):
                    raise TypeError("StormCast evaluate must return a dict")
                result = torch.stack([
                    torch.as_tensor(result[name]).detach().cpu().reshape(())
                    for name in fields
                ])
            if path.suffix == ".pkl":
                with path.open("wb") as handle:
                    pickle.dump(result, handle)
            else:
                with path.open("w", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False, default=str)
            checks[current_index] = (
                checkpoint_names[current_index], True, f"Output saved to {path}"
            )
        else:
            checks[current_index] = (
                checkpoint_names[current_index], True,
                "No output_path configured, skipping save",
            )
    except Exception:
        checks[current_index] = (checkpoint_names[current_index], bool(False), str(traceback.format_exc()))
    return checks


def check_infer_output_validity(config: dict, artifact_path=None):
    name = "stormcast_inference:output_validity"
    try:
        path = Path(config["output_path"])
        if not path.is_absolute():
            path = CANDIDATE_DIR / path
        with path.open("rb") as handle:
            result = pickle.load(handle)
        values = torch.as_tensor(result)
        passed = bool(values.numel() == 2 and torch.isfinite(values).all() and (values >= 0).all())
        return [(name, passed, f"values={values.tolist()}")]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_evaluate_oracle(config: dict, caller_globals=None, caller_locals=None):
    """Base evaluate-oracle flow with shared candidate arguments and named outputs."""
    names = ["evaluate_oracle:shape_match", "evaluate_oracle:value_match"]
    checks = [(name, False, "Not executed due to earlier failure") for name in names]
    shared_args = {}

    def load_and_run(section: dict, is_candidate: bool):
        loader = _load_candidate_modules if is_candidate else _load_reference_modules
        module = loader(section["run_file"])
        context = {"config": {}, "dtype": torch.float32, "device": DEVICE}
        if is_candidate:
            model_module = _load_candidate_modules(section.get("model_file", "stormcast.py"))
            model_class = _load_class(
                model_module, section.get("model_class", "StormCast")
            )
            model_args = {**DEFAULT_MODEL_ARGS, **section.get("model_args", {})}
            context["model"] = model_class(**_resolve_args(model_args, context)).to(DEVICE)
        context = _execute_hooks(
            module,
            section.get("pre_infer_hooks", []),
            context,
            caller_globals={"torch.randn": torch.randn},
        )
        function = getattr(module, section["infer_function"], None)
        if not callable(function):
            raise ValueError(f"Configured inference function is not callable: {section!r}")
        arguments = _resolve_args(section.get("infer_args", {}), context)
        if config.get("share_candidate_args", False):
            if is_candidate:
                shared_args["value"] = copy.deepcopy(arguments)
            else:
                arguments = copy.deepcopy(shared_args["value"])
        return (
            function(*arguments.values())
            if section.get("args_as_positional", False)
            else function(**arguments)
        )

    try:
        candidate = config["candidate"]
        reference = config["reference"]
        candidate_result = load_and_run(candidate, True)
        dependencies = {
            Path(filename).stem: _load_reference_modules(filename)
            for filename in reference.get("import_files", [])
        }
        with mock.patch.dict(sys.modules, dependencies):
            reference_result = load_and_run(reference, False)
        if not isinstance(candidate_result, dict) or not isinstance(reference_result, dict):
            raise TypeError("StormCast evaluate must return a dict")
        candidate_tensor = torch.stack([
            torch.as_tensor(candidate_result[name]).detach().cpu().reshape(())
            for name in config["output_fields"]
        ]).to(torch.float64)
        reference_tensor = torch.stack([
            torch.as_tensor(reference_result[name]).detach().cpu().reshape(())
            for name in config["output_fields"]
        ]).to(torch.float64)
        shape_ok = candidate_tensor.shape == reference_tensor.shape
        checks[0] = (names[0], bool(shape_ok), f"candidate={tuple(candidate_tensor.shape)}, reference={tuple(reference_tensor.shape)}")
        value_ok = bool(
            shape_ok and torch.isfinite(candidate_tensor).all()
            and torch.isfinite(reference_tensor).all()
            and torch.allclose(
                candidate_tensor, reference_tensor,
                atol=config.get("atol", 1e-6), rtol=config.get("rtol", 1e-5),
            )
        )
        checks[1] = (names[1], value_ok, f"candidate={candidate_tensor.tolist()}, reference={reference_tensor.tolist()}")
    except Exception:
        first_pending = next(
            (
                index for index, item in enumerate(checks)
                if item[2] == "Not executed due to earlier failure"
            ),
            len(checks) - 1,
        )
        checks[first_pending] = (names[first_pending], False, traceback.format_exc())
    return checks
