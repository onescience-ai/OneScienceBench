#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PINTO Kovasznay evaluator built from Base with task-specific checks."""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import math
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from base import base_evaluator
from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _resolve_args,
    check_config_content,
    check_files,
    check_infer_output_schema,
    check_infer_structure,
    check_interface,
    check_parameter_update,
)
from base.model_structure_checker import ModelStructureChecker


SEED = 1234


def check_train_structure(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Keep Base results and recognize loops delegated to a training-step helper."""
    checks = base_evaluator.check_train_structure(config)
    loop_index = next(
        (
            index
            for index, (name, _passed, _message) in enumerate(checks)
            if name == "train_start:has_loop"
        ),
        None,
    )
    if loop_index is None or checks[loop_index][1]:
        return checks

    try:
        all_nodes = []
        for relative in config.get("files") or []:
            path = Path(CANDIDATE_DIR, relative)
            all_nodes.extend(ast.walk(ast.parse(path.read_text(encoding="utf-8"))))

        functions = {
            node.name: node
            for node in all_nodes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        training_helpers = {
            name
            for name, function in functions.items()
            if any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in {"backward", "step"}
                for child in ast.walk(function)
            )
        }
        delegated_helpers = set()
        for node in all_nodes:
            if not isinstance(node, (ast.For, ast.While)):
                continue
            called = {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            delegated_helpers.update(called & training_helpers)

        if delegated_helpers:
            checks[loop_index] = (
                checks[loop_index][0],
                True,
                "Training loop delegates parameter updates to: "
                + ", ".join(sorted(delegated_helpers)),
            )
    except Exception:
        checks[loop_index] = (
            checks[loop_index][0],
            False,
            traceback.format_exc(),
        )
    return checks


def check_model_structure(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Apply Base structure rules while preserving every configured result."""
    rules = config.get("rules")
    if not rules:
        return [
            (
                "model_structure:rules",
                False,
                "config['rules'] is missing or empty",
            )
        ]

    checkpoint_names = ["model_structure:build_model_ok"]
    checkpoint_names += [
        f"model_structure:{rule.get('id', '<missing>')}" for rule in rules
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update_check(index: int, passed: bool, message: Any):
        checks[index] = (checkpoint_names[index], bool(passed), str(message))

    current_index = 0
    try:
        model_file = config.get("model_file")
        model_class_name = config.get("model_class")
        if not model_file or not model_class_name:
            raise ValueError("model_file and model_class are required")
        model_module = _load_candidate_modules(model_file)
        model_cls = _load_class(model_module, model_class_name)

        candidate_config = (
            _load_candidate_config(config["config_name"])
            if config.get("config_name")
            else {}
        )
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)
        model_args = config.get("model_args")
        if not model_args:
            raise ValueError("config['model_args'] is required")
        model = model_cls(**_resolve_args(model_args, context)).to(DEVICE).eval()
        update_check(0, True, f"Model built with config: {candidate_config}")

        checker = ModelStructureChecker()
        checker.add_rules_from_rules(rules)
        rule_results = checker.check(model)
        for current_index, rule in enumerate(rules, 1):
            matched = [
                result for result in rule_results if result.rule_id == rule["id"]
            ]
            update_check(
                current_index,
                bool(matched) and all(result.passed for result in matched),
                "; ".join(result.message for result in matched)
                or "No module matched the rule",
            )
    except Exception:
        update_check(
            min(current_index, len(checks) - 1),
            False,
            traceback.format_exc(),
        )
    return checks


def check_model_forward_execution(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Run the public PINTO interface on two dynamic query/boundary sizes."""
    shapes = [tuple(int(value) for value in item) for item in config.get("shapes", [])]
    checkpoint_names = [
        f"model_forward:shape_finite_sensitive_b{batch}_n{queries}_s{boundary}"
        for batch, queries, boundary in shapes
    ]
    if config.get("check_backward"):
        checkpoint_names.append("model_forward:backward")
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update_check(index: int, passed: bool, message: Any):
        checks[index] = (checkpoint_names[index], bool(passed), str(message))

    try:
        model_module = _load_candidate_modules(config["model_file"])
        model_cls = _load_class(model_module, config["model_class"])
        candidate_config = (
            _load_candidate_config(config["config_name"])
            if config.get("config_name")
            else {}
        )
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)
        model_args = _resolve_args(config.get("model_args") or {}, context)
    except Exception:
        detail = traceback.format_exc()
        for index in range(len(checks)):
            update_check(index, False, detail)
        return checks

    for index, (batch_size, query_count, boundary_count) in enumerate(shapes):
        try:
            torch.manual_seed(int(config.get("seed", SEED)) + index)
            model = model_cls(**model_args).to(DEVICE).eval()
            query_xy = torch.rand(batch_size, query_count, 2, device=DEVICE)
            query_xy[..., 0] = query_xy[..., 0] * 1.5 - 0.5
            query_xy[..., 1] = query_xy[..., 1] * 2.0 - 0.5
            boundary_xy = torch.rand(batch_size, boundary_count, 2, device=DEVICE)
            boundary_xy[..., 0] = boundary_xy[..., 0] * 1.5 - 0.5
            boundary_xy[..., 1] = boundary_xy[..., 1] * 2.0 - 0.5
            boundary_values = torch.randn(
                batch_size, boundary_count, 3, device=DEVICE
            )
            originals = (
                query_xy.clone(),
                boundary_xy.clone(),
                boundary_values.clone(),
            )
            with torch.no_grad():
                output = model(query_xy, boundary_xy, boundary_values)
                shifted = model(query_xy, boundary_xy, boundary_values + 0.125)
            is_tensor = torch.is_tensor(output) and torch.is_tensor(shifted)
            actual_shape = tuple(output.shape) if torch.is_tensor(output) else None
            finite = is_tensor and bool(torch.isfinite(output).all())
            unchanged = all(
                torch.equal(current, original)
                for current, original in zip(
                    (query_xy, boundary_xy, boundary_values), originals
                )
            )
            sensitive = is_tensor and not torch.allclose(output, shifted)
            expected_shape = (batch_size, query_count, 3)
            update_check(
                index,
                is_tensor
                and actual_shape == expected_shape
                and finite
                and unchanged
                and sensitive,
                f"shape={actual_shape}, expected={expected_shape}, finite={finite}, "
                f"inputs_unchanged={unchanged}, boundary_sensitive={sensitive}",
            )
        except Exception:
            update_check(index, False, traceback.format_exc())

    if config.get("check_backward"):
        index = len(shapes)
        try:
            torch.manual_seed(int(config.get("seed", SEED)) + 100)
            model = model_cls(**model_args).to(DEVICE).train()
            query_xy = torch.rand(1, 6, 2, device=DEVICE, requires_grad=True)
            boundary_xy = torch.rand(1, 8, 2, device=DEVICE, requires_grad=True)
            boundary_values = torch.randn(
                1, 8, 3, device=DEVICE, requires_grad=True
            )
            output = model(query_xy, boundary_xy, boundary_values)
            if not torch.is_tensor(output) or output.shape != (1, 6, 3):
                raise TypeError("forward must return a Tensor with shape [B,N,3]")
            output.square().mean().backward()
            parameter_gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            input_gradients = (
                query_xy.grad,
                boundary_xy.grad,
                boundary_values.grad,
            )
            passed = (
                all(gradient is not None for gradient in input_gradients)
                and all(
                    bool(torch.isfinite(gradient).all())
                    for gradient in input_gradients
                    if gradient is not None
                )
                and bool(parameter_gradients)
                and all(
                    bool(torch.isfinite(gradient).all())
                    for gradient in parameter_gradients
                )
            )
            update_check(
                index,
                passed,
                f"input_gradients={[item is not None for item in input_gradients]}, "
                f"parameter_gradients={len(parameter_gradients)}",
            )
        except Exception:
            update_check(index, False, traceback.format_exc())
    return checks


def check_method_correctness(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Check the four PINTO-specific contracts not expressible by Base."""
    contract = config.get("contract")
    definitions = {
        "pinto_data": (
            "required_interfaces",
            "paper_config",
            "analytic_solution",
            "batch_schema",
            "query_domain",
            "latin_hypercube_strata",
            "boundary_geometry",
            "boundary_solution",
            "finite_values",
        ),
        "pinto_model": (
            "model_build",
            "public_modules",
            "two_head_attention",
            "swish_activation",
            "independent_output_heads",
            "forward_signature",
            "required_modules_used",
            "independent_implementation",
        ),
        "pinto_training": (
            "required_interfaces",
            "adam_learning_rate",
            "supervised_boundary_residual_losses",
            "navier_stokes_autodiff",
            "real_optimization",
            "checkpoint_and_results",
            "live_metrics",
            "python_syntax",
            "no_repository_download",
        ),
        "pinto_inference": (
            "script_entrypoints",
            "relative_l2_reference",
            "test_viscosities",
            "checkpoint_restore",
            "evaluation_mode",
            "result_persistence",
            "metric_reporting",
            "python_syntax",
        ),
    }
    names = definitions.get(contract)
    if names is None:
        return [("method_check:contract", False, f"Unsupported contract: {contract}")]
    checkpoint_names = [f"method_check:{contract}:{name}" for name in names]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update_check(index: int, passed: bool, message: Any):
        checks[index] = (checkpoint_names[index], bool(passed), str(message))

    try:
        if contract == "pinto_data":
            data_module = _load_candidate_modules(config["data_file"])
            candidate_config = _load_candidate_config(config["config_name"])
            required = (
                "kovasznay_solution",
                "latin_hypercube_2d",
                "build_kovasznay_batch",
            )
            update_check(
                0,
                all(callable(getattr(data_module, name, None)) for name in required),
                required,
            )
            if not checks[0][1]:
                return checks

            config_values: dict[str, list[Any]] = {}
            pending = [candidate_config]
            while pending:
                current = pending.pop()
                if isinstance(current, Mapping):
                    for key, value in current.items():
                        config_values.setdefault(str(key).lower(), []).append(value)
                        pending.append(value)
                elif isinstance(current, (list, tuple)):
                    pending.extend(current)

            train_values = []
            test_values = []
            for key, destination in (
                ("train_nue", train_values),
                ("test_nue", test_values),
            ):
                for value in config_values.get(key, []):
                    if not isinstance(value, (list, tuple)):
                        continue
                    converted = []
                    try:
                        for item in value:
                            if isinstance(item, str) and "/" in item:
                                numerator, denominator = item.split("/", 1)
                                converted.append(float(numerator) / float(denominator))
                            else:
                                converted.append(float(item))
                    except (TypeError, ValueError, ZeroDivisionError):
                        converted = []
                    if converted:
                        destination.extend(converted)
                        break
            expected_train = [1 / 20, 1 / 30, 1 / 50, 1 / 80]
            expected_test = [1 / 10, 1 / 20, 1 / 30, 1 / 40, 1 / 50]
            domain_start = next(
                (
                    value
                    for value in config_values.get("domain_start", [])
                    if isinstance(value, (list, tuple))
                ),
                None,
            )
            domain_stop = next(
                (
                    value
                    for value in config_values.get("domain_stop", [])
                    if isinstance(value, (list, tuple))
                ),
                None,
            )
            domain_mapping = candidate_config.get("domain")
            if isinstance(domain_mapping, Mapping):
                if domain_start is None:
                    domain_start = [
                        domain_mapping.get("x_min"),
                        domain_mapping.get("y_min"),
                    ]
                if domain_stop is None:
                    domain_stop = [
                        domain_mapping.get("x_max"),
                        domain_mapping.get("y_max"),
                    ]
            paper_config = (
                domain_start == [-0.5, -0.5]
                and domain_stop == [1.0, 1.5]
                and 64 in config_values.get("grid_size", [])
                and 2000 in config_values.get("domain_samples", [])
                and 20 in config_values.get("seq_len", [])
                and len(train_values) == len(expected_train)
                and all(
                    math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
                    for left, right in zip(train_values, expected_train)
                )
                and len(test_values) == len(expected_test)
                and all(
                    math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
                    for left, right in zip(test_values, expected_test)
                )
            )
            update_check(
                1,
                paper_config,
                f"domain={domain_start}->{domain_stop}, train_nue={train_values}, "
                f"test_nue={test_values}",
            )

            fixed_xy = torch.tensor(
                [[[0.0, 0.0], [0.25, 0.5], [-0.25, 0.75]]],
                dtype=torch.float32,
            )
            observed = data_module.kovasznay_solution(fixed_xy, 0.05)
            x = fixed_xy[..., 0:1]
            y = fixed_xy[..., 1:2]
            nue = torch.tensor(0.05, dtype=fixed_xy.dtype)
            zeta = 0.5 / nue - torch.sqrt(
                1.0 / (4.0 * nue.square()) + 4.0 * math.pi**2
            )
            expected = torch.cat(
                (
                    1.0 - torch.exp(zeta * x) * torch.cos(2.0 * math.pi * y),
                    zeta
                    / (2.0 * math.pi)
                    * torch.exp(zeta * x)
                    * torch.sin(2.0 * math.pi * y),
                    0.5 * (1.0 - torch.exp(2.0 * zeta * x)),
                ),
                dim=-1,
            )
            analytic_ok = (
                torch.is_tensor(observed)
                and observed.shape == expected.shape
                and torch.allclose(observed.cpu(), expected, rtol=1e-5, atol=1e-6)
            )
            update_check(2, analytic_ok, f"observed_shape={getattr(observed, 'shape', None)}")

            batch = data_module.build_kovasznay_batch(
                [0.05, 0.025],
                domain_samples=16,
                seq_len=4,
                domain=candidate_config["domain"],
                generator=torch.Generator().manual_seed(
                    int(config.get("seed", SEED))
                ),
                device=DEVICE,
            )
            required_keys = (
                "query_xy",
                "boundary_xy",
                "boundary_values",
                "nue",
            )
            actual_shapes = {
                key: tuple(batch[key].shape)
                for key in required_keys
                if isinstance(batch, Mapping) and torch.is_tensor(batch.get(key))
            }
            schema_ok = (
                isinstance(batch, Mapping)
                and len(actual_shapes) == len(required_keys)
                and actual_shapes["query_xy"] == (2, 16, 2)
                and len(actual_shapes["boundary_xy"]) == 3
                and actual_shapes["boundary_xy"][0] == 2
                and actual_shapes["boundary_xy"][2] == 2
                and actual_shapes["boundary_values"][:2]
                == actual_shapes["boundary_xy"][:2]
                and actual_shapes["boundary_values"][2] == 3
                and batch["nue"].numel() == 2
                and actual_shapes["nue"][0] == 2
                and (
                    "target" not in batch
                    or (
                        torch.is_tensor(batch["target"])
                        and tuple(batch["target"].shape) == (2, 16, 3)
                    )
                )
            )
            update_check(3, schema_ok, actual_shapes)
            if not schema_ok:
                return checks

            query_xy = batch["query_xy"]
            query_domain_ok = bool(
                (query_xy[..., 0] >= -0.5).all()
                and (query_xy[..., 0] <= 1.0).all()
                and (query_xy[..., 1] >= -0.5).all()
                and (query_xy[..., 1] <= 1.5).all()
            )
            update_check(4, query_domain_ok, "query points lie inside the paper domain")

            normalized = torch.stack(
                (
                    (query_xy[0, :, 0] + 0.5) / 1.5,
                    (query_xy[0, :, 1] + 0.5) / 2.0,
                ),
                dim=-1,
            ).clamp(0.0, 1.0 - 1e-7)
            bins = torch.floor(normalized * query_xy.shape[1]).to(torch.int64)
            strata_ok = all(
                torch.unique(bins[:, dimension]).numel() >= 14
                for dimension in range(2)
            )
            update_check(5, strata_ok, f"unique_strata={[torch.unique(bins[:, i]).numel() for i in range(2)]}")

            boundary_xy = batch["boundary_xy"]
            on_edge = (
                torch.isclose(boundary_xy[..., 0], boundary_xy.new_tensor(-0.5))
                | torch.isclose(boundary_xy[..., 0], boundary_xy.new_tensor(1.0))
                | torch.isclose(boundary_xy[..., 1], boundary_xy.new_tensor(-0.5))
                | torch.isclose(boundary_xy[..., 1], boundary_xy.new_tensor(1.5))
            )
            update_check(6, bool(on_edge.all()), f"edge_points={int(on_edge.sum())}/{on_edge.numel()}")

            expected_targets = []
            expected_boundaries = []
            for batch_index, viscosity in enumerate((0.05, 0.025)):
                expected_targets.append(
                    data_module.kovasznay_solution(
                        query_xy[batch_index : batch_index + 1], viscosity
                    )
                )
                expected_boundaries.append(
                    data_module.kovasznay_solution(
                        boundary_xy[batch_index : batch_index + 1], viscosity
                    )
                )
            expected_targets = torch.cat(expected_targets, dim=0)
            expected_boundaries = torch.cat(expected_boundaries, dim=0)
            boundary_ok = torch.allclose(
                batch["boundary_values"], expected_boundaries, rtol=1e-5, atol=1e-6
            ) and (
                "target" not in batch
                or torch.allclose(
                    batch["target"], expected_targets, rtol=1e-5, atol=1e-6
                )
            )
            update_check(
                7,
                boundary_ok,
                "boundary values and any supplied targets match the analytic solution",
            )

            finite_ok = all(
                bool(torch.isfinite(batch[key]).all()) for key in required_keys
            ) and (
                "target" not in batch
                or bool(torch.isfinite(batch["target"]).all())
            )
            update_check(8, finite_ok, "all generated tensors are finite")
            return checks

        if contract == "pinto_model":
            model_module = _load_candidate_modules(config["model_file"])
            model_cls = _load_class(model_module, config["model_class"])
            candidate_config = _load_candidate_config(config["config_name"])
            context = {"config": candidate_config}
            context.update(candidate_config)
            model_args = _resolve_args(config.get("model_args") or {}, context)
            model = model_cls(**model_args).to(DEVICE).eval()
            update_check(0, True, type(model).__name__)

            required_modules = (
                "query_lifting",
                "boundary_position_encoder",
                "boundary_value_encoder",
                "cross_attention",
                "post_attention",
                "u_head",
                "v_head",
                "p_head",
            )
            public_ok = all(
                isinstance(getattr(model, name, None), torch.nn.Module)
                for name in required_modules
            )
            update_check(1, public_ok, required_modules)

            attention = getattr(model, "cross_attention", None)
            observed_heads = []
            if isinstance(attention, torch.nn.Module):
                for module in attention.modules():
                    for attribute in ("num_heads", "heads"):
                        value = getattr(module, attribute, None)
                        if isinstance(value, int):
                            observed_heads.append(value)
                            break
            configured_heads = (
                candidate_config.get("model", {}).get("num_heads")
                if isinstance(candidate_config.get("model"), Mapping)
                else None
            )
            source = Path(CANDIDATE_DIR, config["model_file"]).read_text(
                encoding="utf-8"
            )
            attention_ok = (
                isinstance(attention, torch.nn.Module)
                and any(True for _ in attention.children())
                and (
                    bool(observed_heads) and all(value == 2 for value in observed_heads)
                    or configured_heads == 2 and "num_heads" in source
                )
            )
            update_check(
                2,
                attention_ok,
                f"container={type(attention).__name__}, "
                f"observed_heads={observed_heads}, configured_heads={configured_heads}",
            )
            modules = list(model.modules())
            activation_ok = any(
                isinstance(module, torch.nn.SiLU) for module in modules
            ) or "swish" in source.lower()
            update_check(3, activation_ok, "SiLU module or explicit swish implementation")

            heads = [getattr(model, name, None) for name in ("u_head", "v_head", "p_head")]
            independent_heads = (
                all(isinstance(head, torch.nn.Module) for head in heads)
                and len({id(head) for head in heads}) == 3
            )
            update_check(4, independent_heads, f"head_ids={[id(head) for head in heads]}")

            forward_signature = inspect.signature(model.forward)
            forward_ok = all(
                name in forward_signature.parameters
                for name in ("query_xy", "boundary_xy", "boundary_values")
            )
            update_check(5, forward_ok, f"forward={forward_signature}")

            seen: set[str] = set()
            handles = []
            try:
                for name in required_modules:
                    module = getattr(model, name, None)
                    if isinstance(module, torch.nn.Module):
                        observed_modules = (
                            list(module.children())
                            if isinstance(module, torch.nn.ModuleList)
                            else [module]
                        )
                        for observed_module in observed_modules:
                            handles.append(
                                observed_module.register_forward_hook(
                                    lambda _module, _inputs, _output, label=name: seen.add(label)
                                )
                            )
                torch.manual_seed(int(config.get("seed", SEED)))
                with torch.no_grad():
                    output = model(
                        torch.rand(1, 7, 2, device=DEVICE),
                        torch.rand(1, 8, 2, device=DEVICE),
                        torch.randn(1, 8, 3, device=DEVICE),
                    )
                used_ok = set(required_modules).issubset(seen) and output.shape == (1, 7, 3)
                update_check(6, used_ok, f"observed={sorted(seen)}")
            finally:
                for handle in handles:
                    handle.remove()

            forbidden = (
                "git clone",
                "github.com",
                "codegen.standard",
                "codegen.evaluators",
            )
            found = [token for token in forbidden if token in source.lower()]
            update_check(7, not found, f"forbidden_references={found}")
            return checks

        files = config.get("files") or []
        sources: dict[str, str] = {}
        syntax: dict[str, bool] = {}
        for relative in files:
            path = Path(CANDIDATE_DIR, relative)
            source = path.read_text(encoding="utf-8")
            sources[Path(relative).name] = source
            if path.suffix == ".py":
                ast.parse(source, filename=str(path))
                syntax[Path(relative).name] = True
        candidate_config = _load_candidate_config(config["config_name"])
        combined = "\n".join(sources.values())
        lower = combined.lower()
        compact = "".join(lower.split())
        config_text = json.dumps(candidate_config, ensure_ascii=False).lower()
        config_values: dict[str, list[Any]] = {}
        pending = [candidate_config]
        while pending:
            current = pending.pop()
            if isinstance(current, Mapping):
                for key, value in current.items():
                    config_values.setdefault(str(key).lower(), []).append(value)
                    pending.append(value)
            elif isinstance(current, (list, tuple)):
                pending.extend(current)

        if contract == "pinto_training":
            train_module = _load_candidate_modules(files[0])
            required = (
                "load_config",
                "build_model",
                "pde_residual_loss",
                "train_one_step",
                "main",
            )
            values = (
                (
                    all(callable(getattr(train_module, name, None)) for name in required),
                    required,
                ),
                (
                    "adam" in lower
                    and any(
                        isinstance(value, (int, float))
                        and math.isclose(float(value), 5e-4, rel_tol=1e-9, abs_tol=1e-12)
                        for key in ("learning_rate", "lr")
                        for value in config_values.get(key, [])
                    ),
                    "Adam with learning_rate=5e-4",
                ),
                (
                    all(token in lower for token in ("data_loss", "boundary_loss", "residual_loss"))
                    and any(
                        token in lower
                        for token in (
                            "mse_loss",
                            "mseloss",
                            "mean_squared_error",
                            ".square()",
                            ".pow(",
                            "**2",
                            "** 2",
                        )
                    ),
                    "data, boundary, and residual loss composition",
                ),
                (
                    "autograd.grad" in compact
                    and all(token in lower for token in ("continuity", "momentum"))
                    and any(token in lower for token in ("u_x", "ux", "du_dx"))
                    and any(token in lower for token in ("v_y", "vy", "dv_dy")),
                    "steady incompressible Navier-Stokes derivatives",
                ),
                (
                    ".backward(" in compact
                    and "optimizer.step(" in compact
                    and "zero_grad(" in compact,
                    "zero_grad, backward, and optimizer.step",
                ),
                (
                    "weight/best_model.pt" in config_text
                    and "results" in config_text
                    and any(token in lower for token in ("torch.save", "save_checkpoint")),
                    "checkpoint and result paths",
                ),
                (
                    "print(" in compact
                    and all(
                        token in lower
                        for token in (
                            "train_loss",
                            "data_loss",
                            "boundary_loss",
                            "residual_loss",
                        )
                    ),
                    "live loss reporting",
                ),
                (len(syntax) == len(files), syntax),
                (
                    not any(token in lower for token in ("git clone", "wget ", "curl ", "github.com")),
                    "independent implementation",
                ),
            )
        else:
            inference_module = _load_candidate_modules(
                next(relative for relative in files if relative.endswith("inference.py"))
            )
            result_module = _load_candidate_modules(
                next(relative for relative in files if relative.endswith("result.py"))
            )
            test_values = next(
                (
                    value
                    for value in config_values.get("test_nue", [])
                    if isinstance(value, (list, tuple))
                ),
                [],
            )
            try:
                test_values = [float(value) for value in test_values]
            except (TypeError, ValueError):
                test_values = []
            expected_test = [1 / 10, 1 / 20, 1 / 30, 1 / 40, 1 / 50]
            values = (
                (
                    callable(getattr(inference_module, "main", None))
                    and callable(getattr(result_module, "main", None)),
                    "inference.main and result.main",
                ),
                (
                    "relative_l2" in lower,
                    "relative_l2 metric referenced by inference or result code",
                ),
                (
                    len(test_values) == len(expected_test)
                    and all(
                        math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
                        for left, right in zip(test_values, expected_test)
                    ),
                    f"test_nue={test_values}",
                ),
                (
                    any(token in lower for token in ("load_state_dict", "torch.load", "load_checkpoint"))
                    and "best_model.pt" in config_text,
                    "checkpoint restoration",
                ),
                (
                    ".eval(" in compact
                    and any(token in lower for token in ("no_grad", "inference_mode")),
                    "eval mode without gradient recording",
                ),
                (
                    "results" in config_text
                    and any(token in lower for token in ("json.dump", "savez", "write_text", "save_json")),
                    "metrics or predictions persisted under results",
                ),
                (
                    "relative_l2" in lower
                    and any(token in lower for token in ("print(", "logger")),
                    "relative L2 reporting",
                ),
                (len(syntax) == len(files), syntax),
            )

        for index, (passed, detail) in enumerate(values):
            update_check(index, passed, detail)
    except Exception:
        detail = traceback.format_exc()
        for index, (_, passed, _) in enumerate(checks):
            if not passed:
                update_check(index, False, detail)
    return checks


def check_train_loss(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Dynamically validate the PINTO Navier-Stokes residual loss."""
    checkpoint_names = [
        "train_loss:pde_residual_callable",
        "train_loss:finite_scalar",
        "train_loss:query_gradient",
        "train_loss:model_gradients",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update_check(index: int, passed: bool, message: Any):
        checks[index] = (checkpoint_names[index], bool(passed), str(message))

    current_index = 0
    try:
        train_module = _load_candidate_modules(config["train_file"])
        residual_function = getattr(train_module, "pde_residual_loss", None)
        update_check(0, callable(residual_function), "pde_residual_loss")
        if not callable(residual_function):
            return checks

        data_module = _load_candidate_modules(config["data_file"])
        model_module = _load_candidate_modules(config["model_file"])
        model_cls = _load_class(model_module, config["model_class"])
        candidate_config = _load_candidate_config(config["config_name"])
        context = {"config": candidate_config}
        context.update(candidate_config)
        model_args = _resolve_args(config.get("model_args") or {}, context)
        torch.manual_seed(int(config.get("seed", SEED)))
        model = model_cls(**model_args).to(DEVICE).train()
        batch = data_module.build_kovasznay_batch(
            [0.05],
            domain_samples=6,
            seq_len=2,
            domain=candidate_config["domain"],
            generator=torch.Generator().manual_seed(int(config.get("seed", SEED))),
            device=DEVICE,
        )
        query_xy = batch["query_xy"].to(DEVICE).detach().clone().requires_grad_(True)
        boundary_xy = batch["boundary_xy"].to(DEVICE)
        boundary_values = batch["boundary_values"].to(DEVICE)
        nue = batch["nue"].to(DEVICE)
        prediction = model(query_xy, boundary_xy, boundary_values)
        available = {
            "model": model,
            "query_xy": query_xy,
            "query": query_xy,
            "coords": query_xy,
            "x": query_xy,
            "xy": query_xy,
            "points": query_xy,
            "domain_xy": query_xy,
            "collocation_xy": query_xy,
            "collocation_points": query_xy,
            "boundary_xy": boundary_xy,
            "boundary_points": boundary_xy,
            "boundary_coords": boundary_xy,
            "bc_xy": boundary_xy,
            "boundary_values": boundary_values,
            "boundary_data": boundary_values,
            "boundary_target": boundary_values,
            "bc_values": boundary_values,
            "nue": nue,
            "nu": nue,
            "viscosity": nue,
            "viscosities": nue,
            "prediction": prediction,
            "predictions": prediction,
            "pred": prediction,
            "output": prediction,
            "outputs": prediction,
        }
        signature = inspect.signature(residual_function)
        arguments = {}
        missing = []
        for name, parameter in signature.parameters.items():
            if name in available:
                arguments[name] = available[name]
            elif parameter.default is inspect._empty and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                missing.append(name)
        if missing:
            raise TypeError(f"Unsupported required residual arguments: {missing}")
        result = residual_function(**arguments)

        tensors = []
        pending = [result]
        while pending:
            value = pending.pop()
            if torch.is_tensor(value):
                tensors.append(value)
            elif isinstance(value, Mapping):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
        if not tensors:
            raise TypeError("pde_residual_loss must return Tensor residuals or losses")
        terms = [
            value if value.ndim == 0 else value.square().mean()
            for value in tensors
            if value.is_floating_point()
        ]
        if not terms:
            raise TypeError("pde_residual_loss returned no floating Tensor")
        loss = sum(terms)
        current_index = 1
        finite_scalar = loss.ndim == 0 and bool(torch.isfinite(loss)) and float(loss.detach()) >= 0.0
        update_check(1, finite_scalar, f"loss={float(loss.detach()):.6g}")

        loss.backward()
        current_index = 2
        query_ok = query_xy.grad is not None and bool(torch.isfinite(query_xy.grad).all())
        update_check(2, query_ok, f"query_gradient={query_xy.grad is not None}")

        current_index = 3
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradients_ok = bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        update_check(3, gradients_ok, f"parameter_gradients={len(gradients)}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_smoke_training(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Run the required one-step training API with a lightweight config."""
    checkpoint_names = [
        "training:module_and_function",
        "training:finite_loss",
        "training:loss_components",
        "training:parameter_update",
        "training:relative_l2",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update_check(index: int, passed: bool, message: Any):
        checks[index] = (checkpoint_names[index], bool(passed), str(message))

    current_index = 0
    try:
        train_module = _load_candidate_modules(config["run_file"])
        train_function = getattr(train_module, config.get("train_fn_name", "train_one_step"), None)
        update_check(0, callable(train_function), config.get("train_fn_name", "train_one_step"))
        if not callable(train_function):
            return checks

        runtime_config = deepcopy(_load_candidate_config(config["config_name"]))
        runtime_config.setdefault("data", {})
        runtime_config["data"].update(
            {"domain_samples": 6, "seq_len": 2, "train_nue": [0.05]}
        )
        runtime_config.setdefault("model", {})
        runtime_config["model"].update(
            {"hidden_dim": 16, "num_heads": 2, "activation": "swish"}
        )
        runtime_config.setdefault("training", {})
        runtime_config["training"].update(
            {"epochs": 1, "batch_size": 1, "learning_rate": 5e-4}
        )

        with tempfile.TemporaryDirectory(prefix="pinto_kovasznay_smoke_") as temporary:
            work_dir = Path(temporary)
            model_builder = getattr(train_module, "build_model", None)
            batch_builder = getattr(train_module, "build_kovasznay_batch", None)
            if not callable(model_builder) or not callable(batch_builder):
                raise AttributeError("build_model and build_kovasznay_batch must be callable")

            torch.manual_seed(int(config.get("seed", SEED)))
            model = model_builder(runtime_config).to(DEVICE)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(runtime_config["training"]["learning_rate"]),
            )
            generator = torch.Generator().manual_seed(int(config.get("seed", SEED)))
            batch_values = {
                "nue_values": runtime_config["data"]["train_nue"],
                "nue_list": runtime_config["data"]["train_nue"],
                "nu_values": runtime_config["data"]["train_nue"],
                "viscosities": runtime_config["data"]["train_nue"],
                "viscosity_values": runtime_config["data"]["train_nue"],
                "nues": runtime_config["data"]["train_nue"],
                "train_nue": runtime_config["data"]["train_nue"],
                "domain_samples": runtime_config["data"]["domain_samples"],
                "samples": runtime_config["data"]["domain_samples"],
                "seq_len": runtime_config["data"]["seq_len"],
                "domain": runtime_config.get("domain"),
                "generator": generator,
                "device": DEVICE,
                "seed": int(config.get("seed", SEED)),
            }
            batch_arguments = {}
            missing_batch_arguments = []
            for name, parameter in inspect.signature(batch_builder).parameters.items():
                if name in batch_values and batch_values[name] is not None:
                    batch_arguments[name] = batch_values[name]
                elif parameter.default is inspect._empty and parameter.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    missing_batch_arguments.append(name)
            if missing_batch_arguments:
                raise TypeError(
                    "Unsupported required build_kovasznay_batch arguments: "
                    f"{missing_batch_arguments}"
                )
            batch = batch_builder(**batch_arguments)
            if not isinstance(batch, Mapping):
                raise TypeError("build_kovasznay_batch must return a mapping")
            batch = {
                key: value.to(DEVICE) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            signature = inspect.signature(train_function)
            available = {
                "model": model,
                "optimizer": optimizer,
                "batch": batch,
                "config": runtime_config,
                "cfg": runtime_config,
                "output_dir": work_dir,
                "work_dir": work_dir,
                "root_dir": work_dir,
            }
            arguments = {}
            missing = []
            for name, parameter in signature.parameters.items():
                if name in available:
                    arguments[name] = available[name]
                elif parameter.default is inspect._empty and parameter.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    missing.append(name)
            if missing:
                raise TypeError(f"Unsupported required train_one_step arguments: {missing}")
            before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            captured_stdout = io.StringIO()
            with contextlib.redirect_stdout(captured_stdout):
                result = train_function(**arguments)

            train_loss = None
            if isinstance(result, Mapping):
                for key in ("train_loss", "loss", "total_loss"):
                    if key in result:
                        value = result[key]
                        train_loss = float(value.detach()) if torch.is_tensor(value) else float(value)
                        break
            elif isinstance(result, (list, tuple)) and result:
                last = result[-1]
                if isinstance(last, Mapping):
                    for key in ("train_loss", "loss", "total_loss"):
                        if key in last:
                            value = last[key]
                            train_loss = float(value.detach()) if torch.is_tensor(value) else float(value)
                            break
            elif torch.is_tensor(result) and result.numel() == 1:
                train_loss = float(result.detach())
            current_index = 1
            update_check(
                1,
                train_loss is not None and math.isfinite(train_loss),
                f"train_loss={train_loss}",
            )

            current_index = 2
            required_metrics = (
                "train_loss",
                "data_loss",
                "boundary_loss",
                "residual_loss",
            )
            components_ok = isinstance(result, Mapping) and all(
                key in result
                and isinstance(result[key], (int, float))
                and math.isfinite(float(result[key]))
                for key in required_metrics
            )
            update_check(
                2,
                components_ok,
                {key: result.get(key) for key in required_metrics}
                if isinstance(result, Mapping)
                else type(result).__name__,
            )

            current_index = 3
            changed = [
                name
                for name, parameter in model.named_parameters()
                if name in before and not torch.equal(before[name], parameter.detach())
            ]
            update_check(
                3,
                bool(changed),
                f"updated_parameters={len(changed)}",
            )

            current_index = 4
            relative_l2 = result.get("relative_l2") if isinstance(result, Mapping) else None
            update_check(
                4,
                isinstance(relative_l2, (int, float))
                and math.isfinite(float(relative_l2))
                and float(relative_l2) >= 0.0,
                f"relative_l2={relative_l2}",
            )
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_execution(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Run model inference and compute Relative L2 on analytic data."""
    checkpoint_names = [
        "inference_execution:module_load",
        "inference_execution:model_build",
        "inference_execution:infer_call",
        "inference_execution:metric_compute",
        "inference_execution:output_save",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update_check(index: int, passed: bool, message: Any):
        checks[index] = (checkpoint_names[index], bool(passed), str(message))

    current_index = 0
    try:
        infer_module = _load_candidate_modules(config["run_file"])
        data_module = _load_candidate_modules(config["data_file"])
        update_check(0, True, f"loaded {config['run_file']} and {config['data_file']}")

        current_index = 1
        model_module = _load_candidate_modules(config["model_file"])
        model_cls = _load_class(model_module, config["model_class"])
        candidate_config = _load_candidate_config(config["config_name"])
        context = {"config": candidate_config}
        context.update(candidate_config)
        model_args = _resolve_args(config.get("model_args") or {}, context)
        torch.manual_seed(int(config.get("seed", SEED)))
        model = model_cls(**model_args).to(DEVICE).eval()
        update_check(1, True, type(model).__name__)

        batch = data_module.build_kovasznay_batch(
            [0.025],
            domain_samples=16,
            seq_len=4,
            domain=candidate_config["domain"],
            generator=torch.Generator().manual_seed(int(config.get("seed", SEED))),
            device=DEVICE,
        )
        query_xy = batch["query_xy"].to(DEVICE)
        boundary_xy = batch["boundary_xy"].to(DEVICE)
        boundary_values = batch["boundary_values"].to(DEVICE)
        if torch.is_tensor(batch.get("target")):
            target = batch["target"].to(DEVICE)
        else:
            viscosities = batch["nue"].reshape(-1)
            target = torch.cat(
                [
                    data_module.kovasznay_solution(
                        query_xy[index : index + 1],
                        float(viscosities[index].detach().cpu()),
                    )
                    for index in range(query_xy.shape[0])
                ],
                dim=0,
            ).to(DEVICE)
        with torch.no_grad():
            prediction = model(query_xy, boundary_xy, boundary_values)
        current_index = 2
        inference_ok = (
            torch.is_tensor(prediction)
            and prediction.shape == target.shape == (1, 16, 3)
            and bool(torch.isfinite(prediction).all())
        )
        update_check(2, inference_ok, f"prediction_shape={getattr(prediction, 'shape', None)}")
        if not inference_ok:
            return checks

        differences = (prediction - target).reshape(1, -1, 3)
        flattened_target = target.reshape(1, -1, 3)
        per_channel = (
            torch.linalg.vector_norm(differences, dim=1)
            / torch.linalg.vector_norm(flattened_target, dim=1).clamp_min(1e-12)
        ).mean(dim=0)
        metric_ok = bool(torch.isfinite(per_channel).all()) and bool(
            (per_channel >= 0.0).all()
        )
        current_index = 3
        update_check(3, metric_ok, f"relative_l2={per_channel.detach().cpu().tolist()}")

        payload = {
            "prediction": prediction.detach().cpu().tolist(),
            "target": target.detach().cpu().tolist(),
            "relative_l2_u": float(per_channel[0]),
            "relative_l2_v": float(per_channel[1]),
            "relative_l2_p": float(per_channel[2]),
            "relative_l2_mean": float(per_channel.mean()),
        }
        output_path = Path(config["output_path"])
        if not output_path.is_absolute():
            output_path = Path(CANDIDATE_DIR, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        current_index = 4
        update_check(
            4,
            output_path.is_file() and output_path.stat().st_size > 0,
            output_path,
        )
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_output_validity(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Validate the semantic content written by this evaluation run."""
    checkpoint_names = [
        "inference_output:prediction_target_shape",
        "inference_output:finite_predictions",
        "inference_output:required_metrics",
        "inference_output:metric_values",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update_check(index: int, passed: bool, message: Any):
        checks[index] = (checkpoint_names[index], bool(passed), str(message))

    current_index = 0
    try:
        output_path = Path(config["output_path"])
        if not output_path.is_absolute():
            output_path = Path(CANDIDATE_DIR, output_path)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        prediction = torch.as_tensor(result.get("prediction"))
        target = torch.as_tensor(result.get("target"))
        expected_shape = tuple(int(value) for value in config["expected_shape"])
        shape_ok = tuple(prediction.shape) == tuple(target.shape) == expected_shape
        update_check(0, shape_ok, f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}")

        current_index = 1
        finite_ok = bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target).all())
        update_check(1, finite_ok, "prediction and target are finite")

        metric_names = tuple(config.get("metric_names") or ())
        current_index = 2
        metrics_present = all(name in result for name in metric_names)
        update_check(2, metrics_present, f"keys={sorted(result)}")

        current_index = 3
        metric_values = [result.get(name) for name in metric_names]
        metric_ok = metrics_present and all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in metric_values
        )
        update_check(3, metric_ok, dict(zip(metric_names, metric_values)))
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_metrics(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Recompute Relative L2 from saved predictions and validate channel metrics."""
    checkpoint_names = [
        "infer_metric:output_load",
        "infer_metric:per_channel_formula",
        "infer_metric:mean_formula",
        "infer_metric:lower_is_better",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in checkpoint_names
    ]

    def update_check(index: int, passed: bool, message: Any):
        checks[index] = (checkpoint_names[index], bool(passed), str(message))

    current_index = 0
    try:
        output_path = Path(config["output_path"])
        if not output_path.is_absolute():
            output_path = Path(CANDIDATE_DIR, output_path)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        prediction = torch.as_tensor(result.get("prediction"), dtype=torch.float64)
        target = torch.as_tensor(result.get("target"), dtype=torch.float64)
        channel_order = tuple(config.get("channel_order") or ("u", "v", "p"))
        output_ok = (
            prediction.shape == target.shape
            and prediction.ndim >= 2
            and prediction.shape[-1] == len(channel_order)
            and bool(torch.isfinite(prediction).all())
            and bool(torch.isfinite(target).all())
        )
        update_check(
            0,
            output_ok,
            f"path={output_path}, prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}",
        )
        if not output_ok:
            return checks

        flattened_prediction = prediction.reshape(-1, len(channel_order))
        flattened_target = target.reshape(-1, len(channel_order))
        recomputed = (
            torch.linalg.vector_norm(flattened_prediction - flattened_target, dim=0)
            / torch.linalg.vector_norm(flattened_target, dim=0).clamp_min(1e-12)
        )
        stored_channels = [result.get(f"relative_l2_{name}") for name in channel_order]
        channels_ok = all(isinstance(value, (int, float)) for value in stored_channels)
        if channels_ok:
            channels_ok = all(
                math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-5,
                    abs_tol=1e-7,
                )
                for observed, expected in zip(stored_channels, recomputed.tolist())
            )
        current_index = 1
        update_check(
            1,
            channels_ok,
            f"stored={stored_channels}, recomputed={recomputed.tolist()}, "
            f"order={channel_order}",
        )

        stored_mean = result.get("relative_l2_mean")
        recomputed_mean = float(recomputed.mean())
        mean_ok = isinstance(stored_mean, (int, float)) and math.isclose(
            float(stored_mean), recomputed_mean, rel_tol=1e-5, abs_tol=1e-7
        )
        current_index = 2
        update_check(2, mean_ok, f"stored={stored_mean}, recomputed={recomputed_mean}")

        current_index = 3
        metric_values = recomputed.tolist() + [recomputed_mean]
        direction_ok = config.get("mode") == "min" and all(
            math.isfinite(value) and value >= 0.0 for value in metric_values
        )
        update_check(3, direction_ok, f"mode={config.get('mode')}, values={metric_values}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks
