#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XiHe evaluator built from Base with only necessary task-specific checks."""

from __future__ import annotations

import ast
import inspect
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import full, randn

from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _execute_hooks,
    _convert_arg_to_type,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _move_tensors_to_device,
    _resolve_args,
    check_files,
    check_interface,
)
from base.model_structure_checker import ModelStructureChecker


def check_model_structure(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Base structure check with optional config and model pre-hooks.

    Base requires ``config_name`` and passes model arguments through unchanged.
    XiHe has no candidate config file and its Tensor mask is created by the
    same hook/context mechanism used elsewhere in Base.
    """
    rules = config.get("rules")
    if not rules:
        return [
            (
                "model_structure:rules",
                False,
                "config['rules'] is missing or empty",
            )
        ]

    CHECKPOINT_NAMES = ["model_structure:build_model_ok"]
    CHECKPOINT_NAMES += [
        f"model_structure:{rule.get('id', '<missing>')}" for rule in rules
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        # ------------------ 1. 加载模型模块 ------------------
        model_file = config.get("model_file")
        if not model_file:
            raise ValueError("config['model_file'] is missing or empty")
        model_module = _load_candidate_modules(model_file)

        model_class_name = config.get("model_class")
        if not model_class_name:
            raise ValueError("config['model_class'] is missing or empty")
        model_cls = _load_class(model_module, model_class_name)

        # ------------------ 2. 构建模型 ------------------
        config_name = config.get("config_name")
        candidate_config = (
            _load_candidate_config(config_name) if config_name else {}
        )
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)
        context = _execute_hooks(
            model_module,
            config.get("pre_model_hooks", []),
            context,
            globals(),
            context,
        )

        model_args = config.get("model_args")
        if not model_args:
            raise ValueError("config['model_args'] is required")
        resolved_model_args = _resolve_args(model_args, context)
        model = model_cls(**resolved_model_args).to(DEVICE).eval()
        update_check(
            current_index,
            True,
            f"Model built with config: {candidate_config}",
        )

        # ------------------ 3. 执行公共结构规则 ------------------
        checker = ModelStructureChecker()
        checker.add_rules_from_rules(rules)
        results = checker.check(model)
        for current_index, rule in enumerate(rules, 1):
            matched = [
                result for result in results if result.rule_id == rule["id"]
            ]
            update_check(
                current_index,
                bool(matched) and all(result.passed for result in matched),
                "; ".join(result.message for result in matched),
            )
    except Exception:
        update_check(
            min(current_index, len(checks) - 1),
            False,
            traceback.format_exc(),
        )
    return checks


def check_xihe_structure_contract(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Check only XiHe structure that the public rule engine cannot express."""
    import torch.nn as nn

    CHECKPOINT_NAMES = [
        "model_structure:xihe_model_build",
        "model_structure:group_vectors",
        "model_structure:registered_ocean_mask",
        "model_structure:self_contained_imports",
        "model_structure:paper_defaults",
        "model_structure:local_window_size",
        "model_structure:local_attention_heads",
        "model_structure:global_attention_heads",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

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
        context = _execute_hooks(
            model_module,
            config.get("pre_model_hooks", []),
            context,
            globals(),
            context,
        )
        model_args = _resolve_args(config.get("model_args") or {}, context)
        model = model_cls(**model_args).cpu().eval()
        update_check(0, True, model.__class__.__name__)
    except Exception:
        update_check(0, False, traceback.format_exc())
        return checks

    try:
        target = model
        target_path = config.get(
            "group_vectors_path",
            "ocean_blocks.0.global_sie.group_vectors",
        )
        for part in target_path.split("."):
            target = target[int(part)] if part.isdigit() else getattr(target, part)
        expected_shape = config.get("group_vectors_shape", [None, 16])
        actual_shape = tuple(int(item) for item in target.shape)
        shape_matches = len(actual_shape) == len(expected_shape) and all(
            expected is None or int(expected) == actual
            for actual, expected in zip(actual_shape, expected_shape)
        )
        passed = (
            isinstance(target, nn.Parameter)
            and target.requires_grad
            and shape_matches
        )
        update_check(
            1,
            passed,
            f"shape={actual_shape}, requires_grad={target.requires_grad}",
        )
    except Exception:
        update_check(1, False, traceback.format_exc())

    try:
        mask_names = set(
            config.get("mask_names")
            or ("mask", "ocean_mask", "land_sea_mask")
        )
        expected_shape = tuple(int(item) for item in config["mask_shape"])
        matches = [
            (name, value)
            for name, value in model.named_buffers()
            if name.rsplit(".", 1)[-1] in mask_names
        ]
        passed = bool(matches) and tuple(matches[0][1].shape) == expected_shape
        detail = (
            f"name={matches[0][0]}, shape={tuple(matches[0][1].shape)}"
            if matches
            else f"no registered buffer named one of {sorted(mask_names)}"
        )
        update_check(2, passed, detail)
    except Exception:
        update_check(2, False, traceback.format_exc())

    try:
        model_path = (Path(CANDIDATE_DIR) / model_file).resolve()
        tree = ast.parse(model_path.read_text(encoding="utf-8"))
        forbidden_roots = {
            str(item).lower()
            for item in config.get(
                "forbidden_import_roots",
                ("timm", "transformers", "mmcv", "modulus", "onescience"),
            )
        }
        forbidden_substrings = [
            str(item).lower()
            for item in config.get("forbidden_import_substrings", ("xihe",))
        ]
        disallowed = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                lowered = name.lower()
                if (
                    lowered.split(".", 1)[0] in forbidden_roots
                    or any(token in lowered for token in forbidden_substrings)
                ):
                    disallowed.append(name)
        update_check(
            3,
            not disallowed,
            f"disallowed_imports={sorted(set(disallowed))}",
        )
    except Exception:
        update_check(3, False, traceback.format_exc())

    try:
        signature = inspect.signature(model_cls.__init__)
        expected_defaults = config.get("expected_defaults") or {
            "img_size": [2041, 4320],
            "in_chans": 96,
            "out_chans": 94,
        }
        mismatches = []
        for name, expected in expected_defaults.items():
            parameter = signature.parameters.get(name)
            actual = None if parameter is None else parameter.default
            if isinstance(actual, tuple):
                actual = list(actual)
            if actual != expected:
                mismatches.append(f"{name}: expected {expected}, got {actual}")
        update_check(4, not mismatches, "; ".join(mismatches) or "defaults match")
    except Exception:
        update_check(4, False, traceback.format_exc())

    try:
        local_sie = model
        for part in config.get(
            "local_sie_path", "ocean_blocks.0.local_sies.0"
        ).split("."):
            local_sie = (
                local_sie[int(part)] if part.isdigit() else getattr(local_sie, part)
            )
        window_size = getattr(local_sie, "window_size", None)
        window_ok = (
            isinstance(window_size, (tuple, list))
            and len(window_size) == 2
            and all(int(item) > 0 for item in window_size)
        )
        update_check(5, window_ok, f"window_size={window_size}")

        local_heads = getattr(local_sie, "num_heads", None)
        local_dim = getattr(local_sie, "dim", None)
        local_heads_ok = (
            isinstance(local_heads, int)
            and local_heads > 0
            and isinstance(local_dim, int)
            and local_dim % local_heads == 0
        )
        update_check(6, local_heads_ok, f"heads={local_heads}, dim={local_dim}")
    except Exception:
        detail = traceback.format_exc()
        update_check(5, False, detail)
        update_check(6, False, detail)

    try:
        global_sie = model
        for part in config.get(
            "global_sie_path", "ocean_blocks.0.global_sie"
        ).split("."):
            global_sie = (
                global_sie[int(part)]
                if part.isdigit()
                else getattr(global_sie, part)
            )
        global_heads = getattr(global_sie, "num_heads", None)
        global_dim = getattr(global_sie, "dim", None)
        global_heads_ok = (
            isinstance(global_heads, int)
            and global_heads > 0
            and isinstance(global_dim, int)
            and global_dim % global_heads == 0
        )
        update_check(7, global_heads_ok, f"heads={global_heads}, dim={global_dim}")
    except Exception:
        update_check(7, False, traceback.format_exc())

    return checks


def check_model_forward_execution(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Base forward check with optional config and hook-created arguments.

    The Base flow is retained. XiHe adds finite-output and optional backward
    checks because both are explicit requirements of this paper-repro task.
    """
    CHECKPOINT_NAMES = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
        "forward_executable:finite_output",
        "forward_executable:input_unchanged",
    ]
    if config.get("check_backward"):
        CHECKPOINT_NAMES.append("forward_executable:backward")
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        # ------------------ 1. 加载模型模块 ------------------
        model_file = config.get("model_file")
        if not model_file:
            raise ValueError("config['model_file'] is missing or empty")
        model_module = _load_candidate_modules(model_file)

        # ------------------ 2. 加载可选配置 ------------------
        config_name = config.get("config_name")
        model_config = _load_candidate_config(config_name) if config_name else {}
        context = {"config": model_config}
        if isinstance(model_config, dict):
            context.update(model_config)

        # ------------------ 3. 执行前置钩子 ------------------
        context = _execute_hooks(
            model_module,
            config.get("pre_forward_hooks", []),
            context,
            globals(),
            context,
        )

        # ------------------ 4. 构建模型 ------------------
        model_class_name = config.get("model_class")
        if not model_class_name:
            raise ValueError("config['model_class'] is missing or empty")
        model_cls = _load_class(model_module, model_class_name)

        model_args = config.get("model_args")
        if not model_args:
            raise ValueError("config['model_args'] is required")
        resolved_model_args = _resolve_args(model_args, context)
        model = model_cls(**resolved_model_args).to(DEVICE).eval()

        # ------------------ 5. 解析并执行 forward ------------------
        resolved_forward_args = _resolve_args(
            config.get("forward_args", {}), context
        )
        forward_method = getattr(model, "forward", None)
        if forward_method is None or not callable(forward_method):
            raise ValueError("Model does not have a callable 'forward' method")
        signature = inspect.signature(forward_method)
        param_types = {
            name: parameter.annotation
            for name, parameter in signature.parameters.items()
            if name not in ("self", "args", "kwargs")
            and parameter.annotation != inspect._empty
        }
        resolved_forward_args = {
            name: _convert_arg_to_type(value, param_types.get(name))
            for name, value in resolved_forward_args.items()
        }
        resolved_forward_args = _move_tensors_to_device(resolved_forward_args)

        original_x = (
            resolved_forward_args["x"].clone()
            if torch.is_tensor(resolved_forward_args.get("x"))
            else None
        )
        with torch.no_grad():
            output = model(**resolved_forward_args)
        update_check(0, True, "Forward pass executed successfully")

        current_index = 1
        expected_shape = tuple(int(item) for item in config["out_shape"])
        is_tensor = torch.is_tensor(output)
        actual_shape = tuple(output.shape) if is_tensor else None
        update_check(
            1,
            is_tensor and actual_shape == expected_shape,
            f"actual={actual_shape}, expected={expected_shape}",
        )
        update_check(
            2,
            is_tensor and bool(torch.isfinite(output).all()),
            "all output values must be finite",
        )
        current_index = 3
        input_unchanged = (
            original_x is not None
            and torch.equal(resolved_forward_args["x"], original_x)
        )
        update_check(3, input_unchanged, "forward must not mutate x")

        if config.get("check_backward"):
            current_index = 4
            model.train()
            backward_args = dict(resolved_forward_args)
            x = backward_args.get("x")
            if not torch.is_tensor(x) or not x.is_floating_point():
                raise TypeError("forward_args.x must be a floating Tensor")
            backward_args["x"] = x.detach().clone().requires_grad_(True)
            loss = model(**backward_args).square().mean()
            loss.backward()
            parameter_grads = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            gradient_ok = (
                backward_args["x"].grad is not None
                and bool(torch.isfinite(backward_args["x"].grad).all())
                and bool(parameter_grads)
                and all(bool(torch.isfinite(grad).all()) for grad in parameter_grads)
            )
            update_check(4, gradient_ok, f"parameter_grads={len(parameter_grads)}")
    except Exception:
        update_check(
            min(current_index, len(checks) - 1),
            False,
            traceback.format_exc(),
        )
    return checks


def check_xihe_internal_behavior(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Observe that the required XiHe stages participate in real forward flow."""
    CHECKPOINT_NAMES = [
        "model_internal:build_model",
        "model_internal:forward_run",
        "model_internal:required_modules_used",
        "model_internal:stage_order",
        "model_internal:group_mixers_used",
        "model_internal:local_mask_affects_output",
        "model_internal:local_land_isolation",
        "model_internal:group_vectors_affect_output",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    handles = []
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
        context = _execute_hooks(
            model_module,
            config.get("pre_model_hooks", []),
            context,
            globals(),
            context,
        )
        model_args = _resolve_args(config.get("model_args") or {}, context)
        model = model_cls(**model_args).to(DEVICE).eval()
        update_check(0, True, model.__class__.__name__)

        expected_order = list(config.get("expected_stage_order") or [])
        required_internal = list(config.get("required_internal_modules") or [])
        observed = []
        for target_path in [*expected_order, *required_internal]:
            target = model
            for part in target_path.split("."):
                target = target[int(part)] if part.isdigit() else getattr(target, part)
            handles.append(
                target.register_forward_hook(
                    lambda _module, _args, _output, name=target_path: observed.append(name)
                )
            )

        resolved_forward_args = _resolve_args(
            config.get("forward_args") or {}, context
        )
        resolved_forward_args = _move_tensors_to_device(resolved_forward_args)
        with torch.no_grad():
            output = model(**resolved_forward_args)
        current_index = 1
        update_check(1, torch.is_tensor(output), f"output_type={type(output).__name__}")

        current_index = 2
        missing = [name for name in expected_order if name not in observed]
        update_check(2, not missing, f"missing={missing}; observed={observed}")

        current_index = 3
        filtered = [name for name in observed if name in expected_order]
        update_check(
            3,
            filtered == expected_order,
            f"observed={filtered}; expected={expected_order}",
        )

        current_index = 4
        missing_internal = [
            name for name in required_internal if name not in observed
        ]
        update_check(
            4,
            not missing_internal,
            f"missing={missing_internal}; observed={observed}",
        )

        feature_shape = tuple(
            int(item)
            for item in config.get("local_feature_shape", [1, 4, 6, 16])
        )
        generator = torch.Generator().manual_seed(
            int(config.get("internal_seed", 211))
        )
        features = torch.randn(*feature_shape, generator=generator).to(DEVICE)
        height, width = feature_shape[1:3]
        ocean_mask = torch.ones(height, width, dtype=torch.bool, device=DEVICE)
        mixed_mask = ocean_mask.clone()
        split = max(1, width // 2)
        mixed_mask[:, :split] = False

        local_sie = model.ocean_blocks[0].local_sies[0]
        with torch.no_grad():
            full_local = local_sie(features, ocean_mask)
            masked_local = local_sie(features, mixed_mask)
            land_changed = features.clone()
            land_changed[:, :, :split] += 100.0
            changed_local = local_sie(land_changed, mixed_mask)

        current_index = 5
        update_check(
            5,
            not torch.allclose(full_local, masked_local),
            "mixed local mask must alter Local SIE output",
        )
        current_index = 6
        update_check(
            6,
            torch.allclose(
                masked_local[:, :, split:],
                changed_local[:, :, split:],
                rtol=1e-5,
                atol=1e-6,
            ),
            f"ocean_columns={split}:{width}",
        )

        global_sie = model.ocean_blocks[0].global_sie
        current_index = 7
        with torch.no_grad():
            original_vectors = global_sie.group_vectors.detach().clone()
            baseline_global = global_sie(features, ocean_mask)
            delta = torch.linspace(
                -0.2,
                0.2,
                global_sie.group_vectors.numel(),
                dtype=global_sie.group_vectors.dtype,
                device=global_sie.group_vectors.device,
            ).reshape_as(global_sie.group_vectors)
            try:
                global_sie.group_vectors.add_(delta)
                changed_global = global_sie(features, ocean_mask)
            finally:
                global_sie.group_vectors.copy_(original_vectors)
        update_check(
            7,
            not torch.allclose(baseline_global, changed_global),
            "Global SIE output must depend on learnable group_vectors",
        )
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    finally:
        for handle in handles:
            handle.remove()
    return checks


def check_method_correctness(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Run grouped constructor or forward rejection cases for XiHe."""
    invalid_cases = config.get("invalid_cases") or []
    CHECKPOINT_NAMES = [
        f"invalid_input:{case.get('name', index)}"
        for index, case in enumerate(invalid_cases)
    ]
    if not CHECKPOINT_NAMES:
        return [("invalid_input:cases", False, "invalid_cases is missing or empty")]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

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
        context = _execute_hooks(
            model_module,
            config.get("pre_hooks", []),
            context,
            globals(),
            context,
        )
        model_args = _resolve_args(config.get("model_args") or {}, context)
        model = model_cls(**model_args).cpu().eval()
    except Exception:
        detail = traceback.format_exc()
        for index in range(len(checks)):
            update_check(index, False, detail)
        return checks

    for index, case in enumerate(invalid_cases):
        try:
            kind = case.get("kind")
            value = _resolve_args({"value": case.get("value")}, context)["value"]
            if kind == "constructor_mask":
                invalid_args = dict(model_args)
                invalid_args["mask"] = value
                model_cls(**invalid_args)
            elif kind == "forward_x":
                model(value)
            else:
                raise ValueError(f"Unsupported invalid case kind: {kind}")
        except (ValueError, TypeError, RuntimeError, AssertionError) as exc:
            update_check(index, True, f"rejected with {type(exc).__name__}")
        except Exception:
            update_check(index, False, traceback.format_exc())
        else:
            update_check(index, False, "invalid input was accepted")
    return checks


def check_infer_metrics(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Check XiHe mask semantics and invalid-input behavior without an oracle."""
    import torch

    behaviors = list(
        config.get("required_behaviors")
        or (
            "deterministic_eval",
            "input_sensitive",
            "mask_affects_prediction",
            "land_values_do_not_affect_ocean",
            "global_information_exchange",
        )
    )
    CHECKPOINT_NAMES = [f"evaluate_oracle:{name}" for name in behaviors]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]
    indices = {name: index for index, name in enumerate(behaviors)}

    def update_check(name: str, passed: bool, msg: str):
        if name in indices:
            index = indices[name]
            checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    case = config.get("case") or {}
    try:
        model_module = _load_candidate_modules(config["model_file"])
        model_cls = _load_class(model_module, config["model_class"])
        img_size = tuple(int(item) for item in case["img_size"])
        patch_size = tuple(int(item) for item in case["patch_size"])
        in_chans = int(case["in_chans"])
        out_chans = int(case["out_chans"])
        embed_dim = int(case["embed_dim"])
        seed = int(case.get("seed", 101))
    except Exception:
        detail = traceback.format_exc()
        for name in behaviors:
            update_check(name, False, detail)
        return checks

    def build(mask, model_seed=seed):
        torch.manual_seed(model_seed)
        return model_cls(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dim=embed_dim,
            mask=mask,
        ).cpu().eval()

    generator = torch.Generator().manual_seed(seed + 100)
    x = torch.randn(1, in_chans, *img_size, generator=generator)
    split = (img_size[1] // (2 * patch_size[1])) * patch_size[1]
    split = max(patch_size[1], min(split, img_size[1] - patch_size[1]))

    try:
        ocean_model = build(torch.ones(img_size))
    except Exception:
        detail = traceback.format_exc()
        for name in behaviors:
            update_check(name, False, detail)
        return checks

    try:
        with torch.no_grad():
            first_output = ocean_model(x)
            repeated_output = ocean_model(x)
            changed_input_output = ocean_model(x + 0.25)
        update_check(
            "deterministic_eval",
            torch.equal(first_output, repeated_output),
            "two eval-mode calls with the same input must match exactly",
        )
        update_check(
            "input_sensitive",
            not torch.allclose(first_output, changed_input_output),
            "prediction must change when the input changes",
        )
    except Exception:
        detail = traceback.format_exc()
        update_check("deterministic_eval", False, detail)
        update_check("input_sensitive", False, detail)

    mixed_mask = torch.ones(img_size)
    mixed_mask[:, :split] = 0.0
    try:
        full_model = build(torch.ones(img_size), seed)
        masked_model = build(mixed_mask, seed)
        with torch.no_grad():
            full_output = full_model(x)
            masked_output = masked_model(x)
        update_check(
            "mask_affects_prediction",
            not torch.allclose(full_output, masked_output),
            "a mixed mask must alter the prediction",
        )

        land_changed = x.clone()
        land_changed[..., :, :split] += 100.0
        with torch.no_grad():
            base_output = masked_model(x)
            changed_output = masked_model(land_changed)
        update_check(
            "land_values_do_not_affect_ocean",
            torch.allclose(
                base_output[..., :, split:],
                changed_output[..., :, split:],
                rtol=1e-5,
                atol=1e-6,
            ),
            f"ocean_columns={split}:{img_size[1]}",
        )
    except Exception:
        detail = traceback.format_exc()
        update_check("mask_affects_prediction", False, detail)
        update_check("land_values_do_not_affect_ocean", False, detail)

    try:
        localized = x.clone()
        localized[..., : patch_size[0], : patch_size[1]] += 1.0
        with torch.no_grad():
            base_output = ocean_model(x)
            localized_output = ocean_model(localized)
        remote_delta = (
            localized_output[..., -patch_size[0] :, -patch_size[1] :]
            - base_output[..., -patch_size[0] :, -patch_size[1] :]
        ).abs().max()
        update_check(
            "global_information_exchange",
            bool(remote_delta > 1e-9),
            f"remote_max_abs_delta={float(remote_delta)}",
        )
    except Exception:
        update_check("global_information_exchange", False, traceback.format_exc())

    return checks
