#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RNO Darcy evaluator built from Base with task-specific paper checks."""

from __future__ import annotations

import ast
import inspect
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _resolve_args,
    check_files,
    check_infer_output_schema,
    check_infer_structure,
    check_interface,
    check_parameter_update,
    check_train_structure,
)
from base.model_structure_checker import ModelStructureChecker


MODEL_CLASS = "RadonNeuralOperator2D"
DATA_ROOT = Path(
    "/public/share/sugonhpcapp01/onestore/onedatasets/CFD_Benchmark/darcy"
)
DATA_FILES = (
    "piececonst_r421_N1024_smooth1.mat",
    "piececonst_r421_N1024_smooth2.mat",
)
SEED = 20260825


def check_data_availability(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Base-style availability check adapted for the task's MATLAB files."""
    CHECKPOINT_NAMES = [
        "dataset_availability:root_exists",
        "dataset_availability:files_exist",
        "dataset_availability:files_readable",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        root = Path(config.get("data_root", DATA_ROOT)).expanduser().resolve()
        root_exists = root.is_dir()
        update_check(0, root_exists, f"data_root={root}")
        if not root_exists:
            return checks

        files = [root / name for name in config.get("data_files", DATA_FILES)]
        current_index = 1
        missing = [str(path) for path in files if not path.is_file()]
        update_check(1, not missing, f"missing={missing}")
        if missing:
            return checks

        current_index = 2
        unreadable = []
        sizes = {}
        for path in files:
            try:
                with path.open("rb") as stream:
                    header = stream.read(128)
                sizes[path.name] = path.stat().st_size
                if not header or sizes[path.name] <= 128:
                    unreadable.append(str(path))
            except OSError:
                unreadable.append(str(path))
        update_check(2, not unreadable, f"sizes={sizes}; unreadable={unreadable}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_data_schema(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Inspect MAT metadata without loading the full Darcy arrays."""
    CHECKPOINT_NAMES = [
        "dataset_schema:mat_metadata",
        "dataset_schema:required_fields",
        "dataset_schema:paired_shapes",
        "dataset_schema:paper_resolution_and_samples",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        root = Path(config.get("data_root", DATA_ROOT)).expanduser().resolve()
        names = list(config.get("data_files", DATA_FILES))
        required_fields = tuple(config.get("required_fields", ("coeff", "sol")))
        native_resolution = int(config.get("native_resolution", 421))
        sample_requirements = (
            int(config.get("train_samples", 1000)),
            int(config.get("test_samples", 200)),
        )
        metadata: dict[str, dict[str, tuple[int, ...]]] = {}
        for name in names:
            path = root / name
            fields: dict[str, tuple[int, ...]] = {}
            try:
                from scipy.io import whosmat

                fields = {
                    field_name: tuple(int(item) for item in shape)
                    for field_name, shape, _ in whosmat(path)
                }
            except (ImportError, NotImplementedError, OSError, ValueError):
                import h5py

                with h5py.File(path, "r") as handle:
                    fields = {
                        field_name: tuple(int(item) for item in dataset.shape)
                        for field_name, dataset in handle.items()
                        if hasattr(dataset, "shape")
                    }
            metadata[name] = fields
        update_check(0, len(metadata) == len(names), metadata)

        current_index = 1
        required_ok = all(
            all(field in fields for field in required_fields)
            for fields in metadata.values()
        )
        update_check(1, required_ok, f"required fields: {required_fields}")

        current_index = 2
        paired_ok = required_ok and all(
            len(required_fields) >= 2
            and fields[required_fields[0]] == fields[required_fields[1]]
            for fields in metadata.values()
        )
        update_check(2, paired_ok, metadata)

        current_index = 3
        paper_ok = required_ok
        details = []
        for index, name in enumerate(names):
            shape = metadata[name][required_fields[0]]
            spatial_ok = sum(dimension == native_resolution for dimension in shape) >= 2
            sample_ok = any(
                dimension >= sample_requirements[min(index, 1)]
                for dimension in shape
            )
            details.append(
                f"{name}: shape={shape}, spatial={spatial_ok}, samples={sample_ok}"
            )
            paper_ok = paper_ok and spatial_ok and sample_ok
        update_check(3, paper_ok, "; ".join(details))
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_structure(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Apply Base structure rules to an explicitly configured RNO model."""
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

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        model_file = config.get("model_file")
        if not model_file:
            raise ValueError("config['model_file'] is missing or empty")
        model_module = _load_candidate_modules(model_file)

        model_class_name = config.get("model_class")
        if not model_class_name:
            raise ValueError("config['model_class'] is missing or empty")
        model_cls = _load_class(model_module, model_class_name)

        config_name = config.get("config_name")
        candidate_config = (
            _load_candidate_config(config_name) if config_name else {}
        )
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)

        model_args = config.get("model_args")
        if not model_args:
            raise ValueError("config['model_args'] is required")
        resolved_model_args = _resolve_args(model_args, context)
        model = model_cls(**resolved_model_args).to(DEVICE).eval()
        update_check(
            0,
            True,
            f"Model built with config: {candidate_config}",
        )

        checker = ModelStructureChecker()
        checker.add_rules_from_rules(rules)
        rule_results = checker.check(model)
        for current_index, rule in enumerate(rules, 1):
            matched = [result for result in rule_results if result.rule_id == rule["id"]]
            update_check(
                current_index,
                bool(matched) and all(result.passed for result in matched),
                "; ".join(result.message for result in matched)
                or "No module matched the rule",
            )
    except Exception:
        update_check(min(current_index, len(checks) - 1), False, traceback.format_exc())
    return checks


def check_rno_structure_contract(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Check RNO structure that the public rule engine cannot express."""
    CHECKPOINT_NAMES = [
        "model_structure:rno_model_build",
        "model_structure:paper_config",
        "model_structure:attention_operation",
        "model_structure:radon_operation",
        "model_structure:angle_operation",
        "model_structure:operator_stack",
        "model_structure:independent_implementation",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
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
        model_args = _resolve_args(config.get("model_args") or {}, context)
        model = model_cls(**model_args).to(DEVICE).eval()
        update_check(0, True, model.__class__.__name__)
    except Exception:
        update_check(0, False, traceback.format_exc())
        return checks

    try:
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

        paper_config = {
            "channels_128": any(
                value == 128
                for key in ("hidden_dim", "channels")
                for value in config_values.get(key, [])
            ),
            "layers_8": any(
                value == 8
                for key in ("num_layers", "layers")
                for value in config_values.get(key, [])
            ),
            "heads_8": any(
                value == 8
                for key in ("num_heads", "heads")
                for value in config_values.get(key, [])
            ),
            "slices_32": any(
                value == 32
                for key in ("slice_num", "slices", "num_slices")
                for value in config_values.get(key, [])
            ),
            "angles_32": any(
                value == 32
                for key in ("num_angles", "angles")
                for value in config_values.get(key, [])
            ),
            "radon_blocks_4": any(
                value == 4
                for key in ("num_radon_blocks", "radon_blocks")
                for value in config_values.get(key, [])
            ),
            "scalar_output": any(
                value == 1
                for key in ("output_dim", "out_channels")
                for value in config_values.get(key, [])
            ),
        }
        update_check(1, all(paper_config.values()), paper_config)

        modules = list(model.modules())
        class_names = [type(module).__name__.lower() for module in modules]
        model_source = Path(CANDIDATE_DIR, model_file).read_text(encoding="utf-8")
        source = model_source.lower()

        attention_ok = "softmax" in source and any(
            token in source for token in ("heads", "num_heads", "operator_blocks")
        )
        update_check(2, attention_ok, "attention weights and multi-head operator path")

        radon_ok = isinstance(getattr(model, "radon_blocks", None), torch.nn.ModuleList)
        radon_ok = radon_ok and "radon" in source and any(
            token in source for token in ("grid_sample", "backprojection", "fft")
        )
        update_check(3, radon_ok, "differentiable Radon stage")

        angle_modules = [
            module
            for module in getattr(model, "radon_blocks", ()).modules()
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
        ]
        angle_ok = (
            "angle_kernel_size" in source
            and "angle_hidden_channels" in source
            and bool(angle_modules)
        )
        update_check(
            4,
            angle_ok,
            f"angle convolution modules={len(angle_modules)}",
        )

        radon_blocks = getattr(model, "radon_blocks", None)
        operator_blocks = getattr(model, "operator_blocks", None)
        stack_ok = (
            isinstance(radon_blocks, torch.nn.ModuleList)
            and len(radon_blocks) == int(model_args.get("radon_blocks", 0))
            and isinstance(operator_blocks, torch.nn.ModuleList)
            and len(operator_blocks) == int(model_args.get("layers", 0))
        )
        update_check(
            5,
            stack_ok,
            f"radon_blocks={len(radon_blocks) if isinstance(radon_blocks, torch.nn.ModuleList) else None}; "
            f"operator_blocks={len(operator_blocks) if isinstance(operator_blocks, torch.nn.ModuleList) else None}",
        )

        forbidden = (
            "github.com/wenbin-lu",
            "git clone",
            "codegen.standard",
            "codegen.evaluators",
        )
        found = [token for token in forbidden if token in source]
        update_check(6, not found, f"forbidden references={found}")
    except Exception:
        detail = traceback.format_exc()
        for index in range(1, len(checks)):
            if not checks[index][1]:
                update_check(index, False, detail)
    return checks


def check_rno_internal_behavior(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Observe that Radon and operator stages participate in forward flow."""
    CHECKPOINT_NAMES = [
        "model_internal:build_model",
        "model_internal:operator_modules",
        "model_internal:radon_modules",
        "model_internal:forward_run",
        "model_internal:operator_modules_used",
        "model_internal:radon_modules_used",
        "model_internal:interleaved_order",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    handles = []
    current_index = 0
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
        model = model_cls(**model_args).to(DEVICE).eval()
        update_check(0, True, model.__class__.__name__)

        operator_modules = [
            (f"operator_blocks.{index}", module)
            for index, module in enumerate(model.operator_blocks)
        ]
        radon_modules = [
            (f"radon_blocks.{index}", module)
            for index, module in enumerate(model.radon_blocks)
        ]
        current_index = 1
        update_check(1, bool(operator_modules), [name for name, _ in operator_modules])
        current_index = 2
        update_check(2, bool(radon_modules), [name for name, _ in radon_modules])

        observed: list[tuple[str, str]] = []
        for name, module in operator_modules:
            handles.append(
                module.register_forward_hook(
                    lambda _module, _args, _output, name=name: observed.append(
                        ("operator", name)
                    )
                )
            )
        for name, module in radon_modules:
            handles.append(
                module.register_forward_hook(
                    lambda _module, _args, _output, name=name: observed.append(
                        ("radon", name)
                    )
                )
            )

        coefficient = torch.linspace(-1.0, 1.0, 169, dtype=torch.float32)
        coefficient = coefficient.reshape(1, 1, 13, 13).to(DEVICE)
        with torch.no_grad():
            output = model(coefficient)
        current_index = 3
        output_ok = (
            torch.is_tensor(output)
            and tuple(output.shape) == tuple(coefficient.shape)
            and bool(torch.isfinite(output).all())
        )
        update_check(3, output_ok, f"output_type={type(output).__name__}")

        operator_used = {name for kind, name in observed if kind == "operator"}
        radon_used = {name for kind, name in observed if kind == "radon"}
        current_index = 4
        update_check(
            4,
            bool(operator_modules)
            and all(name in operator_used for name, _ in operator_modules),
            f"observed={sorted(operator_used)}",
        )
        current_index = 5
        update_check(
            5,
            bool(radon_modules) and all(name in radon_used for name, _ in radon_modules),
            f"observed={sorted(radon_used)}",
        )
        kinds = [kind for kind, _ in observed]
        current_index = 6
        order_ok = "operator" in kinds and "radon" in kinds
        if order_ok:
            order_ok = kinds.index("radon") < kinds.index("operator")
        update_check(6, order_ok, observed)
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    finally:
        for handle in handles:
            handle.remove()
    return checks


def check_model_forward_execution(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Run the RNO on multiple square grids with behavior checks."""
    sizes = [tuple(item) for item in config.get("spatial_sizes", ((17, 17), (13, 13)))]
    CHECKPOINT_NAMES = [
        f"model_forward:shape_finite_dynamic_{height}x{width}"
        for height, width in sizes
    ]
    if config.get("check_backward", False):
        CHECKPOINT_NAMES.append("model_forward:backward")
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
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
        model_args = _resolve_args(config.get("model_args") or {}, context)
    except Exception:
        detail = traceback.format_exc()
        for index in range(len(checks)):
            update_check(index, False, detail)
        return checks

    for index, (height, width) in enumerate(sizes):
        try:
            torch.manual_seed(SEED + index)
            model = model_cls(**model_args).to(DEVICE).eval()
            values = torch.linspace(-1.0, 1.0, height * width, dtype=torch.float32)
            coefficient = values.reshape(1, 1, height, width).repeat(2, 1, 1, 1).to(DEVICE)
            original = coefficient.clone()
            with torch.no_grad():
                output = model(coefficient)
                shifted = model(coefficient + 0.125)
            is_tensor = torch.is_tensor(output) and torch.is_tensor(shifted)
            finite = is_tensor and bool(torch.isfinite(output).all())
            unchanged = torch.equal(coefficient, original)
            sensitive = is_tensor and not torch.allclose(output, shifted)
            actual_shape = tuple(output.shape) if torch.is_tensor(output) else None
            passed = (
                is_tensor
                and actual_shape == tuple(coefficient.shape)
                and finite
                and unchanged
                and sensitive
            )
            update_check(
                index,
                passed,
                f"shape={actual_shape} finite={finite} "
                f"input_unchanged={unchanged} input_sensitive={sensitive}",
            )
        except Exception:
            update_check(index, False, traceback.format_exc())

    if config.get("check_backward", False):
        index = len(sizes)
        try:
            torch.manual_seed(SEED + 100)
            model = model_cls(**model_args).to(DEVICE).train()
            coefficient = torch.linspace(-1.0, 1.0, 121, dtype=torch.float32)
            coefficient = coefficient.reshape(1, 1, 11, 11).to(DEVICE)
            coefficient.requires_grad_(True)
            output = model(coefficient)
            if not torch.is_tensor(output):
                raise TypeError("forward must return a Tensor")
            output.square().mean().backward()
            parameter_gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            passed = (
                coefficient.grad is not None
                and bool(torch.isfinite(coefficient.grad).all())
                and bool(parameter_gradients)
                and all(
                    bool(torch.isfinite(gradient).all())
                    for gradient in parameter_gradients
                )
            )
            update_check(
                index,
                passed,
                f"input_gradient={coefficient.grad is not None}; "
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
    """Check grouped Darcy, training, or inference paper contracts."""
    contract = config.get("contract")
    definitions = {
        "darcy_data": (
            "config_mapping",
            "configured_data_root",
            "source_files",
            "coeff_sol_keys",
            "paper_split_1000_200",
            "resolution_421_stride5_to85",
            "float32_bchw",
            "train_only_normalization",
            "validation_and_finite_checks",
            "python_syntax",
            "no_repository_download",
        ),
        "paper_training": (
            "main_entrypoint",
            "paper_schedule",
            "adamw_and_scheduler",
            "physical_relative_and_gradient_loss",
            "real_optimization",
            "checkpoint_and_resume",
            "checkpoint_state",
            "live_and_persistent_logging",
        ),
        "paper_inference": (
            "main_entrypoint",
            "checkpoint_cli",
            "data_and_output_overrides",
            "checkpoint_normalizer",
            "paper_relative_l2_and_gap",
            "json_and_predictions",
        ),
        "model_invalid_input": (
            "build_model",
            "reject_rank",
            "reject_channel_count",
        ),
    }
    names = definitions.get(contract)
    if names is None:
        return [("method_check:contract", False, f"Unsupported contract: {contract}")]
    CHECKPOINT_NAMES = [f"method_check:{contract}:{name}" for name in names]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    try:
        if contract == "model_invalid_input":
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
            model = model_cls(**model_args).cpu().eval()
            update_check(0, True, type(model).__name__)
            invalid_inputs = (
                torch.randn(1, 13, 15),
                torch.randn(1, 2, 13, 13),
            )
            for index, value in enumerate(invalid_inputs, 1):
                try:
                    model(value)
                except (ValueError, TypeError, RuntimeError, AssertionError) as error:
                    update_check(index, True, f"rejected with {type(error).__name__}")
                except Exception:
                    update_check(index, False, traceback.format_exc())
                else:
                    update_check(index, False, "invalid input was accepted")
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
        combined = "\n".join(sources.values()).lower()
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

        if contract == "darcy_data":
            roots = [
                value
                for key in ("root", "data_root", "data_path")
                for value in config_values.get(key, [])
            ]
            root_ok = False
            for value in roots:
                try:
                    if Path(str(value)).expanduser().resolve() == DATA_ROOT.resolve():
                        root_ok = True
                        break
                except (OSError, TypeError, ValueError):
                    pass
            dataset_source = sources.get("dataset.py", "").lower()
            compact_dataset = "".join(dataset_source.split())
            values = (
                (isinstance(candidate_config, dict), f"keys={len(candidate_config)}"),
                (root_ok, f"configured roots={roots}"),
                (
                    any(
                        token in dataset_source
                        for token in ("loadmat", "h5py", "mat73")
                    )
                    and any(
                        token in dataset_source
                        for token in (".mat", "glob(", "rglob(")
                    ),
                    "MAT source discovery and loading",
                ),
                (
                    all(
                        name in dataset_source or name in config_text
                        for name in ("coeff", "sol")
                    ),
                    "coeff and sol aliases",
                ),
                (
                    any(
                        value == 1000
                        for key in ("n_train", "train_samples", "train_count")
                        for value in config_values.get(key, [])
                    )
                    and any(
                        value == 200
                        for key in ("n_test", "test_samples", "test_count")
                        for value in config_values.get(key, [])
                    ),
                    "train=1000 test=200",
                ),
                (
                    85 in config_values.get("resolution", [])
                    and (
                        "(source-1)//(self.resolution-1)" in compact_dataset
                        or "(source-1)//(resolution-1)" in compact_dataset
                        or any(
                            value == 5
                            for key in ("downsample", "subsample", "stride")
                            for value in config_values.get(key, [])
                        )
                    ),
                    "paper resolution 85 with endpoint-preserving subsampling",
                ),
                (
                    any(token in dataset_source for token in ("float32", ".float()"))
                    and any(token in dataset_source for token in ("unsqueeze(1)", "[:, none", "[:,none", "reshape")),
                    "float32 NCHW conversion",
                ),
                (
                    "normalize" in dataset_source
                    and "train.stats" in compact_dataset
                    and "test" in dataset_source,
                    "training statistics reused by held-out splits",
                ),
                (
                    "keyerror" in dataset_source
                    and any(
                        token in dataset_source
                        for token in ("valueerror", "typeerror", "available keys")
                    ),
                    "missing keys and invalid data are rejected",
                ),
                (len(syntax) == len([name for name in files if name.endswith(".py")]), syntax),
                (
                    not any(token in combined for token in ("git clone", "github.com/wenbin-lu", "wget ", "curl ")),
                    "independent implementation",
                ),
            )
        elif contract == "paper_training":
            train_source = sources.get("train.py", "")
            train_lower = train_source.lower()
            tree = ast.parse(train_source)
            has_main = any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and any(
                    isinstance(item, ast.Constant) and item.value == "__main__"
                    for item in node.test.comparators
                )
                for node in ast.walk(tree)
            ) or any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "main"
                for node in ast.walk(tree)
            )
            values = (
                (has_main, "main entrypoint"),
                (
                    500 in config_values.get("epochs", [])
                    and any(
                        isinstance(value, (int, float))
                        and math.isclose(
                            float(value),
                            1.0e-3,
                            rel_tol=1.0e-9,
                            abs_tol=1.0e-12,
                        )
                        for key in ("learning_rate", "lr")
                        for value in config_values.get(key, [])
                    ),
                    "epochs=500 lr=1e-3",
                ),
                ("adamw" in combined and any(token in combined for token in ("onecycle", "scheduler")), "AdamW and scheduler"),
                ("relative" in combined and "gradient" in combined and "0.1" in config_text, "relative + 0.1 gradient loss"),
                ("backward(" in combined and any(token in combined for token in ("optimizer.step", "scaler.step")), "backward and optimizer step"),
                (any(token in combined for token in ("torch.save", "atomic_torch_save")) and "resume" in train_lower, "checkpoint and resume"),
                (all(token in combined for token in ("model", "optimizer", "scheduler", "config", "epoch")), "complete resumable checkpoint state"),
                (any(token in train_lower for token in ("print(", "logger", "tqdm")) and any(token in combined for token in ("jsonl", "csv", "history", "log")), "live and persistent logging"),
            )
        else:
            evaluate_source = sources.get("evaluate.py", "")
            evaluate_lower = evaluate_source.lower()
            tree = ast.parse(evaluate_source)
            has_main = any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and any(
                    isinstance(item, ast.Constant) and item.value == "__main__"
                    for item in node.test.comparators
                )
                for node in ast.walk(tree)
            ) or any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "main"
                for node in ast.walk(tree)
            )
            values = (
                (has_main, "main entrypoint"),
                ("--checkpoint" in evaluate_lower and "--config" in evaluate_lower, "checkpoint and config CLI"),
                (
                    "build_datasets" in evaluate_lower
                    and any(
                        token in config_text
                        for token in ("output_dir", "results_file", "predictions_file")
                    ),
                    "config-driven data and output paths",
                ),
                (
                    "build_datasets" in evaluate_lower
                    and "train.stats" in "".join(combined.split())
                    and not any(
                        token in evaluate_lower
                        for token in ("normalizer.fit", "fit_normalizer")
                    ),
                    "training statistics reused without test refit",
                ),
                (
                    "relative" in evaluate_lower
                    and "paper_relative_l2" in evaluate_lower
                    and "absolute_gap_to_paper" in evaluate_lower
                    and "0.0051" in config_text,
                    "paper relative L2 and absolute gap",
                ),
                (any(token in evaluate_lower for token in ("json", "write_json", "save_json")) and any(token in evaluate_lower for token in ("savez", "npz", "prediction")), "metrics and predictions persisted"),
            )

        for index, (passed, detail) in enumerate(values):
            update_check(index, passed, detail)
    except Exception:
        detail = traceback.format_exc()
        for index, (_, passed, _) in enumerate(checks):
            if not passed:
                update_check(index, False, detail)
    return checks


def check_smoke_training(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Perform one real optimizer step on a small RNO instance."""
    CHECKPOINT_NAMES = [
        "training:model_and_optimizer",
        "training:finite_loss_and_gradients",
        "training:parameter_update",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        torch.manual_seed(int(config.get("seed", SEED)))
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
        model = model_cls(**model_args).to(DEVICE).train()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("Model has no trainable parameters")
        optimizer = torch.optim.AdamW(parameters, lr=float(config.get("learning_rate", 1.0e-3)))
        update_check(0, True, f"trainable_parameters={sum(item.numel() for item in parameters)}")

        values = torch.linspace(-1.0, 1.0, 81, dtype=torch.float32)
        coefficient = values.reshape(1, 1, 9, 9).repeat(2, 1, 1, 1).to(DEVICE)
        target = torch.cos(coefficient * torch.pi)
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.zero_grad(set_to_none=True)
        output = model(coefficient)
        if not torch.is_tensor(output) or output.shape != target.shape:
            raise TypeError("forward must return a Tensor with the target shape")
        loss = (output - target).square().mean()
        loss.backward()
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        gradients_ok = bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        current_index = 1
        update_check(1, bool(torch.isfinite(loss)) and gradients_ok, f"loss={float(loss.detach()):.6g} finite_gradients={len(gradients)}")

        optimizer.step()
        changed = any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, parameters)
        )
        current_index = 2
        update_check(2, changed, f"parameters_changed={changed}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_train_loss(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Dynamically validate Relative L2 and gradient loss behavior."""
    CHECKPOINT_NAMES = [
        "train_loss:relative_l2_callable",
        "train_loss:gradient_loss_callable",
        "train_loss:zero_at_identity",
        "train_loss:positive_for_error",
        "train_loss:finite_scalar_outputs",
        "train_loss:differentiable",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        functions: dict[str, Any] = {}
        locations: dict[str, str] = {}
        for relative in config.get("files", []):
            module = _load_candidate_modules(relative)
            for name in ("relative_l2", "gradient_loss", "spatial_gradient_loss"):
                candidate = getattr(module, name, None)
                if callable(candidate):
                    if name == "spatial_gradient_loss" and "gradient_loss" not in functions:
                        functions["gradient_loss"] = candidate
                        locations["gradient_loss"] = relative
                    elif name != "spatial_gradient_loss" and name not in functions:
                        functions[name] = candidate
                        locations[name] = relative
        update_check(0, "relative_l2" in functions, locations.get("relative_l2"))
        current_index = 1
        update_check(1, "gradient_loss" in functions, locations.get("gradient_loss"))
        if len(functions) != 2:
            return checks

        target = torch.linspace(0.25, 1.25, 60, dtype=torch.float32)
        target = target.reshape(2, 1, 5, 6)
        target[..., 0, :] = 0.0
        target[..., -1, :] = 0.0
        target[..., :, 0] = 0.0
        target[..., :, -1] = 0.0
        identical = target.detach().clone()
        relative_zero = functions["relative_l2"](identical, target)
        gradient_zero = functions["gradient_loss"](identical, target)
        relative_zero_value = float(relative_zero.detach()) if torch.is_tensor(relative_zero) else float(relative_zero)
        gradient_zero_value = float(gradient_zero.detach()) if torch.is_tensor(gradient_zero) else float(gradient_zero)
        current_index = 2
        update_check(
            2,
            abs(relative_zero_value) <= 1.0e-7 and abs(gradient_zero_value) <= 1.0e-7,
            f"relative={relative_zero_value:.3e}; gradient={gradient_zero_value:.3e}",
        )

        prediction = (target * 0.8 + 0.1).detach().requires_grad_(True)
        relative = functions["relative_l2"](prediction, target)
        gradient = functions["gradient_loss"](prediction, target)
        relative_value = float(relative.detach()) if torch.is_tensor(relative) else float(relative)
        gradient_value = float(gradient.detach()) if torch.is_tensor(gradient) else float(gradient)
        current_index = 3
        update_check(3, relative_value > 0.0 and gradient_value > 0.0, f"relative={relative_value:.6g}; gradient={gradient_value:.6g}")

        current_index = 4
        scalar_finite = (
            torch.is_tensor(relative)
            and relative.ndim == 0
            and bool(torch.isfinite(relative))
            and torch.is_tensor(gradient)
            and gradient.ndim == 0
            and bool(torch.isfinite(gradient))
        )
        update_check(4, scalar_finite, "both losses must be finite scalar Tensors")

        current_index = 5
        total = relative + float(config.get("gradient_weight", 0.1)) * gradient
        total.backward()
        differentiable = prediction.grad is not None and bool(torch.isfinite(prediction.grad).all())
        update_check(5, differentiable, f"prediction_gradient={prediction.grad is not None}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_execution(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Execute the candidate's held-out evaluation function on synthetic data."""
    CHECKPOINT_NAMES = [
        "inference_execution:module_load",
        "inference_execution:model_build",
        "inference_execution:function_resolve",
        "inference_execution:infer_call",
        "inference_execution:output_save",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        infer_module = _load_candidate_modules(config["run_file"])
        update_check(0, True, config["run_file"])

        current_index = 1
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
        model = model_cls(**model_args).cpu().eval()
        update_check(1, True, type(model).__name__)

        current_index = 2
        infer_function = None
        function_name = None
        requested_names = list(
            config.get("infer_functions", ("evaluate_test_split", "evaluate"))
        )
        for name in requested_names:
            candidate = getattr(infer_module, name, None)
            if callable(candidate):
                infer_function = candidate
                function_name = name
                break
        if infer_function is None:
            function_name = "synthetic_model_forward"
        update_check(2, True, function_name)

        coefficient = torch.linspace(-1.0, 1.0, 81, dtype=torch.float32)
        coefficient = coefficient.reshape(1, 1, 9, 9)
        target = torch.cos(coefficient * torch.pi)

        current_index = 3
        if infer_function is None:
            with torch.no_grad():
                prediction = model(coefficient)
            if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
                raise ValueError(
                    "Synthetic inference must return a finite Tensor with target shape"
                )
            numerator = torch.linalg.vector_norm((prediction - target).reshape(1, -1), dim=1)
            denominator = torch.linalg.vector_norm(target.reshape(1, -1), dim=1).clamp_min(1.0e-12)
            result = {"test_relative_l2": float((numerator / denominator).mean())}
        else:
            identity_normalizer = SimpleNamespace(
                encode=lambda value: value,
                decode=lambda value: value,
            )
            available = {
                "model": model,
                "loader": [(coefficient, target)],
                "data_loader": [(coefficient, target)],
                "test_loader": [(coefficient, target)],
                "dataloader": [(coefficient, target)],
                "normalizer": identity_normalizer,
                "output_normalizer": identity_normalizer,
                "device": torch.device("cpu"),
                "config": {"save_predictions": False},
                "save_predictions": False,
            }
            signature = inspect.signature(infer_function)
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
                raise TypeError(f"Unsupported required inference arguments: {missing}")
            with torch.no_grad():
                result = infer_function(**arguments)
        if result is None:
            raise ValueError("Inference function returned None")
        update_check(3, True, f"result_type={type(result).__name__}")

        current_index = 4
        output_path = Path(config["output_path"])
        if not output_path.is_absolute():
            output_path = Path(CANDIDATE_DIR, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=lambda value: (
                    value.detach().cpu().tolist()
                    if torch.is_tensor(value)
                    else value.tolist()
                    if hasattr(value, "tolist")
                    else str(value)
                    if isinstance(value, Path)
                    else None
                ),
            ),
            encoding="utf-8",
        )
        update_check(4, output_path.is_file() and output_path.stat().st_size > 0, output_path)
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_output_validity(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Validate semantic content of the synthetic inference result."""
    CHECKPOINT_NAMES = [
        "inference_output:structured_result",
        "inference_output:relative_l2_present",
        "inference_output:metric_nonnegative",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        output_path = Path(config["output_path"])
        if not output_path.is_absolute():
            output_path = Path(CANDIDATE_DIR, output_path)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        structured = isinstance(result, Mapping) and bool(result)
        update_check(0, structured, f"type={type(result).__name__}")
        if not structured:
            return checks

        metric_names = tuple(
            config.get(
                "metric_names",
                ("relative_l2", "relative_l2_error", "test_relative_l2"),
            )
        )
        metric_name = next((name for name in metric_names if name in result), None)
        current_index = 1
        update_check(1, metric_name is not None, f"keys={sorted(result)}")
        if metric_name is None:
            return checks

        metric = result[metric_name]
        if torch.is_tensor(metric):
            metric = float(metric.detach().cpu())
        current_index = 2
        valid = isinstance(metric, (int, float)) and math.isfinite(float(metric)) and float(metric) >= 0.0
        update_check(2, valid, f"{metric_name}={metric}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_metrics(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Validate the paper Relative L2 formula and declared lower-is-better target."""
    CHECKPOINT_NAMES = [
        "infer_metric:relative_l2_callable",
        "infer_metric:per_sample_formula",
        "infer_metric:paper_baseline",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        function = None
        location = None
        for relative in config.get("files", []):
            module = _load_candidate_modules(relative)
            candidate = getattr(module, config.get("function", "relative_l2"), None)
            if callable(candidate):
                function = candidate
                location = relative
                break
        if function is None:
            raise AttributeError("relative_l2 callable was not found")
        update_check(0, True, f"loaded from {location}")

        target = torch.stack(
            (torch.ones(1, 2, 2), torch.full((1, 2, 2), 2.0)), dim=0
        )
        prediction = torch.stack(
            (torch.zeros(1, 2, 2), torch.ones(1, 2, 2)), dim=0
        )
        result = function(prediction, target)
        value = float(result.detach().cpu()) if torch.is_tensor(result) else float(result)
        expected = 0.75
        current_index = 1
        update_check(1, math.isfinite(value) and math.isclose(value, expected, rel_tol=1.0e-5, abs_tol=1.0e-6), f"actual={value:.8g} expected={expected}")

        baseline = float(config.get("baseline", 0.0051))
        mode = str(config.get("mode", "min")).lower()
        current_index = 2
        update_check(2, math.isclose(baseline, 0.0051, abs_tol=1.0e-12) and mode == "min", f"relative_l2 baseline={baseline} mode={mode}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks
