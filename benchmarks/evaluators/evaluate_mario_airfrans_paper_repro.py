#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base-oriented evaluator for the MARIO AirfRANS scarce reproduction task."""

from __future__ import annotations

import ast
import inspect
import json
import math
import traceback
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


DATA_ROOT = Path(
    "/public/share/sugonhpcapp01/onestore/onedatasets/"
    "Transolver-Airfoil-Design/Dataset"
)
SEED = 20260810


def check_train_structure(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Keep Base results while accepting MARIO's INR training conventions."""
    checks = base_evaluator.check_train_structure(config)
    train_mode_index = next(
        (
            index
            for index, (name, _passed, _message) in enumerate(checks)
            if name == "train_start:has_train_mode"
        ),
        None,
    )
    dataloader_index = next(
        (
            index
            for index, (name, _passed, _message) in enumerate(checks)
            if name == "train_start:has_dataloader"
        ),
        None,
    )
    needs_train_mode = (
        train_mode_index is not None and not checks[train_mode_index][1]
    )
    needs_batching = (
        dataloader_index is not None and not checks[dataloader_index][1]
    )
    if not needs_train_mode and not needs_batching:
        return checks

    try:
        train_calls: list[str] = []
        manual_batch_loops: list[str] = []
        for relative in config.get("files") or []:
            path = Path(CANDIDATE_DIR, relative)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "train"
                ):
                    train_calls.append(ast.unparse(node.func.value))
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                    and len(node.iter.args) >= 3
                    and "batch" in ast.unparse(node.iter.args[2]).lower()
                ):
                    manual_batch_loops.append(f"{relative}:{getattr(node, 'lineno', '?')}")
        if needs_train_mode and train_calls:
            checks[train_mode_index] = (
                checks[train_mode_index][0],
                True,
                "train() called on: " + ", ".join(sorted(set(train_calls))),
            )
        if needs_batching and manual_batch_loops:
            checks[dataloader_index] = (
                checks[dataloader_index][0],
                True,
                "Explicit mini-batch range loops found at: "
                + ", ".join(manual_batch_loops),
            )
    except Exception:
        failure = traceback.format_exc()
        if needs_train_mode:
            checks[train_mode_index] = (
                checks[train_mode_index][0],
                False,
                failure,
            )
        if needs_batching:
            checks[dataloader_index] = (
                checks[dataloader_index][0],
                False,
                failure,
            )
    return checks


def check_data_availability(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Base-style availability check for the AirfRANS scarce split."""
    CHECKPOINT_NAMES = [
        "dataset_availability:root_exists",
        "dataset_availability:manifest_exists",
        "dataset_availability:paper_split_counts",
        "dataset_availability:splits_disjoint",
        "dataset_availability:sample_vtk_triplets",
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
        update_check(0, root.is_dir(), f"data_root={root}")
        if not root.is_dir():
            return checks

        current_index = 1
        manifest_path = root / str(config.get("manifest_name", "manifest.json"))
        update_check(1, manifest_path.is_file(), manifest_path)
        if not manifest_path.is_file():
            return checks

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_name = str(config.get("train_split", "scarce_train"))
        test_name = str(config.get("test_split", "full_test"))
        train_cases = list(manifest.get(train_name, []))
        test_cases = list(manifest.get(test_name, []))
        expected_train = int(config.get("expected_train_cases", 200))
        expected_test = int(config.get("expected_test_cases", 200))

        current_index = 2
        counts_ok = (
            len(train_cases) == expected_train
            and len(test_cases) == expected_test
        )
        update_check(
            2,
            counts_ok,
            f"{train_name}={len(train_cases)}; {test_name}={len(test_cases)}",
        )

        current_index = 3
        disjoint = set(train_cases).isdisjoint(test_cases)
        update_check(3, disjoint, f"overlap={len(set(train_cases) & set(test_cases))}")

        current_index = 4
        missing: list[str] = []
        for case_name in train_cases[:1] + test_cases[:1]:
            case_dir = root / str(case_name)
            for suffix in ("internal.vtu", "aerofoil.vtp", "freestream.vtp"):
                path = case_dir / f"{case_name}_{suffix}"
                if not path.is_file():
                    missing.append(str(path))
        update_check(4, bool(train_cases and test_cases) and not missing, f"missing={missing}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_structure(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Apply Base structure rules with MARIO constructor-alias adaptation."""
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
        logical_args = _resolve_args(config.get("model_args") or {}, context)
        parameters = inspect.signature(model_cls.__init__).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        aliases = {
            "input_dim": ("input_dim", "in_dim", "dim_in"),
            "output_dim": ("output_dim", "out_dim", "dim_out"),
            "latent_dim": ("latent_dim", "condition_dim", "cond_dim", "modulation_dim"),
            "width": ("width", "hidden_dim", "dim_hidden"),
            "depth": ("depth", "num_layers", "n_layers"),
            "num_frequencies": (
                "num_frequencies",
                "num_fourier_features",
                "fourier_features",
                "mapping_size",
            ),
            "hypernetwork_depth": (
                "hypernetwork_depth",
                "depth_hnn",
                "hypernet_depth",
                "hyper_layers",
            ),
            "fourier_scale": ("fourier_scale", "scale", "sigma"),
            "fourier_scales": ("fourier_scales", "scales", "sigmas"),
            "scalar_output_dim": ("scalar_output_dim", "scalar_out_dim"),
        }
        required = {
            "input_dim",
            "output_dim",
            "latent_dim",
            "width",
            "depth",
            "num_frequencies",
            "hypernetwork_depth",
        }
        model_args: dict[str, Any] = {}
        missing: list[str] = []
        for name, value in logical_args.items():
            names = aliases.get(name, (name,))
            matched = next((item for item in names if item in parameters), None)
            if matched is not None:
                model_args[matched] = value
            elif accepts_kwargs:
                model_args[name] = value
            elif name in required:
                missing.append("/".join(names))
        if missing:
            raise TypeError(f"Missing required constructor fields: {missing}")

        model = model_cls(**model_args).to(DEVICE).eval()
        update_check(0, True, f"model={model_class_name}; args={sorted(model_args)}")

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


def check_method_correctness(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Validate the MARIO-specific contracts that Base cannot express."""
    contract = str(config.get("contract", ""))
    contract_checkpoints = {
        "mario_airfrans_data": [
            "data_contract:configs_loaded",
            "data_contract:dataset_root",
            "data_contract:scarce_protocol",
            "data_contract:vtk_reader",
            "data_contract:feature_schema",
            "data_contract:independent_implementation",
        ],
        "mario_architecture": [
            "model_contract:required_classes",
            "model_contract:forward_interfaces",
            "model_contract:fourier_encoding",
            "model_contract:latent_hypernetwork",
            "model_contract:shift_modulation",
            "model_contract:multiscale_branches",
            "model_contract:field_output_projection",
            "model_contract:independent_implementation",
        ],
        "mario_latent_adaptation": [
            "latent_adaptation:source_contract",
            "latent_adaptation:model_build",
            "latent_adaptation:finite_gradient",
            "latent_adaptation:latent_update",
            "latent_adaptation:finite_outer_loss",
            "latent_adaptation:model_gradient",
        ],
        "mario_training": [
            "training_contract:two_stage_entrypoints",
            "training_contract:two_stage_loops",
            "training_contract:sdf_latent_stage",
            "training_contract:conditioned_flow_stage",
            "training_contract:optimizer_and_mse",
            "training_contract:loss_reporting",
            "training_contract:checkpoint_state",
            "training_contract:trainings_artifacts",
        ],
        "mario_inference": [
            "inference_contract:entrypoint",
            "inference_contract:checkpoint_loading",
            "inference_contract:eval_without_gradients",
            "inference_contract:data_and_model_rebuild",
            "inference_contract:field_metrics",
            "inference_contract:result_persistence",
        ],
    }
    CHECKPOINT_NAMES = contract_checkpoints.get(contract, ["method_contract:known"])
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        if contract == "mario_airfrans_data":
            configs = [
                _load_candidate_config(name)
                for name in config.get("config_names", [])
            ]
            if len(configs) != 2 or not all(isinstance(item, Mapping) for item in configs):
                raise TypeError("config_sdf.yaml and config_out.yaml must be mappings")
            update_check(0, True, f"loaded_configs={len(configs)}")

            sources = []
            for relative in config.get("files", []):
                sources.append(Path(CANDIDATE_DIR, relative).read_text(encoding="utf-8"))
            source = "\n".join(sources)
            lower = source.lower()
            config_text = json.dumps(configs, ensure_ascii=False, default=str).lower()
            expected_root = str(config.get("data_root", DATA_ROOT)).lower()

            current_index = 1
            root_ok = expected_root in config_text or (
                "airfrans_root" in config_text and expected_root in lower
            )
            update_check(1, root_ok, f"expected_root={expected_root}")

            current_index = 2
            scarce_ok = "scarce" in config_text or "scarce" in lower
            split_ok = any(token in lower for token in ("scarce_train", "full_test", "manifest"))
            update_check(2, scarce_ok and split_ok, "scarce train/full-test protocol")

            current_index = 3
            vtk_ok = (
                any(token in lower for token in ("pyvista", "airfrans", "vtk"))
                and any(token in lower for token in (".vtu", "internal"))
                and any(token in lower for token in (".vtp", "aerofoil", "freestream"))
            )
            update_check(3, vtk_ok, "VTK internal/aerofoil/freestream reader")

            current_index = 4
            feature_ok = (
                "sdf" in lower
                and "normal" in lower
                and any(token in lower for token in ("velocity", "pressure", "target", "output"))
                and any(token in lower for token in ("coordinate", "coords", "position", "pos"))
            )
            update_check(4, feature_ok, "coordinates, SDF, normals and flow targets")

            current_index = 5
            forbidden = (
                "git clone",
                "github.com",
                "codegen.standard",
                "codegen.evaluators",
            )
            found = [token for token in forbidden if token in lower]
            update_check(5, not found, f"forbidden_references={found}")

        elif contract == "mario_architecture":
            model_file = str(config["model_file"])
            module = _load_candidate_modules(model_file)
            source = Path(CANDIDATE_DIR, model_file).read_text(encoding="utf-8")
            lower = source.lower()
            class_names = list(config.get("model_classes", []))
            classes = [getattr(module, name, None) for name in class_names]

            current_index = 0
            class_ok = bool(classes) and all(
                inspect.isclass(item) and issubclass(item, torch.nn.Module)
                for item in classes
            )
            update_check(0, class_ok, f"classes={class_names}")

            current_index = 1
            interface_ok = class_ok and all(
                callable(getattr(item, "modulated_forward", None))
                or getattr(item, "forward", None) is not torch.nn.Module.forward
                for item in classes
            )
            update_check(1, interface_ok, "forward or modulated_forward")

            current_index = 2
            fourier_ok = (
                "fourier" in lower
                and any(token in lower for token in ("torch.sin", ".sin(", "sin("))
                and any(token in lower for token in ("torch.cos", ".cos(", "cos("))
            )
            update_check(2, fourier_ok, "sine/cosine Fourier coordinate encoding")

            current_index = 3
            hyper_ok = (
                "latent" in lower
                and "modulat" in lower
                and any(token in lower for token in ("hyper", "depth_hnn", "num_modulations"))
                and "linear" in lower
            )
            update_check(3, hyper_ok, "latent-to-layer modulation network")

            current_index = 4
            additive_modulation = any(
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Add)
                and "modulat" in ast.unparse(node).lower()
                for node in ast.walk(ast.parse(source))
            )
            shift_ok = (
                "shift_modulation" in lower
                or additive_modulation
                or (
                    "modulat" in lower
                    and any(token in lower for token in ("residual", "res *", "res*"))
                )
            )
            update_check(4, shift_ok, "shift-style modulated residual path")

            current_index = 5
            multi_cls = getattr(module, "MultiScaleModulatedFourierFeatures", None)
            multi_source = inspect.getsource(multi_cls).lower() if inspect.isclass(multi_cls) else ""
            multiscale_ok = (
                "scale" in multi_source
                and "fourier" in multi_source
                and any(token in multi_source for token in ("modulelist", "for ", "cat(", "concatenate"))
            )
            update_check(5, multiscale_ok, "multiple Fourier scales and feature fusion")

            current_index = 6
            projection_ok = (
                any(token in multi_source for token in ("output_dim", "out_dim", "dim_out"))
                and any(token in multi_source for token in ("linear", "output", "projection", "last_layer"))
            )
            update_check(6, projection_ok, "field output projection")

            current_index = 7
            forbidden = (
                "git clone",
                "github.com",
                "codegen.standard",
                "codegen.evaluators",
            )
            found = [token for token in forbidden if token in lower]
            update_check(7, not found, f"forbidden_references={found}")

        elif contract == "mario_latent_adaptation":
            model_file = str(config["model_file"])
            module = _load_candidate_modules(model_file)
            train_path = Path(CANDIDATE_DIR, model_file).with_name("train_sdf.py")
            source = (
                Path(CANDIDATE_DIR, model_file).read_text(encoding="utf-8")
                + "\n"
                + train_path.read_text(encoding="utf-8")
            ).lower()
            source_ok = (
                "latent" in source
                and any(token in source for token in ("autograd.grad", "inner_loop", "adapt_latent", "graph_outer_step"))
            )
            update_check(0, source_ok, "differentiable latent adaptation source path")

            model_cls = _load_class(module, str(config["model_class"]))
            candidate_config = (
                _load_candidate_config(config["config_name"])
                if config.get("config_name")
                else {}
            )
            context = {"config": candidate_config}
            if isinstance(candidate_config, dict):
                context.update(candidate_config)
            logical_args = _resolve_args(config.get("model_args") or {}, context)
            parameters = inspect.signature(model_cls.__init__).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            aliases = {
                "input_dim": ("input_dim", "in_dim", "dim_in"),
                "output_dim": ("output_dim", "out_dim", "dim_out"),
                "latent_dim": ("latent_dim", "condition_dim", "cond_dim", "modulation_dim"),
                "width": ("width", "hidden_dim", "dim_hidden"),
                "depth": ("depth", "num_layers", "n_layers"),
                "num_frequencies": ("num_frequencies", "num_fourier_features", "fourier_features", "mapping_size"),
                "hypernetwork_depth": ("hypernetwork_depth", "depth_hnn", "hypernet_depth", "hyper_layers"),
                "fourier_scale": ("fourier_scale", "scale", "sigma"),
            }
            required = {
                "input_dim", "output_dim", "latent_dim", "width", "depth",
                "num_frequencies", "hypernetwork_depth",
            }
            model_args: dict[str, Any] = {}
            missing: list[str] = []
            for name, value in logical_args.items():
                names = aliases.get(name, (name,))
                matched = next((item for item in names if item in parameters), None)
                if matched is not None:
                    model_args[matched] = value
                elif accepts_kwargs:
                    model_args[name] = value
                elif name in required:
                    missing.append("/".join(names))
            if missing:
                raise TypeError(f"Missing required constructor fields: {missing}")
            torch.manual_seed(int(config.get("seed", SEED)))
            model = model_cls(**model_args).to(DEVICE).train()
            current_index = 1
            update_check(1, True, f"args={sorted(model_args)}")

            coordinates = torch.linspace(-0.8, 0.8, 40, device=DEVICE).reshape(2, 10, 2)
            target = torch.sin(coordinates[..., :1] * torch.pi)
            latent = torch.zeros(2, 8, device=DEVICE, requires_grad=True)
            forward_method = getattr(model, "modulated_forward", None)
            output = forward_method(coordinates, latent) if callable(forward_method) else model(coordinates, latent)
            if isinstance(output, tuple):
                output = output[0]
            inner_loss = torch.nn.functional.mse_loss(output, target)
            latent_gradient = torch.autograd.grad(inner_loss, latent, create_graph=True)[0]

            current_index = 2
            gradient_ok = (
                latent_gradient.shape == latent.shape
                and bool(torch.isfinite(latent_gradient).all())
                and float(latent_gradient.abs().sum().detach()) > 0.0
            )
            update_check(2, gradient_ok, f"gradient_norm={float(latent_gradient.norm().detach()):.6g}")

            adapted = latent - float(config.get("inner_learning_rate", 0.01)) * latent_gradient
            current_index = 3
            update_check(3, not torch.equal(adapted.detach(), latent.detach()), "latent changed after inner step")

            adapted_output = forward_method(coordinates, adapted) if callable(forward_method) else model(coordinates, adapted)
            if isinstance(adapted_output, tuple):
                adapted_output = adapted_output[0]
            outer_loss = torch.nn.functional.mse_loss(adapted_output, target)
            current_index = 4
            update_check(4, outer_loss.ndim == 0 and bool(torch.isfinite(outer_loss)), f"loss={float(outer_loss.detach()):.6g}")

            outer_loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            current_index = 5
            model_gradient_ok = bool(gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            update_check(5, model_gradient_ok, f"finite_model_gradients={len(gradients)}")

        elif contract == "mario_training":
            sdf_path = Path(CANDIDATE_DIR, str(config["train_sdf_file"]))
            flow_path = Path(CANDIDATE_DIR, str(config["train_file"]))
            model_path = Path(CANDIDATE_DIR, str(config["model_file"]))
            sdf_source = sdf_path.read_text(encoding="utf-8")
            flow_source = flow_path.read_text(encoding="utf-8")
            model_source = model_path.read_text(encoding="utf-8")
            sdf_lower = sdf_source.lower()
            flow_lower = flow_source.lower()
            model_lower = model_source.lower()
            sdf_tree = ast.parse(sdf_source, filename=str(sdf_path))
            flow_tree = ast.parse(flow_source, filename=str(flow_path))

            current_index = 0
            entry_ok = all(
                any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "main"
                    for node in ast.walk(tree)
                )
                or any(
                    isinstance(node, ast.If)
                    and "__main__" in ast.get_source_segment(source, node.test)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.If) and ast.get_source_segment(source, node.test)
                )
                for tree, source in ((sdf_tree, sdf_source), (flow_tree, flow_source))
            )
            update_check(0, entry_ok, "main entrypoint in both stages")

            current_index = 1
            loop_results = []
            for tree in (sdf_tree, flow_tree):
                stage_loop = False
                for node in ast.walk(tree):
                    if isinstance(node, (ast.For, ast.While)):
                        calls = [
                            child.func.attr
                            for child in ast.walk(node)
                            if isinstance(child, ast.Call)
                            and isinstance(child.func, ast.Attribute)
                        ]
                        if "backward" in calls and "step" in calls:
                            stage_loop = True
                            break
                loop_results.append(stage_loop)
            update_check(1, all(loop_results), f"stage_loops={loop_results}")

            current_index = 2
            sdf_bundle = f"{sdf_lower}\n{model_lower}"
            sdf_ok = (
                "modulatedfourierfeatures" in sdf_bundle
                and "latent" in sdf_bundle
                and any(token in sdf_bundle for token in ("autograd.grad", "inner_loop", "adapt_latent", "graph_outer_step"))
            )
            update_check(2, sdf_ok, "SDF INR and geometry-latent inner adaptation")

            current_index = 3
            flow_bundle = f"{flow_lower}\n{model_lower}"
            flow_ok = (
                "multiscalemodulatedfourierfeatures" in flow_bundle
                and any(token in flow_bundle for token in ("condition", "cond", "freestream", "velocity"))
                and any(token in flow_bundle for token in ("latent", "modulation"))
            )
            update_check(3, flow_ok, "geometry latent and inlet-conditioned flow INR")

            current_index = 4
            stage_sources = (sdf_lower, flow_lower)
            optimizer_and_mse = all(
                any(token in item for token in ("optim.adam", "optim.adamw", "adam(", "adamw("))
                and any(token in item for token in ("mse_loss", "mseloss", ".square(", "** 2", "**2"))
                and ".backward(" in item
                for item in stage_sources
            )
            update_check(4, optimizer_and_mse, "optimizer, MSE and backward in both stages")

            current_index = 5
            reporting = all("print(" in item or "tqdm" in item for item in stage_sources)
            update_check(5, reporting, "print or tqdm loss reporting in both stages")

            current_index = 6
            checkpoint_ok = all(
                any(token in item for token in ("torch.save", "save_checkpoint"))
                for item in stage_sources
            ) and all(token in f"{sdf_lower}\n{flow_lower}" for token in ("epoch", "optimizer"))
            update_check(6, checkpoint_ok, "resumable two-stage checkpoint state")

            configs = [
                _load_candidate_config(name)
                for name in config.get("config_names", [])
            ]
            artifact_bundle = (
                json.dumps(configs, ensure_ascii=False, default=str).lower()
                + "\n"
                + sdf_lower
                + "\n"
                + flow_lower
            )
            current_index = 7
            artifacts_ok = (
                "trainings" in artifact_bundle
                and any(token in artifact_bundle for token in ("checkpoint", ".pt"))
                and any(token in artifact_bundle for token in ("latent", "modulation", ".npz"))
                and any(token in artifact_bundle for token in ("curve", "loss.png", "history", ".csv"))
            )
            update_check(7, artifacts_ok, "checkpoint, latent and curve paths under trainings/")

        elif contract == "mario_inference":
            evaluate_path = Path(CANDIDATE_DIR, str(config["evaluate_file"]))
            model_path = Path(CANDIDATE_DIR, str(config["model_file"]))
            source = evaluate_path.read_text(encoding="utf-8")
            model_source = model_path.read_text(encoding="utf-8")
            lower = source.lower()
            bundle = f"{lower}\n{model_source.lower()}"
            tree = ast.parse(source, filename=str(evaluate_path))

            current_index = 0
            entry_ok = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in {"main", "evaluate", "predict_graph"}
                for node in ast.walk(tree)
            )
            update_check(0, entry_ok, "evaluation entrypoint")

            current_index = 1
            checkpoint_ok = any(token in lower for token in ("torch.load", "load_checkpoint", "safe_torch_load")) and any(
                token in bundle for token in ("load_state_dict", "load_inr")
            )
            update_check(1, checkpoint_ok, "checkpoint restoration")

            current_index = 2
            eval_ok = ".eval(" in bundle and any(token in lower for token in ("no_grad", "inference_mode"))
            update_check(2, eval_ok, "eval mode with gradients disabled")

            current_index = 3
            rebuild_ok = (
                any(token in lower for token in ("build_dataset", "dataloader", "airfrans"))
                and any(token in bundle for token in ("multiscalemodulatedfourierfeatures", "create_inr_instance", "load_model"))
            )
            update_check(3, rebuild_ok, "AirfRANS split and model reconstruction")

            current_index = 4
            metric_ok = (
                any(token in lower for token in ("mse", "rmse", "mae", "relative_l2", "r2"))
                and any(token in lower for token in ("velocity", "pressure", "output_fields", "field"))
            )
            update_check(4, metric_ok, "per-field aerodynamic regression metrics")

            candidate_config = _load_candidate_config(str(config["config_name"]))
            config_text = json.dumps(candidate_config, ensure_ascii=False, default=str).lower()
            current_index = 5
            persistence_ok = (
                "trainings" in f"{lower}\n{config_text}"
                and any(token in lower for token in ("json.dump", "write_text", "to_csv", "torch.save", "np.save"))
                and ("print(" in lower or "tqdm" in lower)
            )
            update_check(5, persistence_ok, "reported metrics persisted under trainings/")

        else:
            update_check(0, False, f"Unsupported contract: {contract!r}")
    except Exception:
        update_check(min(current_index, len(checks) - 1), False, traceback.format_exc())
    return checks


def check_model_forward_execution(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Run both MARIO field models without requiring Standard equivalence."""
    models = list(config.get("models", []))
    CHECKPOINT_NAMES: list[str] = []
    for item in models:
        name = str(item.get("name", "model"))
        CHECKPOINT_NAMES.extend(
            [
                f"model_forward:{name}:build_model",
                f"model_forward:{name}:forward_run",
                f"model_forward:{name}:tensor_shape",
                f"model_forward:{name}:finite",
                f"model_forward:{name}:nonconstant",
                f"model_forward:{name}:input_sensitivity",
                f"model_forward:{name}:latent_sensitivity",
                f"model_forward:{name}:backward",
            ]
        )
    if not CHECKPOINT_NAMES:
        return [("model_forward:models", False, "config['models'] is missing or empty")]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    torch.manual_seed(int(config.get("seed", SEED)))
    for model_index, item in enumerate(models):
        base_index = model_index * 8
        current_index = base_index
        try:
            module = _load_candidate_modules(str(item["model_file"]))
            model_cls = _load_class(module, str(item["model_class"]))
            candidate_config = (
                _load_candidate_config(str(item["config_name"]))
                if item.get("config_name")
                else {}
            )
            context = {"config": candidate_config}
            if isinstance(candidate_config, dict):
                context.update(candidate_config)
            logical_args = _resolve_args(item.get("model_args") or {}, context)
            parameters = inspect.signature(model_cls.__init__).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            aliases = {
                "input_dim": ("input_dim", "in_dim", "dim_in"),
                "output_dim": ("output_dim", "out_dim", "dim_out"),
                "latent_dim": ("latent_dim", "condition_dim", "cond_dim", "modulation_dim"),
                "width": ("width", "hidden_dim", "dim_hidden"),
                "depth": ("depth", "num_layers", "n_layers"),
                "num_frequencies": ("num_frequencies", "num_fourier_features", "fourier_features", "mapping_size"),
                "hypernetwork_depth": ("hypernetwork_depth", "depth_hnn", "hypernet_depth", "hyper_layers"),
                "fourier_scale": ("fourier_scale", "scale", "sigma"),
                "fourier_scales": ("fourier_scales", "scales", "sigmas"),
                "scalar_output_dim": ("scalar_output_dim", "scalar_out_dim"),
            }
            required = {
                "input_dim", "output_dim", "latent_dim", "width", "depth",
                "num_frequencies", "hypernetwork_depth",
            }
            model_args: dict[str, Any] = {}
            missing: list[str] = []
            for name, value in logical_args.items():
                names = aliases.get(name, (name,))
                matched = next((alias for alias in names if alias in parameters), None)
                if matched is not None:
                    model_args[matched] = value
                elif accepts_kwargs:
                    model_args[name] = value
                elif name in required:
                    missing.append("/".join(names))
            if missing:
                raise TypeError(f"Missing required constructor fields: {missing}")
            model = model_cls(**model_args).to(DEVICE).eval()
            update_check(base_index, True, f"args={sorted(model_args)}")

            input_shape = tuple(int(value) for value in item["input_shape"])
            latent_shape = tuple(int(value) for value in item["latent_shape"])
            expected_shape = tuple(int(value) for value in item["output_shape"])
            coordinates = torch.linspace(
                -0.9,
                0.9,
                math.prod(input_shape),
                dtype=torch.float32,
                device=DEVICE,
            ).reshape(input_shape)
            latent = torch.linspace(
                -0.4,
                0.6,
                math.prod(latent_shape),
                dtype=torch.float32,
                device=DEVICE,
            ).reshape(latent_shape)
            forward_method = getattr(model, "modulated_forward", None)
            with torch.no_grad():
                output = forward_method(coordinates, latent) if callable(forward_method) else model(coordinates, latent)
            if isinstance(output, tuple):
                output = output[0]
            current_index = base_index + 1
            update_check(current_index, True, f"output_type={type(output).__name__}")

            current_index = base_index + 2
            tensor_shape_ok = torch.is_tensor(output) and tuple(output.shape) == expected_shape
            update_check(current_index, tensor_shape_ok, f"actual={tuple(output.shape) if torch.is_tensor(output) else None}; expected={expected_shape}")

            current_index = base_index + 3
            finite = torch.is_tensor(output) and bool(torch.isfinite(output).all())
            update_check(current_index, finite, "all output values finite")

            current_index = base_index + 4
            nonconstant = finite and output.numel() > 1 and float(output.std()) > 1.0e-7
            update_check(current_index, nonconstant, f"std={float(output.std()) if finite else float('nan'):.6g}")

            with torch.no_grad():
                changed_input = forward_method(coordinates + 0.05, latent) if callable(forward_method) else model(coordinates + 0.05, latent)
                changed_latent = forward_method(coordinates, latent + 0.1) if callable(forward_method) else model(coordinates, latent + 0.1)
            if isinstance(changed_input, tuple):
                changed_input = changed_input[0]
            if isinstance(changed_latent, tuple):
                changed_latent = changed_latent[0]

            current_index = base_index + 5
            input_difference = float((changed_input - output).abs().max()) if tensor_shape_ok else 0.0
            update_check(current_index, input_difference > 1.0e-7, f"max_difference={input_difference:.6g}")

            current_index = base_index + 6
            latent_difference = float((changed_latent - output).abs().max()) if tensor_shape_ok else 0.0
            update_check(current_index, latent_difference > 1.0e-7, f"max_difference={latent_difference:.6g}")

            current_index = base_index + 7
            if bool(config.get("check_backward", True)):
                model.train()
                model.zero_grad(set_to_none=True)
                train_output = forward_method(coordinates, latent) if callable(forward_method) else model(coordinates, latent)
                if isinstance(train_output, tuple):
                    train_output = train_output[0]
                loss = train_output.square().mean()
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                backward_ok = bool(gradients) and all(
                    bool(torch.isfinite(gradient).all()) for gradient in gradients
                )
                update_check(current_index, backward_ok, f"finite_gradients={len(gradients)}")
            else:
                update_check(current_index, True, "backward check disabled")
        except Exception:
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_train_loss(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Check that both training stages implement an optimized MSE objective."""
    CHECKPOINT_NAMES = [
        "train_loss:sdf_mse_objective",
        "train_loss:flow_mse_objective",
        "train_loss:backward_connections",
        "train_loss:reported_values",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        sdf_source = Path(CANDIDATE_DIR, str(config["train_sdf_file"])).read_text(encoding="utf-8")
        flow_source = Path(CANDIDATE_DIR, str(config["train_file"])).read_text(encoding="utf-8")
        sources = (sdf_source.lower(), flow_source.lower())

        objective_results = [
            any(token in source for token in ("mse_loss", "mseloss", ".square(", "** 2", "**2"))
            and any(token in source for token in ("loss =", "loss=", "loss +=", "loss+="))
            for source in sources
        ]
        update_check(0, objective_results[0], "SDF reconstruction MSE")
        current_index = 1
        update_check(1, objective_results[1], "flow-field reconstruction MSE")

        current_index = 2
        backward_results = [".backward(" in source for source in sources]
        update_check(2, all(backward_results), f"stage_backward={backward_results}")

        current_index = 3
        report_results = [
            ("print(" in source or "tqdm" in source) and "loss" in source
            for source in sources
        ]
        update_check(3, all(report_results), f"stage_reporting={report_results}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_smoke_training(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Run one optimizer step for the SDF and flow MARIO models."""
    models = list(config.get("models", []))
    CHECKPOINT_NAMES: list[str] = []
    for item in models:
        name = str(item.get("name", "model"))
        CHECKPOINT_NAMES.extend(
            [
                f"smoke_training:{name}:model_and_optimizer",
                f"smoke_training:{name}:finite_loss_and_gradients",
                f"smoke_training:{name}:parameter_update",
            ]
        )
    if not CHECKPOINT_NAMES:
        return [("smoke_training:models", False, "config['models'] is missing or empty")]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    torch.manual_seed(int(config.get("seed", SEED)))
    for model_index, item in enumerate(models):
        base_index = model_index * 3
        current_index = base_index
        try:
            module = _load_candidate_modules(str(item["model_file"]))
            model_cls = _load_class(module, str(item["model_class"]))
            candidate_config = (
                _load_candidate_config(str(item["config_name"]))
                if item.get("config_name")
                else {}
            )
            context = {"config": candidate_config}
            if isinstance(candidate_config, dict):
                context.update(candidate_config)
            logical_args = _resolve_args(item.get("model_args") or {}, context)
            parameters = inspect.signature(model_cls.__init__).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            aliases = {
                "input_dim": ("input_dim", "in_dim", "dim_in"),
                "output_dim": ("output_dim", "out_dim", "dim_out"),
                "latent_dim": ("latent_dim", "condition_dim", "cond_dim", "modulation_dim"),
                "width": ("width", "hidden_dim", "dim_hidden"),
                "depth": ("depth", "num_layers", "n_layers"),
                "num_frequencies": ("num_frequencies", "num_fourier_features", "fourier_features", "mapping_size"),
                "hypernetwork_depth": ("hypernetwork_depth", "depth_hnn", "hypernet_depth", "hyper_layers"),
                "fourier_scale": ("fourier_scale", "scale", "sigma"),
                "fourier_scales": ("fourier_scales", "scales", "sigmas"),
                "scalar_output_dim": ("scalar_output_dim", "scalar_out_dim"),
            }
            required = {
                "input_dim", "output_dim", "latent_dim", "width", "depth",
                "num_frequencies", "hypernetwork_depth",
            }
            model_args: dict[str, Any] = {}
            missing: list[str] = []
            for name, value in logical_args.items():
                names = aliases.get(name, (name,))
                matched = next((alias for alias in names if alias in parameters), None)
                if matched is not None:
                    model_args[matched] = value
                elif accepts_kwargs:
                    model_args[name] = value
                elif name in required:
                    missing.append("/".join(names))
            if missing:
                raise TypeError(f"Missing required constructor fields: {missing}")
            model = model_cls(**model_args).to(DEVICE).train()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if not trainable:
                raise ValueError("Model has no trainable parameters")
            optimizer = torch.optim.Adam(
                trainable,
                lr=float(config.get("learning_rate", 1.0e-3)),
            )
            update_check(base_index, True, f"trainable_parameters={sum(item.numel() for item in trainable)}")

            input_shape = tuple(int(value) for value in item["input_shape"])
            latent_shape = tuple(int(value) for value in item["latent_shape"])
            expected_shape = tuple(int(value) for value in item["output_shape"])
            coordinates = torch.randn(*input_shape, device=DEVICE)
            latent = torch.randn(*latent_shape, device=DEVICE)
            target = torch.sin(coordinates[..., :1]).expand(expected_shape).contiguous()
            if expected_shape[-1] > 1:
                channel_scale = torch.linspace(0.7, 1.3, expected_shape[-1], device=DEVICE)
                target = target * channel_scale

            before = [parameter.detach().clone() for parameter in trainable]
            optimizer.zero_grad(set_to_none=True)
            forward_method = getattr(model, "modulated_forward", None)
            output = forward_method(coordinates, latent) if callable(forward_method) else model(coordinates, latent)
            if isinstance(output, tuple):
                output = output[0]
            if not torch.is_tensor(output) or tuple(output.shape) != expected_shape:
                raise TypeError(f"Expected Tensor shape {expected_shape}, got {getattr(output, 'shape', None)}")
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
            current_index = base_index + 1
            gradient_ok = bool(gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            update_check(current_index, bool(torch.isfinite(loss)) and gradient_ok, f"loss={float(loss.detach()):.6g}; finite_gradients={len(gradients)}")

            optimizer.step()
            changed = any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, trainable)
            )
            current_index = base_index + 2
            update_check(current_index, changed, f"parameters_changed={changed}")
        except Exception:
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_execution(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Execute a synthetic MARIO flow-field inference and save typed output."""
    CHECKPOINT_NAMES = [
        "inference_execution:module_load",
        "inference_execution:model_build",
        "inference_execution:infer_call",
        "inference_execution:metric_compute",
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
        run_file = str(config.get("run_file", "test_mario_paper_repro/evaluate.py"))
        _load_candidate_modules(run_file)
        update_check(0, True, f"loaded={run_file}")

        current_index = 1
        module = _load_candidate_modules(str(config["model_file"]))
        model_cls = _load_class(module, str(config["model_class"]))
        candidate_config = (
            _load_candidate_config(str(config["config_name"]))
            if config.get("config_name")
            else {}
        )
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)
        logical_args = _resolve_args(config.get("model_args") or {}, context)
        parameters = inspect.signature(model_cls.__init__).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        aliases = {
            "input_dim": ("input_dim", "in_dim", "dim_in"),
            "output_dim": ("output_dim", "out_dim", "dim_out"),
            "latent_dim": ("latent_dim", "condition_dim", "cond_dim", "modulation_dim"),
            "width": ("width", "hidden_dim", "dim_hidden"),
            "depth": ("depth", "num_layers", "n_layers"),
            "num_frequencies": ("num_frequencies", "num_fourier_features", "fourier_features", "mapping_size"),
            "hypernetwork_depth": ("hypernetwork_depth", "depth_hnn", "hypernet_depth", "hyper_layers"),
            "fourier_scales": ("fourier_scales", "scales", "sigmas"),
            "scalar_output_dim": ("scalar_output_dim", "scalar_out_dim"),
        }
        required = {
            "input_dim", "output_dim", "latent_dim", "width", "depth",
            "num_frequencies", "hypernetwork_depth",
        }
        model_args: dict[str, Any] = {}
        missing: list[str] = []
        for name, value in logical_args.items():
            names = aliases.get(name, (name,))
            matched = next((alias for alias in names if alias in parameters), None)
            if matched is not None:
                model_args[matched] = value
            elif accepts_kwargs:
                model_args[name] = value
            elif name in required:
                missing.append("/".join(names))
        if missing:
            raise TypeError(f"Missing required constructor fields: {missing}")
        torch.manual_seed(int(config.get("seed", SEED)))
        model = model_cls(**model_args).cpu().eval()
        update_check(1, True, f"args={sorted(model_args)}")

        input_shape = tuple(int(value) for value in config["input_shape"])
        latent_shape = tuple(int(value) for value in config["latent_shape"])
        expected_shape = tuple(int(value) for value in config["output_shape"])
        coordinates = torch.linspace(-1.0, 1.0, math.prod(input_shape), dtype=torch.float32).reshape(input_shape)
        latent = torch.linspace(-0.5, 0.5, math.prod(latent_shape), dtype=torch.float32).reshape(latent_shape)
        forward_method = getattr(model, "modulated_forward", None)
        current_index = 2
        with torch.inference_mode():
            prediction = forward_method(coordinates, latent) if callable(forward_method) else model(coordinates, latent)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction_ok = (
            torch.is_tensor(prediction)
            and tuple(prediction.shape) == expected_shape
            and bool(torch.isfinite(prediction).all())
        )
        update_check(2, prediction_ok, f"shape={tuple(prediction.shape) if torch.is_tensor(prediction) else None}")
        if not prediction_ok:
            return checks

        target = torch.sin(coordinates[..., :1] * torch.pi).expand(expected_shape).contiguous()
        target = target * torch.linspace(0.7, 1.3, expected_shape[-1])
        mse = float(torch.nn.functional.mse_loss(prediction, target))
        current_index = 3
        update_check(3, math.isfinite(mse) and mse >= 0.0, f"mse={mse:.8g}")

        current_index = 4
        output_path = Path(str(config["output_path"]))
        if not output_path.is_absolute():
            output_path = Path(CANDIDATE_DIR, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prediction": {
                "data": prediction.detach().cpu().tolist(),
                "dtype": str(prediction.dtype).removeprefix("torch."),
            },
            "target": {
                "data": target.detach().cpu().tolist(),
                "dtype": str(target.dtype).removeprefix("torch."),
            },
            "shape": list(prediction.shape),
            "finite": bool(torch.isfinite(prediction).all()),
            "mse": mse,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
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
    """Validate semantic content of the synthetic MARIO inference output."""
    CHECKPOINT_NAMES = [
        "inference_output:structured_result",
        "inference_output:prediction_payload",
        "inference_output:expected_shape",
        "inference_output:finite_nonconstant",
        "inference_output:metrics_present",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: Any):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), str(msg))

    current_index = 0
    try:
        output_path = Path(str(config["output_path"]))
        if not output_path.is_absolute():
            output_path = Path(CANDIDATE_DIR, output_path)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        structured = isinstance(result, Mapping) and bool(result)
        update_check(0, structured, f"type={type(result).__name__}")
        if not structured:
            return checks

        current_index = 1
        payload = result.get("prediction")
        payload_ok = (
            isinstance(payload, Mapping)
            and isinstance(payload.get("data"), list)
            and isinstance(payload.get("dtype"), str)
        )
        update_check(1, payload_ok, f"prediction_keys={sorted(payload) if isinstance(payload, Mapping) else None}")
        if not payload_ok:
            return checks

        prediction = torch.tensor(payload["data"])
        expected_shape = tuple(int(value) for value in config["expected_shape"])
        current_index = 2
        update_check(2, tuple(prediction.shape) == expected_shape, f"actual={tuple(prediction.shape)}; expected={expected_shape}")

        current_index = 3
        finite_nonconstant = (
            prediction.numel() > 1
            and bool(torch.isfinite(prediction).all())
            and float(prediction.std()) > 1.0e-7
            and bool(result.get("finite"))
        )
        update_check(3, finite_nonconstant, f"std={float(prediction.std()):.6g}")

        current_index = 4
        metric_names = tuple(config.get("metric_names", ("mse",)))
        metrics_ok = all(
            name in result
            and isinstance(result[name], (int, float))
            and math.isfinite(float(result[name]))
            and float(result[name]) >= 0.0
            for name in metric_names
        )
        update_check(4, metrics_ok, f"metrics={[(name, result.get(name)) for name in metric_names]}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_metrics(
    config: dict[str, Any],
    caller_globals=None,
    caller_locals=None,
) -> list[tuple[str, bool, str]]:
    """Recompute MSE and validate the declared paper comparison direction."""
    CHECKPOINT_NAMES = [
        "infer_metric:prediction_and_target",
        "infer_metric:mse_formula",
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
        output_path = Path(str(config["output_path"]))
        if not output_path.is_absolute():
            output_path = Path(CANDIDATE_DIR, output_path)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        prediction_payload = result.get("prediction")
        target_payload = result.get("target")
        payload_ok = (
            isinstance(prediction_payload, Mapping)
            and isinstance(target_payload, Mapping)
            and isinstance(prediction_payload.get("data"), list)
            and isinstance(target_payload.get("data"), list)
        )
        update_check(0, payload_ok, f"keys={sorted(result) if isinstance(result, Mapping) else None}")
        if not payload_ok:
            return checks

        prediction = torch.tensor(prediction_payload["data"], dtype=torch.float64)
        target = torch.tensor(target_payload["data"], dtype=torch.float64)
        actual = float((prediction - target).square().mean())
        metric_name = str(config.get("metric_name", "mse"))
        reported = float(result[metric_name])
        current_index = 1
        formula_ok = math.isfinite(actual) and math.isclose(
            actual,
            reported,
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        )
        update_check(1, formula_ok, f"recomputed={actual:.8g}; reported={reported:.8g}")

        baseline = float(config.get("baseline", 0.0096))
        mode = str(config.get("mode", "min")).lower()
        current_index = 2
        baseline_ok = math.isclose(baseline, 0.0096, abs_tol=1.0e-12) and mode == "min"
        update_check(2, baseline_ok, f"mse baseline={baseline}; mode={mode}")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks
