#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NequIX energy-offset Case; OnescienceBench baseline 737e4eac3ed9.

Base same-name bodies keep their loading/hook/call/result order. Task adapters
handle the JSON-header checkpoint, ASE data, dictionary outputs and offset-only
training. No Standard implementation or saved nn.Module is used as an oracle.
Unknown private interfaces are REVIEW_REQUIRED, not scientific failure evidence.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import math
import sys
import tempfile
import textwrap
import traceback
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import yaml
from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    FORWARD_COMPARE_ATOL,
    TypeInferencer,
    _compute_mse,
    _convert_arg_to_type,
    _execute_hooks,
    _initialize_parameters_with_constants,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _load_reference_config,
    _load_reference_modules,
    _move_tensors_to_device,
    _resolve_args,
    check_files,
    check_interface,
)



def prepare_nequix_batch(module, data_path, frame_index, device,
                         checkpoint_file=None, checkpoint_config=None):
    """Bind the two reviewed batch APIs; never construct a graph for the candidate."""
    parameters = inspect.signature(module.build_finetune_batch).parameters
    if {"structures", "config"}.issubset(parameters):
        if checkpoint_config is None:
            with Path(checkpoint_file).open("rb") as stream:
                checkpoint_config = json.loads(stream.readline().decode("utf-8"))
        structures = module.load_training_structures(str(data_path), index=frame_index)
        return module.build_finetune_batch(
            structures=structures, config=checkpoint_config, device=str(device))
    if {"path", "index"}.issubset(parameters):
        return module.build_finetune_batch(path=str(data_path), index=frame_index, device=str(device))
    raise ValueError(f"REVIEW_REQUIRED: bind build_finetune_batch signature {tuple(parameters)}")


def prepare_nequix_probe(model_file: str, checkpoint_file: str,
                         checkpoint_sha256: str, data_path=None,
                         frame_index: int = 0) -> dict:
    """Shared real-input hook: safe fixed payload + candidate loader observations.

    Bind the reviewed adapter-returning and backbone-returning load_model APIs.
    Wrapping a returned backbone uses the candidate's own public adapter class,
    as in its training entry; no evaluator freeze or parameter repair is applied.
    Does not initialize/repair candidate parameters or trust its returned state
    as the reference. Missing files/dependencies remain visible failures.
    """
    path = Path(checkpoint_file)
    with path.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash != checkpoint_sha256:
        raise ValueError(f"Fixed checkpoint SHA256 mismatch: {actual_hash}")
    with path.open("rb") as stream:
        parsed_config = json.loads(stream.readline().decode("utf-8"))
        payload = torch.load(io.BytesIO(stream.read()), map_location="cpu", weights_only=True)
    if not isinstance(parsed_config, dict) or not isinstance(payload, dict):
        raise TypeError("Fixed checkpoint must contain a JSON mapping and a state mapping")
    # The payload envelope is explicitly distinguished from an nn.Module.
    state = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise TypeError("REVIEW_REQUIRED: bind the fixed checkpoint tensor-dictionary envelope")
    strict_calls, load_calls = [], []
    original_load_state = nn.Module.load_state_dict
    original_load = torch.load

    def observed_state_load(instance, state_dict, *args, **kwargs):
        strict = kwargs.get("strict", args[0] if args else True)
        strict_calls.append((instance, strict))
        return original_load_state(instance, state_dict, *args, **kwargs)

    def observed_load(*args, **kwargs):
        value = original_load(*args, **kwargs)
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, nn.Module):
                raise TypeError("Saved nn.Module deserialization is forbidden by this Case")
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
        load_calls.append(True)
        return value

    with patch.object(torch, "load", observed_load), patch.object(nn.Module, "load_state_dict", observed_state_load):
        module = _load_candidate_modules(model_file)
        loaded = module.load_model(str(path), str(DEVICE))
    model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    if isinstance(model, nn.Module) and type(model).__name__ == "NequixTorch":
        model = _load_class(module, "NequixEnergyOffsetAdapter")(model)
    if not isinstance(model, nn.Module):
        raise TypeError("REVIEW_REQUIRED: load_model result is not a model")
    base_paths = [name for name in ("base", "backbone") if isinstance(getattr(model, name, None), nn.Module)]
    if len(base_paths) != 1:
        raise TypeError(f"REVIEW_REQUIRED: one reviewed backbone binding required: {base_paths}")
    base_path = base_paths[0]
    base = getattr(model, base_path)
    model = model.to(DEVICE)
    probe = {
        "model": model, "base": base, "base_path": base_path, "module": module,
        "checkpoint_config": parsed_config, "state": state,
        "checkpoint_sha256": actual_hash,
        "torch_load_called": bool(load_calls),
        "strict_base_load": any(obj is base and strict is True for obj, strict in strict_calls),
    }
    if data_path is not None:
        probe["batch"] = prepare_nequix_batch(module, data_path, frame_index, DEVICE,
                                               checkpoint_config=parsed_config)
    return probe



def check_config_content(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple[str, bool, str]]:
    """
    检查训练 YAML 配置文件是否存在且有效。

    config 示例（JSON格式）:
    {
        "config_name": "train_config.yaml",
        "required_config_keys": [
            {
                "key": "epochs",
                "type": "int",
                "value": 100
            },
            {
                "key": "batch_size",
                "type": "int"
            },
            {
                "key": "learning_rate",
                "type": "float",
                "value": 0.001
            },
            {
                "key": "optimizer",
                "type": "str"
            },
            {
                "key": "model_name"  // 只检查键是否存在
            },
            {
                "key": "dropout",
                "type": "float"  // 检查键存在且类型为float
            },
            {
                "key": "use_gpu",
                "type": "bool",
                "value": true
            },
            {
                "key": "mixed_precision",
                "type": "bool|None"  // 支持联合类型
            },
            {
                "key": "loss_function",
                "type": "str",
                "value": null  // null表示跳过值检查
            }
        ]
    }

    检查点:
        - config_content:{key}: 检查必需字段是否存在
        - config_content:{key}_type: 检查字段值类型是否正确
        - config_content:{key}_value: 检查字段值是否符合预期

    Args:
        config: 包含配置信息的字典

    Returns:
        检查结果列表，每个元素为 (检查点名称, 是否通过, 消息)
    """
    # ================================================================
    # 0. 初始化
    # ================================================================
    checks: List[Tuple[str, bool, str]] = []

    def update_check(index: int, passed: bool, msg: str):
        """更新指定索引的检查结果"""
        if index < len(checks):
            name = checks[index][0]
            checks[index] = (name, passed, msg)

    def add_check(name: str, passed: bool = False, msg: str = "Not executed"):
        """添加新的检查项"""
        checks.append((name, passed, msg))
        return len(checks) - 1

    try:
        # ------------------ 1. 加载配置 ------------------
        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("config['config_name'] is missing or empty")

        required_keys_spec = config.get("required_config_keys", [])
        if not required_keys_spec:
            raise ValueError("config['required_config_keys'] is missing or empty")

        # 验证并规范化 required_config_keys 格式
        normalized_specs = []
        for idx, spec in enumerate(required_keys_spec):
            if isinstance(spec, str):
                # 兼容旧格式：直接使用字符串
                normalized_specs.append({"key": spec, "type": None, "value": None})
            elif isinstance(spec, dict):
                # JSON格式
                key = spec.get("key")
                if not key:
                    raise ValueError(f"Missing 'key' field in spec at index {idx}: {spec}")

                normalized_specs.append({
                    "key": key,
                    "type": spec.get("type"),  # 可以为None
                    "value": spec.get("value")  # 可以为None
                })
            else:
                raise ValueError(f"Invalid spec type at index {idx}: {type(spec)}")

        # 加载配置文件
        config_data = _load_candidate_config(config_name)
        # Only map the reviewed nested YAML paths; Base owns all comparisons.
        if config.get("config_paths"):
            original_config = config_data
            config_data = dict(config_data)
            for key, path in config["config_paths"].items():
                paths = path if isinstance(path, list) else [path]
                values = _resolve_args({"values": ["$config." + item for item in paths]},
                                       {"config": original_config})["values"]
                values = [value for value in values if value is not None]
                if values and any(value != values[0] for value in values[1:]):
                    raise ValueError(f"REVIEW_REQUIRED: conflicting config paths for {key}: {paths}")
                if values:
                    config_data[key] = values[0]
        # Match only explicitly bound numeric consumers; Base still compares values.
        for key, value_type in config.get("config_value_types", {}).items():
            if key in config_data:
                config_data[key] = _convert_arg_to_type(config_data[key], value_type)

        # ------------------ 2. 检查必需字段 ------------------
        for spec in normalized_specs:
            key = spec["key"]
            expected_type = spec["type"]
            expected_value = spec["value"]

            # 2.1 检查键是否存在
            if key not in config_data:
                add_check(
                    f"config_content:{key}",
                    False,
                    f"Required key '{key}' not found in config"
                )
                continue
            else:
                add_check(
                    f"config_content:{key}",
                    True,
                    f"Required key '{key}' found"
                )

            current_value = config_data[key]

            # 2.2 检查类型（如果指定了类型）
            if expected_type is not None:
                # 处理类型映射
                type_map = {
                    "int": int,
                    "float": float,
                    "str": str,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "NoneType": type(None),
                }

                # 支持用 | 分隔的多个类型，例如 "int|float"
                type_names = expected_type.split("|") if "|" in expected_type else [expected_type]
                expected_types = []
                for type_name in type_names:
                    type_name = type_name.strip()
                    if type_name in type_map:
                        expected_types.append(type_map[type_name])
                    else:
                        # 尝试直接获取类型
                        try:
                            expected_types.append(eval(type_name))
                        except:
                            raise ValueError(f"Unknown type: {type_name}")

                # 检查值是否为期望的类型
                is_correct_type = any(isinstance(current_value, t) for t in expected_types)

                if not is_correct_type:
                    add_check(
                        f"config_content:{key}_type",
                        False,
                        f"Key '{key}' has type {type(current_value).__name__}, expected {expected_type}"
                    )
                    # 类型错误，不继续检查值
                    continue
                else:
                    add_check(
                        f"config_content:{key}_type",
                        True,
                        f"Key '{key}' has correct type {type(current_value).__name__}"
                    )

            # 2.3 检查值（如果指定了期望值，且不为None）
            if expected_value is not None:
                # 支持值比较
                if isinstance(expected_value, (int, float, str, bool)):
                    if current_value != expected_value:
                        add_check(
                            f"config_content:{key}_value",
                            False,
                            f"Key '{key}' has value {current_value}, expected {expected_value}"
                        )
                    else:
                        add_check(
                            f"config_content:{key}_value",
                            True,
                            f"Key '{key}' has correct value {current_value}"
                        )
                else:
                    # 复杂类型的值比较（如列表、字典）
                    if current_value != expected_value:
                        add_check(
                            f"config_content:{key}_value",
                            False,
                            f"Key '{key}' does not match expected value"
                        )
                    else:
                        add_check(
                            f"config_content:{key}_value",
                            True,
                            f"Key '{key}' matches expected value"
                        )

    except FileNotFoundError as e:
        add_check("config_file", False, f"Config file not found: {e}")
    except yaml.YAMLError as e:
        add_check("config_file", False, f"Invalid YAML format: {e}")
    except Exception as e:
        error_detail = traceback.format_exc()
        add_check("config_loading", False, f"Error loading config: {str(e)}\n{error_detail}")

    return checks





def check_dataset_pipeline(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """ASE-LMDB/periodic-graph contract; Base has no ASE-LMDB graph comparator.

    Independent ASE labels are the reference, not candidate-returned metadata.
    Graph attribute paths are explicit interface bindings; absent bindings require
    review. No data downloading, generated replacement data or Standard oracle.
    """
    CHECKPOINT_NAMES = [
        "data:fixed_file", "data:periodic_structures", "data:batch_shape",
        "data:real_labels", "data:periodic_neighbor_graph",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = Path(config["data_path"])
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != config["data_sha256"]:
            raise ValueError(f"Fixed dataset SHA256 mismatch: {digest}")
        update_check(current_index, True, f"Fixed data verified: {digest}")
        current_index += 1
        from ase.db import connect
        from ase.neighborlist import neighbor_list
        # Close the ASE-LMDB connection before the candidate opens the same file.
        with connect(str(path), readonly=True, use_lock_file=False) as database:
            rows = list(database.select())
            atoms = rows[config["frame_index"]].toatoms()
            target_energy = float(rows[config["frame_index"]].energy)
            target_forces = np.array(rows[config["frame_index"]].forces, copy=True)
        module = _load_candidate_modules(config["model_file"])
        structures = module.load_training_structures(str(path))
        periodic = len(structures) == config["structure_count"] == len(rows)
        periodic = periodic and all(bool(np.asarray(item.pbc).all()) for item in structures)
        update_check(current_index, periodic, f"Candidate/reference structures={len(structures)}/{len(rows)}; periodicity checked")
        current_index += 1
        batch = prepare_nequix_batch(module, path, config["frame_index"], DEVICE,
                                      checkpoint_file=config.get("checkpoint_file"))
        data, energy, forces = batch.get("data", batch), batch["energy"], batch["forces"]
        fields = config["graph_fields"]
        positions = data[fields["positions"]] if isinstance(data, dict) else getattr(data, fields["positions"])
        count = config["frame_num_atoms"]
        shape_ok = (torch.is_tensor(positions) and tuple(positions.shape) == (count, 3)
                    and torch.is_tensor(energy) and energy.numel() == 1
                    and torch.is_tensor(forces) and tuple(forces.shape) == (count, 3))
        update_check(current_index, shape_ok, "One graph; real scalar energy, [28,3] positions/forces (no private scalar layout required)")
        current_index += 1
        if not shape_ok:
            raise ValueError("Cannot compare labels with an invalid batch layout")
        actual_positions = positions.detach().cpu().double().numpy()
        # A different atom permutation needs a reviewed, explicit atom-index map.
        labels_ok = (np.allclose(actual_positions, atoms.positions, rtol=1e-6, atol=1e-6)
                     and math.isclose(float(energy.detach().item()), target_energy, rel_tol=1e-6, abs_tol=1e-4)
                     and np.allclose(forces.detach().cpu().double().numpy(), target_forces, rtol=1e-5, atol=1e-6))
        update_check(current_index, bool(labels_ok), "Compare energy/forces/positions to ASE frame index=0; no hard-coded energy oracle")
        current_index += 1
        fields = config["graph_fields"]
        required = [fields["edge_index"], fields["shifts"]]
        if any(name not in data if isinstance(data, dict) else not hasattr(data, name) for name in required):
            raise ValueError(f"REVIEW_REQUIRED: bind actual periodic graph fields {required}; do not rename candidate fields")
        edges = (data[fields["edge_index"]] if isinstance(data, dict) else getattr(data, fields["edge_index"])).detach().cpu().numpy()
        shifts = (data[fields["shifts"]] if isinstance(data, dict) else getattr(data, fields["shifts"])).detach().cpu().double().numpy()
        if edges.ndim != 2 or edges.shape[0] != 2 or shifts.shape != (edges.shape[1], 3):
            raise ValueError("REVIEW_REQUIRED: graph binding expects [2,E] indices and [E,3] shifts")
        # Normalize the declared edge row convention; never rewrite candidate data.
        edges = edges[fields.get("edge_index_rows", [0, 1])]
        senders, receivers, expected_shifts = neighbor_list("ijS", atoms, config["cutoff"], self_interaction=False)
        # Compare periodic edges independent of ordering; shifts may be Cartesian
        # or integer cell shifts as specified by the reviewed configuration.
        if fields.get("shifts_are_cartesian", True):
            shifts = shifts @ np.linalg.inv(np.asarray(atoms.cell))
        rounded = np.rint(shifts)
        if not np.allclose(shifts, rounded, rtol=0, atol=1e-5):
            raise ValueError("Periodic shifts do not correspond to lattice vectors")
        actual_edges = sorted((int(i), int(j), *map(int, s)) for i, j, s in zip(edges[0], edges[1], rounded))
        reference_edges = sorted((int(i), int(j), *map(int, s)) for i, j, s in zip(senders, receivers, expected_shifts))
        update_check(current_index, actual_edges == reference_edges,
                     f"ASE cutoff={config['cutoff']} periodic edges: candidate={len(actual_edges)}, reference={len(reference_edges)}")
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_checkpoint_loading(config: dict, caller_globals=None, caller_locals=None) -> list:
    """
    验证 checkpoint 能否正确加载到模型中。

    config 示例:
    {
        "config_name": "config.yaml"
        "model_file": "model.py",
        "model_class": "MyModel",
        "model_args": {"space_dim": "${config.space_dim}", "n_layers": 3, "hidden_size": 512},  # 模型参数（支持占位符）
        "strict": True  # 是否严格加载，默认 True
    }

    检查点:
        - checkpoint_loading:module_load: 检查能否加载模型模块
        - checkpoint_loading:config_load: 检查能否加载配置
        - checkpoint_loading:checkpoint_load: 检查能否加载 checkpoint
        - checkpoint_loading:model_build: 检查能否构建模型
        - checkpoint_loading:strict_load: 检查能否严格加载 state_dict
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "checkpoint_loading:module_load",
        "checkpoint_loading:checkpoint_load",
        "checkpoint_loading:model_build",
        "checkpoint_loading:strict_load",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    # ================================================================
    # 主流程
    # ================================================================
    current_index = 0

    try:
        # ------------------ 1. 加载配置 ------------------
        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("config['config_name'] is missing or empty")

        model_config = _load_candidate_config(config_name)
        context = {"config": model_config}
        # ------------------ 2. 加载模型模块 ------------------
        model_file = config.get("model_file")
        if not model_file:
            raise ValueError("config['model_file'] is missing or empty")

        model_module = _load_candidate_modules(model_file)
        update_check(current_index, True, f"Model module loaded from {model_file}")

        # ------------------ 3. 加载 checkpoint ------------------
        current_index += 1
        checkpoint_path = config.get("checkpoint_file")
        if not checkpoint_path:
            raise ValueError("config['checkpoint_file'] is missing or empty")

        checkpoint_path_resolved = Path(checkpoint_path).resolve()
        if not checkpoint_path_resolved.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path_resolved}")

        if config.get("json_header_checkpoint"):
            context = _execute_hooks(model_module, config.get("pre_load_hooks", []), context, globals(), caller_locals)
            checkpoint = {"state_dict": context["probe"]["state"]}
        else:
            checkpoint = torch.load(
                checkpoint_path_resolved,
                map_location="cpu",
                weights_only=True,
            )

        # 支持多种 checkpoint 格式
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint  # 直接是 state_dict

        # 记录 checkpoint 元信息（便于调试）
        checkpoint_info = {
            "epoch": checkpoint.get("epoch", None),
            "step": checkpoint.get("step", None),
            "metadata": checkpoint.get("metadata", None),
        }
        context["checkpoint_info"] = checkpoint_info

        update_check(
            current_index,
            True,
            f"Checkpoint loaded from {checkpoint_path_resolved} "
            f"(epoch={checkpoint_info['epoch']}, step={checkpoint_info['step']})"
        )

        # ------------------ 4. 构建模型 ------------------
        current_index += 1
        model_class_name = config.get("model_class")
        if not model_class_name:
            raise ValueError("config['model_class'] is missing or empty")

        # 获取模型参数（支持占位符）
        model_args = config.get("model_args", {})
        resolved_model_args = _resolve_args(model_args, context)

        model_cls = getattr(model_module, model_class_name)
        if model_cls is None or not callable(model_cls):
            raise ValueError(f"Model class '{model_class_name}' not found or not callable")

        model = (context["probe"]["base"] if config.get("json_header_checkpoint")
                 else model_cls(**resolved_model_args)).to(DEVICE)

        update_check(
            current_index,
            True,
            f"Model built with args: {resolved_model_args}"
        )

        # ------------------ 5. 严格加载 state_dict ------------------
        current_index += 1
        strict_mode = config.get("strict", True)

        if config.get("json_header_checkpoint"):
            # Verify the actual candidate load; do not fix its state in the evaluator.
            if not context["probe"]["torch_load_called"] or not context["probe"]["strict_base_load"]:
                raise ValueError("Candidate did not execute torch.load and strict base.load_state_dict")
            actual = model.state_dict()
            missing, unexpected = list(set(state_dict) - set(actual)), list(set(actual) - set(state_dict))
            if not missing and not unexpected and not all(torch.equal(actual[key].detach().cpu(), state_dict[key].cpu()) for key in actual):
                raise ValueError("Candidate base state differs from fixed checkpoint tensors")
        else:
            missing, unexpected = model.load_state_dict(
                state_dict,
                strict=strict_mode,
            )

        if strict_mode and (len(missing) > 0 or len(unexpected) > 0):
            raise ValueError(
                f"Strict load failed: missing={len(missing)} keys, unexpected={len(unexpected)} keys"
            )
        elif len(missing) > 0 or len(unexpected) > 0:
            # 非严格模式，记录警告信息
            warning_msg = (
                f"Non-strict load completed with "
                f"missing={len(missing)} keys, unexpected={len(unexpected)} keys"
            )
            update_check(
                current_index,
                True,
                warning_msg
            )
        else:
            update_check(
                current_index,
                True,
                f"Strict load passed: missing={len(missing)}, unexpected={len(unexpected)}"
            )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_model_structure(config: dict, caller_globals=None, caller_locals=None):
    """
    检查模型的结构是否符合预期。

    config 示例:
    {
        "model_file": "model.py",
        "model_class": "MyModel",
        "config_name": "config.yaml",
        "model_args": {"space_dim": "${config.space_dim}", "n_layers": 3, "hidden_size": 512},  # 模型参数（支持占位符）
        "rules": [
                {
                    "id": "preprocess_mlp",
                    "type": "sequential_mlp",
                    "name": "Check preprocess MLP",
                    "params": {
                        "target_path": "preprocess",
                        "in_features": 2,
                        "hidden_features": 128,
                        "out_features": 64
                    }
                },
                {
                    "id": "blocks_count",
                    "type": "modulelist_length",
                    "name": "Check blocks count",
                    "params": {
                        "target_path": "blocks",
                        "length": 3
                    }
                },
                {
                    "id": "block_0_pattern",
                    "type": "block_pattern",
                    "name": "Check block 0 pattern",
                    "params": {
                        "target_path": "blocks.0",
                        "pattern": ["LayerNorm", "TestAttention", "LayerNorm", "Sequential"]
                    }
                },
                {
                    "id": "attn_out_proj",
                    "type": "linear_shape",
                    "name": "Check attention output projection",
                    "params": {
                        "target_path": "blocks.0.attn.out_proj",
                        "in_features": 64,
                        "out_features": 64
                    }
                },
                {
                    "id": "top_level_children",
                    "type": "children_count",
                    "name": "Check top-level children",
                    "params": {
                        "target_path": "",
                        "count": 6
                    }
                }
            ]
    }

    检查点:
        - model_structure:build_model_ok: 检查能否成功构建模型
        - model_structure:structure: 模型结构
    """
    import torch.nn as nn

    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "model_structure:build_model_ok",
    ]
    if config.get("adapter_contract"):
        CHECKPOINT_NAMES.extend([
            "model_structure:state_size",
            "model_structure:only_offset_trainable", "model_structure:offset_shape_initial",
            "model_structure:checkpoint_config",
        ])
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

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
        if not config_name:
            raise ValueError("config['config_name'] is missing or empty")

        candidate_config = _load_candidate_config(config_name)
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)

        # 获取模型参数（必须配置 model_args）
        model_args = config.get("model_args")
        if not model_args and not config.get("model_ref"):
            raise ValueError("config['model_args'] is required")

        context = _execute_hooks(model_module, config.get("pre_model_hooks", []), context, globals(), caller_locals)
        resolved_model_args = _resolve_args(model_args, context)
        model = (context["probe"][config["model_ref"]]
                 if config.get("model_ref") else model_cls(**resolved_model_args))
        model = model.to(DEVICE).eval()

        update_check(
            current_index,
            True,
            f"Model built with config: {candidate_config}"
        )

        from base.model_structure_checker import (
            ModuleInfoExtractor,
            ModelStructureChecker,
        )
        checker = ModelStructureChecker()

        rules = config.get("rules")
        if rules is None:
            raise ValueError("config['rules'] is missing")
        if config.get("model_ref"):
            rules = _resolve_args({"rules": rules}, {"base_path": context["probe"]["base_path"]})["rules"]
        checker.add_rules_from_rules(rules)

        # 提取结构信息
        extractor = ModuleInfoExtractor(model)
        root_info = extractor.extract()

        # 执行检查
        for rule in checker.rules:
            modules_to_check = checker._get_modules_for_rule(root_info, rule, None)

            if rule.rule_type == "module_count":
                expected = (rule.params or {}).get("count")
                actual = len(modules_to_check)
                passed = expected is not None and actual == expected
                message = (
                    f"Module count={actual} matches"
                    if passed
                    else f"Module count: expected {expected}, got {actual}"
                )
                checks.append((f"model_structure:{rule.id}", passed, message))
                continue

            for module_info in modules_to_check:
                passed, msg = checker.evaluate_rule(rule, module_info)

                # 生成检查点名称
                if module_info.module_path:
                    name = f"model_structure:{rule.id}:{module_info.module_path}"
                else:
                    name = f"model_structure:{rule.id}"
                checks.append((name, passed, msg))

        if config.get("adapter_contract"):
            contract = config["adapter_contract"]
            probe = context["probe"]
            base = probe["base"]
            named = dict(model.named_parameters())
            base_named = dict(base.named_parameters())
            offset = named.get("energy_offset")
            actual_config = dict(probe["checkpoint_config"])
            expected_config = dict(contract["config_values"])
            if "hidden_irreps" in expected_config:
                from e3nn import o3
                actual_config["hidden_irreps"] = o3.Irreps(actual_config.get("hidden_irreps"))
                expected_config["hidden_irreps"] = o3.Irreps(expected_config["hidden_irreps"])
            flags = {
                "model_structure:state_size": len(probe["state"]) == contract["state_count"] and
                    sum(value.numel() for value in probe["state"].values()) == contract["state_numel"],
                "model_structure:only_offset_trainable": len(base_named) == contract["base_parameter_tensors"] and
                    len(named) == contract["base_parameter_tensors"] + 1 and
                    [name for name, value in named.items() if value.requires_grad] == ["energy_offset"] and
                    all(not value.requires_grad for value in base_named.values()),
                "model_structure:offset_shape_initial": isinstance(offset, nn.Parameter) and
                    tuple(offset.shape) == (1,) and torch.equal(offset.detach(), torch.zeros_like(offset)),
                "model_structure:checkpoint_config": all(
                    actual_config.get(name) == value for name, value in expected_config.items()) and
                    len(probe["checkpoint_config"].get("atomic_numbers", [])) == contract["elements"] and
                    all(name in probe["checkpoint_config"] for name in contract["config_fields"]),
            }
            for index, (name, passed) in enumerate(flags.items(), 1):
                update_check(index, bool(passed), "Actual candidate loader/model vs fixed payload; no evaluator zeroing or initialization")

    except Exception as e:
        error_detail = traceback.format_exc()
        if checks[0][1]:
            checks.append(("model_structure:structure_execution", False, error_detail))
        else:
            update_check(0, False, error_detail)

    return checks


def check_model_forward_executable(config: dict, caller_globals=None, caller_locals=None):
    """
    检查模型的前向传播是否可以正常执行。

    config 示例:
    {
        "model_file": "model.py",
        "model_class": "MyModel",
        "config_name": "config.yaml",
        "model_args": {"space_dim": "${config.space_dim}", "n_layers": 3, "hidden_size": 512},  # 模型参数（支持占位符）
        "forward_args": {"x": [[1.0, 2.0], [3.0, 4.0]], "fx": "None", "T": $t"},  # 前向传播参数（支持占位符）
        "out_shape": [2,2,8],
        "pre_forward_hooks": [  # 可选：前置钩子
            {"function": "load_weights", "args": {"checkpoint": "ckpt.pth"}}
        ]
    }

    检查点:
        - forward_executable:forward_run: 检查前向传播能否执行
        - forward_executable:returns_tensor_shape: 检查输出是否为 Tensor 且维度正确
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
    ]
    if config.get("output_shapes"):
        CHECKPOINT_NAMES.append("forward_executable:finite")
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    # ================================================================
    # 主流程
    # ================================================================
    current_index = 0

    try:
        # ------------------ 1. 加载模型模块 ------------------
        model_file = config.get("model_file")
        if not model_file:
            raise ValueError("config['model_file'] is missing or empty")

        model_module = _load_candidate_modules(model_file)
        context = {}  # 初始化上下文

        # ------------------ 2. 加载配置 ------------------
        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("config['config_name'] is missing or empty")

        model_config = _load_candidate_config(config_name)
        context["config"] = model_config
        if isinstance(model_config, dict):
            context.update(model_config)

        # ------------------ 3. 执行前置钩子（如果有） ------------------
        pre_hooks = config.get("pre_forward_hooks", [])
        context = _execute_hooks(model_module, pre_hooks, context, globals(), caller_locals)

        # ------------------ 4. 构建模型 ------------------
        model_class_name = config.get("model_class")
        if not model_class_name:
            raise ValueError("config['model_class'] is missing or empty")

        model_cls = getattr(model_module, model_class_name)
        if model_cls is None or not callable(model_cls):
            raise ValueError(f"Model class '{model_class_name}' not found or not callable")

        # 获取模型参数（必须配置 model_args）
        model_args = config.get("model_args")
        if not model_args and not config.get("model_ref"):
            raise ValueError("config['model_args'] is required")

        resolved_model_args = _resolve_args(model_args, context)
        model = (context["probe"][config["model_ref"]]
                 if config.get("model_ref") else model_cls(**resolved_model_args))
        model = model.to(DEVICE).eval()

        # ------------------ 7. 执行前向传播 ------------------
        # 获取前向传播参数（可选，默认为空字典）
        context.update(context.get("probe", {}))
        forward_args_config = config.get("forward_args", {})
        resolved_forward_args = _resolve_args(forward_args_config, context)

        # 获取模型的 forward 方法
        forward_method = getattr(model, 'forward', None)
        if forward_method is None or not callable(forward_method):
            raise ValueError("Model does not have a callable 'forward' method")

        # 获取参数类型注解
        sig = inspect.signature(forward_method)
        param_types = {}
        for param_name, param in sig.parameters.items():
            # 跳过 self 和 args/kwargs
            if param_name in ('self', 'args', 'kwargs'):
                continue
            # 获取类型注解
            if param.annotation != inspect._empty:
                param_types[param_name] = param.annotation

        # ------------------ 7. 根据类型转换参数 ------------------
        converted_forward_args = {param_name: _convert_arg_to_type(value, param_types.get(param_name)) for
                                  param_name, value in resolved_forward_args.items()}
        converted_forward_args = _move_tensors_to_device(converted_forward_args)

        # Forces/stress may be coordinate derivatives even for a frozen model.
        with torch.set_grad_enabled(config.get("enable_grad", False)):
            out = (getattr(model_module, config["forward_function"])(model, **converted_forward_args)
                   if config.get("forward_function") else model(**converted_forward_args))

        # 更新检查点：forward_run 通过
        update_check(
            current_index,
            True,
            f"Forward pass executed successfully with args"
        )

        # ------------------ 8. 检查输出是否为 Tensor 且维度正确 ------------------
        current_index += 1

        if config.get("output_shapes"):
            if not isinstance(out, dict):
                raise TypeError("REVIEW_REQUIRED: bind the candidate energy/forces output mapping")
            shapes = config["output_shapes"]
            ok = all(torch.is_tensor(out.get(key)) and list(out[key].shape) in allowed for key, allowed in shapes.items())
            update_check(current_index, ok, f"Expected output shapes={shapes}; actual=" +
                         str({key: list(value.shape) for key, value in out.items() if torch.is_tensor(value)}))
            current_index += 1
            update_check(current_index, all(torch.is_tensor(out.get(key)) and bool(torch.isfinite(out[key]).all()) for key in shapes),
                         "Energy/force finite-value check")
        else:
            # 保留 Base 的 Tensor 检查主体，仅修正 out_shape 键及配置形状转换。
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"Output is not a Tensor: {type(out)}")

            same_shape = (tuple(out.shape) == tuple(config.get("out_shape", [])))
            update_check(
                current_index,
                same_shape,
                f"Output is Tensor with shape={tuple(out.shape)}, expected out_shape={config.get('out_shape', [])}"
            )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    检查候选模型与标准模型的前向传播是否等价。

    config 示例:
    {
        # 候选模型配置
        "candidate": {
            "model_file": "model.py",
            "model_class": "MyModel",
            "config_name": "config_candidate.yaml",
            "pre_forward_hooks": [
                {"function": "load_model", "args": {"device": "cuda"}, "save_as": "model"},
                {"function": "load_weights", "args": {"checkpoint": "ckpt.pth"}}
            ],
            "model_args": {"config": "${config}", "hidden_size": 512},
            "forward_args":{"batch":"$x"}
        },
        # 标准模型配置
        "reference": {
            "model_file": "reference_model.py",
            "model_class": "RefModel",
            "config_name": "config_reference.yaml",
            "pre_forward_hooks": [
                {"function": "load_model", "args": {"device": "cpu"}}
            ],
            "model_args": {"config": "${config}", "hidden_size": 512},
            "forward_args":{"batch":"$x"}
        },
        # 比较参数
        "compare_atol": 1e-5
    }

    检查点:
        - forward_equivalence:same_forward: 检查两个模型的前向传播输出是否一致
    """

    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "forward_equivalence:same_forward",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    def load_model_from_config(impl_config: dict, context: dict, is_candidate: bool = True) -> torch.nn.Module:
        """
        根据配置加载模块并构建模型

        Args:
            impl_config: 实现配置
            context: 上下文
            is_candidate: 是否为候选实现
        """
        # 验证必要配置
        model_file = impl_config.get("model_file")
        if not model_file:
            raise ValueError("model_file is required")

        model_class_name = impl_config.get("model_class")
        if not model_class_name:
            raise ValueError("model_class is required")

        config_name = impl_config.get("config_name")
        if not config_name:
            raise ValueError("config_name is required")

        # 加载模块
        if is_candidate or impl_config.get("from_candidate", False):
            module = _load_candidate_modules(model_file)
            model_config = _load_candidate_config(config_name)
        else:
            module = _load_reference_modules(model_file)
            model_config = _load_reference_config(config_name)

        # 将配置添加到上下文中
        context["config"] = model_config
        if isinstance(model_config, dict):
            context.update(model_config)


        # 执行前置钩子
        pre_hooks = impl_config.get("pre_forward_hooks", [])
        context = _execute_hooks(module, pre_hooks, context, globals(), caller_locals)
        if impl_config.get("model_ref"):
            context.update(context.get("probe", {}))
            return context["probe"][impl_config["model_ref"]]

        # 获取模型类
        model_class = getattr(module, model_class_name, None)
        if model_class is None or not callable(model_class):
            raise ValueError(f"Model class '{model_class_name}' not found or not callable")

        # 获取模型参数并解析占位符
        model_args = impl_config.get("model_args", {})
        resolved_model_args = _resolve_args(model_args, context)

        # 构建模型
        model = model_class(**resolved_model_args)
        return model

    def compare_outputs(standard_out: Any, generated_out: Any, atol: float) -> tuple:
        """
        比较两个输出是否一致

        Returns:
            (same_shape, same_values, shape_info, diff_info)
        """
        # 如果是 dict，递归比较
        if isinstance(standard_out, dict) and isinstance(generated_out, dict):
            if standard_out.keys() != generated_out.keys():
                return False, False, "dict keys mismatch", f"keys: {standard_out.keys()} vs {generated_out.keys()}"

            shape_matches = []
            value_matches = []
            for key in standard_out.keys():
                same_shape, same_values, shape_info, diff_info = compare_outputs(
                    standard_out[key], generated_out[key], atol
                )
                shape_matches.append(same_shape)
                value_matches.append(same_values)
                if not same_values:
                    return all(shape_matches), False, shape_info, diff_info

            return all(shape_matches), all(value_matches), "All outputs match", None

        # 如果是 tuple/list，递归比较
        if isinstance(standard_out, (tuple, list)) and isinstance(generated_out, (tuple, list)):
            if len(standard_out) != len(generated_out):
                return False, False, "length mismatch", f"length: {len(standard_out)} vs {len(generated_out)}"

            shape_matches = []
            value_matches = []
            for i, (s, g) in enumerate(zip(standard_out, generated_out)):
                same_shape, same_values, shape_info, diff_info = compare_outputs(s, g, atol)
                shape_matches.append(same_shape)
                value_matches.append(same_values)
                if not same_values:
                    return all(shape_matches), False, f"item {i}: {shape_info}", diff_info

            return all(shape_matches), all(value_matches), "All outputs match", None

        # 如果是标量或非 tensor，直接比较
        if not torch.is_tensor(standard_out) or not torch.is_tensor(generated_out):
            try:
                same = standard_out == generated_out
                return True, same, "scalar comparison", None if same else f"{standard_out} != {generated_out}"
            except Exception:
                return False, False, "cannot compare scalar", "incomparable types"

        # Tensor 比较
        standard_out = standard_out.cpu()
        generated_out = generated_out.cpu()

        shape_match = standard_out.shape == generated_out.shape
        shape_info = f"shape: standard={standard_out.shape}, generated={generated_out.shape}"

        if not shape_match:
            return False, False, shape_info, None

        # 检查值
        max_abs_diff = float((standard_out - generated_out).abs().max().item())
        value_match = max_abs_diff <= atol

        if not value_match:
            diff_info = f"max_abs_diff={max_abs_diff:.6e}, atol={atol}"
            return True, False, shape_info, diff_info

        return True, True, shape_info, None

    def convert_forward_args(model, resolved_forward_args):
        """
        通过 AST 静态分析推断参数类型
        """
        forward = getattr(model, 'forward', None)
        if forward is None or not callable(forward):
            raise AttributeError(f"{model.__class__.__name__} 没有 forward 方法")

        # 获取参数类型注解
        sig = inspect.signature(forward)
        param_types = {}
        for param_name, param in sig.parameters.items():
            # 跳过 self 和 args/kwargs
            if param_name in ('self', 'args', 'kwargs'):
                continue
            # 获取类型注解
            if param.annotation != inspect._empty:
                param_types[param_name] = param.annotation

        missing_param_types = set(resolved_forward_args) - set(param_types)
        if missing_param_types:
            # ``inspect.getsource`` preserves the indentation of methods in
            # their containing class.  Remove that common indentation before
            # parsing; otherwise a method source starts with an unexpected
            # indent and ``ast.parse`` raises ``IndentationError``.
            try:
                source = textwrap.dedent(inspect.getsource(forward))
                tree = ast.parse(source)
            except (OSError, TypeError):
                # Built-in or dynamically-created methods may not have source.
                tree = None
            except (IndentationError, SyntaxError) as exc:
                # Keep argument conversion usable even when a dynamically
                # generated method cannot be parsed.  Signature annotations,
                # when present, have already been collected above.
                tree = None

            func_def = None
            if tree is not None:
                # ``inspect.getsource`` can return a decorated function or a
                # source fragment nested in a class.  Walking the tree handles
                # both forms and also supports async methods.
                func_def = next(
                    (
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == 'forward'
                    ),
                    None,
                )

            if tree is None or func_def is None:
                return {
                    param_name: _convert_arg_to_type(value, param_types.get(param_name))
                    for param_name, value in resolved_forward_args.items()
                }

            else:
                # 提取参数名
                param_names = []
                # 位置参数（含仅位置参数）
                for arg in (*func_def.args.posonlyargs, *func_def.args.args):
                    if arg.arg not in ('self', 'cls'):
                        param_names.append(arg.arg)
                # 仅关键字参数
                for arg in func_def.args.kwonlyargs:
                    if arg.arg not in ('self', 'cls'):
                        param_names.append(arg.arg)

                # 如果没有参数，返回空结果
                if not param_names:
                    param_types = {}
                else:
                    # 分析函数体
                    inferencer = TypeInferencer(param_names)
                    inferencer.visit(func_def)

                    for param_name, param_type in inferencer.infer_types().items():
                        if param_name in missing_param_types and param_type is not None:
                            param_types[param_name] = param_type

        converted_forward_args = {param_name: _convert_arg_to_type(value, param_types.get(param_name)) for
                                param_name, value in resolved_forward_args.items()}
        return converted_forward_args

    # ================================================================
    # 主流程
    # ================================================================
    current_index = 0

    try:
        # 获取候选和参考配置
        candidate_config = config.get("candidate")
        reference_config = config.get("reference")

        if not candidate_config:
            raise ValueError("config['candidate'] is required")
        if not reference_config:
            raise ValueError("config['reference'] is required")

        # 获取比较精度
        atol = config.get("compare_atol", FORWARD_COMPARE_ATOL)

        # ================================================================
        # 1. 加载候选模型
        # ================================================================
        candidate_context = {}
        candidate_model = load_model_from_config(candidate_config, candidate_context, is_candidate=True)

        # ================================================================
        # 2. 加载标准模型
        # ================================================================
        reference_context = {}
        reference_model = load_model_from_config(reference_config, reference_context, is_candidate=False)

        # ================================================================
        # 3. 初始化参数
        # ================================================================
        if config.get("initialize_parameters", True):
            _initialize_parameters_with_constants(candidate_model)
            _initialize_parameters_with_constants(reference_model)
        if "energy_offset" in config:
            # Test mutations affect only a newly loaded probe, never disk weights.
            with torch.no_grad():
                candidate_model.energy_offset.fill_(config["energy_offset"])

        candidate_model = candidate_model.to(DEVICE)
        reference_model = reference_model.to(DEVICE)

        # 设置 eval 模式
        candidate_model.eval()
        reference_model.eval()

        # ================================================================
        # 5. 执行前向传播
        # ================================================================
        with torch.set_grad_enabled(config.get("enable_grad", False)):
            # 获取前向传播的额外参数（如果有）
            candidate_forward_args_config = config.get("candidate").get("forward_args")
            reference_forward_args_config = config.get("reference").get("forward_args")
            candidate_resolved_forward_args = _resolve_args(candidate_forward_args_config, candidate_context)
            reference_resolved_forward_args = _resolve_args(reference_forward_args_config, reference_context)
            candidate_resolved_forward_args = convert_forward_args(candidate_model, candidate_resolved_forward_args)
            reference_resolved_forward_args = convert_forward_args(reference_model, reference_resolved_forward_args)
            candidate_resolved_forward_args = _move_tensors_to_device(candidate_resolved_forward_args)
            reference_resolved_forward_args = _move_tensors_to_device(reference_resolved_forward_args)

            print(reference_model)
            candidate_fn = candidate_config.get("forward_function")
            reference_fn = reference_config.get("forward_function")
            with ExitStack() as forward_scope:
                if config.get("same_forward_reference", False):
                    captured = []

                    def capture_base(_module, _args, output):
                        # Copy before the adapter can mutate the returned tensors.
                        values = output.values() if isinstance(output, dict) else output
                        if not isinstance(output, (dict, tuple, list)) or not all(torch.is_tensor(value) for value in values):
                            raise ValueError("REVIEW_REQUIRED: bind the actual base energy/forces/stress output layout")
                        captured.append({key: value.detach().clone() for key, value in output.items()} if isinstance(output, dict)
                                        else type(output)(value.detach().clone() for value in output))

                    handle = candidate_context["probe"]["base"].register_forward_hook(capture_base)
                    forward_scope.callback(handle.remove)
                candidate_out = (getattr(candidate_context["probe"]["module"], candidate_fn)(candidate_model, **candidate_resolved_forward_args)
                                 if candidate_fn else candidate_model(**candidate_resolved_forward_args))
                if config.get("same_forward_reference", False):
                    handle.remove()
                    if len(captured) != 1:
                        raise ValueError(f"REVIEW_REQUIRED: expected one actual base forward, observed {len(captured)}")
                    # Only the reference path replays this real output. The actual
                    # candidate model/forward is never replaced or recomputed.
                    forward_scope.enter_context(patch.object(reference_model, "forward", return_value=captured[0]))
                reference_out = (getattr(reference_context["probe"]["module"], reference_fn)(reference_model, **reference_resolved_forward_args)
                                 if reference_fn else reference_model(**reference_resolved_forward_args))
            if config.get("compare_keys"):
                keys = config["compare_keys"]
                if not isinstance(candidate_out, dict) or not isinstance(reference_out, dict) or any(
                        not torch.is_tensor(candidate_out.get(key)) or not torch.is_tensor(reference_out.get(key)) for key in keys):
                    raise ValueError(f"REVIEW_REQUIRED: expose actual adapter/base outputs {keys}; missing stress is not a pass")
                candidate_out = {key: candidate_out[key] for key in keys}
                reference_out = {key: reference_out[key] for key in keys}
                if "energy_offset" in config:
                    reference_out["energy"] = reference_out["energy"] + config["energy_offset"]

        # ================================================================
        # 6. 比较输出
        # ================================================================
        same_shape, same_values, shape_info, diff_info = compare_outputs(
            reference_out, candidate_out, atol
        )

        if not same_shape:
            update_check(current_index, False, f"Shape mismatch: {shape_info}")
        elif not same_values:
            update_check(current_index, False, f"Value mismatch: {diff_info}")
        else:
            update_check(
                current_index,
                True,
                f"Outputs match: {shape_info}, atol={atol}"
            )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_train_structure(config: dict) -> list[tuple]:
    """
    静态检查训练脚本是否具备基本的训练启动逻辑。

    config 示例:
    {
        "files": ["train.py", "train_model.py"],          # 训练文件列表（支持多个）
        "check_config": {
            "entry_names": ["train", "main"],                   # 入口函数名称列表
            "model_names": ["Model", "MyModel"],                # 模型类/函数名称列表
            "model_attributes": ["forward", "backward"],        # 模型属性/方法名称列表
            "dataloader_names": ["DataLoader"],                 # DataLoader 名称列表
            "optimizer_names": ["SGD", "Adam", "AdamW"],        # 优化器名称列表
            "loss_names": ["CrossEntropyLoss", "MSELoss"]       # 损失函数名称列表
        }
    }

    注意：所有检查项都会按照 train_check_config 中配置的内容进行检测，
    未配置的检查项将使用默认值。检查点如下：

    检查点（按执行顺序）:
        - train_start:has_main: 检查是否有训练入口
        - train_start:has_model: 检查是否有模型调用
        - train_start:has_dataloader: 检查是否有 DataLoader
        - train_start:has_loop: 检查是否有训练循环
        - train_start:has_train_mode: 检查是否调用 model.train()
        - train_start:has_optimizer: 检查是否有优化器
        - train_start:has_loss: 检查是否有损失函数
        - train_start:has_backward: 检查是否调用 backward()
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "train_start:has_main",
        "train_start:has_model",
        "train_start:has_dataloader",
        "train_start:has_loop",
        "train_start:has_train_mode",
        "train_start:has_optimizer",
        "train_start:has_loss",
        "train_start:has_backward",
    ]
    if config.get("source_contract"):
        CHECKPOINT_NAMES.append("train:local_source_contract")
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        """更新指定索引的检查结果"""
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ------------------ 1. 检查训练文件是否存在 ------------------
        train_files = config.get("files")
        if not train_files:
            raise ValueError("config['files'] is missing or empty")

        # 检查所有文件是否存在
        existing_files = []
        missing_files = []
        for train_file in train_files:
            train_path = (CANDIDATE_DIR / train_file).resolve()
            if train_path.exists() and train_path.is_file():
                existing_files.append(train_path)
            else:
                missing_files.append(str(train_path))

        if missing_files:
            raise FileNotFoundError(f"Training files not found: {', '.join(missing_files)}")

        # ------------------ 2. 解析所有文件的源代码为 AST ------------------
        # 合并所有文件的 AST 节点，用于后续检查
        all_nodes = []
        file_paths = []
        code_contents = []
        for file_path in existing_files:
            source = file_path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(file_path))
            all_nodes.extend(list(ast.walk(tree)))
            file_paths.append(str(file_path))
            code_contents.append(source)

        # 合并所有代码内容（用于文本匹配检查）
        combined_code = "\n".join(code_contents)

        # 从配置中获取检查参数，若无则使用默认值
        train_check_config = config.get("check_config", {})

        # 入口函数名称（用于检测训练脚本的入口点）
        entry_names = set(train_check_config.get(
            "entry_names",
            ["train", "main", "run_training", "train_model", "fit"]
        ))

        # 模型类/函数名称（用于检测模型定义或调用）
        model_names = set(train_check_config.get(
            "model_names",
            ["Model", "Net", "Network", "MyModel", "ResNet", "CNN", "LSTM"]
        ))
        # 模型属性/方法名称（如 forward, backward 等）
        model_attrs = set(train_check_config.get(
            "model_attributes",
            ["forward", "backward", "parameters", "train", "eval"]
        ))

        # DataLoader 名称
        dataloader_names = set(train_check_config.get(
            "dataloader_names",
            ["DataLoader", "DataLoaderV2"]
        ))

        # 优化器名称
        optimizer_names = set(train_check_config.get(
            "optimizer_names",
            ["SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Optimizer"]
        ))

        # 损失函数名称
        loss_names = set(train_check_config.get(
            "loss_names",
            ["CrossEntropyLoss", "MSELoss", "L1Loss", "BCELoss", "BCEWithLogitsLoss",
             "NLLLoss", "KLDivLoss", "HuberLoss", "SmoothL1Loss"]
        ))

        # ------------------ 3. 检查训练入口 ------------------
        # 检测是否存在入口函数（函数定义匹配 entry_names）
        has_entry = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.lower() in entry_names
            for node in all_nodes
        ) or any(
            # 或者检测是否存在 if __name__ == "__main__" 代码块
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and any(
                isinstance(value, ast.Name) and value.id == "__name__"
                for value in node.test.comparators
            )
            for node in all_nodes
        )
        update_check(
            current_index,
            has_entry,
            f"Entry found: {has_entry}, names: {entry_names}"
        )

        # ------------------ 4. 检查模型调用 ------------------
        current_index += 1
        # 检测模型调用（通过类名、函数名或属性/方法名）
        has_model = any(
            isinstance(node, ast.Call)
            and (
                # 匹配函数/类名，如 Model() 或 build_model()
                    (isinstance(node.func, ast.Name) and node.func.id.lower() in model_names)
                    # 匹配属性/方法调用，如 model.forward() 或 model.train()
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr.lower() in model_attrs
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id.lower() in {"model", "net", "network"})
                    # 匹配 torch.nn.Module 的实例化
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr.lower() in model_names
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "torch")
            )
            for node in all_nodes
        ) or any(
            # 检测是否定义了模型类
            isinstance(node, ast.ClassDef)
            and node.name.lower() in model_names
            for node in all_nodes
        )
        update_check(
            current_index,
            has_model,
            f"Model found: {has_model}, names: {model_names}"
        )

        # ------------------ 5. 检查 DataLoader ------------------
        current_index += 1
        has_dataloader = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in dataloader_names
            for node in all_nodes
        ) or any(
            # 检测是否导入了 DataLoader（import 语句）
            isinstance(node, ast.ImportFrom)
            and any(alias.name in dataloader_names for alias in node.names)
            for node in all_nodes
        )
        has_dataloader = has_dataloader or any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func) in train_check_config.get("qualified_dataloader_calls", [])
            for node in all_nodes)
        update_check(
            current_index,
            has_dataloader,
            f"DataLoader found: {has_dataloader}, names: {dataloader_names}"
        )

        # ------------------ 6. 检查训练循环 ------------------
        current_index += 1
        has_loop = any(
            isinstance(node, (ast.For, ast.While))
            for node in all_nodes
        )
        # 更精确的检查：检测循环中是否包含训练相关操作
        has_training_loop = False
        for node in all_nodes:
            if isinstance(node, (ast.For, ast.While)):
                # 检查循环体是否包含 backward() 或 optimizer.step() 等训练操作
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if (isinstance(child.func, ast.Attribute)
                                and child.func.attr in {"backward", "step"}):
                            has_training_loop = True
                            break
                if has_training_loop:
                    break
        has_loop = has_loop and has_training_loop
        if config.get("single_step_function"):
            has_loop = has_loop or any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                      and node.func.id == config["single_step_function"] for node in all_nodes)
        update_check(
            current_index,
            has_loop,
            f"Training loop found: {has_loop}"
        )

        # ------------------ 7. 检查 model.train() ------------------
        current_index += 1
        has_train_mode = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id.lower() in {"model", "net", "network"}
            for node in all_nodes
        )
        if config.get("frozen_eval_allowed"):
            has_train_mode = has_train_mode or any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                                   and node.func.attr == "eval" for node in all_nodes)
        update_check(
            current_index,
            has_train_mode,
            f"model.train() found: {has_train_mode}"
        )

        # ------------------ 8. 检查优化器 ------------------
        current_index += 1
        has_optimizer = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in optimizer_names
            for node in all_nodes
        ) or any(
            # 检测是否实例化了优化器，如 torch.optim.Adam()
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "torch"
            and node.func.value.attr == "optim"
            and node.func.attr in optimizer_names
            for node in all_nodes
        ) or any(
            # 检测 optimizer.step() 调用
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "optimizer"
            for node in all_nodes
        )
        update_check(
            current_index,
            has_optimizer,
            f"Optimizer found: {has_optimizer}, names: {optimizer_names}"
        )

        # ------------------ 9. 检查损失函数 ------------------
        current_index += 1
        has_loss = any(
            isinstance(node, ast.Call)
            and (
                # 直接调用损失函数，如 CrossEntropyLoss()
                    (isinstance(node.func, ast.Name) and node.func.id in loss_names)
                    # 通过 torch.nn 调用，如 torch.nn.CrossEntropyLoss()
                    or (isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "torch"
                        and node.func.attr in loss_names)
                    # 通过 nn 别名调用，如 nn.CrossEntropyLoss()
                    or (isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "nn"
                        and node.func.attr in loss_names)
                    # 检测 loss.backward() 调用
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "backward"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id.lower() in {"loss", "loss_fn", "criterion"})
            )
            for node in all_nodes
        ) or any(
            # 检测 loss 变量的赋值
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id.lower() in {"loss", "loss_fn", "criterion"}
                for target in node.targets
            )
            for node in all_nodes
        )
        update_check(
            current_index,
            has_loss,
            f"Loss function found: {has_loss}, names: {loss_names}"
        )

        # ------------------ 10. 检查 backward() 调用 ------------------
        current_index += 1
        has_backward = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "backward"
            for node in all_nodes
        )
        update_check(
            current_index,
            has_backward,
            f"backward() found: {has_backward}"
        )

        if config.get("source_contract"):
            current_index += 1
            prohibited = set(config["source_contract"]["prohibited_imports"])
            violations = []
            paths = list(CANDIDATE_DIR.rglob("*.py"))
            for path in paths:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        names = [node.module]
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            names = [node.args[0].value]
                    for name in names:
                        if name.split(".")[0] in prohibited:
                            violations.append(f"{path.relative_to(CANDIDATE_DIR)}:{node.lineno}:{name}")
            update_check(current_index, not violations and (CANDIDATE_DIR / config["source_contract"]["required_package"]).is_dir(),
                         f"Static local-package import evidence: {violations}; includes portable_nequix; not arbitrary dynamic-import proof")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_parameter_update(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    静态检查训练脚本中的参数更新逻辑。

    config 示例:
    {
        "files": ["train.py", "model.py"]  # 支持多个文件
    }

    检查点:
        - parameter_update:has_optimizer: 检查是否有优化器
        - parameter_update:has_zero_grad: 检查是否调用 zero_grad
        - parameter_update:has_backward: 检查是否调用 backward
        - parameter_update:has_step: 检查是否调用 step
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "parameter_update:has_optimizer",
        "parameter_update:has_zero_grad",
        "parameter_update:has_backward",
        "parameter_update:has_step",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ------------------ 1. 获取训练文件列表 ------------------
        train_files = config.get("files")
        if not train_files:
            raise ValueError("config['train_files'] is missing or empty")

        # 确保是列表格式
        if isinstance(train_files, str):
            train_files = [train_files]
        elif not isinstance(train_files, list):
            raise TypeError("config['train_files'] must be a string or list of strings")

        # ------------------ 2. 解析所有文件的 AST ------------------
        has_optimizer = False
        has_zero_grad = False
        has_backward = False
        has_step = False

        optimizer_names = {
            "adam", "adamw", "sgd", "rmsprop", "adagrad",
            "adadelta", "adamax", "lbfgs", "adamax"
        }

        for train_file in train_files:
            train_path = (CANDIDATE_DIR / train_file).resolve()

            # 文件存在性由其他函数检查，这里直接解析
            tree = ast.parse(
                train_path.read_text(encoding="utf-8-sig"),
                filename=str(train_path),
            )

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                if isinstance(node.func, ast.Name):
                    if node.func.id.lower() in optimizer_names:
                        has_optimizer = True

                elif isinstance(node.func, ast.Attribute):
                    # Handle qualified constructors such as torch.optim.AdamW(...).
                    if node.func.attr.lower() in optimizer_names:
                        has_optimizer = True
                    elif node.func.attr == "zero_grad":
                        has_zero_grad = True
                    elif node.func.attr == "backward":
                        has_backward = True
                    elif node.func.attr == "step":
                        has_step = True

        # ------------------ 3. 更新检查结果 ------------------
        update_check(
            current_index,
            has_optimizer,
            f"Optimizer found: {has_optimizer}"
        )

        current_index += 1
        update_check(
            current_index,
            has_zero_grad,
            f"zero_grad found: {has_zero_grad}"
        )

        current_index += 1
        update_check(
            current_index,
            has_backward,
            f"backward found: {has_backward}"
        )

        current_index += 1
        update_check(
            current_index,
            has_step,
            f"step found: {has_step}"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_smoke_training(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple]:
    """
    运行冒烟测试，使用最小配置验证训练流程能否启动。

    config 示例:
    {
        "run_file": "train.py",           # 训练模块文件
        "config_name": "config.yaml",          # 配置文件名称
        "train_fn_name": "train",           # 训练入口函数名
        "smoke_config": {                   # 冒烟测试覆盖配置
            "epochs": 1,
            "batch_size": 8
        },
        "pre_train_hooks": [                # 前置钩子（可选，参考 check_training）
            {
                "function": "build_model",
                "args": {"model_name": "resnet50"},
                "save_as": "model"
            }
        ],
        "train_args": {                     # 训练函数参数
            "model": "$model",
            "config": "$config"
        }
    }

    检查点:
        - training:create_work_dir: 检查能否创建工作目录
        - training:smoke_test: 检查冒烟测试是否通过
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "training:create_work_dir",
        "training:smoke_test",
    ]
    if config.get("offset_contract"):
        CHECKPOINT_NAMES.extend([
            "training:optimizer_contract", "training:actual_weighted_loss",
            "training:step_sequence", "training:offset_and_frozen",
            "training:checkpoint_payload",
        ])
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    observation_scope = ExitStack()
    observations = []
    active = {}

    try:
        # ------------------ 1. 获取配置 ------------------
        train_fn_name = config.get("train_fn_name", "train")
        train_file = config.get("run_file")
        if not train_file:
            raise ValueError("run_file not provided in config")

        # ------------------ 2. 检查训练接口 ------------------
        # 使用 _load_candidate_modules 加载模块（与 check_training 一致）
        train_module = _load_candidate_modules(train_file)
        train_fn = getattr(train_module, train_fn_name, None)
        if train_fn is None or not callable(train_fn):
            raise ImportError(f"Train function '{train_fn_name}' not found or not callable in {train_file}")

        # ------------------ 3. 创建工作目录 ------------------
        # work_dir = Path(tempfile.mkdtemp(prefix="training-smoke")).resolve()
        work_dir = (CANDIDATE_DIR / "smoke-training").resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = work_dir / "checkpoints"
        results_dir = work_dir / "results"
        if config.get("offset_contract") and (results_dir.exists() or results_dir.is_symlink()):
            # Preserve old artifacts, but never accept them as this train call's output.
            previous_dir = Path(tempfile.mkdtemp(prefix="previous-results-", dir=work_dir))
            results_dir.rename(previous_dir / "results")
            print(f"PREVIOUS_SMOKE_RESULTS={previous_dir / 'results'}")

        update_check(current_index, True, f"Work directory created: {work_dir}")

        # ------------------ 4. 加载配置 ------------------
        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("train_config_name not provided in config")

        # 加载完整配置（与 check_training 一致）
        candidate_config = _load_candidate_config(config_name)
        runtime_config = deepcopy(candidate_config)

        # ------------------ 5. 构造冒烟配置（轻量覆盖） ------------------
        current_index += 1

        # 默认冒烟配置（epochs=1, batch_size=8 等轻量参数）
        smoke_config = config.get("smoke_config", {})
        context = {"checkpoint_dir": checkpoint_dir, "results_dir": results_dir, "candidate_dir": str(CANDIDATE_DIR)}
        smoke_config = _resolve_args(smoke_config, context)

        runtime_config.update(smoke_config)
        context["config"] = runtime_config
        if isinstance(runtime_config, dict):
            context.update(runtime_config)

        dict_train_entry = (config.get("dict_train_args") is not None
                            and tuple(inspect.signature(train_fn).parameters) == ("config",))
        if dict_train_entry:
            context["train_config"] = deepcopy(runtime_config)
            context["train_config"].update(_resolve_args(config.get("dict_train_overrides", {}), context))

        # ------------------ 6. 执行前置钩子（新增，参考 check_training） ------------------
        pre_hooks = config.get("pre_train_hooks", [])
        for idx, hook_config in enumerate(pre_hooks):
            func_name = hook_config.get("function")
            args_config = hook_config.get("args", {})

            context = _execute_hooks(train_module, [hook_config], context, globals(), caller_locals)
            save_as = hook_config.get("save_as")
            if save_as and isinstance(runtime_config, dict):
                runtime_config[save_as] = context[save_as]

        if config.get("offset_contract"):
            contract = config["offset_contract"]
            original_step = getattr(train_module, contract["step_function"])
            original_predict = getattr(train_module, contract["prediction_function"], None)
            direct_prediction = not callable(original_predict) and contract.get("observe_model_forward", False)
            if not callable(original_predict) and not direct_prediction:
                raise ValueError("REVIEW_REQUIRED: bind the prediction function called by candidate train_step")
            original_backward = torch.Tensor.backward
            original_clip = torch.nn.utils.clip_grad_norm_

            def observed_model_forward(_model, _args, output):
                if active:
                    active["events"].append("forward")
                    active["prediction"] = {key: output[key].detach().clone() for key in ("energy", "forces")}

            def observed_predict(*args, **kwargs):
                output = original_predict(*args, **kwargs)
                observed_model_forward(None, None, output)
                return output

            def observed_backward(tensor, *args, **kwargs):
                if active:
                    active["events"].append("backward")
                    active["losses"].append(tensor.detach().clone())
                return original_backward(tensor, *args, **kwargs)

            def observed_clip(parameters, max_norm, *args, **kwargs):
                if active:
                    # Materialize once, then pass the same real parameters to PyTorch.
                    parameters = [parameters] if torch.is_tensor(parameters) else list(parameters)
                    active["events"].append("clip")
                    active["clip_norms"].append(float(max_norm))
                    active["clip_targets"].append(any(
                        value is active["named"]["energy_offset"] and value.grad is not None
                        for value in parameters))
                return original_clip(parameters, max_norm, *args, **kwargs)

            def observed_step(*args, **kwargs):
                # Actual public signature is an explicit binding, not retried.
                bound = inspect.signature(original_step).bind(*args, **kwargs)
                values = [bound.arguments[name] for name in contract["step_arguments"]]
                model, optimizer, batch, step_config = values
                named = dict(model.named_parameters())
                base = getattr(model, context["probe"]["base_path"])
                base_named = dict(base.named_parameters())
                offset = named.get("energy_offset")
                if not isinstance(offset, nn.Parameter):
                    raise ValueError("energy_offset must be a registered Parameter")
                record = {
                    "model": model, "optimizer": optimizer, "events": [], "losses": [],
                    "clip_norms": [], "clip_targets": [], "gradients": [], "before": offset.detach().clone(),
                    "frozen_before": {name: value.detach().clone() for name, value in base_named.items()},
                    "batch": batch, "named": named, "optimizer_settings": [],
                    "source_state_match": set(base.state_dict()) == set(context["probe"]["state"]) and all(
                        torch.equal(value.detach().cpu(), context["probe"]["state"][name].cpu()) for name, value in base.state_dict().items()),
                }
                reference_batch = context["probe"]["batch"]
                data, reference_data = batch.get("data", batch), reference_batch.get("data", reference_batch)
                positions = data["positions"] if isinstance(data, dict) else data.positions
                reference_positions = reference_data["positions"] if isinstance(reference_data, dict) else reference_data.positions
                record["batch_matches"] = (all(torch.allclose(batch[key].detach().cpu().reshape(-1), reference_batch[key].detach().cpu().reshape(-1),
                                                         rtol=1e-6, atol=1e-6) for key in ("energy", "forces")) and
                                           torch.allclose(positions.detach().cpu(), reference_positions.detach().cpu(), rtol=1e-6, atol=1e-6))
                active.update(record)
                zero_grad, step = optimizer.zero_grad, optimizer.step

                def observed_zero(*a, **kw):
                    active["events"].append("zero_grad")
                    return zero_grad(*a, **kw)

                def observed_optimizer_step(*a, **kw):
                    active["events"].append("step")
                    active["optimizer_settings"].append([
                        (group["lr"], group["weight_decay"], list(group["params"])) for group in optimizer.param_groups])
                    return step(*a, **kw)

                def observed_gradient(gradient):
                    active["gradients"].append(gradient.detach().clone())
                    # No return value: observe only, never replace the gradient.

                handle = offset.register_hook(observed_gradient) if offset.requires_grad else None
                prediction_handle = model.register_forward_hook(observed_model_forward) if direct_prediction else None
                try:
                    with patch.object(optimizer, "zero_grad", observed_zero), patch.object(optimizer, "step", observed_optimizer_step):
                        returned = original_step(*args, **kwargs)
                    record.update(active)
                    record["after"] = offset.detach().clone()
                    record["frozen_same"] = all(torch.equal(value.detach(), record["frozen_before"][name]) for name, value in base_named.items())
                    record["trainable_report"] = list(getattr(train_module, contract["trainable_selector"])(model))
                    observations.append(record)
                    return returned
                finally:
                    if handle is not None:
                        handle.remove()
                    if prediction_handle is not None:
                        prediction_handle.remove()
                    active.clear()

            observation_scope.enter_context(patch.object(train_module, contract["step_function"], observed_step))
            if callable(original_predict):
                observation_scope.enter_context(patch.object(train_module, contract["prediction_function"], observed_predict))
            observation_scope.enter_context(patch.object(torch.Tensor, "backward", observed_backward))
            # Support a directly imported clip_grad_norm_ without a new alias registry.
            for name, value in list(vars(train_module).items()):
                if value is original_clip:
                    observation_scope.enter_context(patch.object(train_module, name, observed_clip))
            observation_scope.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", observed_clip))

        # ------------------ 7. 执行冒烟测试 ------------------
        # 解析训练函数参数（与 check_training 一致）
        train_args_config = config.get("dict_train_args") if dict_train_entry else config.get("train_args", {})
        resolved_train_args = _resolve_args(train_args_config, context)

        # 执行训练
        result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)

        update_check(
            current_index,
            True,
            f"Smoke test completed successfully: {train_fn_name}(...) with {config_name}"
        )

        if config.get("offset_contract"):
            current_index += 1
            if len(observations) != 1:
                raise ValueError(f"Expected one actual train_step; observed {len(observations)}")
            observed = observations[0]
            model, optimizer = observed["model"], observed["optimizer"]
            offset = observed["named"]["energy_offset"]
            settings = observed["optimizer_settings"]
            groups = settings[0] if len(settings) == 1 else []
            optimizer_parameters = [value for lr, decay, parameters in groups for value in parameters]
            optimizer_ok = (isinstance(optimizer, torch.optim.AdamW) and len(optimizer_parameters) == 1
                            and optimizer_parameters[0] is offset and all(
                                lr == contract["lr"] and decay == contract["weight_decay"] for lr, decay, parameters in groups))
            update_check(current_index, optimizer_ok, "Actual AdamW settings and optimizer-owned parameters")
            current_index += 1
            prediction, batch = observed.get("prediction"), observed["batch"]
            if prediction is None:
                raise ValueError("Candidate train_step did not call the bound real prediction function")
            energy = prediction["energy"].reshape(-1)
            target_energy = batch["energy"].detach().to(energy).reshape(-1)
            forces, target_forces = prediction["forces"], batch["forces"].detach().to(prediction["forces"])
            expected_loss = contract["energy_weight"] * _compute_mse(energy, target_energy) + contract["force_weight"] * _compute_mse(forces, target_forces)
            expected_gradient = 2 * contract["energy_weight"] * (energy - target_energy).mean()
            actual_losses = observed["losses"]
            gradients = observed["gradients"]
            loss_ok = (observed["batch_matches"] and len(actual_losses) == 1 and actual_losses[0].numel() == 1 and math.isclose(
                float(actual_losses[0]), expected_loss, rel_tol=1e-5, abs_tol=1e-6) and bool(gradients)
                and all(bool(torch.isfinite(value).all()) for value in gradients)
                and torch.allclose(sum(gradients), expected_gradient.reshape(1), rtol=1e-5, atol=1e-6))
            update_check(current_index, bool(loss_ok), f"Actual backward loss vs 1*energy MSE+5*force MSE: expected={expected_loss}; offset derivative checked")
            current_index += 1
            events = observed["events"]
            sequence = all(name in events for name in ("zero_grad", "forward", "backward", "clip", "step"))
            sequence = sequence and events.index("zero_grad") < events.index("backward") < events.index("clip") < events.index("step")
            sequence = sequence and events.index("forward") < events.index("backward")
            update_check(current_index, sequence and observed["clip_norms"] == [contract["gradient_clip_norm"]]
                         and observed["clip_targets"] == [True],
                         f"Actual train_step events={events}; clip norms={observed['clip_norms']}; clip contains offset gradient={observed['clip_targets']}")
            current_index += 1
            actual_trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
            reported = observed["trainable_report"]
            # train() may change the backbone after train_step() has returned.
            final_base = getattr(model, context["probe"]["base_path"])
            final_base_named = dict(final_base.named_parameters())
            final_frozen_same = set(final_base_named) == set(observed["frozen_before"]) and all(
                torch.equal(value.detach().cpu(), observed["frozen_before"][name].cpu())
                for name, value in final_base_named.items())
            unchanged = (len(observed["frozen_before"]) == contract["base_parameter_tensors"] and observed["source_state_match"] and observed["frozen_same"]
                         and final_frozen_same and all(not value.requires_grad for value in final_base.parameters()))
            trainable_ok = ([name for name, value in actual_trainable] == ["energy_offset"] and
                            len(reported) == 1 and reported[0][0] == "energy_offset" and reported[0][1] is offset)
            update_check(current_index, unchanged and trainable_ok and not torch.equal(observed["before"], observed["after"]),
                         f"Only offset changes; frozen backbone checked at train_step return and train return; final_frozen_same={final_frozen_same}")
            current_index += 1
            output_dir = Path(_resolve_args({"path": contract["output_dir"]}, context)["path"])
            payload = torch.load(output_dir / contract["checkpoint_name"], map_location="cpu", weights_only=True)
            expected_keys = set(contract["checkpoint_keys"])
            if isinstance(payload, dict):
                payload = dict(payload)
                for key, paths in contract.get("checkpoint_key_paths", {}).items():
                    present = [path for path in paths if path in payload]
                    if len(present) > 1:
                        raise ValueError(f"REVIEW_REQUIRED: ambiguous checkpoint keys for {key}: {present}")
                    if present:
                        payload[key] = payload[present[0]]
            if not isinstance(payload, dict) or not expected_keys.issubset(payload):
                raise ValueError(f"Checkpoint must contain {sorted(expected_keys)}; no format/model_name requirement")
            state = model.state_dict()
            saved = payload["model"]
            saved_state_ok = isinstance(saved, dict) and set(saved) == set(state) and all(
                torch.is_tensor(saved[name]) and torch.equal(saved[name].cpu(), value.detach().cpu()) for name, value in state.items())
            opt = payload["optimizer"]
            config_payload = payload["config"]
            header = context["probe"]["checkpoint_config"]
            # Prompt says parsed config, not exclusively the checkpoint JSON header.
            parsed_config_ok = isinstance(config_payload, dict) and any(
                source and all(config_payload.get(key) == value for key, value in source.items())
                for source in (candidate_config, header))
            metrics = [json.loads(line) for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            # Absolute path: an old log in the process cwd must not satisfy this run.
            log_checks = check_files({"required_files": [str(output_dir / "train.log")], "phase": "smoke_train"})
            payload_ok = (all(item[1] for item in log_checks) and bool(metrics) and all(isinstance(item, dict) for item in metrics) and saved_state_ok and isinstance(opt, dict) and bool(opt.get("state")) and bool(opt.get("param_groups"))
                          and payload["completed_steps"] == 1 and payload["trainable_parameter_names"] == ["energy_offset"]
                          and payload["source_checkpoint_sha256"] == contract["source_checkpoint_sha256"]
                          and parsed_config_ok and isinstance(payload["history"], list) and bool(payload["history"]))
            update_check(current_index, bool(payload_ok), f"Complete current-run artifacts, saved model, non-empty optimizer state, steps/names/source hash, parsed config and history; log={log_checks}")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    finally:
        observation_scope.close()
    return checks
