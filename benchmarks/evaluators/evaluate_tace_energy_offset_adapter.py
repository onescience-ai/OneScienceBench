#!/usr/bin/env python3
"""TACE energy-offset adapter: Prompt-first checks/params evaluator（r5 接口接入）.

Base 底稿：OnescienceBench 47778295a69c。直接复用文件/接口；同名方法完整复制后局部适配。
固定数据/权重仍需真实资源；不读取旧 evaluate()，不自建 Runner，不修改 Standard/Agent。
当前没有完整 TACE 数学/周期邻接图证明；数据与部分源码项保持旧评测深度，见配套说明。
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import math
import textwrap
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple
from contextlib import ExitStack
from unittest.mock import patch

import torch
import yaml

from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    FORWARD_COMPARE_ATOL,
    TypeInferencer,
    _load_candidate_modules,
    _load_reference_modules,
    _load_candidate_config,
    _load_reference_config,
    _load_class,
    _resolve_args,
    _execute_hooks,
    _convert_arg_to_type,
    _move_tensors_to_device,
    _initialize_parameters_with_constants,
    _compute_mse,
    check_files,
    check_interface,
)


def load_tace_adapter(module, config, checkpoint_path):
    """接入已核对的两种加载流程；主干返回值交给候选自己的 build_model 包装。"""
    loaded = getattr(module, config.get("model_loader", "load_model"))(str(checkpoint_path), "cpu")
    if isinstance(loaded, (tuple, list)):
        model = loaded[0]
    elif isinstance(loaded, module.TACEEnergyOffsetAdapter):
        model = loaded
    elif isinstance(loaded, torch.nn.Module):
        train_module = _load_candidate_modules(config["train_file"])
        runtime_config = train_module.load_config(str(CANDIDATE_DIR / config["config_name"]))
        # 不由 evaluator 构造适配器、冻结或补加载；执行候选真实训练构建入口。
        model = train_module.build_model(runtime_config)
    else:
        raise TypeError("REVIEW_REQUIRED: bind the candidate load_model return value")
    paths = [name for name in ("base", "backbone") if isinstance(getattr(model, name, None), torch.nn.Module)]
    if len(paths) != 1:
        raise ValueError(f"REVIEW_REQUIRED: bind one frozen backbone on the actual adapter: {paths}")
    return model, paths[0]


def build_tace_batch(module, config, base=None):
    """接入路径/index 或 structures/config 两种数据接口，原生 batch 不重写。"""
    path, index = str(config["data_path"]), config["frame_index"]
    parameters = inspect.signature(module.build_finetune_batch).parameters
    if "path" in parameters and "index" in parameters:
        structures = module.load_training_structures(path)
        batch = module.build_finetune_batch(path, index, "cpu")
    elif "structures" in parameters and "config" in parameters:
        structures = module.load_training_structures(path, index=None)
        if base is None:
            # 数据构造需要真实元素映射；只读固定导出包配置，不代做邻居图或标签处理。
            checkpoint_path = Path(config["checkpoint_file"])
            with checkpoint_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != config["checkpoint_sha256"]:
                raise ValueError(f"Fixed checkpoint SHA256 mismatch: {digest}")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            graph_config = deepcopy(payload["cfg"])
            for key in ("statistics", "target_property", "embedding_property"):
                graph_config.setdefault(key, payload[key])
            dtype = torch.float32
        else:
            graph_config = base.config
            dtype = next(base.parameters()).dtype
        batch = module.build_finetune_batch([structures[index]], graph_config, "cpu", dtype)
    else:
        raise TypeError("REVIEW_REQUIRED: bind the actual build_finetune_batch signature")
    return structures, batch


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
        if config.get("config_loader_file"):
            config_module = _load_candidate_modules(config["config_loader_file"])
            config_data = getattr(config_module, config.get("config_loader", "load_config"))(str(CANDIDATE_DIR / config_name))
        # 已核对的 TACE 配置：training 分组，或平铺字段（index 对应 structure_index）。
        if (config_name == "tace_energy_offset.yaml"
                and config.get("config_loader_file") == "train_tace_energy_offset.py"
                and config.get("config_section") == "training" and "training" not in config_data):
            config_data = dict(config_data)
            if "index" in config_data:
                config_data["structure_index"] = config_data["index"]
        else:
            for section in config.get("config_section", "").split("."):
                if section:
                    config_data = config_data[section]

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
                # TACE train_step 按 float 使用该阈值；只接入本字段，后续沿用 Base 值比较。
                if (config_name == "tace_energy_offset.yaml"
                        and config.get("config_loader_file") == "train_tace_energy_offset.py"
                        and config.get("config_section") == "training"
                        and key == "gradient_clip_norm" and isinstance(current_value, str)
                        and type(expected_value) in (int, float)):
                    current_value = _convert_arg_to_type(current_value, float)
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

        if config.get("checkpoint_sha256"):
            with checkpoint_path_resolved.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_hash != config["checkpoint_sha256"]:
                raise ValueError(f"Fixed checkpoint SHA256 mismatch: {actual_hash}")

        checkpoint = torch.load(
            checkpoint_path_resolved,
            map_location="cpu",
            weights_only=True,
        )

        if config.get("tace_checkpoint"):
            required = {"cfg", "state_dict", "statistics", "target_property", "embedding_property"}
            if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
                raise ValueError("TACE checkpoint envelope is incomplete")
            state = checkpoint["state_dict"]
            if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
                raise TypeError("checkpoint.state_dict must be a tensor mapping")
            if len(state) != config["state_count"] or sum(v.numel() for v in state.values()) != config["state_numel"]:
                raise ValueError("Fixed TACE state count/numel mismatch")
            if list(checkpoint["target_property"]) != config["target_property"]:
                raise ValueError("Fixed target_property mismatch")

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
        observed_loads, observed_strict = [], []
        if config.get("tace_checkpoint"):
            original_load, original_state_load = torch.load, torch.nn.Module.load_state_dict

            def observe_load(*args, **kwargs):
                # 不允许为完成验证而执行不安全的候选反序列化。
                observed_loads.append(kwargs.get("weights_only") is True)
                if kwargs.get("weights_only") is not True:
                    raise ValueError("Candidate checkpoint load must explicitly use weights_only=True")
                return original_load(*args, **kwargs)

            def observe_state_load(instance, state, *args, **kwargs):
                strict = args[0] if args else kwargs.get("strict", True)
                observed_strict.append((instance, strict is True))
                return original_state_load(instance, state, *args, **kwargs)

            with patch.object(torch, "load", observe_load), patch.object(torch.nn.Module, "load_state_dict", observe_state_load):
                adapter, base_path = load_tace_adapter(model_module, config, checkpoint_path_resolved)
            model = adapter.get_submodule(base_path)
            resolved_model_args = {"checkpoint": str(checkpoint_path_resolved)}
        else:
            model_class_name = config.get("model_class")
            if not model_class_name:
                raise ValueError("config['model_class'] is missing or empty")

            # 获取模型参数（支持占位符）
            model_args = config.get("model_args", {})
            resolved_model_args = _resolve_args(model_args, context)

            model_cls = getattr(model_module, model_class_name)
            if model_cls is None or not callable(model_cls):
                raise ValueError(f"Model class '{model_class_name}' not found or not callable")

            model = model_cls(**resolved_model_args).to(DEVICE)

        update_check(
            current_index,
            True,
            f"Model built with args: {resolved_model_args}"
        )

        # ------------------ 5. 严格加载 state_dict ------------------
        current_index += 1
        strict_mode = config.get("strict", True)

        if config.get("tace_checkpoint"):
            current = model.state_dict()
            missing = list(set(state_dict) - set(current))
            unexpected = list(set(current) - set(state_dict))
            if not observed_loads or not all(observed_loads):
                raise ValueError("No observed safe candidate checkpoint load")
            if not any(instance is model and strict for instance, strict in observed_strict):
                raise ValueError("REVIEW_REQUIRED: no strict load_state_dict observed on the configured base model")
            if missing or unexpected or not all(torch.equal(current[k].detach().cpu(), value.detach().cpu()) for k, value in state_dict.items()):
                raise ValueError("Candidate loaded base differs from the independently read fixed state_dict")
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
        - model_structure:{rule.id}[:module_path]: 原 Base 公共结构规则
        - TACE 分支追加 scalar_offset、only_offset_trainable、frozen_base
    """
    import torch.nn as nn

    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "model_structure:build_model_ok",
    ]
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
        if not model_class_name and not config.get("tace_checkpoint"):
            raise ValueError("config['model_class'] is missing or empty")

        if not config.get("tace_checkpoint"):
            model_cls = _load_class(model_module, model_class_name)

        # ------------------ 2. 构建模型 ------------------
        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("config['config_name'] is missing or empty")

        candidate_config = _load_candidate_config(config_name)
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)

        if config.get("tace_checkpoint"):
            model, base_path = load_tace_adapter(model_module, config, config["checkpoint_file"])
            model.eval()
        else:
            # 获取模型参数（必须配置 model_args）
            model_args = config.get("model_args")
            if not model_args:
                raise ValueError("config['model_args'] is required")

            resolved_model_args = _resolve_args(model_args, context)
            model = model_cls(**resolved_model_args).to(DEVICE).eval()

        update_check(
            current_index,
            True,
            f"Model built with config: {candidate_config}"
        )

        from base.model_structure_checker import ModuleInfoExtractor, ModelStructureChecker
        checker = ModelStructureChecker()

        rules = config.get("rules")
        if rules is None:
            raise ValueError("config['rules'] is missing")
        if config.get("tace_checkpoint") and base_path != config.get("base_path", "base"):
            rules = deepcopy(rules)
            for rule in rules:
                path = rule.get("params", {}).get("target_path", "")
                source_path = config.get("base_path", "base")
                if path == source_path or path.startswith(source_path + "."):
                    rule["params"]["target_path"] = base_path + path[len(source_path):]
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

        if config.get("tace_checkpoint"):
            base = model.get_submodule(base_path)
            offset = dict(model.named_parameters()).get("energy_offset")
            named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            checks.extend([
                ("model_structure:scalar_offset", isinstance(offset, torch.nn.Parameter)
                 and list(offset.shape) == [1] and bool(torch.equal(offset.detach(), torch.zeros_like(offset))),
                 "energy_offset is a registered [1] parameter initially zero"),
                ("model_structure:only_offset_trainable", [name for name, _ in named] == ["energy_offset"],
                 f"Trainable parameter names={[name for name, _ in named]}"),
                ("model_structure:frozen_base", len(list(base.parameters())) == config["base_param_count"]
                 and all(not p.requires_grad for p in base.parameters()),
                 f"Frozen base parameter tensors={len(list(base.parameters()))}"),
            ])

    except Exception as e:
        error_detail = traceback.format_exc()
        if not checks[0][1]:
            update_check(0, False, error_detail)
        else:
            checks.append(("model_structure:structure_execution", False, error_detail))

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
        - forward_executable:returns_tensor_shape: 普通分支检查 Tensor；TACE 分支检查所配字典输出的形状及有限性
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
    ]
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
        context = _execute_hooks(model_module, pre_hooks, context)

        # ------------------ 4. 构建模型 ------------------
        if config.get("tace_checkpoint"):
            model, base_path = load_tace_adapter(model_module, config, config["checkpoint_file"])
            _, context["batch"] = build_tace_batch(model_module, config, model.get_submodule(base_path))
            model.train()  # 原 TACE 前向检查的模式；forces 可能通过 autograd 计算。
        else:
            model_class_name = config.get("model_class")
            if not model_class_name:
                raise ValueError("config['model_class'] is missing or empty")

            model_cls = getattr(model_module, model_class_name)
            if model_cls is None or not callable(model_cls):
                raise ValueError(f"Model class '{model_class_name}' not found or not callable")

            # 获取模型参数（必须配置 model_args）
            model_args = config.get("model_args")
            if not model_args:
                raise ValueError("config['model_args'] is required")

            resolved_model_args = _resolve_args(model_args, context)
            model = model_cls(**resolved_model_args).to(DEVICE).eval()

        # ------------------ 7. 执行前向传播 ------------------
        # 获取前向传播参数（可选，默认为空字典）
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

        if config.get("tace_checkpoint"):
            with torch.enable_grad():
                out = model_module.predict_energy_forces(model, context["batch"])
        else:
            with torch.no_grad():
                out = model(**converted_forward_args)

        # 更新检查点：forward_run 通过
        update_check(
            current_index,
            True,
            f"Forward pass executed successfully with args"
        )

        # ------------------ 8. 检查输出是否为 Tensor 且维度正确 ------------------
        current_index += 1

        if config.get("tace_checkpoint"):
            expected = config["output_shapes"]
            same_shape = isinstance(out, dict) and all(torch.is_tensor(out.get(k)) and list(out[k].shape) == shape
                                                       for k, shape in expected.items())
            finite = same_shape and all(bool(torch.isfinite(out[k]).all()) for k in expected)
            update_check(current_index, bool(same_shape and finite),
                         f"Required finite TACE outputs: {expected}; shapes_match={same_shape}; finite={finite}")
        else:
            # 检查是否为 Tensor
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"Output is not a Tensor: {type(out)}")

            same_shape = (out.shape == config.get("out_shape,[]"))
            update_check(
                current_index,
                same_shape,
                f"Output is Tensor with shape={tuple(out.shape)}, expected out_dim={config.get('out_shape,[]')}"
            )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    普通分支检查候选与标准模型前向等价；TACE 分支比较适配层与其冻结主干。
    TACE 零偏移和非零偏移由 JSON 分别配置，不构造独立的完整 TACE oracle。

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
        if is_candidate:
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
        context = _execute_hooks(module, pre_hooks, context)

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
        if config.get("tace_checkpoint"):
            module = _load_candidate_modules(config["model_file"])
            candidate_model, base_path = load_tace_adapter(module, config, config["checkpoint_file"])
            candidate_model.eval()
            reference_model = candidate_model.get_submodule(base_path)
            _, batch = build_tace_batch(module, config, reference_model)
            atol = config.get("compare_atol", FORWARD_COMPARE_ATOL)
            with torch.no_grad():
                candidate_model.energy_offset.fill_(config.get("energy_offset", 0.0))
            # 不改固定主干参数；参考就是适配层内的冻结主干，而非另一份 Standard。
            captured_stress = {}
            handles = []
            try:
                for label, observed in (("adapter", candidate_model), ("base", reference_model)):
                    def observe_output(instance, args, output, label=label):
                        if isinstance(output, dict) and torch.is_tensor(output.get("stress")):
                            captured_stress[label] = output["stress"].detach().clone()
                    handles.append(observed.register_forward_hook(observe_output))
                with torch.enable_grad():
                    actual = module.predict_energy_forces(candidate_model, deepcopy(batch))
                    expected = module.predict_energy_forces(reference_model, deepcopy(batch))
            finally:
                for handle in handles:
                    handle.remove()
            # predict_energy_forces 可以只返回 energy/forces；stress 从实际模型前向观察。
            if "stress" not in actual and "adapter" in captured_stress:
                actual = {**actual, "stress": captured_stress["adapter"]}
            if "stress" not in expected and "base" in captured_stress:
                expected = {**expected, "stress": captured_stress["base"]}
            keys = config.get("output_keys", ["energy", "forces", "stress"])
            if any(key not in actual or key not in expected for key in keys):
                raise ValueError("REVIEW_REQUIRED: requested energy/forces/stress output was not exposed; bind the actual stress-enabled forward")
            candidate_out = {key: actual[key].detach().clone() for key in keys}
            reference_out = {key: expected[key].detach().clone() for key in keys}
            reference_out["energy"] = reference_out["energy"] + config.get("energy_offset", 0.0)
        else:
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
            _initialize_parameters_with_constants(candidate_model)
            _initialize_parameters_with_constants(reference_model)

            candidate_model = candidate_model.to(DEVICE)
            reference_model = reference_model.to(DEVICE)

            # 设置 eval 模式
            candidate_model.eval()
            reference_model.eval()

            # ================================================================
            # 5. 执行前向传播
            # ================================================================
            with torch.no_grad():
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
                candidate_out = candidate_model(**candidate_resolved_forward_args)
                reference_out = reference_model(**reference_resolved_forward_args)

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
    if config.get("tace_training"):
        CHECKPOINT_NAMES.extend([
            "training:offset_only_optimizer", "training:real_loss_and_gradient",
            "training:offset_updated", "training:frozen_base_unchanged", "training:artifacts_entry",
        ])
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    handles = []

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

        update_check(current_index, True, f"Work directory created: {work_dir}")

        # ------------------ 4. 加载配置 ------------------
        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("train_config_name not provided in config")

        # 加载完整配置（与 check_training 一致）
        candidate_config = (train_module.load_config(str(CANDIDATE_DIR / config_name))
                            if config.get("tace_training") else _load_candidate_config(config_name))
        runtime_config = deepcopy(candidate_config)

        # ------------------ 5. 构造冒烟配置（轻量覆盖） ------------------
        current_index += 1

        # 默认冒烟配置（epochs=1, batch_size=8 等轻量参数）
        smoke_config = config.get("smoke_config", {})
        context = {"checkpoint_dir": checkpoint_dir, "results_dir": results_dir}
        smoke_config = _resolve_args(smoke_config, context)

        runtime_config.update(smoke_config)
        context["config"] = runtime_config
        if isinstance(runtime_config, dict):
            context.update(runtime_config)

        # ------------------ 6. 执行前置钩子（新增，参考 check_training） ------------------
        pre_hooks = config.get("pre_train_hooks", [])
        for idx, hook_config in enumerate(pre_hooks):
            func_name = hook_config.get("function")
            args_config = hook_config.get("args", {})

            hook_func = getattr(train_module, func_name, None)
            if hook_func is None or not callable(hook_func):
                raise ValueError(f"Pre-hook {idx}: function '{func_name}' not found or not callable")

            resolved_args = _resolve_args(args_config, context)
            hook_result = hook_func(**resolved_args)

            save_as = hook_config.get("save_as")
            if save_as:
                context[save_as] = hook_result
                if isinstance(runtime_config, dict):
                    runtime_config[save_as] = hook_result

        if config.get("tace_training"):
            if "model" in inspect.signature(train_module.build_trainer).parameters:
                model = train_module.build_model(runtime_config)
                optimizer = train_module.build_trainer(model, runtime_config)
            else:
                model, optimizer = train_module.build_trainer(runtime_config, "cpu")
            module = _load_candidate_modules(config["model_file"])
            paths = [name for name in ("base", "backbone") if isinstance(getattr(model, name, None), torch.nn.Module)]
            if len(paths) != 1:
                raise ValueError(f"REVIEW_REQUIRED: bind the actual training backbone: {paths}")
            base = model.get_submodule(paths[0])
            _, batch = build_tace_batch(module, config, base)
            named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            trainable_ids = {id(p) for _, p in named}
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            base_before = {name: p.detach().clone() for name, p in base.named_parameters()}
            observed_gradients, observed_losses, events = [], [], []

            def observe_gradient(gradient):
                observed_gradients.append(gradient.detach().clone())
                events.append("backward")
                # 返回 None，只观察真实 backward，不改变候选梯度。

            offset = dict(model.named_parameters()).get("energy_offset")
            if offset is not None and offset.requires_grad:
                handles.append(offset.register_hook(observe_gradient))
            # 独立数值依据来自同一个模型/真实标签，不要求候选额外返回某个私有 metric 键。
            with torch.enable_grad():
                predicted = module.predict_energy_forces(model, deepcopy(batch))
            reference_energy = (predicted["energy"].detach().reshape(-1) - batch["energy"].reshape(-1)).mean()
            expected_gradient = 2 * float(config["energy_weight"]) * reference_energy
            expected_loss = (float(config["energy_weight"]) * _compute_mse(predicted["energy"].detach().reshape_as(batch["energy"]), batch["energy"])
                             + float(config["force_weight"]) * _compute_mse(predicted["forces"].detach(), batch["forces"]))
            observed_gradients.clear()
            events.clear()
            context.update(model=model, optimizer=optimizer, batch=batch)

        # ------------------ 7. 执行冒烟测试 ------------------
        # 解析训练函数参数（与 check_training 一致）
        train_args_config = config.get("train_args", {})
        resolved_train_args = _resolve_args(train_args_config, context)

        # 执行训练
        if config.get("tace_training"):
            original_backward = torch.Tensor.backward
            original_clip = torch.nn.utils.clip_grad_norm_
            original_zero, original_step = optimizer.zero_grad, optimizer.step

            def observe_backward(tensor, *args, **kwargs):
                if tensor.numel() == 1:
                    observed_losses.append(float(tensor.detach()))
                return original_backward(tensor, *args, **kwargs)

            def observe_clip(parameters, max_norm, *args, **kwargs):
                events.append(("clip", float(max_norm)))
                return original_clip(parameters, max_norm, *args, **kwargs)

            def observe_zero(*args, **kwargs):
                events.append("zero_grad")
                return original_zero(*args, **kwargs)

            def observe_step(*args, **kwargs):
                events.append("step")
                return original_step(*args, **kwargs)

            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.Tensor, "backward", observe_backward))
                stack.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", observe_clip))
                stack.enter_context(patch.object(optimizer, "zero_grad", observe_zero))
                stack.enter_context(patch.object(optimizer, "step", observe_step))
                for name, value in list(vars(train_module).items()):
                    if value is original_clip:
                        stack.enter_context(patch.object(train_module, name, observe_clip))
                result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)
        else:
            result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)

        update_check(
            current_index,
            True,
            f"Smoke test completed successfully: {train_fn_name}(...) with {config_name}"
        )

        if config.get("tace_training"):
            current_index += 1
            groups_ok = isinstance(optimizer, torch.optim.AdamW) and all(
                float(group["lr"]) == config["lr"] and float(group["weight_decay"]) == config["weight_decay"]
                for group in optimizer.param_groups)
            update_check(current_index, [name for name, _ in named] == ["energy_offset"]
                         and optimizer_ids == trainable_ids and groups_ok,
                         f"Trainable={[name for name, _ in named]}; optimizer matches={optimizer_ids == trainable_ids}; AdamW hyperparameters={groups_ok}")
            current_index += 1
            loss_ok = bool(observed_losses) and all(math.isfinite(value)
                and math.isclose(value, float(expected_loss), rel_tol=1e-5, abs_tol=1e-6) for value in observed_losses)
            gradient_ok = bool(observed_gradients) and all(
                bool(torch.isfinite(g).all()) and torch.allclose(g, expected_gradient.expand_as(g), rtol=1e-5, atol=1e-6)
                for g in observed_gradients)
            required_events = ["zero_grad", "backward", ("clip", float(config["gradient_clip_norm"])), "step"]
            cursor = 0
            for event in events:
                if cursor < len(required_events) and event == required_events[cursor]:
                    cursor += 1
            update_check(current_index, bool(loss_ok and gradient_ok and cursor == len(required_events)),
                         f"Actual backward losses={observed_losses}, expected={float(expected_loss)}; "
                         f"offset gradient matches={gradient_ok}; ordered update calls={events}")
            current_index += 1
            changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p.detach())]
            update_check(current_index, changed == ["energy_offset"], f"Changed parameter names={changed}")
            current_index += 1
            base_ok = len(base_before) == config["base_param_count"] and all(
                not p.requires_grad and torch.equal(base_before[name], p.detach()) for name, p in base.named_parameters())
            update_check(current_index, bool(base_ok), f"Frozen base unchanged={base_ok}")
            current_index += 1
            results_dir.mkdir(parents=True, exist_ok=True)
            for filename in config["expected_outputs"]:
                (results_dir / filename).unlink(missing_ok=True)
            # 沿用旧评测的第二次单步调用，检查完整 train 入口产物；不代替候选训练。
            if "config_path" in inspect.signature(train_module.train).parameters:
                train_module.train(str(CANDIDATE_DIR / config_name), str(results_dir), "cpu")
            else:
                output_config = deepcopy(runtime_config)
                output_config["output_dir"] = str(results_dir)
                train_module.train(output_config)
            update_check(current_index, True, f"Candidate train entry completed: {results_dir}")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    finally:
        for handle in handles:
            handle.remove()

    return checks




def check_dataset_pipeline(config: dict, caller_globals=None, caller_locals=None):
    """真实固定数据的既有检查深度；不把参考/候选双 loader 方法改成另一套协议。"""
    CHECKPOINT_NAMES = ["data:module_load", "data:dataset_process", "data:dmc_energy_force_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        module = _load_candidate_modules(config["model_file"])
        update_check(current_index, True, "Candidate data interfaces loaded")
        current_index += 1
        path = Path(config["data_path"])
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != config["data_sha256"]:
            raise ValueError(f"Fixed dataset SHA256 mismatch: {actual_hash}")
        structures, batch = build_tace_batch(module, config)
        update_check(current_index, True, "Candidate real-data loader and batch builder executed")
        current_index += 1
        data = batch["data"] if "data" in batch else batch
        shapes = {"energy": list(batch["energy"].shape), "forces": list(batch["forces"].shape),
                  "positions": list(data["positions"].shape)}
        shapes_ok = (shapes["energy"] in config.get("energy_shapes", [config["shapes"]["energy"]])
                     and all(shapes[key] == config["shapes"][key] for key in ("forces", "positions")))
        ok = (len(structures) > config["frame_index"] and shapes_ok
              and abs(float(batch["energy"].item()) - config["frame_energy"]) <= config["energy_atol"]
              and bool(torch.isfinite(batch["forces"]).all()))
        update_check(current_index, bool(ok), f"DMC hash={actual_hash}; shapes={shapes}; energy={float(batch['energy'].item())}; periodic graph and complete force-label equality not independently proved")
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_source_contract(config: dict, caller_globals=None, caller_locals=None):
    """本地包和禁止依赖的源码证据；不把任意 AST 标记当作运行时加载证明。"""
    CHECKPOINT_NAMES = ["source:portable_package", "source:prohibited_imports"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        package = CANDIDATE_DIR / config["package_path"]
        update_check(current_index, (package / "__init__.py").is_file(), f"Local package exists={package.is_dir()}; empty __init__.py is allowed")
        current_index += 1
        prohibited = []
        for path in (CANDIDATE_DIR / config["source_dir"]).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
                prohibited.extend(f"{path.relative_to(CANDIDATE_DIR)}:{node.lineno}:{name}" for name in names if name.split(".")[0] in config["prohibited_imports"])
        update_check(current_index, not prohibited, f"Prohibited absolute-import declarations={prohibited}; includes portable package; static evidence only")
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks


def check_adapter_checkpoint(config: dict, caller_globals=None, caller_locals=None):
    """检查真实 train 入口产出的 checkpoint；不要求 Prompt 未规定的 format/model_name。"""
    CHECKPOINT_NAMES = ["artifacts:checkpoint_load", "artifacts:checkpoint_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = CANDIDATE_DIR / config["checkpoint_path"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        update_check(current_index, True, f"Generated checkpoint safely read: {path}")
        current_index += 1
        required = {"model", "optimizer", "completed_steps", "trainable_parameter_names", "source_checkpoint_sha256", "config", "history"}
        # 两种已核对的序列化键；只建立本地检查视图，不改候选文件或 checkpoint。
        if isinstance(payload, dict):
            payload = dict(payload)
            for key, alias in (("model", "model_state_dict"), ("optimizer", "optimizer_state_dict")):
                if key not in payload and alias in payload:
                    payload[key] = payload[alias]
            if "training_config" in payload:
                payload["config"] = payload["training_config"]
        ok = (isinstance(payload, dict) and required.issubset(payload)
              and payload["completed_steps"] == 1
              and payload["trainable_parameter_names"] == ["energy_offset"]
              and payload["source_checkpoint_sha256"] == config["checkpoint_sha256"]
              and isinstance(payload["model"], dict) and bool(payload["model"])
              and isinstance(payload["optimizer"], dict) and bool(payload["optimizer"].get("state"))
              and isinstance(payload["config"], dict) and bool(payload["config"])
              and isinstance(payload["history"], list) and len(payload["history"]) == 1)
        if ok:
            module = _load_candidate_modules(config["model_file"])
            model, _ = load_tace_adapter(module, config, config["source_checkpoint"])
            initial = model.state_dict()
            saved = payload["model"]
            same_keys = set(initial) == set(saved)
            unchanged = same_keys and all(torch.is_tensor(saved[key]) and saved[key].shape == value.shape
                and (bool(torch.isfinite(saved[key]).all()) if key == "energy_offset" else torch.equal(saved[key].cpu(), value.cpu()))
                for key, value in initial.items())
            ok = ok and unchanged
        update_check(current_index, bool(ok), f"Checkpoint required fields={sorted(required)}; present={sorted(payload) if isinstance(payload, dict) else type(payload).__name__}; full model keys/shapes and frozen values compared")
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks
