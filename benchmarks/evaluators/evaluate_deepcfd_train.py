"""evaluate_deepcfd_train：已核对 Agent 接口接入与误判修正（r5）。

保留 Base 方法主体/普通分支；公共运行底稿为 Dev e5234af，Base SHA256 53f95abf。
本轮只改本 Case JSON/evaluator，不修改 Prompt、Standard、Agent、公共 Base/checker/Runner。
接入范围为已核对的 Standard 与原始 Agent；未识别接口明确报错，不代替候选实现。
验证为本地小数据；正式 SCNet 成绩由运行包记录。
"""
from __future__ import annotations

import ast  # noqa: F401  # 复制体内部静态检查使用
import inspect  # noqa: F401
import json  # noqa: F401
import os as _os
import pickle  # noqa: F401
import textwrap  # noqa: F401
import traceback  # noqa: F401
from copy import deepcopy  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Tuple  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import torch  # noqa: F401
import yaml  # noqa: F401


from base.base_evaluator import (  # noqa: E402
    CANDIDATE_DIR,
    DEVICE,
    FORWARD_COMPARE_ATOL,
    REFERENCE_DIR,
    TypeInferencer,
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
    check_interface,
    check_parameter_update,
)
import math
from typing import Mapping  # noqa: F401
from torch import ones  # noqa: E402,F401  # 前置钩子 function: "ones" 经 globals() 解析

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
        if config.get("config_paths"):
            config_data = bind_config_paths(config_data, config["config_paths"])

        # ------------------ 2. 检查必需字段 ------------------
        for spec in normalized_specs:
            key = spec["key"]
            expected_type = spec["type"]
            expected_value = spec["value"]

            # 2.0 支持点号嵌套键（如 training.lr）：逐段导航取值
            _missing = object()
            _current_value = config_data
            for _segment in key.split("."):
                _current_value = (
                    _current_value.get(_segment, _missing)
                    if isinstance(_current_value, dict)
                    else _missing
                )
                if _current_value is _missing:
                    break
            key_found = _current_value is not _missing

            # 2.1 检查键是否存在
            if not key_found:
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

            current_value = _current_value

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
    if isinstance(config.get("rules"), list):
        CHECKPOINT_NAMES += [f"model_structure:{rule.get('id', rule.get('type'))}" for rule in config["rules"]]
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
        if config.get("config_paths"):
            candidate_config = bind_config_paths(candidate_config, config["config_paths"])
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)

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
        checker.add_rules_from_rules(rules)

        # 提取结构信息
        extractor = ModuleInfoExtractor(model)
        root_info = extractor.extract()

        # 执行检查：构建成功后才按真实模块展开规则，准备失败保留已登记项。
        checks[:] = checks[:1]
        CHECKPOINT_NAMES[:] = CHECKPOINT_NAMES[:1]
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
                CHECKPOINT_NAMES.append(f"model_structure:{rule.id}")
                checks.append((CHECKPOINT_NAMES[-1], passed, message))
                continue

            for module_info in modules_to_check:
                passed, msg = checker.evaluate_rule(rule, module_info)

                # 生成检查点名称
                if module_info.module_path:
                    name = f"model_structure:{rule.id}:{module_info.module_path}"
                else:
                    name = f"model_structure:{rule.id}"
                CHECKPOINT_NAMES.append(name)
                checks.append((name, passed, msg))

    except Exception as e:
        error_detail = traceback.format_exc()
        if not checks[0][1]:
            update_check(0, False, error_detail)
        else:
            CHECKPOINT_NAMES.append("model_structure:structure_execution")
            checks.append((CHECKPOINT_NAMES[-1], False, error_detail))
            if isinstance(config.get("rules"), list):
                for specification in config["rules"]:
                    pending = f"model_structure:{specification.get('id', specification.get('type'))}"
                    if not any(name == pending or name.startswith(pending + ":") for name, _, _ in checks):
                        CHECKPOINT_NAMES.append(pending)
                        checks.append((pending, False, "Not executed due to earlier failure"))

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
    if config.get("check_nan_inf"):
        CHECKPOINT_NAMES.append("forward_executable:finite_output")
    if config.get("check_p4_equivariance"):
        CHECKPOINT_NAMES.append("forward_executable:p4_equivariance")
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
        if config.get("config_paths"):
            model_config = bind_config_paths(model_config, config["config_paths"])
        context["config"] = model_config
        if isinstance(model_config, dict):
            context.update(model_config)

        # ------------------ 3. 执行前置钩子（如果有） ------------------
        pre_hooks = config.get("pre_forward_hooks", [])
        context = _execute_hooks(model_module, pre_hooks, context, globals(), context)

        # ------------------ 4. 构建模型 ------------------
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

        # 检查是否为 Tensor
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Output is not a Tensor: {type(out)}")

        same_shape = (list(out.shape) == config.get("out_shape", []))
        update_check(
            current_index,
            same_shape,
            f"Output is Tensor with shape={tuple(out.shape)}, expected out_dim={config.get('out_shape')}"
        )

        if config.get("check_nan_inf"):
            current_index = CHECKPOINT_NAMES.index("forward_executable:finite_output")
            update_check(current_index, bool(torch.isfinite(out).all()), "Output has only finite values")

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
        if is_candidate:
            module = _load_candidate_modules(model_file)
            model_config = _load_candidate_config(config_name)
        else:
            module = _load_reference_modules(model_file)
            model_config = _load_reference_config(config_name)

        # 将配置添加到上下文中
        if impl_config.get("config_paths"):
            model_config = bind_config_paths(model_config, impl_config["config_paths"])
        context["config"] = model_config
        if isinstance(model_config, dict):
            context.update(model_config)


        # 执行前置钩子
        pre_hooks = impl_config.get("pre_forward_hooks", [])
        context = _execute_hooks(module, pre_hooks, context, globals(), context)

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


def check_train_dataset_loading(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    静态检查训练脚本中的数据加载逻辑。

    config 示例:
    {
        "files": ["train.py", "trainer.py"],
        "dataset_names": ["Dataset", "MyDataset"],  # 可选
        "dataloader_names": ["DataLoader", "GraphDataLoader"]  # 可选
    }

    检查点（根据配置动态生成）:
        - data_loading:dataset_call: 检查是否有 Dataset 调用
        - data_loading:dataloader_call: 检查是否有 DataLoader 调用
        - data_loading:dataset_import: 检查是否有 Dataset 导入
        - data_loading:dataloader_import: 检查是否有 DataLoader 导入

    注意: dataset_names 和 dataloader_names 至少需要提供一个
    """
    # ================================================================
    # 0. 获取配置并验证
    # ================================================================
    dataset_names = set(config.get("dataset_names", []))
    dataloader_names = set(config.get("dataloader_names", []))

    # 验证至少有一个配置
    if not dataset_names and not dataloader_names:
        return [("config:validation", False, "Neither dataset_names nor dataloader_names provided")]

    # 确定需要检查的项目
    check_dataset = bool(dataset_names)
    check_dataloader = bool(dataloader_names)

    # ================================================================
    # 1. 动态生成检查点名称
    # ================================================================
    CHECKPOINT_NAMES = []
    if check_dataset:
        CHECKPOINT_NAMES.extend([
            "data_loading:dataset_call",
            "data_loading:dataset_import",
        ])
    if check_dataloader:
        CHECKPOINT_NAMES.extend([
            "data_loading:dataloader_call",
            "data_loading:dataloader_import",
        ])

    # 初始化检查结果
    checks = [(name, False, "Not executed") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
      checks[index] = (checks[index][0], passed, msg)

    try:
        # ------------------ 2. 获取文件配置 ------------------
        train_files = config.get("files", [])
        if isinstance(train_files, str):
            train_files = [train_files]

        if not train_files:
            raise ValueError("config['files'] is missing or empty")

        # 初始化检查结果
        has_dataset_call = False
        has_dataloader_call = False
        has_dataset_import = False
        has_dataloader_import = False

        # 构建名称到索引的映射
        name_to_index = {name: idx for idx, (name, _, _) in enumerate(checks)}

        # ------------------ 3. 扫描所有训练文件 ------------------
        for train_file in train_files:
            # 使用传入的 caller_globals/locals 或默认路径
            # 注意: 这里假设 CANDIDATE_DIR 在外部定义，或需要传入
            train_path = Path(train_file)
            if not train_path.is_absolute():
                candidate_path = (CANDIDATE_DIR / train_file).resolve()
                if candidate_path.exists() and candidate_path.is_file():
                    train_path = candidate_path
            if not train_path.exists() or not train_path.is_file():
                continue

            tree = ast.parse(
                train_path.read_text(encoding="utf-8"),
                filename=str(train_path),
            )

            # 检查调用
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = None
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr

                    if check_dataset and func_name in dataset_names:
                        has_dataset_call = True
                    if check_dataloader and func_name in dataloader_names:
                        has_dataloader_call = True

                # 检查导入语句
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if check_dataset and alias.name in dataset_names:
                            has_dataset_import = True
                        if check_dataloader and alias.name in dataloader_names:
                            has_dataloader_import = True

                # 检查导入的模块属性访问（如 torch.utils.data.DataLoader）
                if isinstance(node, ast.Attribute):
                    attr_name = node.attr
                    if check_dataset and attr_name in dataset_names:
                        has_dataset_import = True
                    if check_dataloader and attr_name in dataloader_names:
                        has_dataloader_import = True

                # 提前退出条件
                all_checked = True
                if check_dataset:
                    all_checked = all_checked and has_dataset_call and has_dataset_import
                if check_dataloader:
                    all_checked = all_checked and has_dataloader_call and has_dataloader_import

                if all_checked:
                    break

            # 如果所有检查项都已满足，提前结束文件扫描
            all_checked = True
            if check_dataset:
                all_checked = all_checked and has_dataset_call and has_dataset_import
            if check_dataloader:
                all_checked = all_checked and has_dataloader_call and has_dataloader_import

            if all_checked:
                break

        # ------------------ 4. 更新检查结果 ------------------
        # 更新 Dataset 检查
        if check_dataset:
            dataset_call_idx = name_to_index.get("data_loading:dataset_call", -1)
            dataset_import_idx = name_to_index.get("data_loading:dataset_import", -1)

            if dataset_call_idx >= 0:
                update_check(
                    dataset_call_idx,
                    has_dataset_call,
                    f"Dataset call found: {has_dataset_call}, names: {list(dataset_names)}"
                )
            if dataset_import_idx >= 0:
                update_check(
                    dataset_import_idx,
                    has_dataset_import,
                    f"Dataset import found: {has_dataset_import}, names: {list(dataset_names)}"
                )

        # 更新 DataLoader 检查
        if check_dataloader:
            dataloader_call_idx = name_to_index.get("data_loading:dataloader_call", -1)
            dataloader_import_idx = name_to_index.get("data_loading:dataloader_import", -1)

            if dataloader_call_idx >= 0:
                update_check(
                    dataloader_call_idx,
                    has_dataloader_call,
                    f"DataLoader call found: {has_dataloader_call}, names: {list(dataloader_names)}"
                )
            if dataloader_import_idx >= 0:
                update_check(
                    dataloader_import_idx,
                    has_dataloader_import,
                    f"DataLoader import found: {has_dataloader_import}, names: {list(dataloader_names)}"
                )

    except Exception as e:
        error_detail = traceback.format_exc()
        # 更新所有未完成的检查点
        for i in range(len(checks)):
            if not checks[i][1]:  # 如果尚未通过
                update_check(i, False, error_detail)

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
    if config.get("observe_training"):
        CHECKPOINT_NAMES += ["training:real_backward", "training:finite_gradients", "training:parameter_change", "training:fresh_artifacts"]
    if config.get("task_contract"):
        CHECKPOINT_NAMES.append("training:task_loss_rollout")
    if config.get("expected_optimizer"):
        CHECKPOINT_NAMES.append("training:optimizer_configuration")
    if config.get("expected_scheduler"):
        CHECKPOINT_NAMES.append("training:scheduler_configuration")
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

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

        # ------------------ 4. 加载配置（本 case 配置内嵌代码，config_name 可选） ------------------
        config_name = config.get("config_name")

        # 加载完整配置（与 check_training 一致）
        candidate_config = _load_candidate_config(config_name) if config_name else {}
        runtime_config = deepcopy(candidate_config)

        # ------------------ 5. 构造冒烟配置（轻量覆盖） ------------------
        current_index += 1

        # 默认冒烟配置（epochs=1, batch_size=8 等轻量参数）
        smoke_config = config.get("smoke_config", {})
        context = {"checkpoint_dir": checkpoint_dir, "results_dir": results_dir,
                   "probe_dir": work_dir / "data", "smoke_config_path": work_dir / "smoke_config.yaml"}
        if config.get("synthetic_data"):
            preparation = globals()[config["synthetic_data"]["function"]]
            preparation(context["probe_dir"], **config["synthetic_data"].get("args", {}))
        smoke_config = _resolve_args(smoke_config, context)

        def _deep_merge(base_cfg, override_cfg):
            merged = deepcopy(base_cfg)
            for _merge_key, _merge_value in (override_cfg or {}).items():
                if config.get("observe_training") and _merge_key not in merged:
                    raise ValueError(f"REVIEW_REQUIRED: bind the existing smoke override key {_merge_key!r}; do not create ignored configuration fields")
                if isinstance(_merge_value, dict) and isinstance(merged.get(_merge_key), dict):
                    merged[_merge_key] = _deep_merge(merged[_merge_key], _merge_value)
                else:
                    merged[_merge_key] = _merge_value
            return merged

        if "paths" in candidate_config and "max_epochs" in candidate_config.get("training", {}):
            data_dir = context["probe_dir"]
            results = context["results_dir"]
            smoke_config = {"device": "cpu", "data": {
                "input_path": str(data_dir / "dataX.pkl"), "target_path": str(data_dir / "dataY.pkl"),
                "expected_samples": config["synthetic_data"]["args"]["count"]},
                "training": {"max_epochs": 1, "eval_interval": 1},
                "paths": {name: str(results / ("inference/metrics.json" if name == "metrics" else
                    "best_model.pt" if name == "best_model" else Path(value).name))
                    for name, value in candidate_config["paths"].items()}}
        runtime_config = _deep_merge(runtime_config, smoke_config)
        if config_name and config.get("cli_mode"):
            with context["smoke_config_path"].open("w", encoding="utf-8") as handle:
                yaml.safe_dump(runtime_config, handle, allow_unicode=True)
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

        # ------------------ 7. 执行冒烟测试 ------------------
        # 解析训练函数参数（与 check_training 一致）
        train_args_config = config.get("train_args", {})
        resolved_train_args = _resolve_args(train_args_config, context)


        if config.get("task_contract"):
            task_index = CHECKPOINT_NAMES.index("training:task_loss_rollout")
            try:
                detail = probe_training_contract(train_module, config["task_contract"])
                update_check(task_index, True, detail)
            except Exception:
                update_check(task_index, False, traceback.format_exc())
        observation = {"backward": 0, "steps": 0, "finite_gradients": True, "changed": False,
                       "optimizer_ok": True, "schedulers": []}
        observed_paths = [CANDIDATE_DIR / name for name in config.get("fresh_artifacts", [])]
        if config.get("observe_training"):
            stale = [str(path) for path in observed_paths if path.exists()]
            if stale:
                raise ValueError(f"OUTPUT_ALREADY_EXISTS: evaluate a fresh artifact snapshot; do not delete original files: {stale}")
        from contextlib import ExitStack
        from unittest.mock import patch
        original_backward = torch.autograd.backward
        optimizer_class = getattr(torch.optim, config.get("optimizer_class", "Adam"))
        original_step = optimizer_class.step

        def observed_backward(*args, **kwargs):
            result = original_backward(*args, **kwargs)
            observation["backward"] += 1
            return result

        def observed_step(optimizer, *args, **kwargs):
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            before = [p.detach().clone() for p in parameters]
            for key, expected in config.get("expected_optimizer", {}).items():
                actual = optimizer.defaults.get(key)
                observation["optimizer_ok"] &= isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
            result = original_step(optimizer, *args, **kwargs)
            gradients = [p.grad for p in parameters if p.grad is not None]
            observation["finite_gradients"] &= bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
            observation["changed"] |= any(not torch.equal(a, p.detach()) for a, p in zip(before, parameters))
            observation["steps"] += 1
            return result

        scheduler_spec = config.get("expected_scheduler")
        if scheduler_spec:
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_spec["class"])
            original_scheduler_init = scheduler_class.__init__
            def observed_scheduler_init(instance, *args, **kwargs):
                original_scheduler_init(instance, *args, **kwargs)
                observation["schedulers"].append(all(getattr(instance, key, None) == value
                    for key, value in scheduler_spec.items() if key != "class"))

        # 执行训练：只观察候选真实调用，不替候选做 backward 或 optimizer.step。
        with ExitStack() as observation_stack:
            if scheduler_spec:
                observation_stack.enter_context(patch.object(scheduler_class, "__init__", observed_scheduler_init))
            if config.get("observe_training"):
                observation_stack.enter_context(patch.object(torch.autograd, "backward", observed_backward))
                observation_stack.enter_context(patch.object(optimizer_class, "step", observed_step))
            if config.get("cli_mode"):
                # CLI 入口（prompt 声明的运行方式）：切到候选目录、补丁 sys.argv 后调用无参 main()。
                # 冒烟参数两种传递方式：
                #  - cli_args：候选脚本原生 argparse 参数（本 case 的 train.py），无配置文件；
                #  - 否则：冒烟配置写临时 yaml，以 --config 传入（deepcfd 型入口）。
                import os as _cli_os
                import sys as _cli_sys

                _old_argv = _cli_sys.argv
                _old_cwd = _cli_os.getcwd()
                try:
                    _cli_os.chdir(CANDIDATE_DIR)
                    _cli_args = config.get("cli_args")
                    if _cli_args is not None:
                        _cli_sys.argv = [str(train_file)] + [str(_a) for _a in _resolve_args({"args": _cli_args}, context)["args"]]
                    else:
                        import yaml as _cli_yaml

                        smoke_yaml_path = work_dir / "smoke_config.yaml"
                        with open(smoke_yaml_path, "w", encoding="utf-8") as _cli_f:
                            _cli_yaml.safe_dump(runtime_config, _cli_f, allow_unicode=True)
                        _cli_sys.argv = [str(train_file), "--config", str(smoke_yaml_path)]
                    result = train_fn()
                finally:
                    _cli_sys.argv = _old_argv
                    _cli_os.chdir(_old_cwd)
            else:
                result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)
        update_check(
            current_index,
            True,
            f"Smoke test completed successfully: {train_fn_name}(...) with {config_name}"
        )


        if config.get("expected_optimizer"):
            update_check(CHECKPOINT_NAMES.index("training:optimizer_configuration"),
                         observation["steps"] > 0 and observation["optimizer_ok"],
                         f"Actual optimizer defaults match={observation['optimizer_ok']}")
        if scheduler_spec:
            update_check(CHECKPOINT_NAMES.index("training:scheduler_configuration"),
                         bool(observation["schedulers"]) and all(observation["schedulers"]),
                         f"Actual scheduler declarations={observation['schedulers']}")
        if config.get("observe_training"):
            update_check(CHECKPOINT_NAMES.index("training:real_backward"), observation["backward"] > 0,
                         f"Observed successful candidate backward calls={observation['backward']}")
            update_check(CHECKPOINT_NAMES.index("training:finite_gradients"),
                         observation["steps"] > 0 and observation["finite_gradients"],
                         f"Candidate optimizer steps={observation['steps']}; finite nonempty gradients={observation['finite_gradients']}")
            update_check(CHECKPOINT_NAMES.index("training:parameter_change"), observation["changed"],
                         f"Optimized parameter changed={observation['changed']}")
            fresh = bool(observed_paths) and all(path.is_file() and path.stat().st_size > 0 for path in observed_paths)
            update_check(CHECKPOINT_NAMES.index("training:fresh_artifacts"), fresh,
                         f"Fresh nonempty output files={fresh}; paths={[str(p) for p in observed_paths]}")

    except (Exception, SystemExit) as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_infer_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    动态执行一次推理流程，检查推理代码是否能够正常运行。

    config 示例:
    {
        "run_file": "inference.py",
        "config_name": "config.yaml",
        "infer_function": "run_inference",
        "pre_infer_hooks": [
            {
                "function": "build_model",
                "args": {"model_name": "model_name", "device": "device"},
                "save_as": "model"
            }
        ],
        "infer_args": {
            "model": "model",
            "data_path": "data_path",
            "output_dir": "output_dir"
        },
        "output_path": "outputs/result.json"
    }

    检查点:
        - inference_execution:module_load: 检查能否加载推理模块
        - inference_execution:pre_hooks: 检查前置钩子是否执行成功
        - inference_execution:infer_call: 检查推理调用是否成功
        - inference_execution:output_save: 检查输出是否保存成功
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "inference_execution:module_load",
        "inference_execution:pre_hooks",
        "inference_execution:infer_call",
        "inference_execution:output_save",
    ]
    if config.get("metric_formula_probe"):
        CHECKPOINT_NAMES.append("inference_execution:metric_formula")
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    def _json_safe(value: Any) -> Any:
        """Convert inference results to JSON without losing array dtype metadata."""
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            return {
                "data": tensor.tolist(),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
        if isinstance(value, np.ndarray):
            return {
                "data": value.tolist(),
                "dtype": str(value.dtype),
            }
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Preserve the previous serializer's behavior for miscellaneous
        # metadata while keeping tensors/arrays lossless and typed above.
        return str(value)

    try:
        # ------------------ 1. 加载推理模块 ------------------
        infer_file = config.get("run_file")
        if not infer_file:
            raise ValueError("config['run_file'] is missing or empty")

        infer_module = _load_candidate_modules(infer_file)
        update_check(current_index, True, f"Module loaded from {infer_file}")

        # ------------------ 2. 加载配置 ------------------
        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("config['config_name'] is missing or empty")

        candidate_config = _load_candidate_config(config_name)
        if config.get("runtime_config_name"):
            candidate_config = _load_candidate_config(config["runtime_config_name"])
        context = {"config": candidate_config}
        context.update(candidate_config)

        # ------------------ 3. 获取推理函数 ------------------
        infer_func_name = config.get("infer_function", "inference")
        infer_func = getattr(infer_module, infer_func_name, None)
        if infer_func is None or not callable(infer_func):
            raise ValueError(f"No callable inference entry '{infer_func_name}' found")

        # ------------------ 4. 执行前置钩子 ------------------
        current_index += 1
        pre_hooks = config.get("pre_infer_hooks", [])
        for idx, hook_config in enumerate(pre_hooks):
            func_name = hook_config.get("function")
            args_config = hook_config.get("args", {})

            hook_func = getattr(infer_module, func_name, None)
            if hook_func is None or not callable(hook_func):
                raise ValueError(f"Hook {idx}: function '{func_name}' not found or not callable")

            resolved_args = _resolve_args(args_config, context)
            hook_result = hook_func(**resolved_args)

            save_as = hook_config.get("save_as")
            if save_as:
                context[save_as] = hook_result

        update_check(current_index, True, f"All {len(pre_hooks)} pre-hooks executed")

        # ------------------ 5. 执行推理 ------------------
        current_index += 1
        infer_args_config = config.get("infer_args", {})
        resolved_infer_args = _resolve_args(infer_args_config, context)

        if config.get("cli_mode"):
            # CLI 入口（如 python inference.py --config xxx）：切到候选目录、
            # 补丁 sys.argv 后调用无参 main()，与真实命令行行为一致。
            import os as _cli_os
            import sys as _cli_sys

            _old_argv = _cli_sys.argv
            _old_cwd = _cli_os.getcwd()
            try:
                _cli_os.chdir(CANDIDATE_DIR)
                _cli_sys.argv = [str(infer_file), "--config", str(CANDIDATE_DIR / config.get("runtime_config_name", config_name))]
                result = infer_func()
            finally:
                _cli_sys.argv = _old_argv
                _cli_os.chdir(_old_cwd)
        else:
            result = infer_func(**resolved_infer_args)
        update_check(current_index, True, "Inference executed successfully")

        # ------------------ 6. 保存输出 ------------------
        current_index += 1
        output_path = config.get("output_path")
        if output_path:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = CANDIDATE_DIR / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == '.json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(_json_safe(result), f, ensure_ascii=False, indent=2)
            elif output_path.suffix == '.pkl':
                with open(output_path, 'wb') as f:
                    pickle.dump(result, f)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(_json_safe(result), f, ensure_ascii=False, indent=2)

            update_check(current_index, True, f"Output saved to {output_path}")
        else:
            update_check(current_index, True, "No output_path configured, skipping save")


        if config.get("metric_formula_probe"):
            current_index = CHECKPOINT_NAMES.index("inference_execution:metric_formula")
            metric_fn = getattr(infer_module, "compute_sample_normalized_grid_mse", None)
            if not callable(metric_fn):
                raise ValueError("Prompt-required compute_sample_normalized_grid_mse is missing")
            predictions = torch.arange(2 * 3 * 172 * 79, dtype=torch.float32).reshape(2, 3, 172, 79) / 8192
            targets = predictions.flip(-1) * 0.3 - 0.4
            squared = (predictions - targets).square().sum(dim=(0, 2, 3)) / predictions.shape[0]
            expected = dict(zip(("Ux_MSE", "Uy_MSE", "p_MSE"), squared.tolist()))
            expected["Total_MSE"] = sum(expected.values())
            actual = metric_fn(predictions, targets)
            if not isinstance(actual, dict):
                raise TypeError("metric function must return the four named Prompt metrics")
            for key, value in expected.items():
                item = actual.get(key)
                if torch.is_tensor(item) and item.numel() == 1:
                    item = item.detach().item()
                if isinstance(item, bool) or not isinstance(item, (int, float, np.number)):
                    raise ValueError(f"{key} must be a numeric scalar, got {item!r}")
                if not math.isfinite(float(item)) or not math.isclose(float(item), value, rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError(f"{key}: expected sample-normalized grid sum {value}, got {item}")
            update_check(current_index, True, "Candidate metric matches grid SSE / sample count, per channel and total")

    except (Exception, SystemExit) as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_infer_output_schema(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple[str, bool, str]]:
    """
    检查推理输出是否存在且有效。

    config 示例:
    {
        "output_path": "outputs/result.json",
        "output_shape": [10, 5],
        "check_nan_inf": true,
        "expected_dtype": "float32"
    }

    检查点:
        - inference_output:file_exists: 检查输出文件是否存在
        - inference_output:file_load: 检查能否加载输出文件
        - inference_output:not_empty: 检查输出是否为空
        - inference_output:finite_check: 检查是否包含 NaN/Inf
        - inference_output:dtype_check: 检查数据类型（如果配置）
        - inference_output:shape_check: 检查输出形状（如果配置）
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "inference_output:file_exists",
        "inference_output:file_load",
        "inference_output:not_empty",
        "inference_output:finite_check",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    # 动态添加 dtype 和 shape 检查点
    if "expected_dtype" in config:
        CHECKPOINT_NAMES.append("inference_output:dtype_check")
        checks.append(("inference_output:dtype_check", False, "Not executed"))
    if "output_shape" in config:
        CHECKPOINT_NAMES.append("inference_output:shape_check")
        checks.append(("inference_output:shape_check", False, "Not executed"))

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    metric_checks = []
    if config.get("required_keys") or config.get("finite_metric_keys"):
        metric_checks = [("inference_output:metric_values", False, "Not executed due to earlier failure")]

    try:
        # ------------------ 1. 检查文件是否存在 ------------------
        output_path = config.get("output_path")
        if not output_path:
            raise ValueError("output_path is required in config")

        file_path = Path(output_path)
        if not file_path.is_absolute():
            file_path = CANDIDATE_DIR / file_path

        configured_file_path = file_path
        if not file_path.exists() and file_path.suffix.lower() == ".json":
            sibling = file_path.with_suffix(".pkl")
            if sibling.exists():
                file_path = sibling

        if not file_path.exists():
            raise FileNotFoundError(f"Output file not found: {configured_file_path}")
        if file_path != configured_file_path:
            update_check(
                current_index,
                True,
                f"Configured output missing: {configured_file_path}; using sibling {file_path}",
            )
        else:
            update_check(current_index, True, f"File exists: {file_path}")

        # ------------------ 2. 加载文件 ------------------
        current_index += 1
        suffix = file_path.suffix.lower()
        loaded_from = file_path
        if suffix == ".json":
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            # Older runs used ``default=str`` and left a truncated
            # ``tensor(...)`` representation in JSON.  If the same run has
            # produced a newer pickle beside it, prefer that lossless result.
            # A tensor repr cannot reliably recover shape or dtype from JSON.
            if isinstance(result, str) and result.lstrip().startswith("tensor("):
                sibling = file_path.with_suffix(".pkl")
                if sibling.exists() and sibling.stat().st_mtime >= file_path.stat().st_mtime:
                    try:
                        with open(sibling, "rb") as f:
                            result = pickle.load(f)
                        loaded_from = sibling
                    except (OSError, pickle.PickleError, EOFError, RuntimeError):
                        # Keep the JSON result so the schema check below can
                        # report the unsupported representation clearly.
                        pass
        elif suffix == ".csv":
            result = pd.read_csv(file_path)
        elif suffix == ".pkl":
            with open(file_path, "rb") as f:
                result = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")
        if loaded_from != file_path:
            load_msg = f"File loaded: {suffix}; used newer sibling {loaded_from.name}"
        else:
            load_msg = f"File loaded: {suffix}"
        update_check(current_index, True, load_msg)

        # ------------------ 3. 检查是否为空 ------------------
        current_index += 1
        if result is None:
            raise ValueError("Inference output is None")

        if isinstance(result, (list, tuple, dict)):
            if len(result) == 0:
                raise ValueError("Inference output is empty")

        try:
            if isinstance(result, pd.DataFrame) and result.empty:
                raise ValueError("Inference output is empty DataFrame")
        except ImportError:
            pass

        update_check(
            current_index,
            True,
            f"Output is not empty, type={type(result).__name__}"
        )

        if metric_checks:
            try:
                required_keys = config.get("required_keys", [])
                if not isinstance(result, dict):
                    raise TypeError("Named metrics must be stored as a JSON object")
                metric_result = bind_config_paths(result, config.get("metric_paths", {}))
                missing_keys = [key for key in required_keys if key not in metric_result]
                if missing_keys:
                    raise ValueError(f"Output missing required keys: {missing_keys}")
                for metric_key in config.get("finite_metric_keys", []):
                    metric_value = metric_result[metric_key]
                    if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)) or not math.isfinite(metric_value) or metric_value < 0:
                        raise ValueError(f"Metric {metric_key} must be a finite nonnegative number, got {metric_value!r}")
                if config.get("metric_total_check") and not math.isclose(
                        metric_result["Total_MSE"], sum(metric_result[key] for key in ("Ux_MSE", "Uy_MSE", "p_MSE")), rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError("Total_MSE must equal Ux_MSE + Uy_MSE + p_MSE")
                metric_checks[0] = (metric_checks[0][0], True, "Required finite numeric metrics and channel total are valid")
            except Exception:
                metric_checks[0] = (metric_checks[0][0], False, traceback.format_exc())

        def _collect_array_info(data: Any) -> List[dict]:
            arrays_info = []

            def _dtype_name(dtype: Any) -> str:
                """Return a comparable dtype name for NumPy and Torch values."""
                name = str(dtype)
                if name.startswith("torch."):
                    name = name[len("torch."):]
                return name

            def _json_array_info(value: dict) -> Optional[dict]:
                """Read an explicitly typed array payload from JSON.

                JSON numbers do not preserve the source dtype.  Producers that
                need dtype validation can therefore write ``{"data": [...],
                "dtype": "float32"}`` (``values``/``array`` are also accepted).
                The payload is still converted to a NumPy array so its shape is
                validated from the actual values rather than from metadata.
                """
                dtype = value.get("dtype")
                if dtype is None:
                    return None

                payload_keys = [
                    key for key in (
                        "data", "values", "array", "tensor", "output", "prediction", "result"
                    )
                    if key in value
                ]
                if len(payload_keys) != 1:
                    return None

                dtype_name = _dtype_name(dtype).strip()
                if dtype_name.startswith("torch."):
                    dtype_name = dtype_name[len("torch."):]
                try:
                    # The dtype metadata is authoritative: JSON numbers may
                    # have been produced from float32/bfloat16 values but are
                    # parsed as Python numbers.  Use NumPy only to validate
                    # the payload and obtain its shape, then retain the
                    # declared dtype separately.
                    np_dtype = np.dtype(dtype_name)
                    normalized_dtype = _dtype_name(np_dtype)
                except (TypeError, ValueError, RuntimeError):
                    torch_dtype = getattr(torch, dtype_name, None)
                    if not isinstance(torch_dtype, torch.dtype):
                        return None
                    normalized_dtype = dtype_name
                try:
                    array = np.asarray(value[payload_keys[0]])
                except (TypeError, ValueError, RuntimeError):
                    return None
                if array.dtype == np.dtype("O") or not np.issubdtype(array.dtype, np.number):
                    return None
                return {
                    "shape": tuple(array.shape),
                    "dtype": normalized_dtype,
                }

            try:
                if isinstance(data, pd.DataFrame):
                    if not data.empty:
                        arrays_info.append({"shape": tuple(data.shape), "dtype": "DataFrame"})
                    return arrays_info

                if isinstance(data, pd.Series):
                    if not data.empty:
                        arrays_info.append({"shape": tuple(data.shape), "dtype": _dtype_name(data.dtype)})
                    return arrays_info

                # Preserve the dtype of tensors loaded from pickle/other
                # binary formats.  JSON serialization turns tensors into
                # lists (or strings), so it cannot recover this information.
                if isinstance(data, torch.Tensor):
                    arrays_info.append({
                        "shape": tuple(data.shape),
                        "dtype": _dtype_name(data.dtype),
                    })
                    return arrays_info

                if isinstance(data, np.ndarray):
                    arrays_info.append({"shape": tuple(data.shape), "dtype": _dtype_name(data.dtype)})
                    return arrays_info

                if isinstance(data, (list, tuple)):
                    # A homogeneous JSON number list is the JSON equivalent
                    # of an array.  Convert it once to obtain both shape and
                    # the dtype that is actually representable in JSON.
                    if data:
                        try:
                            array = np.asarray(data)
                            if array.dtype != np.dtype("O") and np.issubdtype(array.dtype, np.number):
                                arrays_info.append({
                                    "shape": tuple(array.shape),
                                    "dtype": _dtype_name(array.dtype),
                                })
                                return arrays_info
                        except (TypeError, ValueError, RuntimeError):
                            pass
                    for item in data:
                        arrays_info.extend(_collect_array_info(item))
                    return arrays_info

                if isinstance(data, dict):
                    typed_array = _json_array_info(data)
                    if typed_array is not None:
                        arrays_info.append(typed_array)
                        return arrays_info
                    for value in data.values():
                        arrays_info.extend(_collect_array_info(value))
                    return arrays_info

                return arrays_info

            except Exception:
                return arrays_info

        def _check_finite(data: Any) -> bool:
            try:

                # 处理 Pandas DataFrame/Series
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    numeric_data = data.select_dtypes(include=[np.number])
                    if numeric_data.empty:
                        return True
                    if numeric_data.isna().any().any():
                        return False
                    if np.isinf(numeric_data.values).any():
                        return False
                    return True

                # 处理 PyTorch Tensor
                if isinstance(data, torch.Tensor):
                    # 如果张量为空，返回 True
                    if data.numel() == 0:
                        return True

                    # 只检查数值类型的张量
                    if not torch.is_floating_point(data) and not torch.is_complex(data):
                        return True

                    # 检查 NaN 和 Inf
                    # 使用 torch.isfinite() 检查所有元素
                    # 注意：torch.isfinite() 对非数值类型会返回 False
                    try:
                        return torch.isfinite(data).all().item()
                    except Exception:
                        # 如果转换失败（如非数值类型），返回 True
                        return True

                # 处理字典
                if isinstance(data, dict):
                    for value in data.values():
                        if not _check_finite(value):
                            return False
                    return True

                # 处理列表/元组
                if isinstance(data, (list, tuple)):
                    for item in data:
                        if not _check_finite(item):
                            return False
                    return True

                # 处理 NumPy 数组
                if isinstance(data, np.ndarray):
                    if not np.issubdtype(data.dtype, np.number):
                        return True
                    return np.isfinite(data).all()

                # 处理标量数值
                if isinstance(data, (int, float)):
                    return np.isfinite(data)

                return True

            except Exception:
                return True
        # ------------------ 4. 检查 NaN/Inf ------------------
        current_index += 1
        check_nan_inf = config.get("check_nan_inf", True)
        if check_nan_inf:
            finite_ok = _check_finite(result)
            if not finite_ok:
                raise ValueError("Inference output contains NaN or Inf")
            update_check(current_index, True, "All values are finite")
        else:
            update_check(current_index, True, "NaN/Inf check skipped")

        # ------------------ 5. 收集数组信息 ------------------
        arrays_info = _collect_array_info(result)

        if isinstance(result, str) and result.lstrip().startswith("tensor("):
            raise ValueError(
                "JSON output contains a stringified torch.Tensor; rerun inference "
                "with the evaluator's typed JSON serializer or use a .pkl output"
            )

        if not arrays_info and not isinstance(result, (int, float, str, bool)):
            # 纯标量输出
            update_check(
                len(CHECKPOINT_NAMES) - (2 if "expected_dtype" in config else 1) - (
                    1 if "utput_shape" in config else 1),
                True,
                f"Scalar output: {result}"
            )
            return checks + metric_checks

        # ------------------ 6. 检查数据类型 ------------------
        if "expected_dtype" in config:
            check_idx = len(CHECKPOINT_NAMES) - (2 if "output_shape" in config else 1)
            current_index = check_idx
            expected_dtype = str(config["expected_dtype"]).lower().strip()
            if expected_dtype.startswith("torch."):
                expected_dtype = expected_dtype[len("torch."):]
            dtype_ok = False

            if arrays_info:
                for info in arrays_info:
                    dtype = str(info["dtype"]).lower()
                    if expected_dtype in dtype or dtype in expected_dtype:
                        dtype_ok = True
                        break
            else:
                dtype_ok = True

            if not dtype_ok:
                raise ValueError(f"Expected dtype={expected_dtype}, but got {[info['dtype'] for info in arrays_info]}")
            checks[check_idx] = (
                "inference_output:dtype_check",
                True,
                f"dtype check passed: expected={expected_dtype}"
            )

        # ------------------ 7. 检查形状 ------------------
        if "output_shape" in config:
            check_idx = len(CHECKPOINT_NAMES) - 1
            current_index = check_idx
            expected_shape = tuple(config["output_shape"])

            if not arrays_info:
                shape_ok = False
                shape_msg = "No array found to check shape"
            else:
                first_array = arrays_info[0]
                actual_shape = first_array["shape"]

                if len(actual_shape) != len(expected_shape):
                    shape_ok = False
                    shape_msg = f"actual_shape={actual_shape}, expected_shape={expected_shape}"
                else:
                    shape_ok = all(expected is None or actual == expected for actual, expected in
                                   zip(actual_shape, expected_shape))
                    shape_msg = f"actual_shape={actual_shape}, expected_shape={expected_shape}"

            if not shape_ok:
                raise ValueError(f"Shape mismatch: {shape_msg}")
            checks[check_idx] = (
                "inference_output:shape_check",
                True,
                shape_msg
            )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks + metric_checks

# 当前任务的数据/产物接入；普通分支保留 Base 流程。







def deterministic_tensor(size, seed=17, dtype=None, device=None):
    """前向/参考对照共用同一局部随机序列；不改变全局 RNG。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(tuple(size), generator=generator, dtype=dtype or torch.float32)
    return _move_tensors_to_device(value) if device is None else value.to(device)


def bind_config_paths(config_data, paths):
    """只接入 JSON 明示的配置路径；不填默认值、不吞歧义、不改候选配置。"""
    resolved = deepcopy(config_data)
    for target, alternatives in paths.items():
        found = []
        for path in alternatives:
            value = _resolve_args({"value": "$config." + path}, {"config": config_data})["value"]
            if value is not None:
                found.append((path, value))
        if not found:
            raise ValueError(f"REVIEW_REQUIRED: bind configuration for {target!r}; checked {alternatives}")
        if any(value != found[0][1] for _, value in found[1:]):
            raise ValueError(f"REVIEW_REQUIRED: ambiguous configuration for {target!r}: {found}")
        destination = resolved
        parts = target.split(".")
        for part in parts[:-1]:
            if part not in destination:
                destination[part] = {}
            if not isinstance(destination[part], dict):
                raise ValueError(f"REVIEW_REQUIRED: configuration path collision at {target!r}")
            destination = destination[part]
        destination[parts[-1]] = found[0][1]
    return resolved


def check_datasset_pipeline(config: dict, caller_globals=None, caller_locals=None) -> list:
    """
    检查候选 dataset_loader 与标准 dataset_loader 的全流程处理结果。

    config 示例:
    {
        "candidate": {
            "dataset_loader": "candidate_loader.py",
            "loader_name": "loader",           # 可选，默认 "loader"
            "dataset_config": {                 # 可选，默认 {}
                "data_path": "data/",
                "batch_size": 32
            }
        },
        "reference": {
            "dataset_loader": "reference_loader.py",
            "loader_name": "loader",           # 可选，默认 "loader"
            "dataset_config": {                 # 可选，默认 {}
                "data_path": "data/",
                "batch_size": 32
            }
        },
        # 全局可选参数
        "atol": 1e-5,                          # 绝对容差，默认 1e-5
        "rtol": 1e-5                           # 相对容差，默认 1e-5
    }

    检查点:
        - data:dataset_loader_pipeline: 检查候选与标准实现的处理结果是否一致
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "data:dataset_loader_pipeline:modules_load",
        "data:dataset_loader_pipeline:process",
        "data:dataset_loader_pipeline:compare",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    # ================================================================
    # 从 config 中提取参数，并设置默认值
    # ================================================================
    candidate_config = config.get("candidate", {})
    reference_config = config.get("reference", {})

    # 提取 candidate 参数
    candidate_file = candidate_config.get("dataset_loader")
    if candidate_config.get("common_module") and not (CANDIDATE_DIR / str(candidate_file)).is_file():
        candidate_file = candidate_config["common_module"]
    candidate_loader_name = candidate_config.get("loader_name", "loader")
    candidate_dataset_config = candidate_config.get("dataset_config", {})

    # 提取 reference 参数
    reference_file = reference_config.get("dataset_loader")
    reference_loader_name = reference_config.get("loader_name", "loader")
    reference_dataset_config = reference_config.get("dataset_config", {})

    # 全局参数
    atol = config.get("atol", 1e-5)
    rtol = config.get("rtol", 1e-5)

    try:
        # ------------------ 1. 加载候选和标准模块 ------------------
        if not candidate_file:
            raise ValueError("config['candidate']['dataset_loader'] is missing or empty")
        if not reference_file and not config.get("task_probe"):
            raise ValueError("config['reference']['dataset_loader'] is missing or empty")

        candidate_module = _load_candidate_modules(candidate_file)
        reference_module = _load_reference_modules(reference_file) if not config.get("task_probe") else None

        update_check(
            current_index,
            True,
            f"Candidate: {candidate_file}, Reference: {reference_file}"
        )

        # ------------------ 2. 获取 loader 函数 ------------------
        candidate_loader = getattr(candidate_module, candidate_loader_name, None)
        reference_loader = getattr(reference_module, reference_loader_name, None)

        if not config.get("task_probe") and (candidate_loader is None or not callable(candidate_loader)):
            raise AttributeError(
                f"Candidate does not have callable '{candidate_loader_name}' "
                f"in module {candidate_file}"
            )
        if not config.get("task_probe") and (reference_loader is None or not callable(reference_loader)):
            raise AttributeError(
                f"Reference does not have callable '{reference_loader_name}' "
                f"in module {reference_file}"
            )

        # ------------------ 3. 执行处理 ------------------
        current_index += 1

        if config.get("task_probe"):
            candidate_output, reference_output = probe_data_pipeline(candidate_module, candidate_config)
        else:
            reference_output = reference_loader(reference_dataset_config)
            candidate_output = candidate_loader(candidate_dataset_config)

        update_check(
            current_index,
            True,
            ("Candidate executed against an independent synthetic fixture; "
             "this is not the full fixed dataset" if config.get("task_probe") else
             f"Both candidate and reference executed successfully. Candidate loader: {candidate_loader_name}, "
             f"Reference loader: {reference_loader_name}")
        )

        # ------------------ 4. 比较结果 ------------------
        current_index += 1

        def compare(reference: Any, candidate: Any) -> bool:
            # 字典
            if isinstance(reference, dict):
                return (
                        isinstance(candidate, dict)
                        and set(reference) == set(candidate)
                        and all(
                    compare(reference[key], candidate[key])
                    for key in reference
                )
                )

            # List / Tuple
            if isinstance(reference, (list, tuple)):
                return (
                        isinstance(candidate, (list, tuple))
                        and len(reference) == len(candidate)
                        and all(
                    compare(ref_item, candidate_item)
                    for ref_item, candidate_item in zip(
                        reference,
                        candidate,
                    )
                )
                )

            # PyTorch Tensor
            try:
                import torch

                if isinstance(reference, torch.Tensor):
                    if not isinstance(candidate, torch.Tensor):
                        return False

                    try:
                        torch.testing.assert_close(
                            reference.detach().cpu(),
                            candidate.detach().cpu(),
                            atol=atol,
                            rtol=rtol,
                        )
                        return True
                    except AssertionError:
                        return False

            except ImportError:
                pass

            # NumPy ndarray
            if isinstance(reference, np.ndarray):
                if not isinstance(candidate, np.ndarray):
                    return False

                if np.issubdtype(reference.dtype, np.number):
                    return np.allclose(
                        reference,
                        candidate,
                        atol=atol,
                        rtol=rtol,
                        equal_nan=True,
                    )

                return np.array_equal(reference, candidate)

            # 数值标量
            if isinstance(reference, (int, float, complex, np.number)):
                try:
                    return bool(
                        np.isclose(
                            reference,
                            candidate,
                            atol=atol,
                            rtol=rtol,
                            equal_nan=True,
                        )
                    )
                except (TypeError, ValueError):
                    return False

            # 其他类型
            try:
                result = reference == candidate

                if isinstance(result, np.ndarray):
                    return bool(np.all(result))

                return bool(result)

            except Exception:
                return False

        passed = compare(reference_output, candidate_output)

        if not passed:
            raise ValueError("Candidate and reference outputs differ")

        update_check(
            current_index,
            True,
            "Task synthetic-fixture outputs match the independent expected values" if config.get("task_probe") else "dataset_loader reference implementations are consistent"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_checkpoint_content(config: dict, caller_globals=None, caller_locals=None):
    """读取本次训练文件；不把 .pt 路径存在当作有效 checkpoint。"""
    specifications = config.get("checkpoints", [])
    CHECKPOINT_NAMES = ["artifacts:checkpoint:" + spec["path"] for spec in specifications]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    for index, spec in enumerate(specifications):
        try:
            path = CANDIDATE_DIR / spec["path"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or not payload:
                raise ValueError("checkpoint must contain a nonempty state mapping")
            state = payload
            for key in spec.get("model_keys", ["model_state_dict", "model_state", "model", "state_dict"]):
                if key in payload:
                    state = payload[key]
                    break
            if not isinstance(state, dict) or not state:
                raise ValueError("model state is empty or is not a mapping")
            if not all(isinstance(k, str) and torch.is_tensor(v) and bool(torch.isfinite(v).all())
                       for k, v in state.items()):
                raise ValueError("model state must contain named finite tensors")
            missing = [key for key in spec.get("required_keys", []) if key not in payload]
            if missing:
                raise ValueError(f"Prompt-required checkpoint entries missing: {missing}")
            if spec.get("kno_contract"):
                if not isinstance(payload["model_config"], dict) or not payload["model_config"]:
                    raise ValueError("model_config must be a nonempty parsed configuration")
                if not isinstance(payload["data_config"], dict) or not payload["data_config"]:
                    raise ValueError("data_config must be a nonempty parsed configuration")
                if list(payload["spatial_shape"]) != [64, 64]:
                    raise ValueError("spatial_shape must be [64,64]")
                if type(payload["epoch"]) is not int or payload["epoch"] < 0:
                    raise ValueError("epoch must be a nonnegative integer")
                metric = payload["best_relative_l2"]
                if type(metric) not in (int, float) or not math.isfinite(metric) or metric < 0:
                    raise ValueError("best_relative_l2 must be finite and nonnegative")
            checks[index] = (CHECKPOINT_NAMES[index], True, "Loaded nonempty finite model state and required metadata")
        except Exception:
            checks[index] = (CHECKPOINT_NAMES[index], False, traceback.format_exc())
    if not specifications:
        return [("artifacts:checkpoint_config", False, "checkpoints configuration is empty")]
    return checks


def make_deepcfd_probe(directory, count=981, height=6, width=7):
    """写小型真实 pickle；保留正式 981 样本划分，空间缩小仅用于数据单测。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    inputs = rng.normal(size=(count, 3, height, width)).astype(np.float32)
    targets = rng.normal(size=inputs.shape).astype(np.float32)
    targets += np.arange(count, dtype=np.float32)[:, None, None, None] / max(count, 1)
    for name, values in (("dataX.pkl", inputs), ("dataY.pkl", targets)):
        with (directory / name).open("wb") as handle:
            pickle.dump(values, handle)
    return inputs, targets


def probe_data_pipeline(module, config):
    """执行已绑定的候选 loader，独立核对划分、dtype、样本及训练集 RMS。"""
    import random
    import tempfile
    with tempfile.TemporaryDirectory(prefix="deepcfd-data-") as directory:
        inputs, targets = make_deepcfd_probe(directory)
        loader = getattr(module, config["loader_name"], None)
        if callable(loader):
            train, test, weights = loader(directory=directory, split_ratio=0.7, seed=0,
                                          batch_size=16, num_workers=0)
        elif all(callable(getattr(module, name, None)) for name in
                 ("load_config", "load_dataset", "make_loader", "compute_channel_rms")):
            runtime = module.load_config(CANDIDATE_DIR / "config.yaml")
            runtime["data"].update(input_path=str(Path(directory)/"dataX.pkl"),
                target_path=str(Path(directory)/"dataY.pkl"), expected_samples=len(inputs),
                sample_shape=list(inputs.shape[1:]))
            dataset, train_ids, test_ids = module.load_dataset(runtime)
            train = module.make_loader(dataset, train_ids, runtime, shuffle=True)
            test = module.make_loader(dataset, test_ids, runtime, shuffle=False)
            weights = module.compute_channel_rms(dataset, train_ids, runtime["data"]["batch_size"])
        else:
            raise ValueError("REVIEW_REQUIRED: bind the candidate's actual data-loader entry")
        indices = list(range(981))
        random.Random(0).shuffle(indices)
        train_ids, test_ids = indices[:686], indices[686:]
        actual = {"train_count": len(train.dataset), "test_count": len(test.dataset),
                  "train_x": torch.stack([train.dataset[i][0] for i in range(len(train.dataset))]),
                  "train_y": torch.stack([train.dataset[i][1] for i in range(len(train.dataset))]),
                  "test_x": torch.stack([test.dataset[i][0] for i in range(len(test.dataset))]),
                  "test_y": torch.stack([test.dataset[i][1] for i in range(len(test.dataset))]),
                  "weights": torch.as_tensor(weights).reshape(-1),
                  "train_shuffle": isinstance(train.sampler, torch.utils.data.RandomSampler),
                  "test_shuffle": isinstance(test.sampler, torch.utils.data.RandomSampler),
                  "batch_sizes": (train.batch_size, test.batch_size),
                  "workers": (train.num_workers, test.num_workers)}
        expected = {"train_count": 686, "test_count": 295,
                    "train_x": torch.from_numpy(inputs[train_ids]),
                    "train_y": torch.from_numpy(targets[train_ids]),
                    "test_x": torch.from_numpy(inputs[test_ids]),
                    "test_y": torch.from_numpy(targets[test_ids]),
                    "weights": torch.from_numpy(targets[train_ids]).square().mean((0, 2, 3)).sqrt(),
                    "train_shuffle": True, "test_shuffle": False,
                    "batch_sizes": (16, 16), "workers": (0, 0)}
        return actual, expected


def probe_training_contract(train_module, config):
    """只调用候选 Loss；不替候选实现训练或更新。"""
    function = getattr(train_module, config["loss_entry"], None)
    loss_class = getattr(train_module, "RMSWeightedLoss", None)
    if not callable(function) and not callable(loss_class):
        raise ValueError("REVIEW_REQUIRED: bind candidate RMS-weighted training loss")
    prediction = torch.linspace(-2., 3., 2 * 3 * 4 * 5).reshape(2, 3, 4, 5).requires_grad_()
    target = prediction.detach().flip(-1) * .3 + .4
    weights = torch.tensor([2., 3., 5.]).reshape(3, 1, 1)
    actual = function(prediction, target, weights) if callable(function) else loss_class(weights.flatten())(prediction, target)
    differences = prediction - target
    expected = (differences[:, :2].square() / weights[:2]).sum() + (differences[:, 2].abs() / weights[2]).sum()
    # Prompt 固定三通道权重/误差种类，未指定空间 sum/mean；两种合法标量归约均核对值及梯度。
    if not torch.allclose(actual, expected):
        expected = expected / differences[:, 0].numel()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    return "Candidate weighted velocity squared error / pressure absolute error and gradients match"


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    检查候选提交目录中是否存在配置指定的必需文件。

    config 示例:
    {
        "required_files": ["model.py", "config.yaml", "inference.py"]
    }

    检查点:
        - required_file: 检查 required_files 配置是否存在且非空
    """
    CHECKPOINT_NAMES = []
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        # ------------------ 1. 获取配置 ------------------
        required_files = config.get("required_files")
        if required_files is None or not len(required_files):
            raise ValueError("config['required_files'] is missing")

        required_files = [next((name for name in config.get("file_options", {}).get(item, [item])
                                if (CANDIDATE_DIR / name).is_file()), item)
                          for item in required_files]
        file_type = config.get("phase")
        if file_type is None:
            raise ValueError("config['phase'] is missing")

        # ------------------ 2. 定义 CHECKPOINT_NAMES ------------------
        for file in required_files:
            CHECKPOINT_NAMES.append(f"{file_type}.{file}")
            checks.append((f"{file_type}.{file}", False, "Not executed due to earlier failure"))

        # ------------------ 3. 检查文件是否存在且非空  ------------------
        for relative in required_files:
            file_path = Path(relative)
            exists = file_path.exists()
            if not exists:
                file_path = CANDIDATE_DIR / relative
            exists = file_path.exists()
            if exists:
                if relative.endswith(".pt"):
                    update_check(current_index,
                                True,
                                "File exist and not empty: {file_path}")
                else:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if len(content.strip()):
                            update_check(current_index,
                                        True,
                                        f"File exist and not empty: {file_path}")
                        else:
                            update_check(current_index,
                                        False,
                                        f"File exist but empty: {file_path}")
            else:
                update_check(current_index,
                             False,
                             f"File not found: {file_path}")
            current_index += 1

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

        train_files = ["common.py" if name == "dataset.py" and not (CANDIDATE_DIR / name).exists()
                       and (CANDIDATE_DIR / "common.py").is_file() else name for name in train_files]
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
            source = file_path.read_text(encoding="utf-8")
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
        # 统一转为小写进行比较，避免大小写不匹配
        model_names_lower = {name.lower() for name in model_names}
        model_attrs_lower = {attr.lower() for attr in model_attrs}
        # 检测模型调用（通过类名、函数名或属性/方法名）
        has_model = any(
            isinstance(node, ast.Call)
            and (
                # 匹配函数/类名，如 Model() 或 build_model()
                    (isinstance(node.func, ast.Name) and node.func.id.lower() in model_names_lower)
                    # 匹配属性/方法调用，如 model.forward() 或 model.train()
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr.lower() in model_attrs_lower
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id.lower() in {"model", "net", "network"})
                    # 匹配 torch.nn.Module 的实例化
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr.lower() in model_names_lower
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "torch")
            )
            for node in all_nodes
        ) or any(
            # 检测是否定义了模型类
            isinstance(node, ast.ClassDef)
            and node.name.lower() in model_names_lower
            for node in all_nodes
        )
        update_check(
            current_index,
            has_model,
            f"Model found: {has_model}, names: {model_names}"
        )

        # ------------------ 5. 检查 DataLoader ------------------
        current_index += 1
        dataloader_names_lower = {name.lower() for name in dataloader_names}
        has_dataloader = any(
            isinstance(node, ast.Call)
            and (
                # 直接调用: DataLoader(...)
                (isinstance(node.func, ast.Name) and node.func.id in dataloader_names)
                # 属性调用: torch.utils.data.DataLoader(...) 或 xxx.DataLoader(...)
                or (isinstance(node.func, ast.Attribute) and node.func.attr in dataloader_names)
            )
            for node in all_nodes
        ) or any(
            # 检测是否导入了 DataLoader（import 语句）
            isinstance(node, ast.ImportFrom)
            and any(alias.name in dataloader_names for alias in node.names)
            for node in all_nodes
        ) or any(
            # 检测 import torch.utils.data 形式
            isinstance(node, ast.Import)
            and any(
                alias.name.lower().endswith("data") and alias.asname is not None
                and alias.asname.lower() in dataloader_names_lower
                for alias in node.names
            )
            for node in all_nodes
        )
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

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_infer_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    静态检查推理脚本是否具备基本的推理流程。

    config 示例:
    {
        "files": ["inference.py", "predict.py"],          # 推理文件列表（支持多个）
        "check_config": {
            "entry_function_names": ["main", "infer"],              # 入口函数名称列表
            "model_load_functions": ["load_model"],                 # 模型加载函数名称列表
            "model_load_attributes": ["load_state_dict"],           # 模型加载属性/方法名称列表
            "input_parameter_names": ["data", "input"],             # 输入参数名称列表
            "input_load_functions": ["load_data"],                  # 输入加载函数名称列表
            "input_load_attributes": ["loadtxt", "read_csv"],       # 输入加载属性/方法名称列表
            "infer_call_functions": ["predict"],                    # 推理调用函数名称列表
            "infer_call_attributes": ["forward", "predict"],        # 推理调用属性/方法名称列表
            "output_attributes": ["save", "to_csv"]                 # 输出属性/方法名称列表
        }
    }

    注意：所有检查项都会按照 infer_check_config 中配置的内容进行检测，
    未配置的检查项将使用默认值。检查点如下：

    检查点（按执行顺序）:
        - inference_start:has_entry: 检查是否有推理入口
        - inference_start:has_model: 检查是否有模型加载
        - inference_start:has_eval: 检查是否设置 eval 模式
        - inference_start:has_input: 检查是否有输入数据
        - inference_start:has_inference: 检查是否有推理调用
        - inference_start:has_no_grad: 检查是否禁用梯度（可选）
        - inference_start:has_output: 检查是否有输出保存
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "inference_start:has_entry",
        "inference_start:has_model",
        "inference_start:has_eval",
        "inference_start:has_input",
        "inference_start:has_inference",
        "inference_start:has_no_grad",
        "inference_start:has_output",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        """更新指定索引的检查结果"""
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ------------------ 1. 检查推理文件是否存在 ------------------
        infer_files = config.get("files")
        if not infer_files:
            raise ValueError("config['files'] is missing or empty")

        # 确保 infer_files 是列表类型，如果不是则转换为列表
        if isinstance(infer_files, str):
            infer_files = [infer_files]
        elif not isinstance(infer_files, list):
            raise TypeError(f"config['run_infer_files'] must be a list or string, got {type(infer_files)}")

        infer_files = list(infer_files) + [name for name in config.get("support_files", []) if (CANDIDATE_DIR / name).is_file()]
        # 检查所有文件是否存在
        existing_files = []
        missing_files = []
        for infer_file in infer_files:
            infer_path = (CANDIDATE_DIR / infer_file).resolve()
            if infer_path.exists() and infer_path.is_file():
                existing_files.append(infer_path)
            else:
                missing_files.append(str(infer_path))

        if missing_files:
            raise FileNotFoundError(f"Inference files not found: {', '.join(missing_files)}")

        # ------------------ 2. 解析所有文件的源代码为 AST ------------------
        # 合并所有文件的 AST 节点，用于后续检查
        all_nodes = []
        file_paths = []
        for file_path in existing_files:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
            all_nodes.extend(list(ast.walk(tree)))
            file_paths.append(str(file_path))

        # 从配置中获取检查参数，若无则使用默认值
        infer_check_config = config.get("check_config", {})

        # 入口函数名称（用于检测推理脚本的入口点）
        entry_func_names = set(infer_check_config.get(
            "entry_function_names",
            ["main", "inference", "infer", "predict", "run_inference", "run_predict"]
        ))

        # 模型加载函数名称（如 load_model, load_checkpoint 等）
        model_load_funcs = set(infer_check_config.get(
            "model_load_functions",
            ["load_model", "load_checkpoint", "load_weights"]
        ))
        # 模型加载属性/方法名称（如 load_state_dict 等）
        model_load_attrs = set(infer_check_config.get(
            "model_load_attributes",
            ["load_state_dict", "load_checkpoint", "load_weights"]
        ))

        # 输入参数名称（用于检测函数参数中的输入数据）
        input_arg_names = set(infer_check_config.get(
            "input_parameter_names",
            ["input", "inputs", "data", "sample", "samples", "x", "batch"]
        ))

        # 输入加载函数名称（如 DataLoader, load_data 等）
        input_load_funcs = set(infer_check_config.get(
            "input_load_functions",
            ["DataLoader", "Dataset", "load_data", "load_dataset"]
        ))
        # 输入加载属性/方法名称（如 loadtxt, read_csv 等）
        input_load_attrs = set(infer_check_config.get(
            "input_load_attributes",
            ["load", "loadtxt", "read_csv", "read_json"]
        ))

        # 推理调用函数名称（如 predict, inference 等）
        infer_call_funcs = set(infer_check_config.get(
            "infer_call_functions",
            ["predict", "inference", "infer", "run_inference", "predict_batch"]
        ))
        # 推理调用属性/方法名称（如 forward, predict 等）
        infer_call_attrs = set(infer_check_config.get(
            "infer_call_attributes",
            ["forward", "predict", "inference", "infer"]
        ))

        # 输出属性/方法名称（用于检测结果保存操作）
        output_attrs = set(infer_check_config.get(
            "output_attributes",
            ["save", "savez", "dump", "write", "to_csv", "to_json", "to_pickle"]
        ))

        # ------------------ 3. 检查推理入口 -----------------
        # 检测是否存在入口函数（函数定义匹配 entry_func_names）
        has_entry = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.lower() in entry_func_names
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
            f"Entry found: {has_entry}, names: {entry_func_names}"
        )

        # ------------------ 4. 检查模型加载 ------------------
        current_index += 1
        # 检测模型加载调用（通过函数名或属性/方法名）
        has_model_load = any(
            isinstance(node, ast.Call)
            and (
                # 匹配函数名，如 load_model()
                    (isinstance(node.func, ast.Name) and node.func.id.lower() in model_load_funcs)
                    # 匹配属性/方法名，如 model.load_state_dict()
                    or (isinstance(node.func, ast.Attribute) and node.func.attr.lower() in model_load_attrs)
                    # 特殊处理 torch.load()
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch")
            )
            for node in all_nodes
        )
        # 注意：不再单独检测模型构建函数（build_model等），
        # 因为已通过 model_load_funcs 和 model_load_attrs 覆盖了模型初始化和 load_state_dict 检查
        has_model = has_model_load
        update_check(
            current_index,
            has_model,
            f"Model found: {has_model}"
        )

        # ------------------ 5. 检查 eval 模式 ------------------
        current_index += 1
        # 检测是否调用了 .eval() 方法（PyTorch 模型评估模式）
        has_eval_mode = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "eval"
            for node in all_nodes
        )
        update_check(
            current_index,
            has_eval_mode,
            f"eval mode found: {has_eval_mode}"
        )

        # ------------------ 6. 检查输入数据 ------------------
        current_index += 1
        # 检测输入数据加载（通过函数名或属性/方法名）
        has_input_loader = any(
            isinstance(node, ast.Call)
            and (
                    (isinstance(node.func, ast.Name) and node.func.id in input_load_funcs)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr in input_load_attrs)
            )
            for node in all_nodes
        )
        # 或检测函数参数中包含输入参数名
        has_input_parameter = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(arg.arg.lower() in input_arg_names for arg in node.args.args)
            for node in all_nodes
        )
        has_input = has_input_loader or has_input_parameter
        update_check(
            current_index,
            has_input,
            f"Input found: {has_input}"
        )

        # ------------------ 7. 检查推理调用 ------------------
        current_index += 1
        # 检测推理调用（通过函数名或属性/方法名）
        has_infer_call = any(
            isinstance(node, ast.Call)
            and (
                    (isinstance(node.func, ast.Name) and node.func.id.lower() in infer_call_funcs)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr.lower() in infer_call_attrs)
            )
            for node in all_nodes
        )
        # 或检测是否直接调用 model()、net() 等变量
        has_model_call = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.lower() in {"model", "network", "net"}
            for node in all_nodes
        )
        has_infer = has_infer_call or has_model_call
        update_check(
            current_index,
            has_infer,
            f"Inference call found: {has_infer}"
        )

        # ------------------ 8. 检查梯度禁用（可选检查） ------------------
        current_index += 1
        # 检测是否使用了 torch.no_grad() 或 torch.inference_mode()
        has_no_grad = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr in {"no_grad", "inference_mode"}
            for node in all_nodes
        )
        update_check(
            current_index,
            has_no_grad,
            f"no_grad found: {has_no_grad}"
        )

        # ------------------ 9. 检查结果输出 ------------------
        current_index += 1
        # 检测输出保存操作（通过属性/方法名，如 save、to_csv 等）
        # 注意：只检查文件保存相关的属性/方法，不检查 print 等输出函数
        has_output = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in output_attrs
            for node in all_nodes
        )
        update_check(
            current_index,
            has_output,
            f"Output found: {has_output}"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks
