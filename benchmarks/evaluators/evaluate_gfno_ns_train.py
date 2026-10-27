"""evaluate_gfno_ns_train：已核对 Agent 接口接入与误判修正（r5）。

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
import math
import traceback  # noqa: F401
from copy import deepcopy  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Mapping, Tuple  # noqa: F401

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
    check_files,
    check_interface,
    check_parameter_update,
    check_train_structure,
)
from torch import ones  # noqa: E402,F401  # 前置钩子 function: "ones" 经 globals() 解析

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
    if config.get("bind_gfno_roles"):
        CHECKPOINT_NAMES.append("model_structure:bound_roles")
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
        # 本 case 配置内嵌于代码（无配置文件），config_name 可选
        candidate_config = _load_candidate_config(config_name) if config_name else {}
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
        if config.get("bind_gfno_roles"):
            paths = gfno_structure_paths(model, config)
            update_check(CHECKPOINT_NAMES.index("model_structure:bound_roles"), True,
                         f"Actual GFNO role bindings={paths}; mathematical forward checked separately")
            rules = deepcopy(rules)
            for rule in rules:
                params = rule.get("params", {})
                target = params.get("target_path", "")
                if target.startswith("role:"):
                    binding = paths[target[5:]]
                    if isinstance(binding, list):
                        # ModuleList 分组和四个复合 Block 的组织不同；仍检查真实模块数。
                        classes = {model.get_submodule(path).__class__.__name__ for path in binding}
                        if len(classes) != 1:
                            raise ValueError("REVIEW_REQUIRED: mixed GFNO role classes")
                        rule["type"] = "module_count"
                        params.pop("target_path", None)
                        params["count"] = params.pop("length")
                        params["selector"] = {"class_name": next(iter(classes)), "registered_name": binding[0].rsplit(".", 1)[-1]}
                    else:
                        params["target_path"] = binding
        checker.add_rules_from_rules(rules)

        # 提取结构信息
        extractor = ModuleInfoExtractor(model)
        root_info = extractor.extract()

        # 执行检查：构建成功后才按真实模块展开规则，准备失败保留已登记项。
        checks[:] = [item for item in checks if item[2] != "Not executed due to earlier failure"]
        CHECKPOINT_NAMES[:] = [item[0] for item in checks]
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
        # 本 case 配置内嵌于代码（无配置文件），config_name 可选
        model_config = _load_candidate_config(config_name) if config_name else {}
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

        # ---- 可选：p4 90° 旋转等变性检查（本 case prompt 核心契约） ----
        if config.get("check_p4_equivariance"):
            current_index = CHECKPOINT_NAMES.index("forward_executable:p4_equivariance")
            try:
                _x = converted_forward_args.get("x")
                if not torch.is_tensor(_x):
                    raise TypeError("p4 check requires tensor forward_args['x']")
                if _x.ndim < 4:
                    raise ValueError(f"p4 check expects [B,H,W,...] input, got {tuple(_x.shape)}")
                with torch.no_grad():
                    _rot_out = model(**{**converted_forward_args, "x": torch.rot90(_x, 1, dims=(1, 2))})
                _expected = torch.rot90(out, 1, dims=(1, 2))
                _diff = torch.linalg.vector_norm((_rot_out - _expected).reshape(len(_expected), -1), dim=1)
                _scale = torch.linalg.vector_norm(_expected.reshape(len(_expected), -1), dim=1).clamp_min(1e-12)
                _rel = float((_diff / _scale).max())
                _rtol = float(config.get("p4_rtol", 1e-4))
                update_check(
                    current_index,
                    _rel <= _rtol,
                    f"rotation equivariance relative_l2={_rel:.3e}, rtol={_rtol}",
                )
            except Exception as _p4_exc:
                update_check(current_index, False, f"p4 equivariance check error: {_p4_exc}")

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

        # 加载模块（本 case 配置内嵌于代码，config_name 可选）
        if is_candidate:
            module = _load_candidate_modules(model_file)
            model_config = _load_candidate_config(config_name) if config_name else {}
        else:
            module = _load_reference_modules(model_file)
            model_config = _load_reference_config(config_name) if config_name else {}

        # 将配置添加到上下文中
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

        cli_args = config.get("cli_args")
        if callable(getattr(train_module, "run_training", None)) and callable(getattr(train_module, "build_loaders", None)):
            cli_args = ["--data-path", str(context["probe_dir"] / "probe.mat"), "--output-dir", str(results_dir),
                "--device", "cpu", "--ntrain", "2", "--nval", "2", "--ntest", "2", "--batch-size", "20",
                "--epochs", "1", "--modes", "4", "--width", "4", "--cpu-threads", "4"]
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
                    _cli_args = cli_args
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

# 当前任务的数据/产物接入；普通分支保留 Base 流程。










def deterministic_tensor(size, seed=17, dtype=None, device=None):
    """前向/参考对照共用同一局部随机序列；不改变全局 RNG。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(tuple(size), generator=generator, dtype=dtype or torch.float32)
    return _move_tensors_to_device(value) if device is None else value.to(device)


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


def make_ns_probe(directory, count=1207, height=4, width=4, frames=30):
    """MAT v5 + HDF5 两份相同方形场；非均匀数值与样本编号用于核对轴和切分。"""
    from scipy.io import savemat
    import h5py
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(count, height, width, frames)).astype(np.float32)
    raw += np.arange(count, dtype=np.float32)[:, None, None, None] / max(count, 1)
    savemat(directory / "probe.mat", {"u": raw})
    with h5py.File(directory / "probe_hdf5.mat", "w") as handle:
        handle.create_dataset("u", data=raw.transpose(3, 2, 1, 0))
    return torch.from_numpy(raw[..., None])


def probe_data_pipeline(module, config):
    """真实运行 MAT reader/loader；只用合成小网格，不冒充固定文件全量验证。"""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="gfno-data-") as directory:
        values = make_ns_probe(directory)
        reader = getattr(module, config["reader_name"], None)
        loader = getattr(module, config["loader_name"], None)
        if callable(getattr(module, "load_ns_splits", None)) and callable(getattr(module, "build_datasets", None)):
            actual, expected = {}, {}
            for filename in ("probe.mat", "probe_hdf5.mat"):
                splits, _ = module.load_ns_splits(str(Path(directory)/filename))
                for key, reference in (("train",values[:1000]), ("validation",values[-200:-100]), ("test",values[-100:])):
                    actual[filename+key] = splits[key].unsqueeze(-1)
                    expected[filename+key] = reference
            datasets, normalizer, _ = module.build_datasets(str(Path(directory)/"probe.mat"))
            tools = _load_candidate_modules("utils.py")
            trainer = _load_candidate_modules("train.py")
            runtime = tools.load_config([])
            loaders = trainer.build_loaders(datasets, runtime, torch.device("cpu"), torch.Generator().manual_seed(17))
            for label, reference in (("train",values[:1000]), ("validation",values[-200:-100]), ("test",values[-100:])):
                length = len(reference)*20 if label == "train" else len(reference)
                actual[label+"count"] = len(datasets[label]); expected[label+"count"] = length
                actual[label+"shuffle"] = isinstance(loaders[label].sampler, torch.utils.data.RandomSampler)
                expected[label+"shuffle"] = label == "train"
                for index in (0,1,length-1):
                    x,y = datasets[label][index]
                    trajectory,step = divmod(index,20) if label == "train" else (index,0)
                    actual[f"{label}_{index}_x"] = x; expected[f"{label}_{index}_x"] = reference[trajectory,...,step:step+10,:]
                    actual[f"{label}_{index}_y"] = y; expected[f"{label}_{index}_y"] = reference[trajectory,...,step+10:(step+11 if label=="train" else 30),:]
            enabled = "standard"
            from scipy.io import savemat
            changed = values[...,0].numpy().copy(); changed[1000:] += 500.
            savemat(Path(directory)/"heldout_changed.mat", {"u":changed})
            normalized,norm,_ = module.build_datasets(str(Path(directory)/"probe.mat"), normalization=enabled)
            other,_,_ = module.build_datasets(str(Path(directory)/"heldout_changed.mat"), normalization=enabled)
            actual["train_statistics_invariant"] = normalized["train"][0][0]; expected["train_statistics_invariant"] = other["train"][0][0]
            actual["normalization_active"] = not torch.equal(normalized["train"][0][0],datasets["train"][0][0]); expected["normalization_active"] = True
            actual["decode_train"] = norm.decode(normalized["train"][0][0]); expected["decode_train"] = values[0,...,:10,:]
            return actual,expected
        if not callable(reader) or not callable(loader):
            raise ValueError("REVIEW_REQUIRED: bind candidate MAT reader and data-loader entries")
        actual = {"mat": reader(str(Path(directory) / "probe.mat")),
                  "hdf5": reader(str(Path(directory) / "probe_hdf5.mat"))}
        expected = {"mat": values, "hdf5": values}
        bundle = loader(data_path=str(Path(directory) / "probe.mat"),
                        expected_resolution=None, pin_memory=False)
        for label, current, reference in (
            ("train", bundle.train_loader, values[:1000]),
            ("valid", bundle.valid_loader, values[-200:-100]),
            ("test", bundle.test_loader, values[-100:]),
        ):
            length = len(reference) * 20 if label == "train" else len(reference)
            actual[label + "_count"] = len(current.dataset)
            expected[label + "_count"] = length
            actual[label + "_sampler"] = isinstance(current.sampler, torch.utils.data.RandomSampler)
            expected[label + "_sampler"] = label == "train"
            for index in (0, 1, length - 1):
                x, y = current.dataset[index]
                actual[f"{label}_{index}_x"] = x
                actual[f"{label}_{index}_y"] = y.reshape(reference.shape[1], reference.shape[2], -1, 1)
                trajectory, step = divmod(index, 20) if label == "train" else (index, 0)
                expected[f"{label}_{index}_x"] = reference[trajectory, ..., step:step + 10, :]
                stop = step + 11 if label == "train" else 30
                expected[f"{label}_{index}_y"] = reference[trajectory, ..., step + 10:stop, :]
        actual["normalization_default_none"] = bundle.normalizer is None
        expected["normalization_default_none"] = True
        normalized = loader(data_path=str(Path(directory) / "probe.mat"), expected_resolution=None,
                            normalization=config["enabled_normalization"], pin_memory=False)
        # 同一 train，改变 held-out 数据；训练统计/归一化结果必须不受影响。
        from scipy.io import savemat
        changed = values[..., 0].numpy().copy()
        changed[1000:] += 500.0
        savemat(Path(directory) / "heldout_changed.mat", {"u": changed})
        other = loader(data_path=str(Path(directory) / "heldout_changed.mat"), expected_resolution=None,
                       normalization=config["enabled_normalization"], pin_memory=False)
        actual["train_statistics_invariant"] = normalized.train_loader.dataset[0][0]
        expected["train_statistics_invariant"] = other.train_loader.dataset[0][0]
        normal_x = normalized.train_loader.dataset[0][0]
        actual["normalization_active"] = not torch.equal(normal_x, bundle.train_loader.dataset[0][0])
        expected["normalization_active"] = True
        actual["decode_train"] = normalized.normalizer.decode(normal_x).reshape_as(values[0, ..., :10, :])
        expected["decode_train"] = values[0, ..., :10, :]
        return actual, expected


def probe_training_contract(train_module, config):
    """候选 Relative L2 和 20 步反馈；一阶训练更新另由原 main 的观察器验证。"""
    tools = _load_candidate_modules(config["utils_file"])
    loss_class = getattr(tools, config["loss_entry"], None)
    rollout = getattr(tools, config["rollout_entry"], None)
    if not callable(loss_class) or not callable(rollout):
        raise ValueError("REVIEW_REQUIRED: bind candidate Relative L2 and autoregressive rollout")
    prediction = torch.linspace(-.8, 1.2, 30).reshape(2, 15, 1).requires_grad_()
    target = torch.linspace(.2, 1.7, 30).reshape(2, 15, 1)
    loss_kwargs = {"size_average": False} if "size_average" in inspect.signature(loss_class).parameters else {"reduction": "sum"}
    actual = loss_class(**loss_kwargs)(prediction, target)
    expected = (torch.linalg.vector_norm((prediction-target).reshape(2, -1), dim=1) /
                torch.linalg.vector_norm(target.reshape(2, -1), dim=1)).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, prediction, retain_graph=True)[0],
                               torch.autograd.grad(expected, prediction)[0])
    histories = []
    class OneStep(torch.nn.Module):
        def forward(self, x):
            histories.append(x.detach().clone())
            return x[..., -1:, :] + .25
    x = torch.arange(2 * 3 * 4 * 10, dtype=torch.float32).reshape(2, 3, 4, 10, 1) / 100
    actual = rollout(OneStep(), x.clone(), 20)
    expected = torch.cat([x[..., -1:, :] + (i + 1)*.25 for i in range(20)], dim=-2)
    torch.testing.assert_close(actual, expected)
    if len(histories) != 20:
        raise ValueError("Expected 20 actual one-step model calls")
    state = x
    for seen in histories:
        torch.testing.assert_close(seen, state)
        state = torch.cat((state[..., 1:, :], state[..., -1:, :] + .25), dim=-2)
    return "Candidate Relative L2 value/gradient and 20-step autoregressive feedback match"


def gfno_structure_paths(model, config):
    """定位已知 p4 布局中的真实模块；不把 registered_name 当验收条件。"""
    paths = {}
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, torch.nn.ModuleList) and len(blocks) == 4 and all(
            all(isinstance(getattr(block, role, None), torch.nn.Module) for role in ("spectral","mlp","residual")) for block in blocks):
        return {"spectral": [f"blocks.{i}.spectral" for i in range(4)],
                "mlp": [f"blocks.{i}.mlp" for i in range(4)], "skip": [f"blocks.{i}.residual" for i in range(4)]}
    for name, value in model.named_modules():
        if isinstance(value, torch.nn.ModuleList) and len(value) == 4:
            if all(any(p.is_complex() for p in item.parameters()) for item in value):
                role = "spectral"
            elif all(len(list(item.children())) == 0 and len(list(item.parameters())) > 0 for item in value):
                role = "skip"
            elif all(len(list(item.children())) >= 2 for item in value):
                role = "mlp"
            else:
                continue
            if role in paths:
                raise ValueError(f"REVIEW_REQUIRED: ambiguous GFNO {role} binding")
            paths[role] = name
    if set(paths) != {"spectral", "skip", "mlp"}:
        raise ValueError("REVIEW_REQUIRED: bind the candidate's actual G-Fourier block layout")
    lifting = [name for name, value in model.named_modules() if getattr(value, "first_layer", False)]
    projection = [name for name, value in model.named_modules() if getattr(value, "last_layer", False)]
    if len(lifting) != 1 or len(projection) != 1:
        raise ValueError("REVIEW_REQUIRED: bind candidate group lifting/projection roles")
    paths.update(lifting=lifting[0], projection=projection[0])
    return paths


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
        if not config_name and not config.get("config_source"):
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
        if config.get("config_source"):
            source = config["config_source"]
            module = _load_candidate_modules(source["file"])
            factory = getattr(module, source["function"], None)
            serialize = getattr(module, source["serialize"], None)
            if callable(factory) and callable(serialize):
                config_data = serialize(factory())
            elif callable(getattr(module, "load_config", None)) and hasattr(module, "ExperimentConfig"):
                from dataclasses import asdict
                config_data = asdict(module.load_config([]))
            else:
                raise ValueError("REVIEW_REQUIRED: bind candidate experiment-config factory")
        else:
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
