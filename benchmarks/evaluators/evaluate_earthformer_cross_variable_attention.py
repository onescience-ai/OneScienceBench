"""Earthformer 跨变量注意力：旧测试范围的 checks/params 标准化。

仅在专用文件适配本轮上传的 Base；不读取旧 run()、不修改候选源码。
结构不绑定未写入 Prompt 的内部属性名或模块注册顺序。
当前范围不含非法输入、[V]/None mask、masked MSE 数值对照或注意力数学证明。
"""
from __future__ import annotations

import ast
import inspect
import traceback
from copy import deepcopy
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np

import torch

from base.base_evaluator import (
    CANDIDATE_DIR,
    SUFFIX_MAP,
    DATA_LOADER_REGISTRY,
    DEVICE,
    _convert_arg_to_type,
    _execute_hooks,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _move_tensors_to_device,
    _resolve_args,
    check_files,
    check_interface,
)


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
        - model_structure:structure: 无可定位规则时，仅在结构准备失败路径追加
        - model_structure:{rule.id}: 每条已配置规则；非聚合规则追加实际模块路径，MHA 按实际类型检查
    """
    import torch.nn as nn

    # ================================================================
    # 0. 定义检查点；规则路径确定后同步展开名称和结果
    # ================================================================
    CHECKPOINT_NAMES = [
        "model_structure:build_model_ok",
    ]
    # 先保留配置中的规则；取得实际模块后，同步展开名称和结果，不改变原规则判定。
    if isinstance(config.get("rules"), list):
        CHECKPOINT_NAMES += [f"model_structure:{rule.get('id', str(rule.get('type')))}"
                             for rule in config["rules"] if isinstance(rule, dict)]
    rule_count = len(CHECKPOINT_NAMES) - 1
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
        context = {}  # 初始化上下文

        model_class_name = config.get("model_class")
        if not config.get("pre_model_hooks"):
            if not model_class_name:
                raise ValueError("config['model_class'] is missing or empty")

            model_cls = _load_class(model_module, model_class_name)

        # ------------------ 2. 构建模型 ------------------
        config_name = config.get("config_name")
        model_config = _load_candidate_config(config_name) if config_name else {}
        context["config"] = model_config
        if isinstance(model_config, dict):
            context.update(model_config)

        if config.get("pre_model_hooks"):
            # 沿用旧评测的 train.build_model(config)，不要求候选新增 YAML。
            context = _execute_hooks(model_module, config["pre_model_hooks"], context)
            model = context.get("model")
            if not isinstance(model, nn.Module):
                raise TypeError("pre_model_hooks must save an nn.Module as 'model'")
            model = model.to(DEVICE).eval()
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
            f"Model built with config: {model_config}"
        )

        current_index = 1
        from base.model_structure_checker import ModuleInfoExtractor, ModelStructureChecker
        checker = ModelStructureChecker()

        rules = config.get("rules")
        if rules is None:
            raise ValueError("config['rules'] is missing")
        checker.add_rules_from_rules(rules)
        # 按 Base 解析出的原顺序登记；保留映射/文件入口及其默认规则名。
        names = [f"model_structure:{rule.id}" for rule in checker.rules]
        CHECKPOINT_NAMES[1:1 + rule_count] = names
        checks[1:1 + rule_count] = [
            (name, False, "Not executed due to earlier failure") for name in names
        ]

        # 提取结构信息
        extractor = ModuleInfoExtractor(model)
        root_info = extractor.extract()

        # 执行检查
        for rule in checker.rules:
            # MHA 必须按真实对象类型检查；名称匹配会误拒绝子类并接受同名假模块。
            if (rule.rule_type == "module_type"
                    and (rule.params or {}).get("expected") == "MultiheadAttention"
                    and (rule.params or {}).get("match") == "any"):
                paths = [path for path, module in model.named_modules()
                         if isinstance(module, nn.MultiheadAttention)]
                update_check(current_index, bool(paths),
                             f"nn.MultiheadAttention instances (including subclasses): {paths}")
                current_index += 1
                continue

            modules_to_check = checker._get_modules_for_rule(root_info, rule, None)

            if (rule.params or {}).get("match") == "any":
                # 只聚合原生规则的判定；不改 checker 的选择器或规则算法。
                results = [(module_info.module_path, *checker.evaluate_rule(rule, module_info))
                           for module_info in modules_to_check]
                passed = any(ok for _, ok, _ in results)
                message = "; ".join(f"{path or '<root>'}: {msg}" for path, _, msg in results)
                update_check(current_index, passed, message or "No matching modules")
                current_index += 1
                continue

            if rule.rule_type == "module_count":
                expected = (rule.params or {}).get("count")
                actual = len(modules_to_check)
                passed = expected is not None and actual == expected
                message = (
                    f"Module count={actual} matches"
                    if passed
                    else f"Module count: expected {expected}, got {actual}"
                )
                update_check(current_index, passed, message)
                current_index += 1
                continue

            # 模块路径只能在构建后取得；同步替换本规则的预留项，保留后续未执行项。
            names = [f"model_structure:{rule.id}:{module_info.module_path}"
                     if module_info.module_path else f"model_structure:{rule.id}"
                     for module_info in modules_to_check]
            CHECKPOINT_NAMES[current_index:current_index + 1] = names
            checks[current_index:current_index + 1] = [
                (name, False, "Not executed due to earlier failure") for name in names
            ]
            for module_info in modules_to_check:
                passed, msg = checker.evaluate_rule(rule, module_info)
                update_check(current_index, passed, msg)
                current_index += 1

    except Exception as e:
        error_detail = traceback.format_exc()
        if current_index >= len(checks):
            CHECKPOINT_NAMES.append("model_structure:structure")
            checks.append((CHECKPOINT_NAMES[-1], False, "Not executed due to earlier failure"))
        update_check(current_index, False, error_detail)

    return checks



def check_model_forward_executable(config: dict, caller_globals=None, caller_locals=None):
    """
    检查模型前向；按配置校验形状／有限性、mask 置零，或同模型变量置换响应。

    config 示例:
    {
        "model_file": "model.py",
        "model_class": "MyModel",
        "config_name": "config.yaml",
        "model_args": {"space_dim": "${config.space_dim}", "n_layers": 3, "hidden_size": 512},  # 模型参数（支持占位符）
        "forward_args": {"x": [[1.0, 2.0], [3.0, 4.0]], "fx": "None", "T": "$t"},  # 前向传播参数（支持占位符）
        "out_shape": [2,2,8],
        "pre_forward_hooks": [  # 可选：前置钩子
            {"function": "load_weights", "args": {"checkpoint": "ckpt.pth"}}
        ]
    }

    检查点:
        - forward_executable:forward_run: 检查前向传播能否执行
        - forward_executable:returns_tensor_shape: 配置 out_shape 时检查 Tensor 形状和有限性
        - forward_executable:masked_variables_zero: 配置 mask_atol 时检查置零容差
        - variable_semantics:permutation_changes_output: 配置 permutation 时检查响应，不证明注意力轴正确
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
    ]
    if "mask_atol" in config:
        CHECKPOINT_NAMES.append("forward_executable:masked_variables_zero")
    if "permutation" in config:
        CHECKPOINT_NAMES.append("variable_semantics:permutation_changes_output")
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
        model_config = _load_candidate_config(config_name) if config_name else {}
        context["config"] = model_config
        if isinstance(model_config, dict):
            context.update(model_config)

        # ------------------ 3. 执行前置钩子（如果有） ------------------
        pre_hooks = config.get("pre_forward_hooks", [])
        for hook_config in pre_hooks:
            source_module = (
                torch
                if hook_config.get("function") in ("randn", "randn_like", "tensor")
                else model_module
            )
            context = _execute_hooks(source_module, [hook_config], context)

        # ------------------ 4. 构建模型 ------------------
        if "model" in context:
            # 前置调用保存的对象就是旧评测的 build_model 返回值。
            model = context["model"]
            if not isinstance(model, torch.nn.Module):
                raise TypeError("pre_forward_hooks must save an nn.Module as 'model'")
            model = model.to(DEVICE).eval()
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

        with torch.no_grad():
            # 旧脚本按位置传 x/mask；不额外约束候选的形参命名。
            out = model(*converted_forward_args.values())

        # 更新检查点：forward_run 通过
        update_check(
            current_index,
            True,
            f"Forward pass executed successfully with args"
        )

        # ------------------ 8. 检查 Tensor；按配置检查形状和有限性 ------------------
        current_index += 1

        # 检查是否为 Tensor
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Output is not a Tensor: {type(out)}")

        # 置换场景原本只比较两次输出；未配置 out_shape 时不增加有限性／形状断言。
        if "out_shape" not in config and "permutation" in config:
            same_shape, finite = True, True
        else:
            same_shape = (tuple(out.shape) == tuple(config.get("out_shape", [])))
            finite = bool(torch.isfinite(out).all())
        update_check(
            current_index,
            same_shape and finite,
            (f"Output is Tensor with shape={tuple(out.shape)}, expected out_dim={config.get('out_shape', [])}; finite={finite}"
             if "out_shape" in config else "Output shape/finite check skipped: permutation-only fixture")
        )

        if "mask_atol" in config:
            # 保留旧版 [B,V] mask 样例和 atol=1e-7，不增加新的 mask 边界。
            current_index += 1
            mask = converted_forward_args["variable_mask"]
            masked = out * (~mask[:, None, :, None, None]).to(out)
            masked_zero = bool(torch.allclose(masked, torch.zeros_like(masked), atol=config["mask_atol"]))
            update_check(current_index, masked_zero, "invalid variables are zeroed within mask_atol")

        if "permutation" in config:
            # 复用上面同一次原输入前向结果，仅追加原有置换输入的第二次调用。
            current_index += 1
            x = converted_forward_args["x"]
            mask = converted_forward_args["variable_mask"]
            permuted = x[:, :, config["permutation"]]
            with torch.no_grad():
                out_permuted = model(permuted, mask)
            changed = not bool(torch.allclose(out, out_permuted))
            update_check(current_index, changed,
                         "Permuting variable values changes output; this alone does not prove variable-axis attention")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks



def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> List[Tuple[str, bool, str]]:
    """
    检查数据 schema（形状、字段及配置的 dtype）；pre_hooks 存在时检查评测侧内存数据。
    内存场景不读取候选数据文件，不证明候选接口接受这些输入。

    config 示例:
    {
        "config_name": "config.yaml",  # 文件模式必填；pre_hooks 内存模式不需要
        "schema": [  # 数据文件的检查规则
            {
                "path_key": "sigma_path",  # 文件模式为配置路径键；内存模式为 pre_hooks 的 save_as 键
                "shape": {"ndim": 2, "dims": [None, None]},  # shape要求
                "fields": ["field1", "field2"],  # 字段要求（针对结构化数据）
            },
            {
                "path_key": "xy_path",
                "shape": {"ndim": 3, "dims": [None, 2, None]},
            }
        ]
    }

    检查点:
        - dataset_schema:file_load / data_load: 文件加载或评测侧内存数据准备
        - dataset_schema:shape_check: 检查数据形状是否符合预期
        - dataset_schema:field_check: 字段检查，未配置字段时沿用 Base 的跳过结果
        - dataset_schema:dtype_check: schema 配置 dtype 时检查实际数据类型
    """

    # ================================================================
    # 0. 定义检查点名称和辅助函数
    # ================================================================
    CHECKPOINT_NAMES = [
        "dataset_schema:file_load",
        "dataset_schema:shape_check",
        "dataset_schema:field_check",
    ]
    if config.get("pre_hooks"):
        CHECKPOINT_NAMES[0] = "dataset_schema:data_load"
    if isinstance(config.get("schema"), list) and any(
            isinstance(spec, dict) and "dtype" in spec for spec in config["schema"]):
        CHECKPOINT_NAMES.append("dataset_schema:dtype_check")
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ================================================================
        # 1. 获取配置信息
        # ================================================================
        config_name = config.get("config_name")
        if config.get("pre_hooks"):
            config_data = _execute_hooks(torch, config["pre_hooks"], {"config": {}})
        else:
            if not config_name:
                raise ValueError("config_name not found in config")
            config_data = _load_candidate_config(config_name)

        # ================================================================
        # 2. 获取schema配置
        # ================================================================
        schema_config = config.get("schema")
        if not schema_config:
            raise ValueError("schema not found in config")

        if not isinstance(schema_config, list):
            raise TypeError(f"schema must be a list, got {type(schema_config)}")

        # 验证schema配置
        for idx, file_spec in enumerate(schema_config):
            if not isinstance(file_spec, dict):
                raise TypeError(f"schema[{idx}] must be a dict, got {type(file_spec)}")

            path_key = file_spec.get("path_key")
            if not path_key:
                raise ValueError(f"path_key not found in schema[{idx}]")
            if not isinstance(path_key, str):
                raise TypeError(f"path_key must be a string, got {type(path_key)}")

        # ================================================================
        # 3. 执行检查
        # ================================================================
        loaded_data = {}
        file_exist_msgs = []
        file_load_msgs = []
        shape_check_msgs = []
        field_check_msgs = []
        dtype_check_msgs = []

        # 处理每个数据文件
        for file_spec in schema_config:
            current_index = 0
            path_key = file_spec["path_key"]

            if config.get("pre_hooks"):
                # path_key 在本场景仅作为内存对象的键，不创建伪数据文件。
                data = config_data[path_key]
                file_path = f"<evaluator-memory:{path_key}>"
            else:
                # 从配置数据中获取实际路径
                file_path = config_data.get(path_key)
                if file_path is None:
                    raise KeyError(f"Path key '{path_key}' not found in config: {config_name}")

                if not isinstance(file_path, (str, Path)):
                    raise TypeError(f"Path for '{path_key}' must be string or Path, got {type(file_path)}")

                file_path = Path(file_path).resolve()

                # 检查文件是否存在
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path} (path_key: {path_key})")

                file_exist_msgs.append(f"{path_key}: {file_path}")

                # 从文件后缀推断文件类型
                suffix = ''.join(file_path.suffixes)  # 支持 .tar.gz 等多后缀
                if not suffix:
                    suffix = file_path.suffix.lower()

                file_type = SUFFIX_MAP.get(suffix.lower())

                if not file_type:
                    # 尝试使用最后一个后缀
                    last_suffix = file_path.suffix.lower()
                    file_type = SUFFIX_MAP.get(last_suffix)

                if not file_type:
                    raise ValueError(
                        f"Unsupported file extension: '{file_path.suffix}' for {path_key}. "
                        f"Supported extensions: {list(SUFFIX_MAP.keys())}"
                    )

                # 获取对应的加载器
                loader = DATA_LOADER_REGISTRY.get(file_type)
                if not loader:
                    raise KeyError(
                        f"File type '{file_type}' (from extension {file_path.suffix}) "
                        f"not supported by registry. Available: {list(DATA_LOADER_REGISTRY.keys())}"
                    )

            # 获取数据；原文件模式的加载和检查分支保留
            try:
                if not config.get("pre_hooks"):
                    data = loader(file_path)
                loaded_data[path_key] = data
                file_load_msgs.append(f"{path_key}: loaded from {file_path}")

                # ------------------ 形状检查 ------------------
                current_index = CHECKPOINT_NAMES.index("dataset_schema:shape_check")
                shape_spec = file_spec.get("shape")
                if shape_spec and hasattr(data, 'shape'):
                    shape_ok = True
                    shape_details = []

                    # 检查维度数
                    if "ndim" in shape_spec:
                        if data.ndim != shape_spec["ndim"]:
                            shape_ok = False
                            shape_details.append(f"ndim: expected {shape_spec['ndim']}, got {data.ndim}")

                    # 检查具体维度
                    if "dims" in shape_spec:
                        expected_dims = shape_spec["dims"]
                        if len(expected_dims) != data.ndim:
                            shape_ok = False
                            shape_details.append(
                                f"dimensions count mismatch: expected {len(expected_dims)}, got {data.ndim}"
                            )
                        else:
                            for i, expected_dim in enumerate(expected_dims):
                                if expected_dim is not None and data.shape[i] != expected_dim:
                                    shape_ok = False
                                    shape_details.append(f"dim[{i}]: expected {expected_dim}, got {data.shape[i]}")

                    if not shape_ok:
                        raise ValueError(
                            f"Shape validation failed for {path_key}: {', '.join(shape_details)}. "
                            f"Actual shape: {data.shape}"
                        )

                    shape_check_msgs.append(f"{path_key}: shape {data.shape} validated")

                if "dtype" in file_spec:
                    current_index = CHECKPOINT_NAMES.index("dataset_schema:dtype_check")
                    expected_dtype = getattr(torch, file_spec["dtype"])
                    if data.dtype != expected_dtype:
                        raise ValueError(f"dtype mismatch for {path_key}: expected {expected_dtype}, got {data.dtype}")
                    dtype_check_msgs.append(f"{path_key}: dtype {data.dtype} validated")

                # ------------------ 字段检查（针对结构化数据） ------------------
                current_index = CHECKPOINT_NAMES.index("dataset_schema:field_check")
                fields_spec = file_spec.get("fields")
                if fields_spec:
                    if not isinstance(fields_spec, list):
                        raise TypeError(f"fields for {path_key} must be a list, got {type(fields_spec)}")

                    # 检查是否为结构化数组或字典
                    if isinstance(data, np.ndarray) and data.dtype.names:
                        # numpy 结构化数组
                        missing_fields = [f for f in fields_spec if f not in data.dtype.names]
                        if missing_fields:
                            raise ValueError(
                                f"Fields missing in {path_key}: {missing_fields}. "
                                f"Available fields: {data.dtype.names}"
                            )
                        field_check_msgs.append(f"{path_key}: fields {fields_spec} validated")

                    elif isinstance(data, dict):
                        # 字典数据
                        missing_fields = [f for f in fields_spec if f not in data]
                        if missing_fields:
                            raise ValueError(
                                f"Fields missing in {path_key}: {missing_fields}. "
                                f"Available fields: {list(data.keys())}"
                            )
                        field_check_msgs.append(f"{path_key}: fields {fields_spec} validated")

                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        # 列表字典数据
                        available_fields = set().union(*[item.keys() for item in data])
                        missing_fields = [f for f in fields_spec if f not in available_fields]
                        if missing_fields:
                            raise ValueError(
                                f"Fields missing in {path_key}: {missing_fields}. "
                                f"Available fields: {sorted(available_fields)}"
                            )
                        field_check_msgs.append(f"{path_key}: fields {fields_spec} validated (list of dicts)")

                    elif hasattr(data, '__getitem__') and hasattr(data, 'keys'):
                        # 其他类似字典的对象
                        try:
                            missing_fields = [f for f in fields_spec if f not in data]
                            if missing_fields:
                                raise ValueError(
                                    f"Fields missing in {path_key}: {missing_fields}. "
                                    f"Available fields: {list(data.keys())}"
                                )
                            field_check_msgs.append(f"{path_key}: fields {fields_spec} validated")
                        except (AttributeError, TypeError):
                            # 如果无法检查字段，跳过
                            field_check_msgs.append(f"{path_key}: field validation skipped (unsupported data type)")

            except Exception as e:
                raise RuntimeError(f"Failed to load/validate {path_key} from {file_path}: {e}")

        # ================================================================
        # 4. 更新所有检查点
        # ================================================================

        # 检查点1: file_load / data_load
        current_index = 0
        update_check(
            current_index,
            True,
            f"All data loaded: {'; '.join(file_load_msgs)}"
        )
        current_index += 1

        # 检查点2: shape_check
        if shape_check_msgs:
            update_check(current_index, True, f"Shape validation passed: {'; '.join(shape_check_msgs)}")
        else:
            update_check(current_index, True, "No shape validation required")
        current_index += 1

        # 检查点3: field_check
        if field_check_msgs:
            update_check(current_index, True, f"Field validation passed: {'; '.join(field_check_msgs)}")
        else:
            update_check(current_index, True, "No field validation required")

        if any("dtype" in spec for spec in schema_config):
            current_index += 1
            update_check(current_index, True, "; ".join(dtype_check_msgs))

    except Exception as e:
        error_detail = traceback.format_exc()
        # 只保留已经完整完成的汇总；不把 shape/dtype/fields 错误误记成加载错误。
        if current_index != 0 and len(loaded_data) == len(schema_config):
            update_check(0, True, f"All data loaded: {'; '.join(file_load_msgs)}")
        if current_index not in (0, 1) and shape_check_msgs and len(shape_check_msgs) == sum(bool(spec.get("shape")) for spec in schema_config):
            update_check(1, True, f"Shape validation passed: {'; '.join(shape_check_msgs)}")
        # 更新实际失败的检查点；其他未完成汇总仍保留默认 False。
        update_check(current_index, False, error_detail)

    return checks


def check_parameter_update(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    静态检查参数更新调用；支持配置训练入口的外部 optimizer 形参，不证明参数实际变化。

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

            # 外部优化器：只为指定训练入口补充形参使用证据；保留原内部构造识别。
            train_fn_name = config.get("train_fn_name")
            optimizer_param_index = config.get("optimizer_param_index", 3)
            for function in tree.body:
                if isinstance(function, ast.FunctionDef) and function.name == train_fn_name:
                    arguments = function.args.posonlyargs + function.args.args
                    if not 0 <= optimizer_param_index < len(arguments):
                        raise ValueError("optimizer_param_index is outside the configured function signature")
                    optimizer_name = arguments[optimizer_param_index].arg
                    has_optimizer = has_optimizer or any(
                        isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                        and isinstance(item.func.value, ast.Name) and item.func.value.id == optimizer_name
                        and item.func.attr in {"zero_grad", "step"}
                        for item in ast.walk(function)
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
    运行一次 train_step，并检查该次返回 Loss 有限、同模型参数前后发生变化。

    config 示例:
    {
        "run_file": "train.py",           # 训练模块文件
        "config_name": "config.yaml",          # 可选；指定后加载失败不回退
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
        - training:smoke_test: 检查 train_step 是否正常返回
        - train_step:finite_loss: 返回值有限，不比较 masked MSE 公式
        - train_step:parameters_updated: 至少一个参数变化，不检查梯度数值
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "training:create_work_dir",
        "training:smoke_test",
        "train_step:finite_loss",
        "train_step:parameters_updated",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    # 入口/配置属于冒烟准备；只有实际创建目录时才更新目录结果。
    current_index = 1

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
        current_index = 0
        # work_dir = Path(tempfile.mkdtemp(prefix="training-smoke")).resolve()
        work_dir = (CANDIDATE_DIR / "smoke-training").resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = work_dir / "checkpoints"
        results_dir = work_dir / "results"

        update_check(current_index, True, f"Work directory created: {work_dir}")

        # ------------------ 4. 加载配置 ------------------
        current_index += 1
        config_name = config.get("config_name")
        # 加载完整配置（与 check_training 一致）
        candidate_config = _load_candidate_config(config_name) if config_name else {}
        runtime_config = deepcopy(candidate_config)

        # ------------------ 5. 构造冒烟配置（轻量覆盖） ------------------

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

            source_module = (
                torch
                if func_name in ("randn", "randn_like", "tensor")
                else train_module
            )
            hook_func = getattr(source_module, func_name, None)
            if hook_func is None or not callable(hook_func):
                raise ValueError(f"Pre-hook {idx}: function '{func_name}' not found or not callable")

            resolved_args = _resolve_args(args_config, context)
            hook_result = hook_func(**resolved_args)

            save_as = hook_config.get("save_as")
            if save_as:
                context[save_as] = hook_result
                if isinstance(runtime_config, dict):
                    runtime_config[save_as] = hook_result

        # 本 Case 的 optimizer 由调用方提供，在模型迁移后创建。
        if "optimizer_args" in config:
            context["model"] = context["model"].to(DEVICE)
            context["optimizer"] = torch.optim.AdamW(
                context["model"].parameters(), **config["optimizer_args"]
            )

        # ------------------ 7. 执行冒烟测试 ------------------
        # 解析训练函数参数（与 check_training 一致）
        train_args_config = config.get("train_args", {})
        resolved_train_args = _resolve_args(train_args_config, context)

        # 保存本次训练模型的参数；只在下面执行一次 train_fn。
        model = context["model"]
        before = [p.detach().clone() for p in model.parameters()]

        # 执行训练
        if config.get("args_as_positional", False):
            result = train_fn(*resolved_train_args.values())
        else:
            result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)

        update_check(
            current_index,
            True,
            f"Smoke test completed successfully: {train_fn_name}(...) with {config_name}"
        )

        # ------------------ 8. 检查本次训练返回值 ------------------
        current_index += 1
        finite = bool(torch.isfinite(torch.as_tensor(result)))
        update_check(
            current_index,
            finite,
            f"loss={float(result):.8f}"
        )

        # ------------------ 9. 检查本次训练的参数变化 ------------------
        current_index += 1
        changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        update_check(
            current_index,
            changed,
            f"train_step returned; changed={changed}; no masked-MSE value or gradient-value comparison"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks
