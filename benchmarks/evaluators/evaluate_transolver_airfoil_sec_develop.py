#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Transolver AirfRANS 二次开发评测（OnescienceBench）。

check_files/check_interface 原名直接导入；同名方法以当前 Base 原文为底稿。
模型类名和构造形参由 JSON 明确绑定，不维护自动别名解析器。
仅要求 Prompt 的 N×7、N×2 → N×4；不与二次开发 Standard 比较全模型输出。
原始训练基线仅用于 dataset 和 backbone 注册结构保持检查，不作新 Head 答案。
check_smoke_training 沿用 Base 主流程，候选 train_epoch 执行反向/更新；保留两组 Loss 数值检查。
不运行 398 epochs/完整 VTK 数据管线；接口未接入的 REVIEW_REQUIRED 不能归为算法错误。
"""
from __future__ import annotations

import ast
import inspect
import json
import math
import sys
import traceback
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import yaml
from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _repo_root,
    _load_candidate_modules,
    _load_candidate_config,
    _load_module_from_path,
    _load_class,
    _resolve_args,
    _execute_hooks,
    _convert_arg_to_type,
    _move_tensors_to_device,
    _compute_mse,
    check_files,
    check_interface,
)


def resolve_airfoil_roles(model: nn.Module, role_paths: dict) -> dict:
    """使用显式候选路径定位角色，不按待测通道数/层数择优；未知布局需接入。"""
    resolved = {}
    for role, paths in role_paths.items():
        found = {}
        for path in paths:
            try:
                module = model.get_submodule(path)
            except AttributeError:
                continue
            found.setdefault(id(module), path)
        if len(found) != 1:
            raise ValueError(f"REVIEW_REQUIRED: role {role!r} requires one binding: {list(found.values())}")
        resolved[role] = next(iter(found.values()))
    # 只在已绑定的 Head/输入 MLP 内找末层/首层 Linear；不会选全模型无关模块。
    for role in ("velocity", "pressure", "viscosity", "input"):
        if role not in resolved:
            continue
        path = resolved[role]
        linears = [(name, module) for name, module in model.get_submodule(path).named_modules()
                   if isinstance(module, nn.Linear)]
        if not linears:
            raise ValueError(f"REVIEW_REQUIRED: {role} has no registered Linear projection")
        name, _ = linears[0 if role == "input" else -1]
        resolved[role + "_linear"] = path + ("." + name if name else "")
    return resolved

def airfoil_tensor(part: str, points: int = 32, seed: int = 20260819) -> torch.Tensor:
    """公共 hooks 的小输入；只使用 Prompt 规定的二维接口及参考网格范围。"""
    generator = torch.Generator().manual_seed(seed)
    pos = torch.rand(points, 2, generator=generator) * torch.tensor([6.0, 3.0]) - torch.tensor([2.0, 1.5])
    x = torch.randn(points, 7, generator=generator)
    x[:, :2] = pos
    target = torch.randn(points, 4, generator=generator)
    surface = torch.arange(points) < max(1, points // 3)
    return _move_tensors_to_device({"x": x, "pos": pos, "target": target, "surface": surface}[part])

def prepare_airfoil_smoke(config: dict, model_file: str, model_class: str,
                          model_args: dict, dataset_file: str, sample_class: str,
                          weight_paths: dict, points: int = 32,
                          runtime_dir=None, runtime_overrides=None) -> dict:
    """准备训练输入/对象；路径入口仅写临时配置，数据构造在调用边界注入小样本。"""
    module = _load_candidate_modules(model_file)
    model = _load_class(module, model_class)(**model_args).to(DEVICE)
    x, pos, target, surface = [airfoil_tensor(part, points).cpu() for part in ("x", "pos", "target", "surface")]
    sample = {"x": x, "pos": pos, "y": target, "surf": surface}
    if sample_class is not None:
        data_cls = _load_class(_load_candidate_modules(dataset_file), sample_class)
        sample = data_cls(**sample)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=2)

    # 保留 v3 的两组权重；未声明 volume 权重时沿用系数 1。
    paths_by_field = {}
    for field, paths in weight_paths.items():
        found = [(path, _resolve_args({"v": "$config." + path}, {"config": config})["v"]) for path in paths]
        found = [(path, value) for path, value in found if value is not None]
        if field == "volume" and not found:
            continue
        if len(found) != 1 or isinstance(found[0][1], bool) or not isinstance(found[0][1], (int, float)) or not math.isfinite(found[0][1]):
            raise ValueError(f"REVIEW_REQUIRED: bind exactly one finite numeric weight for {field}: {found}")
        paths_by_field[field] = found[0][0]
    original_weights = {"volume": 1.0, **{
        key: _resolve_args({"v": "$config." + path}, {"config": config})["v"]
        for key, path in paths_by_field.items()}}
    changed_weights = dict(original_weights, velocity=0.7, pressure=2.3, viscosity=1.1, surface=1.7)
    if "volume" in paths_by_field:
        changed_weights["volume"] = 0.4
    probes = []
    for weights in (original_weights, changed_weights):
        runtime_config = deepcopy(config)
        for key, path in paths_by_field.items():
            parts = path.split(".")
            destination = runtime_config
            for part in parts[:-1]:
                destination = destination[part]
            destination[parts[-1]] = weights[key]
        probes.append((runtime_config, weights))
    probe = {
        "model": model,
        "dataset": [sample],
        "optimizer": optimizer,
        "scheduler": scheduler,
        "x": x,
        "pos": pos,
        "target": target,
        "surface": surface,
        "loss_probes": probes,
        "training_config": probes[-1][0],
    }

    if runtime_dir is not None:
        directory = Path(runtime_dir)
        directory.mkdir(parents=True, exist_ok=True)
        runtime_config = deepcopy(probes[-1][0])
        for path, value in (runtime_overrides or {}).items():
            parts, destination = path.split("."), runtime_config
            for part in parts[:-1]:
                destination = destination[part]
            destination[parts[-1]] = str(value) if isinstance(value, Path) else value
        config_path = directory / "airfoil_smoke_config.yaml"
        config_path.write_text(yaml.safe_dump(runtime_config, allow_unicode=True), encoding="utf-8")
        probe["config_path"] = str(config_path)
    return probe


def check_config_content(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple[str, bool, str]]:
    """
    检查训练 YAML 配置文件是否存在且有效。
    config_paths 仅接入 JSON 中明确列出的嵌套路径；字段比较沿用 Base。

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
            raw_config = config_data
            config_data = dict(raw_config)
            for key, paths in config["config_paths"].items():
                values = [_resolve_args({"v": "$config." + path}, {"config": raw_config})["v"]
                          for path in paths]
                values = [value for value in values if value is not None]
                if values and all(value == values[0] for value in values):
                    config_data[key] = values[0]
                else:
                    config_data.pop(key, None)


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

def check_dataset_pipeline(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """检查不可修改的 dataset.py 与显式原始训练基线一致。

    不用二次开发 Standard 冒充原始输入，也不把文件相同说成 VTK 数据行为已验证。
    baseline_dir 必须先由实际原始代码核对；缺少基线时 REVIEW_REQUIRED，不自比较。
    """
    CHECKPOINT_NAMES = ["data:source_parse", "data:dataset_unchanged"]
    checks = [(name, False, "Not executed") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        path = CANDIDATE_DIR / config["dataset_file"]
        source = path.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(path))
        update_check(0, True, "Candidate dataset.py parses; no VTK files loaded")
        current_index = 1
        reference_dir = (_repo_root() / config["baseline_dir"]).resolve()
        reference = reference_dir / config["dataset_file"]
        if reference_dir == CANDIDATE_DIR.resolve() or not reference.is_file():
            raise ValueError("REVIEW_REQUIRED: original training dataset baseline is absent or self-referential")
        original = reference.read_text(encoding="utf-8-sig")
        update_check(1, source == original,
                     f"Dataset source equals original training baseline (BOM/newline normalized): {reference}; "
                     "this proves preservation, not independent correctness of the baseline")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: dict, caller_globals=None, caller_locals=None):
    """
    检查模型的结构是否符合预期。
    model_class/model_args 使用实际候选接口；不自动猜测类名或构造形参。
    role_paths 只定位待测模块；backbone_reference 排除明确允许替换的输出尾部。
    公共 rules 循环不变；注册结构相同不证明全部前向数学语义。

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
                    "severity": "error",
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
                    "severity": "error",
                    "params": {
                        "target_path": "blocks",
                        "length": 3
                    }
                },
                {
                    "id": "block_0_pattern",
                    "type": "block_pattern",
                    "severity": "error",
                    "params": {
                        "target_path": "blocks.0",
                        "pattern": ["LayerNorm", "TestAttention", "LayerNorm", "Sequential"]
                    }
                },
                {
                    "id": "attn_out_proj",
                    "type": "linear_shape",
                    "severity": "error",
                    "params": {
                        "target_path": "blocks.0.attn.out_proj",
                        "in_features": 64,
                        "out_features": 64
                    }
                },
                {
                    "id": "top_level_children",
                    "type": "children_count",
                    "severity": "error",
                    "params": {
                        "target_path": "",
                        "count": 6
                    }
                }
            ]
    }

    检查点:
        - model_structure:build_model_ok: 检查能否成功构建模型
        - model_structure:<rule id>[:<path>]: 公共规则结果
        - model_structure:backbone_reference: 原始主干绑定失败时的复核记录
        - model_structure:block_components: 原有逐 Block 组成证据
        - model_structure:structure_execution: 构建成功后结构执行异常
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
        if not model_args:
            raise ValueError("config['model_args'] is required")

        resolved_model_args = _resolve_args(model_args, context)
        model = model_cls(**resolved_model_args).to(DEVICE).eval()

        update_check(
            current_index,
            True,
            f"Model built with config: {candidate_config}"
        )

        from base.model_structure_checker import (
            BuiltinRules,
            ModuleInfoExtractor,
            ModelStructureChecker,
            StructureLearner,
        )
        checker = ModelStructureChecker()

        rules = config.get("rules")
        if rules is None:
            raise ValueError("config['rules'] is missing")
        roles = resolve_airfoil_roles(model, config["role_paths"]) if config.get("role_paths") else {}
        rules = _resolve_args({"rules": rules}, roles)["rules"]
        # 对齐原始训练基线的 backbone，而不是二次开发 Standard 的全模型/Head。
        reference_spec = config.get("backbone_reference")
        if reference_spec:
            try:
                reference_dir = (_repo_root() / reference_spec["directory"]).resolve()
                if reference_dir == CANDIDATE_DIR.resolve():
                    raise ValueError("Original backbone baseline cannot be the candidate itself")
                reference_module = _load_module_from_path("airfoil_original_backbone", reference_dir / config["model_file"])
                ref_cls = _load_class(reference_module, reference_spec.get("model_class", model_class_name))
                ref_args = _resolve_args(reference_spec.get("model_args", model_args), context)
                reference_model = ref_cls(**ref_args).to(DEVICE).eval()
                reference_roles = resolve_airfoil_roles(reference_model, reference_spec["role_paths"])
                for role in ("input", "blocks"):
                    info = ModuleInfoExtractor(reference_model.get_submodule(reference_roles[role])).extract()
                    structure = StructureLearner._info_to_structure(info)
                    tail = reference_spec.get("replaceable_output_tail")
                    if role == "blocks" and tail:
                        block = structure["children"][tail["block_index"]]
                        names = tail["children"]
                        if not names or [item.get("name") for item in block["children"][-len(names):]] != names:
                            raise ValueError("REVIEW_REQUIRED: original output-tail binding no longer matches")
                        block["children"] = block["children"][:-len(names)]
                        block["exact"] = False  # Base 原生前缀结构：保留核心，允许输出尾部迁出或留在该块内。
                    candidate_info = ModuleInfoExtractor(model.get_submodule(roles[role])).extract()
                    positional_structure = deepcopy(structure)
                    stack = [(structure, candidate_info), (positional_structure, None)]
                    while stack:
                        item, actual = stack.pop()
                        expected_children = item.get("children", [])
                        actual_children = actual.children if actual is not None else []
                        # 同名子层只作定位，不把普通自定义模块的登记顺序直接当作 forward 调用顺序。
                        # Sequential/ModuleList 等有序容器不重排，原输出尾部的边界不变。
                        if (item.get("module_type") == "custom" and actual is not None
                                and actual.type.value == "custom"):
                            by_name = {child.get("name"): child for child in expected_children}
                            names = [child.registered_name for child in actual_children[:len(expected_children)]]
                            if (None not in by_name and len(by_name) == len(expected_children)
                                    and len(names) == len(expected_children) and set(names) == set(by_name)):
                                item["children"] = expected_children = [by_name[name] for name in names]
                        item.pop("name", None)  # private registration names are not the task contract
                        if item.get("module_type") == "custom":
                            item.pop("class_name", None)
                        stack.extend((child, actual_children[index] if index < len(actual_children) else None)
                                     for index, child in enumerate(expected_children))
                    # 保留原有的无名称约束比较；同名定位不能反而拒绝已支持的合法改名。
                    if BuiltinRules.check_structure(candidate_info, {"structure": positional_structure})[0]:
                        structure = positional_structure
                    rules.append({"id": "preserved_" + role, "type": "structure", "severity": "error",
                                  "params": {"target_path": roles[role], "structure": structure}})
            except Exception:
                checks.append(("model_structure:backbone_reference", False,
                               "REVIEW_REQUIRED: verify original backbone source and non-Head scope\n" + traceback.format_exc()))
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

        if roles:
            blocks = list(model.get_submodule(roles["blocks"]).children())
            component_evidence = []
            for block in blocks:
                modules = list(block.modules())
                has_attention = any("attention" in type(m).__name__.lower() for m in modules)
                has_norm = any(isinstance(m, nn.LayerNorm) for m in modules)
                has_mlp = any(isinstance(m, nn.Linear) for m in modules)
                heads = [getattr(m, name) for m in modules for name in ("heads", "num_heads", "n_heads", "n_head")
                         if isinstance(getattr(m, name, None), int)]
                component_evidence.append(has_attention and has_norm and has_mlp and 8 in heads)
            checks.append(("model_structure:block_components", bool(blocks) and all(component_evidence),
                           f"Per-block attention/LayerNorm/MLP/8-head registration evidence={component_evidence}; "
                           "not a full Physics-Attention mathematical proof"))
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
    保留 Base 的加载、hooks、构造、参数转换与前向流程。
    role_paths 启用原有 Head 顺序、Block 调用、参考网格检查；check_nan_inf 检查有限值。

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
        - forward_executable:finite: check_nan_inf 启用时检查有限值
        - forward_executable:task_head_order: role_paths 启用时检查三 Head 输出顺序
        - forward_executable:backbone_calls: role_paths 启用时观察八个 Block 调用
        - forward_executable:reference_grid_distances: role_paths 启用时核对参考网格距离
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
    ]
    if config.get("check_nan_inf", False):
        CHECKPOINT_NAMES.append("forward_executable:finite")
    if config.get("role_paths"):
        CHECKPOINT_NAMES.extend([
            "forward_executable:task_head_order",
            "forward_executable:backbone_calls",
            "forward_executable:reference_grid_distances",
        ])
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    # ================================================================
    # 主流程
    # ================================================================
    current_index = 0
    handles = []

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

        observed = {}
        roles = resolve_airfoil_roles(model, config["role_paths"]) if config.get("role_paths") else {}
        for role in ("velocity", "pressure", "viscosity"):
            if role in roles:
                handles.append(model.get_submodule(roles[role]).register_forward_hook(
                    lambda module, args, output, role=role: observed.__setitem__(role, output.detach().clone())
                    if isinstance(output, torch.Tensor) else None))
        if "input_linear" in roles:
            handles.append(model.get_submodule(roles["input_linear"]).register_forward_pre_hook(
                lambda module, args: observed.__setitem__("input", args[0].detach().clone())))
        block_calls = set()
        if "blocks" in roles:
            for index, block in enumerate(model.get_submodule(roles["blocks"]).children()):
                handles.append(block.register_forward_hook(
                    lambda module, args, output, index=index: block_calls.add(index)))
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

        same_shape = (tuple(out.shape) == tuple(config.get("out_shape", [])))
        update_check(
            current_index,
            same_shape,
            f"Output is Tensor with shape={tuple(out.shape)}, expected out_shape={config.get('out_shape', [])}"
        )

        if config.get("check_nan_inf", False):
            current_index += 1
            update_check(current_index, bool(torch.isfinite(out).all()), "Finite candidate output")
        if roles:
            current_index += 1
            tasks = ("velocity", "pressure", "viscosity")
            ordered = all(role in observed for role in tasks)
            if ordered:
                pieces = [observed[role].reshape(-1, width) for role, width in zip(tasks, (2, 1, 1))]
                concatenated = torch.cat(pieces, dim=-1)
                ordered = concatenated.shape == out.shape and bool(torch.allclose(out, concatenated))
            update_check(current_index, bool(ordered),
                         "Executed Head outputs concatenate as [ux,uy,p,nut]; no Head architecture equality required")
            current_index += 1
            update_check(current_index, len(block_calls) == 8,
                         f"Executed backbone block indices={sorted(block_calls)}")
            current_index += 1
            x, pos = converted_forward_args["x"], converted_forward_args["pos"]
            gx, gy = torch.meshgrid(torch.linspace(-2, 4, 8, device=DEVICE),
                                   torch.linspace(-1.5, 1.5, 8, device=DEVICE), indexing="ij")
            # 小距离的 float32 cdist 参考存在抵消误差；提高参考计算精度，不放宽判定容差。
            expected_dist = torch.cdist(pos.double(), torch.stack((gx.flatten(), gy.flatten()), dim=-1).double()).to(pos.dtype)
            actual = observed.get("input")
            encoded = False
            if actual is not None and actual.numel() == x.shape[0] * 71:
                actual = actual.reshape(x.shape[0], 71)
                for features, distances in ((actual[:, :7], actual[:, 7:]), (actual[:, -7:], actual[:, :-7])):
                    encoded = encoded or (bool(torch.allclose(features, x, atol=1e-5, rtol=1e-5)) and
                        bool(torch.allclose(distances.sort(-1).values, expected_dist.sort(-1).values, atol=1e-5, rtol=1e-5)))
            update_check(current_index, bool(encoded),
                         "Observed input projection contains original 7 features and 64 Euclidean reference-grid distances; grid column order unrestricted")
    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    finally:
        for handle in handles:
            handle.remove()
    return checks

def check_train_structure(config: dict) -> list[tuple]:
    """
    静态检查训练脚本是否具备基本的训练启动逻辑。
    Base 原八项保持原顺序；airfoil_contract 只在末尾追加原有四项任务证据。
    唯一源码读取适配为 utf-8-sig，兼容合法 UTF-8 BOM。

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
        airfoil_contract 启用时另有：train:positional_config_cli、train:onecycle_step、
        train:resampled_validation、train:checkpoint_history；均为原有有限静态证据。
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

        if config.get("airfoil_contract", False):
            calls = [node for node in all_nodes if isinstance(node, ast.Call)]
            parser_positional = any(isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                                    and node.args and isinstance(node.args[0], ast.Constant)
                                    and isinstance(node.args[0].value, str) and not node.args[0].value.startswith("-")
                                    for node in calls)
            positional_argv = any(isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                                  and node.value.attr == "argv" and isinstance(node.slice, ast.Constant)
                                  and node.slice.value == 1 for node in all_nodes)
            checks.append(("train:positional_config_cli", parser_positional or positional_argv,
                           "Static evidence for python train.py config.yaml; a --config-only CLI is not substituted"))
            scheduler_names = {target.id for node in all_nodes if isinstance(node, ast.Assign)
                               and isinstance(node.value, ast.Call)
                               and getattr(node.value.func, "attr", getattr(node.value.func, "id", "")) == "OneCycleLR"
                               for target in node.targets if isinstance(target, ast.Name)}
            scheduler_step = any(isinstance(node.func, ast.Attribute) and node.func.attr == "step"
                                 and isinstance(node.func.value, ast.Name) and node.func.value.id in scheduler_names
                                 for node in calls)
            checks.append(("train:onecycle_step", bool(scheduler_names) and scheduler_step,
                           "OneCycleLR constructor and same-object step call; not a complete scheduling proof"))
            identifiers = {node.id.lower() for node in all_nodes if isinstance(node, ast.Name)}
            identifiers.update(node.name.lower() for node in all_nodes if isinstance(node, ast.FunctionDef))
            validation = any("valid" in name or name.startswith("val_") for name in identifiers)
            repeat = any("repeat" in name for name in identifiers)
            disabled_grad = any(getattr(node.func, "attr", "") in ("no_grad", "inference_mode") for node in calls)
            checks.append(("train:resampled_validation", validation and repeat and disabled_grad,
                           "Finite static evidence of repeated validation; 10/20 defaults checked in configuration, not full 398-epoch execution"))
            save = any(getattr(node.func, "attr", "") == "save" for node in calls)
            history = any("history" in name for name in identifiers) or any(
                isinstance(call.func, ast.Attribute) and call.func.attr == "dump"
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "json"
                and call.args and isinstance(call.args[0], ast.Name)
                and call.args[0].id in train_check_config.get("history_variables", [])
                for call in calls
            )
            checks.append(("train:checkpoint_history", save and history,
                           "Checkpoint/history save declarations; no fixed filename or normalization-state artifact requirement"))

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
    airfoil_probe 可选：前置 hook 准备小样本，候选训练函数执行更新；
    调用后保留加权 Loss 数值、实际梯度和参数变化检查，不代做 backward/step。

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
        - training:weighted_loss: 可选，两组 Loss 数值及真实训练输出梯度的加权公式检查
        - training:finite_gradients: 可选，候选训练过程中观察到有限梯度
        - training:parameter_update: 可选，候选训练实际改变模型参数
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "training:create_work_dir",
        "training:smoke_test",
    ]
    if config.get("airfoil_probe"):
        CHECKPOINT_NAMES.extend([
            "training:weighted_loss",
            "training:finite_gradients",
            "training:parameter_update",
        ])
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    gradient_handles = []
    observed_gradients = []
    loss_backward_records = []
    loss_observation_errors = []
    entry_context = ExitStack()

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
        if not callable(train_fn) and config.get("entry_adapters"):
            # 只选择 JSON 明确接入的现有入口；不按试运行结果重试，不修改候选函数。
            names = [name for name in config["entry_adapters"] if callable(getattr(train_module, name, None))]
            if len(names) != 1:
                raise ValueError(f"REVIEW_REQUIRED: bind one native training entry: {names}")
            train_fn_name = names[0]
            config = {**config, **config["entry_adapters"][train_fn_name]}
            train_fn = getattr(train_module, train_fn_name)
        if train_fn is None or not callable(train_fn):
            raise ImportError(f"REVIEW_REQUIRED: Train function '{train_fn_name}' not found or not callable in {train_file}")

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
        candidate_config = _load_candidate_config(config_name)
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
            # 复用公共 hook 查找；仍逐项执行并同步 save_as 到运行配置。
            context = _execute_hooks(
                train_module, [hook_config], context,
                globals(), caller_locals,
            )
            save_as = hook_config.get("save_as")
            if save_as and isinstance(runtime_config, dict):
                runtime_config[save_as] = context[save_as]

        if config.get("airfoil_probe"):
            probe = context[config["airfoil_probe"]]

            def observe_training_output(module, args, kwargs, output):
                # 只观察本次小样本训练的输出梯度，不替换 Loss 或代做反向。
                if not torch.is_grad_enabled() or not isinstance(output, torch.Tensor) or not output.requires_grad:
                    return
                try:
                    batch_x = (args[0] if args else kwargs["x"]).detach().reshape(-1, 7)
                    sample_x = probe["x"].to(batch_x.device)
                    matches = (batch_x[:, None, :] == sample_x[None, :, :]).all(dim=-1)
                    if not bool((matches.sum(dim=-1) == 1).all()):
                        raise ValueError("training input cannot be mapped to the supplied smoke sample")
                    indices = matches.to(torch.int64).argmax(dim=-1)
                    target = probe["target"].to(output.device)[indices]
                    surface = probe["surface"].to(output.device)[indices].bool()
                    prediction = output.detach().reshape(-1, 4)
                    if prediction.shape != target.shape:
                        raise ValueError("training output is not aligned with the supplied [N,4] target")
                    weights = probe["loss_probes"][-1][1]
                    expected = torch.zeros_like(prediction)
                    # 对原六项区域/字段 MSE 求解析梯度；支持采样顺序变化和 B=1 包装。
                    for region, mask in (("surface", surface), ("volume", ~surface)):
                        for field, columns in (("velocity", slice(0, 2)), ("pressure", slice(2, 3)),
                                               ("viscosity", slice(3, 4))):
                            difference = prediction[mask, columns] - target[mask, columns]
                            if difference.numel():
                                expected[mask, columns] = (2 * weights[region] * weights[field]
                                                           / difference.numel()) * difference
                    record = {"expected": expected, "actual": None}
                    loss_backward_records.append(record)

                    def observe_output_gradient(gradient):
                        actual = gradient.detach().reshape(-1, 4).clone()
                        record["actual"] = actual if record["actual"] is None else record["actual"] + actual
                        # 返回 None：不改变候选梯度；同一输出的多次反向按累计梯度比较。

                    gradient_handles.append(output.register_hook(observe_output_gradient))
                except Exception:
                    loss_observation_errors.append("REVIEW_REQUIRED: smoke input/output binding\n" + traceback.format_exc())

            def observe_model(model):
                probe["model"] = model
                gradient_handles.append(model.register_forward_hook(observe_training_output, with_kwargs=True))
                probe["before"] = [parameter.detach().clone() for parameter in model.parameters()]
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        gradient_handles.append(parameter.register_hook(
                            lambda gradient: observed_gradients.append(bool(torch.isfinite(gradient).all()))
                        ))
                return model

            if config.get("native_entry"):
                binding = config["native_entry"]
                dataset_class = getattr(train_module, binding["dataset_class"], None)
                factory = getattr(train_module, binding["model_factory"], None)
                if not callable(dataset_class) or not callable(factory):
                    raise ValueError("REVIEW_REQUIRED: native dataset constructor/model factory is not bound")

                def observed_factory(*args, **kwargs):
                    return observe_model(factory(*args, **kwargs))

                # 仅注入测试数据；真实模型工厂、DataLoader、Loss、反向、更新仍由候选执行。
                entry_context.enter_context(patch.object(train_module, binding["dataset_class"],
                                                         lambda *args, **kwargs: probe["dataset"]))
                entry_context.enter_context(patch.object(train_module, binding["model_factory"], observed_factory))
            else:
                observe_model(probe["model"])

        # ------------------ 7. 执行冒烟测试 ------------------
        # 解析训练函数参数（与 check_training 一致）
        train_args_config = config.get("train_args", {})
        resolved_train_args = _resolve_args(train_args_config, context)

        # 执行训练
        result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)

        update_check(
            current_index,
            True,
            f"Smoke test completed successfully: {train_fn_name}(...) with {config_name}"
        )

        if config.get("airfoil_probe"):
            current_index += 1
            if "before" not in probe:
                raise ValueError("Candidate training did not construct the configured model")
            model, before = probe["model"], probe["before"]
            loss_name = config["loss_function"]
            loss_fn = getattr(train_module, loss_name, None)
            if not callable(loss_fn):
                raise ValueError(f"REVIEW_REQUIRED: bind the actual candidate loss function instead of {loss_name!r}")
            fields = {"velocity": slice(0, 2), "pressure": slice(2, 3), "viscosity": slice(3, 4)}
            x, pos, target, surface = _move_tensors_to_device(
                [probe[part] for part in ("x", "pos", "target", "surface")]
            )
            numerical = []
            # 只调用候选模型/Loss 作数值对照；不在评测器中反向或更新。
            with torch.no_grad():
                for loss_config, weights in probe["loss_probes"]:
                    prediction = model(x, pos)
                    if not isinstance(prediction, torch.Tensor) or prediction.shape != target.shape:
                        raise ValueError("Candidate output must have shape [N,4]")
                    loss_context = {"config": loss_config, "prediction": prediction, "target": target, "surface": surface}
                    args = _resolve_args(config["loss_args"], loss_context)
                    result = loss_fn(*args.values()) if config.get("loss_args_as_positional", False) else loss_fn(**args)
                    loss = result[config.get("loss_result_index", 0)] if isinstance(result, (tuple, list)) else result
                    if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
                        raise TypeError("Actual candidate loss must be a finite scalar Tensor; configure its result index if necessary")
                    expected = sum(weights[region] * weights[field] * _compute_mse(prediction[mask, indices], target[mask, indices])
                                   for region, mask in (("surface", surface), ("volume", ~surface))
                                   for field, indices in fields.items())
                    numerical.append(math.isclose(float(loss.detach()), expected, rel_tol=1e-5, abs_tol=1e-6))
            backward_matches = [
                bool(torch.allclose(record["actual"], record["expected"], rtol=1e-5, atol=1e-6))
                for record in loss_backward_records if record["actual"] is not None
            ]
            update_check(current_index,
                         all(numerical) and bool(backward_matches) and all(backward_matches) and not loss_observation_errors,
                         f"Actual weighted MSE probes={numerical}; training_output_gradient_matches={backward_matches}; "
                         f"observation_errors={loss_observation_errors}")
            current_index += 1
            update_check(current_index, bool(observed_gradients) and all(observed_gradients),
                         f"Candidate {train_fn_name} gradient observations={len(observed_gradients)}; no required gradient fraction")
            current_index += 1
            changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters()))
            update_check(current_index, changed, f"Candidate {train_fn_name} changed model parameters; evaluator did not call backward/step")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)
    finally:
        entry_context.close()
        for handle in gradient_handles:
            handle.remove()

    return checks
