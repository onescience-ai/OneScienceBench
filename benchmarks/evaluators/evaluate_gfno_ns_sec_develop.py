"""GFNO NS：Prompt 边界、旧 evaluator 深度、Base 执行骨架。

直接复用 check_files/check_interface/check_parameter_update。
同名重写均以完整 Base 原函数为底稿，只接入无 YAML、空 model_args、GFNO 浅层结构/前向/训练契约等必要差异。
旧 evaluator 仅用于确定检查边界与证据深度；不恢复 Standard 权重等价、私有 state_dict 结构或深层数据流追踪。
MATLAB 7.3 reader 保留一个小型运行时增强；其余数据、模型内部和训练控制流主要保持有限静态证据。
"""
from __future__ import annotations

import ast
import inspect
import operator
import tempfile
import traceback
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import torch
import yaml
from base.base_evaluator import (
    CANDIDATE_DIR, DEVICE, _convert_arg_to_type, _execute_hooks,
    _load_candidate_config, _load_candidate_modules, _load_class,
    _resolve_args, _move_tensors_to_device,
    check_files, check_interface, check_parameter_update,
)


def check_data_contract(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """按旧 evaluator 深度检查数据源码契约，并保留 MATLAB 7.3 小型 reader 探针。

    大多数项目仅检查有限源码证据，不运行完整数据管线。reader_calls 只用于在已知入口存在时
    增强验证；未匹配时退回静态证据，不把私有函数名当成 Prompt 接口。
    """
    CHECKPOINT_NAMES = [
        "data:mat_reader",
        "data:configured_dataset",
        "data:float32_five_dimensional",
        "data:official_split",
        "data:rollout_training_samples",
        "data:rollout_samples",
        "data:normalization_default_none",
        "data:train_only_statistics",
        "data:normalization_statistics",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), msg)

    current_index = 0
    try:
        dataset_path = (CANDIDATE_DIR / config["dataset_file"]).resolve()
        source = dataset_path.read_text(encoding="utf-8-sig")
        lower = source.lower()
        compact = "".join(lower.split())
        project_source = "\n".join(
            (CANDIDATE_DIR / name).read_text(encoding="utf-8-sig")
            for name in config.get("project_files", [config["dataset_file"]])
            if (CANDIDATE_DIR / name).is_file()
        )

        static_reader = (
            ("loadmat" in lower or "h5py" in lower)
            and (f"'{config['variable'].lower()}'" in lower or f'"{config["variable"].lower()}"' in lower)
        )
        runtime_detail = "runtime probe not bound; static source evidence only"
        runtime_ok = None
        data_module = _load_candidate_modules(config["dataset_file"])
        matches = [
            item for item in config.get("reader_calls", [])
            if callable(getattr(data_module, item["function"], None))
        ]
        if len(matches) == 1:
            call = matches[0]
            reader = getattr(data_module, call["function"])
            import h5py
            import numpy as np

            fixture = config["reader_fixture"]
            with tempfile.TemporaryDirectory(prefix="gfno-reader-") as directory:
                path = Path(directory) / "synthetic_u.mat"
                values = np.arange(
                    np.prod(fixture["raw_shape"]), dtype=fixture["dtype"]
                ).reshape(fixture["raw_shape"])
                with h5py.File(path, "w", userblock_size=512) as handle:
                    array = handle.create_dataset(config["variable"], data=values)
                    array.attrs["MATLAB_class"] = np.bytes_(fixture["matlab_class"])
                with path.open("r+b") as handle:
                    header_bytes = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, GFNO synthetic fixture"
                    handle.write(header_bytes.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM")

                reads = []
                original_getitem = h5py.Group.__getitem__

                def observe_dataset(group, key):
                    value = original_getitem(group, key)
                    if isinstance(value, h5py.Dataset) and value.ndim == 4:
                        reads.append(value.name.lstrip("/"))
                    return value

                args = _resolve_args(call["args"], {"fixture": str(path)})
                inspect.signature(reader).bind(**args)
                try:
                    with patch.object(h5py.Group, "__getitem__", observe_dataset):
                        reader(**args)
                    runtime_ok = config["variable"] in reads
                    runtime_detail = (
                        f"Reader {call['function']} returned; HDF5 accesses={reads}; "
                        "synthetic format probe only"
                    )
                except Exception:
                    runtime_ok = False
                    runtime_detail = "CANDIDATE_READ_ERROR\n" + traceback.format_exc()
        elif len(matches) > 1:
            runtime_detail = "multiple configured reader entries matched; static source evidence only"

        update_check(
            0,
            static_reader and runtime_ok is not False,
            f"static_reader={static_reader}; {runtime_detail}",
        )

        current_index = 1
        dataset_name = config["dataset_name"].lower()
        update_check(
            1,
            dataset_name in project_source.lower(),
            f"Configured dataset filename present={dataset_name in project_source.lower()}: {config['dataset_name']}",
        )

        current_index = 2
        five_dimensional = "float32" in lower and any(
            token in lower
            for token in (
                "[..., none]", "np.newaxis", "newaxis", "unsqueeze",
                "ndim != 5", "ndim==5", "ndim == 5",
            )
        )
        update_check(
            2,
            five_dimensional,
            "Static float32 and five-dimensional trajectory evidence; data pipeline not executed",
        )

        current_index = 3
        ntrain, nvalid, ntest = config["split_counts"]
        train_slice = any(token in compact for token in (
            f"data[:{ntrain}]", "data[:ntrain]", "data[:train_size]",
            f"range({ntrain})", f"range(0,{ntrain})",
        ))
        valid_slice = any(token in compact for token in (
            f"data[-{nvalid + ntest}:-{ntest}]",
            "data[-(ntest+nvalid):-ntest]",
            "data[-(nvalid+ntest):-ntest]",
            "data[n-(val_size+test_size):n-test_size]",
            "data[n-(test_size+val_size):n-test_size]",
        ))
        test_slice = any(token in compact for token in (
            f"data[-{ntest}:]", "data[-ntest:]", "data[n-test_size:]",
        ))
        update_check(
            3,
            train_slice and valid_slice and test_slice,
            f"Static official split evidence: train={train_slice}, valid={valid_slice}, test={test_slice}",
        )

        current_index = 4
        rollout_training = (
            ("rollout_steps" in lower or "t_out" in lower)
            and any(token in lower for token in ("future", "target", "pde"))
            and str(config["target_steps"]) in lower
        )
        update_check(
            4,
            rollout_training,
            "Static explicit multi-step future-target evidence; sample generation not executed",
        )

        current_index = 5
        rollout_samples = "rollout" in lower and any(token in lower for token in ("history", "t_in"))
        update_check(
            5,
            rollout_samples,
            f"Static history/rollout evidence for {config['history_steps']} input and {config['target_steps']} target steps",
        )

        current_index = 6
        normalization_none = "normalization" in lower and (
            '"none"' in lower or "'none'" in lower
        )
        update_check(6, normalization_none, "Static normalization='none' support evidence")

        current_index = 7
        train_only = any(
            token in compact
            for token in ("fit(train", "fit(cls,train", "splits.train", "train_data.mean", "train_data.std")
        ) or "train_indices" in lower
        update_check(7, train_only, "Static training-split statistics evidence; no full leakage proof")

        current_index = 8
        update_check(8, "mean" in lower and "std" in lower, "Static mean/std normalization statistics evidence")

    except Exception:
        update_check(current_index, False, traceback.format_exc())

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
        - model:registered_four_stages: 配置时检查长度为指定值的已注册 ModuleList
        - model:defaults: 配置时检查 Prompt 指定的默认属性
        - model:p4_group_convolution / group_spectral / group_mlp / residual_1x1 / lifting_projection:
          配置时追加与旧 evaluator 同深度的有限源码证据
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
        candidate_config = _load_candidate_config(config_name) if config_name else {}
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)

        # 获取模型参数（必须配置 model_args）
        model_args = config.get("model_args")
        if model_args is None:
            raise ValueError("config['model_args'] is required")

        resolved_model_args = _resolve_args(model_args, context)
        model = model_cls(**resolved_model_args).to(config.get("device", DEVICE)).eval()

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

        if "registered_stage_count" in config:
            expected = config["registered_stage_count"]
            registered = [
                (name, len(module))
                for name, module in model.named_modules()
                if isinstance(module, nn.ModuleList)
            ]
            passed = any(length == expected for _, length in registered)
            checks.append((
                "model:registered_four_stages",
                passed,
                f"Registered ModuleList lengths={registered}; expected any length={expected}; "
                "container evidence only",
            ))

        if "constructor_defaults" in config:
            signature = inspect.signature(model_cls).parameters
            actual = {
                name: getattr(model, name, signature[name].default if name in signature else None)
                for name in config["constructor_defaults"]
            }
            checks.append((
                "model:defaults",
                actual == config["constructor_defaults"],
                f"Default model properties={actual}",
            ))

        if config.get("gfno_source_contract", False):
            source = (CANDIDATE_DIR / model_file).read_text(encoding="utf-8-sig").lower()
            compact = "".join(source.split())
            module_names = [type(module).__name__.lower() for module in model.modules()]
            source_checks = [
                (
                    "model:p4_group_convolution",
                    "p4" in source and "rot90" in source and ("roll" in source or "group" in source),
                    "Static p4/rot90/group source evidence only",
                ),
                (
                    "model:group_spectral",
                    "spectral" in source and "rfft2" in source and "irfft2" in source,
                    "Static spectral/rfft2/irfft2 source evidence only",
                ),
                (
                    "model:group_mlp",
                    any("mlp" in name for name in module_names) and "gelu" in source,
                    "Static MLP/GELU module evidence only",
                ),
                (
                    "model:residual_1x1",
                    ("skip" in source or "residual" in source or "bypass" in source)
                    and ("kernel_size=1" in compact or "gconv1x1" in compact or "1x1group" in compact),
                    "Static 1x1 group skip/residual/bypass evidence only",
                ),
                (
                    "model:lifting_projection",
                    ("lifting" in source or "self.p" in source)
                    and ("projection" in source or "self.q" in source),
                    "Static lifting/projection source evidence only",
                ),
            ]
            checks.extend(source_checks)

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(-1, False, error_detail)

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
        - forward_executable:output_finite: 配置时检查输出有限值
        - forward_executable:p4_equivariance: 配置时检查一次 90° 旋转的相对 L2
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
    ]
    if config.get("check_nan_inf", False):
        CHECKPOINT_NAMES.append("forward_executable:output_finite")
    if "rotation" in config:
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
        model_config = _load_candidate_config(config_name) if config_name else {}
        context["config"] = model_config
        if isinstance(model_config, dict):
            context.update(model_config)

        # ------------------ 3. 执行前置钩子（如果有） ------------------
        pre_hooks = config.get("pre_forward_hooks", [])
        hook_globals = dict(caller_globals or {})
        hook_globals.setdefault("randn", torch.randn)
        context = _execute_hooks(
            model_module, pre_hooks, context,
            caller_globals=hook_globals, caller_locals=caller_locals
        )

        # ------------------ 4. 构建模型 ------------------
        model_class_name = config.get("model_class")
        if not model_class_name:
            raise ValueError("config['model_class'] is missing or empty")

        model_cls = getattr(model_module, model_class_name)
        if model_cls is None or not callable(model_cls):
            raise ValueError(f"Model class '{model_class_name}' not found or not callable")

        # 获取模型参数（必须配置 model_args）
        model_args = config.get("model_args")
        if model_args is None:
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
            out = (
                model(*converted_forward_args.values())
                if config.get("args_as_positional", False)
                else model(**converted_forward_args)
            )

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
            finite = bool(torch.isfinite(out).all())
            update_check(current_index, finite, f"Output values finite={finite}")

        if "rotation" in config:
            current_index += 1
            rotation = config["rotation"]
            rotated_args = dict(converted_forward_args)
            rotated_args[rotation["argument"]] = torch.rot90(
                converted_forward_args[rotation["argument"]],
                rotation["k"],
                dims=tuple(rotation["dims"]),
            )
            with torch.no_grad():
                rotated_output = (
                    model(*rotated_args.values())
                    if config.get("args_as_positional", False)
                    else model(**rotated_args)
                )
            expected = torch.rot90(out, rotation["k"], dims=tuple(rotation["dims"]))
            error = float("inf")
            if isinstance(rotated_output, torch.Tensor) and tuple(rotated_output.shape) == tuple(config["out_shape"]):
                numerator = torch.linalg.vector_norm((rotated_output - expected).reshape(-1))
                denominator = torch.linalg.vector_norm(expected.reshape(-1)).clamp_min(
                    rotation["denominator_epsilon"]
                )
                error = float((numerator / denominator).item())
            update_check(
                current_index,
                error <= rotation["rtol"],
                f"Rotation relative L2={error:.8g}; tolerance={rotation['rtol']}; "
                "one configured rotation, not a full equivariance proof",
            )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks

def check_config_content(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple[str, bool, str]]:
    """
    检查训练配置内容；保留 Base 的字段存在、类型和值比较。

    config_name 指定时仍读取原配置文件；未指定时，config_file/factories 可提供
    候选 Python 配置对象。factories 沿用已审查的 function/args/attributes 映射，
    必须恰好匹配一个，不推断、不重试，不要求候选新增 YAML。
    对象来源的 result_directory 规范化为完整相对路径，不只比较 basename。
    本 Case 不配置 type，不额外增加字段类型要求；不证明全部 CLI 覆盖优先级。

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
        if not config_name and not config.get("config_file"):
            raise ValueError("config['config_name'] or config['config_file'] is required")

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

        # 指定文件时保留 Base 原加载；仅无 config_name 时接入 Python 配置对象。
        if config_name:
            config_data = _load_candidate_config(config_name)
        else:
            module = _load_candidate_modules(config["config_file"])
            matches = [item for item in config["factories"]
                       if callable(getattr(module, item["function"], None))]
            if len(matches) != 1:
                raise ValueError("REVIEW_REQUIRED: configure one applicable default-configuration factory; no fixed candidate API required")
            factory = matches[0]
            settings = getattr(module, factory["function"])(**factory.get("args", {}))
            config_data = {key: operator.attrgetter(path)(settings)
                           for key, path in factory["attributes"].items()}
            if "result_directory" in config_data:
                config_data["result_directory"] = str(Path(str(config_data["result_directory"]).rstrip("/")))

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

    原生八项按 check_config 配置检测，未配置时使用原默认值。
    可选 experiment 沿用既有 evaluation_functions/checkpoint_files/result_files 等配置，追加与旧 evaluator 同深度的有限源码证据；
    不加载配置对象、不执行训练入口，默认数值另由 check_config_content 检查。
    追加项：train:adam_scheduler、train:rollout_relative_l2、
    train:validation_rollout_each_epoch、train:best_and_last_checkpoints、
    train:reload_best_for_test、train:rotation_metric、train:configured_dataset、
    train:result_files、train:default_result_directory。它们不证明完整运行控制流。
    检查点如下：

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
    if "experiment" in config:
        CHECKPOINT_NAMES += [
            "train:adam_scheduler", "train:rollout_relative_l2",
            "train:validation_rollout_each_epoch", "train:best_and_last_checkpoints",
            "train:reload_best_for_test", "train:rotation_metric",
            "train:configured_dataset", "train:result_files",
            "train:default_result_directory",
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

        # ------------------ 11. 已有实验源码补充；复用上面的 AST ------------------
        if "experiment" in config:
            current_index = 8
            experiment = config["experiment"]
            aliases = {}
            for node in all_nodes:
                if isinstance(node, ast.Import):
                    aliases.update({item.asname or item.name.split('.')[0]: item.name if item.asname else item.name.split('.')[0] for item in node.names})
                elif isinstance(node, ast.ImportFrom):
                    aliases.update({item.asname or item.name: f'{node.module}.{item.name}' for item in node.names})
            calls = set()
            for node in all_nodes:
                if isinstance(node, ast.Call):
                    name = ast.unparse(node.func)
                    first, separator, rest = name.partition('.')
                    calls.add(aliases.get(first, first) + (separator + rest if separator else ''))
            doc_nodes = {id(node.body[0].value) for node in all_nodes
                         if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                         and node.body and isinstance(node.body[0], ast.Expr)
                         and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)}
            strings = [node.value for node in all_nodes if isinstance(node, ast.Constant)
                       and isinstance(node.value, str) and id(node) not in doc_nodes]
            names = {Path(value).name for value in strings}
            update_check(current_index, 'torch.optim.Adam' in calls and any(name.startswith('torch.optim.lr_scheduler.') for name in calls),
                         'Static Adam/scheduler construction evidence; not proof of every runtime branch')
            current_index += 1
            has_norm = any(name.split('.')[-1] in ('norm', 'vector_norm') for name in calls)
            has_ratio = any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) for node in all_nodes)
            update_check(current_index, has_norm and has_ratio,
                         'Static relative-norm ingredients; recurrence checked by training smoke; complete loss aggregation not proven')
            current_index += 1
            evaluation_names = set(experiment['evaluation_functions'])
            validation = any(isinstance(node, (ast.For, ast.While)) and any(
                isinstance(child, ast.Call) and ast.unparse(child.func).split('.')[-1] in evaluation_names
                for child in ast.walk(node)) for node in all_nodes)
            update_check(current_index, validation,
                         'Configured evaluation call occurs inside a loop; default horizon is checked separately by check_config_content; full control flow not verified')
            current_index += 1
            checkpoints = set(experiment['checkpoint_files']).issubset(names)
            update_check(current_index, checkpoints and 'torch.save' in calls,
                         'Literal best/last filenames and serialization call present; files not generated by this check')
            current_index += 1
            update_check(current_index, checkpoints and 'torch.load' in calls and any(name.endswith('.load_state_dict') for name in calls),
                         'Static checkpoint read/state restoration evidence; reload-before-test order not dynamically verified')
            current_index += 1
            update_check(current_index, any(name.split('.')[-1] == 'rot90' for name in calls) and has_norm,
                         'Static rotation/error ingredients; numerical model equivariance is a separate check')
            current_index += 1
            dataset_name = experiment["dataset_name"].lower()
            update_check(current_index, dataset_name in combined_code.lower(),
                         f"Static configured dataset filename evidence={dataset_name in combined_code.lower()}: {experiment['dataset_name']}")
            current_index += 1
            result_files = set(experiment["result_files"])
            result_files_present = all(name.lower() in combined_code.lower() for name in result_files)
            update_check(current_index, result_files_present,
                         f"Static result file declarations present={result_files_present}: {sorted(result_files)}")
            current_index += 1
            normalized_source = combined_code.replace("\\", "/")
            expected_dir = experiment["result_directory"].rstrip("/")
            update_check(current_index, expected_dir in normalized_source,
                         f"Static default result directory evidence={expected_dir in normalized_source}: {expected_dir}")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks

def check_rollout_training_smoke(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """按旧 evaluator 深度执行小型多步自回归、Relative L2、反向和一次参数更新。

    该检查只验证 GFNO 模型可参与多步可微 rollout，不调用候选私有训练入口，也不证明
    候选完整训练脚本的每个分支或每一步误差聚合方式。
    """
    CHECKPOINT_NAMES = [
        "training:rollout_shape",
        "training:rollout_relative_l2",
        "training:finite_gradients",
        "training:parameter_updates",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), msg)

    current_index = 0
    try:
        model_module = _load_candidate_modules(config["model_file"])
        model_cls = _load_class(model_module, config["model_class"])
        model = model_cls(**config.get("model_args", {})).to(config.get("device", DEVICE)).train()
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        if not parameters:
            raise ValueError("Model has no trainable parameters")

        generator = torch.Generator(device="cpu").manual_seed(config.get("seed", 20260820))
        history = torch.randn(tuple(config["input_shape"]), generator=generator).to(config.get("device", DEVICE))
        target = torch.randn(tuple(config["target_shape"]), generator=generator).to(config.get("device", DEVICE))
        optimizer = torch.optim.Adam(parameters.values(), **config["optimizer_args"])
        before = {name: value.detach().clone() for name, value in parameters.items()}

        predictions = []
        current = history
        for _ in range(config["rollout_steps"]):
            prediction = model(current)
            if not isinstance(prediction, torch.Tensor):
                raise TypeError(f"Model output is not a Tensor: {type(prediction)}")
            predictions.append(prediction)
            current = torch.cat((current[..., 1:, :], prediction), dim=-2)
        rollout = torch.cat(predictions, dim=-2)

        update_check(
            0,
            tuple(rollout.shape) == tuple(target.shape),
            f"actual={tuple(rollout.shape)} expected={tuple(target.shape)}",
        )

        current_index = 1
        difference = torch.linalg.vector_norm((rollout - target).reshape(len(target), -1), dim=1)
        scale = torch.linalg.vector_norm(target.reshape(len(target), -1), dim=1).clamp_min(1.0e-12)
        loss = (difference / scale).mean()
        update_check(1, bool(torch.isfinite(loss)), f"rollout Relative L2={float(loss.detach()):.8g}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        current_index = 2
        finite_gradients = [
            name for name, parameter in parameters.items()
            if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        ]
        update_check(
            2,
            bool(finite_gradients),
            f"finite gradient tensors={len(finite_gradients)}/{len(parameters)}",
        )

        optimizer.step()
        current_index = 3
        changed = [
            name for name, parameter in parameters.items()
            if not torch.equal(before[name], parameter.detach())
        ]
        update_check(
            3,
            bool(changed),
            f"updated parameter tensors={len(changed)}/{len(parameters)}; evaluator-owned one-step probe",
        )

    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks

