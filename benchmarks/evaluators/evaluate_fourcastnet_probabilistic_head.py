"""FourCastNet：以 Base 同名检查承接结构、前向、训练、推理与输入契约。

只保留本 Case 必需适配；频域探针及各测试配置沿用既有判定，不修改公共文件或候选源码。
"""
from __future__ import annotations

import ast
import inspect
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple
import sys
import traceback
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn, randn, linspace, reshape, full
from torch.optim import AdamW
from torch.overrides import TorchFunctionMode

from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _convert_arg_to_type,
    _move_tensors_to_device,
    REFERENCE_DIR,
    _execute_hooks,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _load_reference_modules,
    _load_reference_config,
    _resolve_args,
    _safe_isinstance_check,
    check_files,
    check_infer_output_schema,
)
from base.model_structure_checker import ModelStructureChecker

EVALUATOR_REVISION = "fourcastnet-v15-afno-spectral"
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "float32"
DEFAULT_SEED = 17

# randn/linspace/reshape/full/AdamW 是配置中实际使用的库接口，供 _execute_hooks 查找。
# 未导入本地重写的同名公共评分方法；不新增通用工厂或注册框架。


@contextmanager
def _test_runtime(config: dict):
    """设置本次测试的设备、数据类型和随机种子，结束后恢复 RNG/梯度状态。"""
    device = torch.device(config.get("device", DEFAULT_DEVICE))
    dtype = getattr(torch, config.get("dtype", DEFAULT_DTYPE), None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("params.dtype must name a real floating-point torch dtype")
    if type(config.get("seed", DEFAULT_SEED)) is not int:
        raise TypeError("params.seed must be an integer")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This evaluator supports explicit cpu or cuda devices")

    cuda_devices = []
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA/DCU device unavailable; no CPU fallback")
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        cuda_devices = [index]

    with torch.random.fork_rng(devices=cuda_devices), torch.enable_grad():
        seed = config.get("seed", DEFAULT_SEED)
        torch.random.default_generator.manual_seed(seed)
        for index in cuda_devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        yield device, dtype


@_test_runtime({})
def check_model_structure(config: dict, caller_globals=None, caller_locals=None):
    """
    检查模型结构；配置 input_projection/distinct_modules 时追加投影与 Head 对象独立性检查。

    config 示例:
    {
        "model_file": "model.py",
        "model_class": "MyModel",
        "config_name": "config.yaml",
        "model_args": {"space_dim": "${config.space_dim}", "n_layers": 3, "hidden_size": 512},  # 模型参数（支持占位符）1
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
        - model_structure:{rule.id}[:module_path]: 公共结构规则
        - model_internal:input_projection / heads_distinct: 配置时检查，不检查底层参数存储共享
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
    if config.get("input_projection") is not None:
        CHECKPOINT_NAMES.append("model_internal:input_projection")
    if config.get("distinct_modules") is not None:
        CHECKPOINT_NAMES.append("model_internal:heads_distinct")
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
        if not model_args:
            raise ValueError("config['model_args'] is required")

        resolved_model_args = _resolve_args(model_args, context)
        model = model_cls(**resolved_model_args).to(DEVICE).eval()

        update_check(
            current_index,
            True,
            f"Model built with config: {candidate_config}"
        )

        current_index = 1
        from base.model_structure_checker import ModuleInfoExtractor, ModelStructureChecker, BuiltinRules
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

        # 输入投影仍按真实 Conv2d 选择；通道比较直接使用公共 conv_shape 规则。
        if config.get("input_projection") is not None:
            current_index = CHECKPOINT_NAMES.index("model_internal:input_projection")
            projections = [
                (name, *BuiltinRules.check_conv_shape(ModuleInfoExtractor(child).extract(), config["input_projection"]))
                for name, child in model.named_modules() if isinstance(child, nn.Conv2d)
            ]
            update_check(current_index, any(ok for _, ok, _ in projections),
                         f"Input projections: {projections}")
        if config.get("distinct_modules") is not None:
            current_index = CHECKPOINT_NAMES.index("model_internal:heads_distinct")
            head_names = config["distinct_modules"]
            try:
                heads = [model.get_submodule(name) for name in head_names]
                distinct = len(heads) >= 2 and len({id(head) for head in heads}) == len(heads)
                update_check(current_index, distinct,
                             f"Distinct module objects: {head_names}; {distinct}")
            except Exception:
                update_check(current_index, False, traceback.format_exc())

    except Exception as e:
        error_detail = traceback.format_exc()
        if current_index >= len(checks):
            CHECKPOINT_NAMES.append("model_structure:structure")
            checks.append((CHECKPOINT_NAMES[-1], False, "Not executed due to earlier failure"))
        update_check(current_index, False, error_detail)

    return checks


@_test_runtime({})
def check_model_forward_executable(config: dict, caller_globals=None, caller_locals=None):
    """
    检查模型前向传播；按配置检查输出，或观察尺度关系、Block 调用及既有频域路径。

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
        - forward_executable:returns_tensor_shape / returns_dict_shape: 配置 out_shape 时检查形状
        - forward_executable:{field}_finite / {field}_positive: 配置时检查有限性／正值
        - model_internal:*: 配置时观察建模、softplus、Block 形状和频域路径；不是完整 AFNO 数学证明
    """
    # ================================================================
    # 0. 定义基础及按配置追加的检查点；建模结果在 Forward 之前更新
    # ================================================================
    CHECKPOINT_NAMES = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
    ]
    output_shapes = config.get("out_shape", [])
    fields = list(output_shapes) if isinstance(output_shapes, dict) else ["output"]
    if isinstance(output_shapes, dict):
        CHECKPOINT_NAMES[1] = "forward_executable:returns_dict_shape"
    if config.get("check_nan_inf", False):
        CHECKPOINT_NAMES += [f"forward_executable:{name}_finite" for name in fields]
    CHECKPOINT_NAMES += [f"forward_executable:{name}_positive" for name in config.get("positive_outputs", [])]
    relation = config.get("scale_transform")
    block_rule = config.get("block_validation")
    spectral_enabled = bool(block_rule and block_rule.get("check_spectral_path", False))
    if relation is not None or block_rule is not None:
        CHECKPOINT_NAMES.append("model_internal:build_model")
    if relation is not None:
        CHECKPOINT_NAMES.append("model_internal:softplus_min_scale")
    if block_rule is not None:
        CHECKPOINT_NAMES.append("model_internal:blocks_used")
        CHECKPOINT_NAMES += [f"model_internal:block_{i}_shape" for i in range(block_rule["count"])]
        if spectral_enabled:
            CHECKPOINT_NAMES += [f"model_internal:block_{i}_{name}"
                                 for i in range(block_rule["count"])
                                 for name in ("fft_calls", "spectral_path", "input_bypass")]
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
        context = {"dtype": torch.float32, "device": DEVICE}  # 初始化上下文

        # ------------------ 2. 加载配置 ------------------
        config_name = config.get("config_name")
        model_config = _load_candidate_config(config_name) if config_name else {}
        context["config"] = model_config
        if isinstance(model_config, dict):
            context.update(model_config)

        # ------------------ 3. 执行前置钩子（如果有） ------------------
        pre_hooks = config.get("pre_forward_hooks", [])
        if relation is None and block_rule is None:
            context = _execute_hooks(model_module, pre_hooks, context, globals(), context)

        # ------------------ 4. 构建模型 ------------------
        if relation is not None or block_rule is not None:
            current_index = CHECKPOINT_NAMES.index("model_internal:build_model")
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
        if relation is not None or block_rule is not None:
            update_check(current_index, True, "Model initialized")
            current_index = 0
            context["model"] = model
            # 内部观察样例保持先建模、后生成输入的原顺序。
            context = _execute_hooks(model_module, pre_hooks, context, globals(), context)

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

        if relation is not None or block_rule is not None:
            head_observations, block_observations = [], {}
            block_inputs = {}
            # 在安装旧Forward观察器前复制，避免复制带有闭包的临时forward。
            spectral_blocks = {
                path: deepcopy(child) for path, child in model.named_modules()
                if spectral_enabled and type(child).__name__ == block_rule["class_name"]
            }
            with ExitStack() as stack:
                if relation is not None:
                    head = model.get_submodule(relation["head"])
                    for path, child in head.named_modules():
                        def observe_head(*a, _forward=child.forward, _path=path,
                                         _softplus=isinstance(child, nn.Softplus), **kw):
                            raw = a[0] if a else kw.get("input")
                            if _softplus and isinstance(raw, torch.Tensor):
                                head_observations.append((_path + ":input", raw.detach().clone()))
                            value = _forward(*a, **kw)
                            if isinstance(value, torch.Tensor):
                                head_observations.append((_path or "<head>", value.detach().clone()))
                            return value
                        stack.enter_context(patch.object(child, "forward", observe_head))
                if block_rule is not None:
                    for path, child in model.named_modules():
                        if type(child).__name__ != block_rule["class_name"]:
                            continue
                        def observe_block(*a, _forward=child.forward, _path=path, **kw):
                            raw = a[0] if a else next(iter(kw.values()))
                            if spectral_enabled:
                                block_inputs[_path] = raw.detach().clone()
                            value = _forward(*a, **kw)
                            block_observations[_path] = (tuple(raw.shape), tuple(value.shape))
                            return value
                        stack.enter_context(patch.object(child, "forward", observe_block))
                with torch.no_grad():
                    out = (model(*converted_forward_args.values()) if config.get("args_as_positional", False)
                           else model(**converted_forward_args))
        else:
            with torch.no_grad():
                out = (model(*converted_forward_args.values()) if config.get("args_as_positional", False)
                       else model(**converted_forward_args))

        # 更新检查点：forward_run 通过
        update_check(
            current_index,
            True,
            f"Forward pass executed successfully with args"
        )

        # ------------------ 8. 按 out_shape 检查 Tensor／字典形状 ------------------
        current_index += 1

        if "out_shape" not in config and (relation is not None or block_rule is not None):
            # 该观察样例原本未配置最终输出 shape；不新增一组输出形状约束。
            same_shape = True
            values = {}
        elif isinstance(output_shapes, dict):
            same_shape = isinstance(out, Mapping) and set(out) == set(output_shapes)
            values = {name: out.get(name) if isinstance(out, Mapping) else None for name in fields}
            same_shape = same_shape and all(
                isinstance(values[name], torch.Tensor) and tuple(values[name].shape) == tuple(shape)
                for name, shape in output_shapes.items()
            )
        else:
            # 保留原单 Tensor 路径；修正原 out_shape 配置键及尺寸比较。
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"Output is not a Tensor: {type(out)}")
            same_shape = tuple(out.shape) == tuple(output_shapes)
            values = {"output": out}
        update_check(current_index, same_shape,
                     f"Output shape contract: {output_shapes}" if "out_shape" in config
                     else "Output shape check skipped: this observation fixture has no out_shape")

        if config.get("check_nan_inf", False):
            for name in fields:
                current_index += 1
                value = values[name]
                finite = isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                update_check(current_index, finite, f"{name} finite={finite}")
        for name in config.get("positive_outputs", []):
            current_index += 1
            value = values[name]
            positive = isinstance(value, torch.Tensor) and bool((value > 0).all())
            update_check(current_index, positive, f"{name} positive={positive}")

        if relation is not None or block_rule is not None:
            current_index = CHECKPOINT_NAMES.index("model_internal:build_model") + 1
            if relation is not None:
                scale = out["scale"]
                comparisons = []
                for path, raw in head_observations:
                    if raw.shape == scale.shape and raw.is_floating_point():
                        expected = nn.functional.softplus(raw) + relation["min_scale"]
                        passed = bool(torch.isfinite(scale).all()) and bool(torch.allclose(
                            scale, expected, atol=relation["atol"], rtol=relation["rtol"]))
                        comparisons.append((path, passed))
                update_check(current_index, any(passed for _, passed in comparisons),
                             f"min_scale={relation['min_scale']}; observations={comparisons}")
                current_index += 1
            if block_rule is not None:
                observed = list(block_observations.items())
                update_check(current_index, len(observed) == block_rule["count"],
                             f"Used blocks={list(block_observations)}")
                current_index += 1
                for index in range(block_rule["count"]):
                    if index < len(observed):
                        path, (input_shape, output_shape) = observed[index]
                        passed = input_shape == output_shape == tuple(block_rule["shape"])
                        update_check(current_index, passed, f"{path}: {input_shape} -> {output_shape}")
                    current_index += 1
                if spectral_enabled:
                    forward_ops = {torch.fft.fft, torch.fft.fft2, torch.fft.fftn,
                                   torch.fft.rfft, torch.fft.rfft2, torch.fft.rfftn}
                    inverse_ops = {torch.fft.ifft, torch.fft.ifft2, torch.fft.ifftn,
                                   torch.fft.irfft, torch.fft.irfft2, torch.fft.irfftn}

                    class SpectralTrace(TorchFunctionMode):
                        """仅在Block副本运行期间观察；不替换全局FFT或修改候选源码。"""
                        def __init__(self, detach_inverse=False):
                            super().__init__()
                            self.detach_inverse = detach_inverse
                            self.forward_values, self.inverse_values, self.calls = [], [], []

                        def __torch_function__(self, func, types, args=(), kwargs=None):
                            value = func(*args, **(kwargs or {}))
                            if func in forward_ops:
                                self.calls.append(func.__name__)
                                self.forward_values.append(value)
                            elif func in inverse_ops:
                                self.calls.append(func.__name__)
                                input_value = args[0] if args else kwargs["input"]
                                self.inverse_values.append((input_value, value))
                                if self.detach_inverse:
                                    return value.detach()
                            return value

                    for index in range(block_rule["count"]):
                        start_index = current_index
                        current_index += 3
                        if index >= len(observed):
                            for offset in range(3):
                                update_check(start_index + offset, False, "No observed Block to probe")
                            continue
                        path = observed[index][0]
                        stage_index = start_index
                        try:
                            # 原模型不重新执行；每次探针使用同一权重/输入的新副本。
                            with _test_runtime(config):
                                probe_input = block_inputs[path].detach().clone().requires_grad_(True)
                                with SpectralTrace() as trace:
                                    probe_output = deepcopy(spectral_blocks[path])(probe_input)
                                called = bool(trace.forward_values) and bool(trace.inverse_values)
                                update_check(start_index, called, f"{path}: FFT calls={trace.calls}")
                                stage_index += 1
                                # 先确认FFT的输入来源可追溯到当前Block输入。
                                linked_forward = []
                                for value in trace.forward_values:
                                    if value.requires_grad:
                                        grad = torch.autograd.grad(
                                            value, probe_input, torch.ones_like(value).contiguous(),
                                            retain_graph=True, allow_unused=True)[0]
                                        if grad is not None and bool(torch.isfinite(grad).all()):
                                            linked_forward.append(value)
                                connected = []
                                if linked_forward and isinstance(probe_output, torch.Tensor) and probe_output.requires_grad:
                                    for inverse_input, inverse_output in trace.inverse_values:
                                        if not inverse_input.requires_grad or not inverse_output.requires_grad:
                                            continue
                                        used = torch.autograd.grad(
                                            probe_output, inverse_output, torch.ones_like(probe_output).contiguous(),
                                            retain_graph=True, allow_unused=True)[0]
                                        if used is None or not bool(torch.isfinite(used).all()):
                                            connected.append(False)
                                            continue
                                        upstream = torch.autograd.grad(
                                            inverse_input, linked_forward, torch.ones_like(inverse_input).contiguous(),
                                            retain_graph=True, allow_unused=True)
                                        connected.append(any(grad is not None and bool(torch.isfinite(grad).all())
                                                             for grad in upstream))
                                # 允许零梯度：Softshrink/零初始化可以合法地产生暂时为零的频域支路。
                                update_check(start_index + 1, any(connected),
                                             f"{path}: input->FFT->inverse FFT->output graph paths={connected}")
                        except Exception:
                            update_check(stage_index, False, f"{path}: spectral probe error\n{traceback.format_exc()}")
                        try:
                            # 仅在另一份探针中断开逆FFT梯度，数值不变；观察绕过频域的输入连接。
                            with _test_runtime(config):
                                probe_input = block_inputs[path].detach().clone().requires_grad_(True)
                                with SpectralTrace(detach_inverse=True) as trace:
                                    probe_output = deepcopy(spectral_blocks[path])(probe_input)
                                bypass = None
                                if isinstance(probe_output, torch.Tensor) and probe_output.requires_grad:
                                    weights = torch.linspace(-0.7, 0.9, probe_output.numel(),
                                                             device=probe_output.device, dtype=probe_output.dtype)
                                    bypass = torch.autograd.grad(probe_output, probe_input,
                                                                 weights.reshape_as(probe_output).contiguous(),
                                                                 allow_unused=True)[0]
                                passed = (bool(trace.inverse_values) and bypass is not None
                                          and bool(torch.isfinite(bypass).all()))
                                update_check(start_index + 2, passed,
                                             f"{path}: input bypass after cutting inverse FFT={passed}; "
                                             "graph evidence only, not an identity-residual proof")
                        except Exception:
                            update_check(start_index + 2, False, f"{path}: bypass probe error\n{traceback.format_exc()}")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


@_test_runtime({})
def check_method_correctness(config: dict,caller_globals=None, caller_locals=None) -> List[tuple]:
    """
    检查候选方法的正确性，通过与期望输出对比或与参考方法对比。

    config 格式:
    {
        "candidate": {
            "file": "train.py",  # 候选方法所在文件
            "function": "relevate2_loss",  # 当前方法读取模块级函数
            "args": {  # 调用参数（支持占位符）
                "data": "${input_data}",
                "config": "${config}"
            },
            "pre_hooks": [  # 前置钩子，执行后结果保存到上下文
                {
                    "function": "load_data",
                    "args": {"path": "data.npy"},
                    "save_as": "input_data"  # 保存到上下文的键名
                }
            ],
            "expected": {  # 可选：期望输出
                "output": "${expected_output}",
                "type": "tensor",  # 或 "list", "numpy", "dict", "any"
                "rtol": 1e-6,
                "atol": 1e-8,
                "shape": [10, 10]  # 可选：期望形状
            }
        },
        "reference": {  # 可选：参考方法，与candidate配置格式相同
            "file": "run_train.py",
            "function": "relevate2_loss",
            "args": {
                "data": "${input_data}",
                "config": "${config}"
            },
            "pre_hooks": [
                {
                    "function": "load_data",
                    "args": {"path": "data.npy"},
                    "save_as": "input_data"
                }
            ]
        }
    }

    校验规则:
        - expected.output 存在时比较数值；只有 expected.type 时检查类型
        - 未指定 expected.output 而有 reference 时，与参考方法比较
        - 如果两者都没有，则只检查方法是否能执行

    检查点:
        - method_check:candidate_loaded: 候选方法加载成功
        - method_check:candidate_executed: 候选方法执行成功
        - method_check:output_matches: 按配置检查类型或比较输出；无比较配置时显式跳过
        - method_check:shape_matches: 配置形状时比较；未配置时保留显式跳过结果
    """

    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "method_check:candidate_loaded",
        "method_check:candidate_executed",
        "method_check:output_matches",
        "method_check:shape_matches",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    # ================================================================
    # 主流程
    # ================================================================
    current_index = 0
    context = {"config": {}, "dtype": torch.float32, "device": DEVICE}  # 初始化上下文
    candidate_module = None
    reference_module = None

    try:
        # ------------------ 检查配置 ------------------
        candidate_config = config.get("candidate")
        if not candidate_config:
            raise ValueError("config['candidate'] is required")

        reference_config = config.get("reference")

        # ------------------ 1. 加载候选方法 ------------------
        candidate_file = candidate_config.get("file")
        candidate_fn_name = candidate_config.get("function")

        if not candidate_file or not candidate_fn_name:
            raise ValueError("candidate requires both 'file' and 'function'")

        # 加载候选模块
        candidate_module = _load_candidate_modules(candidate_file)
        # 获取候选函数
        candidate_method = getattr(candidate_module, candidate_fn_name, None)
        if candidate_method is None:
            raise ValueError(f"Function '{candidate_fn_name}' not found in '{candidate_file}'")
        if not callable(candidate_method):
            raise ValueError(f"'{candidate_fn_name}' is not callable")

        # 将模块保存到上下文中，供钩子使用
        context["candidate_module"] = candidate_module

        update_check(
            current_index,
            True,
            f"Candidate function '{candidate_fn_name}' loaded from '{candidate_file}'"
        )
        current_index += 1

        # ------------------ 2. 执行候选方法的前置钩子 ------------------
        pre_hooks = candidate_config.get("pre_hooks", [])
        for hook_config in pre_hooks:
            context = _execute_hooks(candidate_module, [hook_config], context, globals(), context)
            hook_result = context.get(hook_config.get("save_as"))
            # 下一条 hook 可能执行模型；先与 Base 的 $device 保持一致。
            if isinstance(hook_result, nn.Module):
                hook_result.to(DEVICE)

        # ------------------ 3. 解析候选方法的参数 ------------------
        args_config = candidate_config.get("args", {})
        resolved_args = _resolve_args(args_config, context)
        resolved_args = _move_tensors_to_device(resolved_args)

        # ------------------ 4. 在候选执行前准备期望，避免原地修改污染参考 ------------------
        expected_config = candidate_config.get("expected") or {}
        expected_output = None
        compare_mode = "type_only"
        if "output" in expected_config:
            expected_output = deepcopy(_resolve_args({"output": expected_config["output"]}, context)["output"])
            compare_mode = "expected"
        elif reference_config:
            reference_module = _load_reference_modules(reference_config["file"])
            reference_method = getattr(reference_module, reference_config["function"])
            # 支持原 reference.pre_hooks；未配置时直接使用已准备的公共上下文。
            reference_context = _execute_hooks(reference_module, reference_config.get("pre_hooks", []),
                                               dict(context), globals(), context)
            reference_args = _resolve_args(reference_config.get("args", args_config), reference_context)
            # 非叶 Tensor 不能直接 deepcopy；先保存 Tensor 快照再复制容器。
            pending, tensor_copies = [reference_args], {}
            while pending:
                value = pending.pop()
                if isinstance(value, torch.Tensor):
                    tensor_copies.setdefault(id(value), value.detach().clone())
                elif isinstance(value, Mapping):
                    pending.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    pending.extend(value)
            reference_args = deepcopy(reference_args, tensor_copies)
            with torch.no_grad():
                expected_output = (reference_method(*reference_args.values()) if reference_config.get("args_as_positional", False)
                                   else reference_method(**reference_args))
            expected_output = (expected_output.detach().clone() if isinstance(expected_output, torch.Tensor)
                               else deepcopy(expected_output))
            compare_mode = "reference"

        # ------------------ 5. 执行候选方法 ------------------
        candidate_result = (candidate_method(*resolved_args.values()) if candidate_config.get("args_as_positional", False)
                            else candidate_method(**resolved_args))
        context["candidate_result"] = candidate_result

        update_check(
            current_index,
            True,
            f"Candidate function executed successfully with args={resolved_args}"
        )
        current_index += 1
        if not expected_config and not reference_config:
            # 模式3：保留 Base 的结果项和显式跳过分支；跳过不代表完成数值或形状校验。
            update_check(
                current_index,
                True,
                "No expected or reference config, skipping output comparison"
            )
            current_index += 1
            update_check(
                current_index,
                True,
                "Shape check skipped (no comparison mode)"
            )
            return checks

        # ------------------ 6. 对比输出 ------------------
        rtol = expected_config.get("rtol", 1e-5) if expected_config else 1e-5
        atol = expected_config.get("atol", 1e-8) if expected_config else 1e-8
        expected_type = expected_config.get("type", "auto") if expected_config else "auto"

        # 该 Base 快照未定义 _compare_outputs；保留类型和数值对照的最小实现。
        output_match = True
        if expected_type not in ("auto", "any"):
            target_type = nn.Module if expected_type == "nn.Module" else expected_type
            output_match, _ = _safe_isinstance_check(candidate_result, target_type)
        if compare_mode != "type_only":
            actual = torch.as_tensor(candidate_result).detach().cpu()
            expected_value = torch.as_tensor(expected_output).detach().cpu()
            output_match = output_match and actual.shape == expected_value.shape
            output_match = output_match and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected_value).all())
            output_match = output_match and bool(torch.allclose(actual.to(torch.float64), expected_value.to(torch.float64),
                                                               rtol=rtol, atol=atol))
        match_msg = f"type={type(candidate_result).__name__}; mode={compare_mode}"

        update_check(
            current_index,
            output_match,
            f"Output comparison ({compare_mode}): {match_msg}"
        )
        current_index += 1

        # ------------------ 7. 检查形状（如果指定） ------------------
        expected_shape = expected_config.get("shape") if expected_config else None

        if expected_shape is not None:
            shape_match = hasattr(candidate_result, "shape") and list(candidate_result.shape) == expected_shape
            shape_msg = f"actual_shape={getattr(candidate_result, 'shape', None)}, expected_shape={expected_shape}"
            update_check(
                current_index,
                shape_match,
                f"Shape check: {shape_msg}"
            )
        else:
            update_check(
                current_index,
                True,
                "Shape check skipped (not specified)"
            )

    except Exception as e:
        error_detail = traceback.format_exc()
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
                train_path.read_text(encoding="utf-8"),
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


@_test_runtime({})
def check_smoke_training(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple]:
    """
    运行训练入口冒烟，或按 gradient_inputs 配置执行标量函数的可微性冒烟。

    config 示例:
    {
        "run_file": "train.py",           # 训练模块文件
        "config_name": "config.yaml",          # 可选配置文件名称
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

    配置 gradient_inputs 时，调用标量函数并检查指定 Tensor 的梯度；不检查解析梯度数值。
    检查点:
        - method_grad:loaded / executed / gradients / scalar_shape: 仅 gradient_inputs 场景
        - training:create_work_dir: 检查能否创建工作目录
        - training:smoke_test: 检查冒烟测试是否通过
    """
    # ================================================================
    # 0. 定义基础及按配置追加的检查点；loaded 在目录创建前更新
    # ================================================================
    CHECKPOINT_NAMES = [
        "training:create_work_dir",
        "training:smoke_test",
    ]
    if "gradient_inputs" in config:
        CHECKPOINT_NAMES += ["method_grad:loaded", "method_grad:executed",
                             "method_grad:gradients", "method_grad:scalar_shape"]
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
        if "gradient_inputs" in config:
            current_index = CHECKPOINT_NAMES.index("method_grad:loaded")
        train_module = _load_candidate_modules(train_file)
        train_fn = getattr(train_module, train_fn_name, None)
        if train_fn is None or not callable(train_fn):
            raise ImportError(f"Train function '{train_fn_name}' not found or not callable in {train_file}")

        if "gradient_inputs" in config:
            update_check(CHECKPOINT_NAMES.index("method_grad:loaded"), True, "Candidate callable loaded")

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
        # 未指定配置文件时使用空配置；指定文件的加载错误仍按原流程返回失败。
        candidate_config = _load_candidate_config(config_name) if config_name else {}
        runtime_config = deepcopy(candidate_config)

        # ------------------ 5. 构造冒烟配置（轻量覆盖） ------------------

        # 默认冒烟配置（epochs=1, batch_size=8 等轻量参数）
        smoke_config = config.get("smoke_config", {})
        context = {"checkpoint_dir": checkpoint_dir, "results_dir": results_dir, "dtype": torch.float32, "device": DEVICE}
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

            hook_module = (_load_candidate_modules(hook_config["file"])
                           if hook_config.get("file") else train_module)
            hook_func = getattr(hook_module, func_name, None)
            if hook_func is None:
                parts = func_name.split(".")
                hook_func = context.get(parts[0], globals().get(parts[0]))
                for part in parts[1:]:
                    hook_func = getattr(hook_func, part, None)
            if hook_func is None or not callable(hook_func):
                raise ValueError(f"Pre-hook {idx}: function '{func_name}' not found or not callable")

            resolved_args = _resolve_args(args_config, context)
            hook_result = hook_func(**resolved_args)
            # 在后续模型调用或 optimizer 构造之前，与输入使用同一设备。
            if isinstance(hook_result, nn.Module):
                hook_result = hook_result.to(DEVICE)

            save_as = hook_config.get("save_as")
            if save_as:
                context[save_as] = hook_result
                if isinstance(runtime_config, dict):
                    runtime_config[save_as] = hook_result

        # ------------------ 7. 执行冒烟测试 ------------------
        # 解析训练函数参数（与 check_training 一致）
        train_args_config = config.get("train_args", {})
        resolved_train_args = _resolve_args(train_args_config, context)

        if "gradient_inputs" in config:
            gradient_inputs = _resolve_args(config["gradient_inputs"], context)
            if not isinstance(gradient_inputs, dict) or not gradient_inputs:
                raise ValueError("gradient_inputs must be a nonempty label-to-Tensor mapping")
            for value in gradient_inputs.values():
                if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                    raise TypeError("Each gradient input must be a floating-point Tensor")
                value.requires_grad_(True)

        # 执行配置入口：训练函数或独立标量函数，每个 Check 只调用一次。
        if config.get("args_as_positional", False):
            result = train_fn(*resolved_train_args.values())
        else:
            result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)

        update_check(
            current_index,
            True,
            f"Smoke test completed successfully: {train_fn_name}(...) with {config_name}"
        )

        if "gradient_inputs" in config:
            current_index = CHECKPOINT_NAMES.index("method_grad:executed")
            update_check(current_index, True, "Candidate call completed")
            current_index += 1
            passed = (isinstance(result, torch.Tensor) and result.ndim == 0 and result.is_floating_point()
                      and bool(torch.isfinite(result)) and result.requires_grad)
            valid = []
            if passed:
                gradients = torch.autograd.grad(result, list(gradient_inputs.values()), allow_unused=True)
                valid = [gradient is not None and gradient.shape == value.shape
                         and bool(torch.isfinite(gradient).all())
                         for value, gradient in zip(gradient_inputs.values(), gradients)]
                passed = all(valid)
            update_check(current_index, passed, f"gradient_paths={dict(zip(gradient_inputs, valid))}")
            current_index += 1
            update_check(current_index, isinstance(result, torch.Tensor) and result.ndim == 0, "Expected scalar Tensor")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


@_test_runtime({})
def check_infer_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    动态执行一次推理；配置 check_no_grad 时观察这次模型调用是否由候选关闭梯度。

    config 示例:
    {
        "run_file": "inference.py",
        "config_name": "config.yaml",  # 可选
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
        - inference_execution:output_save: 检查输出保存；未配置 output_path 时记为跳过
        - inference_no_grad:forward_grad_disabled: 仅 check_no_grad=True 时观察，不替候选关闭梯度
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
    if config.get("check_no_grad", False):
        CHECKPOINT_NAMES.append("inference_no_grad:forward_grad_disabled")
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
        # 删除本次输出的旧副本；本轮执行失败时不能沿用上轮结果。
        if config.get("output_path"):
            previous_output = Path(config["output_path"])
            if not previous_output.is_absolute():
                previous_output = CANDIDATE_DIR / previous_output
            previous_output.unlink(missing_ok=True)

        # ------------------ 1. 加载推理模块 ------------------
        infer_file = config.get("run_file")
        if not infer_file:
            raise ValueError("config['run_file'] is missing or empty")

        infer_module = _load_candidate_modules(infer_file)
        update_check(current_index, True, f"Module loaded from {infer_file}")

        # ------------------ 2. 加载配置 ------------------
        # 模块已加载；后续准备错误不能覆盖 module_load 的成功结果。
        current_index += 1
        config_name = config.get("config_name")
        candidate_config = _load_candidate_config(config_name) if config_name else {}
        context = {"config": candidate_config, "dtype": torch.float32, "device": DEVICE}
        context.update(candidate_config)

        # ------------------ 3. 获取推理函数 ------------------
        infer_func_name = config.get("infer_function", "inference")
        infer_func = getattr(infer_module, infer_func_name, None)
        if infer_func is None or not callable(infer_func):
            raise ValueError(f"No callable inference entry '{infer_func_name}' found")

        if config.get("model_file"):
            model_module = _load_candidate_modules(config["model_file"])
            model_class = _load_class(model_module, config["model_class"])
            context["model"] = model_class(**_resolve_args(config["model_args"], context)).to(DEVICE)

        # ------------------ 4. 执行前置钩子 ------------------
        pre_hooks = config.get("pre_infer_hooks", [])
        for idx, hook_config in enumerate(pre_hooks):
            func_name = hook_config.get("function")
            args_config = hook_config.get("args", {})

            hook_module = (_load_candidate_modules(hook_config["file"])
                           if hook_config.get("file") else infer_module)
            hook_func = getattr(hook_module, func_name, None)
            if hook_func is None:
                parts = func_name.split(".")
                hook_func = context.get(parts[0], globals().get(parts[0]))
                for part in parts[1:]:
                    hook_func = getattr(hook_func, part, None)
            if hook_func is None or not callable(hook_func):
                raise ValueError(f"Hook {idx}: function '{func_name}' not found or not callable")

            resolved_args = _resolve_args(args_config, context)
            hook_result = hook_func(**resolved_args)
            # 在后续推理调用之前，与输入使用同一设备。
            if isinstance(hook_result, nn.Module):
                hook_result = hook_result.to(DEVICE)

            save_as = hook_config.get("save_as")
            if save_as:
                context[save_as] = hook_result

        update_check(current_index, True, f"All {len(pre_hooks)} pre-hooks executed")

        # ------------------ 5. 执行推理 ------------------
        current_index += 1
        infer_args_config = config.get("infer_args", {})
        resolved_infer_args = _resolve_args(infer_args_config, context)

        if config.get("check_no_grad", False):
            grad_states = []
            def observe_grad(_module, _inputs):
                grad_states.append(torch.is_grad_enabled())
            observed_model = context[config.get("observed_model_key", "model")]
            handle = observed_model.register_forward_pre_hook(observe_grad)
            try:
                result = (infer_func(*resolved_infer_args.values()) if config.get("args_as_positional", False)
                          else infer_func(**resolved_infer_args))
            finally:
                handle.remove()
        else:
            result = (infer_func(*resolved_infer_args.values()) if config.get("args_as_positional", False)
                      else infer_func(**resolved_infer_args))
        update_check(current_index, True, "Inference executed successfully")

        # ------------------ 6. 保存输出 ------------------
        current_index += 1
        output_path = config.get("output_path")
        if output_path:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = CANDIDATE_DIR / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 仅为保存而规范有序字段；完整容器兼容由已有 _output_values 处理。
            if config.get("output_fields"):
                result = torch.stack([value.reshape(()) for value in _output_values(result, config["output_fields"])])

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

        if config.get("check_no_grad", False):
            current_index += 1
            update_check(current_index, bool(grad_states) and not any(grad_states), f"grad_states={grad_states}")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def _output_values(output, names):
    """按配置的字段名/顺序取得数值 Tensor；供输出检查和参考对照共用。

    不理解指标含义，不检查公式或有限性。支持字典、属性对象和有序数值容器。
    """
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("Output fields must be a nonempty list with unique names")
    if isinstance(output, Mapping):
        values = [output[name] for name in names]
    elif all(hasattr(output, name) for name in names):
        values = [getattr(output, name) for name in names]
    elif isinstance(output, (torch.Tensor, np.ndarray)):
        values = output.reshape(-1)
    elif isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
        values = output
    else:
        raise TypeError(f"Unsupported output container: {type(output).__name__}")
    if len(values) != len(names):
        raise ValueError(f"Expected {len(names)} fields, got {len(values)}")
    return [value.detach().cpu() if isinstance(value, torch.Tensor)
            else torch.as_tensor(value) for value in values]


@_test_runtime({})
def check_evaluate_oracle(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    通过与标准实现的推理结果进行对比，验证当前推理实现的正确性。

    config 示例:
    {
        "candidate": {
            "run_file": "inference.py",
            "config_name": "config.yaml",
            "infer_function": "run_inference",
            "pre_infer_hooks": [
                {"function": "load_model", "args": {"device": "cuda"}, "save_as": "model"},
                {"function": "preprocess", "args": {"normalize": True}}
            ],
            "infer_args": {"model": "${model}"}
        },
        "reference": {
            "run_file": "run_inference.py",
            "config_name": "config.yaml",
            "infer_function": "run_inference",
            "pre_infer_hooks": [
                {"function": "load_model", "args": {"device": "cpu"}, "save_as": "model"},
                {"function": "preprocess", "args": {"normalize": False}}
            ],
            "infer_args": {"model": "${model}"}
        }
    }

    检查点:
        - evaluate_oracle:shape_match: 检查形状是否匹配
        - evaluate_oracle:value_match: 按配置容差或逐值相等比较
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "evaluate_oracle:shape_match",
        "evaluate_oracle:value_match",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    shared_args = {}

    def to_cpu_tensor(data: Any) -> torch.Tensor:
        """
        将数据转换为 CPU tensor
        """
        try:
            if config.get("output_fields"):
                values = _output_values(data, config["output_fields"])
                if config.get("scalar_outputs", False):
                    return torch.stack([value.reshape(()) for value in values])
                return torch.stack(values)
            # 如果是 torch.Tensor，移动到 CPU
            if torch.is_tensor(data):
                return data.cpu()
            # 如果是 numpy array，转换为 torch.Tensor
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            # 如果是 list/tuple，尝试转换为 torch.Tensor
            if isinstance(data, (list, tuple)):
                return torch.tensor(data)
            # 尝试直接转换
            return torch.tensor(data)
        except:
            raise ValueError(f"Cannot convert {type(data)} to torch.Tensor")

    def run_inference_pipeline(module, pipeline_config: dict, context: dict, is_candidate: bool) -> Any:
        """
        执行推理流程（包括 pre_infer_hooks）
        """
        # 执行前置钩子
        pre_hooks = pipeline_config.get("pre_infer_hooks", [])
        context = _execute_hooks(module, pre_hooks, context, globals(), context)

        # 执行推理
        infer_func_name = pipeline_config.get("infer_function", "inference")
        infer_func = getattr(module, infer_func_name, None)
        if infer_func is None or not callable(infer_func):
            raise ValueError(f"No callable inference entry '{infer_func_name}' found")

        infer_args_config = pipeline_config.get("infer_args", {})
        resolved_infer_args = _resolve_args(infer_args_config, context)
        if config.get("share_candidate_args", False):
            if is_candidate:
                shared_args["value"] = deepcopy(resolved_infer_args)
            else:
                resolved_infer_args = deepcopy(shared_args["value"])
        result = (infer_func(*resolved_infer_args.values()) if pipeline_config.get("args_as_positional", False)
                  else infer_func(**resolved_infer_args))

        return result

    def load_and_run(impl_config: dict, context: dict, is_candidate: bool = True) -> Any:
        """
        加载模块并执行推理

        Args:
            impl_config: 实现配置
            context: 上下文
            is_candidate: 是否为候选实现，True 使用 _load_candidate_modules/_load_candidate_config，
                         False 使用 _load_reference_modules/_load_reference_config
        """
        # 验证必要配置
        run_infer_file = impl_config.get("run_file")
        if not run_infer_file:
            raise ValueError("run_infer_file is required")

        infer_config_name = impl_config.get("config_name")
        # if not infer_config_name:
        #     raise ValueError("infer_config_name is required")

        # 根据 is_candidate 选择不同的加载函数
        config_data = {}
        if is_candidate:
            # 加载候选模块和配置
            module = _load_candidate_modules(run_infer_file)
            if infer_config_name and len(infer_config_name):
                config_data = _load_candidate_config(infer_config_name)
        else:
            # 加载标准模块和配置
            module = _load_reference_modules(run_infer_file)
            if infer_config_name and len(infer_config_name):
                if not (REFERENCE_DIR / infer_config_name).is_file():
                    raise FileNotFoundError(f"Configured reference file missing: {infer_config_name}")
                config_data = _load_reference_config(infer_config_name)

        context = {"config": config_data, "dtype": torch.float32, "device": DEVICE}
        context.update(config_data)

        if impl_config.get("model_file"):
            model_module = (_load_candidate_modules(impl_config["model_file"]) if is_candidate
                            else _load_reference_modules(impl_config["model_file"]))
            model_class = _load_class(model_module, impl_config["model_class"])
            context["model"] = model_class(**_resolve_args(impl_config["model_args"], context)).to(DEVICE)

        # 执行推理
        result = run_inference_pipeline(module, impl_config, context, is_candidate)

        return result

    def compare_tensors_on_cpu(candidate_result: Any, reference_result: Any) -> tuple:
        """
        直接在 CPU 上比较两个推理结果的形状和值
        返回: (shape_match, value_match, shape_info, diff_info)
        """
        # 转换为 CPU tensor
        candidate_tensor = to_cpu_tensor(candidate_result)
        reference_tensor = to_cpu_tensor(reference_result)

        # 检查形状
        shape_match = candidate_tensor.shape == reference_tensor.shape
        shape_info = f"candidate_shape={candidate_tensor.shape}, reference_shape={reference_tensor.shape}"

        if not shape_match:
            return False, False, shape_info, None

        # 比较 Tensor 值；配置 atol/rtol 时使用容差，否则逐值相等
        if "atol" in config or "rtol" in config:
            value_match = (bool(torch.isfinite(candidate_tensor).all()) and bool(torch.isfinite(reference_tensor).all())
                           and torch.allclose(candidate_tensor.to(torch.float64), reference_tensor.to(torch.float64),
                                              atol=config.get("atol", 1e-8), rtol=config.get("rtol", 1e-5)))
        else:
            value_match = torch.equal(candidate_tensor, reference_tensor)

        if not value_match:
            # 找出差异位置
            diff_mask = candidate_tensor != reference_tensor
            if torch.any(diff_mask):
                # 获取第一个差异位置的索引
                if candidate_tensor.dim() == 1:
                    first_diff_idx = torch.where(diff_mask)[0][0].item()
                    diff_info = (
                        f"Values differ at index {first_diff_idx}: "
                        f"{candidate_tensor[first_diff_idx].item()} vs {reference_tensor[first_diff_idx].item()}"
                    )
                else:
                    # 对于多维 tensor，使用 unravel_index 的 tensor 版本
                    flat_idx = torch.argmax(diff_mask.flatten().int())
                    first_diff_idx = np.unravel_index(flat_idx.item(), candidate_tensor.shape)
                    diff_info = (
                        f"Values differ at position {first_diff_idx}: "
                        f"{candidate_tensor[first_diff_idx].item()} vs {reference_tensor[first_diff_idx].item()}"
                    )
            else:
                diff_info = "Tensors are not equal for unknown reason"
            return True, False, shape_info, diff_info

        return True, True, shape_info, None

    current_index = 0
    try:
        # ================================================================
        # 1. 获取候选实现和标准实现的配置
        # ================================================================
        candidate_config = config.get("candidate")
        reference_config = config.get("reference")

        if not candidate_config:
            raise ValueError("config['candidate'] is required")
        if not reference_config:
            raise ValueError("config['reference'] is required")

        # ================================================================
        # 2. 执行候选推理（使用候选实现的完整配置）
        # ================================================================
        candidate_result = load_and_run(candidate_config, {}, is_candidate=True)

        # ================================================================
        # 3. 执行标准推理（使用标准实现的完整配置）
        # ================================================================
        dependencies = {Path(file).stem: _load_reference_modules(file)
                        for file in reference_config.get("import_files", [])}
        with patch.dict(sys.modules, dependencies):
            reference_result = load_and_run(reference_config, {}, is_candidate=False)

        # ================================================================
        # 4. 检查点: shape_match - 检查形状
        # ================================================================
        shape_match, value_match, shape_info, diff_info = compare_tensors_on_cpu(
            candidate_result, reference_result
        )

        update_check(
            current_index,
            shape_match,
            f"Shapes match: {shape_info}" if shape_match else f"Shape mismatch: {shape_info}"
        )
        current_index += 1

        # ================================================================
        # 5. 检查点: value_match - 检查值
        # ================================================================
        update_check(
            current_index,
            value_match,
            "Results match under the configured comparison" if value_match else diff_info
        )

    except Exception:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_interface(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    检查指定文件中的接口定义；配置非法输入时追加当前 Case 的合法基线和拒绝检查。

    config 示例:
    {
        "files": ["train.py", "trainer.py"],
        "required_interfaces": ["train", "main", "Trainer.train"]
    }

    检查点:
        - train_interface:{file}_load: 原 Base 的模块加载结果名；并非仅指训练文件
        - {interface_name}: 检查接口是否在至少一个文件中存在且可调用
        - invalid_*: 仅配置非法输入时执行；准备失败不是拒绝证据
        - interface:configuration: 仅在结果尚无法初始化的配置错误路径追加
        input_contracts 可按顺序列出多个 candidate/非法输入组；单组原配置仍可使用。
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = []
    checks = []

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ------------------ 1. 获取配置 ------------------
        interface_files_files = config.get("files", [])
        if isinstance(interface_files_files, str):
            interface_files_files = [interface_files_files]

        if interface_files_files is None or not len(interface_files_files):
            raise ValueError("config['interface_files'] is missing or empty")

        required_interfaces = config.get("required_interfaces", [])
        if required_interfaces is None or not len(required_interfaces):
            raise ValueError("config['required_interfaces'] is missing or empty")

        interface_type = config.get("phase")
        if interface_type is None:
            raise ValueError("config['phase'] is missing")

        # ------------------ 2. 初始化检查点 ------------------
        for interface_file in interface_files_files:
            check_name = f"train_interface:{interface_file}_load"
            CHECKPOINT_NAMES.append(check_name)
            checks.append((check_name, False, "Not executed"))

        for interface_name in required_interfaces:
            CHECKPOINT_NAMES.append(interface_name)
            checks.append((interface_name, False, "Not executed"))


        # 各输入契约保留自己的 candidate；原单组配置按一个契约处理。
        invalid_start_index = len(CHECKPOINT_NAMES)
        # 非法样例配置尚未解析完成；失败时追加配置错误，不覆盖任何接口/拒绝结果。
        current_index = invalid_start_index
        input_contracts = config.get("input_contracts")
        if input_contracts is None:
            input_contracts = [config] if "invalid_values" in config or "invalid_inputs" in config else []
        invalid_calls = []
        for input_config in input_contracts:
            candidate = input_config["candidate"]
            valid_args_config = candidate["args"]
            if not isinstance(valid_args_config, dict):
                raise TypeError("candidate.args must be a mapping")
            if ("invalid_values" in input_config) == ("invalid_inputs" in input_config):
                raise ValueError("Configure exactly one of invalid_values or invalid_inputs")

            if "invalid_values" in input_config:
                invalid_values = input_config["invalid_values"]
                if not isinstance(invalid_values, dict) or not invalid_values:
                    raise ValueError("invalid_values must be a nonempty parameter-to-values mapping")
                for parameter, values in invalid_values.items():
                    if parameter not in valid_args_config:
                        raise ValueError(f"Unknown parameter in invalid_values: {parameter}")
                    if not isinstance(values, list) or not values:
                        raise ValueError(f"invalid_values[{parameter!r}] must be a nonempty list")
                    for value_index, value in enumerate(values):
                        invalid_args = deepcopy(valid_args_config)
                        invalid_args[parameter] = deepcopy(value)
                        label = f"{parameter}[{value_index}]"
                        invalid_calls.append((candidate, label, [], invalid_args, f"{parameter}={value!r}"))
            else:
                invalid_inputs = input_config["invalid_inputs"]
                if not isinstance(invalid_inputs, dict) or not invalid_inputs:
                    raise ValueError("invalid_inputs must be a nonempty named-input mapping")
                for label, specification in invalid_inputs.items():
                    if not isinstance(label, str) or not label.strip():
                        raise ValueError("Each invalid input needs a nonempty label")
                    if not isinstance(specification, dict) or not isinstance(specification.get("args"), dict):
                        raise TypeError(f"invalid_inputs[{label!r}].args must be a complete mapping")
                    hooks = specification.get("pre_hooks", [])
                    if not isinstance(hooks, list):
                        raise TypeError(f"invalid_inputs[{label!r}].pre_hooks must be a list")
                    invalid_calls.append((candidate, label, hooks, specification["args"], f"input={label}"))


        invalid_names = [
            f"invalid_input:{label}:{result}"
            for _, label, _, _, _ in invalid_calls
            for result in ("valid_call", "rejected")
        ]
        CHECKPOINT_NAMES += invalid_names
        checks += [(name, False, "Not executed due to earlier failure") for name in invalid_names]

        # ------------------ 3. 加载所有模块 ------------------
        modules = []
        for idx, interface_file in enumerate(interface_files_files):
            current_index = idx
            try:
                module = _load_candidate_modules(interface_file)
                modules.append({"file": interface_file, "module": module})
                checks[idx] = (
                    f"train_interface:{interface_file}_load",
                    True,
                    f"Module loaded from {interface_file}"
                )
            except Exception as e:
                checks[idx] = (
                    f"train_interface:{interface_file}_load",
                    False,
                    f"Failed to load {interface_file}: {str(e)}"
                )

        if not modules:
            return checks

        # ------------------ 4. 检查每个接口 ------------------
        base_idx = len(interface_files_files)
        for idx, interface_name in enumerate(required_interfaces):
            check_idx = base_idx + idx
            current_index = check_idx
            found_in_any = False
            found_files = []

            for module_info in modules:
                module = module_info["module"]
                file_name = module_info["file"]

                # 类方法
                if "." in interface_name:
                    class_name, method_name = interface_name.split(".", 1)
                    cls = getattr(module, class_name, None)
                    ok = (
                            cls is not None
                            and callable(getattr(cls, method_name, None))
                    )
                # 模块级对象
                else:
                    obj = getattr(module, interface_name, None)
                    ok = callable(obj)

                if ok:
                    found_in_any = True
                    found_files.append(file_name)

            if found_in_any:
                checks[check_idx] = (
                    interface_name,
                    True,
                    f"{interface_name} found in {', '.join(found_files)}"
                )
            else:
                checks[check_idx] = (
                    interface_name,
                    False,
                    f"{interface_name} not found in any train file"
                )


        # ------------------ 5. 按配置顺序检查各接口的非法输入拒绝行为 ------------------
        if invalid_calls:
            if not all(item[1] for item in checks[:invalid_start_index]):
                return checks
            for input_index, (candidate, label, hooks, invalid_args_config, description) in enumerate(invalid_calls):
                current_index = invalid_start_index + input_index * 2
                try:
                    # ------------------ 1. 每项重新准备运行对象，不共享上次候选模型 ------------------
                    module = _load_candidate_modules(candidate["file"])
                    with _test_runtime(config) as (device, dtype):
                        context = {"dtype": dtype, "device": device, "config": {}}
                        context = _execute_hooks(module, candidate.get("pre_hooks", []), context, globals(), context)
                        # 保留 v10 顺序：公共准备 -> 本项非法 Tensor -> 合法调用 -> 非法调用。
                        context = _execute_hooks(module, hooks, context, globals(), context)
                        function = getattr(module, candidate["function"], None)
                        if function is None:
                            function = context.get(candidate["function"])
                        if not callable(function):
                            raise TypeError("Configured candidate callable was not found")
                        # 合法与非法输入均由 Base 的 $device 生成，模型必须同设备。
                        if isinstance(function, nn.Module):
                            function = function.to(DEVICE)

                        # ------------------ 2. 先确认本项合法基线可执行 ------------------
                        valid_args = _resolve_args(candidate["args"], context)
                        function(**valid_args)
                        update_check(current_index, True, f"{description}; valid baseline completed")
                        current_index += 1

                        # ------------------ 3. 只把非法候选调用中的异常视为拒绝证据 ------------------
                        invalid_args = _resolve_args(invalid_args_config, context)
                        try:
                            function(**invalid_args)
                        except Exception as error:
                            update_check(current_index, bool(str(error).strip()),
                                         f"{description}; {type(error).__name__}: {error}")
                        else:
                            update_check(current_index, False,
                                         f"{description}; expected exception, but call returned")
                except Exception:
                    update_check(current_index, False, f"{description}; {traceback.format_exc()}")
                    # 当前样例失败不退出整组；其他结果仍保留并按序执行。

    except Exception as e:
        error_detail = traceback.format_exc()
        if current_index >= len(checks):
            CHECKPOINT_NAMES.append("interface:configuration")
            checks.append((CHECKPOINT_NAMES[-1], False, "Not executed"))
        update_check(current_index, False, error_detail)

    return checks
