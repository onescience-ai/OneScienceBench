#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FengWu 分组加权 MSE 的规则评测器。

直接复用：check_files。
同名适配：check_method_correctness、check_interface、check_smoke_training。
非法输入并入接口检查；Loss 前向/反向并入冒烟，不增加优化器或参数更新。
本 Case 用 Prompt 公式的 expected，不读取 Standard；reference 仅在显式配置时执行。
"""
from __future__ import annotations

import inspect
import traceback
from copy import deepcopy
from typing import List, Tuple

import torch
from torch import tensor

from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _load_candidate_config,
    _execute_hooks,
    _load_candidate_modules,
    _load_reference_modules,
    _load_class,
    _move_tensors_to_device,
    _resolve_args,
    check_files,
)


def build_fengwu_inputs(pressure_level, batch_size, height, width, seed,
                       part, container, requires_grad=False):
    """生成原评测器的六分支输入，不增加通用 Tensor 配置语言。

    两次调用使用同一局部 Generator 起点。targets 跳过前六次抽样，
    因而与原版“先六个预测，再六个目标”的随机序列完全一致。
    输入准备不计分，不改变全局随机状态。
    """
    if part not in ("predictions", "targets") or container not in ("tuple", "list"):
        raise ValueError("part must be predictions/targets; container must be tuple/list")
    if any(not isinstance(v, int) or v <= 0 for v in (pressure_level, batch_size, height, width)):
        raise ValueError("FengWu fixture dimensions must be positive integers")
    shapes = [(batch_size, 4, height, width)] + [(batch_size, pressure_level, height, width)] * 5
    generator = torch.Generator().manual_seed(seed)
    values = [torch.randn(shape, generator=generator) for shape in shapes]
    if part == "targets":
        values = [torch.randn(shape, generator=generator) for shape in shapes]
    values = [value.to(DEVICE).detach().requires_grad_(requires_grad) for value in values]
    return tuple(values) if container == "tuple" else values


@torch.no_grad()
def _ref_loss(preds, tgts, kw):
    """Prompt 公式：按压力层数取均值，不除以层权之和。"""
    lw = kw.get("level_weights")
    if lw is not None:
        lw = torch.as_tensor(lw, dtype=preds[0].dtype, device=preds[0].device)
    sl = (preds[0] - tgts[0]).square().mean()
    ul = []
    for p, t in zip(preds[1:], tgts[1:]):
        se = (p - t).square()
        ul.append(se.mean() if lw is None else (se.mean(dim=(0, 2, 3)) * lw).mean())
    return kw["surface_weight"] * sl + kw["upper_air_weight"] * torch.stack(ul).mean()


def _execute_hooks_with_save(module, hooks: list, context: dict) -> dict:
    """执行前置调用并保存返回值；FengWu 传 None，只从评测侧查找函数。"""
    for idx, hook_config in enumerate(hooks):
        func_name = hook_config.get("function")
        if not func_name:
            raise ValueError(f"Hook {idx}: 'function' is required")
        hook_func = getattr(module, func_name, None)
        if hook_func is None:
            hook_func = globals().get(func_name)
        if hook_func is None or not callable(hook_func):
            raise ValueError(f"Hook {idx}: function '{func_name}' not found or not callable")
        args_config = hook_config.get("args", {})
        resolved_args = _resolve_args(args_config, context)
        hook_result = hook_func(**resolved_args)
        save_as = hook_config.get("save_as")
        if save_as:
            context[save_as] = hook_result
    return context


def _call_function(function, args):
    """补齐缺失调用辅助：按真实签名绑定，不转换输入容器、不捕获后重试。"""
    signature = inspect.signature(function)
    keywords = dict(args)
    positional = []
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            if parameter.name in keywords:
                positional.append(keywords.pop(parameter.name))
            elif parameter.default is not inspect.Parameter.empty:
                positional.append(parameter.default)
            else:
                raise TypeError(f"Missing positional-only argument: {parameter.name}")
    signature.bind(*positional, **keywords)
    return function(*positional, **keywords)


def _execute_reference_method(reference_config, context, reference_module=None):
    """补齐 Base 的模块级参考函数调用；不读取 YAML，不构建额外执行框架。"""
    reference_file = reference_config.get("file")
    reference_fn_name = reference_config.get("function")
    if not reference_file or not reference_fn_name:
        raise ValueError("reference requires both 'file' and 'function'")
    if reference_module is None:
        reference_module = _load_reference_modules(reference_file)
    reference_method = getattr(reference_module, reference_fn_name, None)
    if reference_method is None or not callable(reference_method):
        raise ValueError(f"Reference function '{reference_fn_name}' not found or not callable")

    # candidate_module 不是输入数据，不能 deepcopy；其余运行输入与候选隔离。
    reference_context = deepcopy({key: value for key, value in context.items()
                                  if key != "candidate_module"})
    reference_context = _execute_hooks_with_save(
        reference_module, reference_config.get("pre_hooks", []), reference_context
    )
    resolved_args = _resolve_args(reference_config.get("args", {}), reference_context)
    resolved_args = _move_tensors_to_device(resolved_args)
    with torch.no_grad():
        reference_result = _call_function(reference_method, resolved_args)
    return (reference_result.detach().clone()
            if isinstance(reference_result, torch.Tensor) else reference_result)


def _compare_outputs(candidate_result, expected_output, rtol=1e-5, atol=1e-8,
                     expected_type="auto"):
    """补齐当前任务的有限 Tensor 数值比较，不转换输入/输出容器。"""
    if expected_type not in ("tensor", "auto"):
        raise ValueError("FengWu output comparison supports tensor/auto only")
    output_match = False
    if not isinstance(candidate_result, torch.Tensor) or not isinstance(expected_output, torch.Tensor):
        match_msg = f"Expected Tensor outputs, got {type(candidate_result).__name__}/{type(expected_output).__name__}"
    elif candidate_result.shape != expected_output.shape:
        match_msg = f"Shape mismatch: actual={tuple(candidate_result.shape)}, expected={tuple(expected_output.shape)}"
    elif not candidate_result.numel() or not bool(torch.isfinite(candidate_result).all()):
        match_msg = "Candidate output is empty or contains NaN/Inf"
    elif not bool(torch.isfinite(expected_output).all()):
        match_msg = "Evaluator expected output contains NaN/Inf"
    else:
        dtype = torch.promote_types(candidate_result.dtype, expected_output.dtype)
        actual = candidate_result.detach().to(device="cpu", dtype=dtype)
        expected = expected_output.detach().to(device="cpu", dtype=dtype)
        output_match = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
        max_abs_diff = float((actual - expected).abs().max().item())
        match_msg = f"finite Tensor comparison: max_abs_diff={max_abs_diff:.8g}, rtol={rtol}, atol={atol}"
    return output_match, match_msg


def _check_shape_with_message(candidate_result, expected_shape):
    """Tensor 精确形状比较；空列表 [] 表示零维标量。"""
    shape_match = (isinstance(candidate_result, torch.Tensor)
                   and tuple(candidate_result.shape) == tuple(expected_shape))
    shape_msg = (f"actual={tuple(candidate_result.shape)}, expected={tuple(expected_shape)}"
                 if isinstance(candidate_result, torch.Tensor)
                 else f"Expected Tensor, got {type(candidate_result).__name__}")
    return shape_match, shape_msg


def check_method_correctness(config: dict,caller_globals=None, caller_locals=None) -> List[tuple]:
    """
    检查候选方法的正确性，通过与期望输出对比或与参考方法对比。

    config 格式:
    {
        "candidate": {
            "file": "train.py",  # 候选方法所在文件
            "function": "relevate2_loss",  # 函数名或类.方法名
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
                "type": "tensor",  # 当前比较辅助只支持 tensor/auto
                "rtol": 1e-6,
                "atol": 1e-8,
                "shape": [10, 10]  # 可选：期望形状
            }
        },
        "reference": {  # 可选：模块级参考函数；当前 Case 未配置
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
        - 如果 candidate 中有 expected，则与 expected 对比
        - 如果 candidate 中没有 expected，但有 reference，则与 reference 对比
        - 如果两者都没有，则只检查方法是否能执行

    检查点:
        - method_check:candidate_loaded: 候选方法加载成功
        - method_check:candidate_executed: 候选方法执行成功
        - method_check:output_matches: 输出匹配（与expected或reference对比）
        - method_check:shape_matches: 形状匹配（如果指定了形状）
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
    context = {"config": {}}  # 初始化上下文
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
        if "." in candidate_fn_name:
            model_class_name, method_name = candidate_fn_name.split(".", 1)
            model_cls = _load_class(candidate_module, model_class_name)
            if not issubclass(model_cls, torch.nn.Module):
                raise TypeError(f"{model_class_name} must be a torch.nn.Module subclass")
            candidate_method = getattr(model_cls, method_name, None)
        else:
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
        context = _execute_hooks_with_save(None, pre_hooks, context)

        # ------------------ 3. 解析候选方法的参数 ------------------
        args_config = candidate_config.get("args", {})
        resolved_args = _resolve_args(args_config, context)
        resolved_args = _move_tensors_to_device(resolved_args)

        # ------------------ 4. 执行候选方法 ------------------
        if "." in candidate_fn_name:
            model_args = candidate_config.get("model_args")
            if model_args is None:
                raise ValueError("candidate['model_args'] is required for a class method")
            resolved_model_args = _resolve_args(model_args, context)
            model = model_cls(**resolved_model_args).to(DEVICE).eval()
            candidate_method = getattr(model, method_name)
        candidate_result = _call_function(candidate_method, resolved_args)
        context["candidate_result"] = candidate_result

        update_check(
            current_index,
            True,
            f"Candidate function executed successfully with args={resolved_args}"
        )
        current_index += 1

        # ------------------ 5. 确定期望输出 ------------------
        expected_config = candidate_config.get("expected")

        if expected_config:
            # 模式1：与期望输出对比
            expected_output = _resolve_args(
                {"output": expected_config.get("output")},
                context
            )["output"]
            compare_mode = "expected"

        elif reference_config:
            # 模式2：与参考方法对比
            expected_output = _execute_reference_method(
                reference_config,
                context,
                reference_module
            )
            compare_mode = "reference"

        else:
            # 模式3：只检查执行，不对比
            update_check(
                current_index,
                True,
                "No expected or reference config, skipping output comparison"
            )
            current_index += 1
            # 跳过形状检查
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

        output_match, match_msg = _compare_outputs(
            candidate_result,
            expected_output,
            rtol=rtol,
            atol=atol,
            expected_type=expected_type
        )

        update_check(
            current_index,
            output_match,
            f"Output comparison ({compare_mode}): {match_msg}"
        )
        current_index += 1

        # ------------------ 7. 检查形状（如果指定） ------------------
        expected_shape = expected_config.get("shape") if expected_config else None

        if expected_shape is not None:
            shape_match, shape_msg = _check_shape_with_message(candidate_result, expected_shape)
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


def check_smoke_training(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple]:
    """
    运行最小入口冒烟；当前 Case 构造 Loss 并执行前向／反向，不运行 optimizer 或参数更新。

    config 示例:
    {
        "run_file": "train.py",           # 通用示例；本 Case 使用 fengwu_loss_modification.py
        "config_name": "config.yaml",          # 可选；指定后加载失败不回退
        "train_fn_name": "train",           # 通用示例；本 Case 为 forward，另配 model_class/model_args
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
        - training:smoke_test: 检查配置入口是否正常返回
        - loss_backward:scalar_finite / requires_grad / backward_completed / gradients_finite:
          配置 model_class 时检查有限标量、反向和六分支梯度；允许零梯度
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "training:create_work_dir",
        "training:smoke_test",
    ]
    if config.get("model_class"):
        CHECKPOINT_NAMES += [
            "loss_backward:scalar_finite",
            "loss_backward:requires_grad",
            "loss_backward:backward_completed",
            "loss_backward:gradients_finite",
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
        if config.get("model_class"):
            train_fn = _load_class(train_module, config["model_class"])
            if not issubclass(train_fn, torch.nn.Module):
                raise TypeError("Loss class must be a torch.nn.Module subclass")
        else:
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

            if config.get("model_class"):
                # Loss 样例生成函数属于评测侧，不能使用候选模块的同名函数。
                hook_func = globals().get(func_name)
            else:
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

        if config.get("model_class"):
            model_args = _resolve_args(config["model_args"], context)
            model = train_fn(**model_args).to(DEVICE)
            train_fn = getattr(model, train_fn_name)
            resolved_train_args = _move_tensors_to_device(resolved_train_args)
            predictions = resolved_train_args["predictions"]
            if len(predictions) != 6 or any(not value.is_leaf or not value.requires_grad for value in predictions):
                raise ValueError("Evaluator backward inputs must be six gradient-enabled leaf Tensors")

        # 执行配置入口：当前 Case 为 Loss.forward；普通函数分支保留。
        if config.get("model_class"):
            with torch.enable_grad():
                result = _call_function(train_fn, resolved_train_args)
        else:
            result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)

        update_check(
            current_index,
            True,
            f"Smoke test completed successfully: {train_fn_name}(...) with {config_name}"
        )

        # ------------------ 8. 当前 Loss 的最小反向传播闭环 ------------------
        if config.get("model_class"):
            current_index += 1
            if not isinstance(result, torch.Tensor) or result.ndim != 0 or not bool(torch.isfinite(result)):
                raise ValueError("Loss must be a finite scalar Tensor")
            update_check(current_index, True, f"scalar loss={result.detach().item()}")

            current_index += 1
            if not result.requires_grad:
                raise ValueError("Loss is detached from autograd")
            update_check(current_index, True, "Loss requires grad")

            current_index += 1
            result.backward()
            update_check(current_index, True,
                         "loss.backward() completed; no optimizer or training loop was run")

            current_index += 1
            failed_branches = [index for index, value in enumerate(predictions)
                               if value.grad is None or not bool(torch.isfinite(value.grad).all())]
            update_check(current_index, not failed_branches,
                         f"Missing/nonfinite prediction gradients at branch indices={failed_branches}; zero gradients allowed")

    except Exception as e:
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
        invalid_values 与 invalid_inputs 可在同一模型配置下同时提供，先检查构造参数再检查 Forward。
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


        # 当前 Case 的非法输入结果随接口结果一起预分配；没有配置时只运行原接口检查。
        invalid_start_index = len(CHECKPOINT_NAMES)
        # 非法样例配置尚未解析完成；失败时追加配置错误，不覆盖任何接口/拒绝结果。
        current_index = invalid_start_index
        invalid_values = config.get("invalid_values", {})
        invalid_inputs = config.get("invalid_inputs", [])
        if "invalid_values" in config or "invalid_inputs" in config:
            if not invalid_values and not invalid_inputs:
                raise ValueError("Configure at least one nonempty invalid_values/invalid_inputs group")
            invalid_names = [f"invalid_constructor:{key}:{index}"
                             for key, values in invalid_values.items() for index in range(len(values))]
            invalid_names += [f"invalid_forward:{item['id']}" for item in invalid_inputs]
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


        # ------------------ 5. 检查当前接口的非法输入拒绝行为 ------------------
        if "invalid_values" in config or "invalid_inputs" in config:
            if not all(item[1] for item in checks[:invalid_start_index]):
                return checks
            try:
                model_module = _load_candidate_modules(config["model_file"])
                model_cls = _load_class(model_module, config["model_class"])
                if not issubclass(model_cls, torch.nn.Module):
                    raise TypeError("Loss class must be a torch.nn.Module subclass")
                context = {"config": {}}
                context.update(nan=float("nan"), inf=float("inf"))
                model_args = _resolve_args(config["model_args"], context)
            except Exception as e:
                detail = "Setup failed, not rejection evidence: " + traceback.format_exc()
                for index in range(invalid_start_index, len(checks)):
                    update_check(index, False, detail)
                return checks

            current_index = invalid_start_index
            for parameter, values in invalid_values.items():
                for value in values:
                    try:
                        # 合法构造在异常判定之外；不能把总是报错的构造器算作拒绝通过。
                        model_cls(**deepcopy(model_args))
                        invalid_model_args = deepcopy(model_args)
                        invalid_model_args[parameter] = _resolve_args({"value": value}, context)["value"]
                    except Exception as e:
                        update_check(current_index, False,
                            "Legal baseline/setup failed: " + traceback.format_exc()
                        )
                        current_index += 1
                        continue
                    try:
                        model_cls(**invalid_model_args)
                    except (ValueError, TypeError, RuntimeError) as e:
                        update_check(current_index, True,
                            f"Rejected {parameter}={value!r}: {type(e).__name__}: {e}"
                        )
                    except Exception as e:
                        update_check(current_index, False,
                            "Unexpected exception, not rejection evidence: " + traceback.format_exc()
                        )
                    else:
                        update_check(current_index, False,
                            f"Accepted invalid {parameter}={value!r}"
                        )
                    current_index += 1

            for item in invalid_inputs:
                try:
                    # 每个样例重新生成输入、构建模型；合法调用不共享给非法调用的可变状态。
                    input_context = _execute_hooks(None, config.get("pre_forward_hooks", []), dict(context),
                                                   globals(), caller_locals)
                    forward_args = _move_tensors_to_device(_resolve_args(config["forward_args"], input_context))
                    model = model_cls(**deepcopy(model_args)).to(DEVICE).eval()
                    with torch.no_grad():
                        output = _call_function(model.forward, deepcopy(forward_args))
                    if not isinstance(output, torch.Tensor) or output.ndim != 0 or not bool(torch.isfinite(output)):
                        raise ValueError("Legal baseline did not return a finite scalar Tensor")
                    model = model_cls(**deepcopy(model_args)).to(DEVICE).eval()

                    argument = item["argument"]
                    if argument not in ("predictions", "targets", "both"):
                        raise ValueError(f"Unsupported argument: {argument}")
                    arguments = ("predictions", "targets") if argument == "both" else (argument,)
                    for argument in arguments:
                        values = forward_args[argument]
                        if "length" in item:
                            length = item["length"]
                            if not isinstance(length, int) or length < 0 or length == 6:
                                raise ValueError("Invalid branch-count fixture length")
                            changed = list(values[:length])
                            while len(changed) < length:
                                changed.append(values[-1].clone())
                            forward_args[argument] = type(values)(changed)
                        elif "container" in item:
                            if item["container"] != "dict":
                                raise ValueError("Only dict container rejection is configured for this task")
                            forward_args[argument] = dict(enumerate(values))
                        elif "shape" in item or "value" in item:
                            branch = item["branch"]
                            changed = list(values)
                            changed[branch] = (torch.zeros(item["shape"], device=values[branch].device,
                                                           dtype=values[branch].dtype)
                                               if "shape" in item else item["value"])
                            forward_args[argument] = type(values)(changed)
                        else:
                            raise ValueError(f"Invalid input fixture has no replacement: {item}")
                except Exception as e:
                    update_check(current_index, False,
                        "Legal baseline/setup failed: " + traceback.format_exc()
                    )
                    current_index += 1
                    continue

                try:
                    with torch.no_grad():
                        _call_function(model.forward, forward_args)
                except (ValueError, TypeError, RuntimeError) as e:
                    update_check(current_index, True,
                        f"Rejected {item['id']}: {type(e).__name__}: {e}"
                    )
                except Exception as e:
                    update_check(current_index, False,
                        "Unexpected exception, not rejection evidence: " + traceback.format_exc()
                    )
                else:
                    update_check(current_index, False, f"Accepted invalid input: {item['id']}")
                current_index += 1

    except Exception as e:
        error_detail = traceback.format_exc()
        if current_index >= len(checks):
            CHECKPOINT_NAMES.append("interface:configuration")
            checks.append((CHECKPOINT_NAMES[-1], False, "Not executed"))
        update_check(current_index, False, error_detail)

    return checks
