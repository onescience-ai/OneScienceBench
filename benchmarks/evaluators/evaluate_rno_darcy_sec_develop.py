#!/usr/bin/env python3
"""RNO-Darcy evaluator for OnescienceBench (Base-reuse revision 14).

The Prompt is the acceptance contract.  The legacy evaluator is used only when
an existing check boundary is unclear.  Public Base methods are reused first;
same-name overrides keep the Base execution skeleton and add only the minimum
RNO-specific checks that the public method cannot express directly.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
import logging
import pickle
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import inspect
import traceback
from pathlib import Path
from typing import Any, List, Tuple

import torch
import yaml

from base.base_evaluator import (
    CANDIDATE_DIR, DEVICE,
    _load_candidate_modules, _load_candidate_config, _load_class,
    _resolve_args, _execute_hooks, _convert_arg_to_type, _move_tensors_to_device,
    _compute_mse, _compute_mae,
    check_files, check_interface,
)


def _rno_model_roles(model, spec):
    """Resolve explicitly reviewed role paths, not required candidate attribute names.

    Unknown/ambiguous bindings require review. Selection never filters on the
    expected channel/kernel/count: a malformed bound module must still be tested.
    """
    modules = dict(model.named_modules())
    roles = {}
    for role in ("fine", "coarse"):
        paths = [p for p in spec[role] if p in modules]
        if len(paths) != 1:
            raise ValueError(f"REVIEW_REQUIRED: bind {role} Radon branch; matched={paths}")
        path = paths[0]
        blocks = []
        for name, block in modules[path].named_children():
            weights = [(key, child) for key, child in block.named_children()
                       if key in spec["angle_network"]]
            if len(weights) != 1:
                raise ValueError(f"REVIEW_REQUIRED: bind angle network in {path}.{name}")
            key, network = weights[0]
            block_path = f"{path}.{name}"
            convs = [(f"{block_path}.{key}.{n}", m) for n, m in network.named_modules()
                     if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d))]
            blocks.append((block_path, block, f"{block_path}.{key}", network, convs))
        roles[role] = (path, modules[path], blocks)
    return roles


@contextmanager
def _rno_probe_fixture(candidate_config, spec, parent):
    """Tiny real MAT inputs and isolated runtime config for candidate entry tests.

    Only paths, sample/batch counts and duration are overridden. Model sizes,
    physics formulae and candidate source bytes are not changed. Layout adapters
    are explicit; an unknown loader/config schema is REVIEW_REQUIRED, not PASS.
    """
    import scipy.io
    with tempfile.TemporaryDirectory(prefix="rno-probe-", dir=parent) as directory:
        work = Path(directory)
        cfg = deepcopy(candidate_config)
        replacements = {
            "data_root": str(work), "train_samples": 2, "test_samples": 1,
            "batch_size": 2, "eval_batch_size": 1, "epochs": 1,
            "num_workers": 0, "output_dir": str(work / "results"),
            "checkpoint_dir": str(work / "results" / "checkpoint"),
            "cache_dir": str(work / "cache"), "device": "cpu", "amp": False,
        }
        for role, paths in spec["overrides"].items():
            changed = 0
            for path in paths:
                keys = path.split("."); node = cfg
                for key in keys[:-1]:
                    node = node.get(key, {}) if isinstance(node, dict) else {}
                if isinstance(node, dict) and keys[-1] in node:
                    node[keys[-1]] = replacements[role]; changed += 1
            if not changed and role in {"data_root", "train_samples", "test_samples", "epochs", "batch_size", "output_dir"}:
                raise ValueError(f"REVIEW_REQUIRED: bind runtime override {role}")
        # Smooth + discontinuous positive permeability, nonzero pressure mean.
        axis = np.linspace(0, 1, 421, dtype=np.float32)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        a = 2 + x + 0.3*y + 3*(x > 0.5)
        u = (1 + 0.2*x)*np.sin(np.pi*x)*np.sin(np.pi*y)
        train_a = np.stack([a, a + 0.7]); train_u = np.stack([u, u * 1.2])
        test_a = (a + 1.4)[None]; test_u = (u * 1.4)[None]
        for name, aa, uu in ((cfg["data"]["train_file"], train_a, train_u),
                             (cfg["data"]["test_file"], test_a, test_u)):
            scipy.io.savemat(work / name, {"coeff": aa, "sol": uu})
        (work / "results" / "checkpoint").mkdir(parents=True)
        config_path = work / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        dataset_module = _load_candidate_modules(spec["dataset_file"])
        matches = [b for b in spec["dataset_builders"]
                   if callable(getattr(dataset_module, b["function"], None))]
        if len(matches) != 1:
            raise ValueError("REVIEW_REQUIRED: bind one candidate dataset builder")
        binding = matches[0]
        context = _execute_hooks(dataset_module, [{"function": binding["function"],
            "args": {"config": "$config"}, "save_as": "datasets"}], {"config": cfg})
        built = context["datasets"]
        train_set, test_set = built[:2]
        if binding["normalizers"] == "objects":
            xm, ym = built[2:4]
            stats = dict(input_mean=float(xm.mean.item()), input_std=float(xm.std.item()),
                         target_mean=float(ym.mean.item()), target_std=float(ym.std.item()))
        else:
            stats = built[2]
        probe = dict(work=work, config=cfg, config_path=config_path, binding=binding,
                     train_set=train_set, test_set=test_set, stats=stats,
                     train_a=torch.from_numpy(train_a[:, ::5, ::5].copy()),
                     train_u=torch.from_numpy(train_u[:, ::5, ::5].copy()),
                     test_a=torch.from_numpy(test_a[:, ::5, ::5].copy()),
                     test_u=torch.from_numpy(test_u[:, ::5, ::5].copy()))
        argv = sys.argv[:]
        profile = sys.getprofile()
        loggers = {name: (obj, obj.handlers[:], obj.level, obj.propagate)
                   for name, obj in logging.Logger.manager.loggerDict.items()
                   if isinstance(obj, logging.Logger)}
        try:
            yield probe
        finally:
            sys.setprofile(profile)
            sys.argv = argv
            for name, obj in list(logging.Logger.manager.loggerDict.items()):
                if not isinstance(obj, logging.Logger):
                    continue
                old = loggers.get(name, (obj, [], obj.level, obj.propagate))
                for handler in obj.handlers:
                    if handler not in old[1]:
                        handler.close()
                obj.handlers = old[1]; obj.setLevel(old[2]); obj.propagate = old[3]



def _rno_training_observer(probe):
    """Observe actual candidate calls; do not replace loss/model/optimizer results.

    The reviewed call bindings cover split-loss and composite-loss APIs. Unknown
    inlined implementations need an adapter. Profiling is scoped to one entry call.
    """
    observed = {key: [] for key in ("pressure", "pde", "boundary", "cache", "losses", "gradients", "updates")}
    observed["errors"] = []
    observed["details"] = {}
    current = {}
    optimizer_frames = {}
    stats = probe["stats"]
    candidate_prefix = str(CANDIDATE_DIR.resolve()) + "/"
    def flat(value):
        return value.detach().cpu().reshape(value.shape[0], -1)
    def near(left, right):
        return left.shape == right.shape and torch.allclose(left.to(torch.float64), right.to(torch.float64), atol=2e-4, rtol=2e-4)
    def physical_variants(prediction):
        raw = flat(prediction) * stats["target_std"] + stats["target_mean"]
        zeroed = raw.reshape(-1, 85, 85).clone()
        zeroed[:, (0, -1), :] = 0; zeroed[:, :, (0, -1)] = 0
        return raw, zeroed.reshape(raw.shape)
    def record(frame, event, value):
        try:
            if event not in {"call", "return"}:
                return
            name = frame.f_code.co_name
            if name == "step" and isinstance(frame.f_locals.get("self"), torch.optim.Optimizer):
                optimizer = frame.f_locals["self"]
                parameters = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                if event == "call":
                    grads = [p.grad for p in parameters if p.grad is not None]
                    observed["gradients"].append(bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads))
                    optimizer_frames[id(frame)] = [p.detach().clone() for p in parameters]
                else:
                    before = optimizer_frames.pop(id(frame), None)
                    if before is not None:
                        observed["updates"].append(any(not torch.equal(p.detach(), v) for p,v in zip(parameters,before)))
                return
            if name == "backward" and event == "call":
                loss = frame.f_locals.get("self", frame.f_locals.get("tensors"))
                values = [loss] if isinstance(loss,torch.Tensor) else loss if isinstance(loss,(tuple,list)) else []
                if values and all(isinstance(v,torch.Tensor) for v in values):
                    observed["losses"].append(all(bool(torch.isfinite(v).all()) for v in values))
            if not frame.f_code.co_filename.startswith(candidate_prefix):
                return
            local = frame.f_locals
            instance = local.get("self")
            if name == "forward" and isinstance(instance, torch.nn.Module) and type(instance).__name__ == "RadonNeuralOperator2D":
                if event == "call" and torch.is_grad_enabled():
                    field = local.get("field", local.get("fx"))
                    if isinstance(field, torch.Tensor):
                        physical = flat(field) * stats["input_std"] + stats["input_mean"]
                        references = probe["train_a"].reshape(2, -1)
                        indices = ((physical[:, None] - references[None]).square().mean(-1)).argmin(-1)
                        current.clear(); current.update(indices=indices, pressure_seen=False,
                            coefficient=references[indices], target=probe["train_u"].reshape(2, -1)[indices],
                            coarse=local.get("coarse_field"))
                elif event == "return" and torch.is_grad_enabled() and isinstance(value, torch.Tensor):
                    current["prediction"] = value
                return
            if not torch.is_grad_enabled() or "prediction" not in current:
                return
            raw, bc = physical_variants(current["prediction"])
            pressure_call = name == "relative_l2" or (name == "forward" and type(instance).__name__ == "RelativeL2Loss")
            if event == "call" and pressure_call and not current["pressure_seen"]:
                pred = local.get("prediction", local.get("pred")); target = local.get("target")
                if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
                    current["pressure_seen"] = True
                    target_ok = near(flat(target), current["target"])
                    pred_ok = near(flat(pred), raw) or near(flat(pred), bc)
                    observed["pressure"].append(target_ok and pred_ok)
                    observed["details"].setdefault("pressure",[]).append({"target_in_physical_units":target_ok,"prediction_in_physical_units":pred_ok})
            if event == "call" and name in {"darcy_pde_residual", "darcy_pde_residual_loss"}:
                pred = local.get("prediction", local.get("u"))
                coefficient = local.get("coefficient", local.get("a"))
                coefficient_ok = near(flat(coefficient), current["coefficient"])
                pred_ok = near(flat(pred), raw) or near(flat(pred), bc)
                observed["pde"].append(coefficient_ok and pred_ok)
                observed["details"].setdefault("pde",[]).append({"coefficient_in_physical_units":coefficient_ok,"pressure_in_physical_units":pred_ok})
                indices = probe["binding"].get("aux_indices")
                weights = local.get("importance", local.get("interface_weights"))
                if weights is None:
                    # Composite API supplies the weight to its immediate caller.
                    weights = frame.f_back.f_locals.get("interface_weights")
                coarse = current.get("coarse")
                cache_ok = False
                if indices is not None and isinstance(coarse, torch.Tensor) and isinstance(weights, torch.Tensor):
                    rows = [probe["train_set"][int(i)] for i in current["indices"]]
                    expected_c = torch.stack([torch.as_tensor(row[indices[0]]) for row in rows]).reshape(raw.shape[0], -1)
                    expected_w = torch.stack([torch.as_tensor(row[indices[1]]) for row in rows]).reshape(raw.shape[0], -1)
                    cache_ok = near(flat(coarse), expected_c) and near(flat(weights), expected_w)
                observed["cache"].append(cache_ok)
                observed["details"].setdefault("cache",[]).append({"cached_sample_fields_bound":indices is not None,"coarse_passed_to_model":isinstance(coarse,torch.Tensor),"cached_values_consumed":cache_ok})
            if event == "return" and name in {"boundary_mse", "zero_dirichlet_boundary_loss"}:
                field = local.get("prediction", local.get("u"))
                raw_grid = raw.reshape(-1, 85, 85)
                raw_boundary = torch.cat((raw_grid[:,0], raw_grid[:,-1], raw_grid[:,:,0], raw_grid[:,:,-1]),1)
                value_ok = isinstance(value, torch.Tensor) and bool(torch.isfinite(value))
                # Zero is legitimate when the model itself already enforces the physical boundary.
                active = float(raw_boundary.square().mean()) <= 1e-10
                if value_ok and value.requires_grad and not active:
                    gradient = torch.autograd.grad(value, current["prediction"], retain_graph=True, allow_unused=True)[0]
                    active = float(value.detach()) > 1e-10 and gradient is not None and bool((gradient.detach() != 0).any())
                physical = near(flat(field),raw)
                observed["boundary"].append(value_ok and active and physical)
                observed["details"].setdefault("boundary",[]).append({"physical_raw_pressure":physical,"finite_penalty":value_ok,"active_penalty_or_model_hard_boundary":active})
        except Exception as exc:
            observed["errors"].append(f"{type(exc).__name__}: {exc}")
    return observed, record


def _rno_pde_numeric_probe():
    """Compare the candidate's finite-volume helper on a variable positive field.

    Uses unit forcing and unit importance, so normalized/unnormalized weighted
    means coincide. Never compares a candidate model with Standard weights.
    """
    module = _load_candidate_modules("utils.py")
    axis = torch.linspace(0, 1, 7, dtype=torch.float64)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    u = (x*x + 0.3*x*y + y*y)[None]
    a = (1 + x + 3*(x > 0.5) + 0.2*y)[None]
    h = 1.0 / 6
    east = 2*a[:,:,:-1]*a[:,:,1:]/(a[:,:,:-1]+a[:,:,1:])
    north = 2*a[:,:-1,:]*a[:,1:,:]/(a[:,:-1,:]+a[:,1:,:])
    fx = east*(u[:,:,1:]-u[:,:,:-1])/h
    fy = north*(u[:,1:,:]-u[:,:-1,:])/h
    reference = -(fx[:,1:-1,1:]-fx[:,1:-1,:-1]+fy[:,1:,1:-1]-fy[:,:-1,1:-1])/h - 1
    functions = [n for n in ("darcy_pde_residual", "darcy_pde_residual_loss") if callable(getattr(module,n,None))]
    if len(functions) != 1:
        raise ValueError("REVIEW_REQUIRED: bind one candidate finite-volume helper")
    name = functions[0]
    if name == "darcy_pde_residual":
        actual = getattr(module,name)(u[:,None], a[:,None], 1.0, h).reshape(reference.shape)
        expected = reference
    else:
        actual = getattr(module,name)(u.reshape(1,-1), a.reshape(1,-1), torch.ones(1,49,dtype=u.dtype),7,h)
        expected = reference.square().mean()
    passed = (isinstance(actual,torch.Tensor) and bool(torch.isfinite(actual).all())
              and torch.allclose(actual,expected,atol=1e-6,rtol=1e-6))
    return passed, f"Candidate {name} on variable positive permeability; max difference={float((actual-expected).abs().max()):.6g}; harmonic finite-volume oracle"


def check_config_content(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple[str, bool, str]]:
    """
    检查配置声明；保留 Base 的字段存在、类型和值比较。

    config_paths 只列已核对的候选 YAML 路径，用 Base _resolve_args 读取。
    不再扫描源码猜默认值或按同名叶键猜层级；未知/冲突接入记 REVIEW_REQUIRED。
    普通模式、字段存在、类型和值比较沿用 Base。声明正确不等于运行时使用正确。

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

        unresolved = {}
        if config.get("config_paths"):
            context = {"config": config_data}
            config_data = {}
            for spec in normalized_specs:
                key = spec["key"]
                paths = config["config_paths"].get(key, [])
                values = _resolve_args({path: "$config." + path for path in paths}, context)
                located = [(path, value) for path, value in values.items() if value is not None]
                if key in config.get("path_keys", []):
                    located = [(path, str(Path(str(value)))) for path, value in located]
                if not located or any(value != located[0][1] for _, value in located[1:]):
                    unresolved[key] = f"REVIEW_REQUIRED: bind an unambiguous declaration for {key!r}: {located}; not a mandatory private YAML key"
                else:
                    config_data[key] = located[0][1]

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
                    unresolved.get(key, f"Required key '{key}' not found in config")
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
    """检查真实微型数据管线；不再用源码变量名推断下采样和归一化。

    runtime_probe 的 builder/样本索引是接入描述，不是 Prompt 私有接口要求。
    临时 MAT 仍为 421×421，2 个训练/1 个测试样本；默认 1000/200 由配置检查。
    与 Base 的 candidate/reference 全流程对比方法职责不同，不读取 Standard。
    缓存仅检查已接入内存布局的生产和样本取用；训练消费另由训练 smoke 检查。
    """
    CHECKPOINT_NAMES = [
        "data:source_parse",
        "data:mat_sample_and_downsample",
        "data:train_only_normalization",
        "data:multiscale_cache_and_interface",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], bool(passed), msg)

    current_index = 0
    try:
        dataset_file = config.get("dataset_file", "dataset.py")
        path = CANDIDATE_DIR / dataset_file
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        update_check(current_index, True, f"Python syntax parsed: {dataset_file}")

        current_index += 1
        with _rno_probe_fixture(_load_candidate_config(config["config_name"]),
                                config["runtime_probe"], CANDIDATE_DIR) as probe:
            binding, stats = probe["binding"], probe["stats"]
            train_set, test_set = probe["train_set"], probe["test_set"]
            matched = len(train_set) == 2 and len(test_set) == 1
            details = []
            for split, dataset in (("train", train_set), ("test", test_set)):
                samples = [dataset[index] for index in range(len(dataset))]
                coefficients = torch.stack([
                    torch.as_tensor(row[binding["input_index"]]) for row in samples
                ]).reshape(len(samples), -1)
                targets = torch.stack([
                    torch.as_tensor(row[binding["target_index"]]) for row in samples
                ]).reshape(len(samples), -1)
                coefficients = coefficients * stats["input_std"] + stats["input_mean"]
                if split == "train" or binding["test_target_normalized"]:
                    targets = targets * stats["target_std"] + stats["target_mean"]
                expected_a = probe[split + "_a"].reshape(len(samples), -1)
                expected_u = probe[split + "_u"].reshape(len(samples), -1)
                shape_ok = coefficients.shape == expected_a.shape and targets.shape == expected_u.shape
                values_ok = (
                    shape_ok
                    and torch.allclose(coefficients.double(), expected_a.double(), rtol=2e-5, atol=2e-5)
                    and torch.allclose(targets.double(), expected_u.double(), rtol=2e-5, atol=2e-5)
                )
                matched = matched and values_ok
                details.append(f"{split}: shape={shape_ok}, decoded coeff/sol match stride-5 reference={values_ok}")
            update_check(current_index, matched, "; ".join(details) + "; temporary data only")

            current_index += 1
            normalization_ok = True
            for role, expected in (("input", probe["train_a"]), ("target", probe["train_u"])):
                mean, std = stats[role + "_mean"], stats[role + "_std"]
                # Prompt 未固定 population/sample std；两种全局标量约定均接受。
                correct_mean = (
                    np.isfinite(mean)
                    and np.isclose(mean, float(expected.mean()), rtol=2e-5, atol=2e-5)
                )
                correct_std = np.isfinite(std) and std > 0 and any(
                    np.isclose(std, float(expected.std(correction=correction)), rtol=2e-5, atol=2e-5)
                    for correction in (0, 1))
                normalization_ok = normalization_ok and correct_mean and correct_std
            update_check(current_index, normalization_ok and matched,
                         f"Actual scalar train statistics={stats}; train/test decoded values matched={matched}; "
                         "training-only statistics checked on 2/1 temporary samples, not a universal leakage proof")

            current_index += 1
            indices = binding.get("aux_indices")
            if indices is None:
                update_check(current_index, False,
                             "Executed dataset builder returns the reviewed fine-only layout; no coarse/interface fields are supplied. "
                             "A different cache-consumer interface requires an explicit adapter; dead source markers cannot pass.")
            else:
                one, again = train_set[0], train_set[0]
                coarse, weights = [torch.as_tensor(one[i]) for i in indices]
                coarse2, weights2 = [torch.as_tensor(again[i]) for i in indices]
                valid = (coarse.numel() == 43*43 and weights.numel() == 85*85
                         and bool(torch.isfinite(coarse).all() and torch.isfinite(weights).all())
                         and bool((weights >= 1).all() and (weights <= 8).all())
                         and float(weights.max()) > float(weights.min()))
                repeated = torch.equal(coarse, coarse2) and torch.equal(weights, weights2)
                stored = []
                for value in vars(train_set).values():
                    values = value.values() if isinstance(value, dict) else [value]
                    for array in values:
                        if isinstance(array, (torch.Tensor, np.ndarray)) and array.ndim and array.shape[0] == len(train_set):
                            stored.append(torch.as_tensor(array[0]))
                backing = [[row for row in stored if row.shape == wanted.shape and torch.equal(row, wanted)]
                           for wanted in (coarse, weights)]
                live = False
                if all(len(matches) == 1 for matches in backing):
                    cached_c, cached_w = backing[0][0], backing[1][0]
                    old_c, old_w = cached_c.clone(), cached_w.clone()
                    try:
                        cached_c.fill_(0.123); cached_w.fill_(2.345)
                        consumed = train_set[0]
                        live = (torch.equal(torch.as_tensor(consumed[indices[0]]), cached_c)
                                and torch.equal(torch.as_tensor(consumed[indices[1]]), cached_w))
                    finally:
                        cached_c.copy_(old_c); cached_w.copy_(old_w)
                update_check(current_index, valid and repeated and live,
                             f"Actual temporary dataset: coarse/interface shapes valid={valid}, "
                             f"repeat values stable={repeated}, sample consumes stored buffers={live}; "
                             "RAM-cache layout probe, not a universal disk/lazy-cache proof. Training consumption checked separately.")
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: dict, caller_globals=None, caller_locals=None):
    """
    检查模型的结构是否符合预期。

    model_config_arg 可显式指定完整 YAML 的形参名；仅在候选声明该参数时注入。
    不猜测子配置、不捕获 TypeError 重试；未声明时保持原 model_args 构造。

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
        - model_structure:structure: 模型结构
        - model_structure:transformer_layers: Transformer 层/注意力容器的有限注册证据
        - model_structure:angle_conv_in/out: JSON 公共 conv_shape rules，检查 Conv 1→16 和 16→1 的存在证据
        - model_structure:angle_activations: GELU/Sigmoid 注册或函数调用证据
        - model_structure:multiscale_radon: 实际细/粗分支角度；块数交给公共 children_count
        - model_structure:gated_local_fusion: 门控/局部融合有限源码证据
        - model_structure:radon_grid_cache: Radon 网格缓存有限源码证据

    role_paths 只负责已审查候选的调用接入，不作为 Prompt 的私有命名约束。
    rules 的 $fine_branch/$coarse_branch/$angle_conv_in/$angle_conv_out 在定位
    对应分支后展开，仍交给公共 checker；不会选中 local_feature 等无关卷积。
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
        if model_args is None:
            raise ValueError("config['model_args'] is required; {} requests candidate defaults")

        resolved_model_args = _resolve_args(model_args, context)
        config_arg = config.get("model_config_arg")
        if config_arg and config_arg in inspect.signature(model_cls).parameters:
            resolved_model_args.setdefault(config_arg, context["config"])
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
        roles = _rno_model_roles(model, config["role_paths"]) if config.get("role_paths") else None
        if roles is not None:
            expanded = []
            contexts = [{"fine_branch": roles["fine"][0], "coarse_branch": roles["coarse"][0]}]
            for role, (_, _, blocks) in roles.items():
                for index, (_, _, _, _, convs) in enumerate(blocks):
                    if len(convs) != 2:
                        raise ValueError(f"REVIEW_REQUIRED: bind two angle convolutions in {role}[{index}]")
                    contexts.append({"suffix": f"{role}_{index}", "angle_conv_in": convs[0][0],
                                     "angle_conv_out": convs[1][0], "conv_ndim": len(convs[0][1].kernel_size)})
            for rule in rules:
                locator = rule.get("params", {}).get("target_path", "")
                selected = contexts[1:] if locator in {"$angle_conv_in", "$angle_conv_out"} else contexts[:1]
                for binding in selected:
                    item = _resolve_args(rule, binding)
                    if "suffix" in binding:
                        item["id"] += "_" + binding["suffix"]
                        kernel = item["params"].get("kernel_size")
                        if isinstance(kernel, int):
                            item["params"]["kernel_size"] = [kernel] * binding["conv_ndim"]
                    expanded.append(item)
            rules = expanded
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

        # ------------------ 3. RNO Prompt 特有的有限结构证据 ------------------
        modules = list(model.modules())
        # 不强制 PyTorch 内置注意力类；自定义 attention + LayerNorm 仅作注册组成证据。
        layer_counts = {len(module.layers) for module in modules if isinstance(module, nn.TransformerEncoder)}
        layer_counts.update(len(module) for module in modules
                            if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module)
                            and all(isinstance(layer, nn.TransformerEncoderLayer)
                                    or any(isinstance(child, nn.MultiheadAttention) for child in layer.modules())
                                    or (any("attention" in type(child).__name__.lower() for child in layer.modules())
                                        and any(isinstance(child, nn.LayerNorm) for child in layer.modules()))
                                    for layer in module))
        expected_layers = config.get("transformer_layers")
        if expected_layers is not None:
            checks.append((
                "model_structure:transformer_layers", expected_layers in layer_counts,
                f"Transformer/attention-container counts={sorted(layer_counts)}; expected={expected_layers}; "
                "registration evidence only; unrecognized fused/custom attention requires review",
            ))
        source_nodes = []
        if (config.get("multiscale_source_evidence")
                or not all(any(isinstance(module, kind) for module in modules) for kind in (nn.GELU, nn.Sigmoid))):
            source_paths = {CANDIDATE_DIR / model_file}
            for module in modules:
                source_file = inspect.getsourcefile(type(module))
                if source_file and Path(source_file).resolve().is_relative_to(CANDIDATE_DIR.resolve()):
                    source_paths.add(Path(source_file))
            source_nodes = [node for path in sorted(source_paths)
                            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig")))]
        angle_nodes = []
        angle_modules = []
        for _, _, blocks in (roles or {}).values():
            for _, _, _, network, _ in blocks:
                angle_modules.extend(network.modules())
                angle_nodes.extend(ast.walk(ast.parse(inspect.getsource(type(network)))))
        activation_calls = {ast.unparse(node.func).lower() for node in angle_nodes if isinstance(node, ast.Call)}
        angle_activations = all(any(isinstance(m, kind) for m in angle_modules)
                                or any(name.endswith("." + token) for name in activation_calls)
                                for kind, token in ((nn.GELU, "gelu"), (nn.Sigmoid, "sigmoid")))
        checks.append(("model_structure:angle_activations", angle_activations,
                       "Activation evidence restricted to bound angle networks; their execution is checked separately"))


        if config.get("multiscale_source_evidence"):
            identifiers = {value.lower() for node in source_nodes for value in (
                [node.id] if isinstance(node, ast.Name) else
                [node.attr] if isinstance(node, ast.Attribute) else
                [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [])}
            literals = {node.value.lower() for node in source_nodes
                        if isinstance(node, ast.Constant) and isinstance(node.value, str)}
            evidence = identifiers | literals
            actual_angles = {}
            for role, (_, _, blocks) in (roles or {}).items():
                values = []
                for _, block, _, _, _ in blocks:
                    matches = []
                    for path in config["role_paths"]["angle_count"]:
                        value = block
                        for part in path.split("."):
                            value = getattr(value, part, None)
                        if isinstance(value, int):
                            matches.append(value)
                    values.append(matches[0] if len(set(matches)) == 1 else None)
                actual_angles[role] = values
            angle_ok = all(values and all(v == config["branch_angles"][role] for v in values)
                           for role, values in actual_angles.items()) and len(actual_angles) == 2
            checks.append(("model_structure:multiscale_radon", angle_ok,
                           f"Bound branch angles={actual_angles}; unresolved attributes require review; block counts use public rules"))
            checks.append((
                "model_structure:gated_local_fusion",
                "bilinear" in evidence and any("gate" in name for name in identifiers)
                and any("sigmoid" in name for name in identifiers)
                and any("local" in name for name in identifiers)
                and any("residual" in name or "correction" in name for name in identifiers),
                "Static bilinear/gate/local-residual evidence only; channel-wise fusion is not executed",
            ))
            checks.append((
                "model_structure:radon_grid_cache",
                any("cache" in name for name in identifiers)
                and all(any(part in name for name in evidence)
                        for part in ("resolution", "angle", "device", "dtype")),
                "Static cache/key ingredients only; key composition and cache hits are not executed",
            ))

    except Exception as e:
        error_detail = traceback.format_exc()
        if checks[0][1]:
            checks.append(("model_structure:structure", False, error_detail))
        else:
            update_check(0, False, error_detail)

    return checks

def check_model_forward_executable(config: dict, caller_globals=None, caller_locals=None):
    """
    检查模型的前向传播是否可以正常执行。

    model_config_arg 的可选完整 YAML 接入与结构检查一致。

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
        - forward_executable:returns_tensor_shape: 检查输出是否为 Tensor 且维度正确
        - forward_executable:output_finite: 配置 check_nan_inf 时检查有限值

    model_args={} 仅用于支持无参数构造的候选；其他签名须显式接入，不能据此判候选不合格。
    out_shape 按 Base 职责比较实际形状；不提供自动跳过形状检查的分支。
    args_as_positional 按 JSON 参数顺序调用；hook_source="torch" 使用公共 hook
    构造正常规则网格，不要求候选提供测试 helper，不对候选增设坐标范围约束。
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
    ]
    if config.get("check_nan_inf"):
        CHECKPOINT_NAMES.append("forward_executable:output_finite")
    if config.get("role_paths"):
        CHECKPOINT_NAMES.append("forward_executable:radon_branch_execution")
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
        hook_module = torch if config.get("hook_source") == "torch" else model_module
        context = _execute_hooks(hook_module, pre_hooks, context, caller_globals, caller_locals)

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
            raise ValueError("config['model_args'] is required; {} requests candidate defaults")

        resolved_model_args = _resolve_args(model_args, context)
        config_arg = config.get("model_config_arg")
        if config_arg and config_arg in inspect.signature(model_cls).parameters:
            resolved_model_args.setdefault(config_arg, context["config"])
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
        handles = []
        if config.get("role_paths"):
            roles = _rno_model_roles(model, config["role_paths"])
            for role, (_, _, blocks) in roles.items():
                for block_path, block, angle_path, network, convs in blocks:
                    for path, module in [(block_path, block), (angle_path, network), *convs]:
                        observed[path] = 0
                        def record(_module, _args, _out, path=path):
                            observed[path] += 1
                        handles.append(module.register_forward_hook(record))
        try:
            with torch.no_grad():
                out = model(*converted_forward_args.values()) if config.get("args_as_positional") else model(**converted_forward_args)
        finally:
            for handle in handles:
                handle.remove()

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
        if config.get("check_nan_inf"):
            current_index += 1
            update_check(current_index, bool(torch.isfinite(out).all()), "Candidate output finite-value check")
        if config.get("role_paths"):
            current_index += 1
            update_check(current_index, bool(observed) and all(n > 0 for n in observed.values()),
                         f"Actual bound branch/angle-network forward calls={observed}; not a full connectivity proof")

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
        - train:rno_optimizer_and_clip: AdamW、OneCycleLR、逐步 scheduler 与梯度裁剪
        - train:rno_physics_loss: 物理空间数据项、PDE、边界与课程权重静态证据
        - train:no_validation_or_test_selection: 不设置验证集且不以测试集选 checkpoint
        - train:checkpoint_and_artifacts: last/周期 checkpoint 与结果文件声明
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
        "train:rno_optimizer_and_clip",
        "train:rno_physics_loss",
        "train:no_validation_or_test_selection",
        "train:checkpoint_and_artifacts",
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
        train_tree = None
        file_paths = []
        code_contents = []
        for file_path in existing_files:
            source = file_path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(file_path))
            all_nodes.extend(list(ast.walk(tree)))
            if file_path.name == "train.py":
                train_tree = tree
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

        # ------------------ 11. RNO Prompt 特有训练契约 ------------------
        train_nodes = list(ast.walk(train_tree)) if train_tree is not None else []
        call_names = {
            ast.unparse(node.func).lower()
            for node in all_nodes if isinstance(node, ast.Call)
        }
        identifiers = {
            value.lower()
            for node in all_nodes
            for value in (
                [node.id] if isinstance(node, ast.Name) else
                [node.attr] if isinstance(node, ast.Attribute) else
                [node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else []
            )
        }
        strings = {
            str(node.value).lower()
            for node in all_nodes
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        train_names = {
            value.lower()
            for node in train_nodes
            for value in (
                [node.id] if isinstance(node, ast.Name) else
                [node.attr] if isinstance(node, ast.Attribute) else
                [node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else []
            )
        }

        current_index += 1
        optimizer_ok = any(name.endswith("adamw") for name in call_names)
        scheduler_targets = set()
        for node in all_nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
                if ast.unparse(node.value.func).lower().endswith("onecyclelr"):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    scheduler_targets.update(ast.unparse(target).lower() for target in targets)
        scheduler_ok = any(f"{name}.step" in call_names for name in scheduler_targets)
        clip_ok = any(name.endswith("clip_grad_norm_") for name in call_names)
        update_check(current_index, optimizer_ok and scheduler_ok and clip_ok,
                     f"Static optimizer evidence: AdamW={optimizer_ok}, OneCycleLR/step={scheduler_ok}, clip_grad_norm_={clip_ok}")

        current_index += 1
        relative_l2 = any("relative" in name and "l2" in name for name in identifiers)
        gradient = any("gradient" in name or name.startswith("grad_") for name in identifiers)
        boundary = any("dirichlet" in name or "boundary" in name for name in identifiers)
        pde = any(any(token in name for token in ("pde", "darcy", "divergence", "flux", "residual")) for name in identifiers)
        harmonic = any("harmonic" in name for name in identifiers | strings)
        curriculum = any("curriculum" in name or "progress" in name for name in identifiers)
        update_check(current_index, relative_l2 and gradient and boundary and pde and harmonic and curriculum,
                     f"Static loss evidence: relative_l2={relative_l2}, gradient={gradient}, boundary={boundary}, pde={pde}, harmonic={harmonic}, curriculum={curriculum}; formulas not executed")

        current_index += 1
        validation = any(name in {"val_loader", "valid_loader", "validation_loader",
                                  "val_dataset", "valid_dataset", "validation_dataset"} for name in train_names)
        # 只连接直接测试指标赋值与保存条件，不把 best_train_loss 或测试日志当成选模。
        checkpoint_save_names = {"save", "save_checkpoint", *config.get("checkpoint_save_functions", [])}
        test_metrics = set()
        for node in train_nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
                expression = ast.unparse(node.value).lower()
                if "test" in expression and any(token in expression for token in ("eval", "metric", "loss", "score")):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    test_metrics.update(ast.unparse(target).lower() for target in targets)
        test_selection = False
        for node in train_nodes:
            if isinstance(node, ast.If):
                condition_names = {item.id.lower() for item in ast.walk(node.test) if isinstance(item, ast.Name)}
                test_condition = bool(condition_names & test_metrics) or any(
                    "test" in name and any(token in name for token in ("loss", "metric", "score", "error"))
                    for name in condition_names)
                conditional_save = any(isinstance(item, ast.Call)
                                       and ast.unparse(item.func).split(".")[-1] in checkpoint_save_names
                                       for statement in node.body + node.orelse for item in ast.walk(statement))
                test_selection |= test_condition and conditional_save
        update_check(current_index, not validation and not test_selection,
                     f"Static validation-loader evidence={validation}; direct test-metric checkpoint selection={test_selection}; no full control-flow proof")

        current_index += 1
        last_pt = any(Path(value).name == "last.pt" for value in strings)
        periodic = any(
            isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in checkpoint_save_names
            and any(isinstance(item, ast.JoinedStr) and "epoch" in ast.unparse(item).lower()
                    for argument in [*node.args, *(keyword.value for keyword in node.keywords)] for item in ast.walk(argument))
            for node in all_nodes
        ) or any(
            isinstance(node, ast.If) and any(isinstance(item, ast.Mod) for item in ast.walk(node.test))
            and any(isinstance(item, ast.Call) and ast.unparse(item.func).split(".")[-1] in checkpoint_save_names
                    for statement in node.body for item in ast.walk(statement))
            for node in all_nodes
        )
        artifacts = {"history.json", "summary.json", "train.log"}.issubset({Path(value).name for value in strings})
        save_call = any(name.endswith("save") or name.split(".")[-1] in checkpoint_save_names for name in call_names)
        update_check(current_index, last_pt and periodic and artifacts and save_call,
                     f"Static output evidence: last.pt={last_pt}, periodic={periodic}, artifacts={artifacts}, save_call={save_call}; files not generated")

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
            source = file_path.read_text(encoding="utf-8-sig")
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
    运行候选训练入口；Base 原加载/配置/hooks/调用顺序保留。

    runtime_probe 存在时使用临时 421×421 MAT（2 train/1 test）、一个 epoch，
    模型/训练/Loss 均为候选真实实现；CPU、目录、样本数是试验配置覆盖。
    entry_names 是已审查的调用接入，不是候选必须采用的私有接口。
    追加：physical_pressure_loss、physical_pde_input、independent_boundary_term、
    pipeline_cache_consumed、harmonic_finite_volume，并观察真实 backward/step 的
    finite_loss、finite_gradients、parameter_update；没有观察到调用时需复核。
    普通模式不变；无 runtime_probe 时仍按 Base 调用指定训练函数。

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
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    resources = ExitStack()
    probe = None

    try:
        # ------------------ 1. 获取配置 ------------------
        train_fn_name = config.get("train_fn_name", "train")
        train_file = config.get("run_file")
        if not train_file:
            raise ValueError("run_file not provided in config")

        # ------------------ 2. 检查训练接口 ------------------
        # 使用 _load_candidate_modules 加载模块（与 check_training 一致）
        train_module = _load_candidate_modules(train_file)
        if config.get("runtime_probe"):
            entries = [name for name in config["entry_names"]
                       if callable(getattr(train_module,name,None))
                       and Path(inspect.getsourcefile(getattr(train_module,name))).resolve() == (CANDIDATE_DIR/train_file).resolve()]
            if len(entries) != 1:
                raise ValueError(f"REVIEW_REQUIRED: bind candidate training entry, matched={entries}")
            train_fn_name = entries[0]
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
        candidate_config = _load_candidate_config(config_name)
        runtime_config = deepcopy(candidate_config)

        # ------------------ 5. 构造冒烟配置（轻量覆盖） ------------------
        current_index += 1

        # 默认冒烟配置（epochs=1, batch_size=8 等轻量参数）
        smoke_config = config.get("smoke_config", {})
        context = {"checkpoint_dir": checkpoint_dir, "results_dir": results_dir}
        smoke_config = _resolve_args(smoke_config, context)

        runtime_config.update(smoke_config)
        if config.get("runtime_probe"):
            probe = resources.enter_context(_rno_probe_fixture(runtime_config, config["runtime_probe"], work_dir))
            runtime_config = probe["config"]
            resources.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            sys.argv = [train_file, "--config", str(probe["config_path"])]
            context["runtime_config_path"] = str(probe["config_path"])
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
        train_args_config = config["entry_args"][train_fn_name] if probe is not None else config.get("train_args", {})
        resolved_train_args = _resolve_args(train_args_config, context)

        # 执行训练
        if probe is not None:
            observed, observer = _rno_training_observer(probe)
            previous_profile = sys.getprofile()
            try:
                sys.setprofile(observer)
                result = train_fn(**resolved_train_args)
            finally:
                sys.setprofile(previous_profile)
        else:
            result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)

        update_check(
            current_index,
            True,
            f"Smoke test completed successfully: {train_fn_name}(...) with {config_name}"
        )

        if probe is not None:
            for role, name in (("pressure", "training:physical_pressure_loss"), ("pde", "training:physical_pde_input"),
                               ("boundary", "training:independent_boundary_term"), ("cache", "training:pipeline_cache_consumed"),
                               ("losses", "training:finite_loss"), ("gradients", "training:finite_gradients"),
                               ("updates", "training:parameter_update")):
                values = observed[role]
                passed = bool(values) and all(values) and not observed["errors"]
                detail = f"Actual candidate train call observations={values}; details={observed['details'].get(role,[])}; probe_errors={observed['errors']}"
                if not values:
                    detail = "REVIEW_REQUIRED: declared observation binding not reached; not proof of a candidate implementation error; " + detail
                checks.append((name, passed, detail))
            try:
                passed, detail = _rno_pde_numeric_probe()
            except Exception as exc:
                passed, detail = False, f"REVIEW_REQUIRED: numerical helper binding/execution: {exc}"
            checks.append(("training:harmonic_finite_volume", passed, detail))

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    finally:
        resources.close()

    return checks



def check_infer_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    动态执行候选推理入口，保留 Base 加载/配置/hooks/调用/输出顺序。

    runtime_probe 仅准备临时 MAT、候选自身权重和调用参数；不替换模型输出或指标。
    追加 physical_metrics 和 evaluation_metadata 后置检查。未知入口需要接入复核。
    物理指标同时支持原压力和物理硬边界压力，但不能使用标准化值冒充物理量。

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
        - inference_execution:strict_checkpoint_and_test: 实际 strict 加载和测试输出规模
        - inference_execution:physical_metrics: 四项有限指标，压力指标的物理单位
        - inference_execution:evaluation_metadata: 实际 evaluation.json 元数据

    _compute_mse/_compute_mae 直接使用 Base；不再由结构检查重复猜指标/元数据。
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
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    resources = ExitStack()
    probe = None

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
        context = {"config": candidate_config}
        context.update(candidate_config)

        # ------------------ 3. 获取推理函数 ------------------
        infer_func_name = config.get("infer_function", "inference")
        if config.get("runtime_probe"):
            entries = [name for name in config["entry_names"]
                       if callable(getattr(infer_module,name,None))
                       and Path(inspect.getsourcefile(getattr(infer_module,name))).resolve() == (CANDIDATE_DIR/infer_file).resolve()]
            if len(entries) != 1:
                raise ValueError(f"REVIEW_REQUIRED: bind candidate inference entry, matched={entries}")
            infer_func_name = entries[0]
            probe = resources.enter_context(_rno_probe_fixture(candidate_config, config["runtime_probe"], CANDIDATE_DIR))
            resources.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            context = {"config":probe["config"], **probe["config"]}
            context["runtime_config_path"] = str(probe["config_path"])
            module = _load_candidate_modules(config["model_file"])
            model_cls = _load_class(module,config["model_class"])
            args = _resolve_args(config["model_args"], {"config":probe["config"]})
            key = config.get("model_config_arg")
            if key and key in inspect.signature(model_cls).parameters:
                args.setdefault(key,probe["config"])
            model = model_cls(**args).cpu()
            checkpoint = probe["work"] / "probe.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch":7},checkpoint)
            del model
            context["runtime_checkpoint_path"] = str(checkpoint)
            sys.argv = [infer_file,"--config",str(probe["config_path"]),"--checkpoint",str(checkpoint)]
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
        infer_args_config = config["entry_args"][infer_func_name] if probe is not None else config.get("infer_args", {})
        resolved_infer_args = _resolve_args(infer_args_config, context)

        if probe is not None:
            outputs, strict_loads = [], []
            def record(frame,event,value):
                instance = frame.f_locals.get("self")
                if (event == "call" and frame.f_code is torch.nn.Module.load_state_dict.__code__
                        and isinstance(instance, torch.nn.Module) and type(instance).__name__ == config["model_class"]):
                    strict_loads.append(frame.f_locals.get("strict") is True)
                if (event == "return" and frame.f_code.co_name == "forward"
                        and isinstance(instance,torch.nn.Module) and type(instance).__name__ == config["model_class"]
                        and isinstance(value,torch.Tensor)):
                    outputs.append(value.detach().cpu().reshape(value.shape[0],-1))
            previous_profile = sys.getprofile()
            try:
                sys.setprofile(record)
                result = infer_func(**resolved_infer_args)
            finally:
                sys.setprofile(previous_profile)
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

        if probe is not None:
            report_path = probe["work"] / "results" / "evaluation.json"
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
            metrics = report.get("metrics",report)
            if not outputs:
                raise ValueError("REVIEW_REQUIRED: no bound candidate model output observed")
            prediction = torch.cat(outputs)*probe["stats"]["target_std"] + probe["stats"]["target_mean"]
            target = probe["test_u"].reshape(1,-1)
            bounded = prediction.reshape(-1,85,85).clone()
            bounded[:,(0,-1),:] = 0; bounded[:,:,(0,-1)] = 0
            comparisons = {}
            for label, pred in (("decoded_raw",prediction), ("decoded_then_physical_zero_boundary",bounded.reshape(prediction.shape))):
                diff = pred-target
                expected = {"mse": _compute_mse(pred, target), "mae": _compute_mae(pred, target),
                            "relative_l2":float((torch.linalg.vector_norm(diff,dim=1)/torch.linalg.vector_norm(target,dim=1)).mean())}
                comparisons[label] = expected
            complete_metrics = all(key in metrics and np.isfinite(metrics[key]) for key in
                                   ("relative_l2", "gradient_relative_l2", "mse", "mae"))
            matched = [label for label,expected in comparisons.items()
                       if complete_metrics and prediction.shape == target.shape and all(key in metrics and np.isfinite(metrics[key]) and np.isclose(metrics[key],v,rtol=2e-4,atol=1e-6)
                              for key,v in expected.items())]
            checks.append(("inference_execution:strict_checkpoint_and_test", bool(strict_loads) and all(strict_loads)
                           and prediction.shape == target.shape,
                           f"Actual model strict loads={strict_loads}; output/test shapes={tuple(prediction.shape)}/{tuple(target.shape)}; "
                           "unrecognized loading wrappers require review"))
            checks.append(("inference_execution:physical_metrics",bool(matched),
                           f"Actual entry metrics={metrics}; independent physical references={comparisons}; matched={matched}. "
                           "Requires all four finite metrics; checks pressure units, not a unique gradient stencil or hard-vs-soft boundary convention."))
            checks.append(("inference_execution:evaluation_metadata", report.get("checkpoint_epoch") == 7
                           and ("data_config" in report) and (report.get("parameter_count",report.get("num_parameters",0)) > 0),
                           f"Actual evaluation.json: epoch={report.get('checkpoint_epoch')}, data_config_present={'data_config' in report}, "
                           f"parameter_count={report.get('parameter_count',report.get('num_parameters'))}"))

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    finally:
        resources.close()

    return checks
