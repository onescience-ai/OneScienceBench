#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MeshGraphNet CylinderFlow 二次开发：checks/params 标准化。

直接复用 Base 的文件、配置内容与具名接口检查；同名函数以完整 Base 原函数为底稿。
边界来源是 evaluate_meshgraphnet_cylinderflow_sec_develop.py，不是训练版 evaluator。
数据使用专用静态契约；训练骨架、目标噪声及领域静态证据共用 check_train_structure 的文件/AST 流程。
结构尺寸使用自包含 JSON rules 和公共 checker；MLP 组件及多尺度证据保持旧脚本检查深度。
小图冒烟由评测器构建 MSE/Adam，不运行完整 train.py。
旧 optional 的参考对照、独立评测脚本和已落地产物仍不作为主 checks 硬门槛。
Prompt 明确要求的默认目录、loss_nodes、DataLoader、独立损失权重和结果文件声明在现有训练检查中检查。
静态检查不计注释或说明字符串；通过只表示源码/配置证据，不代表完整调用链和数值语义已验证。
"""
from __future__ import annotations

import ast
import inspect
import traceback
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, List, Tuple

import torch
import torch.nn as nn

from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _convert_arg_to_type,
    _execute_hooks,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _move_tensors_to_device,
    _resolve_args,
    check_files,
    check_config_content,
    check_interface,
)

SEED = 20260821
NODE_COUNT = 8
FORWARD_ALIASES = {
    "x": "node_features", "nodes": "node_features",
    "edge_features": "fine_edge_features", "edge_attr": "fine_edge_features",
    "edge_index": "fine_edge_index", "coarse_indices": "coarse_node_index",
    "interp_index": "interpolation_index", "interp_weight": "interpolation_weight",
    "coarse_node_indices": "coarse_node_index",
    "coarse_to_fine_idx": "interpolation_index",
    "coarse_to_fine_weights": "interpolation_weight",
}

def _same(actual: Any, expected: Any) -> bool:
    """旧版配置数值比较；浮点转换失败为 False，容差不变。"""
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= max(1.0, abs(expected)) * 1.0e-9
        except (TypeError, ValueError):
            return False
    return actual == expected

def _tiny_graph(seed: int = SEED) -> dict[str, Any]:
    """8 节点多尺度夹具，保留旧拓扑并满足 Prompt one-hot/L2 距离契约；不替候选实现 FPS/缓存。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fine_edge_index = torch.tensor(
        (
            (0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4),
            (1, 0, 2, 1, 3, 2, 0, 3, 5, 4, 6, 5, 7, 6, 4, 7),
        ),
        dtype=torch.long,
    )
    coarse_node_index = torch.tensor((0, 3, 6), dtype=torch.long)
    coarse_edge_index = torch.tensor(
        ((0, 1, 1, 2, 2, 0), (1, 0, 2, 1, 0, 2)), dtype=torch.long
    )
    nearest = torch.tensor((0, 0, 1, 1, 1, 2, 2, 2), dtype=torch.long)
    fine = torch.arange(NODE_COUNT, dtype=torch.long)
    down_edge_index = torch.stack((fine, nearest))
    up_edge_index = torch.stack((nearest, fine))
    interpolation_index = torch.stack((nearest, (nearest + 1) % 3), dim=1)
    interpolation_weight = torch.tensor([[0.75, 0.25]]).repeat(NODE_COUNT, 1)
    graph = {
        "node_features": torch.randn((NODE_COUNT, 6), generator=generator),
        "fine_edge_features": torch.randn(
            (fine_edge_index.shape[1], 3), generator=generator
        ),
        "fine_edge_index": fine_edge_index,
        "coarse_node_index": coarse_node_index,
        "coarse_edge_features": torch.randn(
            (coarse_edge_index.shape[1], 3), generator=generator
        ),
        "coarse_edge_index": coarse_edge_index,
        "down_edge_features": torch.randn((NODE_COUNT, 3), generator=generator),
        "down_edge_index": down_edge_index,
        "up_edge_features": torch.randn((NODE_COUNT, 3), generator=generator),
        "up_edge_index": up_edge_index,
        "interpolation_index": interpolation_index,
        "interpolation_weight": interpolation_weight,
        "target": torch.randn((NODE_COUNT, 3), generator=generator),
    }
    # Prompt 的显式特征契约：后四通道为 one-hot，边为相对坐标及其 L2 距离。
    # 保留旧拓扑、8/3 节点数和随机抽样顺序；此夹具不验证候选 FPS 比例或缓存。
    graph["node_features"][:, 2:] = torch.nn.functional.one_hot(
        torch.arange(NODE_COUNT) % 4, num_classes=4
    ).to(graph["node_features"])
    positions = torch.tensor([
        [0.0, 0.0], [0.1, 0.1], [1.9, 0.1], [2.0, 0.0],
        [2.1, 0.1], [3.9, 0.1], [4.0, 0.0], [4.1, 0.1],
    ], dtype=graph["node_features"].dtype)
    coarse_positions = positions[coarse_node_index]
    for name, source_positions, target_positions in (
        ("fine", positions, positions),
        ("coarse", coarse_positions, coarse_positions),
        ("down", positions, coarse_positions),
        ("up", coarse_positions, positions),
    ):
        senders, receivers = graph[name + "_edge_index"]
        relative = source_positions[senders] - target_positions[receivers]
        graph[name + "_edge_features"] = torch.cat(
            (relative, torch.linalg.vector_norm(relative, dim=-1, keepdim=True)), dim=-1
        )
    return graph

def check_data_contract(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """数据静态契约：TFRecord、统计、多尺度缓存与窗口。

    config: {"files": ["dataset.py"], "config_name": "config.yaml"}。
    files 为恰好一个源码文件；config_name 由公共 YAML loader 加载。
    不导入/读取大 TFRecord，不证明 FPS/kNN 数学正确或训练统计无泄漏。
    注释/docstring 不作实现证据；接受 eye/scatter 或四列零矩阵逐元素赋 1 的 one-hot 写法。
    目标/噪声的跨文件证据移至训练阶段，由 check_train_structure 的 teacher_forcing_noise 选项单独检查。
    """
    CHECKPOINT_NAMES = [
        "syntax:dataset.py",
        "data:tfrecord_schema",
        "data:limits",
        "data:fine_graph",
        "data:node_types",
        "data:training_statistics",
        "data:multiscale_pipeline",
        "data:rollout_windows",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        files = config.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("config['files'] must contain exactly one dataset source file")
        source = (CANDIDATE_DIR / files[0]).read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(CANDIDATE_DIR / files[0]))
        update_check(current_index, True, "Python syntax parsed; no dataset entrypoint name is imposed")
        current_index += 1
        # 只从代码提取证据，注释和 docstring 不能代替数据处理实现。
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body.pop(0)
        source = ast.unparse(tree)
        lower, compact = source.lower(), "".join(source.lower().split())
        candidate_config = _load_candidate_config(config["config_name"])
        data = candidate_config.get("data", {})
        data = data if isinstance(data, dict) else {}
        train_count = next((data[name] for name in ("train_trajectories", "train_samples",) if name in data), None)
        valid_count = next((data[name] for name in ("valid_trajectories", "val_trajectories", "val_samples",) if name in data), None)
        test_count = next((data[name] for name in ("test_trajectories", "test_samples",) if name in data), None)
        shared_steps = next((data[name] for name in ("steps", "num_steps") if name in data), None)
        step_values = (
            [shared_steps] * 3
            if shared_steps is not None
            else [next((data[name] for name in ("train_steps",) if name in data), None), next((data[name] for name in ("val_steps",) if name in data), None), next((data[name] for name in ("test_steps",) if name in data), None)]
        )
        fields = ("cells", "mesh_pos", "node_type", "velocity", "pressure")
        triangle_edges = all(any(token in compact for token in pair)
                             for pair in (("[0,1]", "[1,0]"), ("[1,2]", "[2,1]"), ("[2,0]", "[0,2]")))
        # 四列零矩阵 + 二维索引赋 1 也是 one-hot 的实现证据；不绑定数组/常量名。
        four_names = {target.id for node in tree.body if isinstance(node, ast.Assign)
                      and isinstance(node.value, ast.Constant) and node.value.value == 4
                      for target in node.targets if isinstance(target, ast.Name)}
        four_column_arrays = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "zeros"
                    and node.value.args):
                continue
            shape = node.value.args[0]
            if isinstance(shape, (ast.Tuple, ast.List)) and len(shape.elts) == 2:
                width = shape.elts[1]
                if ((isinstance(width, ast.Constant) and width.value == 4)
                        or (isinstance(width, ast.Name) and width.id in four_names)):
                    four_column_arrays.update(target.id for target in node.targets if isinstance(target, ast.Name))
        indexed_one_hot = any(isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                              and node.value.value == 1 and any(isinstance(target, ast.Subscript)
                              and isinstance(target.value, ast.Name) and target.value.id in four_column_arrays
                              and isinstance(target.slice, ast.Tuple) and len(target.slice.elts) == 2
                              for target in node.targets) for node in ast.walk(tree))
        multiscale = (
            ("farthest" in lower or "fps" in lower)
            and ("knn" in lower or ("cdist" in lower and "topk" in lower))
            and "coarse" in lower
            and any(token in lower for token in ("interpolation", "cross_scale", "down_edge", "up_edge"))
        )
        cached = (
            "torch.save" in lower
            or "cache" in lower
            or ("self.trajectories" in lower and "_build_multiscale_graph" in lower)
        )
        values = [
            (
                "data:tfrecord_schema",
                "tfrecord" in lower
                and all(field in lower for field in fields)
                and any(token in lower for token in ("decode_raw", "parsefromstring", "example")),
                fields,
            ),
            (
                "data:limits",
                _same(train_count, 400)
                and _same(valid_count, 10)
                and _same(test_count, 10)
                and all(_same(value, 300) for value in step_values),
                f"counts={train_count}/{valid_count}/{test_count}, steps={step_values}",
            ),
            (
                "data:fine_graph",
                triangle_edges
                and "unique" in lower
                and any(token in lower for token in ("flip", "bidirectional", "senders", "receivers"))
                and any(token in lower for token in ("vector_norm", "linalg.norm", "torch.norm")),
                "deduplicated bidirectional triangle edges with [dx,dy,L2]",
            ),
            (
                "data:node_types",
                (indexed_one_hot or (("one_hot" in lower or "scatter" in lower or "eye(" in compact)
                 and ("num_classes=4" in compact or "torch.zeros" in compact or "eye(4" in compact)))
                and (
                    all(name in source for name in ("NORMAL", "INFLOW", "OUTFLOW", "WALL_BOUNDARY"))
                    or "squeezed-3" in compact
                ),
                "four CylinderFlow node types",
            ),
            (
                "data:training_statistics",
                "edge_stats.json" in lower
                and "node_stats.json" in lower
                and any(token in lower for token in ("std", "scale")),
                "statistics-file/scaling source evidence; does NOT verify training-only provenance",
            ),
            (
                "data:multiscale_pipeline",
                multiscale and cached,
                "FPS coarse nodes, kNN coarse graph, cross-scale mapping, trajectory cache",
            ),
            (
                "data:rollout_windows",
                any(token in lower for token in ("max_rollout_horizon", "horizon", "window_size"))
                and any(token in lower for token in ("velocity_targets", "velocity"))
                and any(token in lower for token in ("pressure_targets", "pressure")),
                "contiguous windows retain future velocity and pressure",
            ),
        ]
        for name, passed, message in values:
            current_index = CHECKPOINT_NAMES.index(name)
            update_check(current_index, bool(passed), "Static source/config evidence: " + str(message))
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

    当前 Case 仅做三处必要适配：按候选构造签名过滤同义 model_args；
    允许显式指定 CPU device；把 rule.params.target_path_aliases 解析为公共
    checker 接收的单个 target_path。其余执行顺序和规则逻辑保持 Base 原样。
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

        # 必要适配：同一 Case 允许等价构造参数名，只传候选类真实接受的参数。
        if config.get("check_config", {}).get("match_model_signature", False):
            signature = inspect.signature(model_cls)
            model_args = {
                name: value for name, value in model_args.items()
                if name in signature.parameters
            }

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

        # 必要适配：每条 rule 自身声明等价路径；解析后仍交给公共 checker。
        registered_paths = dict(model.named_modules())
        normalized_rules = []
        for rule_config in rules:
            normalized = deepcopy(rule_config)
            params = normalized.get("params")
            if isinstance(params, dict) and "target_path_aliases" in params:
                primary = params.get("target_path")
                aliases = params.pop("target_path_aliases")
                candidates = [primary, *aliases] if isinstance(aliases, list) else [primary]
                params["target_path"] = next(
                    (path for path in candidates
                     if isinstance(path, str) and path in registered_paths),
                    primary,
                )
                # 公共 checker 将数字段当作 children 下标；共享模块去重后需换算位置。
                selected = params["target_path"]
                if selected and selected in registered_paths:
                    parent, translated = model, []
                    for part in selected.split("."):
                        children = dict(parent.named_children())
                        translated.append(str(list(children).index(part)) if part.isdigit() else part)
                        parent = children[part]
                    params["target_path"] = ".".join(translated)
            normalized_rules.append(normalized)
        checker.add_rules_from_rules(normalized_rules)

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

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(-1, False, error_detail)

    return checks

def check_model_contract(config: dict, caller_globals=None, caller_locals=None):
    """检查公共结构 rules 无法直接表达的 MeshGraphNet Prompt 契约。

    公共 ``check_model_structure`` 已承接模型类型和线性层形状。本方法按旧脚本
    深度检查：MLP 的 Linear 数量、LayerNorm 组件、模型中的 ReLU 证据，
    以及三段非空且总计 15、多尺度传递、sum 和残差的结构/源码证据。
    不验证每个隐藏位置的激活或 LayerNorm 的真实执行顺序；不要求 5/5/5。
    """
    CHECKPOINT_NAMES = [
        "model:all_mlp_contract",
        "model:fine_coarse_fine_stages",
        "model:multiscale_transfer",
        "model:residual_sum_message_passing",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        model_file = config.get("model_file")
        if not model_file:
            raise ValueError("config['model_file'] is missing or empty")
        model_module = _load_candidate_modules(model_file)

        model_class_name = config.get("model_class")
        if not model_class_name:
            raise ValueError("config['model_class'] is missing or empty")
        model_cls = _load_class(model_module, model_class_name)

        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("config['config_name'] is missing or empty")
        candidate_config = _load_candidate_config(config_name)
        context = {"config": candidate_config}
        if isinstance(candidate_config, dict):
            context.update(candidate_config)

        model_args = config.get("model_args")
        if not model_args:
            raise ValueError("config['model_args'] is required")
        if config.get("check_config", {}).get("match_model_signature", False):
            signature = inspect.signature(model_cls)
            model_args = {
                name: value for name, value in model_args.items()
                if name in signature.parameters
            }
        model = model_cls(**_resolve_args(model_args, context)).to(
            config.get("device", DEVICE)
        ).eval()

        try:
            source_tree = ast.parse(inspect.getsource(model_module))
            for node in ast.walk(source_tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if (node.body and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)):
                        node.body.pop(0)
            source = ast.unparse(source_tree).lower()
        except (OSError, TypeError, SyntaxError):
            source = ""

        decoder = getattr(model, "decoder", None)
        decoder_modules = set(decoder.modules()) if isinstance(decoder, nn.Module) else set()
        mlp_candidates = [
            module for module in model.modules()
            if any(cls.__name__.lower() == "mlp" for cls in type(module).__mro__)
            or (isinstance(module, nn.Sequential)
                and any(isinstance(child, nn.Linear) for child in module.children()))
        ]
        mlps = [
            module for module in mlp_candidates
            if not any(
                other is not module
                and any(child is module for child in list(other.modules())[1:])
                for other in mlp_candidates
            )
        ]
        # 旧深度：检查组件数量/存在，不把注册的 ReLU 对象数当作隐藏层执行次数。
        linear_counts = [sum(isinstance(child, nn.Linear) for child in module.modules())
                         for module in mlps]
        non_decoder_norms = [any(isinstance(child, nn.LayerNorm) for child in module.modules())
                             for module in mlps if module not in decoder_modules]
        decoder_norms = sum(isinstance(module, nn.LayerNorm) for module in decoder_modules)
        relu_evidence = (any(isinstance(module, nn.ReLU) for module in model.modules())
                         or any(token in source for token in ("relu(", "relu_(")))
        update_check(
            current_index,
            bool(mlps) and all(count == 3 for count in linear_counts)
            and all(non_decoder_norms) and decoder_norms == 0 and relu_evidence,
            f"MLPs={len(mlps)}, Linear counts={linear_counts}, "
            f"non-decoder LayerNorm={all(non_decoder_norms)}, decoder LayerNorm={decoder_norms}, "
            f"ReLU evidence={relu_evidence}; component evidence only, activation/norm order NOT verified",
        )
        current_index += 1

        role_modules = dict(model.named_modules())
        stage_blocks = []
        for aliases in (
            ("fine_processor_before", "fine_pre_processor", "fine_processor_pre",
             "processor_fine_before", "processor.fine_blocks"),
            ("coarse_processor", "processor_coarse", "coarse_processors",
             "processor.coarse_blocks"),
            ("fine_processor_after", "fine_post_processor", "fine_processor_post",
             "processor_fine_after", "processor.correction_blocks"),
        ):
            blocks = []
            for name in aliases:
                value = role_modules.get(name)
                if value is None:
                    continue
                value = getattr(value, "blocks", value)
                try:
                    blocks = list(value)
                    break
                except TypeError:
                    continue
            stage_blocks.append(blocks)
        before, coarse, after = stage_blocks
        update_check(
            current_index,
            all(len(stage) > 0 for stage in stage_blocks)
            and sum(len(stage) for stage in stage_blocks) == 15,
            f"stage_blocks={[len(before), len(coarse), len(after)]}; require non-empty total=15",
        )
        current_index += 1

        transfer = (
            (hasattr(model, "down_transfer") and hasattr(model, "up_transfer"))
            or (
                ("fusion_mlp" in role_modules or "processor.fusion_mlp" in role_modules)
                and any(token in source for token in
                        ("interpol", "coarse_on_fine", "coarse_to_fine"))
            )
        )
        update_check(
            current_index,
            transfer,
            "down/up transfer modules or interpolation fusion source evidence; not a full computation-graph proof",
        )
        current_index += 1

        residual_sum = (
            ("index_add" in source or "scatter_add" in source)
            and source.count("+") >= 2
        )
        update_check(
            current_index,
            residual_sum,
            "Static index_add/scatter_add and residual addition evidence; variable names unrestricted",
        )

    except Exception:
        update_check(current_index, False, traceback.format_exc())

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

    当前 Case 仅适配：小型多尺度输入夹具、候选构造/forward 同义形参、
    显式 CPU device、Base 中 out_shape 键名错误，以及 Prompt 所需有限值检查。
    其余加载、占位符解析、类型转换、前向执行和异常流程保持 Base 原样。
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "forward_executable:forward_run",
        "forward_executable:returns_tensor_shape",
    ]
    if config.get("check_nan_inf", False):
        CHECKPOINT_NAMES.append("forward:finite")
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

        check_config = config.get("check_config", {})
        if check_config.get("match_model_signature", False):
            signature = inspect.signature(model_cls)
            model_args = {
                name: value for name, value in model_args.items()
                if name in signature.parameters
            }

        resolved_model_args = _resolve_args(model_args, context)
        if check_config.get("tiny_multiscale_graph", False):
            torch.set_num_threads(1)
            torch.manual_seed(check_config.get("seed", SEED))
        model = model_cls(**resolved_model_args).to(config.get("device", DEVICE)).eval()

        # ------------------ 7. 执行前向传播 ------------------
        if check_config.get("tiny_multiscale_graph", False):
            context.update(_tiny_graph(check_config.get("seed", SEED)))
        # 获取前向传播参数（可选，默认为空字典）
        forward_args_config = config.get("forward_args", {})
        resolved_forward_args = _resolve_args(forward_args_config, context)

        # 获取模型的 forward 方法
        forward_method = getattr(model, 'forward', None)
        if forward_method is None or not callable(forward_method):
            raise ValueError("Model does not have a callable 'forward' method")

        # 获取参数类型注解
        sig = inspect.signature(forward_method)
        if check_config.get("match_forward_signature", False):
            matched_args = {}
            for name, parameter in sig.parameters.items():
                if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                    continue
                key = name if name in resolved_forward_args else FORWARD_ALIASES.get(name)
                if key in resolved_forward_args:
                    matched_args[name] = resolved_forward_args[key]
                elif parameter.default is inspect.Parameter.empty:
                    raise TypeError(f"unsupported required forward parameter: {name}")
            resolved_forward_args = matched_args
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
        if check_config.get("tiny_multiscale_graph", False):
            converted_forward_args = {
                name: value.to(config.get("device", DEVICE))
                if isinstance(value, torch.Tensor) else value
                for name, value in converted_forward_args.items()
            }
        else:
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

        same_shape = (tuple(out.shape) == tuple(config.get("out_shape", [])))
        update_check(
            current_index,
            same_shape,
            f"Output is Tensor with shape={tuple(out.shape)}, expected out_dim={config.get('out_shape', [])}"
        )

        if config.get("check_nan_inf", False):
            current_index += 1
            update_check(current_index, bool(torch.isfinite(out).all()), "All output values finite")

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

    必要适配：读取源码使用 utf-8-sig；以下两个 Case 选项复用本方法的
    文件检查和 AST 解析，二者互斥，未启用时执行上述 Base 八项原逻辑。
    check_config.teacher_forcing_noise=true：files 为 [dataset.py, train.py]，
    只返回 data:teacher_forcing_noise；不读取 utils.py，不运行 Base 八项。
    check_config.meshgraphnet_contract=true：files 为 [train.py, utils.py]，
    返回下方登记的语法、默认配置、课程、调度及交付检查；不重复 Base 八项。
    两个选项均须提供 config_name，保持旧静态检查深度，不证明数值语义。
    Base 八项仍由配套 JSON 的独立检查调用，文件依赖不变。
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    check_options = config.get("check_config", {}) if isinstance(config, dict) else {}
    teacher_forcing_noise = isinstance(check_options, dict) and check_options.get("teacher_forcing_noise", False)
    meshgraphnet_contract = isinstance(check_options, dict) and check_options.get("meshgraphnet_contract", False)
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
    if teacher_forcing_noise:
        CHECKPOINT_NAMES = ["data:teacher_forcing_noise"]
    elif meshgraphnet_contract:
        CHECKPOINT_NAMES = [
            "syntax:train.py",
            "syntax:utils.py",
            "train:cli",
            "train:model_defaults",
            "train:optimizer_defaults",
            "train:adam_step_scheduler",
            "train:curriculum_rollout",
            "train:rollout_boundary_and_losses",
            "train:optimization",
            "train:ddp_amp_resume_early_stop",
            "train:checkpoint_selection",
            "train:rollout_evaluation",
            "train:output_configured",
            "train:loss_nodes",
            "train:independent_loss_weights",
            "train:result_declarations",
        ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        """更新指定索引的检查结果"""
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        if teacher_forcing_noise and meshgraphnet_contract:
            raise ValueError("teacher_forcing_noise and meshgraphnet_contract are mutually exclusive")
        # ------------------ 1. 检查训练文件是否存在 ------------------
        train_files = config.get("files")
        if not train_files:
            raise ValueError("config['files'] is missing or empty")

        if (teacher_forcing_noise or meshgraphnet_contract) and (
                not isinstance(train_files, list) or len(train_files) != 2):
            raise ValueError("Case source checks require exactly two files in their declared order")

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
        trees = []
        for file_path in existing_files:
            if meshgraphnet_contract:
                current_index = len(file_paths)
            source = file_path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(file_path))
            all_nodes.extend(list(ast.walk(tree)))
            file_paths.append(str(file_path))
            code_contents.append(source)
            trees.append(tree)
            if meshgraphnet_contract:
                update_check(current_index, True, "Python syntax parsed: " + file_path.name)
                current_index += 1

        # 合并所有代码内容（用于文本匹配检查）
        combined_code = "\n".join(code_contents)

        if teacher_forcing_noise or meshgraphnet_contract:
            # 复用上面的 AST；仅移除静态证据中的说明字符串，不改变普通分支 AST。
            evidence_trees = deepcopy(trees)
            for tree in evidence_trees:
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        if (node.body and isinstance(node.body[0], ast.Expr)
                                and isinstance(node.body[0].value, ast.Constant)
                                and isinstance(node.body[0].value.value, str)):
                            node.body.pop(0)
            lower = "\n".join(ast.unparse(tree) for tree in evidence_trees).lower()
            candidate_config = _load_candidate_config(config["config_name"])

        if teacher_forcing_noise:
            files = train_files
            compact = "".join(lower.split())
            data = candidate_config.get("data", {})
            data = data if isinstance(data, dict) else {}
            noise = next((data[name] for name in ("noise_std", "noise") if name in data), None)
            target = (
                ("teacher_target" in lower or "target" in lower)
                and "velocity" in lower and "pressure" in lower
                and ("first_delta" in lower or "velocity[1]" in compact
                     or "velocity_next" in lower
                     or any(
                         isinstance(node, ast.Subscript)
                         and "velocity" in ast.unparse(node.value).lower()
                         and isinstance(node.slice, ast.BinOp)
                         and isinstance(node.slice.op, ast.Add)
                         and isinstance(node.slice.right, ast.Constant)
                         and node.slice.right.value == 1
                         for node in all_nodes
                     ))
            )
            noise_correction = any(
                token in compact for token in ("first_delta-=noise", "-noise", "torch.sub(")
            )
            passed = bool(
                target and _same(noise, 0.02)
                and "normal" in lower and noise_correction
            )
            update_check(
                0,
                passed,
                f"Static source/config evidence: files={files}; next velocity increment/pressure; "
                f"NORMAL noise std={noise}; not a numeric noise/mask proof",
            )
            return checks

        if meshgraphnet_contract:
            model = candidate_config.get("model", {})
            model = model if isinstance(model, dict) else {}
            multiscale = dict(model)
            multiscale.update(model["multiscale"] if isinstance(model.get("multiscale"), dict) else {})
            training = candidate_config.get("training", candidate_config.get("train", {}))
            training = training if isinstance(training, dict) else {}
            loss = candidate_config.get("loss", {})
            loss = loss if isinstance(loss, dict) else {}
            curriculum = loss.get("curriculum", training.get("curriculum", {}))
            curriculum = curriculum if isinstance(curriculum, dict) else {}
            evaluation = candidate_config.get("evaluation", candidate_config.get("eval", {}))
            evaluation = evaluation if isinstance(evaluation, dict) else {}
            hidden = next((model[name] for name in ("hidden_dim", "latent_dim",) if name in model), None)
            stages = [
                next((multiscale[name] for name in ("fine_steps_before", "fine_pre_steps", "num_fine_blocks") if name in multiscale), None),
                next((multiscale[name] for name in ("coarse_steps", "num_coarse_blocks") if name in multiscale), None),
                next((multiscale[name] for name in ("fine_steps_after", "fine_post_steps", "num_correction_blocks") if name in multiscale), None),
            ]
            horizons = next((training[name] for name in ("rollout_horizons",) if name in training), None) or next((curriculum[name] for name in ("horizons", "horizon_schedule") if name in curriculum), None) or []
            rollout_steps = next((evaluation[name] for name in ("rollout_steps", "full_rollout_steps",) if name in evaluation), None)
            eval_horizons = next((evaluation[name] for name in ("rollout_horizons", "horizons",) if name in evaluation), None)
            if eval_horizons is None:
                eval_horizons = rollout_steps if isinstance(rollout_steps, (list, tuple)) else []
            output = next((candidate_config["output"][name] for name in ("root", "output_dir") if isinstance(candidate_config.get("output"), dict) and name in candidate_config["output"]), None) or next((training[name] for name in ("output_dir",) if name in training), None)
            separate_velocity_pressure = (
                any(token in lower for token in ("velocity_loss", "velocity_mse", "vel_loss"))
                and any(token in lower for token in ("pressure_loss", "pressure_mse", "pres_loss"))
            )
            defaults_ok = (
                _same(training.get("epochs"), 25)
                and _same(training.get("batch_size"), 1)
                and _same(training.get("learning_rate"), 1.0e-4)
                and _same(training.get("lr_decay_rate", training.get("lr_decay")), 0.9999991)
            )
            model_ok = (
                _same(next((model[name] for name in ("node_input_dim", "node_dim",) if name in model), None), 6)
                and _same(next((model[name] for name in ("edge_input_dim", "edge_dim",) if name in model), None), 3)
                and _same(next((model[name] for name in ("output_dim", "out_dim",) if name in model), None), 3)
                and _same(hidden, 128)
                and _same(next((model[name] for name in ("mlp_hidden_layers", "hidden_layers", "num_mlp_layers") if name in model), None), 2)
                and str(model.get("aggregation", "")).lower() == "sum"
                and all(type(value) is int and value > 0 for value in stages)
                and sum(stages) == 15
                and _same(multiscale.get("coarse_ratio"), 0.25)
            )
            # 旧深度只检查调用/关键词证据，不追踪 scheduler 的赋值和对象身份。
            calls = [ast.unparse(node.func).lower() for tree in evidence_trees
                     for node in ast.walk(tree) if isinstance(node, ast.Call)]
            scheduler_declared = any(token in lower for token in ("exponentiallr", "lambdalr"))
            step_declared = any(name.endswith(".step") for name in calls)
            manual_lr_update = any(
                isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Mult)
                and isinstance(node.target, ast.Subscript)
                and isinstance(node.target.slice, ast.Constant) and node.target.slice.value == "lr"
                for tree in evidence_trees for node in ast.walk(tree)
            ) or any(
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult)
                and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "lr" for target in
                        (node.targets if isinstance(node, ast.Assign) else [node.target]))
                for tree in evidence_trees for node in ast.walk(tree)
            )
            cli_flags = {item.value for item in ast.walk(evidence_trees[0])
                         if isinstance(item, ast.Constant) and isinstance(item.value, str)}
            literals = {item.value for tree in evidence_trees for item in ast.walk(tree)
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)}
            horizon_sequence = any(isinstance(item, (ast.List, ast.Tuple))
                and {1, 50}.issubset({part.value for part in item.elts if isinstance(part, ast.Constant) and isinstance(part.value, int)})
                and any((isinstance(part, ast.Constant) and part.value == 299)
                        or (isinstance(part, (ast.Name, ast.Attribute, ast.Subscript)) and "rollout" in ast.unparse(part).lower())
                        for part in item.elts)
                for tree in evidence_trees for item in ast.walk(tree))
            loss_nodes = next((section[key] for section, key in ((loss, "loss_nodes"), (loss, "nodes"),
                               (training, "loss_nodes"), (candidate_config, "loss_nodes")) if key in section), None)
            # 仅识别可配置项的独立性，不新增权重取值或正负限制。
            weight_names = (("velocity_weight", "vel_weight", "weight_velocity", "w_velocity", "w_v"),
                            ("pressure_weight", "pres_weight", "weight_pressure", "w_pressure", "w_p"))
            configured_weights = [next((section[key] for section in (loss, training, curriculum)
                                   for key in names if key in section), None) for names in weight_names]
            weight_symbols = {node.id for tree in evidence_trees for node in ast.walk(tree) if isinstance(node, ast.Name)}
            weight_symbols.update(node.arg for tree in evidence_trees for node in ast.walk(tree) if isinstance(node, ast.arg))
            weight_symbols.update(node.attr for tree in evidence_trees for node in ast.walk(tree) if isinstance(node, ast.Attribute))
            weight_symbols.update(literals)
            independent_weights = all(any(name in weight_symbols for name in names) for names in weight_names)
            # 两个独立权重也可通过映射或向量传入，不强制 YAML/函数形参采用一组固定键名。
            weight_mapping = loss.get("weights", loss.get("loss_weights"))
            if isinstance(weight_mapping, dict):
                independent_weights = independent_weights or (all(name in weight_mapping for name in ("velocity", "pressure"))
                    and all(name in literals for name in ("velocity", "pressure")))
            elif isinstance(weight_mapping, (list, tuple)) and len(weight_mapping) == 2:
                weighted_indices = {node.slice.value for tree in evidence_trees for node in ast.walk(tree)
                    if isinstance(node, ast.Subscript) and "weight" in ast.unparse(node.value).lower()
                    and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)}
                independent_weights = independent_weights or {0, 1}.issubset(weighted_indices)
            load_evidence = any(name in ("torch.load", "load_checkpoint") or name.endswith((".load_state_dict", ".load_checkpoint"))
                                for name in calls)
            boundary_assignment = any(isinstance(item, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                and any(isinstance(target, ast.Subscript) for target in
                        (item.targets if isinstance(item, ast.Assign) else [item.target]))
                for tree in evidence_trees for item in ast.walk(tree))
            values = [
                (
                    "train:cli",
                    all("--" + name in cli_flags for name in ("config", "data", "output", "device", "epochs", "rollout")),
                    "required CLI overrides",
                ),
                ("train:model_defaults", model_ok, f"hidden={hidden}, stages={stages}"),
                ("train:optimizer_defaults", defaults_ok, dict(training)),
                (
                    "train:adam_step_scheduler",
                    "adam" in lower and ((scheduler_declared and step_declared) or manual_lr_update),
                    f"Adam source evidence; scheduler declaration={scheduler_declared}, step call={step_declared}, "
                    f"manual lr multiplication={manual_lr_update}; object identity/cadence/value use NOT verified",
                ),
                (
                    "train:curriculum_rollout",
                    list(horizons) == [1, 2, 4, 8]
                    and "horizon" in lower
                    and ("detach" in lower or "no_grad" in lower)
                    and "rollout" in lower
                    and ("one_step" in lower or "single_step" in lower or "teacher" in lower),
                    f"horizons={horizons}",
                ),
                (
                    "train:rollout_boundary_and_losses",
                    ("torch.where" in calls or boundary_assignment)
                    and "normal" in lower
                    and "outflow" in lower
                    and separate_velocity_pressure,
                    "NORMAL/OUTFLOW updates and separate velocity/pressure losses",
                ),
                (
                    "train:optimization",
                    any(name.endswith(".zero_grad") for name in calls) and step_declared,
                    "zero_grad/step call evidence; optimizer/backward are checked by Base check_train_structure; "
                    "step receiver identity NOT verified; model update is exercised separately",
                ),
                (
                    "train:ddp_amp_resume_early_stop",
                    ("distributeddataparallel" in lower or "ddp(" in lower)
                    and "autocast" in lower
                    and "gradscaler" in lower
                    and "resume" in lower
                    and ("stale_epochs" in lower or "patience" in lower),
                    "single device/DDP, AMP, resume, early stopping",
                ),
                (
                    "train:checkpoint_selection",
                    "best.pt" in lower
                    and "last.pt" in lower
                    and ("best_val_loss" in lower or "best_validation" in lower)
                    and load_evidence,
                    "best/last declarations and weight-loading source evidence; selection metric and final reload order not proven",
                ),
                (
                    "train:rollout_evaluation",
                    (
                        all(value in eval_horizons for value in (1, 50, 299))
                        or (_same(rollout_steps, 299) and horizon_sequence)
                    )
                    and "velocity" in lower
                    and "pressure" in lower
                    and "rollout" in lower and "rmse" in lower and "mse" in lower,
                    f"rollout_steps={rollout_steps}, horizons={eval_horizons}; velocity/RMSE and pressure/MSE source evidence, not numeric validation",
                ),
                (
                    "train:output_configured",
                    isinstance(output, str) and PurePosixPath(output.replace("\\", "/")) == PurePosixPath("results/meshgraphnet_cylinderflow_seed0"),
                    f"default output={output!r}; expected ./results/meshgraphnet_cylinderflow_seed0/",
                ),
                (
                    "train:loss_nodes",
                    loss_nodes == "all" and "normal_outflow" in literals,
                    f"default loss_nodes={loss_nodes!r}; normal_outflow option evidence, not mask-value proof",
                ),
                (
                    "train:independent_loss_weights",
                    independent_weights,
                    f"separate velocity/pressure parameter/key evidence; configured scalar values={configured_weights}; weighting arithmetic not dynamically tested",
                ),
                (
                    "train:result_declarations",
                    all(name in literals or any(isinstance(value, str) and value.endswith("/" + name) for value in literals)
                        for name in ("history.json", "summary.json"))
                    and any(token in lower for token in ("logging", "logger", "filehandler", ".log")),
                    "history.json/summary.json and logging source declarations; no materialized output required",
                ),
            ]
            for name, passed, message in values:
                current_index = CHECKPOINT_NAMES.index(name)
                update_check(current_index, bool(passed), "Static source/config evidence: " + str(message))
            return checks

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

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks



def check_meshgraphnet_smoke_training(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple]:
    """在小型多尺度图上验证前向、Loss、梯度和一次 Adam 更新。

    这是模型级动态探针，不冒充 Base ``check_smoke_training``，也不运行候选
    train.py 或证明课程式 rollout loss。模型和 forward 同义形参只按候选真实签名绑定。
    """
    CHECKPOINT_NAMES = [
        "training:smoke_test",
        "training:forward_shape",
        "training:finite_loss",
        "training:finite_gradients",
        "training:multiscale_gradients",
        "training:parameter_updates",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        model_class_name = config.get("model_class", config.get("train_fn_name", "MeshGraphNet"))
        model_file = config.get("model_file", config.get("run_file"))
        if not model_file:
            raise ValueError("model_file/run_file not provided in config")
        model_module = _load_candidate_modules(model_file)
        model_cls = getattr(model_module, model_class_name, None)
        if not inspect.isclass(model_cls):
            raise TypeError(f"Model class '{model_class_name}' not found in {model_file}")

        smoke_config = config.get("smoke_config", {})
        model_args = smoke_config.get("model_args")
        optimizer_args = smoke_config.get("optimizer_args")
        if not isinstance(model_args, dict) or not isinstance(optimizer_args, dict):
            raise ValueError("smoke_config.model_args and optimizer_args are required")
        device = smoke_config.get("device", "cpu")
        seed = smoke_config.get("seed", SEED)
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        graph = {name: value.to(device) for name, value in _tiny_graph(seed).items()}

        signature = inspect.signature(model_cls)
        model_kwargs = {
            name: value for name, value in model_args.items()
            if name in signature.parameters
        }
        model = model_cls(**model_kwargs).to(device).train()
        parameters = {
            name: value for name, value in model.named_parameters()
            if value.requires_grad
        }
        if not parameters:
            raise ValueError("No trainable parameters")
        before = {name: value.detach().clone() for name, value in parameters.items()}
        optimizer = torch.optim.Adam(parameters.values(), **optimizer_args)

        context = dict(graph)
        resolved_args = _resolve_args(config.get("forward_args", config.get("train_args", {})), context)
        matched_args = {}
        for name, parameter in inspect.signature(model.forward).parameters.items():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            key = name if name in resolved_args else FORWARD_ALIASES.get(name)
            if key in resolved_args:
                matched_args[name] = resolved_args[key]
            elif parameter.default is inspect.Parameter.empty:
                raise TypeError(f"unsupported required forward parameter: {name}")

        optimizer.zero_grad(set_to_none=True)
        prediction = model(**matched_args)
        loss = torch.nn.functional.mse_loss(prediction, graph["target"])
        loss.backward()
        finite = {
            name for name, value in parameters.items()
            if value.grad is not None and bool(torch.isfinite(value.grad).all())
        }
        finite_parameters = {id(parameters[name]) for name in finite}
        multiscale_gradients = {}
        for group, paths in (
            ("fine", ("fine_processor_before", "fine_pre_processor", "fine_processor_pre",
                      "processor_fine_before", "processor.fine_blocks", "fine_processor_after",
                      "fine_post_processor", "fine_processor_post", "processor_fine_after",
                      "processor.correction_blocks")),
            ("coarse", ("coarse_processor", "processor_coarse", "coarse_processors",
                        "processor.coarse_blocks")),
            ("transfer", ("down_transfer", "up_transfer", "fusion_mlp",
                          "processor.fusion_mlp")),
        ):
            group_parameters = set()
            for path in paths:
                try:
                    group_parameters.update(
                        id(value) for value in model.get_submodule(path).parameters()
                    )
                except AttributeError:
                    continue
            multiscale_gradients[group] = bool(group_parameters & finite_parameters)
        optimizer.step()
        changed = {
            name for name, value in parameters.items()
            if not torch.equal(before[name], value.detach())
        }

        update_check(current_index, True, "Model-only MSE/Adam probe completed")
        current_index += 1
        update_check(current_index, tuple(prediction.shape) == (NODE_COUNT, 3), str(tuple(prediction.shape)))
        current_index += 1
        update_check(current_index, bool(torch.isfinite(loss)), str(float(loss.detach())))
        current_index += 1
        update_check(current_index, len(finite) >= max(1, int(0.7 * len(parameters))), f"{len(finite)}/{len(parameters)}")
        current_index += 1
        update_check(current_index, all(multiscale_gradients.values()), str(multiscale_gradients))
        current_index += 1
        update_check(current_index, len(changed) >= max(1, int(0.6 * len(parameters))), f"{len(changed)}/{len(parameters)}")

    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks
