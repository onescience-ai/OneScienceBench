#!/usr/bin/env python
"""Evaluator template for standalone model-inference tasks.

Choose exactly one final validation mode:

* ``metric`` when evaluator-owned inputs have trusted labels/targets.
* ``oracle`` when candidate outputs must match a trusted implementation.

All ``TODO(task)`` placeholders fail closed until task-specific rules replace
them.  Both final functions return check_type ``metric`` as required.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
import tempfile
import textwrap
import yaml
import pickle
import h5py
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pathlib import Path
from copy import deepcopy
from typing import Tuple, Dict, List, Any, Callable, Optional, Any, get_origin, get_args, Union


FORWARD_COMPARE_ATOL = 1e-6
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO,实际运行时可能有不同，按照实际的情况配置


candidate_env = os.environ.get('candidate_env')
candidate_dir_hints = os.environ.get('candidate_dir_hints')
stardard_name = os.environ.get('stardard_name')



SUFFIX_MAP = {
    '.npy': 'npy',
    '.npz': 'npz',
    '.pkl': 'pkl',
    '.pickle': 'pkl',
    '.pt': 'pt',
    '.pth': 'pt',
    '.h5': 'h5',
    '.hdf5': 'h5',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.txt': 'txt',
    '.csv': 'csv',
}

DATA_LOADER_REGISTRY = {
    'npy': np.load,
    'npz': np.load,
    'json': lambda p: json.load(open(p)),
    'yaml': lambda p: yaml.safe_load(open(p)),
    'yml': lambda p: yaml.safe_load(open(p)),
    'pkl': lambda p: pickle.load(open(p, 'rb')),
    'pickle': lambda p: pickle.load(open(p, 'rb')),
    'txt': lambda p: open(p).read(),
    'csv': lambda p: np.genfromtxt(p, delimiter=','),
    'h5': lambda p: {k: v[:] for k, v in h5py.File(p, 'r').items()},
    'hdf5': lambda p: {k: v[:] for k, v in h5py.File(p, 'r').items()}
}


class TypeInferencer(ast.NodeVisitor):
    """AST 访问器，推断参数类型"""
    
    def __init__(self, param_names: List[str]):
        self.param_names = param_names
        self.usage = {name: [] for name in param_names}
    
    def visit_Call(self, node):
        """检测方法调用: param.method()"""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                param = node.func.value.id
                if param in self.param_names:
                    method = node.func.attr
                    self.usage[param].append(f"调用 .{method}()")
                elif param == "torch" and node.func.attr in {"cat", "stack", "matmul", "einsum"}:
                    for argument in node.args:
                        for child in ast.walk(argument):
                            if isinstance(child, ast.Name) and child.id in self.param_names:
                                self.usage[child.id].append("传入 torch 张量运算")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """检测张量属性访问，例如 ``x.shape``。"""
        if isinstance(node.value, ast.Name) and node.value.id in self.param_names:
            self.usage[node.value.id].append(f"访问 .{node.attr}")
        self.generic_visit(node)
    
    def visit_BinOp(self, node):
        """检测二元运算: param + something"""
        if isinstance(node.left, ast.Name) and node.left.id in self.param_names:
            op = type(node.op).__name__
            self.usage[node.left.id].append(f"参与 {op} 运算")
        if isinstance(node.right, ast.Name) and node.right.id in self.param_names:
            op = type(node.op).__name__
            self.usage[node.right.id].append(f"参与 {op} 运算")
        self.generic_visit(node)
    
    def visit_Subscript(self, node):
        """检测索引操作: param[index]"""
        if isinstance(node.value, ast.Name) and node.value.id in self.param_names:
            self.usage[node.value.id].append("使用索引 [] 访问")
        self.generic_visit(node)
    
    def visit_If(self, node):
        """检测条件判断: if param:"""
        if isinstance(node.test, ast.Name) and node.test.id in self.param_names:
            self.usage[node.test.id].append("作为布尔条件")
        self.generic_visit(node)
    
    def infer_types(self) -> Dict[str, Any]:
        """根据使用模式推断类型"""
        result = {}
        
        # 模式映射
        type_patterns = {
            '.items()': 'dict',
            '.keys()': 'dict', 
            '.values()': 'dict',
            '.append()': 'list',
            '.extend()': 'list',
            '.pop()': 'list',
            '.split()': 'str',
            '.strip()': 'str',
            '.upper()': 'str',
            '.lower()': 'str',
            '.format()': 'str',
            '.startswith()': 'str',
            '.endswith()': 'str',
            '.to()': 'torch.Tensor',
            '.cuda()': 'torch.Tensor',
            '.cpu()': 'torch.Tensor',
            '.detach()': 'torch.Tensor',
            '.item()': 'torch.Tensor',
            '.shape': 'torch.Tensor',
            '.size()': 'torch.Tensor',
            '.view()': 'torch.Tensor',
            '.reshape()': 'torch.Tensor',
            '.dtype': 'torch.Tensor',
            '.device': 'torch.Tensor',
            '传入 torch 张量运算': 'torch.Tensor',
            '使用索引 [] 访问': 'list/tuple/str/dict',
            '参与 Add 运算': 'int/float/str/torch.Tensor',
            '参与 Mult 运算': 'int/float/torch.Tensor',
            '作为布尔条件': 'bool/any object with __bool__',
        }
        
        for param, usages in self.usage.items():
            inferred = set()
            for usage in usages:
                for pattern, ptype in type_patterns.items():
                    if pattern in usage:
                        inferred.add(ptype)
            # Conversion requires a concrete type.  Prefer Tensor whenever a
            # parameter has an explicit tensor-only operation; otherwise leave
            # the value unchanged instead of returning an ambiguous union.
            result[param] = torch.Tensor if 'torch.Tensor' in inferred else None
        
        return result

def _move_tensors_to_device(data):
    """
    递归地将数据中的 tensor 移动到指定设备
    
    Args:
        data: 要移动的数据（可以是 dict, list, tuple, tensor）
        device: 目标设备
    
    Returns:
        移动后的数据
    """
    if isinstance(data, torch.Tensor):
        return data.to(DEVICE)
    elif isinstance(data, dict):
        return {k: _move_tensors_to_device(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_move_tensors_to_device(item) for item in data)
    else:
        return data

def _parse_type_string(type_str: str):
    """
    将类型字符串解析为实际的类型对象
    
    Args:
        type_str: 类型字符串，如 'Tensor', 'Optional[Tensor]', 'List[int]' 等
    
    Returns:
        解析后的类型对象，如果无法解析则返回 None
    """
    if type_str is None:
        return None
    
    # 基础类型映射
    base_mapping = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'tuple': tuple,
        'dict': dict,
        'set': set,
        'None': None,
        'NoneType': type(None),
    }
    
    # torch 类型映射
    torch_mapping = {
        'Tensor': torch.Tensor,
        'torch.Tensor': torch.Tensor,
        'tensor': torch.Tensor,
        'LongTensor': torch.LongTensor,
        'FloatTensor': torch.FloatTensor,
        'DoubleTensor': torch.DoubleTensor,
    }
    
    # 处理 Optional[T] 或 Union[T, None]
    optional_pattern = r'^(Optional|Union)\[(.*)\]$'
    match = re.match(optional_pattern, type_str)
    if match:
        inner_type_str = match.group(2).strip()
        # 如果内部类型是 'None' 或包含 'None'，尝试提取另一个类型
        if inner_type_str == 'None':
            return None
        # 处理 Union[T, None] 的情况
        if 'None' in inner_type_str:
            parts = [p.strip() for p in inner_type_str.split(',')]
            for p in parts:
                if p != 'None':
                    inner_type_str = p
                    break
        return _parse_type_string(inner_type_str)
    
    # 处理 List[T], Tuple[T], Dict[K, V] 等
    generic_pattern = r'^(List|Tuple|Dict|Set)\[(.*)\]$'
    match = re.match(generic_pattern, type_str)
    if match:
        container_type = match.group(1)
        if container_type == 'List':
            return list
        elif container_type == 'Tuple':
            return tuple
        elif container_type == 'Dict':
            return dict
        elif container_type == 'Set':
            return set
    
    # 处理基本类型
    if type_str in base_mapping:
        return base_mapping[type_str]
    
    # 处理 torch 类型
    if type_str in torch_mapping:
        return torch_mapping[type_str]
    
    # 尝试从全局命名空间获取类型
    try:
        # 尝试直接 eval（谨慎使用）
        # 对于 'Tensor'，会从当前命名空间查找
        if type_str in globals():
            return globals()[type_str]
        if type_str in __builtins__:
            return __builtins__.get(type_str)
    except:
        pass
    
    return None

def _safe_isinstance_check(value, target_type):
    """
    安全地进行 isinstance 检查，支持字符串类型
    
    Args:
        value: 要检查的值
        target_type: 目标类型（可以是类型对象或字符串）
    
    Returns:
        (is_instance, resolved_type): 是否是目标类型的实例，以及解析后的类型
    """
    # 如果 target_type 是 None，返回 False
    if target_type is None:
        return False, None
    
    # 如果 target_type 是字符串，尝试解析
    if isinstance(target_type, str):
        resolved_type = _parse_type_string(target_type)
        if resolved_type is None:
            return False, None
        target_type = resolved_type
    
    # 如果 target_type 不是类型对象，尝试获取 origin
    if not isinstance(target_type, type):
        try:
            origin = get_origin(target_type)
            if origin is not None and isinstance(origin, type):
                target_type = origin
            else:
                return False, None
        except:
            return False, None
    
    # 安全地进行 isinstance 检查
    try:
        return isinstance(value, target_type), target_type
    except TypeError:
        return False, target_type

def _convert_arg_to_type(value, target_type):
    """
    将值转换为目标类型。
    支持常见类型：torch.Tensor, list, tuple, dict, int, float, str, bool, None
    支持字符串类型的类型注解，如 'Tensor', 'Optional[Tensor]' 等
    
    Args:
        value: 要转换的值
        target_type: 目标类型（可以是类型对象或字符串）
    
    Returns:
        转换后的值
    """

    # 1. 如果值已经是目标类型，直接返回
    is_instance, resolved_type = _safe_isinstance_check(value, target_type)
    if is_instance:
        return value
    
    # 2. 如果 target_type 是字符串，尝试解析为实际类型
    if isinstance(target_type, str):
        resolved_type = _parse_type_string(target_type)
        if resolved_type is not None:
            target_type = resolved_type
        else:
            # 无法解析，直接返回原值
            return value
    
    # 3. 处理 None
    if value is None or (isinstance(value, str) and value.lower() in ("none", "null")):
        return None
    
    # 4. 处理 torch.Tensor
    if target_type == torch.Tensor or (hasattr(target_type, '__origin__') and target_type.__origin__ is torch.Tensor):
        # 如果输入是 list 或 tuple，转换为 tensor
        if isinstance(value, (list, tuple)):
            # 如果列表为空，返回空 tensor
            if len(value) == 0:
                return torch.tensor([])
            # 检查是否所有元素都是 tensor，如果是则合并
            if all(isinstance(v, torch.Tensor) for v in value):
                try:
                    return torch.stack(value) if len(value) > 0 else torch.tensor([])
                except:
                    return torch.cat(value, dim=0) if len(value) > 0 else torch.tensor([])
            return torch.tensor(value)
        # 如果输入是 dict，尝试转换为 tensor
        if isinstance(value, dict):
            return torch.tensor(list(value.values()))
        # 如果输入是标量，转换为 tensor
        if isinstance(value, (int, float, bool)):
            return torch.tensor(value)
        # 如果输入已经是 tensor，直接返回
        if isinstance(value, torch.Tensor):
            return value
        return value
    
    # 5. 处理 list 类型 (包括 List[T])
    if target_type == list or (hasattr(target_type, '__origin__') and target_type.__origin__ is list):
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return value
    
    # 6. 处理 tuple 类型 (包括 Tuple[T])
    if target_type == tuple or (hasattr(target_type, '__origin__') and target_type.__origin__ is tuple):
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, torch.Tensor):
            return tuple(value.tolist())
        return value
    
    # 7. 处理 dict 类型
    if target_type == dict or (hasattr(target_type, '__origin__') and target_type.__origin__ is dict):
        return value
    
    # 8. 处理基本类型
    if target_type == int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if target_type == float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if target_type == bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        try:
            return bool(value)
        except:
            return value
    if target_type == str:
        return str(value)
    
    # 9. 默认返回原值
    return value

def _compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算均方根误差"""
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse).item()


def _compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算均方误差"""
    return torch.mean((pred - target) ** 2).item()


def _compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算平均绝对误差"""
    return torch.mean(torch.abs(pred - target)).item()


def _compute_mape(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算平均绝对百分比误差"""
    # 避免除零
    mask = target.abs() > 1e-8
    if not mask.any():
        return float('inf')
    return torch.mean(torch.abs((pred[mask] - target[mask]) / target[mask])).item() * 100


def _compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算 R² 决定系数"""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return (1 - ss_res / ss_tot).item()


def _compute_mae_percent(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算平均绝对误差百分比（另一种形式）"""
    return _compute_mae(pred, target) / (torch.mean(torch.abs(target)).item() + 1e-8) * 100


def _compute_huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> float:
    """计算 Huber 损失"""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(loss).item()


METRIC_REGISTRY = {
    "rmse": _compute_rmse,
    "mse": _compute_mse,
    "mae": _compute_mae,
    "mape": _compute_mape,
    "r2": _compute_r2,
    "mae_percent": _compute_mae_percent,
    "huber": _compute_huber_loss,
}


def _resolve_args(args: dict, context: dict) -> dict:
    """
    解析参数中的占位符（如 $device, ${config.key}, $context_key, $checkpoint_dir/checkpoints/elas_best.pt）

    支持的占位符格式:
        - $device: 解析为当前 DEVICE
        - ${config.key}: 从 config 字典中获取（支持嵌套）
        - $context_key: 从 context 中获取
        - $context_key/path/to/file: 从 context 获取前缀并与后续路径拼接
        - ${context_key}/path/to/file: 从 context 获取前缀并与后续路径拼接
    """
    import re
    import os
    
    def resolve_value(value):
        """递归解析单个值"""
        if isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(v) for v in value]
        elif not isinstance(value, str) or not value.startswith("$") or value.startswith("$$"):
            return None if value == "None" else value
        
        # 处理字符串占位符
        result = value
        
        # 1. 先处理 ${...} 格式的占位符
        pattern_brace = r'\$\{([^}]+)\}'
        for match in re.finditer(pattern_brace, value):
            placeholder = match.group(0)
            path = match.group(1)
            
            # 解析占位符内容
            if path == "device":
                resolved_part = DEVICE
            elif path.startswith("config."):
                path_parts = path.split(".")[1:]
                resolved_part = context.get("config", {})
                for part in path_parts:
                    resolved_part = resolved_part.get(part) if isinstance(resolved_part, dict) else None
                    if resolved_part is None:
                        break
                resolved_part = None if resolved_part == "None" else resolved_part
            else:
                resolved_part = context.get(path, None)
            
            # 替换占位符
            if isinstance(resolved_part, str):
                result = result.replace(placeholder, resolved_part)
            else:
                result = result.replace(placeholder, str(resolved_part) if resolved_part is not None else "")
        
        # 2. 处理 $xxx 格式（可能包含路径拼接）
        if result == value:  # 如果没有被 ${} 处理过，或者处理完后还有 $xxx 格式
            # 检查是否包含路径分隔符
            sep_pos = -1
            for sep in ['/', '\\']:
                pos = result.find(sep, 1)
                if pos != -1:
                    sep_pos = pos if sep_pos == -1 or pos < sep_pos else sep_pos
            
            if sep_pos != -1:
                # 有路径拼接: $var_name/path/to/file
                var_name = result[1:sep_pos]
                suffix = result[sep_pos:]
                
                # 解析变量名
                if var_name == "device":
                    resolved_var = DEVICE
                elif var_name.startswith("config."):
                    path_parts = var_name.split(".")[1:]
                    resolved_var = context.get("config", {})
                    for part in path_parts:
                        resolved_var = resolved_var.get(part) if isinstance(resolved_var, dict) else None
                        if resolved_var is None:
                            break
                    resolved_var = None if resolved_var == "None" else resolved_var
                else:
                    resolved_var = context.get(var_name, None)
                
                if isinstance(resolved_var, str):
                    return os.path.join(resolved_var, suffix.lstrip('/\\'))
                else:
                    return None if resolved_var is None else str(resolved_var) + suffix
            else:
                # 没有路径拼接，按简单占位符处理
                name = result[1:]
                if name == "device":
                    return DEVICE
                elif name.startswith("config."):
                    path_parts = name.split(".")[1:]
                    resolved = context.get("config", {})
                    for part in path_parts:
                        resolved = resolved.get(part) if isinstance(resolved, dict) else None
                        if resolved is None:
                            break
                    return None if resolved == "None" else resolved
                else:
                    return context.get(name, None)
        
        return result
    
    return {k: resolve_value(v) for k, v in args.items()}


def _execute_hooks(module, hooks: list, context: dict, caller_globals=None, caller_locals=None) -> dict:
    """执行前置钩子"""
    if not hooks:
        return context

    current_globals = globals()
    for idx, hook_config in enumerate(hooks):
        func_name = hook_config.get("function")
        if not func_name:
            raise ValueError(f"Hook {idx}: 'function' is required")

        args_config = hook_config.get("args", {})

        # 1. 从模型模块查找
        hook_func = getattr(module, func_name, None)

        # 2. 从调用者脚本的全局变量查找
        if hook_func is None and caller_globals:
            hook_func = caller_globals.get(func_name)

        # 3. 从调用者脚本的局部变量查找
        if hook_func is None and caller_locals:
            hook_func = caller_locals.get(func_name)

        # 4. 从当前模块查找
        if hook_func is None:
            hook_func = current_globals.get(func_name)

        if hook_func is None or not callable(hook_func):
            raise ValueError(f"Hook {idx}: function '{func_name}' not found or not callable")

        resolved_args = _resolve_args(args_config, context)
        hook_result = hook_func(**resolved_args)

        save_as = hook_config.get("save_as")
        if save_as:
            context[save_as] = hook_result

    return context


def _evaluator_root() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / 'benchmarks' / 'datasets').is_dir():
            return candidate
    raise FileNotFoundError('Could not locate repository root containing benchmarks/datasets.')


def _find_candidate_dir() -> Path:
    candidates: list[Path] = []

    explicit = os.environ.get(candidate_env)
    if explicit:
        candidates.append(Path(explicit).expanduser())

    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry).expanduser().resolve()
        name = candidate.name
        if name in candidate_dir_hints or any(name.startswith(prefix + "_V") for prefix in candidate_dir_hints):
            candidates.append(candidate)
    if len(candidates):
        return candidates[-1].resolve()

    raise FileNotFoundError(
        "Could not locate an inference submission directory. "
        f"Set {candidate_env} or add a matching directory to PYTHONPATH. "
    )


def _reference_model_path() -> Path:
    return (
        _repo_root()
        / "benchmarks"
        / "standard"
        / stardard_name
    ).resolve()


CANDIDATE_DIR = _find_candidate_dir()
REFERENCE_DIR = _reference_model_path()


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_candidate_modules(file_name: str):
    """file_name example: inference.py"""
    candidate_dir_str = str(CANDIDATE_DIR)
    if CANDIDATE_DIR not in sys.path:
        sys.path.insert(0, candidate_dir_str)

    module_name = Path(file_name).stem
    module = _load_module_from_path(
        f"candidate_{module_name}",
        CANDIDATE_DIR / file_name)

    return module


def _load_reference_modules(file_name: str):
    """file_name example: inference.py"""
    reference_dir_str = str(REFERENCE_DIR)
    if reference_dir_str not in sys.path:
        sys.path.insert(0, reference_dir_str)

    module_name = Path(file_name).stem
    module = _load_module_from_path(
        f"reference_{module_name}",
        REFERENCE_DIR / file_name)

    return module


def _load_class(module, class_name):
    cls = getattr(module, class_name, None)
    if not inspect.isclass(cls):
        raise TypeError(f"{class_name} is not a class")
    return cls


def _load_candidate_config(config_name: str) -> dict[str, Any]:
    """config must be yaml or yml formate"""

    with open(str(CANDIDATE_DIR / config_name), "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("Submission config must contain a mapping")
    return config


def _load_reference_config(config_name: str) -> dict[str, Any]:
    """标准代码实现中的配置文件加载,TODO 集成该方法修改对应的逻辑为当前任务的标准配置文件加载 """
    config_path = REFERENCE_DIR / config_name
    if not config_path.is_file():
        return {}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Reference config must contain a mapping")
    return config


def _initialize_parameters_with_constants(
        model: Any,
        weight_value: float = 0.002,
        bias_value: float = 0.0001,
        embedding_value: float = 0.001,
        norm_value: float = 1.0,
        other_value: float = 0.001,
) -> Any:
    """Initialize every parameter with a deterministic constant so that two
    structurally equivalent models produce identical forward outputs.

    Pattern derived from the functional-attention oracle evaluator: weights
    receive a centered pattern scaled to ``weight_value``; biases, embeddings,
    norms and other parameter categories each get a fixed scalar value.
    """
    import torch

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            name_lower = name.lower()
            if "bias" in name_lower:
                parameter.fill_(bias_value)
            elif (
                    "embedding" in name_lower
                    or "embed" in name_lower
                    or "token" in name_lower
                    or "position" in name_lower
                    or "pos_embed" in name_lower
            ):
                parameter.fill_(embedding_value)
            elif (
                    "norm" in name_lower
                    or "ln" in name_lower
                    or "layernorm" in name_lower
            ):
                parameter.fill_(norm_value)
            elif parameter.ndim >= 2 and "weight" in name_lower:
                pattern = torch.arange(
                    parameter.numel(),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ).reshape(parameter.shape)
                pattern = pattern - pattern.mean()
                scale = pattern.abs().max().clamp_min(1.0)
                parameter.copy_(weight_value * pattern / scale)
            else:
                parameter.fill_(other_value if parameter.ndim == 0 else embedding_value)

    return model


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


def check_config(config: Dict, caller_globals=None, caller_locals=None) -> List[Tuple[str, bool, str]]:
    """
    检查候选提交目录中的配置文件内容是否为字典格式。

    config 示例:
    {
        "config_name": "config.yaml"
    }

    检查点:
        - config: 检查配置文件内容是否可以转换为字典格式
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "config.format",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ------------------ 1. 检查配置文件内容是否为字典格式 ------------------
        candidate_config_file = config.get("config_name")
        if not candidate_config_file:
            raise ValueError("config['config_name'] is missing or empty")

        loaded_config = _load_candidate_config(candidate_config_file)

        # 核心检查：内容必须为字典格式
        if not isinstance(loaded_config, dict):
            raise TypeError(
                f"Config content must be a dictionary (YAML mapping), "
                f"got {type(loaded_config).__name__}: {candidate_config_file}"
            )

        update_check(
            current_index,
            True,
            f"Config is valid dictionary with {len(loaded_config)} key(s): {candidate_config_file}"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


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


def check_interface(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    检查训练脚本中的接口定义。

    config 示例:
    {
        "files": ["train.py", "trainer.py"],
        "required_interfaces": ["train", "main", "Trainer.train"]
    }

    检查点:
        - {interface_type}:{file}_load: 检查能否加载每个训练文件
        - {interface_name}: 检查接口是否在至少一个文件中存在
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = []
    checks = []

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

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
            check_name = f"{interface_type}:{interface_file}_load"
            CHECKPOINT_NAMES.append(check_name)
            checks.append((check_name, False, "Not executed"))

        for interface_name in required_interfaces:
            CHECKPOINT_NAMES.append(interface_name)
            checks.append((interface_name, False, "Not executed"))

        # ------------------ 3. 加载所有模块 ------------------
        modules = []
        for idx, interface_file in enumerate(interface_files_files):
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

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(-1, False, error_detail)

    return checks


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
        context = _execute_hooks_with_save(candidate_module, pre_hooks, context)
        
        # ------------------ 3. 解析候选方法的参数 ------------------
        args_config = candidate_config.get("args", {})
        resolved_args = _resolve_args(args_config, context)
        resolved_args = _move_tensors_to_device(resolved_args)
        
        # ------------------ 4. 执行候选方法 ------------------
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


def check_data_availability(config: Dict, caller_globals=None, caller_locals=None) -> List[Tuple[str, bool, str]]:
    """
    检查数据集的可用性，验证配置中指定的数据文件是否存在且可加载。

    config 示例:
    {
        "config_name": "config.yaml",  # 配置文件路径
        "data_files": ["sigma_path", "xy_path"]  # 在config中的键名列表
    }

    检查点:
        - dataset_availability:file_exist: 检查所有数据文件是否存在
        - dataset_availability:load: 检查能否加载所有数据文件
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "dataset_availability:file_exist",
        "dataset_availability:load",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ================================================================
        # 1. 获取配置信息
        # ================================================================
        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("config_name not found in config")

        # 加载配置文件
        config_data = _load_candidate_config(config_name)

        # ================================================================
        # 2. 获取数据文件键名列表
        # ================================================================
        data_keys = config.get("data_files")
        if not data_keys:
            raise ValueError("data_files not found in config")

        if not isinstance(data_keys, list):
            raise TypeError(f"data_files must be a list, got {type(data_keys)}")

        # 验证每个键名
        for idx, key in enumerate(data_keys):
            if not isinstance(key, str):
                raise TypeError(f"data_files[{idx}] must be a string, got {type(key)}")
            if not key.strip():
                raise ValueError(f"data_files[{idx}] is empty string")

        # ================================================================
        # 3. 解析实际文件路径
        # ================================================================
        resolved_paths = {}
        file_exist_msgs = []

        for path_key in data_keys:
            # 从配置数据中获取实际路径
            actual_path = config_data.get(path_key)
            if actual_path is None:
                raise KeyError(f"Path key '{path_key}' not found in config: {config_name}")

            if not isinstance(actual_path, (str, Path)):
                raise TypeError(f"Path for '{path_key}' must be string or Path, got {type(actual_path)}")

            resolved_path = Path(actual_path).resolve()
            resolved_paths[path_key] = resolved_path

            if not resolved_path.exists():
                raise FileNotFoundError(
                    f"File not found: {resolved_path} (path_key: {path_key})"
                )

            file_exist_msgs.append(f"{path_key}: {resolved_path}")

        update_check(
            current_index,
            True,
            f"All files exist: {'; '.join(file_exist_msgs)}"
        )

        # ================================================================
        # 4. 检查所有数据能否加载（通过后缀自动识别文件类型）
        # ================================================================
        current_index += 1
        load_msgs = []

        for path_key in data_keys:
            resolved_path = resolved_paths[path_key]

            # 从文件后缀推断文件类型
            suffix = ''.join(resolved_path.suffixes)  # 支持 .tar.gz 等多后缀
            if not suffix:
                suffix = resolved_path.suffix.lower()

            file_type = SUFFIX_MAP.get(suffix.lower())

            if not file_type:
                # 尝试使用最后一个后缀
                last_suffix = resolved_path.suffix.lower()
                file_type = SUFFIX_MAP.get(last_suffix)

            if not file_type:
                raise ValueError(
                    f"Unsupported file extension: '{resolved_path.suffix}' for {path_key}. "
                    f"Supported extensions: {list(SUFFIX_MAP.keys())}"
                )

            # 获取对应的加载器
            loader = DATA_LOADER_REGISTRY.get(file_type)
            if not loader:
                raise KeyError(
                    f"File type '{file_type}' (from extension {resolved_path.suffix}) "
                    f"not supported by registry. Available: {list(DATA_LOADER_REGISTRY.keys())}"
                )

            data = loader(resolved_path)

            # 验证加载的数据是否有效
            if hasattr(data, 'shape'):
                load_msgs.append(f"{path_key}: shape={data.shape}")
            elif isinstance(data, (dict, list)):
                load_msgs.append(f"{path_key}: type={type(data).__name__}, len={len(data)}")
            elif hasattr(data, '__len__'):
                load_msgs.append(f"{path_key}: len={len(data)}")
            else:
                load_msgs.append(f"{path_key}: loaded successfully")

        update_check(
            current_index,
            True,
            f"All data loaded: {'; '.join(load_msgs)}"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> List[Tuple[str, bool, str]]:
    """
    检查数据集的 schema（形状、字段等）是否符合预期。

    config 示例:
    {
        "config_name": "config.yaml",  # 包含具体数据路径配置
        "schema": [  # 数据文件的检查规则
            {
                "path_key": "sigma_path",  # 在config中的键名，文件类型通过后缀自动识别
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
        - dataset_schema:file_exist: 检查数据文件是否存在
        - dataset_schema:file_load: 检查数据文件是否能正常加载
        - dataset_schema:shape_check: 检查数据形状是否符合预期
        - dataset_schema:field_check: 检查数据字段是否符合预期
    """

    # ================================================================
    # 0. 定义检查点名称和辅助函数
    # ================================================================
    CHECKPOINT_NAMES = [
        "dataset_schema:file_load",
        "dataset_schema:shape_check",
        "dataset_schema:field_check",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ================================================================
        # 1. 获取配置信息
        # ================================================================
        config_name = config.get("config_name")
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

        # 处理每个数据文件
        for file_spec in schema_config:
            path_key = file_spec["path_key"]

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

            # 加载文件
            try:
                data = loader(file_path)
                loaded_data[path_key] = data
                file_load_msgs.append(f"{path_key}: loaded from {file_path}")

                # ------------------ 形状检查 ------------------
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

                # ------------------ 字段检查（针对结构化数据） ------------------
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

        # 检查点1: file_load
        update_check(
            current_index,
            True,
            f"All files loaded: {'; '.join(file_load_msgs)}"
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

    except Exception as e:
        error_detail = traceback.format_exc()
        # 更新当前检查点为失败
        update_check(current_index, False, error_detail)

    return checks


def check_data_cleaning(config: Dict, caller_globals=None, caller_locals=None):
    """
    调用候选代码的数据清洗函数，检查清洗后的结果。

    支持两种模式:

    【模式1 - 模块级函数】
    data_utils.py 中包含独立的 load_dataset() 和 clean_data(raw_data) 函数
    适用场景：数据处理逻辑简单，无需维护状态

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "clean_func": "clean_data",           # 可选，默认 "clean_data"
        "load_params": {"data_path": "data.csv"}["data.csv"],   # 可选，传递给 load_func 的位置参数列表

    }


    【模式2 - 类方法】
    data_utils.py 中包含数据处理类，通过实例化后调用方法
    适用场景：需要维护状态、配置复杂、或需要复用同一处理流程

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "class_name": "DataProcessor",        # 必填，类名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "clean_func": "clean_data",           # 可选，默认 "clean_data"
        "init_kwargs": {                      # 可选，类初始化的关键字参数字典
            "data_path": "data.csv",
            "mode": "train",
            "batch_size": 32
        }
    }


    检查点:
        - dataset:cleaning_module: 检查能否加载数据模块
        - dataset:clean_result_not_none: 检查清洗结果是否为空
        - dataset:clean_finite: 检查清洗结果是否包含 NaN/Inf
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "dataset:cleaning_module",
        "dataset:clean_result_not_none",
        "dataset:clean_finite",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ------------------ 1. 加载数据模块 ------------------
        dataset_file = config.get("dataset_file")
        if not dataset_file:
            raise ValueError("config['dataset_file'] is missing or empty")

        data_module = _load_candidate_modules(dataset_file)
        update_check(current_index, True, f"Data module loaded from {dataset_file}")

        # ------------------ 2. 获取配置参数 ------------------
        load_func_name = config.get("load_func", "load_dataset")
        clean_func_name = config.get("clean_func", "clean_data")
        class_name = config.get("class_name")

        # 获取参数配置（扁平化结构）
        load_params = config.get("load_params", {})  # 直接获取列表
        init_kwargs = config.get("init_kwargs", {})  # 直接获取字典

        # ------------------ 3. 获取可调用对象 (函数或类实例) ------------------
        if class_name:
            if not hasattr(data_module, class_name):
                raise AttributeError(f"data_module does not have class '{class_name}'")
            cls = getattr(data_module, class_name)
            if not callable(cls):
                raise AttributeError(f"'{class_name}' is not callable")

            instance = cls(**init_kwargs)

            # 获取 load_dataset 方法
            if not hasattr(instance, load_func_name) or not callable(getattr(instance, load_func_name)):
                raise AttributeError(f"Instance does not have callable '{load_func_name}'")
            load_func = getattr(instance, load_func_name)

            # 获取 clean_data 方法
            if not hasattr(instance, clean_func_name) or not callable(getattr(instance, clean_func_name)):
                raise AttributeError(f"Instance does not have callable '{clean_func_name}'")
            clean_func = getattr(instance, clean_func_name)

            update_check(current_index, True, f"Instantiated {class_name} from module")
            # 执行 load 方法（无参数，数据路径已在初始化时传入）
            raw_data = load_func()

        else:
            if not hasattr(data_module, load_func_name) or not callable(getattr(data_module, load_func_name)):
                raise AttributeError(f"data_module does not have callable '{load_func_name}'")
            load_func = getattr(data_module, load_func_name)

            if not hasattr(data_module, clean_func_name) or not callable(getattr(data_module, clean_func_name)):
                raise AttributeError(f"data_module does not have callable '{clean_func_name}'")
            clean_func = getattr(data_module, clean_func_name)

            update_check(current_index, True, f"Found functions {load_func_name} and {clean_func_name} in module")

            raw_data = load_func(**load_params)

        # ------------------ 4. 执行数据清洗 ------------------
        current_index += 1
        cleaned_data = clean_func(raw_data)

        if cleaned_data is None:
            raise ValueError(f"{clean_func_name} returned None")

        update_check(
            current_index,
            True,
            f"cleaned_data type={type(cleaned_data).__name__}"
        )

        # ------------------ 5. 检查数据是否包含 NaN/Inf ------------------
        current_index += 1
        if isinstance(cleaned_data, np.ndarray):
            finite_ok = np.isfinite(cleaned_data).all()
            if not finite_ok:
                raise ValueError("cleaned_data contains NaN or Inf values")
            update_check(
                current_index,
                True,
                f"All values are finite, shape={cleaned_data.shape}"
            )
        else:
            # 非 numpy 数组，尝试转换为数组检查
            try:
                arr = np.array(cleaned_data)
                if np.isfinite(arr).all():
                    update_check(
                        current_index,
                        True,
                        f"All values are finite, shape={arr.shape}"
                    )
                else:
                    raise ValueError("cleaned_data contains NaN or Inf values")
            except Exception as e:
                # 无法转换为数组，跳过检查
                update_check(
                    current_index,
                    True,
                    f"Non-numpy data type, finite check skipped: {type(cleaned_data).__name__}"
                )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_data_filtering(config, caller_globals=None, caller_locals=None):
    """
    调用数据筛选函数，检查筛选后的数据是否满足约束。

    支持两种模式:

    【模式1 - 模块级函数】
    data_utils.py 中包含独立的 load_dataset() 和 filter_data(raw_data) 函数
    适用场景：数据处理逻辑简单，无需维护状态

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "filter_func": "filter_data",         # 可选，默认 "filter_data"
        "load_params": {"data_path": "data.csv"},  # 可选，传递给 load_func 的关键字参数
        "min_value": 0.0,                     # 可选，最小值约束
        "max_value": 1.0                      # 可选，最大值约束
    }

    【模式2 - 类方法】
    data_utils.py 中包含数据处理类，通过实例化后调用方法
    适用场景：需要维护状态、配置复杂、或需要复用同一处理流程

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "class_name": "DataProcessor",        # 必填，类名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "filter_func": "filter_data",         # 可选，默认 "filter_data"
        "init_kwargs": {                      # 可选，类初始化的关键字参数字典
            "data_path": "data.csv",
            "mode": "train",
            "batch_size": 32
        },
        "min_value": 0.0,                     # 可选，最小值约束
        "max_value": 1.0                      # 可选，最大值约束
    }

    检查点:
        - data:filtering_module: 检查能否加载数据模块
        - data:filter_result: 检查筛选结果是否为空
        - data:filter_min_value: 检查数据是否满足最小值约束（如果配置了 min_value）
        - data:filter_max_value: 检查数据是否满足最大值约束（如果配置了 max_value）
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "data:filtering_module",
        "data:filter_result",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    # 动态添加 min/max 检查点
    has_min_check = "min_value" in config
    has_max_check = "max_value" in config

    if has_min_check:
        CHECKPOINT_NAMES.append("data:filter_min_value")
        checks.append(("data:filter_min_value", False, "Not executed due to earlier failure"))
    if has_max_check:
        CHECKPOINT_NAMES.append("data:filter_max_value")
        checks.append(("data:filter_max_value", False, "Not executed due to earlier failure"))

    try:
        # ------------------ 1. 加载数据模块 ------------------
        dataset_file = config.get("dataset_file")
        if not dataset_file:
            raise ValueError("config['dataset_file'] is missing or empty")

        data_module = _load_candidate_modules(dataset_file)
        update_check(current_index, True, f"Data module loaded from {dataset_file}")

        # ------------------ 2. 获取配置参数 ------------------
        load_func_name = config.get("load_func", "load_dataset")
        filter_func_name = config.get("filter_func", "filter_data")
        class_name = config.get("class_name")

        # 获取参数配置
        load_params = config.get("load_params", {})  # 关键字参数字典
        init_kwargs = config.get("init_kwargs", {})  # 类初始化关键字参数

        # ------------------ 3. 获取可调用对象 (函数或类实例) ------------------
        if class_name:
            # 模式2: 类方法
            if not hasattr(data_module, class_name):
                raise AttributeError(f"data_module does not have class '{class_name}'")
            cls = getattr(data_module, class_name)
            if not callable(cls):
                raise AttributeError(f"'{class_name}' is not callable")

            instance = cls(**init_kwargs)

            # 获取 load_dataset 方法
            if not hasattr(instance, load_func_name) or not callable(getattr(instance, load_func_name)):
                raise AttributeError(f"Instance does not have callable '{load_func_name}'")
            load_func = getattr(instance, load_func_name)

            # 获取 filter_data 方法
            if not hasattr(instance, filter_func_name) or not callable(getattr(instance, filter_func_name)):
                raise AttributeError(f"Instance does not have callable '{filter_func_name}'")
            filter_func = getattr(instance, filter_func_name)

            update_check(current_index, True,
                         f"Instantiated {class_name} from module, using {load_func_name} and {filter_func_name}")
            # 执行 load 方法
            raw_data = load_func(**load_params)

        else:
            # 模式1: 模块级函数
            if not hasattr(data_module, load_func_name) or not callable(getattr(data_module, load_func_name)):
                raise AttributeError(f"data_module does not have callable '{load_func_name}'")
            load_func = getattr(data_module, load_func_name)

            if not hasattr(data_module, filter_func_name) or not callable(getattr(data_module, filter_func_name)):
                raise AttributeError(f"data_module does not have callable '{filter_func_name}'")
            filter_func = getattr(data_module, filter_func_name)

            update_check(current_index, True, f"Found functions {load_func_name} and {filter_func_name} in module")
            raw_data = load_func(**load_params)

        # ------------------ 4. 执行数据筛选 ------------------
        current_index += 1
        filtered_data = filter_func(raw_data)

        if filtered_data is None:
            raise ValueError(f"{filter_func_name} returned None")

        update_check(
            current_index,
            True,
            f"filtered_data type={type(filtered_data).__name__}"
        )

        # ------------------ 5. 检查数据范围（如果配置了 min_value/max_value） ------------------
        if isinstance(filtered_data, np.ndarray):
            # 检查最小值
            if has_min_check:
                min_value = config["min_value"]
                check_idx = len(CHECKPOINT_NAMES) - (2 if has_max_check else 1)
                min_ok = np.all(filtered_data >= min_value)
                if not min_ok:
                    raise ValueError(
                        f"Data contains values below min_value: min={filtered_data.min()}, expected >= {min_value}"
                    )
                checks[check_idx] = (
                    "data:filter_min_value",
                    True,
                    f"min={filtered_data.min()}, expected>={min_value}"
                )

            # 检查最大值
            if has_max_check:
                max_value = config["max_value"]
                check_idx = len(CHECKPOINT_NAMES) - 1
                max_ok = np.all(filtered_data <= max_value)
                if not max_ok:
                    raise ValueError(
                        f"Data contains values above max_value: max={filtered_data.max()}, expected <= {max_value}"
                    )
                checks[check_idx] = (
                    "data:filter_max_value",
                    True,
                    f"max={filtered_data.max()}, expected<={max_value}"
                )
        else:
            # 非 numpy 数组，尝试转换为数组检查
            try:
                arr = np.array(filtered_data)
                if has_min_check:
                    check_idx = len(CHECKPOINT_NAMES) - (2 if has_max_check else 1)
                    min_ok = np.all(arr >= config["min_value"])
                    if not min_ok:
                        raise ValueError(
                            f"Data contains values below min_value: min={arr.min()}, expected>={config['min_value']}")
                    checks[check_idx] = (
                        "data:filter_min_value",
                        True,
                        f"min={arr.min()}, expected>={config['min_value']}"
                    )
                if has_max_check:
                    check_idx = len(CHECKPOINT_NAMES) - 1
                    max_ok = np.all(arr <= config["max_value"])
                    if not max_ok:
                        raise ValueError(
                            f"Data contains values above max_value: max={arr.max()}, expected<={config['max_value']}")
                    checks[check_idx] = (
                        "data:filter_max_value",
                        True,
                        f"max={arr.max()}, expected<={config['max_value']}"
                    )
            except Exception as e:
                # 无法转换为数组，跳过检查
                if has_min_check:
                    check_idx = len(CHECKPOINT_NAMES) - (2 if has_max_check else 1)
                    checks[check_idx] = (
                        "data:filter_min_value",
                        True,
                        f"Non-numpy data, min check skipped: {type(filtered_data).__name__}"
                    )
                if has_max_check:
                    check_idx = len(CHECKPOINT_NAMES) - 1
                    checks[check_idx] = (
                        "data:filter_max_value",
                        True,
                        f"Non-numpy data, max check skipped: {type(filtered_data).__name__}"
                    )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_data_normalization(config, caller_globals=None, caller_locals=None) -> list(tuple):
    """
    调用候选归一化函数，检查归一化结果。

    支持两种模式:

    【模式1 - 模块级函数】
    data_utils.py 中包含独立的 load_dataset() 和 normalize_data(raw_data) 函数
    适用场景：数据处理逻辑简单，无需维护状态

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "normalize_func": "normalize_data",   # 可选，默认 "normalize_data"
        "load_params": {"data_path": "data.csv"},  # 可选，传递给 load_func 的关键字参数
        "normalization_method": "reference"   # 必填，"reference" 或 "minmax"
    }

    【模式2 - 类方法】
    data_utils.py 中包含数据处理类，通过实例化后调用方法
    适用场景：需要维护状态、配置复杂、或需要复用同一处理流程

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "class_name": "DataProcessor",        # 必填，类名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "normalize_func": "normalize_data",   # 可选，默认 "normalize_data"
        "init_kwargs": {                      # 可选，类初始化的关键字参数字典
            "data_path": "data.csv",
            "mode": "train",
            "batch_size": 32
        },
        "normalization_method": "reference"   # 必填，"reference" 或 "minmax"
    }

    检查点:
        - data:normalization_module: 检查能否加载数据模块
        - data:normalization_result: 检查归一化结果是否为空
        - data:normalization_finite: 检查归一化结果是否包含 NaN/Inf
        - data:standardization_mean (reference): 检查均值是否接近 0
        - data:standardization_std (reference): 检查标准差是否接近 1
        - data:minmax_range (minmax): 检查数据是否在 [0, 1] 范围内
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "data:normalization_module",
        "data:normalization_result",
        "data:normalization_finite",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ------------------ 1. 加载数据模块 ------------------
        dataset_file = config.get("dataset_file")
        if not dataset_file:
            raise ValueError("config['dataset_file'] is missing or empty")

        data_module = _load_candidate_modules(dataset_file)
        update_check(current_index, True, f"Data module loaded from {dataset_file}")

        # ------------------ 2. 获取配置参数 ------------------
        load_func_name = config.get("load_func", "load_dataset")
        normalize_func_name = config.get("normalize_func", "normalize_data")
        class_name = config.get("class_name")

        # 获取参数配置
        load_params = config.get("load_params", {})  # 关键字参数字典
        init_kwargs = config.get("init_kwargs", {})  # 类初始化关键字参数

        # 获取归一化方法（必填）
        method = config.get("normalization_method")
        if not method:
            raise ValueError("config['normalization_method'] is required: 'reference' or 'minmax'")
        if method not in ["reference", "minmax"]:
            raise ValueError(f"normalization_method must be 'reference' or 'minmax', got '{method}'")

        # ------------------ 3. 获取可调用对象 (函数或类实例) ------------------
        if class_name:
            # 模式2: 类方法
            if not hasattr(data_module, class_name):
                raise AttributeError(f"data_module does not have class '{class_name}'")
            cls = getattr(data_module, class_name)
            if not callable(cls):
                raise AttributeError(f"'{class_name}' is not callable")

            instance = cls(**init_kwargs)

            # 获取 load_dataset 方法
            if not hasattr(instance, load_func_name) or not callable(getattr(instance, load_func_name)):
                raise AttributeError(f"Instance does not have callable '{load_func_name}'")
            load_func = getattr(instance, load_func_name)

            # 获取 normalize_data 方法
            if not hasattr(instance, normalize_func_name) or not callable(getattr(instance, normalize_func_name)):
                raise AttributeError(f"Instance does not have callable '{normalize_func_name}'")
            normalize_func = getattr(instance, normalize_func_name)

            update_check(current_index, True,
                         f"Instantiated {class_name} from module, using {load_func_name} and {normalize_func_name}")
            # 执行 load 方法
            raw_data = load_func(**load_params)

        else:
            # 模式1: 模块级函数
            if not hasattr(data_module, load_func_name) or not callable(getattr(data_module, load_func_name)):
                raise AttributeError(f"data_module does not have callable '{load_func_name}'")
            load_func = getattr(data_module, load_func_name)

            if not hasattr(data_module, normalize_func_name) or not callable(getattr(data_module, normalize_func_name)):
                raise AttributeError(f"data_module does not have callable '{normalize_func_name}'")
            normalize_func = getattr(data_module, normalize_func_name)

            update_check(current_index, True, f"Found functions {load_func_name} and {normalize_func_name} in module")
            raw_data = load_func(**load_params)

        # ------------------ 4. 执行数据归一化 ------------------
        current_index += 1
        normalized = normalize_func(raw_data)

        if normalized is None:
            raise ValueError(f"{normalize_func_name} returned None")

        update_check(
            current_index,
            True,
            f"normalized type={type(normalized).__name__}"
        )

        # ------------------ 5. 检查数据是否包含 NaN/Inf ------------------
        current_index += 1
        if not isinstance(normalized, np.ndarray):
            # 非 numpy 数组，尝试转换为数组
            try:
                normalized = np.array(normalized)
            except Exception as e:
                update_check(
                    current_index,
                    True,
                    f"Non-numpy data, finite check skipped: {type(normalized).__name__}"
                )
                # 无法转换为数组，无法继续后续检查
                return checks

        finite_ok = np.isfinite(normalized).all()
        if not finite_ok:
            raise ValueError("normalized data contains NaN or Inf values")
        update_check(
            current_index,
            True,
            f"All values are finite, shape={normalized.shape}"
        )

        # ------------------ 6. 根据归一化方法检查结果 ------------------
        if method == "reference":
            # 标准化：均值应为 0，标准差应为 1
            # 处理多维数据
            if normalized.ndim > 1:
                mean = normalized.mean(axis=0)
                std = normalized.std(axis=0)
            else:
                mean = normalized.mean()
                std = normalized.std()

            # 添加均值检查
            mean_check_name = "data:standardization_mean"
            CHECKPOINT_NAMES.append(mean_check_name)
            checks.append((mean_check_name, False, "Not executed"))

            mean_ok = np.allclose(mean, 0, atol=1e-5)
            if not mean_ok:
                raise ValueError(f"Mean not close to 0: mean={mean}")
            checks[-1] = (mean_check_name, True, f"mean={mean}")

            # 添加标准差检查
            std_check_name = "data:standardization_std"
            CHECKPOINT_NAMES.append(std_check_name)
            checks.append((std_check_name, False, "Not executed"))

            std_ok = np.allclose(std, 1, atol=1e-5)
            if not std_ok:
                raise ValueError(f"Std not close to 1: std={std}")
            checks[-1] = (std_check_name, True, f"std={std}")

        elif method == "minmax":
            # Min-Max 归一化：数据应在 [0, 1] 范围内（允许微小的数值误差）
            range_check_name = "data:minmax_range"
            CHECKPOINT_NAMES.append(range_check_name)
            checks.append((range_check_name, False, "Not executed"))

            range_ok = np.all(normalized >= -1e-6) and np.all(normalized <= 1 + 1e-6)
            if not range_ok:
                raise ValueError(f"Data outside [0, 1] range: min={normalized.min()}, max={normalized.max()}")
            checks[-1] = (
                range_check_name,
                True,
                f"range=[{normalized.min():.6f}, {normalized.max():.6f}]"
            )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_data_transformation(config, caller_globals=None, caller_locals=None) -> list(tuple):
    """
    调用数据转换函数，验证转换后的结果。

    支持两种模式:

    【模式1 - 模块级函数】
    data_utils.py 中包含独立的 load_dataset() 和 transform_data(raw_data) 函数
    适用场景：数据处理逻辑简单，无需维护状态

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "transform_func": "transform_data",   # 可选，默认 "transform_data"
        "load_params": {"data_path": "data.csv"},  # 可选，传递给 load_func 的关键字参数
        "expected_shape": [10, 5]             # 可选，期望的输出形状
    }

    【模式2 - 类方法】
    data_utils.py 中包含数据处理类，通过实例化后调用方法
    适用场景：需要维护状态、配置复杂、或需要复用同一处理流程

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "class_name": "DataProcessor",        # 必填，类名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "transform_func": "transform_data",   # 可选，默认 "transform_data"
        "init_kwargs": {                      # 可选，类初始化的关键字参数字典
            "data_path": "data.csv",
            "mode": "train",
            "batch_size": 32
        },
        "expected_shape": [10, 5]             # 可选，期望的输出形状
    }

    检查点:
        - data:transformation_module: 检查能否加载数据模块
        - data:transform_result: 检查转换结果是否为空
        - data:transform_finite: 检查转换结果是否包含 NaN/Inf
        - data:transform_shape: 检查转换结果的形状是否符合预期（如果配置了 expected_shape）
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "data:transformation_module",
        "data:transform_result",
        "data:transform_finite",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    # 动态添加 shape 检查点
    has_shape_check = "expected_shape" in config
    if has_shape_check:
        CHECKPOINT_NAMES.append("data:transform_shape")
        checks.append(("data:transform_shape", False, "Not executed due to earlier failure"))

    try:
        # ------------------ 1. 加载数据模块 ------------------
        dataset_file = config.get("dataset_file")
        if not dataset_file:
            raise ValueError("config['dataset_file'] is missing or empty")

        data_module = _load_candidate_modules(dataset_file)
        update_check(current_index, True, f"Data module loaded from {dataset_file}")

        # ------------------ 2. 获取配置参数 ------------------
        load_func_name = config.get("load_func", "load_dataset")
        transform_func_name = config.get("transform_func", "transform_data")
        class_name = config.get("class_name")

        # 获取参数配置
        load_params = config.get("load_params", {})  # 关键字参数字典
        init_kwargs = config.get("init_kwargs", {})  # 类初始化关键字参数

        # ------------------ 3. 获取可调用对象 (函数或类实例) ------------------
        if class_name:
            # 模式2: 类方法
            if not hasattr(data_module, class_name):
                raise AttributeError(f"data_module does not have class '{class_name}'")
            cls = getattr(data_module, class_name)
            if not callable(cls):
                raise AttributeError(f"'{class_name}' is not callable")

            instance = cls(**init_kwargs)

            # 获取 load_dataset 方法
            if not hasattr(instance, load_func_name) or not callable(getattr(instance, load_func_name)):
                raise AttributeError(f"Instance does not have callable '{load_func_name}'")
            load_func = getattr(instance, load_func_name)

            # 获取 transform_data 方法
            if not hasattr(instance, transform_func_name) or not callable(getattr(instance, transform_func_name)):
                raise AttributeError(f"Instance does not have callable '{transform_func_name}'")
            transform_func = getattr(instance, transform_func_name)

            update_check(current_index, True,
                         f"Instantiated {class_name} from module, using {load_func_name} and {transform_func_name}")
            # 执行 load 方法
            raw_data = load_func(**load_params)

        else:
            # 模式1: 模块级函数
            if not hasattr(data_module, load_func_name) or not callable(getattr(data_module, load_func_name)):
                raise AttributeError(f"data_module does not have callable '{load_func_name}'")
            load_func = getattr(data_module, load_func_name)

            if not hasattr(data_module, transform_func_name) or not callable(getattr(data_module, transform_func_name)):
                raise AttributeError(f"data_module does not have callable '{transform_func_name}'")
            transform_func = getattr(data_module, transform_func_name)

            update_check(current_index, True, f"Found functions {load_func_name} and {transform_func_name} in module")
            raw_data = load_func(**load_params)

        # ------------------ 4. 执行数据转换 ------------------
        current_index += 1
        transformed = transform_func(raw_data)

        if transformed is None:
            raise ValueError(f"{transform_func_name} returned None")

        update_check(
            current_index,
            True,
            f"transformed type={type(transformed).__name__}"
        )

        # ------------------ 5. 检查数据是否包含 NaN/Inf ------------------
        current_index += 1
        if isinstance(transformed, np.ndarray):
            finite_ok = np.isfinite(transformed).all()
            if not finite_ok:
                raise ValueError("transformed data contains NaN or Inf values")
            update_check(
                current_index,
                True,
                f"All values are finite, shape={transformed.shape}"
            )
        else:
            # 非 numpy 数组，尝试转换为数组检查
            try:
                arr = np.array(transformed)
                if np.isfinite(arr).all():
                    update_check(
                        current_index,
                        True,
                        f"All values are finite, shape={arr.shape}"
                    )
                else:
                    raise ValueError("transformed data contains NaN or Inf values")
            except Exception as e:
                # 无法转换为数组，跳过检查
                update_check(
                    current_index,
                    True,
                    f"Non-numpy data, finite check skipped: {type(transformed).__name__}"
                )

        # ------------------ 6. 检查形状是否符合预期 ------------------
        if has_shape_check and isinstance(transformed, np.ndarray):
            expected_shape = tuple(config["expected_shape"])
            shape_check_idx = len(CHECKPOINT_NAMES) - 1

            if transformed.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch: expected={expected_shape}, actual={transformed.shape}"
                )
            checks[shape_check_idx] = (
                "data:transform_shape",
                True,
                f"shape={transformed.shape}, expected={expected_shape}"
            )
        elif has_shape_check and not isinstance(transformed, np.ndarray):
            # 非 numpy 数组，无法检查形状
            shape_check_idx = len(CHECKPOINT_NAMES) - 1
            checks[shape_check_idx] = (
                "data:transform_shape",
                True,
                f"Non-numpy data, shape check skipped: {type(transformed).__name__}"
            )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_padding_and_mask(config: Dict, caller_globals=None, caller_locals=None) -> list(tuple):
    """
    调用候选代码的 padding 函数，检查 padding 和 mask。

    支持两种模式:

    【模式1 - 模块级函数】
    data_utils.py 中包含独立的 load_dataset() 和 pad_data(raw_data) 函数
    适用场景：数据处理逻辑简单，无需维护状态

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "pad_func": "pad_data",               # 可选，默认 "pad_data"
        "load_params": {"data_path": "data.csv"}  # 可选，传递给 load_func 的关键字参数
    }

    【模式2 - 类方法】
    data_utils.py 中包含数据处理类，通过实例化后调用方法
    适用场景：需要维护状态、配置复杂、或需要复用同一处理流程

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "class_name": "DataProcessor",        # 必填，类名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "pad_func": "pad_data",               # 可选，默认 "pad_data"
        "init_kwargs": {                      # 可选，类初始化的关键字参数字典
            "data_path": "data.csv",
            "mode": "train",
            "batch_size": 32,
            "max_length": 128
        }
    }

    检查点:
        - data:padding_module: 检查能否加载数据模块
        - data:padding_result: 检查 padding 结果是否为空
        - data:padding_mask_shape: 检查 padded 和 mask 的形状是否匹配
        - data:mask_dtype: 检查 mask 的数据类型是否为 bool
        - data:padding_finite: 检查 padded 数据是否包含 NaN/Inf
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "data:padding_module",
        "data:padding_result",
        "data:padding_mask_shape",
        "data:mask_dtype",
        "data:padding_finite",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        # ------------------ 1. 加载数据模块 ------------------
        dataset_file = config.get("dataset_file")
        if not dataset_file:
            raise ValueError("config['dataset_file'] is missing or empty")

        data_module = _load_candidate_modules(dataset_file)
        update_check(current_index, True, f"Data module loaded from {dataset_file}")

        # ------------------ 2. 获取配置参数 ------------------
        load_func_name = config.get("load_func", "load_dataset")
        pad_func_name = config.get("pad_func", "pad_data")
        class_name = config.get("class_name")

        # 获取参数配置
        load_params = config.get("load_params", {})  # 关键字参数字典
        init_kwargs = config.get("init_kwargs", {})  # 类初始化关键字参数

        # ------------------ 3. 获取可调用对象 (函数或类实例) ------------------
        if class_name:
            # 模式2: 类方法
            if not hasattr(data_module, class_name):
                raise AttributeError(f"data_module does not have class '{class_name}'")
            cls = getattr(data_module, class_name)
            if not callable(cls):
                raise AttributeError(f"'{class_name}' is not callable")

            instance = cls(**init_kwargs)

            # 获取 load_dataset 方法
            if not hasattr(instance, load_func_name) or not callable(getattr(instance, load_func_name)):
                raise AttributeError(f"Instance does not have callable '{load_func_name}'")
            load_func = getattr(instance, load_func_name)

            # 获取 pad_data 方法
            if not hasattr(instance, pad_func_name) or not callable(getattr(instance, pad_func_name)):
                raise AttributeError(f"Instance does not have callable '{pad_func_name}'")
            pad_func = getattr(instance, pad_func_name)

            update_check(current_index, True,
                         f"Instantiated {class_name} from module, using {load_func_name} and {pad_func_name}")
            # 执行 load 方法
            raw_data = load_func(**load_params)

        else:
            # 模式1: 模块级函数
            if not hasattr(data_module, load_func_name) or not callable(getattr(data_module, load_func_name)):
                raise AttributeError(f"data_module does not have callable '{load_func_name}'")
            load_func = getattr(data_module, load_func_name)

            if not hasattr(data_module, pad_func_name) or not callable(getattr(data_module, pad_func_name)):
                raise AttributeError(f"data_module does not have callable '{pad_func_name}'")
            pad_func = getattr(data_module, pad_func_name)

            update_check(current_index, True, f"Found functions {load_func_name} and {pad_func_name} in module")
            raw_data = load_func(**load_params)

        # ------------------ 4. 执行数据 padding ------------------
        current_index += 1
        result = pad_func(raw_data)

        # 处理返回结果：可能是 (padded, mask) 元组，也可能是包含两者的字典
        if isinstance(result, tuple) and len(result) == 2:
            padded, mask = result
        elif isinstance(result, dict) and "padded" in result and "mask" in result:
            padded = result["padded"]
            mask = result["mask"]
        else:
            raise ValueError(
                f"pad_data should return tuple (padded, mask) or dict with 'padded' and 'mask' keys, got {type(result)}"
            )

        if padded is None or mask is None:
            raise ValueError("pad_data returned None for padded or mask")

        update_check(
            current_index,
            True,
            f"padded={padded.shape if hasattr(padded, 'shape') else 'no shape'}, "
            f"mask={mask.shape if hasattr(mask, 'shape') else 'no shape'}"
        )

        # ------------------ 5. 检查 padded 和 mask 是否为 numpy 数组 ------------------
        if not isinstance(padded, np.ndarray):
            try:
                padded = np.array(padded)
            except Exception as e:
                raise TypeError(f"padded cannot be converted to numpy array: {e}")

        if not isinstance(mask, np.ndarray):
            try:
                mask = np.array(mask)
            except Exception as e:
                raise TypeError(f"mask cannot be converted to numpy array: {e}")

        # ------------------ 6. 检查 padded 和 mask 的形状是否匹配 ------------------
        current_index += 1
        # mask 的维度应该与 padded 的前 mask.ndim 个维度匹配
        if padded.ndim < mask.ndim:
            raise ValueError(
                f"padded.ndim ({padded.ndim}) < mask.ndim ({mask.ndim}), "
                f"padded.shape={padded.shape}, mask.shape={mask.shape}"
            )

        shape_match = padded.shape[:mask.ndim] == mask.shape
        if not shape_match:
            raise ValueError(
                f"Shape mismatch: padded.shape[:mask.ndim]={padded.shape[:mask.ndim]}, mask.shape={mask.shape}"
            )
        update_check(
            current_index,
            True,
            f"padded.shape={padded.shape}, mask.shape={mask.shape}"
        )

        # ------------------ 7. 检查 mask 的数据类型 ------------------
        current_index += 1
        # 检查是否布尔类型（支持 np.bool_ 和 Python bool）
        if mask.dtype != np.bool_:
            # 尝试转换为 bool 类型（如果值都是 0/1）
            unique_values = np.unique(mask)
            if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, False, True}):
                # 可以安全转换为 bool
                mask = mask.astype(np.bool_)
                update_check(
                    current_index,
                    True,
                    f"mask dtype was {mask.dtype}, converted to bool"
                )
            else:
                raise ValueError(f"mask dtype should be bool or convertible to bool, got {mask.dtype}")
        else:
            update_check(
                current_index,
                True,
                f"mask dtype={mask.dtype}"
            )

        # ------------------ 8. 检查 padded 数据是否包含 NaN/Inf ------------------
        current_index += 1
        if not isinstance(padded, np.ndarray):
            try:
                padded = np.array(padded)
            except Exception as e:
                update_check(
                    current_index,
                    True,
                    f"Non-numpy data, finite check skipped: {type(padded).__name__}"
                )
                # 无法检查有限值，跳过
                return checks

        finite_ok = np.isfinite(padded).all()
        if not finite_ok:
            raise ValueError("padded data contains NaN or Inf values")
        update_check(
            current_index,
            True,
            f"All values are finite, shape={padded.shape}"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_data_split(config: Dict, caller_globals=None, caller_locals=None) -> list(tuple):
    """
    调用数据划分函数，检查 train/val/test 结果。

    支持两种模式:

    【模式1 - 模块级函数】
    data_utils.py 中包含独立的 load_dataset() 和 split_dataset(raw_data) 函数
    适用场景：数据处理逻辑简单，无需维护状态

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "split_func": "split_dataset",        # 可选，默认 "split_dataset"
        "load_params": {"data_path": "data.csv"},  # 可选，传递给 load_func 的关键字参数
        "split_ratio": [0.7, 0.15, 0.15]      # 可选，用于验证划分比例
    }

    【模式2 - 类方法】
    data_utils.py 中包含数据处理类，通过实例化后调用方法
    适用场景：需要维护状态、配置复杂、或需要复用同一处理流程

    config 示例:
    {
        "dataset_file": "data_utils.py",      # 必填，数据模块文件名
        "class_name": "DataProcessor",        # 必填，类名
        "load_func": "load_dataset",          # 可选，默认 "load_dataset"
        "split_func": "split_dataset",        # 可选，默认 "split_dataset"
        "init_kwargs": {                      # 可选，类初始化的关键字参数字典
            "data_path": "data.csv",
            "mode": "train",
            "random_seed": 42
        },
        "split_ratio": [0.7, 0.15, 0.15]      # 可选，用于验证划分比例
    }

    检查点:
        - data:split_module: 检查能否加载数据模块
        - data:split_not_empty: 检查划分结果是否为空
        - data:split_ratio: 检查划分比例是否符合预期（如果配置了 split_ratio）
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "data:split_module",
        "data:split_not_empty",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    # 动态添加比例检查点
    has_ratio_check = "split_ratio" in config
    if has_ratio_check:
        CHECKPOINT_NAMES.append("data:split_ratio")
        checks.append(("data:split_ratio", False, "Not executed due to earlier failure"))

    try:
        # ------------------ 1. 加载数据模块 ------------------
        dataset_file = config.get("dataset_file")
        if not dataset_file:
            raise ValueError("config['dataset_file'] is missing or empty")

        data_module = _load_candidate_modules(dataset_file)
        update_check(current_index, True, f"Data module loaded from {dataset_file}")

        # ------------------ 2. 获取配置参数 ------------------
        load_func_name = config.get("load_func", "load_dataset")
        split_func_name = config.get("split_func", "split_dataset")
        class_name = config.get("class_name")

        # 获取参数配置
        load_params = config.get("load_params", {})  # 关键字参数字典
        init_kwargs = config.get("init_kwargs", {})  # 类初始化关键字参数

        # 验证 split_ratio 配置（如果提供）
        if has_ratio_check:
            split_ratio = config["split_ratio"]
            if not isinstance(split_ratio, (list, tuple, np.ndarray)):
                raise ValueError(f"split_ratio must be list/tuple/ndarray, got {type(split_ratio)}")
            if len(split_ratio) != 3:
                raise ValueError(f"split_ratio must have exactly 3 elements, got {len(split_ratio)}")
            if not np.allclose(sum(split_ratio), 1.0, atol=1e-6):
                raise ValueError(f"split_ratio must sum to 1.0, got {sum(split_ratio)}")

        # ------------------ 3. 获取可调用对象 (函数或类实例) ------------------
        if class_name:
            # 模式2: 类方法
            if not hasattr(data_module, class_name):
                raise AttributeError(f"data_module does not have class '{class_name}'")
            cls = getattr(data_module, class_name)
            if not callable(cls):
                raise AttributeError(f"'{class_name}' is not callable")

            instance = cls(**init_kwargs)

            # 获取 load_dataset 方法
            if not hasattr(instance, load_func_name) or not callable(getattr(instance, load_func_name)):
                raise AttributeError(f"Instance does not have callable '{load_func_name}'")
            load_func = getattr(instance, load_func_name)

            # 获取 split_dataset 方法
            if not hasattr(instance, split_func_name) or not callable(getattr(instance, split_func_name)):
                raise AttributeError(f"Instance does not have callable '{split_func_name}'")
            split_func = getattr(instance, split_func_name)

            update_check(current_index, True,
                         f"Instantiated {class_name} from module, using {load_func_name} and {split_func_name}")
            # 执行 load 方法
            raw_data = load_func(**load_params)

        else:
            # 模式1: 模块级函数
            if not hasattr(data_module, load_func_name) or not callable(getattr(data_module, load_func_name)):
                raise AttributeError(f"data_module does not have callable '{load_func_name}'")
            load_func = getattr(data_module, load_func_name)

            if not hasattr(data_module, split_func_name) or not callable(getattr(data_module, split_func_name)):
                raise AttributeError(f"data_module does not have callable '{split_func_name}'")
            split_func = getattr(data_module, split_func_name)

            update_check(current_index, True, f"Found functions {load_func_name} and {split_func_name} in module")
            raw_data = load_func(**load_params)

        # ------------------ 4. 执行数据划分 ------------------
        current_index += 1
        split_result = split_func(raw_data)

        # 处理返回结果：可能是 tuple、list 或 dict
        if isinstance(split_result, (tuple, list)):
            if len(split_result) != 3:
                raise ValueError(
                    f"split_dataset should return (train, val, test) tuple/list of length 3, got length {len(split_result)}"
                )
            train, val, test = split_result
        elif isinstance(split_result, dict):
            # 支持字典返回，例如 {'train': ..., 'val': ..., 'test': ...}
            if "train" not in split_result or "val" not in split_result or "test" not in split_result:
                raise ValueError(
                    "split_dataset returned dict but missing one of 'train', 'val', 'test' keys"
                )
            train = split_result["train"]
            val = split_result["val"]
            test = split_result["test"]
        else:
            raise ValueError(
                f"split_dataset should return (train, val, test) tuple/list or dict with keys, got {type(split_result)}"
            )

        # 检查每个划分是否非空
        if train is None or val is None or test is None:
            raise ValueError("split_dataset returned None for one or more splits")

        # 检查是否为空（支持 len 或 shape 检查）
        def get_length(data):
            if hasattr(data, "__len__"):
                return len(data)
            elif isinstance(data, np.ndarray):
                return data.shape[0] if data.ndim > 0 else 1
            else:
                return 1  # 标量或非序列类型

        train_len = get_length(train)
        val_len = get_length(val)
        test_len = get_length(test)

        if train_len == 0 or val_len == 0 or test_len == 0:
            raise ValueError(
                f"Empty split: train={train_len}, val={val_len}, test={test_len}"
            )

        update_check(
            current_index,
            True,
            f"train={train_len}, val={val_len}, test={test_len}"
        )

        # ------------------ 5. 检查划分比例 ------------------
        if has_ratio_check:
            check_idx = len(CHECKPOINT_NAMES) - 1
            total = train_len + val_len + test_len
            actual_ratio = np.array([train_len, val_len, test_len]) / total
            expected_ratio = np.asarray(config["split_ratio"])

            # 检查实际比例是否与期望比例接近（允许 2% 的误差）
            ratio_ok = np.allclose(actual_ratio, expected_ratio, atol=0.02)
            if not ratio_ok:
                raise ValueError(
                    f"Split ratio mismatch: expected={expected_ratio}, actual={actual_ratio}, "
                    f"difference={np.abs(actual_ratio - expected_ratio)}"
                )
            checks[check_idx] = (
                "data:split_ratio",
                True,
                f"expected={expected_ratio.tolist()}, actual={actual_ratio.tolist()}"
            )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


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
        if not reference_file:
            raise ValueError("config['reference']['dataset_loader'] is missing or empty")

        candidate_module = _load_candidate_modules(candidate_file)
        reference_module = _load_reference_modules(reference_file)

        update_check(
            current_index,
            True,
            f"Candidate: {candidate_file}, Reference: {reference_file}"
        )

        # ------------------ 2. 获取 loader 函数 ------------------
        candidate_loader = getattr(candidate_module, candidate_loader_name, None)
        reference_loader = getattr(reference_module, reference_loader_name, None)

        if candidate_loader is None or not callable(candidate_loader):
            raise AttributeError(
                f"Candidate does not have callable '{candidate_loader_name}' "
                f"in module {candidate_file}"
            )
        if reference_loader is None or not callable(reference_loader):
            raise AttributeError(
                f"Reference does not have callable '{reference_loader_name}' "
                f"in module {reference_file}"
            )

        # ------------------ 3. 执行处理 ------------------
        current_index += 1

        reference_output = reference_loader(reference_dataset_config)
        candidate_output = candidate_loader(candidate_dataset_config)

        update_check(
            current_index,
            True,
            f"Both candidate and reference executed successfully. "
            f"Candidate loader: {candidate_loader_name}, "
            f"Reference loader: {reference_loader_name}"
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
            "dataset_loader reference implementations are consistent"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_checkpoint_file(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    检查候选提交目录中是否存在配置指定的必需文件。

    config 示例:
    {
        "config_name": "config.yaml",
        "checkpoint_key": "checkpoint"
    }

    检查点:
        - checkpoint_file: 检查checkpoint_file是否存在
    """
    CHECKPOINT_NAMES = ["checkpoint_file.exist"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        # ------------------ 1. 获取配置 ------------------
        config_name = config.get("config_name")
        if not config_name:
            raise ValueError("config['config_name'] is missing or empty")

        model_config = _load_candidate_config(config_name)
        checkpoint = model_config.get(config.get("checkpoint_key"))
        if checkpoint is None:
            raise ValueError(f"config[{config.get('checkpoint_key')}] is missing or empty")

        checkpoint_path = Path(checkpoint).resolve()
        if checkpoint_path.exists():
            update_check(current_index, True, f"checkpoint file found")
        else:
            update_check(current_index, True, f"checkpoint file not found")

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

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
        - checkpoint_loading:config_load: 检查能否加载配置
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

        checkpoint = torch.load(
            checkpoint_path_resolved,
            map_location="cpu",
            weights_only=True,
        )

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
        - model_structure:structure: 模型结构
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

        from model_structure_checker import ModuleInfoExtractor, ModelStructureChecker
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
                train_path.read_text(encoding="utf-8"),
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

        # 执行训练
        result = train_fn(**resolved_train_args) if resolved_train_args else train_fn(runtime_config)

        update_check(
            current_index,
            True,
            f"Smoke test completed successfully: {train_fn_name}(...) with {config_name}"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_smoke_artifacts(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    验证训练产生的产物文件。

    config 示例:
    {
        "artifacts": {
            "files": ["output.log", "metrics.json"],
            "checkpoints": [
                "checkpoints/best.pth",
                "checkpoints/last.pth"
            ]
        }
    }

    检查点:
        - artifact:file_{name}_exists: 检查普通文件是否存在
    """

    def normalize_items(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [(Path(value).name, value)]
        if isinstance(value, (list, tuple)):
            return [(Path(str(item)).name, str(item)) for item in value]
        if isinstance(value, dict):
            return [(str(name), str(path)) for name, path in value.items()]
        return []

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    CHECKPOINT_NAMES = []
    checks = []
    current_index = 0
    try:
        # ================================================================
        # 0. 定义所有检查点名称（按执行顺序）
        # ================================================================
        artifacts = config.get("artifacts", {})
        if not isinstance(artifacts, dict) or not len(artifacts):
            checks.append(("artifacts", False, "config['artifacts'] must be validate"))
            return checks
        # ------------------ 2. 检查普通文件 ------------------
        file_items = normalize_items(artifacts.get("files"))
        for name, relative_path in file_items:
            CHECKPOINT_NAMES.append((f"artifacts.{name}", False, "Not executed due to earlier failure"))

        for name, relative_path in file_items:
            path = CANDIDATE_DIR / relative_path
            exist = path.exists()
            update_check(current_index, exist, f"{name} is exist" if exist else f"{name} is not exist")
            current_index += 1
    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_training(config: dict, caller_globals=None, caller_locals=None) -> List[Tuple]:
    """
    执行真实训练任务，验证完整训练流程。

    config 示例:
    {
        "run_train_file": "train.py",           # 训练模块文件
        "train_config_name": "config.yaml",     # 配置文件名称
        "train_function": "train",              # 训练入口函数名
        "pre_train_hooks": [                    # 前置钩子（可选）
            {
                "function": "build_model",
                "args": {"model_name": "resnet50"},
                "save_as": "model"
            },
            {
                "function": "build_dataloader",
                "args": {"batch_size": 32},
                "save_as": "dataloader"
            }
        ],
        "train_args": {                         # 训练函数参数
            "model": "model",
            "dataloader": "dataloader",
            "config": "config"
        }
    }

    检查点:
        - training:module_load: 检查能否加载训练模块
        - training:config_load: 检查能否加载配置
        - training:pre_hooks: 检查前置钩子是否执行成功
        - training:train_call: 检查训练调用是否成功
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "training:module_load",
        "training:config_load",
        "training:pre_hooks",
        "training:train_call",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        # ------------------ 1. 加载训练模块 ------------------
        train_file = config.get("run_train_file")
        if not train_file:
            raise ValueError("config['run_train_file'] is missing or empty")

        train_module = _load_candidate_modules(train_file)
        update_check(current_index, True, f"Module loaded from {train_file}")

        # ------------------ 2. 加载配置 ------------------
        current_index += 1
        train_config_name = config.get("train_config_name")
        if not train_config_name:
            raise ValueError("config['train_config_name'] is missing or empty")

        # 加载完整配置作为上下文
        candidate_config = _load_candidate_config(train_config_name)
        runtime_config = deepcopy(candidate_config)
        update_check(current_index, True, f"Config loaded from {train_config_name}")

        # ------------------ 3. 获取训练函数 ------------------
        train_fn_name = config.get("train_function", "train")
        train_fn = getattr(train_module, train_fn_name, None)
        if train_fn is None or not callable(train_fn):
            raise ValueError(f"No callable training entry '{train_fn_name}' found")

        # 构建上下文，用于参数解析
        context = {"config": runtime_config}
        # 将配置中的所有键加入上下文，便于引用
        if isinstance(runtime_config, dict):
            context.update(runtime_config)

        # ------------------ 4. 执行前置钩子 ------------------
        current_index += 1
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
                # 保存到上下文和 runtime_config 中，供后续使用
                context[save_as] = hook_result
                if isinstance(runtime_config, dict):
                    runtime_config[save_as] = hook_result

        update_check(current_index, True, f"All {len(pre_hooks)} pre-hooks executed")

        # ------------------ 5. 执行训练 ------------------
        current_index += 1
        train_args_config = config.get("train_args", {})
        if not train_args_config:
            # 如果没有指定 train_args，默认传入 config
            train_args_config = {}

        resolved_train_args = _resolve_args(train_args_config, context)

        # 执行训练
        result = train_fn(**resolved_train_args)

        # 训练成功完成
        update_check(
            current_index,
            True,
            f"Training completed successfully: {train_fn_name}(...) with {train_config_name}"
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


def check_loss_convergence(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    检查训练过程中记录的 loss 是否具有收敛趋势。

    config 示例:
    {
        "loss_file": "outputs/loss_history.json"  # 或 .txt 文件
    }

    loss 文件格式示例 (JSON):
    {
        "train_loss": [1.25, 0.94, 0.76, 0.63],
        "eval_loss": [1.31, 1.01, 0.82, 0.71]
    }

    或文本格式示例 (每行一条记录):
    train_loss: 0.89, eval_loss: 0.92
    train_loss: 0.76, eval_loss: 0.81
    train_loss: 0.63, eval_loss: 0.71

    检查点:
        - training:loss_file_exists: 检查 loss 文件是否存在
        - training:loss_file_valid: 检查 loss 文件是否有效
        - training:train_loss_finite: 检查 train_loss 是否有限
        - training:train_loss_decrease: 检查 train_loss 是否下降
        - training:train_loss_trend: 检查 train_loss 趋势
        - training:eval_loss_finite: 检查 eval_loss 是否有限
        - training:eval_loss_decrease: 检查 eval_loss 是否下降
        - training:eval_loss_trend: 检查 eval_loss 趋势
    """
    import json
    import numpy as np
    import traceback
    from pathlib import Path
    import re

    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "training:loss_file_exists",
        "training:loss_file_valid",
        "training:train_loss_finite",
        "training:train_loss_decrease",
        "training:train_loss_trend",
        "training:eval_loss_finite",
        "training:eval_loss_decrease",
        "training:eval_loss_trend",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    CANDIDATE_DIR = Path.cwd()  # 假设定义在外部

    try:
        # ------------------ 1. 检查文件是否存在 ------------------
        loss_file = config.get("loss_file")
        if not loss_file:
            raise ValueError("loss_file not provided in config")

        loss_path = (CANDIDATE_DIR / loss_file).resolve()
        if not loss_path.exists():
            raise FileNotFoundError(f"Loss file not found: {loss_path}")
        update_check(current_index, True, f"Loss file exists: {loss_path}")

        # ------------------ 2. 加载 loss 文件 ------------------
        current_index += 1

        # 读取文件内容
        with open(loss_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        train_loss = []
        eval_loss = []

        # 尝试作为 JSON 解析
        try:
            loss_data = json.loads(content)
            train_loss = loss_data.get("train_loss", [])
            eval_loss = loss_data.get("eval_loss", [])

            if isinstance(train_loss, list) and isinstance(eval_loss, list):
                # JSON 格式解析成功
                if len(train_loss) < 2:
                    raise ValueError(f"Insufficient train_loss records: {len(train_loss)} < 2")
                update_check(current_index, True, f"JSON format loaded, train_loss: {len(train_loss)} records")
            else:
                raise ValueError("JSON format invalid: train_loss or eval_loss is not a list")

        except (json.JSONDecodeError, ValueError, TypeError):
            # 如果不是 JSON，尝试作为文本格式解析
            lines = content.split('\n')
            train_loss_pattern = re.compile(r'train_loss\s*[:=]\s*([\d.]+)', re.IGNORECASE)
            eval_loss_pattern = re.compile(r'eval_loss\s*[:=]\s*([\d.]+)', re.IGNORECASE)

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 提取 train_loss
                train_match = train_loss_pattern.search(line)
                if train_match:
                    try:
                        train_loss.append(float(train_match.group(1)))
                    except ValueError:
                        pass

                # 提取 eval_loss
                eval_match = eval_loss_pattern.search(line)
                if eval_match:
                    try:
                        eval_loss.append(float(eval_match.group(1)))
                    except ValueError:
                        pass

            if not train_loss and not eval_loss:
                raise ValueError(f"Unable to parse loss data from file: {loss_path}")

            if len(train_loss) < 2:
                raise ValueError(f"Insufficient train_loss records: {len(train_loss)} < 2")

            update_check(current_index, True,
                         f"Text format loaded, train_loss: {len(train_loss)} records, eval_loss: {len(eval_loss)} records")

        # ------------------ 3. 检查 train_loss 是否有限 ------------------
        current_index += 1
        train_loss_array = np.asarray(train_loss, dtype=float)
        finite_ok = np.isfinite(train_loss_array).all()
        if not finite_ok:
            invalid_indices = np.where(~np.isfinite(train_loss_array))[0].tolist()
            raise ValueError(f"train_loss contains non-finite values at indices: {invalid_indices}")
        update_check(
            current_index,
            True,
            f"min={train_loss_array.min():.6f}, max={train_loss_array.max():.6f}"
        )

        # ------------------ 4. 检查 train_loss 是否下降 ------------------
        current_index += 1
        train_start = float(train_loss_array[0])
        train_end = float(train_loss_array[-1])
        if train_end >= train_start:
            raise ValueError(f"train_loss did not decrease: start={train_start:.6f}, end={train_end:.6f}")
        update_check(
            current_index,
            True,
            f"start={train_start:.6f}, end={train_end:.6f}, decrease={train_end - train_start:.6f}"
        )

        # ------------------ 5. 检查 train_loss 整体趋势 ------------------
        current_index += 1
        x = np.arange(len(train_loss_array))
        slope = np.polyfit(x, train_loss_array, deg=1)[0]
        if slope >= 0:
            raise ValueError(f"train_loss trend is not decreasing: slope={slope:.6e}")
        update_check(current_index, True, f"slope={slope:.6e}")

        # ------------------ 6. 检查 eval_loss ------------------
        if len(eval_loss) >= 2:
            # eval_loss 有限
            current_index += 1
            eval_loss_array = np.asarray(eval_loss, dtype=float)
            eval_finite_ok = np.isfinite(eval_loss_array).all()
            if not eval_finite_ok:
                invalid_indices = np.where(~np.isfinite(eval_loss_array))[0].tolist()
                raise ValueError(f"eval_loss contains non-finite values at indices: {invalid_indices}")
            update_check(
                current_index,
                True,
                f"min={eval_loss_array.min():.6f}, max={eval_loss_array.max():.6f}"
            )

            # eval_loss 下降
            current_index += 1
            eval_start = float(eval_loss_array[0])
            eval_end = float(eval_loss_array[-1])
            if eval_end >= eval_start:
                raise ValueError(f"eval_loss did not decrease: start={eval_start:.6f}, end={eval_end:.6f}")
            update_check(
                current_index,
                True,
                f"start={eval_start:.6f}, end={eval_end:.6f}, decrease={eval_end - eval_start:.6f}"
            )

            # eval_loss 趋势
            current_index += 1
            eval_slope = np.polyfit(np.arange(len(eval_loss_array)), eval_loss_array, deg=1)[0]
            if eval_slope >= 0:
                raise ValueError(f"eval_loss trend is not decreasing: slope={eval_slope:.6e}")
            update_check(current_index, True, f"slope={eval_slope:.6e}")
        else:
            # eval_loss 数据不足，跳过检查
            # 注意：索引 5, 6, 7 对应 eval_loss 的三个检查
            update_check(5, False, f"Insufficient eval_loss records: {len(eval_loss)} < 2, skipping eval_loss checks")
            update_check(6, False, "eval_loss decrease skipped")
            update_check(7, False, "eval_loss trend skipped")

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

    except Exception as e:
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
            return checks

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

    return checks



def check_infer_metrics(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    动态执行推理并计算指定指标，与 baseline 对比验证模型性能。

    config 示例:
    {
        "run_infer_file": "inference.py",
        "infer_config_name": "config.yaml",
        "infer_function": "run_inference",
        "pre_infer_hooks": [
            {"function": "load_model", "args": {"device": "cuda"}, "save_as":"model"},
            {"function": "preprocess", "args": {"normalize": True}}
        ],
        "infer_args": {"model": "${model}", "data": "data/test.json"},
        "ground_truth_path": "data/ground_truth.json",
        "metric": {
            "name": "rmse",  # 支持: rmse, mae, mse, mape, r2, mae_percent, huber
            "comparison": "less_than",  # 或 "greater_than"
            "baseline": 0.1234,
            "tolerance": 1e-5
        }
    }

    检查点:
        - inference_metrics:ground_truth_load: 检查能否加载 ground truth
        - inference_metrics:metric_compute: 检查能否计算指标
        - inference_metrics:metric_validation: 检查指标是否满足条件
    """
    # ================================================================
    # 0. 定义所有检查点名称（按执行顺序）
    # ================================================================
    CHECKPOINT_NAMES = [
        "inference_metrics:ground_truth_load",
        "inference_metrics:metric_compute",
        "inference_metrics:metric_validation",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    def to_cpu_tensor(data: Any) -> torch.Tensor:
        """
        将数据转换为 CPU tensor
        """
        try:
            if torch.is_tensor(data):
                return data.cpu()
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            if isinstance(data, (list, tuple)):
                return torch.tensor(data)
            return torch.tensor(data)
        except Exception:
            raise ValueError(f"Cannot convert {type(data)} to torch.Tensor")

    def run_inference_pipeline(module, config_section: dict, context: dict) -> Any:
        """
        执行推理流程（包括 pre_infer_hooks）
        假设模块加载、配置加载等前置步骤都已成功
        """
        pre_hooks = config_section.get("pre_infer_hooks", [])
        context = _execute_hooks(module, pre_hooks, context)

        infer_func_name = config_section.get("infer_function", "inference")
        infer_func = getattr(module, infer_func_name, None)
        if infer_func is None or not callable(infer_func):
            raise ValueError(f"No callable inference entry '{infer_func_name}' found")

        infer_args_config = config_section.get("infer_args", {})
        resolved_infer_args = _resolve_args(infer_args_config, context)
        result = infer_func(**resolved_infer_args)

        return result

    def parse_metric_config(config: dict) -> dict:
        """解析指标配置"""
        if "metric" not in config:
            raise ValueError(
                "Missing 'metric' configuration. Please provide metric dict with "
                "'name', 'baseline', 'comparison', and 'tolerance' fields."
            )

        metric_config = config["metric"]
        required_fields = ["name", "baseline", "comparison"]
        for field in required_fields:
            if field not in metric_config:
                raise ValueError(f"metric config missing required field: {field}")

        # 验证 comparison 值
        if metric_config["comparison"] not in ["less_than", "greater_than"]:
            raise ValueError(
                f"Invalid comparison '{metric_config['comparison']}'. "
                "Must be 'less_than' or 'greater_than'"
            )

        return {
            "name": metric_config["name"],
            "baseline": metric_config["baseline"],
            "comparison": metric_config["comparison"],
            "tolerance": metric_config.get("tolerance", 1e-5),  # 可选，默认 1e-5
        }

    current_index = 0
    try:
        # ================================================================
        # 1. 加载推理模块和配置（默认成功，不记录检查点）
        # ================================================================
        infer_file = config.get("run_infer_file")
        if not infer_file:
            raise ValueError("config['run_infer_file'] is required")

        infer_module = _load_candidate_modules(infer_file)

        config_name = config.get("infer_config_name")
        if not config_name:
            raise ValueError("config['infer_config_name'] is required")

        candidate_config = _load_candidate_config(config_name)

        # ================================================================
        # 2. 执行推理（包括 pre_infer_hooks，默认成功）
        # ================================================================
        result = run_inference_pipeline(infer_module, config, candidate_config.copy())

        # ================================================================
        # 3. 检查点: ground_truth_load - 加载 ground truth
        # ================================================================
        gt_path = config.get("ground_truth_path")
        if not gt_path:
            raise ValueError("Missing 'ground_truth_path' in config")

        gt_path = Path(gt_path)
        if not gt_path.is_absolute():
            gt_path = CANDIDATE_DIR / gt_path

        ext = os.path.splitext(gt_path)[1].lstrip('.').lower()
        data_loader = DATA_LOADER_REGISTRY.get(ext)
        if data_loader is None:
            raise ValueError(f"gt_path not support loader")
        ground_truth = data_loader(gt_path)
        update_check(
            current_index,
            True,
            f"Ground truth loaded from {gt_path}"
        )
        current_index += 1

        # ================================================================
        # 4. 转换为 CPU tensor 进行比较
        # ================================================================
        pred_tensor = to_cpu_tensor(result)
        gt_tensor = to_cpu_tensor(ground_truth)

        # 确保形状一致
        if pred_tensor.shape != gt_tensor.shape:
            pred_tensor = pred_tensor.flatten()
            gt_tensor = gt_tensor.flatten()

        # 确保数据类型一致
        if pred_tensor.dtype != gt_tensor.dtype:
            pred_tensor = pred_tensor.float()
            gt_tensor = gt_tensor.float()

        # ================================================================
        # 5. 解析指标配置并计算指标
        # ================================================================
        metric_config = parse_metric_config(config)
        metric_name = metric_config["name"]
        baseline = metric_config["baseline"]
        comparison = metric_config["comparison"]
        tolerance = metric_config["tolerance"]

        # 获取指标计算函数并计算
        metric_func = METRIC_REGISTRY.get(metric_name)
        if metric_func is None:
            raise ValueError(
                f"Unsupported metric '{metric_name}'. "
            )
        computed_value = metric_func(pred_tensor, gt_tensor)

        update_check(
            current_index,
            True,
            f"Computed {metric_name}: {computed_value:.6f}"
        )
        current_index += 1

        # ================================================================
        # 6. 检查点: metric_validation - 与 Baseline 对比
        # ================================================================
        diff = abs(computed_value - baseline)

        # 判断是否通过
        if comparison == "less_than":
            # 指标越小越好（如 RMSE, MAE, MSE, MAPE）
            passed = (computed_value - baseline) < tolerance
            condition_desc = f"{metric_name} < {baseline} + {tolerance}"
        elif comparison == "greater_than":
            # 指标越大越好（如 R², 准确率）
            passed = (baseline - computed_value) < tolerance
            condition_desc = f"{metric_name} > {baseline} - {tolerance}"
        else:
            raise ValueError(f"Unsupported comparison type: {comparison}. Use 'less_than' or 'greater_than'")

        # 构建详细的验证信息
        if passed:
            msg = (
                f"{metric_name} validation passed: {computed_value:.6f} vs baseline {baseline:.6f} "
                f"(condition: {condition_desc}), diff = {diff:.6f}"
            )
        else:
            msg = (
                f"{metric_name} validation failed: {computed_value:.6f} vs baseline {baseline:.6f} "
                f"(condition: {condition_desc}), diff = {diff:.6f} >= {tolerance}"
            )

        update_check(current_index, passed, msg)

    except Exception as e:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks


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
        - evaluate_oracle:value_match: 检查值是否完全相同
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

    def to_cpu_tensor(data: Any) -> torch.Tensor:
        """
        将数据转换为 CPU tensor
        """
        try:
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

    def run_inference_pipeline(module, pipeline_config: dict, context: dict) -> Any:
        """
        执行推理流程（包括 pre_infer_hooks）
        """
        # 执行前置钩子
        pre_hooks = pipeline_config.get("pre_infer_hooks", [])
        context = _execute_hooks(module, pre_hooks, context)

        # 执行推理
        infer_func_name = pipeline_config.get("infer_function", "inference")
        infer_func = getattr(module, infer_func_name, None)
        if infer_func is None or not callable(infer_func):
            raise ValueError(f"No callable inference entry '{infer_func_name}' found")

        infer_args_config = pipeline_config.get("infer_args", {})
        resolved_infer_args = _resolve_args(infer_args_config, context)
        result = infer_func(**resolved_infer_args)

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
                config_data = _load_reference_config(infer_config_name)

        context = {"config": config_data}
        context.update(config_data)

        # 执行推理
        result = run_inference_pipeline(module, impl_config, context)

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

        # 检查值（直接在 tensor 上比较）
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
            "Results match exactly: shape and values are identical" if value_match else diff_info
        )

    except Exception:
        error_detail = traceback.format_exc()
        update_check(current_index, False, error_detail)

    return checks
