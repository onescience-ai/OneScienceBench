import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch.nn as nn


# ============================================================================
# 1. 模块信息提取层
# ============================================================================

class ModuleType(Enum):
    """模块类型枚举"""
    LINEAR = "linear"
    CONV = "conv"
    CONVTRANSPOSE = "convtranspose"
    LAYERNORM = "layernorm"
    BATCHNORM = "batchnorm"
    DROPOUT = "dropout"
    ACTIVATION = "activation"
    SEQUENTIAL = "sequential"
    MODULELIST = "modulelist"
    MODULEDICT = "moduledict"
    CUSTOM = "custom"
    IDENTITY = "identity"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    PARAMETER = "parameter"


@dataclass
class ModuleInfo:
    """模块结构信息数据类"""
    name: str
    type: ModuleType
    class_name: str
    module_path: str
    children: List['ModuleInfo'] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    # 形状信息
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    normalized_shape: Optional[tuple] = None
    kernel_size: Optional[tuple] = None
    stride: Optional[tuple] = None
    padding: Optional[tuple] = None

    # 统计信息
    param_count: int = 0
    is_leaf: bool = False
    # Synthetic nodes represent a missing structural selector, not a missing
    # path.  Paths are retained only as diagnostic metadata.
    exists: bool = True
    # The registered child name is useful for selecting a module without
    # making the rule depend on a positional/path representation.
    registered_name: str = ""

    def to_dict(self) -> dict:
        """转为字典便于序列化"""
        return {
            "name": self.name,
            "type": self.type.value,
            "class_name": self.class_name,
            "module_path": self.module_path,
            "registered_name": self.registered_name,
            "params": self.params,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "normalized_shape": self.normalized_shape,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "children": [c.to_dict() for c in self.children],
            "param_count": self.param_count,
            "is_leaf": self.is_leaf,
            "exists": self.exists
        }


class ModuleInfoExtractor:
    """从nn.Module实例中提取结构信息"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.module_info = None

    def extract(self) -> ModuleInfo:
        """提取完整的结构信息树"""
        self.module_info = self._extract_module(
            self.model,
            module_path="",
            registered_name="",
        )
        return self.module_info

    def _extract_module(self, module: nn.Module, module_path: str,
                        registered_name: str = "") -> ModuleInfo:
        """递归提取单个模块的信息"""

        # 获取模块类型和参数
        module_type, params = self._classify_module(module)

        # 创建信息节点
        info = ModuleInfo(
            name=module.__class__.__name__,
            type=module_type,
            class_name=module.__class__.__name__,
            module_path=module_path,
            registered_name=registered_name,
            params=params,
            param_count=sum(p.numel() for p in module.parameters())
        )

        # 提取形状信息
        self._extract_shape_info(module, info)

        # 递归提取子模块
        for name, child in module.named_children():
            child_path = f"{module_path}.{name}" if module_path else name
            child_info = self._extract_module(child, child_path, name)
            info.children.append(child_info)

        # 判断是否为叶子节点
        info.is_leaf = len(info.children) == 0

        return info

    def _classify_module(self, module: nn.Module) -> Tuple[ModuleType, dict]:
        """分类模块并提取参数"""
        params = {}

        if isinstance(module, nn.Linear):
            params = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None
            }
            return ModuleType.LINEAR, params

        elif isinstance(module, nn.LayerNorm):
            params = {
                "normalized_shape": module.normalized_shape,
                "eps": module.eps,
                "elementwise_affine": module.elementwise_affine
            }
            return ModuleType.LAYERNORM, params

        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            params = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "bias": module.bias is not None,
            }
            return ModuleType.CONV, params

        elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d,
                                 nn.ConvTranspose3d)):
            params = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "bias": module.bias is not None,
            }
            return ModuleType.CONVTRANSPOSE, params

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            params = {
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum,
                "affine": module.affine
            }
            return ModuleType.BATCHNORM, params

        elif isinstance(module, nn.BatchNorm2d):
            params = {
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum,
                "affine": module.affine
            }
            return ModuleType.BATCHNORM, params

        elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                                 nn.Dropout3d, nn.AlphaDropout)):
            params = {"p": module.p}
            return ModuleType.DROPOUT, params

        elif isinstance(module, nn.Sequential):
            return ModuleType.SEQUENTIAL, {}

        elif isinstance(module, nn.ModuleList):
            return ModuleType.MODULELIST, {"length": len(module)}

        elif isinstance(module, nn.ModuleDict):
            return ModuleType.MODULEDICT, {"keys": list(module.keys())}

        elif isinstance(module, nn.Embedding):
            return ModuleType.EMBEDDING, {
                "num_embeddings": module.num_embeddings,
                "embedding_dim": module.embedding_dim,
                "padding_idx": module.padding_idx,
            }

        elif isinstance(module, (nn.GroupNorm, nn.InstanceNorm1d,
                                 nn.InstanceNorm2d, nn.InstanceNorm3d)):
            return ModuleType.NORMALIZATION, {
                "num_channels": getattr(module, "num_channels", None),
                "num_groups": getattr(module, "num_groups", None),
            }

        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU,
                                 nn.SELU, nn.Tanh, nn.Sigmoid)):
            params = {"inplace": getattr(module, "inplace", False)}
            return ModuleType.ACTIVATION, params

        elif isinstance(module, nn.Identity):
            return ModuleType.IDENTITY, {}

        else:
            # 检查是否是激活函数（通过名称判断）
            class_name = module.__class__.__name__.lower()
            if any(act in class_name for act in ['relu', 'gelu', 'tanh', 'sigmoid', 'silu']):
                return ModuleType.ACTIVATION, params
            return ModuleType.CUSTOM, params

    def _extract_shape_info(self, module: nn.Module, info: ModuleInfo):
        """提取形状相关信息"""
        if isinstance(module, nn.Linear):
            info.in_features = module.in_features
            info.out_features = module.out_features
        elif isinstance(module, nn.LayerNorm):
            info.normalized_shape = tuple(module.normalized_shape)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                 nn.ConvTranspose1d, nn.ConvTranspose2d,
                                 nn.ConvTranspose3d)):
            info.in_features = module.in_channels
            info.out_features = module.out_channels
            info.kernel_size = module.kernel_size
            info.stride = module.stride
            info.padding = module.padding


# ============================================================================
# 2. 检查规则系统
# ============================================================================

class Severity(Enum):
    """检查结果严重程度"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class CheckRule:
    """检查规则定义"""
    id: str
    name: str
    description: str
    severity: Severity
    check_func: Callable[[ModuleInfo, dict], Tuple[bool, str]]
    params: dict = field(default_factory=dict)
    rule_type: str = ""


class CheckResult:
    """检查结果"""

    def __init__(self, rule_id: str, passed: bool, message: str,
                 severity: Severity, suggestion: str = ""):
        self.rule_id = rule_id
        self.passed = passed
        self.message = message
        self.severity = severity
        self.suggestion = suggestion

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity.value,
            "suggestion": self.suggestion
        }


class BuiltinRules:
    """内置通用检查规则"""

    @staticmethod
    def _matches_expected(info: ModuleInfo, expected: Any) -> bool:
        """Match a class/type pattern without relying on a module path."""
        if expected is None or expected == "*":
            return True
        if isinstance(expected, ModuleType):
            return info.type == expected
        if isinstance(expected, str):
            expected_lower = expected.lower()
            return (info.class_name.lower() == expected_lower or
                    info.type.value.lower() == expected_lower)
        if not isinstance(expected, dict):
            return False

        class_name = expected.get("class_name", expected.get("class"))
        if class_name is not None and info.class_name.lower() != str(class_name).lower():
            return False
        module_type = expected.get("module_type", expected.get("type"))
        if module_type is not None:
            if isinstance(module_type, ModuleType):
                if info.type != module_type:
                    return False
            elif str(module_type).lower() not in {
                    info.type.value.lower(), info.class_name.lower()}:
                return False
        name = expected.get("name", expected.get("registered_name"))
        if name is not None and str(name).lower() not in {
                info.registered_name.lower(), info.class_name.lower()}:
            return False
        for key, actual in info.params.items():
            if key in expected:
                wanted = expected[key]
                if isinstance(actual, tuple):
                    actual = list(actual)
                if isinstance(wanted, tuple):
                    wanted = list(wanted)
                if actual != wanted:
                    return False
        return True

    @staticmethod
    def _format_expected(expected: Any) -> str:
        if isinstance(expected, ModuleType):
            return expected.value
        if isinstance(expected, dict):
            return str(expected)
        return str(expected)

    @staticmethod
    def check_unsupported(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """Return a failure for a rule type this checker does not implement."""
        return False, params.get("error", "Unsupported structure rule")

    @staticmethod
    def check_module_type(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """Check a module's class/type, usable at any tree node."""
        expected = params.get("expected", params.get("module_type", params.get("type")))
        if expected is None:
            return False, "module_type requires an expected/type value"
        if not BuiltinRules._matches_expected(info, expected):
            return False, (f"Expected {BuiltinRules._format_expected(expected)}, "
                           f"got {info.class_name} ({info.type.value})")
        return True, f"Module type {info.class_name} matches"

    @staticmethod
    def check_linear_shape(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """检查Linear层的in_features/out_features"""
        if not info.exists:
            return False, f"Target module path does not exist: {info.module_path!r}"
        if info.type != ModuleType.LINEAR:
            return False, f"Expected a Linear layer, got {info.class_name}"

        expected_in = params.get("in_features")
        expected_out = params.get("out_features")

        failures = []
        if expected_in is not None and info.in_features != expected_in:
            failures.append(f"in_features: expected {expected_in}, got {info.in_features}")
        if expected_out is not None and info.out_features != expected_out:
            failures.append(f"out_features: expected {expected_out}, got {info.out_features}")

        expected_bias = params.get("bias")
        if expected_bias is not None:
            actual_bias = bool(info.params.get("bias"))
            if actual_bias != bool(expected_bias):
                failures.append(f"bias: expected {bool(expected_bias)}, got {actual_bias}")

        if failures:
            return False, "; ".join(failures)
        return True, f"Linear({info.in_features}->{info.out_features}) matches"

    @staticmethod
    def check_layer_norm_shape(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """检查LayerNorm的normalized_shape"""
        if not info.exists:
            return False, f"Target module path does not exist: {info.module_path!r}"
        if info.type != ModuleType.LAYERNORM:
            return False, f"Expected a LayerNorm, got {info.class_name}"

        expected_shape = params.get("normalized_shape")
        if expected_shape is None:
            return True, "No expected shape specified"

        actual_shape = tuple(info.normalized_shape) if info.normalized_shape else None
        if actual_shape != tuple(expected_shape):
            return False, f"normalized_shape: expected {expected_shape}, got {actual_shape}"

        return True, f"LayerNorm{actual_shape} matches"

    @staticmethod
    def check_modulelist_length(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """检查ModuleList的长度"""
        if not info.exists:
            return False, f"Target module path does not exist: {info.module_path!r}"
        if info.type != ModuleType.MODULELIST:
            return False, f"Expected a ModuleList, got {info.class_name}"

        expected_length = params.get("length")
        if expected_length is None:
            return True, "No expected length specified"

        ignored = BuiltinRules._ignored_types(params)
        actual_length = len([c for c in info.children if c.type.value not in ignored and
                             c.class_name.lower() not in ignored])
        if actual_length != expected_length:
            return False, f"length: expected {expected_length}, got {actual_length}"

        return True, f"ModuleList length={actual_length} matches"

    @staticmethod
    def check_children_count(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """检查子模块数量"""
        if not info.exists:
            return False, f"Target module path does not exist: {info.module_path!r}"
        expected_count = params.get("count")
        if expected_count is None:
            return True, "No expected count specified"

        ignored = BuiltinRules._ignored_types(params)
        actual_count = len([c for c in info.children if c.type.value not in ignored and
                            c.class_name.lower() not in ignored])
        if actual_count != expected_count:
            return False, f"children count: expected {expected_count}, got {actual_count}"

        return True, f"Children count={actual_count} matches"

    @staticmethod
    def check_block_pattern(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """检查Block是否按预期模式排列"""
        if not info.exists:
            return False, f"Target module path does not exist: {info.module_path!r}"
        pattern = params.get("pattern", [])
        if not pattern:
            return True, "No pattern specified"

        ignored = BuiltinRules._ignored_types(params, default={"dropout"})
        children = [c for c in info.children
                    if c.type.value not in ignored and c.class_name.lower() not in ignored]

        # A pattern describes the complete ordered list of non-Dropout
        # children.  Use ``exact: false`` only for a deliberately prefix-only
        # contract; otherwise extra modules must be reported as a mismatch.
        if len(children) < len(pattern):
            return False, f"Expected at least {len(pattern)} non-Dropout children, got {len(children)}"
        if params.get("exact", True) and len(children) != len(pattern):
            return False, f"Expected exactly {len(pattern)} non-Dropout children, got {len(children)}"

        for i, expected in enumerate(pattern):
            if i >= len(children):
                return False, f"Missing child at position {i}"

            # 支持类型名称或类型枚举
            if not BuiltinRules._matches_expected(children[i], expected):
                return False, (f"At position {i}: expected "
                               f"{BuiltinRules._format_expected(expected)}, "
                               f"got {children[i].class_name}")

        return True, f"Pattern matches: {[c.class_name for c in children[:len(pattern)]]}"

    @staticmethod
    def check_sequential_mlp(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """检查Sequential是否为MLP结构（Linear -> Activation -> Linear）"""
        if not info.exists:
            return False, f"Target module path does not exist: {info.module_path!r}"
        if info.type != ModuleType.SEQUENTIAL:
            return False, f"Expected a Sequential, got {info.class_name}"

        # 获取叶子节点（排除Dropout）
        leaves = []
        for child in info.children:
            if child.type != ModuleType.DROPOUT:
                leaves.append(child)

        expected_layers = params.get("layers", 3)
        expected_in = params.get("in_features")
        expected_hidden = params.get("hidden_features")
        expected_out = params.get("out_features")

        if len(leaves) != expected_layers:
            return False, f"MLP needs exactly {expected_layers} non-Dropout layers, got {len(leaves)}"
        if len(leaves) < 3:
            return False, "MLP structure requires Linear -> Activation -> Linear"

        failures = []

        # 检查第一层 Linear
        if leaves[0].type != ModuleType.LINEAR:
            failures.append(f"layer1: expected Linear, got {leaves[0].type.value}")
        elif expected_in is not None and leaves[0].in_features != expected_in:
            failures.append(f"layer1.in_features: expected {expected_in}, got {leaves[0].in_features}")

        # 检查第二层 Activation
        if leaves[1].type != ModuleType.ACTIVATION:
            failures.append(f"layer2: expected Activation, got {leaves[1].type.value}")

        # 检查第三层 Linear
        if len(leaves) >= 3:
            if leaves[2].type != ModuleType.LINEAR:
                failures.append(f"layer3: expected Linear, got {leaves[2].type.value}")
            elif expected_out is not None and leaves[2].out_features != expected_out:
                failures.append(f"layer3.out_features: expected {expected_out}, got {leaves[2].out_features}")

            # 检查hidden维度
            if expected_hidden is not None and leaves[0].out_features != expected_hidden:
                failures.append(f"hidden_dim: expected {expected_hidden}, got {leaves[0].out_features}")

        if failures:
            return False, "; ".join(failures)
        return True, f"MLP structure matches: {leaves[0].in_features}->{leaves[0].out_features}->{leaves[-1].out_features}"

    @staticmethod
    def _ignored_types(params: dict, default: Optional[set] = None) -> set:
        values = params.get("ignore_types", default or set())
        if isinstance(values, str):
            values = [values]
        return {str(value.value if isinstance(value, ModuleType) else value).lower()
                for value in values}

    @staticmethod
    def check_conv_shape(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """Check convolution channels and optional kernel/stride/padding."""
        if info.type not in {ModuleType.CONV, ModuleType.CONVTRANSPOSE}:
            return False, f"Expected a convolution, got {info.class_name}"
        failures = []
        for key, actual in (("in_channels", info.in_features),
                            ("out_channels", info.out_features),
                            ("kernel_size", info.kernel_size),
                            ("stride", info.stride),
                            ("padding", info.padding)):
            expected = params.get(key)
            if expected is None:
                continue
            actual_cmp = list(actual) if isinstance(actual, tuple) else actual
            expected_cmp = list(expected) if isinstance(expected, tuple) else expected
            if actual_cmp != expected_cmp:
                failures.append(f"{key}: expected {expected}, got {actual}")
        return (False, "; ".join(failures)) if failures else (True, "Convolution shape matches")

    @staticmethod
    def check_parameter_count(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        expected = params.get("count", params.get("param_count"))
        if expected is None:
            return False, "parameter_count requires count/param_count"
        if info.param_count != expected:
            return False, f"parameter count: expected {expected}, got {info.param_count}"
        return True, f"Parameter count={info.param_count} matches"

    @staticmethod
    def check_module_count(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """Count is evaluated by the checker over selector matches."""
        return True, "Module count selector evaluated"

    @staticmethod
    def check_embedding_shape(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        if info.type != ModuleType.EMBEDDING:
            return False, f"Expected an Embedding layer, got {info.class_name}"
        failures = []
        for key in ("num_embeddings", "embedding_dim", "padding_idx"):
            expected = params.get(key)
            if expected is not None and info.params.get(key) != expected:
                failures.append(f"{key}: expected {expected}, got {info.params.get(key)}")
        return (False, "; ".join(failures)) if failures else (True, "Embedding shape matches")

    @staticmethod
    def check_moduledict_keys(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        if info.type != ModuleType.MODULEDICT:
            return False, f"Expected a ModuleDict, got {info.class_name}"
        expected = params.get("keys")
        if expected is None:
            return False, "moduledict_keys requires keys"
        actual = list(info.params.get("keys", []))
        if params.get("exact", True):
            passed = actual == list(expected)
        else:
            passed = all(key in actual for key in expected)
        if not passed:
            return False, f"keys: expected {list(expected)}, got {actual}"
        return True, f"ModuleDict keys={actual} match"

    @staticmethod
    def check_normalization_shape(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        if info.type == ModuleType.LAYERNORM:
            return BuiltinRules.check_layer_norm_shape(info, params)
        if info.type != ModuleType.NORMALIZATION:
            return False, f"Expected a normalization layer, got {info.class_name}"
        failures = []
        for key in ("num_channels", "num_groups"):
            expected = params.get(key)
            if expected is not None and info.params.get(key) != expected:
                failures.append(f"{key}: expected {expected}, got {info.params.get(key)}")
        return (False, "; ".join(failures)) if failures else (True, "Normalization shape matches")

    @staticmethod
    def check_structure(info: ModuleInfo, params: dict) -> Tuple[bool, str]:
        """Recursively compare a path-free structural contract."""
        expected = params.get("structure", params.get("expected"))
        if expected is None:
            return False, "structure requires a structure/expected contract"

        def compare(node: ModuleInfo, spec: Any, label: str) -> List[str]:
            failures = []
            if isinstance(spec, dict):
                if not BuiltinRules._matches_expected(node, spec):
                    failures.append(f"{label}: expected {spec}, got {node.class_name}")
                for attr in ("in_features", "out_features", "normalized_shape",
                             "kernel_size", "stride", "padding", "param_count"):
                    if attr in spec:
                        expected_value = spec[attr]
                        actual_value = getattr(node, attr)
                        if attr == "normalized_shape" and actual_value is not None:
                            actual_value = list(actual_value)
                        if attr in {"kernel_size", "stride", "padding"} and actual_value is not None:
                            actual_value = list(actual_value)
                        if actual_value != expected_value:
                            failures.append(f"{label}.{attr}: expected {expected_value}, got {actual_value}")
                for key in set(spec).intersection(node.params):
                    if key in {"class_name", "module_type", "children", "name",
                               "registered_name"}:
                        continue
                    expected_value = spec[key]
                    actual_value = node.params[key]
                    if isinstance(actual_value, tuple):
                        actual_value = list(actual_value)
                    if isinstance(expected_value, tuple):
                        expected_value = list(expected_value)
                    if actual_value != expected_value:
                        failures.append(f"{label}.{key}: expected {expected_value}, got {actual_value}")
                expected_children = spec.get("children")
                if expected_children is not None:
                    ignored = BuiltinRules._ignored_types(spec, default=set())
                    actual_children = [child for child in node.children
                                       if child.type.value not in ignored and
                                       child.class_name.lower() not in ignored]
                    exact = spec.get("exact", True)
                    if exact and len(actual_children) != len(expected_children):
                        failures.append(f"{label}.children: expected {len(expected_children)}, got {len(actual_children)}")
                    elif len(actual_children) < len(expected_children):
                        failures.append(f"{label}.children: expected at least {len(expected_children)}, got {len(actual_children)}")
                    for index, child_spec in enumerate(expected_children):
                        if index < len(actual_children):
                            failures.extend(compare(actual_children[index], child_spec,
                                                    f"{label}.children[{index}]"))
            elif not BuiltinRules._matches_expected(node, spec):
                failures.append(f"{label}: expected {BuiltinRules._format_expected(spec)}, got {node.class_name}")
            return failures

        failures = compare(info, expected, "root")
        return (False, "; ".join(failures)) if failures else (True, "Structure matches")


# ============================================================================
# 3. 检查引擎
# ============================================================================

class ModelStructureChecker:
    """模型结构检查引擎"""

    # 规则名称到函数的映射
    RULE_FUNC_MAP = {
        "module_type": BuiltinRules.check_module_type,
        "type": BuiltinRules.check_module_type,
        "linear_shape": BuiltinRules.check_linear_shape,
        "conv_shape": BuiltinRules.check_conv_shape,
        "embedding_shape": BuiltinRules.check_embedding_shape,
        "normalization_shape": BuiltinRules.check_normalization_shape,
        "moduledict_keys": BuiltinRules.check_moduledict_keys,
        "parameter_count": BuiltinRules.check_parameter_count,
        "module_count": BuiltinRules.check_module_count,
        "layer_norm_shape": BuiltinRules.check_layer_norm_shape,
        "modulelist_length": BuiltinRules.check_modulelist_length,
        "children_count": BuiltinRules.check_children_count,
        "block_pattern": BuiltinRules.check_block_pattern,
        "sequential_mlp": BuiltinRules.check_sequential_mlp,
        "structure": BuiltinRules.check_structure,
    }

    def __init__(self):
        self.rules: List[CheckRule] = []

    def add_rule(self, rule: CheckRule):
        """添加检查规则"""
        self.rules.append(rule)

    def add_rules_from_rules(self, rules: Union[List[Dict], Dict, str, Path]):
        """从配置添加规则

        Args:
            rules: 字典配置或YAML文件路径
        """
        if isinstance(rules, (str, Path)):
            config_path = Path(rules)
            with config_path.open("r", encoding="utf-8") as handle:
                rules = yaml.safe_load(handle)

        if isinstance(rules, dict):
            # Accept both a bare rule mapping and the common {"rules": [...]} form.
            rules = rules.get("rules", [rules])
        if not isinstance(rules, list):
            raise TypeError(f"rules must be a list, mapping, or config path; got {type(rules).__name__}")
        if not rules:
            raise ValueError("At least one structure rule is required")

        for rule_config in rules:
            if not isinstance(rule_config, dict):
                raise TypeError(f"Each structure rule must be a mapping; got {type(rule_config).__name__}")
            rule = self._parse_rule_config(rule_config)
            self.rules.append(rule)

    def _parse_rule_config(self, config: dict) -> Optional[CheckRule]:
        """解析规则配置"""
        rule_type = config.get("type")
        if rule_type not in self.RULE_FUNC_MAP:
            message = f"Unsupported structure rule type: {rule_type!r}"
            print(f"Warning: {message}")
            unsupported_params = dict(config.get("params") or {})
            # An unsupported rule is reported once, at its configured target
            # (or at the root), rather than once for every module in a model.
            unsupported_params.setdefault("target_path", "")
            unsupported_params["error"] = message
            return CheckRule(
                id=config.get("id", str(rule_type)),
                name=config.get("name", str(rule_type)),
                description=config.get("description", ""),
                severity=Severity.ERROR,
                check_func=BuiltinRules.check_unsupported,
                params=unsupported_params,
                rule_type=str(rule_type),
            )

        severity_str = config.get("severity", "warning")
        try:
            severity = Severity(severity_str)
        except ValueError:
            severity = Severity.WARNING

        return CheckRule(
            id=config.get("id", rule_type),
            name=config.get("name", rule_type),
            description=config.get("description", ""),
            severity=severity,
            check_func=self.RULE_FUNC_MAP[rule_type],
            params=config.get("params", {}),
            rule_type=rule_type,
        )

    def check(self, model: nn.Module, path_filter: Optional[str] = None,
              verbose: bool = False) -> List[CheckResult]:
        """执行检查

        Args:
            model: 要检查的模型
            path_filter: 路径过滤器，只检查匹配路径的模块
            verbose: 是否打印详细信息

        Returns:
            检查结果列表
        """
        # 提取结构信息
        extractor = ModuleInfoExtractor(model)
        root_info = extractor.extract()

        if verbose:
            print(f"Extracted {len(self._get_all_modules(root_info))} modules")

        # 应用检查
        results = []
        for rule in self.rules:
            modules_to_check = self._get_modules_for_rule(root_info, rule, path_filter)

            if rule.rule_type == "module_count":
                expected = (rule.params or {}).get("count")
                passed = expected is not None and len(modules_to_check) == expected
                msg = (f"Module count: expected {expected}, got {len(modules_to_check)}"
                       if not passed else f"Module count={len(modules_to_check)} matches")
                results.append(CheckResult(
                    rule_id=rule.id,
                    passed=passed,
                    message=msg,
                    severity=rule.severity,
                    suggestion=self._get_suggestion(rule, root_info, passed),
                ))
                continue

            for module_info in modules_to_check:
                passed, msg = self.evaluate_rule(rule, module_info)
                result = CheckResult(
                    rule_id=rule.id,
                    passed=passed,
                    message=f"[{module_info.module_path}] {msg}" if module_info.module_path else msg,
                    severity=rule.severity,
                    suggestion=self._get_suggestion(rule, module_info, passed)
                )
                results.append(result)

        return results

    @staticmethod
    def evaluate_rule(rule: CheckRule, module_info: ModuleInfo) -> Tuple[bool, str]:
        """Evaluate one selected node and fail closed for a missing selection."""
        if not module_info.exists:
            return False, f"Rule target does not exist or matched no modules: {module_info.module_path}"
        return rule.check_func(module_info, rule.params)

    @staticmethod
    def _missing_selection(label: str) -> ModuleInfo:
        """Create a diagnostic node so an unmatched rule cannot disappear."""
        return ModuleInfo(
            name="<missing>",
            type=ModuleType.CUSTOM,
            class_name="<missing>",
            module_path=label,
            registered_name="",
            exists=False,
            is_leaf=True,
        )

    def _selection_or_failure(self, selected: List[ModuleInfo], rule: CheckRule,
                              label: str) -> List[ModuleInfo]:
        """Keep zero matches meaningful only for aggregate count rules."""
        if selected or rule.rule_type == "module_count":
            return selected
        return [self._missing_selection(label)]

    def _get_modules_for_rule(self, root: ModuleInfo, rule: CheckRule,
                              path_filter: Optional[str]) -> List[ModuleInfo]:
        """Select nodes structurally; ``target_path`` is only a legacy hint.

        A path is never checked as part of the model contract.  New rules
        should use ``selector``/``target`` (class/type/name/occurrence) or a
        path-free ``structure`` rule.  Existing positional-path rules remain
        readable through a compatibility lookup, but an absent hint is not
        converted into a successful check or a path-specific assertion.
        """
        if path_filter:
            selected = self._find_modules_by_path_filter(root, path_filter)
            return self._selection_or_failure(
                selected, rule, f"<path_filter:{path_filter}>"
            )

        params = rule.params or {}
        selector = params.get("selector", params.get("target"))
        selector_keys = {"class_name", "class", "module_type", "type",
                         "name", "registered_name", "occurrence", "index"}
        if selector is None and any(key in params for key in selector_keys):
            selector = {key: params[key] for key in selector_keys if key in params}

        nodes = self._get_all_modules(root)
        if selector is not None:
            selected = [node for node in nodes
                        if self._matches_selector(node, selector)]
            occurrence = params.get("occurrence")
            if occurrence is None and isinstance(selector, dict):
                occurrence = selector.get("occurrence", selector.get("index"))
            if occurrence is not None:
                try:
                    index = int(occurrence)
                    selected = [selected[index]] if 0 <= index < len(selected) else []
                except (TypeError, ValueError):
                    selected = []
            return self._selection_or_failure(
                selected, rule, f"<selector:{selector!r}>"
            )

        # ``target_path`` is a deprecated locator retained solely so old
        # evaluation data keeps working.  It is not validated and cannot make
        # a rule pass.  Path-free rules use type-appropriate candidates.
        target_path = params.get("target_path")
        if target_path is not None:
            module = self._find_module_by_path(root, str(target_path))
            selected = [module] if module is not None else []
            return self._selection_or_failure(selected, rule, str(target_path))

        selected = self._default_candidates(root, rule)
        return self._selection_or_failure(
            selected, rule, f"<default:{rule.rule_type or rule.id}>"
        )

    def _matches_selector(self, info: ModuleInfo, selector: Any) -> bool:
        """Return whether ``info`` matches a path-independent selector."""
        if isinstance(selector, str) or isinstance(selector, ModuleType):
            return BuiltinRules._matches_expected(info, selector)
        if not isinstance(selector, dict):
            return False
        class_name = selector.get("class_name", selector.get("class"))
        if class_name is not None and info.class_name.lower() != str(class_name).lower():
            return False
        module_type = selector.get("module_type", selector.get("type"))
        if module_type is not None and not BuiltinRules._matches_expected(info, module_type):
            return False
        name = selector.get("name", selector.get("registered_name"))
        if name is not None and info.registered_name.lower() != str(name).lower():
            return False
        return True

    def _default_candidates(self, root: ModuleInfo, rule: CheckRule) -> List[ModuleInfo]:
        """Choose sensible nodes when a rule has no location selector."""
        rule_type = rule.rule_type
        if not rule_type and rule.id in self.RULE_FUNC_MAP:
            rule_type = rule.id
        # Structural aggregate rules describe the model root by default.
        if rule_type in {"children_count", "block_pattern", "structure",
                         "module_type", "type", "parameter_count"}:
            return [root]
        if rule_type == "module_count":
            selector = (rule.params or {}).get("selector", (rule.params or {}).get("target"))
            if selector is None:
                return self._get_all_modules(root)
        type_by_rule = {
            "linear_shape": ModuleType.LINEAR,
            "conv_shape": {ModuleType.CONV, ModuleType.CONVTRANSPOSE},
            "embedding_shape": ModuleType.EMBEDDING,
            "normalization_shape": {ModuleType.LAYERNORM, ModuleType.NORMALIZATION},
            "layer_norm_shape": ModuleType.LAYERNORM,
            "modulelist_length": ModuleType.MODULELIST,
            "moduledict_keys": ModuleType.MODULEDICT,
            "sequential_mlp": ModuleType.SEQUENTIAL,
        }
        expected_type = type_by_rule.get(rule_type)
        nodes = self._get_all_modules(root)
        if isinstance(expected_type, set):
            return [node for node in nodes if node.type in expected_type]
        if expected_type is not None:
            return [node for node in nodes if node.type == expected_type]
        return [root]

    def _find_module_by_path(self, root: ModuleInfo, path: str) -> Optional[ModuleInfo]:
        """根据路径查找模块"""
        if not path:
            return root

        parts = path.split('.')
        current = root
        for part in parts:
            if part.isdigit():
                idx = int(part)
                if idx < len(current.children):
                    current = current.children[idx]
                else:
                    return None
            else:
                found = None
                for child in current.children:
                    registered_name = child.module_path.rsplit(".", 1)[-1]
                    if (registered_name == part or child.name == part or
                            child.class_name == part):
                        found = child
                        break
                current = found
                if current is None:
                    return None
        return current

    def _find_modules_by_path_filter(self, root: ModuleInfo, path_filter: str) -> List[ModuleInfo]:
        """根据路径过滤器查找模块"""
        results = []
        for module in self._get_all_modules(root):
            if module.module_path.startswith(path_filter):
                results.append(module)
        return results

    def _get_all_modules(self, info: ModuleInfo) -> List[ModuleInfo]:
        """获取所有模块（递归）"""
        modules = [info]
        for child in info.children:
            modules.extend(self._get_all_modules(child))
        return modules

    def _get_suggestion(self, rule: CheckRule, info: ModuleInfo, passed: bool) -> str:
        """获取建议"""
        if passed:
            return ""

        suggestions = {
            "linear_shape": f"Check Linear layer at '{info.module_path}': expected in_features={rule.params.get('in_features')}, out_features={rule.params.get('out_features')}",
            "modulelist_length": f"Check ModuleList at '{info.module_path}': expected length={rule.params.get('length')}",
            "block_pattern": f"Check block pattern at '{info.module_path}': expected {rule.params.get('pattern')}",
            "sequential_mlp": f"Check MLP at '{info.module_path}': expected Linear->Activation->Linear structure",
        }

        return suggestions.get(rule.id, f"Review module at '{info.module_path}'")


class StructureLearner:
    """从参考模型学习结构配置"""

    @staticmethod
    def generate_config_from_model(model: nn.Module, output_path: str = None) -> dict:
        """从模型生成检查配置"""
        extractor = ModuleInfoExtractor(model)
        info = extractor.extract()

        config = [{
            "id": "model_structure",
            "type": "structure",
            "name": "Check complete model structure",
            "severity": "error",
            "params": {
                "structure": StructureLearner._info_to_structure(info),
            },
        }]

        if output_path:
            with open(output_path, 'w') as f:
                yaml.dump({"rules": config}, f, default_flow_style=False)

        return config

    @staticmethod
    def _info_to_structure(info: ModuleInfo) -> dict:
        """Convert a module tree to a portable contract without path fields."""
        spec = {
            "class_name": info.class_name,
            "module_type": info.type.value,
            "children": [StructureLearner._info_to_structure(child)
                         for child in info.children],
        }
        if info.registered_name:
            spec["name"] = info.registered_name
        if info.type in {ModuleType.LINEAR, ModuleType.CONV,
                         ModuleType.CONVTRANSPOSE}:
            spec["in_features"] = info.in_features
            spec["out_features"] = info.out_features
        if info.type == ModuleType.LAYERNORM:
            spec["normalized_shape"] = list(info.normalized_shape or ())
        if info.type in {ModuleType.CONV, ModuleType.CONVTRANSPOSE}:
            spec["kernel_size"] = list(info.kernel_size or ())
            spec["stride"] = list(info.stride or ())
            spec["padding"] = list(info.padding or ())
        for key in ("bias", "num_features", "num_embeddings",
                    "embedding_dim", "padding_idx", "num_channels",
                    "num_groups"):
            if key in info.params:
                spec[key] = info.params[key]
        return spec


# ============================================================================
# 6. 便捷函数
# ============================================================================

def check_model_structure(model: nn.Module, config_path: str = None,
                          config_dict: dict = None, verbose: bool = True) -> List[CheckResult]:
    """便捷函数：检查模型结构

    Args:
        model: 要检查的模型
        config_path: 配置文件路径 (YAML或JSON)
        config_dict: 配置字典 (与config_path二选一)
        verbose: 是否打印详细信息

    Returns:
        检查结果列表
    """
    checker = ModelStructureChecker()

    if config_dict:
        checker.add_rules_from_rules(config_dict)
    elif config_path:
        checker.add_rules_from_rules(config_path)
    else:
        raise ValueError("Either config_path or config_dict must be provided")

    results = checker.check(model, verbose=verbose)

    # 生成并打印报告

    return results


# ============================================================================
# 4. 检查点系统（统一输出格式 - 简化版）
# ============================================================================

@dataclass
class Checkpoint:
    """检查点数据类"""
    name: str
    passed: bool
    message: str
    severity: str = ""


def check_model_structure_v2(
        model: nn.Module,
        config_path: str = None,
        config_dict: dict = None,
        verbose: bool = True
) -> List[Checkpoint]:
    """
    检查模型结构，返回检查点列表

    Args:
        model: 要检查的模型
        config_path: 配置文件路径
        config_dict: 配置字典

    Returns:
        检查点列表，每个检查点为 (name, passed, message, severity)
    """
    checker = ModelStructureChecker()

    if config_dict:
        checker.add_rules_from_rules(config_dict)
    elif config_path:
        checker.add_rules_from_rules(config_path)
    else:
        raise ValueError("Either config_path or config_dict must be provided")

    # 提取结构信息
    extractor = ModuleInfoExtractor(model)
    root_info = extractor.extract()

    if verbose:
        print(f"Extracted {len(checker._get_all_modules(root_info))} modules")

    # 执行检查并转换为Checkpoint格式
    checkpoints = []
    for rule in checker.rules:
        modules_to_check = checker._get_modules_for_rule(root_info, rule, None)

        if rule.rule_type == "module_count":
            expected = (rule.params or {}).get("count")
            passed = expected is not None and len(modules_to_check) == expected
            message = (f"Module count: expected {expected}, got {len(modules_to_check)}"
                       if not passed else f"Module count={len(modules_to_check)} matches")
            checkpoints.append(Checkpoint(
                name=rule.id,
                passed=passed,
                message=message,
                severity=rule.severity.value,
            ))
            continue

        for module_info in modules_to_check:
            passed, msg = checker.evaluate_rule(rule, module_info)

            # 生成检查点名称
            if module_info.module_path:
                name = f"{rule.id}:{module_info.module_path}"
            else:
                name = rule.id

            checkpoint = Checkpoint(
                name=name,
                passed=passed,
                message=f"[{module_info.module_path}] {msg}" if module_info.module_path else msg,
                severity=rule.severity.value
            )
            checkpoints.append(checkpoint)

    return checkpoints


def print_checkpoint_report(checkpoints: List[Checkpoint]):
    """打印检查点报告"""
    total = len(checkpoints)
    passed = sum(1 for c in checkpoints if c.passed)
    failed = total - passed

    print("\n" + "=" * 70)
    print(f"STRUCTURE CHECK REPORT: {passed}/{total} passed ({passed / total * 100:.1f}%)")
    print("=" * 70)

    for c in checkpoints:
        status = "✓ PASS" if c.passed else "✗ FAIL"
        print(f"\n{status} [{c.severity.upper()}] {c.name}")
        print(f"  {c.message}")


def checkpoints_to_tuples(checkpoints: List[Checkpoint]) -> List[tuple]:
    """转换为元组格式，与第二份文件兼容"""
    return [(c.name, c.passed, c.message) for c in checkpoints]
