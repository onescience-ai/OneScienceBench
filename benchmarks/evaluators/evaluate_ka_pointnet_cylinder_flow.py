"""V2 evaluator adapter for the KA-PointNet cylinder-flow case."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

# The shared base evaluator registers h5/hdf5 loaders, but this case only uses
# numpy/yaml/model checks and the evaluation environment may omit h5py.
try:
    import h5py as _h5py
except ImportError:
    import types as _types

    _h5py = _types.ModuleType("h5py")
    _h5py.__spec__ = importlib.util.spec_from_loader("h5py", loader=None)

    def _missing_h5py_file(*args, **kwargs):
        raise ImportError("h5py is required to load HDF5 evaluator inputs")

    _h5py.File = _missing_h5py_file
    sys.modules["h5py"] = _h5py

try:
    import pandas as _pandas
except ImportError:
    import types as _types

    _pandas = _types.ModuleType("pandas")
    _pandas.__spec__ = importlib.util.spec_from_loader("pandas", loader=None)
    _pandas.DataFrame = type("DataFrame", (), {})
    _pandas.Series = type("Series", (), {})

    def _missing_pandas_read_csv(*args, **kwargs):
        raise ImportError("pandas is required to load CSV evaluator inputs")

    _pandas.read_csv = _missing_pandas_read_csv
    sys.modules["pandas"] = _pandas

# The result-script contract imports matplotlib only to define its plotting
# helper; plotting itself is outside the evaluator checkpoints for this case.
try:
    import matplotlib as _matplotlib
except ImportError:
    import types as _types

    _matplotlib = _types.ModuleType("matplotlib")
    _matplotlib.__path__ = []
    _matplotlib.__spec__ = importlib.util.spec_from_loader(
        "matplotlib", loader=None, is_package=True
    )

    def _missing_matplotlib_use(*args, **kwargs):
        return None

    _matplotlib.use = _missing_matplotlib_use
    _pyplot = _types.ModuleType("matplotlib.pyplot")
    _pyplot.__spec__ = importlib.util.spec_from_loader(
        "matplotlib.pyplot", loader=None
    )
    _matplotlib.pyplot = _pyplot
    sys.modules["matplotlib"] = _matplotlib
    sys.modules["matplotlib.pyplot"] = _pyplot

_BASE_EVALUATOR_DIR = None
for _parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    _base_dir = _parent / 'benchmarks' / 'evaluators' / 'base'
    if _base_dir.is_dir():
        _BASE_EVALUATOR_DIR = _base_dir
        break
if _BASE_EVALUATOR_DIR is None:
    raise FileNotFoundError('Could not locate repository evaluators/base directory.')
sys.path.insert(0, str(_BASE_EVALUATOR_DIR))
from base_evaluator import *


def _repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / 'benchmarks' / 'datasets').is_dir():
            return candidate
    raise FileNotFoundError('Could not locate repository root containing benchmarks/datasets.')


candidate_env = os.environ.get("candidate_env")
stardard_name = os.environ.get("stardard_name")

def _candidate_model_path() -> Path:
    if not candidate_env:
        raise RuntimeError("candidate_env is not configured")
    candidate_path = os.environ.get(candidate_env)
    if not candidate_path:
        raise RuntimeError(f"{candidate_env} is not configured")
    return Path(candidate_path).expanduser().resolve()

def _reference_model_path() -> Path:
    return (
        _repo_root()
        / "benchmarks"
        / "standard"
        / stardard_name
    ).resolve()

CANDIDATE_DIR = _candidate_model_path()
REFERENCE_DIR = _reference_model_path()

CASE_ID = "ka_pointnet_cylinder_flow"
REQUIRED_FILES = (
    "models/KAPointNet.py",
    "config/config.yaml",
    "scripts/train.py",
    "scripts/inference.py",
    "scripts/result.py",
)
REQUIRED_CLASS_NAME = "KAPointNet"
DATA_ROOT = Path("/public/share/sugonhpcapp01/onestore/onedatasets/PointNetCFD_data/data")
EXPECTED_RAW_SHAPE = (2215, 1024, 5)
EXPECTED_SPLITS = {
    "training_idx.npy": 1772,
    "validation_idx.npy": 221,
    "test_idx.npy": 222,
}
EXPECTED_MODEL_PARAMS = 5_316_224
FORWARD_COMPARE_ATOL = 1e-6
SEED = 20260818

def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_config(candidate_dir: Path, config_name: str = "config/config.yaml") -> dict[str, Any]:
    import yaml

    config_path = (candidate_dir / str(config_name)).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def load_standard_model_class():
    model_module = import_module_from_path(
        (REFERENCE_DIR / "models" / "KAPointNet.py").resolve(),
        "ka_pointnet_standard_model",
    )
    model_cls = getattr(model_module, REQUIRED_CLASS_NAME, None)
    if model_cls is None:
        raise AttributeError(f"{REQUIRED_CLASS_NAME} is not exported by the standard model.")
    return model_cls

def instantiate_model(model_cls, config: dict[str, Any], torch_module):
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    kwargs = {
        "input_channels": int(model_cfg.get("input_channels", 2)),
        "output_channels": int(model_cfg.get("output_channels", 3)),
        "network_scaling": float(model_cfg.get("network_scaling", 1.0)),
        "poly_degree": int(model_cfg.get("poly_degree", 5)),
        "alpha": float(model_cfg.get("alpha", 1.0)),
        "beta": float(model_cfg.get("beta", 1.0)),
        "num_points": int(data_cfg.get("num_points", 1024)),
    }
    return model_cls(**kwargs)

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "file:models/KAPointNet.py",
        "file:config/config.yaml",
        "file:scripts/train.py",
        "file:scripts/inference.py",
        "file:scripts/result.py",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        required_files = tuple(config.get("required_files", REQUIRED_FILES))
        for rel_path in required_files:
            path = (CANDIDATE_DIR / rel_path).resolve()
            ok, msg = path.is_file(), str(path)
            update_check(current_index, ok, msg)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "data:root_absolute",
        "data:root_matches_shared_path",
        "data:root_exists",
        "data:cfd_file_name",
        "data:cfd_file_exists",
        "data:raw_shape",
        "data:raw_dtype_float",
        "data:input_columns_xy",
        "data:output_columns_uvp",
        "data:num_points_1024",
        "data:training_idx.npy:exists",
        "data:training_idx.npy:length",
        "data:training_idx.npy:int_indices",
        "data:validation_idx.npy:exists",
        "data:validation_idx.npy:length",
        "data:validation_idx.npy:int_indices",
        "data:test_idx.npy:exists",
        "data:test_idx.npy:length",
        "data:test_idx.npy:int_indices",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_config = load_config(CANDIDATE_DIR, config.get("config_name", "config/config.yaml"))
        data_cfg = candidate_config.get("data", {})
        root = Path(data_cfg.get("root", ""))
        ok, msg = root.is_absolute(), str(root)
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = root == DATA_ROOT, str(root)
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = root.is_dir(), str(root)
        update_check(current_index, ok, msg)
        current_index += 1

        cfd_file = data_cfg.get("cfd_file")
        raw_path = root / str(cfd_file)
        ok, msg = cfd_file == "CFDdata.npy", str(cfd_file)
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = raw_path.is_file(), str(raw_path)
        update_check(current_index, ok, msg)
        current_index += 1

        if raw_path.is_file():
            import numpy as np

            raw = np.load(raw_path, mmap_mode="r")
            ok, msg = tuple(raw.shape) == EXPECTED_RAW_SHAPE, f"shape={tuple(raw.shape)}"
            update_check(current_index, ok, msg)
            current_index += 1
            ok, msg = str(raw.dtype).startswith("float"), str(raw.dtype)
            update_check(current_index, ok, msg)
            current_index += 1
        else:
            ok, msg = False, "CFDdata.npy missing"
            update_check(current_index, ok, msg)
            current_index += 1
            ok, msg = False, "CFDdata.npy missing"
            update_check(current_index, ok, msg)
            current_index += 1

        input_cols = data_cfg.get("input_columns", {})
        output_cols = data_cfg.get("output_columns", {})
        ok, msg = input_cols == {"x": 0, "y": 1}, str(input_cols)
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = output_cols == {"p": 2, "u": 3, "v": 4}, str(output_cols)
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = int(data_cfg.get("num_points", -1)) == 1024, str(data_cfg.get("num_points"))
        update_check(current_index, ok, msg)
        current_index += 1

        for filename, expected_len in EXPECTED_SPLITS.items():
            configured = data_cfg.get(filename.replace(".npy", "_file"))
            if configured is None:
                configured = data_cfg.get(filename.replace("_idx.npy", "_idx_file"))
            configured = str(configured or filename)
            split_path = root / configured
            ok, msg = split_path.is_file(), str(split_path)
            update_check(current_index, ok, msg)
            current_index += 1
            if split_path.is_file():
                import numpy as np

                values = np.load(split_path)
                ok, msg = len(values) == expected_len, f"length={len(values)}, expected={expected_len}"
                update_check(current_index, ok, msg)
                current_index += 1
                ok, msg = str(values.dtype).startswith("int"), str(values.dtype)
                update_check(current_index, ok, msg)
                current_index += 1
            else:
                ok, msg = False, f"{filename} missing"
                update_check(current_index, ok, msg)
                current_index += 1
                ok, msg = False, f"{filename} missing"
                update_check(current_index, ok, msg)
                current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "train_script:has_main",
        "train_script:has_parse_args",
        "train_script:uses_adam",
        "train_script:prints_loss",
        "train_script:saves_checkpoint",
        "train_script:checkpoint_path_contract",
        "train_script:results_dir_contract",
        "train_script:learning_rate_contract",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        train_path = (CANDIDATE_DIR / "scripts" / "train.py").resolve()
        source = train_path.read_text(encoding="utf-8")
        train_module = import_module_from_path(
            train_path,
            "ka_pointnet_candidate_train",
        )
        candidate_config = load_config(CANDIDATE_DIR, config.get("config_name", "config/config.yaml"))
        ok, msg = hasattr(train_module, "main"), "main"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = hasattr(train_module, "parse_args"), "parse_args"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = "Adam" in source, "Adam optimizer"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = "train_loss" in source and "val_loss" in source, "train_loss/val_loss"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = "torch.save" in source and "best_model.pth" in json.dumps(candidate_config), "checkpoint"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = candidate_config.get("experiment", {}).get("checkpoint_path") == "weight/best_model.pth", str(candidate_config.get("experiment", {}).get("checkpoint_path"))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = candidate_config.get("experiment", {}).get("results_dir") == "results", str(candidate_config.get("experiment", {}).get("results_dir"))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = abs(float(candidate_config.get("training", {}).get("learning_rate", 0.0)) - 5e-4) < 1e-12, str(candidate_config.get("training", {}).get("learning_rate"))
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "eval_script:inference_has_main",
        "eval_script:result_has_main",
        "eval_script:loads_checkpoint",
        "eval_script:writes_metrics",
        "eval_script:writes_predictions",
        "eval_script:writes_figures",
        "eval_script:default_split_test",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        inference_path = (CANDIDATE_DIR / "scripts" / "inference.py").resolve()
        result_path = (CANDIDATE_DIR / "scripts" / "result.py").resolve()
        inference_source = inference_path.read_text(encoding="utf-8")
        result_source = result_path.read_text(encoding="utf-8")
        inference_module = import_module_from_path(
            inference_path,
            "ka_pointnet_candidate_inference",
        )
        result_module = import_module_from_path(
            result_path,
            "ka_pointnet_candidate_result",
        )
        candidate_config = load_config(CANDIDATE_DIR, config.get("config_name", "config/config.yaml"))
        ok, msg = hasattr(inference_module, "main"), "inference.main"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = hasattr(result_module, "main"), "result.main"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = "load_state_dict" in inference_source and "torch.load" in inference_source, "checkpoint load"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = "metrics" in inference_source and "json.dump" in inference_source, "metrics json"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = "np.savez" in inference_source, "prediction npz"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = "save_field" in result_source or "save_field_triptych" in result_source, "figures"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = candidate_config.get("inference", {}).get("split") == "test", str(candidate_config.get("inference", {}).get("split"))
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "model_structure:class_name",
        "model_structure:is_torch_module",
        "model_structure:has_jacobi_layer_class",
        "model_structure:ten_jacobi_layers",
        "model_structure:nine_batchnorm_layers",
        "model_structure:uses_max_pooling",
        "model_structure:uses_jacobi_coeffs",
        "model_structure:poly_degree_5",
        "model_structure:alpha_beta_1",
        "model_structure:parameter_count",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        model_path = (CANDIDATE_DIR / str(config.get("model_file", "models/KAPointNet.py"))).resolve()
        model_module = import_module_from_path(
            model_path,
            "ka_pointnet_candidate_model",
        )
        candidate_config = load_config(CANDIDATE_DIR, config.get("config_name", "config/config.yaml"))
        model_cls = getattr(model_module, config.get("model_class", REQUIRED_CLASS_NAME), None)
        source = model_path.read_text(encoding="utf-8")
        model = instantiate_model(model_cls, candidate_config, torch)
        module_class_names = [module.__class__.__name__ for module in model.modules()]
        param_count = sum(param.numel() for param in model.parameters())
        ok, msg = model_cls.__name__ == REQUIRED_CLASS_NAME, model_cls.__name__
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = isinstance(model, torch.nn.Module), str(type(model))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = hasattr(model_module, "JacobiKANLayer"), "JacobiKANLayer"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = module_class_names.count("JacobiKANLayer") == 10, f"count={module_class_names.count('JacobiKANLayer')}"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = module_class_names.count("BatchNorm1d") == 9, f"count={module_class_names.count('BatchNorm1d')}"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = "max_pool1d" in source, "PointNet global max pooling"
        update_check(current_index, ok, msg)
        current_index += 1
        has_jacobi_coeffs = any(name.endswith(".jacobi_coeffs") for name, _ in model.named_parameters())
        ok, msg = has_jacobi_coeffs, "jacobi_coeffs parameter"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = int(candidate_config.get("model", {}).get("poly_degree", -1)) == 5, str(candidate_config.get("model", {}).get("poly_degree"))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = float(candidate_config.get("model", {}).get("alpha", 0.0)) == 1.0 and float(candidate_config.get("model", {}).get("beta", 0.0)) == 1.0, f"alpha={candidate_config.get('model', {}).get('alpha')} beta={candidate_config.get('model', {}).get('beta')}"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = param_count == EXPECTED_MODEL_PARAMS, f"params={param_count}"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "forward_executable:returns_tensor",
        "forward_executable:shape",
        "forward_executable:finite",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        torch.manual_seed(SEED)
        model_path = (CANDIDATE_DIR / str(config.get("model_file", "models/KAPointNet.py"))).resolve()
        model_module = import_module_from_path(model_path, "ka_pointnet_candidate_forward_model")
        candidate_config = load_config(CANDIDATE_DIR, config.get("config_name", "config/config.yaml"))
        model_cls = getattr(model_module, config.get("model_class", REQUIRED_CLASS_NAME), None)
        model = instantiate_model(model_cls, candidate_config, torch).eval()
        x = torch.linspace(-1.0, 1.0, steps=2 * 1024, dtype=torch.float32).view(1, 2, 1024)
        with torch.no_grad():
            y = model(x)
        ok, msg = isinstance(y, torch.Tensor), str(type(y))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = tuple(y.shape) == (1, 3, 1024), f"shape={tuple(y.shape)}"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = bool(torch.isfinite(y).all().item()), "all finite"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "forward_equivalence:state_dict_keys",
        "forward_equivalence:state_dict_shapes",
        "forward_equivalence:same_shape",
        "forward_equivalence:same_forward",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        candidate_path = (CANDIDATE_DIR / str(config.get("model_file", "models/KAPointNet.py"))).resolve()
        candidate_module = import_module_from_path(
            candidate_path,
            "ka_pointnet_candidate_equivalence_model",
        )
        candidate_config = load_config(CANDIDATE_DIR, config.get("config_name", "config/config.yaml"))
        model_class_name = config.get("model_class", REQUIRED_CLASS_NAME)
        model_cls = getattr(candidate_module, model_class_name, None)
        checks = checks
        standard_cls = load_standard_model_class()
        torch.manual_seed(SEED)
        standard_model = instantiate_model(standard_cls, candidate_config, torch).eval()
        torch.manual_seed(SEED)
        candidate_model = instantiate_model(model_cls, candidate_config, torch).eval()

        standard_state = standard_model.state_dict()
        candidate_state = candidate_model.state_dict()
        same_keys = list(standard_state.keys()) == list(candidate_state.keys())
        same_shapes = same_keys and all(
            standard_state[key].shape == candidate_state[key].shape for key in standard_state
        )
        ok, msg = same_keys, f"keys_match={same_keys}"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = same_shapes, f"shapes_match={same_shapes}"
        update_check(current_index, ok, msg)
        current_index += 1

        if same_shapes:
            candidate_model.load_state_dict(standard_state, strict=True)
            x = torch.linspace(-0.75, 0.75, steps=2 * 1024, dtype=torch.float32).view(1, 2, 1024)
            with torch.no_grad():
                standard_out = standard_model(x)
                candidate_out = candidate_model(x)
            same_shape = isinstance(candidate_out, torch.Tensor) and tuple(candidate_out.shape) == tuple(standard_out.shape)
            max_abs_diff = (
                float((standard_out - candidate_out).abs().max().item())
                if same_shape
                else float("inf")
            )
            ok, msg = same_shape, f"standard={tuple(standard_out.shape)}, candidate={tuple(candidate_out.shape) if isinstance(candidate_out, torch.Tensor) else type(candidate_out)}"
            update_check(current_index, ok, msg)
            current_index += 1
            ok, msg = max_abs_diff <= FORWARD_COMPARE_ATOL, f"max_abs_diff={max_abs_diff}, atol={FORWARD_COMPARE_ATOL}"
            update_check(current_index, ok, msg)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "traning:single_step_loss_finite",
        "traning:single_step_gradients",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        model_path = (CANDIDATE_DIR / str(config.get("model_file", "models/KAPointNet.py"))).resolve()
        model_module = import_module_from_path(model_path, "ka_pointnet_candidate_training_model")
        candidate_config = load_config(CANDIDATE_DIR, config.get("config_name", "config/config.yaml"))
        model_cls = getattr(model_module, config.get("model_class", REQUIRED_CLASS_NAME), None)
        torch.manual_seed(SEED)
        model = instantiate_model(model_cls, candidate_config, torch).train()
        x = torch.randn(1, 2, 1024)
        target = torch.randn(1, 3, 1024)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        y = model(x)
        loss = torch.mean((y - target) ** 2)
        loss.backward()
        optimizer.step()
        grad_ok = any(param.grad is not None and torch.isfinite(param.grad).all().item() for param in model.parameters())
        ok, msg = bool(torch.isfinite(loss).item()), f"loss={float(loss.item())}"
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = grad_ok, "at least one finite gradient"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "artifacts:checkpoint_path",
        "artifacts:results_dir",
        "artifacts:weight_dir_present",
        "artifacts:results_dir_present",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate_config = load_config(CANDIDATE_DIR, config.get("config_name", "config/config.yaml"))
        experiment = candidate_config.get("experiment", {})
        ok, msg = experiment.get("checkpoint_path") == "weight/best_model.pth", str(experiment.get("checkpoint_path"))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = experiment.get("results_dir") == "results", str(experiment.get("results_dir"))
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = (CANDIDATE_DIR / "weight").is_dir(), str(CANDIDATE_DIR / "weight")
        update_check(current_index, ok, msg)
        current_index += 1
        ok, msg = (CANDIDATE_DIR / "results").is_dir(), str(CANDIDATE_DIR / "results")
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
