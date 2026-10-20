"""V2 adapter for the DeepCFD secondary-development evaluator."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import os
import pickle
import random
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

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

REQUIRED_FILES = ("unetex.py", "dataset.py", "train.py", "inference.py", "config.yaml")
MODEL_CLASSES = ("DoubleConv2D", "Down2D", "Up2D", "UNetEncoder2D", "UNetDecoder2D", "UNetHead2D", "DecoderPath", "UNetEx")
MODEL_CONFIG = {"in_channels": 3, "out_channels": 3, "base_channels": 8, "num_stages": 4, "kernel_size": 5, "bilinear": True, "normtype": "none", "shared_decoder_stages": 2}
SEED = 20260819

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "required:unetex.py", "syntax:unetex.py", "required:dataset.py", "syntax:dataset.py",
        "required:train.py", "syntax:train.py", "required:inference.py", "syntax:inference.py",
        "required:config.yaml",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        for name in REQUIRED_FILES:
            path = CANDIDATE_DIR / name
            ok = path.is_file()
            update_check(current_index, ok, str(path))
            current_index += 1
            if path.suffix == ".py" and path.is_file():
                try:
                    text = path.read_text(encoding="utf-8")
                    ast.parse(text, filename=str(path))
                    ok = bool(text.strip())
                    msg = f"lines={len(text.splitlines())}"
                except Exception as exc:
                    ok = False
                    msg = f"{type(exc).__name__}: {exc}"
                update_check(current_index, ok, msg)
                current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_data_split(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "data:pickle_files", "data:split_contract", "data:loader_config",
        "data:tensor_channels", "data:seeded_shuffle_source", "data:loader_callable",
        "data:loader_runtime",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import numpy as np
        import yaml
        data_config = yaml.safe_load((CANDIDATE_DIR / "config.yaml").read_text(encoding="utf-8"))
        data = data_config.get("data", {})
        model = data_config.get("model", {})
        source = (CANDIDATE_DIR / "dataset.py").read_text(encoding="utf-8")
        ratio = data.get("split_ratio")
        split = int(981 * float(ratio)) if ratio is not None else -1
        ok = Path(str(data.get("input_file"))).name == "dataX.pkl" and Path(str(data.get("target_file"))).name == "dataY.pkl"
        update_check(current_index, ok, f"input={data.get('input_file')} target={data.get('target_file')}")
        current_index += 1
        ok = math.isclose(float(ratio), 0.7, rel_tol=1e-9, abs_tol=1e-12) and data.get("seed") == 0 and split == 686 and 981 - split == 295
        update_check(current_index, ok, f"ratio={ratio} seed={data.get('seed')} counts={split}/{981 - split}")
        current_index += 1
        ok = data.get("batch_size") == 16 and data.get("num_workers") == 0
        update_check(current_index, ok, f"batch_size={data.get('batch_size')} num_workers={data.get('num_workers')}")
        current_index += 1
        ok = model.get("in_channels") == 3 and model.get("out_channels") == 3
        update_check(current_index, ok, f"in={model.get('in_channels')} out={model.get('out_channels')}")
        current_index += 1
        ok = "random.Random" in source and ".shuffle(" in source
        update_check(current_index, ok, "expected random.Random(seed).shuffle(indices)")
        current_index += 1

        spec = importlib.util.spec_from_file_location("deepcfd_sec_dataset", CANDIDATE_DIR / "dataset.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {CANDIDATE_DIR / 'dataset.py'}")
        dataset_module = importlib.util.module_from_spec(spec)
        sys.modules["deepcfd_sec_dataset"] = dataset_module
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(CANDIDATE_DIR))
            spec.loader.exec_module(dataset_module)
        finally:
            sys.path[:] = old_path
        creator = getattr(dataset_module, "create_dataloaders", None)
        ok = callable(creator)
        update_check(current_index, ok, "create_dataloaders")
        current_index += 1
        if not ok:
            raise AttributeError("dataset.py must expose create_dataloaders")
        values = np.arange(10 * 3 * 4 * 5, dtype=np.float32).reshape(10, 3, 4, 5)
        with tempfile.TemporaryDirectory(prefix="deepcfd_eval_") as directory:
            path = Path(directory)
            with (path / "dataX.pkl").open("wb") as stream:
                pickle.dump(values, stream)
            with (path / "dataY.pkl").open("wb") as stream:
                pickle.dump(values * 0.5, stream)
            result = creator(directory=path, input_file="dataX.pkl", target_file="dataY.pkl", split_ratio=0.7, seed=0, batch_size=16, num_workers=0)
        train_loader, test_loader = result[:2]
        expected = list(range(10))
        random.Random(0).shuffle(expected)
        actual_train = list(getattr(train_loader.dataset, "indices", ()))
        actual_test = list(getattr(test_loader.dataset, "indices", ()))
        ok = actual_train == expected[:7] and actual_test == expected[7:] and train_loader.batch_size == 16 and test_loader.num_workers == 0
        update_check(current_index, ok, f"train={actual_train} test={actual_test}")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "train_script:config_cli", "train_script:adamw", "train_script:schedule",
        "train_script:optimization_step", "train_script:unified_validation_selection",
        "train_script:loss_callable", "train_script:total_mse_loss_runtime",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        import yaml
        training_config = yaml.safe_load((CANDIDATE_DIR / "config.yaml").read_text(encoding="utf-8"))
        training = training_config.get("training", {})
        source = (CANDIDATE_DIR / "train.py").read_text(encoding="utf-8")
        lower = source.lower()
        ok = "argparse" in lower and "--config" in source
        update_check(current_index, ok, "train.py --config")
        current_index += 1
        try:
            ok = "adamw" in lower and math.isclose(float(training.get("lr")), 0.001, rel_tol=1e-9, abs_tol=1e-12) and math.isclose(float(training.get("weight_decay")), 0.005, rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            ok = False
        update_check(current_index, ok, f"lr={training.get('lr')} weight_decay={training.get('weight_decay')}")
        current_index += 1
        ok = training.get("num_epochs") == 1000 and training.get("eval_interval") == 10 and training.get("patience") == 200
        update_check(current_index, ok, f"epochs={training.get('num_epochs')} interval={training.get('eval_interval')} patience={training.get('patience')}")
        current_index += 1
        ok = ".backward(" in source and "optimizer.step(" in source and "zero_grad" in source
        update_check(current_index, ok, "zero_grad -> backward -> optimizer.step")
        current_index += 1
        ok = "total_mse" in lower and "val_loss" in lower and "torch.save" in lower and "patience" in lower
        update_check(current_index, ok, "validation/checkpoint/early-stop use Total_MSE as val_loss")
        current_index += 1
        spec = importlib.util.spec_from_file_location("deepcfd_sec_train", CANDIDATE_DIR / "train.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {CANDIDATE_DIR / 'train.py'}")
        train_module = importlib.util.module_from_spec(spec)
        sys.modules["deepcfd_sec_train"] = train_module
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(CANDIDATE_DIR))
            spec.loader.exec_module(train_module)
        finally:
            sys.path[:] = old_path
        loss_function = next((getattr(train_module, name) for name in ("loss_func", "compute_loss", "total_mse_loss") if callable(getattr(train_module, name, None))), None)
        ok = callable(loss_function)
        update_check(current_index, ok, "loss_func/compute_loss/total_mse_loss")
        current_index += 1
        if loss_function is None:
            raise AttributeError("train.py must expose a callable unified loss")
        generator = torch.Generator().manual_seed(SEED)
        predictions = torch.randn((2, 3, 3, 4), generator=generator)
        targets = torch.randn((2, 3, 3, 4), generator=generator)
        actual = loss_function(predictions, targets)
        expected = (predictions - targets).square().mean(dim=(0, 2, 3)).sum()
        ok = isinstance(actual, torch.Tensor) and actual.numel() == 1 and torch.allclose(actual, expected)
        update_check(current_index, ok, f"actual={float(actual)} expected={float(expected)}")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "eval_script:metric_callable", "eval_script:config_cli", "eval_script:metric_helper_called",
        "eval_script:metric_formula", "eval_script:checkpoint_restore", "eval_script:optional_absent",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        path = CANDIDATE_DIR / "inference.py"
        if not path.is_file():
            for _ in range(len(CHECKPOINT_NAMES)):
                update_check(current_index, True, "inference.py absent; optional check passed")
                current_index += 1
        else:
            source = path.read_text(encoding="utf-8")
            lower = source.lower()
            spec = importlib.util.spec_from_file_location("deepcfd_sec_inference", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load {path}")
            inference_module = importlib.util.module_from_spec(spec)
            sys.modules["deepcfd_sec_inference"] = inference_module
            old_path = list(sys.path)
            try:
                sys.path.insert(0, str(CANDIDATE_DIR))
                spec.loader.exec_module(inference_module)
            finally:
                sys.path[:] = old_path
            metric = getattr(inference_module, "compute_sample_normalized_grid_mse", None)
            ok = callable(metric)
            update_check(current_index, ok, "compute_sample_normalized_grid_mse")
            current_index += 1
            generator = torch.Generator().manual_seed(SEED + 1)
            predictions = torch.randn((3, 3, 2, 5), generator=generator)
            targets = torch.randn((3, 3, 2, 5), generator=generator)
            actual = metric(predictions, targets)
            values = {key: float(actual[key]) for key in ("Ux_MSE", "Uy_MSE", "p_MSE", "Total_MSE")}
            expected_values = (predictions - targets).square().mean(dim=(0, 2, 3))
            expected = {"Ux_MSE": float(expected_values[0]), "Uy_MSE": float(expected_values[1]), "p_MSE": float(expected_values[2]), "Total_MSE": float(expected_values.sum())}
            ok = "argparse" in lower and "--config" in source
            update_check(current_index, ok, "inference.py --config")
            current_index += 1
            ok = any(isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == "compute_sample_normalized_grid_mse") or (isinstance(node.func, ast.Attribute) and node.func.attr == "compute_sample_normalized_grid_mse")) for node in ast.walk(ast.parse(source)))
            update_check(current_index, ok, "inference.py directly calls the required helper")
            current_index += 1
            ok = all(math.isclose(values[key], expected[key], rel_tol=1e-6, abs_tol=1e-8) for key in expected)
            update_check(current_index, ok, f"actual={values} expected={expected}")
            current_index += 1
            ok = "torch.load" in lower and "load_state_dict" in lower
            update_check(current_index, ok, "restore weight/best_model.pt")
            current_index += 1
            update_check(current_index, True, "inference.py present")
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "model_structure:required_classes", "model_structure:constructor_defaults",
        "model_structure:shared_encoder", "model_structure:down_stages",
        "model_structure:double_conv_parameter_keys", "model_structure:partial_decoder",
        "model_structure:independent_decoder_compatibility", "model_structure:fully_shared_decoder",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        nn = torch.nn

        def sequence(module: Any) -> list[Any]:
            try:
                return list(module)
            except TypeError:
                return []

        def double_conv_ok(module: Any, expected_in: int, expected_out: int) -> bool:
            layers = sequence(getattr(module, "double_conv", None))
            return len(layers) == 4 and isinstance(layers[0], nn.Conv2d) and isinstance(layers[1], nn.ReLU) and isinstance(layers[2], nn.Conv2d) and isinstance(layers[3], nn.ReLU) and layers[0].in_channels == expected_in and layers[0].out_channels == expected_out and layers[2].out_channels == expected_out and layers[0].bias is None and layers[2].bias is None

        def upsample_ok(stage: Any) -> bool:
            up = getattr(stage, "up", None)
            conv = sequence(getattr(getattr(stage, "conv", None), "double_conv", None))
            scale = getattr(up, "scale_factor", None)
            scale_ok = scale == 2 or scale == 2.0 or scale == (2.0, 2.0)
            return isinstance(up, nn.Upsample) and scale_ok and up.mode == "bilinear" and up.align_corners is True and len(conv) == 4 and isinstance(conv[0], nn.Conv2d) and conv[0].out_channels == conv[0].in_channels // 2

        def import_module(path: Path, name: str):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            old_path = list(sys.path)
            try:
                sys.path.insert(0, str(path.parent))
                spec.loader.exec_module(module)
            finally:
                sys.path[:] = old_path
            return module

        def build_model(module: Any, shared_stages: int = 2) -> Any:
            config_values = dict(MODEL_CONFIG, shared_decoder_stages=shared_stages)
            return module.UNetEx(**config_values)

        generated_module = import_module(CANDIDATE_DIR / "unetex.py", "deepcfd_sec_generated_unetex")
        reference_module = import_module(REFERENCE_DIR / "unetex.py", "deepcfd_sec_reference_unetex")
        missing = [name for name in MODEL_CLASSES if not inspect.isclass(getattr(generated_module, name, None)) or not issubclass(getattr(generated_module, name, None), nn.Module)]
        signature = inspect.signature(generated_module.UNetEx.__init__).parameters
        defaults_ok = signature["num_stages"].default == 4 and signature["shared_decoder_stages"].default == 2
        model = build_model(generated_module)
        encoder = getattr(model, "encoder", None)
        down_stages = getattr(encoder, "down_stages", None)
        channels = (8, 16, 32, 64, 128)
        down_ok = isinstance(down_stages, nn.ModuleList) and len(down_stages) == 4
        if down_ok:
            for stage, in_channels, out_channels in zip(down_stages, channels[:-1], channels[1:]):
                children = sequence(getattr(stage, "maxpool_conv", None))
                down_ok = down_ok and len(children) == 2 and isinstance(children[0], nn.MaxPool2d) and double_conv_ok(children[1], in_channels, out_channels)
        shared_up = getattr(getattr(model, "shared_decoder", None), "up_stages", None)
        decoders = getattr(model, "decoders", None)
        partial_ok = isinstance(shared_up, nn.ModuleList) and len(shared_up) == 2
        partial_ok = partial_ok and isinstance(decoders, nn.ModuleList) and len(decoders) == 3
        head_ids = []
        if partial_ok:
            partial_ok = all(upsample_ok(stage) for stage in shared_up)
            for path in decoders:
                up_stages = getattr(getattr(path, "decoder", None), "up_stages", None)
                head = getattr(getattr(path, "head", None), "conv", None)
                partial_ok = partial_ok and isinstance(path, generated_module.DecoderPath) and isinstance(up_stages, nn.ModuleList) and len(up_stages) == 2 and all(upsample_ok(stage) for stage in up_stages) and isinstance(head, nn.Conv2d) and (head.in_channels, head.out_channels) == (8, 1) and head.kernel_size == (1, 1) and head.padding == (0, 0) and head.bias is not None
                head_ids.append(id(head))
        partial_ok = partial_ok and len(set(head_ids)) == 3
        state_keys = set(model.state_dict())
        exact_conv_keys = "encoder.inc.double_conv.0.weight" in state_keys and "encoder.inc.double_conv.2.weight" in state_keys and "encoder.inc.double_conv.0.bias" not in state_keys and "encoder.inc.double_conv.2.bias" not in state_keys
        independent = build_model(generated_module, 0)
        fully_shared = build_model(generated_module, 4)
        independent_ok = len(independent.shared_decoder.up_stages) == 0 and all(len(path.decoder.up_stages) == 4 for path in independent.decoders) and not any(key.startswith("shared_decoder.up_stages.") for key in independent.state_dict())
        fully_shared_ok = len(fully_shared.shared_decoder.up_stages) == 4 and all(len(path.decoder.up_stages) == 0 for path in fully_shared.decoders)
        reference_keys = set(build_model(reference_module, 0).state_dict())
        compatibility_ok = independent_ok and set(independent.state_dict()) == reference_keys
        checks_to_record = (
            (not missing, f"missing={missing}"),
            (defaults_ok, "num_stages=4 shared_decoder_stages=2"),
            (isinstance(encoder, generated_module.UNetEncoder2D) and double_conv_ok(getattr(encoder, "inc", None), 3, 8), "encoder.inc and channels 3->8"),
            (down_ok, "four MaxPool2d+DoubleConv2D stages: 8->16->32->64->128"),
            (exact_conv_keys, "Conv-ReLU-Conv-ReLU without bias"),
            (partial_ok, "shared 4->3->2; independent 2->1->0; three heads"),
            (compatibility_ok, "shared_decoder_stages=0 reference keys"),
            (fully_shared_ok, "shared_decoder_stages=num_stages"),
        )
        for ok, msg in checks_to_record:
            update_check(current_index, ok, msg)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_executable:required_shape", "forward_executable:odd_grid_alignment", "forward_executable:finite_outputs"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        spec = importlib.util.spec_from_file_location("deepcfd_sec_generated_unetex_forward", CANDIDATE_DIR / "unetex.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {CANDIDATE_DIR / 'unetex.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["deepcfd_sec_generated_unetex_forward"] = module
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(CANDIDATE_DIR))
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path
        model = module.UNetEx(**MODEL_CONFIG).cpu().eval()
        generator = torch.Generator().manual_seed(SEED)
        required_input = torch.randn((1, 3, 172, 79), generator=generator)
        odd_input = torch.randn((2, 3, 35, 21), generator=generator)
        with torch.no_grad():
            required_output = model(required_input)
            odd_output = model(odd_input)
        ok = tuple(required_output.shape) == (1, 3, 172, 79)
        update_check(current_index, ok, f"actual={tuple(required_output.shape)} expected=(1, 3, 172, 79)")
        current_index += 1
        ok = tuple(odd_output.shape) == (2, 3, 35, 21)
        update_check(current_index, ok, f"actual={tuple(odd_output.shape)} expected=(2, 3, 35, 21)")
        current_index += 1
        ok = bool(torch.isfinite(required_output).all()) and bool(torch.isfinite(odd_output).all())
        update_check(current_index, ok, "all output values are finite")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_equivalence:shared_decoder_stages_0", "forward_equivalence:shared_decoder_stages_2", "forward_equivalence:shared_decoder_stages_4"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        def import_module(path: Path, name: str):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            old_path = list(sys.path)
            try:
                sys.path.insert(0, str(path.parent))
                spec.loader.exec_module(module)
            finally:
                sys.path[:] = old_path
            return module

        def build_model(module: Any, shared_stages: int):
            values = dict(MODEL_CONFIG, shared_decoder_stages=shared_stages)
            return module.UNetEx(**values)

        generated_module = import_module(CANDIDATE_DIR / "unetex.py", "deepcfd_sec_generated_equivalence")
        reference_module = import_module(REFERENCE_DIR / "unetex.py", "deepcfd_sec_reference_equivalence")
        generator = torch.Generator().manual_seed(SEED + 2)
        inputs = torch.randn((1, 3, 35, 21), generator=generator)
        for shared_stages in (0, 2, 4):
            generated = build_model(generated_module, shared_stages).cpu().eval()
            reference = build_model(reference_module, shared_stages).cpu().eval()
            with torch.no_grad():
                for parameter in generated.parameters():
                    parameter.fill_(0.001 if parameter.ndim > 1 else 0.01)
                for parameter in reference.parameters():
                    parameter.fill_(0.001 if parameter.ndim > 1 else 0.01)
            with torch.no_grad():
                actual = generated(inputs)
                expected = reference(inputs)
            same = tuple(actual.shape) == tuple(expected.shape) and torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
            difference = float((actual - expected).abs().max()) if same else float("inf")
            update_check(current_index, same, f"shape={tuple(actual.shape)} max_abs_diff={difference:.3e}")
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["traning:finite_total_mse", "traning:backward", "traning:optimizer_update"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        spec = importlib.util.spec_from_file_location("deepcfd_sec_generated_training", CANDIDATE_DIR / "unetex.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {CANDIDATE_DIR / 'unetex.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["deepcfd_sec_generated_training"] = module
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(CANDIDATE_DIR))
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path
        model = module.UNetEx(**MODEL_CONFIG).cpu().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
        generator = torch.Generator().manual_seed(SEED + 3)
        inputs = torch.randn((1, 3, 32, 31), generator=generator)
        targets = torch.randn((1, 3, 32, 31), generator=generator)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = (predictions - targets).square().mean(dim=(0, 2, 3)).sum()
        loss.backward()
        gradients = [value.grad for value in model.parameters() if value.requires_grad]
        gradients_ok = bool(gradients) and all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
        ok = bool(torch.isfinite(loss))
        update_check(current_index, ok, f"loss={float(loss.detach())}")
        current_index += 1
        ok = gradients_ok
        update_check(current_index, ok, f"gradient_tensors={len(gradients)}")
        current_index += 1
        optimizer.step()
        changed = [name for name, value in model.named_parameters() if not torch.equal(before[name], value.detach())]
        ok = bool(changed)
        update_check(current_index, ok, f"changed_parameters={len(changed)}")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["artifacts:training_checkpoint_path", "artifacts:inference_checkpoint_path", "artifacts:metrics_path", "artifacts:materialized_metrics"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import yaml
        data = yaml.safe_load((CANDIDATE_DIR / "config.yaml").read_text(encoding="utf-8"))
        training = data.get("training", {})
        inference = data.get("inference", {})
        checkpoint = Path(str(training.get("output_dir", ""))) / str(training.get("checkpoint_name", ""))
        inference_checkpoint = Path(str(inference.get("checkpoint_path", "")))
        result_path = Path(str(inference.get("result_dir", ""))) / "metrics.json"
        normalized = lambda value: value.as_posix().lstrip("./")
        ok = normalized(checkpoint) == "weight/best_model.pt"
        update_check(current_index, ok, checkpoint)
        current_index += 1
        ok = normalized(inference_checkpoint) == "weight/best_model.pt"
        update_check(current_index, ok, inference_checkpoint)
        current_index += 1
        ok = normalized(result_path) == "result/deepcfd/metrics.json"
        update_check(current_index, ok, result_path)
        current_index += 1
        metrics_file = CANDIDATE_DIR / "result" / "deepcfd" / "metrics.json"
        if metrics_file.is_file():
            try:
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                values = {key: float(metrics[key]) for key in ("Ux_MSE", "Uy_MSE", "p_MSE", "Total_MSE")}
                ok = all(math.isfinite(value) for value in values.values()) and math.isclose(values["Total_MSE"], values["Ux_MSE"] + values["Uy_MSE"] + values["p_MSE"], rel_tol=1e-6, abs_tol=1e-8)
                msg = values
            except Exception as exc:
                ok = False
                msg = f"{type(exc).__name__}: {exc}"
        else:
            ok = True
            msg = "metrics.json absent; optional materialized check passed"
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks
