"""V2 checkpoint adapter for the DeepCFD training evaluator."""

import ast
import importlib.util
import inspect
import json
import math
import os
import pickle
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

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
MODEL_CONFIG = {"in_channels": 3, "out_channels": 3, "base_channels": 8, "num_stages": 4, "kernel_size": 5, "bilinear": True, "normtype": "none"}
EXPECTED_CHANNELS = (8, 16, 32, 64, 128)
EXPECTED_OUTPUT_SHAPE = (1, 3, 172, 79)
EXPECTED_TOTAL_SAMPLES = 981
EXPECTED_TRAIN_SAMPLES = 686
EXPECTED_TEST_SAMPLES = 295
DEEPCFD_DATA_DIR = Path("/public/share/sugonhpcapp01/onestore/onedatasets/DeepCFD/data")
FORWARD_CASE_COUNT = 6
SEED = 20260815
MODEL_INITIALIZATION_SEED = SEED + 10000
METRIC_KEYS = ("Ux_MSE", "Uy_MSE", "p_MSE", "Total_MSE")


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["required:unetex.py", "syntax:unetex.py", "required:dataset.py", "syntax:dataset.py", "required:train.py", "syntax:train.py", "required:inference.py", "syntax:inference.py", "required:config.yaml"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        for name in REQUIRED_FILES:
            path = CANDIDATE_DIR / name
            update_check(current_index, path.is_file(), str(path))
            current_index += 1
            if path.suffix == ".py":
                try:
                    text = path.read_text(encoding="utf-8")
                    ast.parse(text, filename=str(path))
                    ok = path.is_file() and bool(text.strip())
                    message = f"lines={len(text.splitlines())}"
                except Exception as exc:
                    ok = False
                    message = f"{type(exc).__name__}: {exc}"
                update_check(current_index, ok, message)
                current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_data_split(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["data:pickle_files", "data:data_directory", "data:split_ratio", "data:split_seed", "data:split_counts", "data:dataloader_settings", "data:tensor_contract", "data:deterministic_random_split", "data:pickle_tensor_loader", "data:channel_rms"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import yaml
        value = yaml.safe_load((CANDIDATE_DIR / "config.yaml").read_text(encoding="utf-8"))
        value = value if isinstance(value, Mapping) else {}
        data = value.get("data", {}) if isinstance(value.get("data", {}), Mapping) else {}
        dataset = value.get("dataset", {}) if isinstance(value.get("dataset", {}), Mapping) else {}
        def get(paths, default=None):
            for path in paths:
                item = value
                try:
                    for part in path.split("."):
                        item = item[part]
                    return path, item
                except (KeyError, TypeError):
                    pass
            return None, default
        _, directory = get(("data.directory", "data.data_dir", "dataset.directory", "dataset.data_dir"))
        _, input_file = get(("data.input_file", "data.data_x_name", "dataset.input_file"))
        _, target_file = get(("data.target_file", "data.data_y_name", "dataset.target_file"))
        update_check(current_index, Path(str(input_file)).name == "dataX.pkl" and Path(str(target_file)).name == "dataY.pkl", f"directory={directory!r} input={input_file!r} target={target_file!r}")
        current_index += 1
        update_check(current_index, directory is not None and Path(str(directory)).name == "data", f"configured={directory!r} expected=./data")
        current_index += 1
        ratio_key, ratio = get(("data.split_ratio", "dataset.split_ratio"))
        seed_key, seed = get(("data.seed", "dataset.seed", "seed"))
        batch_key, batch = get(("data.batch_size", "dataloader.batch_size", "training.batch_size"))
        worker_key, workers = get(("data.num_workers", "dataloader.num_workers", "training.num_workers"))
        ratio = float(ratio)
        seed = float(seed)
        update_check(current_index, math.isclose(ratio, 0.7, abs_tol=1e-12), f"{ratio_key}={ratio}")
        current_index += 1
        update_check(current_index, math.isclose(seed, 0.0, abs_tol=1e-12), f"{seed_key}={seed}")
        current_index += 1
        split = int(EXPECTED_TOTAL_SAMPLES * ratio)
        update_check(current_index, split == EXPECTED_TRAIN_SAMPLES and EXPECTED_TOTAL_SAMPLES - split == EXPECTED_TEST_SAMPLES, f"total={EXPECTED_TOTAL_SAMPLES} train={split} test={EXPECTED_TOTAL_SAMPLES - split}")
        current_index += 1
        update_check(current_index, float(batch) == 16.0 and float(workers) == 0.0, f"{batch_key}={batch} {worker_key}={workers}")
        current_index += 1
        shape = get(("data.sample_shape", "dataset.sample_shape"), (3, 172, 79))[1]
        model = value.get("model", {}) if isinstance(value.get("model", {}), Mapping) else {}
        shape_ok = tuple(int(item) for item in shape) == (3, 172, 79) and model.get("in_channels") == 3 and model.get("out_channels") == 3
        update_check(current_index, shape_ok, f"sample_shape={shape} model_channels=({model.get('in_channels')}, {model.get('out_channels')})")
        current_index += 1
        source = (CANDIDATE_DIR / "dataset.py").read_text(encoding="utf-8")
        update_check(current_index, "random.Random(" in source and ".shuffle(" in source, "requires random.Random(seed).shuffle(indices)")
        current_index += 1
        combined = "\n".join((CANDIDATE_DIR / name).read_text(encoding="utf-8") for name in ("dataset.py", "train.py", "engine.py", "losses.py", "training_utils.py") if (CANDIDATE_DIR / name).is_file())
        update_check(current_index, "pickle" in source.lower() and ("float32" in source.lower() or "torch.from_numpy" in source.lower() or "torch.as_tensor" in source.lower()) and "dataloader" in source.lower(), "pickle -> float32 tensor -> DataLoader")
        current_index += 1
        lower = combined.lower()
        update_check(current_index, "sqrt" in lower and "mean" in lower and ("**2" in combined or "** 2" in combined or ".square(" in combined), "sqrt(mean(train_targets ** 2)) by channel")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["train_script:config_cli", "train_script:adamw", "train_script:optimizer_hyperparameters", "train_script:schedule", "train_script:optimization_step", "train_script:validation_and_metrics", "train_script:best_checkpoint_and_early_stop", "train_script:rms_weighted_loss"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        import yaml
        value = yaml.safe_load((CANDIDATE_DIR / "config.yaml").read_text(encoding="utf-8"))
        value = value if isinstance(value, Mapping) else {}
        source = (CANDIDATE_DIR / "train.py").read_text(encoding="utf-8")
        combined = source + "\n" + "\n".join((CANDIDATE_DIR / name).read_text(encoding="utf-8") for name in ("engine.py", "losses.py", "training_utils.py") if (CANDIDATE_DIR / name).is_file())
        lower = combined.lower()
        training = value.get("training", {}) if isinstance(value.get("training", {}), Mapping) else {}
        optimizer = value.get("optimizer", {}) if isinstance(value.get("optimizer", {}), Mapping) else {}
        update_check(current_index, "argparse" in source.lower() and "--config" in source, "python train.py --config config.yaml")
        current_index += 1
        update_check(current_index, "adamw" in lower and str(optimizer.get("name", training.get("optimizer", "AdamW"))).lower() == "adamw", "optimizer=AdamW")
        current_index += 1
        update_check(current_index, float(optimizer.get("learning_rate", training.get("lr"))) == 0.001 and float(optimizer.get("weight_decay", training.get("weight_decay"))) == 0.005, "lr=0.001 weight_decay=0.005")
        current_index += 1
        update_check(current_index, training.get("num_epochs") == 1000 and training.get("eval_interval") == 10 and training.get("patience") == 200, "epochs=1000 interval=10 patience=200")
        current_index += 1
        update_check(current_index, ".backward(" in lower and "optimizer.step(" in lower and "zero_grad" in lower, "zero_grad -> backward -> optimizer.step")
        current_index += 1
        update_check(current_index, "val_loss" in lower and "ux" in lower and "uy" in lower and ("channel_errors" in lower or "total_mse" in lower or "p_mse" in lower), "val_loss plus Total/Ux/Uy/p metrics")
        current_index += 1
        update_check(current_index, "val_loss" in lower and "torch.save" in lower and "patience" in lower and ("early stop" in lower or "early_stopping" in lower or "earlystop" in lower), "best checkpoint and early stopping")
        current_index += 1
        spec = importlib.util.spec_from_file_location("deepcfd_submission_train", CANDIDATE_DIR / "train.py")
        if spec is None or spec.loader is None:
            raise ImportError("could not load train.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["deepcfd_submission_train"] = module
        sys.path.insert(0, str(CANDIDATE_DIR))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        function = next((getattr(module, name, None) for name in ("loss_func", "compute_loss", "deepcfd_loss") if callable(getattr(module, name, None))), None)
        update_check(current_index, callable(function), "loss_func/compute_loss/deepcfd_loss")
        current_index += 1
        if function is None:
            raise AttributeError("loss callable is missing")
        predictions = torch.tensor([[[[1.0, -2.0], [0.5, 3.0]], [[-1.0, 2.0], [4.0, -0.5]], [[3.0, -2.0], [1.0, -4.0]]], [[[-0.5, 1.5], [2.0, -1.0]], [[2.5, -3.0], [1.0, 0.5]], [[-1.0, 2.0], [-3.0, 0.25]]]], dtype=torch.float64)
        targets = torch.zeros_like(predictions)
        rms = torch.tensor([2.0, 4.0, 5.0], dtype=torch.float64)
        actual = function(predictions, targets, rms)
        errors = predictions - targets
        expected = torch.sum(torch.stack((errors[:, 0].square(), errors[:, 1].square(), errors[:, 2].abs()), dim=1) / rms.view(1, 3, 1, 1))
        update_check(current_index, isinstance(actual, torch.Tensor) and actual.numel() == 1 and torch.allclose(actual, expected, rtol=1e-10, atol=1e-10), f"actual={actual} expected={expected}")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["eval_script:config_cli", "eval_script:metric_helper_called", "eval_script:sample_normalized_grid_mse", "eval_script:complete_test_inference", "eval_script:checkpoint_eval_mode", "eval_script:metrics_schema_and_persistence"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        path = CANDIDATE_DIR / "inference.py"
        if not path.is_file():
            for _ in CHECKPOINT_NAMES:
                update_check(current_index, True, "inference.py absent; optional check passed")
                current_index += 1
        else:
            import torch
            source = path.read_text(encoding="utf-8")
            lower = source.lower()
            spec = importlib.util.spec_from_file_location("deepcfd_submission_inference", path)
            if spec is None or spec.loader is None:
                raise ImportError("could not load inference.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules["deepcfd_submission_inference"] = module
            sys.path.insert(0, str(CANDIDATE_DIR))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.pop(0)
            metric = getattr(module, "compute_sample_normalized_grid_mse", None)
            update_check(current_index, "argparse" in lower and "--config" in source, "python inference.py --config config.yaml")
            current_index += 1
            tree = ast.parse(source)
            called = any(isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == "compute_sample_normalized_grid_mse") or (isinstance(node.func, ast.Attribute) and node.func.attr == "compute_sample_normalized_grid_mse")) for node in ast.walk(tree))
            update_check(current_index, called, "compute_sample_normalized_grid_mse is called")
            current_index += 1
            if not callable(metric):
                raise AttributeError("metric helper is missing")
            predictions = torch.randn((3, 3, 2, 5), generator=torch.Generator().manual_seed(SEED))
            targets = torch.randn((3, 3, 2, 5), generator=torch.Generator().manual_seed(SEED + 1))
            actual = metric(predictions, targets)
            actual = {key: float(actual[key]) for key in METRIC_KEYS} if isinstance(actual, Mapping) else dict(zip(METRIC_KEYS, actual))
            expected_values = (predictions - targets).square().mean(dim=(0, 2, 3))
            expected = {"Ux_MSE": float(expected_values[0]), "Uy_MSE": float(expected_values[1]), "p_MSE": float(expected_values[2]), "Total_MSE": float(expected_values.sum())}
            update_check(current_index, all(math.isclose(float(actual[key]), expected[key], rel_tol=1e-6, abs_tol=1e-8) for key in METRIC_KEYS), f"actual={actual} expected={expected}")
            current_index += 1
            update_check(current_index, any(term in lower for term in ("test_loader", "test_dataloader", "test_dataset")) and any(term in lower for term in ("torch.cat", "accumul", "sample_count", "all_pred")), "complete test accumulation")
            current_index += 1
            update_check(current_index, "load_state_dict" in lower and ".eval(" in lower and "no_grad" in lower, "checkpoint restore, eval, no_grad")
            current_index += 1
            update_check(current_index, all(key.lower() in lower for key in METRIC_KEYS) and ("json.dump" in lower or "write_text" in lower), "metrics.json contains all required keys")
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["model_structure:required_classes", "model_structure:unetex_constructor", "model_structure:shared_encoder", "model_structure:encoder_registration", "model_structure:encoder_channel_start", "model_structure:down_stages", "model_structure:double_conv_exact_sequence", "model_structure:three_independent_decoder_paths", "model_structure:decoder_up_stages_and_heads", "model_structure:decoder_parameters_not_shared", "model_structure:reference_state_dict_keys_and_shapes", "model_structure:double_conv_checkpoint_keys"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch.nn as nn
        def load(path, name):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            sys.path.insert(0, str(path.parent))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.pop(0)
            return module
        generated = load(CANDIDATE_DIR / "unetex.py", "deepcfd_submission_structure")
        reference = load(REFERENCE_DIR / "unetex.py", "deepcfd_reference_structure")
        model = generated.UNetEx(**MODEL_CONFIG)
        reference_model = reference.UNetEx(**MODEL_CONFIG)
        missing = [name for name in MODEL_CLASSES if not inspect.isclass(getattr(generated, name, None)) or not issubclass(getattr(generated, name), nn.Module)]
        update_check(current_index, not missing, f"missing={missing}")
        current_index += 1
        params = inspect.signature(generated.UNetEx.__init__).parameters
        update_check(current_index, all(name in params for name in MODEL_CONFIG), f"parameters={list(params)}")
        current_index += 1
        encoder = getattr(model, "encoder", None)
        update_check(current_index, isinstance(encoder, generated.UNetEncoder2D), f"encoder={type(encoder).__name__}")
        current_index += 1
        stages = getattr(encoder, "down_stages", None)
        update_check(current_index, isinstance(stages, nn.ModuleList) and len(stages) == 4 and isinstance(getattr(encoder, "inc", None), generated.DoubleConv2D), "encoder ModuleList registration")
        current_index += 1
        inc = getattr(encoder, "inc", None)
        layers = list(getattr(inc, "double_conv", ()))
        update_check(current_index, len(layers) == 4 and isinstance(layers[0], nn.Conv2d) and isinstance(layers[1], nn.ReLU) and isinstance(layers[2], nn.Conv2d) and isinstance(layers[3], nn.ReLU) and (layers[0].in_channels, layers[0].out_channels, layers[2].out_channels) == (3, 8, 8), "encoder channels 3->8->8")
        current_index += 1
        down_ok = isinstance(stages, nn.ModuleList) and len(stages) == 4
        for stage, left, right in zip(stages if down_ok else (), EXPECTED_CHANNELS[:-1], EXPECTED_CHANNELS[1:]):
            children = list(getattr(stage, "maxpool_conv", ()))
            block = children[1] if len(children) == 2 else None
            block_layers = list(getattr(block, "double_conv", ()))
            down_ok = down_ok and len(children) == 2 and isinstance(children[0], nn.MaxPool2d) and len(block_layers) == 4 and isinstance(block_layers[0], nn.Conv2d) and block_layers[0].in_channels == left and block_layers[0].out_channels == right and block_layers[2].out_channels == right
        update_check(current_index, down_ok, "four down stages 8->16->32->64->128")
        current_index += 1
        double_ok = True
        for block in model.modules():
            if isinstance(block, generated.DoubleConv2D):
                items = list(getattr(block, "double_conv", ()))
                double_ok = double_ok and len(items) == 4 and isinstance(items[0], nn.Conv2d) and isinstance(items[1], nn.ReLU) and isinstance(items[2], nn.Conv2d) and isinstance(items[3], nn.ReLU) and items[0].bias is None and items[2].bias is None and items[0].kernel_size == (5, 5) and items[0].padding == (2, 2)
        update_check(current_index, double_ok, "Conv-ReLU-Conv-ReLU, kernel=5, bias=False")
        current_index += 1
        decoders = getattr(model, "decoders", None)
        decoder_ok = isinstance(decoders, nn.ModuleList) and len(decoders) == 3 and len({id(item) for item in decoders}) == 3
        update_check(current_index, decoder_ok, "three independent DecoderPath modules")
        current_index += 1
        paths_ok = decoder_ok and all(isinstance(getattr(path, "decoder", None), generated.UNetDecoder2D) and len(getattr(getattr(path, "decoder", None), "up_stages", ())) == 4 and getattr(getattr(getattr(path, "head", None), "conv", None), "out_channels", None) == 1 for path in decoders)
        update_check(current_index, paths_ok, "decoder stages and one-channel heads")
        current_index += 1
        ids = [{id(parameter) for parameter in path.parameters()} for path in decoders] if decoder_ok else []
        independent = len(ids) == 3 and all(ids[a].isdisjoint(ids[b]) for a in range(3) for b in range(a + 1, 3))
        update_check(current_index, independent, "decoder parameter sets are disjoint")
        current_index += 1
        state, ref_state = model.state_dict(), reference_model.state_dict()
        update_check(current_index, tuple(state) == tuple(ref_state) and all(state[key].shape == ref_state[key].shape for key in ref_state), "reference state-dict keys and shapes")
        current_index += 1
        bias_keys = [key for key in state if "double_conv" in key and key.endswith(".bias")]
        update_check(current_index, not bias_keys and any("double_conv.0.weight" in key for key in state) and any("double_conv.2.weight" in key for key in state), f"unexpected_bias_keys={bias_keys}")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_executable:real_data_cases", "forward_executable:batch_output_shapes", "forward_executable:finite_batch_outputs"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import numpy as np
        import torch
        spec = importlib.util.spec_from_file_location("deepcfd_submission_forward", CANDIDATE_DIR / "unetex.py")
        if spec is None or spec.loader is None:
            raise ImportError("could not load unetex.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["deepcfd_submission_forward"] = module
        sys.path.insert(0, str(CANDIDATE_DIR))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        model = module.UNetEx(**MODEL_CONFIG).cpu().eval()
        with (DEEPCFD_DATA_DIR / "dataX.pkl").open("rb") as stream:
            values = np.asarray(pickle.load(stream), dtype=np.float32)
        indices = list(range(len(values)))
        random.Random(0).shuffle(indices)
        test = indices[int(len(indices) * 0.7):]
        positions = [round(index * (len(test) - 1) / 5) for index in range(6)]
        inputs = torch.from_numpy(np.ascontiguousarray(values[[test[position] for position in positions]], dtype=np.float32))
        with torch.no_grad():
            output = model(inputs)
        expected = (6, 3, 172, 79)
        update_check(current_index, len(positions) == 6 and len(set(positions)) == 6 and tuple(inputs.shape) == expected, f"shape={tuple(inputs.shape)}")
        current_index += 1
        update_check(current_index, tuple(output.shape) == expected and all(tuple(output[index:index + 1].shape) == EXPECTED_OUTPUT_SHAPE for index in range(6)), f"actual={tuple(output.shape)} expected={expected}")
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output).all()), "all generated output values must be finite")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_equivalence:deterministic_parameter_initialization", "forward_equivalence:case_1:values", "forward_equivalence:case_2:values", "forward_equivalence:case_3:values", "forward_equivalence:case_4:values", "forward_equivalence:case_5:values", "forward_equivalence:case_6:values"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import numpy as np
        import torch
        def load(path, name):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            sys.path.insert(0, str(path.parent))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.pop(0)
            return module
        generated = load(CANDIDATE_DIR / "unetex.py", "deepcfd_submission_equivalence")
        reference = load(REFERENCE_DIR / "unetex.py", "deepcfd_reference_equivalence")
        actual_model = generated.UNetEx(**MODEL_CONFIG).cpu().eval()
        expected_model = reference.UNetEx(**MODEL_CONFIG).cpu().eval()
        with torch.no_grad():
            for model in (actual_model, expected_model):
                for parameter in model.parameters():
                    parameter.fill_(0.001 if parameter.ndim > 1 else 0.01)
        actual_parameters = dict(actual_model.named_parameters())
        expected_parameters = dict(expected_model.named_parameters())
        same = set(actual_parameters) == set(expected_parameters) and all(actual_parameters[name].shape == expected_parameters[name].shape and torch.equal(actual_parameters[name], expected_parameters[name]) for name in actual_parameters)
        update_check(current_index, same, "identical deterministic parameter sets")
        current_index += 1
        with (DEEPCFD_DATA_DIR / "dataX.pkl").open("rb") as stream:
            values = np.asarray(pickle.load(stream), dtype=np.float32)
        indices = list(range(len(values)))
        random.Random(0).shuffle(indices)
        test = indices[int(len(indices) * 0.7):]
        positions = [round(index * (len(test) - 1) / 5) for index in range(6)]
        inputs = torch.from_numpy(np.ascontiguousarray(values[[test[position] for position in positions]], dtype=np.float32))
        with torch.no_grad():
            actual = actual_model(inputs)
            expected = expected_model(inputs)
        for index in range(6):
            left, right = actual[index:index + 1], expected[index:index + 1]
            ok = tuple(left.shape) == tuple(right.shape) == EXPECTED_OUTPUT_SHAPE and torch.allclose(left, right, atol=1e-6, rtol=1e-6)
            update_check(current_index, ok, f"case={index + 1}")
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["traning:initialization_mode", "traning:finite_loss", "traning:finite_gradient_coverage", "traning:parameter_update_scope"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        spec = importlib.util.spec_from_file_location("deepcfd_submission_training", CANDIDATE_DIR / "unetex.py")
        if spec is None or spec.loader is None:
            raise ImportError("could not load unetex.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["deepcfd_submission_training"] = module
        sys.path.insert(0, str(CANDIDATE_DIR))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        model = module.UNetEx(**MODEL_CONFIG).cpu().train()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.001 if parameter.ndim > 1 else 0.01)
        update_check(current_index, True, "from_scratch: deterministic direct initialization")
        current_index += 1
        named = list(model.named_parameters())
        trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in named if not parameter.requires_grad]
        optimizer = torch.optim.AdamW((parameter for _, parameter in trainable), lr=0.001, weight_decay=0.005)
        generator = torch.Generator(device="cpu").manual_seed(SEED + 500)
        inputs = 0.5 + 0.1 * torch.randn((1, 3, 32, 33), generator=generator)
        targets = 0.1 * torch.randn((1, 3, 32, 33), generator=generator)
        rms = torch.tensor([1.1, 0.9, 1.3]).view(1, 3, 1, 1)
        before = {name: parameter.detach().clone() for name, parameter in named}
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        errors = predictions - targets
        loss = torch.sum(torch.stack((errors[:, 0].square(), errors[:, 1].square(), errors[:, 2].abs()), dim=1) / rms)
        loss.backward()
        missing = [name for name, parameter in trainable if parameter.grad is None]
        nonfinite = [name for name, parameter in trainable if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
        optimizer.step()
        changed = [name for name, parameter in trainable if not parameter.detach().equal(before[name])]
        frozen_changed = [name for name, parameter in frozen if not parameter.detach().equal(before[name])]
        update_check(current_index, bool(torch.isfinite(loss)), f"loss={float(loss.detach())}")
        current_index += 1
        update_check(current_index, not missing and not nonfinite, f"missing={missing} nonfinite={nonfinite}")
        current_index += 1
        update_check(current_index, len(changed) == len(trainable) and not frozen_changed, f"changed_trainable={len(changed)}/{len(trainable)} changed_frozen={frozen_changed}")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_infer_output_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["artifacts:checkpoint_path", "artifacts:metrics_path", "artifacts:checkpoint_if_materialized", "artifacts:metrics_if_materialized"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import yaml
        value = yaml.safe_load((CANDIDATE_DIR / "config.yaml").read_text(encoding="utf-8"))
        value = value if isinstance(value, Mapping) else {}
        def get(path, default=None):
            item = value
            try:
                for part in path.split("."):
                    item = item[part]
                return item
            except (KeyError, TypeError):
                return default
        declared_checkpoint = str(get("training.best_checkpoint_path", get("training.checkpoint_path", get("inference.checkpoint_path", "")))).replace("./", "").lstrip("/")
        declared_metrics = str(get("inference.metrics_path", "")).replace("./", "").lstrip("/")
        update_check(current_index, declared_checkpoint == "weight/best_model.pt", f"declared={declared_checkpoint}")
        current_index += 1
        update_check(current_index, declared_metrics == "result/deepcfd/metrics.json", f"declared={declared_metrics}")
        current_index += 1
        checkpoint = CANDIDATE_DIR / "weight/best_model.pt"
        checkpoint_ok = True
        detail = "not materialized; full training is outside dynamic evaluation"
        if checkpoint.is_file():
            import torch
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload
            if isinstance(payload, Mapping):
                state = next((payload[key] for key in ("model_state", "model_state_dict", "state_dict") if isinstance(payload.get(key), Mapping)), payload)
            checkpoint_ok = isinstance(state, Mapping) and bool(state)
            detail = f"materialized entries={len(state) if checkpoint_ok else 0}"
        update_check(current_index, checkpoint_ok, detail)
        current_index += 1
        metrics_path = CANDIDATE_DIR / "result/deepcfd/metrics.json"
        metrics_ok = True
        detail = "not materialized; inference execution is outside this evaluator"
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = {key: float(payload[key]) for key in METRIC_KEYS}
            metrics_ok = all(math.isfinite(item) for item in metrics.values()) and math.isclose(metrics["Total_MSE"], metrics["Ux_MSE"] + metrics["Uy_MSE"] + metrics["p_MSE"], rel_tol=1e-6, abs_tol=1e-8)
            detail = f"materialized metrics={metrics}"
        update_check(current_index, metrics_ok, detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks
