"""V2 adapter for the multiscale KNO legacy evaluator."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Dict, Iterable, Mapping

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

REQUIRED_FILES = ("model.py", "dataset.py", "utils.py", "train.py", "config.yaml")
PYTHON_FILES = REQUIRED_FILES[:-1]
SUBMISSION_NAMES = (
    "test_kno_sec_develop",
    "test_kno_sec_develop_oneskills",
    "kno_sec_develop",
    "test_kno_train",
)
CHECK_TYPES = {
    "file",
    "data",
    "train_script",
    "eval_script",
    "model_structure",
    "forward_executable",
    "forward_equivalence",
    "traning",
    "artifacts",
}
REQUIRED_CHECK_TYPES = {
    "file",
    "data",
    "train_script",
    "model_structure",
    "forward_executable",
    "traning",
}
OPTIONAL_CHECK_TYPES = CHECK_TYPES - REQUIRED_CHECK_TYPES
DATA_FILE = "NavierStokes_V1e-5_N1200_T20.mat"
CHECKPOINT_KEYS = {
    "model_state_dict",
    "model_config",
    "data_config",
    "spatial_shape",
    "epoch",
    "best_relative_l2",
}
SEED = 20260824


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _source(directory: Path, names: Iterable[str]) -> str:
    return "\n".join(_read(directory / name) for name in names if (directory / name).is_file())


def _compact(source: str) -> str:
    return "".join(source.lower().split())


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(_read(path))
    if not isinstance(value, dict):
        raise TypeError("config.yaml must contain a mapping")
    root = value.get("root")
    return root if isinstance(root, dict) else value


def _find_value(value: Any, names: Iterable[str]) -> Any:
    wanted = set(names)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in wanted:
                return item
        for item in value.values():
            found = _find_value(item, wanted)
            if found is not None:
                return found
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_model(
    module: Any,
    spatial_shape: tuple[int, int] = (64, 64),
    reduced: bool = False,
) -> Any:
    model_class = getattr(module, "KNO2DNavierStokes", None)
    if not inspect.isclass(model_class):
        raise TypeError("model.py must define class KNO2DNavierStokes")
    op_size, low_modes, high_modes = (8, 4, 8) if reduced else (32, 12, 24)
    values = {
        "input_channels": 10,
        "in_channels": 10,
        "output_channels": 1,
        "out_channels": 1,
        "spatial_shape": spatial_shape,
        "op_size": op_size,
        "low_modes_x": low_modes,
        "low_modes_y": low_modes,
        "high_modes_x": high_modes,
        "high_modes_y": high_modes,
        "modes_x": low_modes,
        "modes_y": low_modes,
        "decompose": 6,
        "evolution_steps": 6,
        "linear_type": True,
        "normalization": False,
        "use_batch_norm": False,
        "gate_init": 0.0,
    }
    signature = inspect.signature(model_class)
    kwargs = {name: values[name] for name in signature.parameters if name in values}
    return model_class(**kwargs)


def _model_forward(model: Any, pos: Any, fields: Any) -> Any:
    parameters = list(inspect.signature(model.forward).parameters)
    if parameters and parameters[0].lower() in {"fx", "fields", "x", "inputs", "state"}:
        return model(fields, pos)
    return model(pos, fields)


def _random_inputs(torch: Any, shape: tuple[int, int]) -> tuple[Any, Any]:
    height, width = shape
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    fields = torch.randn((1, height * width, 10), generator=generator)
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    pos = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1).unsqueeze(0)
    return pos, fields


def _curriculum_horizons(config: Mapping[str, Any]) -> list[int]:
    value = _find_value(config, {"horizons"})
    if isinstance(value, list):
        return [int(item) for item in value]
    stages = _find_value(config, {"stages", "curriculum"})
    if isinstance(stages, list):
        return [int(stage["horizon"]) for stage in stages if isinstance(stage, Mapping) and "horizon" in stage]
    return []


def _evolution_steps(model: Any) -> list[int]:
    values: list[int] = []
    for module in model.modules():
        for name in ("decompose", "evolution_steps", "num_steps"):
            value = getattr(module, name, None)
            if isinstance(value, int):
                values.append(value)
    return values


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint is not a mapping")
    return value


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "file:model.py", "file:dataset.py", "file:utils.py", "file:train.py", "file:config.yaml",
        "file:syntax:model.py", "file:syntax:dataset.py", "file:syntax:utils.py", "file:syntax:train.py",
        "file:config_yaml",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        for name in REQUIRED_FILES:
            path = CANDIDATE_DIR / name
            update_check(current_index, path.is_file(), str(path))
            current_index += 1
        for name in PYTHON_FILES:
            path = CANDIDATE_DIR / name
            if path.is_file():
                try:
                    ast.parse(_read(path), filename=str(path))
                    update_check(current_index, True, "valid Python syntax")
                except SyntaxError as exc:
                    update_check(current_index, False, str(exc))
            current_index += 1
        config_path = CANDIDATE_DIR / "config.yaml"
        if config_path.is_file():
            try:
                _load_config(config_path)
                update_check(current_index, True, "valid YAML mapping")
            except Exception as exc:
                update_check(current_index, False, f"{type(exc).__name__}: {exc}")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "data:source", "data:split", "data:shape_contract", "data:dynamic_windows",
        "data:all_window_indices", "data:flattened_interface", "data:loader_policy",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        reference_dir = REFERENCE_DIR
        del reference_dir
        source = _source(CANDIDATE_DIR, ("dataset.py", "config.yaml"))
        compact = _compact(source)
        config = _load_config(CANDIDATE_DIR / "config.yaml")
        ntrain = _as_float(_find_value(config, {"ntrain", "train_samples"}))
        ntest = _as_float(_find_value(config, {"ntest", "nval", "val_samples", "test_samples"}))
        history = _as_float(_find_value(config, {"t_in", "history", "input_steps"}))
        horizon = _as_float(_find_value(config, {"max_horizon", "t_out", "horizon", "output_steps"}))
        batch = _as_float(_find_value(config, {"batch_size"}))
        window_formula = (
            "-self.t_in-horizon+1" in compact
            or "-self.history-horizon+1" in compact
            or "-t_in-horizon+1" in compact
            or "-input_steps-horizon+1" in compact
        )
        ok = DATA_FILE.lower() in source.lower() and "loadmat" in compact and ("['u']" in compact or '["u"]' in compact)
        update_check(current_index, ok, DATA_FILE)
        current_index += 1
        ok = ntrain == 1000 and ntest == 200 and "[:ntrain" in compact and "[-ntest" in compact
        update_check(current_index, ok, f"ntrain={ntrain}, validation={ntest}")
        current_index += 1
        ok = history == 10 and horizon == 10 and batch == 8
        update_check(current_index, ok, f"history={history}, max_horizon={horizon}, batch={batch}")
        current_index += 1
        ok = "set_horizon" in compact and "windows_per_trajectory" in compact and window_formula
        update_check(current_index, ok, "continuous windows depend on the active horizon")
        current_index += 1
        ok = "divmod(" in compact and "windows_per_trajectory" in compact
        update_check(current_index, ok, "trajectory and start are derived from every valid index")
        current_index += 1
        ok = "reshape(" in compact and "pos" in compact and "meshgrid" in compact
        update_check(current_index, ok, "returns flattened fields and [N,2] coordinates")
        current_index += 1
        ok = "shuffle=true" in compact and "shuffle=false" in compact
        update_check(current_index, ok, "training shuffled; validation deterministic")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "train_script:optimizer", "train_script:scheduler", "train_script:duration", "train_script:curriculum",
        "train_script:teacher_forcing_decay", "train_script:scheduled_sampling", "train_script:composite_loss",
        "train_script:autoregressive_validation", "train_script:legacy_initialization", "train_script:checkpoint_schema",
        "train_script:logging",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        reference_dir = REFERENCE_DIR
        del reference_dir
        source = _source(CANDIDATE_DIR, ("train.py", "utils.py", "model.py", "config.yaml"))
        compact = _compact(source)
        config = _load_config(CANDIDATE_DIR / "config.yaml")
        epochs = _as_float(_find_value(config, {"epochs", "num_epochs"}))
        lr = _as_float(_find_value(config, {"learning_rate", "lr"}))
        decay = _as_float(_find_value(config, {"weight_decay"}))
        step_size = _as_float(_find_value(config, {"scheduler_step_size", "step_size"}))
        gamma = _as_float(_find_value(config, {"scheduler_gamma", "gamma"}))
        horizons = _curriculum_horizons(config)
        tf_start = _as_float(_find_value(config, {"teacher_forcing_start", "teacher_forcing_ratio_start"}))
        tf_end = _as_float(_find_value(config, {"teacher_forcing_end", "teacher_forcing_ratio_end"}))
        if tf_start is None or tf_end is None:
            curriculum = _find_value(config, {"stages"})
            ratios = [
                float(stage.get("teacher_forcing_ratio"))
                for stage in curriculum or []
                if isinstance(stage, Mapping) and stage.get("teacher_forcing_ratio") is not None
            ]
            if ratios:
                tf_start, tf_end = ratios[0], ratios[-1]
        checkpoint_schema = all(key.lower() in source.lower() for key in CHECKPOINT_KEYS)
        ok = "adamw" in compact and lr == 1.0e-3 and decay == 1.0e-5
        update_check(current_index, ok, f"AdamW lr={lr}, weight_decay={decay}")
        current_index += 1
        ok = "steplr" in compact and step_size == 100 and gamma == 0.5
        update_check(current_index, ok, f"StepLR step_size={step_size}, gamma={gamma}")
        current_index += 1
        update_check(current_index, epochs == 500, f"epochs={epochs}")
        current_index += 1
        ok = horizons == [1, 2, 5, 10] and "set_train_horizon" in compact
        update_check(current_index, ok, f"horizons={horizons}")
        current_index += 1
        ok = tf_start is not None and tf_end is not None and tf_start > tf_end and "teacher_forcing" in compact
        update_check(current_index, ok, f"start={tf_start}, end={tf_end}")
        current_index += 1
        ok = "torch.where(" in compact and "torch.rand(" in compact and "torch.cat(" in compact
        update_check(current_index, ok, "per-sample teacher forcing with autoregressive feedback")
        current_index += 1
        ok = "mse" in compact and "relative_l2" in compact and "spectral" in compact and ("far_step_weight" in compact or "time_weights" in compact)
        update_check(current_index, ok, "MSE + Relative L2 + spectral loss with far-step weighting")
        current_index += 1
        ok = ("teacher_forcing_ratio=0" in compact or ",0.0,training=false" in compact or ",0,training=false" in compact) and ("training=false" in compact or "model.eval()" in compact)
        update_check(current_index, ok, "validation rolls out without teacher forcing")
        current_index += 1
        ok = "init_checkpoint" in compact and "strict=false" in compact and "high" in compact and "missing" in compact
        update_check(current_index, ok, "legacy weights initialize the low-frequency branch")
        current_index += 1
        update_check(current_index, checkpoint_schema, f"required keys={sorted(CHECKPOINT_KEYS)}")
        current_index += 1
        ok = "train.log" in compact and ("history.json" in compact or "metrics" in compact)
        update_check(current_index, ok, "training log and metric history are declared")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "eval_script:optional", "eval_script:checkpoint", "eval_script:inference_mode", "eval_script:rollout_metric",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        reference_dir = REFERENCE_DIR
        del reference_dir
        paths = [CANDIDATE_DIR / name for name in ("eval.py", "evaluate.py", "inference.py")]
        path = next((item for item in paths if item.is_file()), None)
        if path is None:
            update_check(current_index, True, "no standalone evaluation script required")
            current_index += 1
        else:
            current_index += 1
            compact = _compact(_read(path))
            ok = "torch.load(" in compact and "model_state_dict" in compact
            update_check(current_index, ok, path.name)
            current_index += 1
            ok = "model.eval()" in compact and ("no_grad" in compact or "inference_mode" in compact)
            update_check(current_index, ok, path.name)
            current_index += 1
            ok = "relative_l2" in compact and ("rollout" in compact or "torch.cat(" in compact)
            update_check(current_index, ok, path.name)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
        current_index += 1

    checks = checks[:current_index]
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "model_structure:classes", "model_structure:low_branch", "model_structure:high_branch",
        "model_structure:learnable_gate", "model_structure:fourier_evolution", "model_structure:frequency_separation",
        "model_structure:pointwise_convs", "model_structure:residual_path", "model_structure:no_batchnorm",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        from torch import nn

        generated_module = _load_module("kno_sec_generated_model", CANDIDATE_DIR / "model.py")
        model = _build_model(generated_module).cpu().eval()
        source = _compact(inspect.getsource(generated_module))
        complex_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.is_complex()]
        low = [
            (name, parameter)
            for name, parameter in complex_parameters
            if tuple(parameter.shape) == (32, 32, 12, 12) and "high" not in name.lower()
        ]
        high = [
            (name, parameter)
            for name, parameter in complex_parameters
            if parameter.ndim == 4 and tuple(parameter.shape[:2]) == (32, 32)
            and tuple(parameter.shape[-2:]) != (12, 12) and min(parameter.shape[-2:]) > 12
        ]
        gates = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if "gate" in name.lower() and parameter.requires_grad
        ]
        convs = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        required_classes = ("EncoderConv2D", "DecoderConv2D", "MultiScaleKoopmanOperator2D", "KNO2D", "KNO2DNavierStokes")
        ok = all(inspect.isclass(getattr(generated_module, name, None)) for name in required_classes)
        update_check(current_index, ok, str(required_classes))
        current_index += 1
        ok = bool(low) and "koopman_matrix" in low[0][0]
        update_check(current_index, ok, str([f"{name}:{tuple(value.shape)}" for name, value in low]))
        current_index += 1
        update_check(current_index, bool(high), str([f"{name}:{tuple(value.shape)}" for name, value in high]))
        current_index += 1
        ok = bool(gates) and "sigmoid(" in source
        update_check(current_index, ok, str([f"{name}:{tuple(value.shape)}" for name, value in gates]))
        current_index += 1
        ok = "rfft2(" in source and "irfft2(" in source and 'einsum("btxy,tfxy->bfxy"' in source
        update_check(current_index, ok, "rfft2/irfft2 and required einsum")
        current_index += 1
        ok = "low" in source and "high" in source and ("mask" in source or "band" in source)
        update_check(current_index, ok, "parallel low/high frequency bands")
        current_index += 1
        ok = bool(convs) and all(module.kernel_size == (1, 1) for module in convs)
        update_check(current_index, ok, f"Conv2d kernels={[module.kernel_size for module in convs]}")
        current_index += 1
        ok = 6 in _evolution_steps(model) and "+update" in source and "tanh(" in source
        update_check(current_index, ok, f"evolution_steps={_evolution_steps(model)}")
        current_index += 1
        update_check(current_index, not batch_norms, f"active BatchNorm2d={len(batch_norms)}")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "forward_executable:shape", "forward_executable:finite", "forward_executable:six_evolutions",
        "forward_executable:pos_compatibility",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        generated_module = _load_module("kno_sec_generated_model", CANDIDATE_DIR / "model.py")
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        model = _build_model(generated_module, (64, 64)).cpu().eval()
        pos, fields = _random_inputs(torch, (64, 64))
        operator_class = getattr(generated_module, "MultiScaleKoopmanOperator2D")
        calls = 0

        def count_call(_module: Any, _inputs: Any, _output: Any) -> None:
            nonlocal calls
            calls += 1

        handles = [
            module.register_forward_hook(count_call)
            for module in model.modules()
            if isinstance(module, operator_class)
        ]
        try:
            with torch.inference_mode():
                output = _model_forward(model, pos, fields)
                shifted_pos_output = _model_forward(model, pos + 7.0, fields)
        finally:
            for handle in handles:
                handle.remove()
        ok = tuple(output.shape) == (1, 4096, 1)
        update_check(current_index, ok, f"actual={tuple(output.shape)}, expected=(1, 4096, 1)")
        current_index += 1
        update_check(current_index, bool(torch.isfinite(output).all()), "all output values are finite")
        current_index += 1
        update_check(current_index, calls == 12, f"6 calls per forward x 2 forwards; observed={calls}")
        current_index += 1
        update_check(current_index, bool(torch.equal(output, shifted_pos_output)), "pos is interface-only and does not affect output")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_equivalence:state_schema", "forward_equivalence:output"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        generated_module = _load_module("kno_sec_generated_model", CANDIDATE_DIR / "model.py")
        reference_module = _load_module("kno_sec_reference_model", REFERENCE_DIR / "model.py")
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        reference = _build_model(reference_module, (16, 16), reduced=True).cpu().eval()
        generated = _build_model(generated_module, (16, 16), reduced=True).cpu().eval()
        reference_state = reference.state_dict()
        generated_state = generated.state_dict()
        compatible = set(reference_state) == set(generated_state) and all(
            generated_state[name].shape == value.shape for name, value in reference_state.items()
        )
        if not compatible:
            update_check(current_index, False, "generated and reference state schemas differ")
            current_index += 1
        else:
            update_check(current_index, True, f"{len(reference_state)} tensors")
            current_index += 1
            generated.load_state_dict(reference_state, strict=True)
            pos, fields = _random_inputs(torch, (16, 16))
            with torch.inference_mode():
                expected = _model_forward(reference, pos, fields)
                actual = _model_forward(generated, pos, fields)
            maximum = float((actual - expected).abs().max())
            update_check(current_index, torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6), f"max_abs_error={maximum:.3e}")
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "traning:rollout_shape", "traning:composite_loss", "traning:finite_gradients",
        "traning:multiscale_gradients", "traning:optimizer_update",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch
        import torch.nn.functional as F

        generated_module = _load_module("kno_sec_generated_model", CANDIDATE_DIR / "model.py")
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        shape = (16, 16)
        model = _build_model(generated_module, shape, reduced=True).cpu().train()
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        before = {name: value.detach().clone() for name, value in parameters.items()}
        optimizer = torch.optim.AdamW(parameters.values(), lr=1.0e-3, weight_decay=1.0e-5)
        pos, state = _random_inputs(torch, shape)
        target = torch.randn((1, shape[0] * shape[1], 3), generator=torch.Generator().manual_seed(SEED + 1))
        predictions = []
        for step in range(3):
            prediction = _model_forward(model, pos, state)
            predictions.append(prediction)
            feedback = target[..., step : step + 1] if step == 0 else prediction
            state = torch.cat((state[..., 1:], feedback), dim=-1)
        rollout = torch.cat(predictions, dim=-1)
        time_weights = torch.tensor([1.0, 1.5, 2.0])
        time_weights /= time_weights.sum()
        mse_terms, relative_terms, spectral_terms = [], [], []
        for step in range(3):
            prediction = rollout[..., step : step + 1]
            truth = target[..., step : step + 1]
            mse_terms.append(F.mse_loss(prediction, truth))
            relative_terms.append(torch.linalg.vector_norm(prediction - truth) / torch.linalg.vector_norm(truth).clamp_min(1.0e-8))
            prediction_grid = prediction.reshape(1, *shape)
            truth_grid = truth.reshape(1, *shape)
            spectral_terms.append(
                torch.linalg.vector_norm(torch.fft.rfft2(prediction_grid) - torch.fft.rfft2(truth_grid))
                / torch.linalg.vector_norm(torch.fft.rfft2(truth_grid)).clamp_min(1.0e-8)
            )
        loss = (
            torch.sum(torch.stack(mse_terms) * time_weights)
            + 0.1 * torch.sum(torch.stack(relative_terms) * time_weights)
            + 0.1 * torch.sum(torch.stack(spectral_terms) * time_weights)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite_gradients = {
            name for name, parameter in parameters.items()
            if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        }
        branch_gradients = [
            name for name in finite_gradients
            if any(token in name.lower() for token in ("koopman", "high", "gate"))
        ]
        optimizer.step()
        changed = [name for name, parameter in parameters.items() if not torch.equal(before[name], parameter.detach())]
        update_check(current_index, tuple(rollout.shape) == tuple(target.shape), f"actual={tuple(rollout.shape)}")
        current_index += 1
        update_check(current_index, bool(torch.isfinite(loss)), f"loss={float(loss.detach()):.6f}")
        current_index += 1
        update_check(current_index, len(finite_gradients) == len(parameters), f"finite={len(finite_gradients)}/{len(parameters)}")
        current_index += 1
        update_check(current_index, bool(branch_gradients), str(branch_gradients))
        current_index += 1
        update_check(current_index, bool(changed), f"changed={len(changed)}/{len(parameters)}")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["artifacts:declarations", "artifacts:materialized_optional", "artifacts:materialized_if_present"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        reference_dir = REFERENCE_DIR
        del reference_dir
        source = _source(CANDIDATE_DIR, ("train.py", "utils.py", "config.yaml")).lower()
        declared = (
            ("best.pt" in source or "best_kno_navier_stokes_v1e5.pt" in source)
            and ("last.pt" in source or "checkpoint" in source)
            and "train.log" in source
            and ("history.json" in source or "metrics" in source)
        )
        update_check(current_index, declared, "best/last checkpoint, train.log, and metric history")
        current_index += 1
        checkpoints = list(CANDIDATE_DIR.glob("**/best.pt")) + list(CANDIDATE_DIR.glob("**/best_kno_navier_stokes_v1e5.pt"))
        if not checkpoints:
            update_check(current_index, True, "full-training artifacts are optional during code evaluation")
            current_index += 1
        else:
            current_index += 1
            try:
                checkpoint = _load_checkpoint(checkpoints[0])
                missing = sorted(CHECKPOINT_KEYS - set(checkpoint))
                log_exists = any(CANDIDATE_DIR.glob("**/train.log"))
                metrics_exist = any(CANDIDATE_DIR.glob("**/history.json")) or any(CANDIDATE_DIR.glob("**/metrics*"))
                update_check(current_index, not missing and log_exists and metrics_exist, f"missing={missing}, log={log_exists}, metrics={metrics_exist}")
            except Exception as exc:
                update_check(current_index, False, f"{type(exc).__name__}: {exc}")
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
        current_index += 1

    checks = checks[:current_index]
    return checks
