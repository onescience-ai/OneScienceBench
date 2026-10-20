"""V2 evaluator adapter for the MTL Airfoil Surrogate evaluator."""

import ast
import importlib.util
import inspect
import os
import sys
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

MODEL_CLASS = "AirflowNNMTLDecoderI"
REQUIRED_FILES = ("models.py", "dataset.py", "train.py")
MODEL_KWARGS = {
    "nb_hidden_layers": 3,
    "size_hidden_layers": 64,
    "bn_bool": True,
    "encoder": [7, 64, 64, 8],
    "decoder_surf": [8, 64, 64, 1],
    "decoder_vol": [8, 64, 64, 4],
}
EXPECTED_CHANNELS = {
    "encoder": [7, 64, 64, 8],
    "nn": [8, 64, 64, 64, 8],
    "decoder_surf": [8, 64, 64, 1],
    "decoder_vol": [8, 64, 64, 4],
}
SEED = 20260818
NODES = 64
SURFACE_NODES = 16
TOLERANCE = 1e-6

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    directory = str(path.parent)
    while directory in sys.path:
        sys.path.remove(directory)
    sys.path.insert(0, directory)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _all_python_source(directory: Path) -> str:
    return "\n".join(
        _read(path)
        for path in sorted(directory.rglob("*.py"))
        if not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )

def _contains(source: str, *tokens: str) -> bool:
    source = source.lower()
    return all(token.lower() in source for token in tokens)

def _model(module: Any, model_kwargs=None):
    model_class = getattr(module, MODEL_CLASS, None)
    if not inspect.isclass(model_class):
        raise TypeError(f"{MODEL_CLASS} is not defined in {module.__file__}")
    return model_class(**(MODEL_KWARGS if model_kwargs is None else model_kwargs))

def _channels(component: Any) -> list[int]:
    import torch.nn as nn
    linears = list(getattr(component, "lins", []))
    if not linears:
        linears = [layer for layer in component.modules() if isinstance(layer, nn.Linear)]
    if not linears or not all(isinstance(layer, nn.Linear) for layer in linears):
        return []
    return [linears[0].in_features, *[layer.out_features for layer in linears]]

def _batch(torch: Any):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    inputs = torch.randn(NODES, 7, generator=generator)
    targets = torch.randn(NODES, 4, generator=generator)
    surface = torch.zeros(NODES, dtype=torch.bool)
    surface[:SURFACE_NODES] = True
    return inputs, targets, surface

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["file:models.py", "file:syntax:models.py", "file:dataset.py", "file:syntax:dataset.py", "file:train.py", "file:syntax:train.py"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        del caller_globals, caller_locals
        required_files = tuple(config.get("required_files", REQUIRED_FILES))
        for name in required_files:
            path = CANDIDATE_DIR / Path(name).name
            update_check(current_index, path.is_file(), path)
            current_index += 1
            if path.is_file():
                try:
                    ast.parse(_read(path), filename=str(path))
                    ok, msg = True, "valid Python"
                except SyntaxError as exc:
                    ok, msg = False, exc
                update_check(current_index, ok, msg)
            current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_data_transformation(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["data:airfrans_meshes", "data:input_features", "data:four_targets", "data:surface_mask", "data:normalization", "data:split_and_sampling"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        del caller_globals, caller_locals
        source = _read(CANDIDATE_DIR / Path(config.get("dataset_file", "dataset.py")).name)
        lower = source.lower()
        update_check(current_index, "internal.vtu" in lower and "aerofoil.vtp" in lower, "volume VTU and aerofoil VTP are read"); current_index += 1
        update_check(current_index, _contains(source, "implicit_distance", "normal") and ("u_inf" in lower or "velocity" in lower), "coordinates/freestream/SDF/normals form seven input channels"); current_index += 1
        update_check(current_index, ("nut" in lower or "turbulent" in lower) and ("pressure" in lower or '"p"' in lower or "['p']" in lower), "u, v, pressure and turbulent viscosity targets"); current_index += 1
        update_check(current_index, "surf" in lower and ("bool" in lower or "mask" in lower), "surface/volume boolean separation"); current_index += 1
        update_check(current_index, ("mean" in lower or "mu" in lower) and ("std" in lower or "sigma" in lower), "training-statistics normalization"); current_index += 1
        update_check(current_index, "train" in lower and "test" in lower and ("subsampl" in lower or "randperm" in lower or "choice" in lower), "train/test split and point subsampling"); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["train:model_usage", "train:two_task_losses", "train:stch", "train:optimization", "train:validation", "train:scheduler"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        del caller_globals, caller_locals
        train_source = _read(CANDIDATE_DIR / Path(config.get("files", ["train.py"])[0]).name)
        project_source = _all_python_source(CANDIDATE_DIR)
        lower = train_source.lower()
        update_check(current_index, MODEL_CLASS.lower() in lower, "training instantiates the MTL surrogate"); current_index += 1
        update_check(current_index, "surf" in lower and ("vol" in lower or "volume" in lower) and ("mse" in lower or "squared" in lower), "surface-pressure and volume-field losses"); current_index += 1
        update_check(current_index, "stch" in project_source.lower(), "Smooth Tchebycheff multi-task weighting"); current_index += 1
        update_check(current_index, "optimizer" in lower and "backward" in lower and ".step(" in lower, "zero-grad/backward/optimizer-step loop"); current_index += 1
        update_check(current_index, ("valid" in lower or "val_" in lower) and ("no_grad" in lower or "inference_mode" in lower), "validation without gradient tracking"); current_index += 1
        update_check(current_index, "scheduler" in lower or "onecycle" in lower or "cosine" in lower, "learning-rate scheduling"); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["eval:optional_absent", "eval:checkpoint", "eval:inference_mode", "eval:metrics"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        del caller_globals, caller_locals
        directory = CANDIDATE_DIR
        optional_files = tuple(config.get("optional_files", ("inference.py", "evaluate.py", "eval.py")))
        path = next((directory / Path(name).name for name in optional_files if (directory / Path(name).name).is_file()), None)
        if path is None:
            update_check(current_index, True, "no evaluation script supplied"); current_index += 1
        else:
            update_check(current_index, True, "evaluation script supplied"); current_index += 1
            source = _read(path); project_source = _all_python_source(directory); lower = source.lower()
            update_check(current_index, "torch.load" in lower and "load_state_dict" in lower, "checkpoint restoration"); current_index += 1
            update_check(current_index, ".eval(" in lower and ("no_grad" in lower or "inference_mode" in lower), "evaluation mode without gradients"); current_index += 1
            update_check(current_index, "mse" in project_source.lower() and ("force" in project_source.lower() or "lift" in project_source.lower()) and ("spearman" in project_source.lower() or "rank" in project_source.lower()), "field and aerodynamic design metrics"); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["model:submodules", "model:channels:encoder", "model:channels:nn", "model:channels:decoder_surf", "model:channels:decoder_vol", "model:backbone_batch_norm", "model:parameter_partition"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch.nn as nn
        del caller_globals, caller_locals
        generated_module = _load_module("mtl_airfoil_generated", CANDIDATE_DIR / "models.py")
        model = _model(generated_module, config.get("model_args", MODEL_KWARGS))
        missing = [name for name in EXPECTED_CHANNELS if not hasattr(model, name)]
        if missing:
            update_check(current_index, False, f"missing={missing}"); current_index += 1
        else:
            update_check(current_index, True, "encoder, shared backbone, surface decoder and volume decoder"); current_index += 1
            for name, expected in EXPECTED_CHANNELS.items():
                actual = _channels(getattr(model, name)); update_check(current_index, actual == expected, f"expected={expected} actual={actual}"); current_index += 1
            batch_norms = [layer for layer in model.nn.modules() if isinstance(layer, nn.BatchNorm1d)]
            update_check(current_index, len(batch_norms) == 3, f"count={len(batch_norms)}"); current_index += 1
            try:
                shared = {id(parameter) for parameter in model.shared_parameters()}; task = {id(parameter) for parameter in model.task_specific_parameters()}; all_parameters = {id(parameter) for parameter in model.parameters()}
                partition_ok = bool(shared and task) and not shared & task and shared | task == all_parameters
                detail = f"shared={len(shared)} task={len(task)} all={len(all_parameters)}"
            except Exception as exc:
                partition_ok, detail = False, f"{type(exc).__name__}: {exc}"
            update_check(current_index, partition_ok, detail); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward:outputs", "forward:shapes", "forward:finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        del caller_globals, caller_locals
        generated_module = _load_module("mtl_airfoil_generated", CANDIDATE_DIR / "models.py")
        model = _model(generated_module, config.get("model_args", MODEL_KWARGS)).cpu().eval()
        inputs, _, surface = _batch(torch)
        with torch.no_grad(): output = model(inputs, surface)
        pair_ok = isinstance(output, (tuple, list)) and len(output) == 2
        update_check(current_index, pair_ok, f"type={type(output).__name__}" if not pair_ok else "two outputs"); current_index += 1
        if pair_ok:
            out_surface, out_volume = output; shapes = (tuple(out_surface.shape), tuple(out_volume.shape)); expected = ((SURFACE_NODES, 1), (NODES - SURFACE_NODES, 4))
            update_check(current_index, shapes == expected, f"expected={expected} actual={shapes}"); current_index += 1
            update_check(current_index, bool(torch.isfinite(out_surface).all() and torch.isfinite(out_volume).all()), "all outputs are finite"); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["equivalence:surface", "equivalence:volume"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        del caller_globals, caller_locals
        torch.manual_seed(SEED)
        reference_module = _load_module("mtl_airfoil_reference", REFERENCE_DIR / "models.py")
        generated_module = _load_module("mtl_airfoil_generated", CANDIDATE_DIR / "models.py")
        model_kwargs = config.get("model_args", MODEL_KWARGS)
        reference = _model(reference_module, model_kwargs).cpu().eval(); generated = _model(generated_module, model_kwargs).cpu().eval()
        generated.load_state_dict(reference.state_dict(), strict=True)
        inputs, _, surface = _batch(torch)
        with torch.no_grad(): reference_output = reference(inputs, surface); generated_output = generated(inputs, surface)
        for name, actual, expected in zip(("surface", "volume"), generated_output, reference_output):
            difference = float((actual - expected).abs().max()); update_check(current_index, torch.allclose(actual, expected, atol=TOLERANCE, rtol=TOLERANCE), f"max_abs_diff={difference:.3e}"); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["training:loss", "training:all_parameters_trainable", "training:gradients", "training:updates"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        del caller_globals, caller_locals
        torch.manual_seed(SEED)
        generated_module = _load_module("mtl_airfoil_generated", CANDIDATE_DIR / "models.py")
        model = _model(generated_module, config.get("model_args", MODEL_KWARGS)).cpu().train()
        parameters = dict(model.named_parameters()); trainable = {name: parameter for name, parameter in parameters.items() if parameter.requires_grad}; before = {name: parameter.detach().clone() for name, parameter in trainable.items()}; optimizer = torch.optim.Adam(trainable.values(), lr=1e-3); inputs, targets, surface = _batch(torch)
        optimizer.zero_grad(); out_surface, out_volume = model(inputs, surface); loss_surface = torch.nn.functional.mse_loss(out_surface, targets[surface, 2:3]); loss_volume = torch.nn.functional.mse_loss(out_volume, targets[~surface]); loss = loss_surface + loss_volume; loss.backward()
        finite_gradients = {name for name, parameter in trainable.items() if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())}; optimizer.step(); changed = {name for name, parameter in trainable.items() if not torch.equal(before[name], parameter.detach())}; missing_gradients = sorted(set(trainable) - finite_gradients); unchanged = sorted(set(trainable) - changed)
        update_check(current_index, bool(torch.isfinite(loss)), f"surface={float(loss_surface):.6g} volume={float(loss_volume):.6g}"); current_index += 1
        update_check(current_index, len(trainable) == len(parameters), f"trainable={len(trainable)} total={len(parameters)}"); current_index += 1
        update_check(current_index, not missing_gradients, f"missing_or_nonfinite={missing_gradients}"); current_index += 1
        update_check(current_index, not unchanged, f"unchanged={unchanged}"); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["artifact:checkpoint", "artifact:metrics", "artifact:normalization_stats"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        del caller_globals, caller_locals
        train_source = _read(CANDIDATE_DIR / Path(config.get("files", ["train.py"])[0]).name).lower(); data_source = _read(CANDIDATE_DIR / Path(config.get("dataset_file", "dataset.py")).name).lower()
        update_check(current_index, "torch.save" in train_source and (".pt" in train_source or ".pth" in train_source), "model checkpoint is saved"); current_index += 1
        update_check(current_index, "json.dump" in train_source or "metrics.json" in train_source or "csv" in train_source, "training/evaluation metrics are saved"); current_index += 1
        update_check(current_index, ("np.save" in data_source or "numpy.save" in data_source) and "mean" in data_source and "std" in data_source, "normalization statistics are saved"); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks
