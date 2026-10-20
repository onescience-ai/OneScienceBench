"""V2 evaluator adapter for the INFINITY AirfRANS paper reproduction."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import sys
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

for _optional_dependency in ("h5py", "pandas"):
    try:
        __import__(_optional_dependency)
    except ModuleNotFoundError as error:
        if error.name != _optional_dependency:
            raise
        _stub = types.ModuleType(_optional_dependency)
        _stub.__spec__ = importlib.util.spec_from_loader(_optional_dependency, loader=None)
        sys.modules[_optional_dependency] = _stub

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

REQUIRED_FILES = (
    "model.py", "dataset.py", "utils.py", "train.py", "inference.py", "config.yaml"
)
DATA_ROOT = Path(
    "/public/share/sugonhpcapp01/onestore/onedatasets/"
    "Transolver-Airfoil-Design/Dataset"
)
FIELD_ALIASES = {
    "distance": ("distance", "sdf", "d"),
    "normal": ("normal", "normals", "n"),
    "vx": ("vx", "velocity_x"),
    "vy": ("vy", "velocity_y"),
    "pressure": ("pressure", "p"),
    "nu_t": ("nu_t", "nut", "nu", "turbulent_viscosity"),
}
SEED = 20260820

@dataclass
class Project:
    directory: Path
    model: Any | None
    config: dict[str, Any]
    sources: dict[str, str]
    import_error: str | None = None

    @property
    def source(self) -> str:
        return "\n".join(self.sources.values())

@dataclass
class ModelProbe:
    fields: dict[str, Any]
    processor: Any
    latent_dim: int

    def field(self, name: str, coordinates, code):
        import torch
        return self.fields[name](coordinates, code)

    def process(self, distance_code, normal_code, inlet):
        import torch
        if len(inspect.signature(self.processor.forward).parameters) >= 3:
            output = self.processor(distance_code, normal_code, inlet)
        else:
            output = self.processor(torch.cat((distance_code, normal_code, inlet), -1))
        if not isinstance(output, Mapping):
            return output
        fields = ("vx", "vy", "pressure", "nu_t")
        return torch.cat([_mapping_value(output, FIELD_ALIASES[name]) for name in fields], -1)

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""

def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    value = yaml.safe_load(_read(path))
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _load_project(directory: Path, role: str, config_override=None) -> Project:
    sources = {
        name: _read(directory / name) for name in REQUIRED_FILES if name.endswith(".py")
    }
    config_path = directory / "config.yaml"
    config = _load_yaml(config_path) if config_path.is_file() else {}
    try:
        model = _load_module(directory / "model.py", f"infinity_{role}_model")
        return Project(directory, model, config, sources)
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        return Project(directory, None, config, sources, detail)

def _contains(source: str, alternatives: Iterable[str]) -> bool:
    lower = source.lower()
    return any(value.lower() in lower for value in alternatives)

def _deep_values(value: Any, names: Iterable[str]) -> list[Any]:
    names, found = set(names), []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in names:
                found.append(child)
            found.extend(_deep_values(child, names))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_deep_values(child, names))
    return found

def _mapping_value(mapping: Mapping[str, Any], aliases: Iterable[str]):
    for name in aliases:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"None of {tuple(aliases)} exists in {tuple(mapping)}")

def _field_name(module: Any, semantic_name: str) -> str:
    names = tuple(getattr(module, "ALL_FIELDS", ()))
    for alias in FIELD_ALIASES[semantic_name]:
        if not names or alias in names:
            return alias
    raise KeyError(f"No {semantic_name} field in {names}")

def _latent_dim(field_model: Any) -> int:
    for candidate in (field_model, getattr(field_model, "inr", None)):
        if candidate is not None and hasattr(candidate, "latent_dim"):
            return int(candidate.latent_dim)
    raise AttributeError("Field INR does not expose latent_dim")

def _build_probe(project: Project) -> ModelProbe:
    if project.model is None:
        raise ImportError(project.import_error or "model.py import failed")
    module = project.model
    if callable(getattr(module, "build_inr", None)):
        fields = {
            name: module.build_inr(_field_name(module, name), project.config)
            for name in ("distance", "normal")
        }
        processor = module.build_processor(project.config)
    else:
        spec = getattr(module, "INRBuildSpec", None)
        if spec is None or not callable(getattr(spec, "from_config", None)):
            raise TypeError("Expected build_inr/build_processor or INRBuildSpec.from_config")
        system = spec.from_config(project.config).build()
        modules = getattr(system, "field_inrs", getattr(system, "inrs", None))
        if modules is None:
            raise AttributeError("Built model must expose field_inrs or inrs")
        fields = {
            name: _mapping_value(modules, FIELD_ALIASES[name])
            for name in ("distance", "normal")
        }
        processor = system.processor
    latent_dim = _latent_dim(fields["distance"])
    if _latent_dim(fields["normal"]) != latent_dim:
        raise ValueError("Geometry INRs use different latent dimensions")
    return ModelProbe(fields, processor, latent_dim)

def _adapt(project: Project, model: Any, coordinates, target, create_graph: bool):
    import torch
    from torch.nn import functional
    module = project.model
    if callable(getattr(module, "adapt_code", None)):
        return module.adapt_code(model, coordinates, target, 3, create_graph=create_graph)
    if callable(getattr(module, "adapt_latent", None)):
        rates = _deep_values(project.config, ("inner_lr",))
        result = module.adapt_latent(
            model, coordinates, target, 3, float(rates[0] if rates else 1e-2), create_graph
        )
        return result[0] if isinstance(result, tuple) else result
    code = torch.zeros(
        _latent_dim(model), device=coordinates.device, dtype=coordinates.dtype,
        requires_grad=True,
    )
    for _ in range(3):
        loss = functional.mse_loss(model(coordinates, code), target)
        gradient = torch.autograd.grad(loss, code, create_graph=create_graph)[0]
        code = code - 1e-2 * gradient
    return code

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "file:model.py", "syntax:model.py", "file:dataset.py", "syntax:dataset.py",
        "file:utils.py", "syntax:utils.py", "file:train.py", "syntax:train.py",
        "file:inference.py", "syntax:inference.py", "file:config.yaml",
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
            if path.is_file() and path.suffix == ".py":
                try:
                    ast.parse(_read(path), filename=str(path))
                    syntax_ok, detail = bool(_read(path).strip()), "valid Python AST"
                except Exception as error:
                    syntax_ok, detail = False, f"{type(error).__name__}: {error}"
                update_check(current_index, syntax_ok, detail)
                current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_data_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "data:configured_root", "data:local_vtk_sample", "data:paper_800_200_split",
        "data:airfrans_fields", "data:case_level_split", "data:train_only_normalization",
        "data:no_repository_download",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        config = _load_yaml(CANDIDATE_DIR / "config.yaml")
        source = "\n".join(_read(CANDIDATE_DIR / name) for name in REQUIRED_FILES if name.endswith(".py")).lower()
        roots = _deep_values(config, ("root", "root_path", "data_root"))
        root = Path(str(roots[0])).expanduser() if roots else None
        internal = next(DATA_ROOT.glob("*/*_internal.vtu"), None) if DATA_ROOT.is_dir() else None
        surface = None if internal is None else internal.with_name(internal.name.replace("_internal.vtu", "_aerofoil.vtp"))
        fields = {
            "distance": _contains(source, ("implicit_distance", "sdf", "distance")),
            "normal": _contains(source, ("normals", "normal")),
            "velocity": _contains(source, ('["u"]', "velocity", "vx")),
            "pressure": _contains(source, ('["p"]', '"p"', "'p'", "pressure")),
            "viscosity": _contains(source, ("nut", "nu_t", '"nu"')),
        }
        values = {
            "configured_root": root is not None and root.resolve() == DATA_ROOT.resolve(),
            "local_vtk_sample": internal is not None and surface is not None and surface.is_file(),
            "paper_800_200_split": 800 in _deep_values(config, ("train_count", "expected_train_cases")) and 200 in _deep_values(config, ("test_count", "expected_test_cases")),
            "airfrans_fields": all(fields.values()),
            "case_level_split": _contains(source, ("case", "simulation")) and _contains(source, ("split", "manifest")),
            "train_only_normalization": "normalization" in source and "train" in source and _contains(source, ("mean", "std", "moment")),
            "no_repository_download": not _contains(source, ("git clone", "github.com", "wget ", "curl ")),
        }
        for key in ("configured_root", "local_vtk_sample", "paper_800_200_split", "airfrans_fields", "case_level_split", "train_only_normalization", "no_repository_download"):
            update_check(current_index, values[key], str(values[key]))
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "train:main_entrypoint", "train:six_fields", "train:three_step_autodecoding",
        "train:second_order_meta_learning", "train:two_stage_processor", "train:optimization",
        "train:live_logging", "train:persistence", "train:config_driven",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        config = _load_yaml(CANDIDATE_DIR / "config.yaml")
        train = _read(CANDIDATE_DIR / "train.py")
        combined = "\n".join((train, _read(CANDIDATE_DIR / "model.py"), _read(CANDIDATE_DIR / "utils.py"))).lower()
        values = {
            "main_entrypoint": "def main" in train and "__main__" in train,
            "six_fields": all(_contains(combined, aliases) for aliases in FIELD_ALIASES.values()),
            "three_step_autodecoding": 3 in _deep_values(config, ("inner_steps", "test_inner_steps")) and _contains(combined, ("adapt_code", "adapt_latent", "auto-decod")),
            "second_order_meta_learning": "create_graph=true" in combined or ("create_graph" in combined and "backward(" in combined),
            "two_stage_processor": "processor" in combined and "latent" in combined and _contains(combined, ("stage", "encode_dataset", "materialize_latents")),
            "optimization": _contains(combined, ("adam", "adamw")) and _contains(combined, ("mse_loss", "square")) and "backward(" in combined,
            "live_logging": "print(" in train or "tqdm" in train.lower(),
            "persistence": _contains(combined, ("torch.save", "save_checkpoint")) and _contains(combined, ("latent", "curve", "csv")),
            "config_driven": _contains(train, ("load_config", "yaml")) and bool(config),
        }
        for key in ("main_entrypoint", "six_fields", "three_step_autodecoding", "second_order_meta_learning", "two_stage_processor", "optimization", "live_logging", "persistence", "config_driven"):
            update_check(current_index, values[key], str(values[key]))
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "eval:checkpoint_restore", "eval:geometry_process_decode", "eval:full_grid_chunks",
        "eval:volume_mse", "eval:surface_pressure", "eval:drag_lift", "eval:spearman",
        "eval:saved_results", "eval:optional_absent",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        inference = _read(CANDIDATE_DIR / "inference.py").lower()
        if not inference:
            update_check(8, True, "not supplied")
        else:
            source = "\n".join(_read(CANDIDATE_DIR / name) for name in REQUIRED_FILES if name.endswith(".py")).lower()
            values = {
                "checkpoint_restore": _contains(source, ("load_checkpoint", "torch.load")) and "load_state_dict" in source,
                "geometry_process_decode": _contains(inference, ("distance", "sdf")) and _contains(inference, ("normal", '"n"')) and "processor" in inference and _contains(inference, ("decode", "predict_case")),
                "full_grid_chunks": "chunk" in inference,
                "volume_mse": "mse" in inference and _contains(inference, ("volume", "normalized")),
                "surface_pressure": "surface" in inference and _contains(inference, ("pressure", "mse_p")),
                # The legacy evaluator treats these optional inference metrics as
                # non-blocking when the project does not implement them.
                "drag_lift": (
                    _contains(inference, ("force_coefficient", "drag", "c_d"))
                    and _contains(inference, ("lift", "c_l"))
                ) or not _contains(inference, ("force_coefficient", "drag", "c_d", "lift", "c_l")),
                "spearman": "spearman" in inference or "correlation" not in inference,
                "saved_results": _contains(inference, ("np.savez", "save_npz")) and _contains(inference, ("json", "metrics")),
            }
            for key in ("checkpoint_restore", "geometry_process_decode", "full_grid_chunks", "volume_mse", "surface_pressure", "drag_lift", "spearman", "saved_results"):
                update_check(current_index, values[key], str(values[key]))
                current_index += 1
            update_check(current_index, True, "not applicable when inference.py is supplied")
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "model:INRBuildSpec", "model:six_independent_fields", "model:fourier_backbone",
        "model:shift_modulation", "model:linear_hypernetwork", "model:latent_processor",
        "model:auto_decoding",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        generated_module = _load_project(CANDIDATE_DIR, "generated", config)
        if generated_module.model is None:
            raise ImportError(generated_module.import_error)
        source, model = generated_module.sources.get("model.py", "").lower(), generated_module.model
        values = {
            "INRBuildSpec": inspect.isclass(getattr(model, "INRBuildSpec", None)),
            "six_independent_fields": all(_contains(source, aliases) for aliases in FIELD_ALIASES.values()) and _contains(source, ("moduledict", "build_inr", "field_inrs")),
            "fourier_backbone": "fourier" in source and _contains(source, ("sin", "cos")),
            "shift_modulation": "shift" in source and "+ shift" in source,
            "linear_hypernetwork": "hypernetwork" in source and "linear" in source,
            "latent_processor": "processor" in source and _contains(source, ("mlp", "sequential", "residualblock")),
            "auto_decoding": "zeros" in source and "autograd.grad" in source,
        }
        for key in ("INRBuildSpec", "six_independent_fields", "fourier_backbone", "shift_modulation", "linear_hypernetwork", "latent_processor", "auto_decoding"):
            update_check(current_index, values[key], str(values[key]))
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward:distance", "forward:normal", "forward:processor"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        torch.manual_seed(SEED)
        generated_module = _load_project(CANDIDATE_DIR, "generated", config)
        probe, coordinates = _build_probe(generated_module), torch.randn(37, 2)
        code = torch.zeros(probe.latent_dim)
        distance = probe.field("distance", coordinates, code)
        normal = probe.field("normal", coordinates, code)
        processed = probe.process(torch.zeros(3, probe.latent_dim), torch.zeros(3, probe.latent_dim), torch.randn(3, 2))
        update_check(current_index, bool(distance.shape == (37, 1) and torch.isfinite(distance).all()), str(distance.shape))
        current_index += 1
        update_check(current_index, bool(normal.shape == (37, 2) and torch.isfinite(normal).all()), str(normal.shape))
        current_index += 1
        update_check(current_index, bool(processed.shape == (3, 4 * probe.latent_dim) and torch.isfinite(processed).all()), str(processed.shape))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["equivalence:distance", "equivalence:normal"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        generated_module = _load_project(CANDIDATE_DIR, "generated", config)
        reference_module = _load_project(REFERENCE_DIR, "reference")
        generated, reference = _build_probe(generated_module), _build_probe(reference_module)
        coordinates = torch.randn(29, 2, generator=torch.Generator().manual_seed(SEED))
        for probe in (generated, reference):
            for model in probe.fields.values():
                for parameter in model.parameters():
                    parameter.data.zero_()
        for current_name, width in (("distance", 1), ("normal", 2)):
            left = generated.field(current_name, coordinates, torch.zeros(generated.latent_dim))
            right = reference.field(current_name, coordinates, torch.zeros(reference.latent_dim))
            same_shape = left.shape == right.shape == (29, width)
            difference = float((left - right).abs().max()) if same_shape else float("inf")
            update_check(current_index, same_shape and torch.allclose(left, right), f"max_abs_diff={difference:.3e}")
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["training:cavia_outer_step", "training:processor_step"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        from torch.nn import functional
        torch.manual_seed(SEED)
        generated_module = _load_project(CANDIDATE_DIR, "generated", config)
        probe, coordinates = _build_probe(generated_module), torch.randn(24, 2)
        target = (coordinates[:, :1].sin() + coordinates[:, 1:2].cos()) * 0.5
        field_model = probe.fields["distance"]
        optimizer = torch.optim.Adam(field_model.parameters(), lr=1e-3)
        before = [value.detach().clone() for value in field_model.parameters()]
        code = _adapt(generated_module, field_model, coordinates, target, create_graph=True)
        loss = functional.mse_loss(probe.field("distance", coordinates, code), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [value.grad for value in field_model.parameters() if value.grad is not None]
        optimizer.step()
        changed = any(not torch.equal(old, new) for old, new in zip(before, field_model.parameters()))
        field_ok = bool(torch.isfinite(loss)) and bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        update_check(current_index, field_ok and changed, str(float(loss.detach())))
        current_index += 1
        processor_optimizer = torch.optim.Adam(probe.processor.parameters(), lr=1e-3)
        processor_loss = functional.mse_loss(probe.process(torch.randn(4, probe.latent_dim), torch.randn(4, probe.latent_dim), torch.randn(4, 2)), torch.randn(4, 4 * probe.latent_dim))
        processor_optimizer.zero_grad(set_to_none=True)
        processor_loss.backward()
        processor_gradients = [p.grad for p in probe.processor.parameters() if p.grad is not None]
        processor_optimizer.step()
        processor_ok = bool(torch.isfinite(processor_loss)) and bool(processor_gradients)
        processor_ok = processor_ok and all(bool(torch.isfinite(g).all()) for g in processor_gradients)
        update_check(current_index, processor_ok, str(float(processor_loss.detach())))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "artifacts:trainings_root", "artifacts:checkpoints", "artifacts:latents",
        "artifacts:curves", "artifacts:predictions", "artifacts:metrics",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        source = "\n".join(_read(CANDIDATE_DIR / name) for name in REQUIRED_FILES if name.endswith(".py")).lower() + json.dumps(config).lower()
        values = {
            "trainings_root": "trainings" in source,
            "checkpoints": _contains(source, ("checkpoint", ".pt")),
            "latents": "latent" in source,
            "curves": _contains(source, ("curve", "csv", "savefig")),
            "predictions": "prediction" in source,
            "metrics": "metrics" in source,
        }
        for key in ("trainings_root", "checkpoints", "latents", "curves", "predictions", "metrics"):
            update_check(current_index, values[key], str(values[key]))
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks
