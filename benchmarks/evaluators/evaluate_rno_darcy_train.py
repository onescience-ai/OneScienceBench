#!/usr/bin/env python3
"""Evaluate generated RNO-Darcy training projects against the task contract."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


TASK_NAME = "rno_darcy_train"
MODEL_CLASS = "RadonNeuralOperator2D"
REQUIRED_FILES = ("config.yaml", "dataset.py", "model.py", "utils.py", "train.py")
PYTHON_FILES = ("dataset.py", "model.py", "utils.py", "train.py", "evaluate.py")
SUBMISSION_NAMES = ("test_rno_darcy_train", "test_rno_darcy_train_oneskills", TASK_NAME)
DATA_ROOT = Path("/public/share/sugonhpcapp01/onestore/onedatasets/CFD_Benchmark/darcy")
TRAIN_FILE = "piececonst_r421_N1024_smooth1.mat"
TEST_FILE = "piececonst_r421_N1024_smooth2.mat"
CHECK_TYPES = (
    "file", "data", "train_script", "eval_script", "model_structure",
    "forward_executable", "forward_equivalence", "traning", "artifacts",
)
REQUIRED_CHECK_TYPES = {
    "file", "data", "train_script", "model_structure", "forward_executable", "traning",
}


@dataclass
class Project:
    root: Path
    config: dict[str, Any]
    model_module: Any


def _check(check_type: str, name: str, ok: bool, detail: str) -> dict[str, Any]:
    if check_type not in CHECK_TYPES:
        raise ValueError(f"unsupported check_type: {check_type}")
    return {"check_type": check_type, "check": name, "ok": bool(ok), "detail": str(detail)}


def _records(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else [value]


def _guard(check_type: str, fn: Callable[[], Any]) -> list[dict[str, Any]]:
    try:
        return _records(fn())
    except Exception as exc:
        return [_check(check_type, f"{check_type}_exception", False, f"{type(exc).__name__}: {exc}")]


def _standard_dir() -> Path:
    explicit = os.environ.get("RNO_DARCY_TRAIN_STANDARD_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "standard" / TASK_NAME


def _submission_dir() -> Path:
    explicit = os.environ.get("RNO_DARCY_TRAIN_SUBMISSION_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if all((cwd / name).is_file() for name in REQUIRED_FILES):
        return cwd
    for name in SUBMISSION_NAMES:
        if (cwd / name).is_dir():
            return (cwd / name).resolve()
    return cwd


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(_read(path))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a YAML mapping")
    return value


def _load_module(path: Path, tag: str) -> Any:
    name = f"_{TASK_NAME}_{tag}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def _load_project(root: Path, tag: str) -> Project:
    return Project(root, _load_yaml(root / "config.yaml"), _load_module(root / "model.py", tag))


def _contains(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return all(needle.lower() in lowered for needle in needles)


def _deep_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _deep_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _deep_values(item)
    else:
        yield value


def _has_value(config: dict[str, Any], expected: Any) -> bool:
    if isinstance(expected, str):
        target = expected.lower()
        return any(target in str(value).lower() for value in _deep_values(config))
    return any(value == expected for value in _deep_values(config))


def _has_main_guard(text: str) -> bool:
    compact = text.replace(" ", "").replace("'", '"')
    return 'if__name__=="__main__"' in compact


def _small_configs(height: int, width: int) -> tuple[dict[str, Any], ...]:
    """Cover the constructor names used by the reference and common submissions."""
    reference = {
        "height": height, "width": width, "in_channels": 3, "out_channels": 1,
        "input_dim": 1, "output_dim": 1,
        "channels": 8, "hidden_dim": 8, "num_layers": 2,
        "num_heads": 2, "num_slices": 4, "num_angles": 4, "num_radon_blocks": 1,
        "slice_num": 4, "ref_grid_size": 4, "unified_pos": True,
        "angle_hidden_channels": 16, "angle_conv_channels": 16,
        "angle_conv_kernel_size": 3, "mlp_ratio": 1, "dropout": 0.0,
    }
    prompt_style = {
        "resolution": height, "hidden_dim": 8, "num_transformer_layers": 2,
        "num_layers": 2, "num_heads": 2, "slice_num": 4,
        "num_radon_angles": 4, "num_angles": 4, "num_radon_blocks": 1,
        "angle_hidden_channels": 16, "mlp_ratio": 1, "dropout": 0.0,
        "in_channels": 3, "out_channels": 1,
    }
    return reference, prompt_style


def _construct(model_class: type, height: int, width: int | None = None):
    errors: list[str] = []
    width = height if width is None else width
    signature = inspect.signature(model_class)

    def remember(label: str, error: Exception) -> None:
        detail = str(error).splitlines()[0][:240]
        errors.append(f"{label}: {type(error).__name__}: {detail}")

    for config in _small_configs(height, width):
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        kwargs = config if accepts_kwargs else {
            key: value for key, value in config.items() if key in signature.parameters
        }
        try:
            return model_class(**kwargs)
        except Exception as exc:
            remember(f"constructor({sorted(kwargs)})", exc)
        for value in (config, {"model": config, "data": {"resolution": height}}):
            try:
                return model_class(value)
            except Exception as exc:
                remember("constructor(config)", exc)
        factory = getattr(model_class, "from_config", None)
        if factory is not None:
            for value in (config, {"model": config, "data": {"resolution": height}}):
                try:
                    return factory(value)
                except Exception as exc:
                    remember("from_config", exc)
    raise RuntimeError("could not construct a small model; " + " | ".join(errors[-4:]))


def _coefficient(batch: int, height: int, width: int):
    import torch

    y = torch.linspace(0.0, 1.0, height)
    x = torch.linspace(0.0, 1.0, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    field = 0.3 + xx + 0.5 * yy + 0.1 * torch.sin(3.0 * xx + yy)
    return field[None, None].repeat(batch, 1, 1, 1).float()


def _positions(batch: int, height: int, width: int):
    import torch

    y = torch.linspace(0.0, 1.0, height)
    x = torch.linspace(0.0, 1.0, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, -1, 2).repeat(batch, 1, 1)


def _canonical_output(output: Any, batch: int, height: int, width: int):
    import torch

    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        output = next((output[key] for key in ("pred", "prediction", "output", "sol") if key in output), None)
    if not torch.is_tensor(output):
        raise TypeError("model output must be a tensor")
    if output.shape == (batch, 1, height, width):
        return output
    if output.shape == (batch, height, width, 1):
        return output.permute(0, 3, 1, 2)
    if output.shape == (batch, height * width, 1):
        return output.transpose(1, 2).reshape(batch, 1, height, width)
    raise ValueError(f"unexpected output shape {tuple(output.shape)}")


def _forward(model: Any, coefficient: Any):
    batch, _, height, width = coefficient.shape
    positions = _positions(batch, height, width).to(coefficient)
    flat = coefficient.flatten(2).transpose(1, 2)
    errors: list[str] = []
    for args in ((coefficient,), (positions, flat), (flat, positions)):
        try:
            return _canonical_output(model(*args), batch, height, width)
        except Exception as exc:
            errors.append(f"{len(args)} argument(s): {exc}")
    raise RuntimeError("forward failed for supported interfaces; " + " | ".join(errors))


def _relative_l2(prediction: Any, target: Any):
    import torch

    numerator = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-8)
    return (numerator / denominator).mean()


def _gradient_relative_l2(prediction: Any, target: Any):
    import torch

    pred_grad = torch.cat(
        ((prediction[:, :, 1:] - prediction[:, :, :-1]).flatten(1),
         (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]).flatten(1)), dim=1,
    )
    target_grad = torch.cat(
        ((target[:, :, 1:] - target[:, :, :-1]).flatten(1),
         (target[:, :, :, 1:] - target[:, :, :, :-1]).flatten(1)), dim=1,
    )
    return (
        torch.linalg.vector_norm(pred_grad - target_grad, dim=1)
        / torch.linalg.vector_norm(target_grad, dim=1).clamp_min(1e-8)
    ).mean()


def _named_value(value: Any, names: set[str]) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in names:
                return item
            found = _named_value(item, names)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _named_value(item, names)
            if found is not None:
                return found
    return None


def check_required_files(root: Path | None = None) -> list[dict[str, Any]]:
    root = (root or _submission_dir()).resolve()
    results = [
        _check("file", f"required_{name}", (root / name).is_file(), f"required file: {name}")
        for name in REQUIRED_FILES
    ]
    for name in PYTHON_FILES:
        path = root / name
        if not path.is_file():
            continue
        try:
            ast.parse(_read(path), filename=str(path))
            results.append(_check("file", f"syntax_{name}", True, f"{name} parses successfully"))
        except SyntaxError as exc:
            results.append(_check("file", f"syntax_{name}", False, str(exc)))
    source = "\n".join(_read(root / name) for name in PYTHON_FILES if (root / name).is_file())
    independent = "codegen.standard" not in source and "codegen.evaluators" not in source
    results.append(_check("file", "submission_independence", independent, "submission must not import evaluator/reference code"))
    return results


def check_data_contract(root: Path | None = None) -> dict[str, Any]:
    root = (root or _submission_dir()).resolve()
    config = _load_yaml(root / "config.yaml")
    data_config = config.get("data", config)
    dataset_text = _read(root / "dataset.py")
    source = dataset_text.lower().replace(" ", "")

    configured_paths = [str(value) for value in _deep_values(config) if isinstance(value, (str, Path))]
    root_ok = any(str(DATA_ROOT) in value for value in configured_paths)
    config_terms = {str(value).lower() for value in _deep_values(config)}
    conditions = {
        "Darcy data root": root_ok,
        "train/test MAT files": (
            _named_value(data_config, {"train_file", "train_path"}) == TRAIN_FILE
            and _named_value(data_config, {"test_file", "test_path"}) == TEST_FILE
        ),
        "coeff and sol fields": (
            _named_value(data_config, {"input_key", "input_field"}) == "coeff"
            and _named_value(data_config, {"target_key", "output_key", "target_field"}) == "sol"
        ),
        "1000/200 split": (
            _named_value(data_config, {"n_train", "train_samples"}) == 1000
            and _named_value(data_config, {"n_test", "test_samples"}) == 200
        ),
        "421→85 stride-5 downsampling": (
            _named_value(data_config, {"native_resolution"}) == 421
            and _named_value(data_config, {"downsample", "subsample", "stride"}) == 5
            and _named_value(data_config, {"resolution"}) == 85
        ),
        "MAT reader": "loadmat" in dataset_text or "h5py" in dataset_text,
        "floating tensors": "float32" in dataset_text.lower() or ".float(" in dataset_text.lower(),
        "global scalar train-only normalization": all(
            token in source for token in ("normalizer", ".mean(", ".std(", "fit(train")
        ) and "fit(test" not in source,
        "no validation split": (
            not config_terms.intersection({"val", "valid", "validation"})
            and not any(token in source for token in ("val_loader", "validation_loader", "random_split("))
        ),
        "no external data download": not any(
            token in dataset_text.lower() for token in ("git clone", "wget ", "curl ")
        ),
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return _check("data", "darcy_data_contract", not failed, "all data requirements satisfied" if not failed else "missing: " + ", ".join(failed))


def check_train_script_contract(root: Path | None = None) -> dict[str, Any]:
    root = (root or _submission_dir()).resolve()
    config = _load_yaml(root / "config.yaml")
    training_config = config.get("training", config)
    loss_config = config.get("loss", config)
    train_text = _read(root / "train.py")
    source = (train_text + "\n" + _read(root / "utils.py")).lower()
    scheduler = str(_named_value(training_config, {"scheduler", "scheduler_name"})).lower().replace("_", "")
    conditions = {
        "main guard": _has_main_guard(train_text),
        "500 epochs": _named_value(training_config, {"epochs", "num_epochs"}) == 500,
        "batch size 4": _named_value(training_config, {"batch_size"}) == 4,
        "AdamW": str(_named_value(training_config, {"optimizer", "optimizer_name"})).lower() == "adamw",
        "learning rate 1e-3": _named_value(training_config, {"learning_rate", "lr"}) == 1e-3,
        "weight decay 1e-5": _named_value(training_config, {"weight_decay"}) == 1e-5,
        "OneCycleLR": (
            scheduler in {"onecycle", "onecyclelr"}
            and "onecyclelr" in source
            and "scheduler.step(" in source
        ),
        "maximum gradient norm 0.1": _named_value(training_config, {"max_grad_norm", "grad_clip"}) == 0.1,
        "Relative L2 + 0.1 gradient Relative L2": (
            _has_value(loss_config, "relative_l2")
            and _named_value(loss_config, {"gradient_weight", "gradient_loss_weight", "grad_loss_weight"}) == 0.1
            and "gradient" in source
        ),
        "physical-space loss": "decode(" in source or "denormal" in source,
        "zero Dirichlet boundary": "dirichlet" in source or "zero_boundary" in source,
        "AdamW update and gradient clipping": "adamw" in source and "clip_grad_norm_" in source,
        "last and periodic checkpoints": "last.pt" in source and "checkpoint_every" in source,
        "complete resume state": all(
            token in source for token in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "normalizer")
        ),
        "no test-selected checkpoint": not any(
            token in source for token in ("best.pt", "best_model", "best_checkpoint", "save_best")
        ),
        "terminal and file logging": "streamhandler" in source and "filehandler" in source,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return _check("train_script", "training_contract", not failed, "training contract satisfied" if not failed else "missing: " + ", ".join(failed))


def check_eval_script_contract(root: Path | None = None) -> dict[str, Any]:
    root = (root or _submission_dir()).resolve()
    evaluate_path = root / "evaluate.py"
    if not evaluate_path.is_file():
        return _check(
            "eval_script", "evaluation_contract", False,
            "optional evaluate.py was not provided",
        )
    text = _read(evaluate_path) + "\n" + _read(root / "utils.py")
    lower = text.lower()
    conditions = {
        "main guard": _has_main_guard(text),
        "checkpoint CLI": "--checkpoint" in text,
        "strict checkpoint load": "strict=true" in lower.replace(" ", ""),
        "evaluation mode": ".eval(" in lower and ("no_grad" in lower or "inference_mode" in lower),
        "paper and auxiliary metrics": all(
            token in lower for token in ("relative_l2", "gradient", "mse", "mae")
        ),
        "evaluation.json": "evaluation.json" in lower,
        "evaluation metadata": all(
            token in lower for token in ("epoch", "parameter_count", "config")
        ),
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return _check("eval_script", "evaluation_contract", not failed, "evaluation contract satisfied" if not failed else "missing: " + ", ".join(failed))


def check_model_structure(root: Path | None = None) -> dict[str, Any]:
    import torch.nn as nn

    project = _load_project((root or _submission_dir()).resolve(), "structure")
    model_class = getattr(project.model_module, MODEL_CLASS, None)
    if not inspect.isclass(model_class):
        return _check("model_structure", "rno_structure", False, f"missing class {MODEL_CLASS}")
    model = _construct(model_class, 11)
    modules = list(model.modules())
    class_names = [module.__class__.__name__.lower() for module in modules]
    conv_pairs = {(module.in_channels, module.out_channels) for module in modules if isinstance(module, nn.Conv1d)}
    config = project.config.get("model", project.config)
    conditions = {
        "128 hidden channels": _named_value(config, {"channels", "hidden_dim", "hidden_channels"}) == 128,
        "8 attention heads": _named_value(config, {"num_heads", "heads"}) == 8,
        "8 Transformer layers": _named_value(config, {"num_layers", "num_transformer_layers", "layers"}) == 8,
        "64 slices": _named_value(config, {"num_slices", "slice_num", "slices"}) == 64,
        "32 Radon angles": _named_value(config, {"angles", "num_angles", "num_radon_angles", "radon_angles"}) == 32,
        "4 Radon blocks": _named_value(config, {"num_radon_blocks", "radon_blocks"}) == 4,
        "16 angle hidden channels": _named_value(
            config, {"angle_hidden_channels", "angle_hidden", "angle_conv_channels"}
        ) == 16,
        "angle convolution kernel 3": _named_value(
            config, {"angle_conv_kernel_size", "angle_kernel_size"}
        ) == 3,
        "8×8 unified reference grid": (
            _named_value(config, {"ref_grid_size", "reference_grid_size"}) == 8
            and _named_value(config, {"unified_pos", "unified_position_encoding"}) is True
        ),
        "attention component": any("attention" in name for name in class_names),
        "Radon component": any("radon" in name for name in class_names),
        "angle weighting Conv1d 1→16→1": (1, 16) in conv_pairs and (16, 1) in conv_pairs,
        "GELU angle activation": any(isinstance(module, nn.GELU) for module in modules),
        "Sigmoid angle gate": any(isinstance(module, nn.Sigmoid) for module in modules),
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return _check("model_structure", "rno_structure", not failed, "RNO structure satisfied" if not failed else "missing: " + ", ".join(failed))


def check_model_forward_executable(root: Path | None = None) -> dict[str, Any]:
    import torch

    torch.manual_seed(0)
    project = _load_project((root or _submission_dir()).resolve(), "forward")
    model_class = getattr(project.model_module, MODEL_CLASS)
    tested: list[str] = []
    for batch, height, width in ((1, 85, 85), (2, 9, 13)):
        model = _construct(model_class, height, width).cpu().eval()
        coefficient = _coefficient(batch, height, width)
        before = coefficient.clone()
        with torch.no_grad():
            positions = _positions(batch, height, width)
            field = coefficient.flatten(2).transpose(1, 2)
            raw_output = model(positions, field)
            required_shape = (batch, height * width, 1)
            if raw_output.shape != required_shape:
                raise AssertionError(
                    f"expected raw output {required_shape}, got {tuple(raw_output.shape)}"
                )
            output = _canonical_output(raw_output, batch, height, width)
        expected_shape = (batch, 1, height, width)
        if output.shape != expected_shape:
            raise AssertionError(f"expected {expected_shape}, got {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise AssertionError("forward output contains NaN or Inf")
        if not torch.equal(coefficient, before):
            raise AssertionError("forward modified its input in-place")
        tested.append(
            f"positions={(batch, height * width, 2)}, "
            f"field={(batch, height * width, 1)}, output={required_shape}"
        )
    return _check("forward_executable", "forward_shapes", True, "; ".join(tested))


def check_model_forward_equivalence(
    root: Path | None = None, standard_root: Path | None = None,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(7)
    candidate = _load_project((root or _submission_dir()).resolve(), "candidate_equivalence")
    reference = _load_project((standard_root or _standard_dir()).resolve(), "reference_equivalence")
    reference_model = _construct(getattr(reference.model_module, MODEL_CLASS), 11).cpu().eval()
    candidate_model = _construct(getattr(candidate.model_module, MODEL_CLASS), 11).cpu().eval()
    candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
    coefficient = _coefficient(2, 11, 11)
    with torch.no_grad():
        expected = _forward(reference_model, coefficient)
        actual = _forward(candidate_model, coefficient)
    passed = torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    difference = float((actual - expected).abs().max())
    return _check("forward_equivalence", "reference_forward", passed, f"maximum absolute difference: {difference:.3e}")


def check_traning(root: Path | None = None) -> dict[str, Any]:
    """Run one genuine optimization step (name retained for harness compatibility)."""
    import torch

    torch.manual_seed(11)
    project = _load_project((root or _submission_dir()).resolve(), "training")
    model = _construct(getattr(project.model_module, MODEL_CLASS), 9).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    coefficient = _coefficient(2, 9, 9)
    target = (0.2 * coefficient + coefficient.square()).detach()
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}

    optimizer.zero_grad(set_to_none=True)
    prediction = _forward(model, coefficient)
    prediction_bc = prediction.clone()
    prediction_bc[:, :, (0, -1), :] = 0.0
    prediction_bc[:, :, :, (0, -1)] = 0.0
    loss = _relative_l2(prediction, target) + 0.1 * _gradient_relative_l2(prediction_bc, target)
    if not torch.isfinite(loss):
        raise AssertionError("training loss is NaN or Inf")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("missing or non-finite gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    changed = any(not torch.equal(before[name], value.detach()) for name, value in model.named_parameters() if name in before)
    return _check("traning", "one_step_training", changed, f"finite loss={float(loss):.6f}; parameters_changed={changed}")


def check_artifacts(root: Path | None = None) -> dict[str, Any]:
    root = (root or _submission_dir()).resolve()
    source = "\n".join(
        _read(root / name).lower() for name in ("train.py", "evaluate.py", "utils.py")
        if (root / name).is_file()
    )
    names = ("last.pt", "history.json", "summary.json", "evaluation.json", "train.log")
    declared = all(name in source for name in names) and (
        "checkpoint_every" in source or "epoch_" in source
    )
    artifact_dir = os.environ.get("RNO_DARCY_TRAIN_ARTIFACT_DIR")
    if not artifact_dir:
        return _check("artifacts", "training_artifacts", declared, "artifact declarations checked; runtime directory not provided")
    output = Path(artifact_dir).expanduser().resolve()
    required = tuple(output / name for name in names)
    periodic = any(path.is_file() and path.stat().st_size > 0 for path in output.glob("epoch_*.pt"))
    present = all(path.is_file() and path.stat().st_size > 0 for path in required) and periodic
    return _check("artifacts", "training_artifacts", declared and present, f"checked artifacts in {output}")


def evaluate(
    submission_dir: Path | None = None, standard_dir: Path | None = None,
) -> dict[str, Any]:
    submission = (submission_dir or _submission_dir()).resolve()
    standard = (standard_dir or _standard_dir()).resolve()
    checks: list[dict[str, Any]] = []
    steps: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("file", lambda: check_required_files(submission)),
        ("data", lambda: check_data_contract(submission)),
        ("train_script", lambda: check_train_script_contract(submission)),
        ("eval_script", lambda: check_eval_script_contract(submission)),
        ("model_structure", lambda: check_model_structure(submission)),
        ("forward_executable", lambda: check_model_forward_executable(submission)),
        ("forward_equivalence", lambda: check_model_forward_equivalence(submission, standard)),
        ("traning", lambda: check_traning(submission)),
        ("artifacts", lambda: check_artifacts(submission)),
    )
    for check_type, step in steps:
        records = _guard(check_type, step)
        for record in records:
            if record.get("check_type") != check_type:
                record = _check(check_type, "invalid_check_type", False, f"returned {record.get('check_type')!r}")
            checks.append(record)

    required = [record for record in checks if record["check_type"] in REQUIRED_CHECK_TYPES]
    passed = bool(required) and all(record["ok"] for record in required)
    return {
        "task": TASK_NAME,
        "submission_dir": str(submission),
        "standard_dir": str(standard),
        "passed": passed,
        "summary": {
            "required_passed": sum(record["ok"] for record in required),
            "required_total": len(required),
            "optional_passed": sum(record["ok"] for record in checks if record["check_type"] not in REQUIRED_CHECK_TYPES),
            "optional_total": sum(record["check_type"] not in REQUIRED_CHECK_TYPES for record in checks),
        },
        "checks": checks,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=None)
    parser.add_argument("--standard-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = evaluate(args.submission_dir, args.standard_dir)
    except Exception as exc:
        report = {
            "task": TASK_NAME,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
