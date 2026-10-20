#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import json
import struct
import subprocess
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

import torch

from base.base_evaluator import (
    CANDIDATE_DIR,
    _load_candidate_modules,
    _load_reference_modules,
    _load_class,
    check_files,
    check_interface,
    _load_candidate_config,
    check_config as _base_check_config,
    check_infer_structure as _base_check_infer_structure,
)
from base.model_structure_checker import check_model_structure_v2, checkpoints_to_tuples

CHECKPOINT_PATH = Path(
    "/public/share/sugonhpcapp01/onestore/paper_repro/weight/"
    "best_kno_navier_stokes_v1e5.pt"
)
DATA_PATH = Path(
    "/public/share/sugonhpcapp01/onestore/onedatasets/CFD_Benchmark/ns/"
    "NavierStokes_V1e-5_N1200_T20.mat"
)
ATOL = 1e-6
RTOL = 1e-6


def check_config_content(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Validate task settings by meaning rather than requiring a flat YAML layout."""
    checks = list(_base_check_config(
        config, caller_globals=caller_globals, caller_locals=caller_locals
    ))
    try:
        settings = _load_candidate_config(config.get("config_name", "config.yaml"))
        sections = [settings]
        for group in ("root", "data", "inference"):
            section = settings.get(group)
            if isinstance(section, Mapping):
                sections.append(section)
        expected = config.get("expected", {
            "t_in": 10, "t_out": 10, "ntest": 200, "batch_size": 8,
        })
        for key, required in expected.items():
            name = f"config_content:{key}"
            try:
                values = [section[key] for section in sections if key in section]
                if not values:
                    if key == "shuffle":
                        # shuffle=False may be an explicit DataLoader default.
                        checks.append((name, True, "not configured; test-order is verified by oracle"))
                        continue
                    raise KeyError(f"{key!r} not configured")
                if any(value != required for value in values):
                    raise AssertionError(f"{key}={values!r}, expected {required!r}")
                checks.append((name, True, f"{key}={required!r} across supported YAML layouts"))
            except Exception:
                checks.append((name, False, traceback.format_exc()))
    except Exception:
        checks.append(("config_content:semantic_settings", False, traceback.format_exc()))
    return checks


def check_checkpoint_loading(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the fixed KNO model_state against both model implementations."""
    names = (
        "checkpoint_loading:assets_and_metadata",
        "checkpoint_loading:reference_strict",
        "checkpoint_loading:candidate_state_layout",
        "checkpoint_loading:candidate_strict",
    )
    checks = [(name, False, "Not executed due to earlier failure") for name in names]

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("KNO checkpoint must be a mapping")
        for key in ("model_state", "model_config", "spatial_shape"):
            if key not in checkpoint:
                raise KeyError(f"KNO checkpoint missing {key!r}")
        state = checkpoint["model_state"]
        metadata = checkpoint["model_config"]
        spatial_shape = tuple(int(dim) for dim in checkpoint["spatial_shape"])
        if not isinstance(state, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("model_state and model_config must be mappings")
        if len(spatial_shape) != 2 or min(spatial_shape) <= 0:
            raise ValueError(f"invalid spatial_shape: {spatial_shape}")
        kwargs = dict(metadata)
        kwargs.pop("name", None)
        checks[0] = (names[0], True, f"{len(state)} state tensors; spatial_shape={spatial_shape}")
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    try:
        reference_cls = _load_class(_load_reference_modules("model.py"), "KNO2DNavierStokes")
        reference = reference_cls(spatial_shape=spatial_shape, **kwargs).cpu().eval()
        reference.load_state_dict(state, strict=True)
        checks[1] = (names[1], True, "reference strict=True accepted model_state")
    except Exception:
        checks[1] = (names[1], False, traceback.format_exc())

    try:
        candidate_cls = _load_class(_load_candidate_modules("model.py"), "KNO2DNavierStokes")
        if not issubclass(candidate_cls, torch.nn.Module):
            raise TypeError("KNO2DNavierStokes must subclass nn.Module")
        candidate = candidate_cls(spatial_shape=spatial_shape, **kwargs).cpu().eval()
        candidate_state = candidate.state_dict()
        if set(candidate_state) != set(state):
            raise AssertionError(
                f"state keys mismatch: missing={sorted(set(state)-set(candidate_state))}, "
                f"unexpected={sorted(set(candidate_state)-set(state))}"
            )
        for key, tensor in state.items():
            actual = candidate_state[key]
            if actual.shape != tensor.shape or actual.dtype != tensor.dtype:
                raise AssertionError(
                    f"{key}: candidate {tuple(actual.shape)}/{actual.dtype}, "
                    f"checkpoint {tuple(tensor.shape)}/{tensor.dtype}"
                )
        checks[2] = (names[2], True, "all state keys, shapes and dtypes match")
    except Exception:
        checks[2] = (names[2], False, traceback.format_exc())
        checks[3] = (names[3], False, "Not executed because candidate construction failed")
        return checks

    try:
        incompatible = candidate.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AssertionError(f"strict load returned {incompatible}")
        for key, tensor in candidate.state_dict().items():
            torch.testing.assert_close(tensor, state[key], rtol=0.0, atol=0.0)
        checks[3] = (names[3], True, "candidate strict=True loaded exact checkpoint values")
    except Exception:
        checks[3] = (names[3], False, traceback.format_exc())
    return checks


def check_model_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Delegate declarative structure rules to the shared model checker."""
    names = (
        "model_structure:build_model",
        "model_structure:declared_rules",
        "model_structure:koopman_complex_matrix",
    )
    checks = [(name, False, "Not executed due to earlier failure") for name in names]
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        metadata = dict(checkpoint["model_config"])
        metadata.pop("name", None)
        spatial_shape = tuple(int(dim) for dim in checkpoint["spatial_shape"])
        model_cls = _load_class(_load_candidate_modules("model.py"), "KNO2DNavierStokes")
        model = model_cls(spatial_shape=spatial_shape, **metadata).cpu().eval()
        rules = config.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError("model_structure requires nonempty rules")
        checks[0] = (names[0], True, "checkpoint-configured candidate model instantiated")
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    try:
        results = checkpoints_to_tuples(
            check_model_structure_v2(model, config_dict={"rules": rules}, verbose=False)
        )
        checks.extend(results)
        failures = [f"{name}: {detail}" for name, ok, detail in results if not ok]
        checks[1] = (
            names[1], not failures,
            "; ".join(failures) if failures else f"{len(results)} structure rules passed",
        )
    except Exception:
        checks[1] = (names[1], False, traceback.format_exc())

    try:
        koopman = [m for m in model.modules() if type(m).__name__ == "KoopmanOperator2D"]
        if len(koopman) != 1:
            raise AssertionError(f"expected one KoopmanOperator2D, found {len(koopman)}")
        width = int(checkpoint["model_config"]["op_size"])
        mx = int(checkpoint["model_config"]["modes_x"])
        my = int(checkpoint["model_config"]["modes_y"])
        expected_shape = (width, width, mx, my)
        matrices = [p for p in koopman[0].parameters() if torch.is_complex(p)]
        if len(matrices) != 1 or tuple(matrices[0].shape) != expected_shape:
            raise AssertionError(
                f"expected one complex Koopman parameter {expected_shape}, "
                f"got {[tuple(p.shape) for p in matrices]}"
            )
        checks[2] = (names[2], True, f"complex Koopman matrix has shape {expected_shape}")
    except Exception:
        checks[2] = (names[2], False, traceback.format_exc())
    return checks


def check_model_forward_equivalence(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Compare KNO inference under identical real checkpoint weights."""
    names = (
        "model_forward:one_step_shapes_and_values",
        "model_forward:reference_equivalence",
        "model_forward:intermediate_activations",
        "model_forward:determinism_and_state_integrity",
    )
    checks = [(name, False, "Not executed due to earlier failure") for name in names]
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        kwargs = dict(checkpoint["model_config"])
        kwargs.pop("name", None)
        height, width = map(int, checkpoint["spatial_shape"])
        candidate_cls = _load_class(_load_candidate_modules("model.py"), "KNO2DNavierStokes")
        reference_cls = _load_class(_load_reference_modules("model.py"), "KNO2DNavierStokes")
        candidate = candidate_cls(spatial_shape=(height, width), **kwargs).cpu().eval()
        reference = reference_cls(spatial_shape=(height, width), **kwargs).cpu().eval()
        candidate.load_state_dict(checkpoint["model_state"], strict=True)
        reference.load_state_dict(checkpoint["model_state"], strict=True)
        y_axis, x_axis = torch.linspace(0, 1, height), torch.linspace(0, 1, width)
        grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
        position = torch.stack((grid_x, grid_y), dim=-1).reshape(1, height * width, 2)
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    try:
        with torch.inference_mode():
            for batch_size in (1, 2):
                pos = position.expand(batch_size, -1, -1).contiguous()
                history = torch.randn(
                    batch_size, height * width, 10,
                    generator=torch.Generator(device="cpu").manual_seed(42),
                )
                prediction = candidate(pos, history)
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError("forward must return a torch.Tensor")
                if tuple(prediction.shape) != (batch_size, height * width, 1):
                    raise AssertionError(f"invalid forward shape {tuple(prediction.shape)}")
                if not torch.isfinite(prediction).all():
                    raise AssertionError("nonfinite model predictions")
        checks[0] = (names[0], True, "B=1,2 output shapes and finite values verified")
    except Exception:
        checks[0] = (names[0], False, traceback.format_exc())

    try:
        comparisons = []
        with torch.inference_mode():
            for batch_size, seed in ((1, 11), (2, 42)):
                pos = position.expand(batch_size, -1, -1).contiguous()
                history = torch.randn(
                    batch_size, height * width, 10,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                )
                actual, expected = candidate(pos, history), reference(pos, history)
                delta = actual.double() - expected.double()
                comparisons.append({
                    "batch_size": batch_size,
                    "seed": seed,
                    "mse": float(delta.square().mean()),
                    "relative_l2": float((
                        torch.linalg.vector_norm(delta.reshape(batch_size, -1), dim=1)
                        / torch.linalg.vector_norm(
                            expected.double().reshape(batch_size, -1), dim=1
                        ).clamp_min(1e-12)
                    ).mean()),
                    "max_abs_error": float(delta.abs().max()),
                })
                torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        checks[1] = (names[1], True, json.dumps(comparisons, ensure_ascii=False))
    except Exception:
        checks[1] = (names[1], False, traceback.format_exc())

    try:
        stages = {"EncoderConv2D", "DecoderConv2D", "KoopmanOperator2D"}
        activations = {role: {stage: [] for stage in stages} for role in ("candidate", "reference")}
        pos = position.contiguous()
        history = torch.randn(
            1, height * width, 10,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        for role, model in (("candidate", candidate), ("reference", reference)):
            handles = []
            try:
                for module in model.modules():
                    stage = type(module).__name__
                    if stage in stages:
                        handles.append(module.register_forward_hook(
                            lambda _mod, _args, output, role=role, stage=stage:
                                activations[role][stage].append(output.detach().cpu())
                        ))
                with torch.inference_mode():
                    model(pos, history)
            finally:
                for handle in handles:
                    handle.remove()
        for stage in stages:
            actual_values = activations["candidate"][stage]
            expected_values = activations["reference"][stage]
            if not actual_values or len(actual_values) != len(expected_values):
                raise AssertionError(
                    f"{stage}: invalid activation call counts "
                    f"{len(actual_values)} != {len(expected_values)}"
                )
            for actual, expected in zip(actual_values, expected_values):
                torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        checks[2] = (names[2], True, "encoder, decoder, Koopman activations match")
    except Exception:
        checks[2] = (names[2], False, traceback.format_exc())

    try:
        before = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
        pos = position.contiguous()
        history = torch.randn(
            1, height * width, 10,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        if any(module.training for module in candidate.modules()):
            raise AssertionError("candidate or child module is not in eval mode")
        with torch.inference_mode():
            first = candidate(pos, history)
            second = candidate(pos, history)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        for name, value in before.items():
            torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0.0, atol=0.0)
        checks[3] = (names[3], True, "deterministic evaluation; checkpoint state unchanged")
    except Exception:
        checks[3] = (names[3], False, traceback.format_exc())
    return checks


def check_infer_structure(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Reuse the shared AST inspection, then check evaluation-mode semantics."""
    checks = list(_base_check_infer_structure(
        config, caller_globals=caller_globals, caller_locals=caller_locals
    ))
    name = "infer_structure:required_eval_no_grad"
    try:
        source = CANDIDATE_DIR / "inference.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    called.add(node.func.id)
        if "eval" not in called or not {"no_grad", "inference_mode"} & called:
            raise AssertionError("inference.py must use eval() and no_grad()/inference_mode()")
        checks.append((name, True, "eval() and no_grad()/inference_mode() detected"))
    except Exception:
        checks.append((name, False, traceback.format_exc()))
    return checks


def check_infer_execution(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Run the actual 10-step inference entrypoint and verify generated files."""
    name = "infer_execution:full_10_step_cli"
    try:
        script = CANDIDATE_DIR / "inference.py"
        if not script.is_file():
            raise FileNotFoundError(script)
        if not DATA_PATH.is_file():
            raise FileNotFoundError(f"Navier–Stokes MAT data unavailable: {DATA_PATH}")
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"KNO checkpoint unavailable: {CHECKPOINT_PATH}")
        output_files = ("prediction_sample.pt", "prediction_sample.png", "inference_summary.json")
        previous = {
            name: (CANDIDATE_DIR / name).stat().st_mtime_ns
            for name in output_files if (CANDIDATE_DIR / name).is_file()
        }
        # The prompt does not prescribe a CLI flag. Support both a config-based
        # entrypoint and a no-argument entrypoint; do not impose --device.
        source = script.read_text(encoding="utf-8")
        arguments = [sys.executable, str(script)]
        if "--config" in source:
            arguments.extend(["--config", "config.yaml"])
        completed = subprocess.run(
            arguments,
            cwd=str(CANDIDATE_DIR), capture_output=True, text=True,
            timeout=int(config.get("timeout", 300)), check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"inference.py exit={completed.returncode}; "
                f"stdout={completed.stdout[-6000:]}; stderr={completed.stderr[-6000:]}"
            )
        for filename in output_files:
            path = CANDIDATE_DIR / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"CLI did not generate {filename}")
            if filename in previous and path.stat().st_mtime_ns <= previous[filename]:
                raise AssertionError(f"CLI did not refresh {filename}")
        return [(name, True, "CLI succeeded; all three output artifacts freshly generated")]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_infer_output_schema(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the three KNO-specific artifact formats and their metadata."""
    names = (
        "infer_output:prediction_tensor",
        "infer_output:visualization_png",
        "infer_output:summary_json",
        "infer_output:artifact_metric_consistency",
    )
    checks = [(name, False, "Not executed due to earlier failure") for name in names]

    try:
        path = CANDIDATE_DIR / "prediction_sample.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(saved, Mapping):
            raise TypeError("prediction_sample.pt must store a mapping")
        for key in ("prediction", "target", "spatial_shape", "mse", "relative_l2"):
            if key not in saved:
                raise KeyError(f"prediction_sample.pt missing {key!r}")
        prediction, target = saved["prediction"], saved["target"]
        if not all(isinstance(t, torch.Tensor) for t in (prediction, target)):
            raise TypeError("prediction/target must be tensors")
        if tuple(prediction.shape) != tuple(target.shape) or prediction.shape[0] != 4:
            raise AssertionError("prediction and target must have equal first-four-sample shapes")
        if prediction.ndim == 3:
            if tuple(prediction.shape) != (4, 4096, 10):
                raise AssertionError(f"invalid flattened tensor shape: {tuple(prediction.shape)}")
        elif prediction.ndim == 4:
            if tuple(prediction.shape) != (4, 64, 64, 10):
                raise AssertionError(f"invalid spatial tensor shape: {tuple(prediction.shape)}")
        else:
            raise AssertionError(f"invalid tensor rank: {prediction.ndim}")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise AssertionError("prediction/target contain NaN or Inf")
        if tuple(saved["spatial_shape"]) != (64, 64):
            raise AssertionError(f"invalid spatial_shape: {saved['spatial_shape']}")
        checks[0] = (names[0], True, "four finite 64x64x10 prediction and target samples")
    except Exception:
        checks[0] = (names[0], False, traceback.format_exc())

    try:
        path = CANDIDATE_DIR / "prediction_sample.png"
        if not path.is_file() or path.stat().st_size < 1000:
            raise FileNotFoundError(f"PNG missing or too small: {path}")
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise ValueError("prediction_sample.png is not a valid PNG header")
        width, height = struct.unpack(">II", header[16:24])
        if min(width, height) < 300:
            raise AssertionError(f"PNG visualization too small: {width}x{height}")
        checks[1] = (names[1], True, f"valid PNG dimensions: {width}x{height}")
    except Exception:
        checks[1] = (names[1], False, traceback.format_exc())

    try:
        summary = json.loads((CANDIDATE_DIR / "inference_summary.json").read_text(encoding="utf-8"))
        if not isinstance(summary, Mapping):
            raise TypeError("inference_summary.json must contain an object")
        for key, aliases in (
            ("mse", ("mse", "MSE")),
            ("relative_l2", ("relative_l2", "Relative_L2")),
        ):
            present = [alias for alias in aliases if alias in summary]
            if not present:
                raise KeyError(f"summary missing {key} (accepted keys: {aliases})")
            values = [float(summary[alias]) for alias in present]
            if not all(torch.isfinite(torch.tensor(v)) and v >= 0 for v in values):
                raise AssertionError(f"invalid {key}: {values}")
            if any(abs(v - values[0]) > ATOL for v in values[1:]):
                raise AssertionError(f"conflicting {key} aliases: {values}")
        count_keys = ("eval_batch_size", "sample_count", "batch_size")
        present_counts = [name for name in count_keys if name in summary]
        if not present_counts or any(int(summary[name]) != 8 for name in present_counts):
            raise AssertionError("summary must report the first test batch of 8 samples")
        checks[2] = (names[2], True, "finite MSE/Relative L2; evaluated first 8 test samples")
    except Exception:
        checks[2] = (names[2], False, traceback.format_exc())

    try:
        saved = torch.load(CANDIDATE_DIR / "prediction_sample.pt", map_location="cpu", weights_only=True)
        summary = json.loads((CANDIDATE_DIR / "inference_summary.json").read_text(encoding="utf-8"))
        for key, aliases in (
            ("mse", ("mse", "MSE")),
            ("relative_l2", ("relative_l2", "Relative_L2")),
        ):
            reported = [float(summary[alias]) for alias in aliases if alias in summary]
            if not reported or any(abs(float(saved[key]) - value) > ATOL for value in reported):
                raise AssertionError(f"{key}: tensor metadata differs from JSON summary")
        checks[3] = (names[3], True, "PT and JSON metric metadata agree")
    except Exception:
        checks[3] = (names[3], False, traceback.format_exc())
    return checks


def check_evaluate_oracle(config: dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Independently validate last-200 first-batch rollout and its two metrics."""
    names = (
        "infer_oracle:dataset_last_200_first_batch",
        "infer_oracle:autoregressive_first_four",
        "infer_oracle:full_batch_mse",
        "infer_oracle:full_batch_relative_l2",
    )
    checks = [(name, False, "Not executed due to earlier failure") for name in names]
    try:
        import numpy as np
        from scipy.io import loadmat

        data = loadmat(DATA_PATH)
        if "u" not in data:
            raise KeyError("Navier–Stokes MAT data must contain 'u'")
        flow = np.asarray(data["u"], dtype=np.float32)
        if flow.shape != (1200, 64, 64, 20):
            raise AssertionError(f"expected data [1200,64,64,20], got {flow.shape}")
        first_batch = flow[-200:][:8]
        target = torch.from_numpy(np.array(first_batch[..., 10:20], copy=True)).reshape(8, 4096, 10)
        history = torch.from_numpy(np.array(first_batch[..., :10], copy=True)).reshape(8, 4096, 10)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        kwargs = dict(checkpoint["model_config"])
        kwargs.pop("name", None)
        spatial_shape = tuple(int(dim) for dim in checkpoint["spatial_shape"])
        if spatial_shape != (64, 64):
            raise AssertionError(f"checkpoint grid size mismatches MAT data: {spatial_shape}")
        reference_cls = _load_class(_load_reference_modules("model.py"), "KNO2DNavierStokes")
        reference = reference_cls(spatial_shape=spatial_shape, **kwargs).cpu().eval()
        reference.load_state_dict(checkpoint["model_state"], strict=True)
        y_axis, x_axis = torch.linspace(0, 1, 64), torch.linspace(0, 1, 64)
        grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
        positions = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 4096, 2).expand(8, -1, -1).contiguous()
        frames = []
        with torch.inference_mode():
            for step in range(10):
                frame = reference(positions, history)
                if tuple(frame.shape) != (8, 4096, 1):
                    raise AssertionError(f"reference step {step} returned {tuple(frame.shape)}")
                frames.append(frame)
                history = torch.cat((history[..., 1:], frame), dim=-1)
        expected = torch.cat(frames, dim=-1)
        residual = expected.double() - target.double()
        expected_mse = float(residual.square().mean())
        expected_l2 = float((
            torch.linalg.vector_norm(residual.reshape(8, -1), dim=1)
            / torch.linalg.vector_norm(target.double().reshape(8, -1), dim=1).clamp_min(1e-8)
        ).mean())
        saved = torch.load(CANDIDATE_DIR / "prediction_sample.pt", map_location="cpu", weights_only=True)
        summary = json.loads((CANDIDATE_DIR / "inference_summary.json").read_text(encoding="utf-8"))
    except Exception:
        return [(name, False, traceback.format_exc()) for name in names]

    try:
        recorded_truth = saved["target"].detach().cpu().reshape(4, 4096, 10)
        torch.testing.assert_close(recorded_truth, target[:4], rtol=0.0, atol=0.0)
        checks[0] = (names[0], True, "first four targets match global test indices 1000-1003")
    except Exception:
        checks[0] = (names[0], False, traceback.format_exc())

    try:
        recorded = saved["prediction"].detach().cpu().reshape(4, 4096, 10)
        torch.testing.assert_close(recorded, expected[:4], rtol=RTOL, atol=ATOL)
        checks[1] = (names[1], True, "saved first-four 10-step rollout matches reference")
    except Exception:
        checks[1] = (names[1], False, traceback.format_exc())

    try:
        reported = [("PT", saved["mse"])] + [
            ("summary:" + key, summary[key]) for key in ("mse", "MSE") if key in summary
        ]
        if len(reported) == 1:
            raise KeyError("summary must report MSE")
        for label, value in reported:
            if abs(float(value) - expected_mse) > max(ATOL, RTOL * abs(expected_mse)):
                raise AssertionError(f"{label} MSE={value}, reference={expected_mse}")
        checks[2] = (names[2], True, f"whole-batch MSE={expected_mse:.9g}")
    except Exception:
        checks[2] = (names[2], False, traceback.format_exc())

    try:
        reported = [("PT", saved["relative_l2"])] + [
            ("summary:" + key, summary[key])
            for key in ("relative_l2", "Relative_L2") if key in summary
        ]
        if len(reported) == 1:
            raise KeyError("summary must report Relative L2")
        for label, value in reported:
            if abs(float(value) - expected_l2) > max(ATOL, RTOL * abs(expected_l2)):
                raise AssertionError(f"{label} Relative L2={value}, reference={expected_l2}")
        checks[3] = (names[3], True, f"batch-mean flattened Relative L2={expected_l2:.9g}")
    except Exception:
        checks[3] = (names[3], False, traceback.format_exc())
    return checks
