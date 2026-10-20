#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task-specific evaluator methods for the standalone AlphaGenome-style case.

Reuse shared evaluator utilities whenever their contracts match the task. Keep
only AlphaGenome-specific semantic checks in this module.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch
from torch import nn


os.environ.setdefault("candidate_env", "ALPHAGENOME_SUBMISSION_DIR")
os.environ.setdefault("candidate_dir_hints", "alphagenome_inference")
os.environ.setdefault("standard_name", "alphagenome_inference")

_candidate_env_name = os.environ["candidate_env"]
_artifact_dir = os.environ.get("EVAL_ARTIFACT_DIR")
if _artifact_dir and not os.environ.get(_candidate_env_name):
    os.environ[_candidate_env_name] = _artifact_dir


from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _initialize_parameters_with_constants,
    _load_candidate_modules,
    _load_class,
    _load_reference_modules,
    check_files,
    check_infer_structure as _base_check_infer_structure,
    check_interface,
)
from base.model_structure_checker import check_model_structure_v2
from base.model_structure_checker import checkpoints_to_tuples


EXPECTED_TRACK_CHANNELS = {
    "atac": 3,
    "dnase": 2,
    "cage": 4,
    "rna_seq": 2,
    "chip_tf": 5,
    "chip_histone": 3,
}
RTOL = 1e-6
ATOL = 1e-7
SEED = 20260729


def check_method_correctness(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Validate AlphaGenome-specific helper/component semantics."""
    scope = config.get("scope")

    if scope == "dna":
        checks = [
            ("method_correctness:dna_preprocessing", False, "Not executed due to earlier failure"),
            ("method_correctness:dna_channel_mapping", False, "Not executed due to earlier failure"),
            ("method_correctness:invalid_dna_rejection", False, "Not executed due to earlier failure"),
        ]
        try:
            candidate_module = _load_candidate_modules("run_inference.py")
            reference_module = _load_reference_modules("run_inference.py")
        except Exception:
            detail = traceback.format_exc()
            return [(name, False, detail) for name, _, _ in checks]

        try:
            sequences = [" acgtACGT ", "\ntgcaTGCA\n"]
            candidate = candidate_module.encode_dna(sequences)
            reference = reference_module.encode_dna(sequences)
            if not isinstance(candidate, torch.Tensor):
                raise AssertionError("encode_dna must return torch.Tensor")
            if candidate.dtype != torch.float32:
                raise AssertionError(f"encode_dna dtype={candidate.dtype}, expected=torch.float32")
            if tuple(candidate.shape) != (2, 8, 4):
                raise AssertionError(f"encode_dna shape={tuple(candidate.shape)}, expected=(2, 8, 4)")
            torch.testing.assert_close(candidate, reference, rtol=0.0, atol=0.0)
            torch.testing.assert_close(candidate.sum(dim=-1), torch.ones(2, 8), rtol=0.0, atol=0.0)
            if not torch.all((candidate == 0) | (candidate == 1)):
                raise AssertionError("encode_dna must produce one-hot values")
            checks[0] = (checks[0][0], True, "DNA preprocessing matches the reference")
        except Exception:
            checks[0] = (checks[0][0], False, traceback.format_exc())

        try:
            sequences = ["AAAA", "CCCC", "GGGG", "TTTT", "acgt", " tgca "]
            candidate = candidate_module.encode_dna(sequences)
            reference = reference_module.encode_dna(sequences)
            torch.testing.assert_close(candidate, reference, rtol=0.0, atol=0.0)
            expected_channels = torch.tensor(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [0, 1, 2, 3],
                    [3, 2, 1, 0],
                ],
                dtype=torch.long,
            )
            torch.testing.assert_close(candidate.argmax(dim=-1), expected_channels, rtol=0.0, atol=0.0)
            round_trip = "".join(
                reference_module.DNA_ALPHABET[index]
                for index in candidate[-1].argmax(dim=-1).tolist()
            )
            if round_trip != "TGCA":
                raise AssertionError(f"DNA channel round trip={round_trip!r}, expected='TGCA'")
            checks[1] = (checks[1][0], True, "A/C/G/T channel mapping matches the reference")
        except Exception:
            checks[1] = (checks[1][0], False, traceback.format_exc())

        try:
            invalid_batches = [
                ("empty DNA batch", []),
                ("too-short DNA sequence", ["ACG"]),
                ("unequal DNA batch lengths", ["ACGT", "ACGTA"]),
                ("unsupported DNA base", ["ACGN"]),
            ]
            for description, value in invalid_batches:
                try:
                    candidate_module.encode_dna(value)
                except ValueError:
                    continue
                except Exception as error:
                    raise AssertionError(
                        f"{description} raised {type(error).__name__}, expected ValueError"
                    ) from error
                raise AssertionError(f"{description} did not raise ValueError")
            checks[2] = (checks[2][0], True, "all invalid DNA inputs raise ValueError")
        except Exception:
            checks[2] = (checks[2][0], False, traceback.format_exc())

        return checks

    if scope == "model_components":
        checks = [
            ("method_correctness:residual_block", False, "Not executed due to earlier failure"),
            ("method_correctness:transformer_block", False, "Not executed due to earlier failure"),
        ]
        try:
            candidate_module = _load_candidate_modules("run_inference.py")
            reference_module = _load_reference_modules("run_inference.py")
        except Exception:
            detail = traceback.format_exc()
            return [(name, False, detail) for name, _, _ in checks]

        try:
            torch.manual_seed(SEED)
            reference = reference_module.ResidualConvBlock(16).eval()
            candidate = candidate_module.ResidualConvBlock(16).eval()
            candidate.load_state_dict(reference.state_dict(), strict=True)
            value = torch.linspace(-1.0, 1.0, steps=2 * 16 * 9).reshape(2, 16, 9)
            with torch.no_grad():
                expected = reference(value)
                actual = candidate(value)
            if tuple(actual.shape) != tuple(value.shape):
                raise AssertionError("ResidualConvBlock must preserve input shape")
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
            checks[0] = (checks[0][0], True, "ResidualConvBlock matches the reference")
        except Exception:
            checks[0] = (checks[0][0], False, traceback.format_exc())

        try:
            torch.manual_seed(SEED)
            reference = reference_module.AlphaGenomeTransformerBlock(32, 4, 64).eval()
            candidate = candidate_module.AlphaGenomeTransformerBlock(32, 4, 64).eval()
            candidate.load_state_dict(reference.state_dict(), strict=True)
            value = torch.linspace(-0.75, 0.75, steps=2 * 7 * 32).reshape(2, 7, 32)
            with torch.no_grad():
                expected = reference(value)
                actual = candidate(value)
            if tuple(actual.shape) != tuple(value.shape):
                raise AssertionError("AlphaGenomeTransformerBlock must preserve input shape")
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
            checks[1] = (checks[1][0], True, "AlphaGenomeTransformerBlock matches the reference")
        except Exception:
            checks[1] = (checks[1][0], False, traceback.format_exc())

        return checks

    if scope == "custom_configuration":
        name = "method_correctness:custom_configuration"
        try:
            candidate_module = _load_candidate_modules("run_inference.py")
            reference_module = _load_reference_modules("run_inference.py")
            configuration = {
                "hidden_dim": 8,
                "num_heads": 4,
                "num_transformer_layers": 1,
                "track_channels": {"signal": 1, "accessibility": 2},
            }
            torch.manual_seed(20260731)
            reference = reference_module.AlphaGenomeModel(**configuration).eval()
            candidate = candidate_module.AlphaGenomeModel(**configuration).eval()
            candidate.load_state_dict(reference.state_dict(), strict=True)
            for sequences in (["ACGTA", "TGCAT"], ["ACGTACGTACGTA"]):
                reference_input = reference_module.encode_dna(sequences)
                candidate_input = candidate_module.encode_dna(sequences)
                with torch.no_grad():
                    expected = reference(reference_input)
                    actual = candidate(candidate_input)
                if set(actual) != {"signal", "accessibility"}:
                    raise AssertionError(f"custom output keys={sorted(actual)}")
                for track_name, channels in {"signal": 1, "accessibility": 2}.items():
                    expected_shape = (len(sequences), reference_input.shape[1], channels)
                    if tuple(actual[track_name].shape) != expected_shape:
                        raise AssertionError(
                            f"{track_name} shape={tuple(actual[track_name].shape)}, expected={expected_shape}"
                        )
                    torch.testing.assert_close(actual[track_name], expected[track_name], rtol=RTOL, atol=ATOL)
            return [(name, True, "custom AlphaGenome configuration matches the reference")]
        except Exception:
            return [(name, False, traceback.format_exc())]

    return [("method_correctness:scope", False, f"Unsupported scope: {scope}")]


def check_model_structure(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Validate the declared AlphaGenome architecture using shared structure rules."""
    name = "model_structure:declared_rules"
    try:
        candidate_module = _load_candidate_modules("run_inference.py")
        model_cls = _load_class(candidate_module, "AlphaGenomeModel")
        model = model_cls().to(DEVICE).eval()
        rules = config.get("rules")
        if not rules:
            raise ValueError("config['rules'] is missing or empty")
        results = checkpoints_to_tuples(
            check_model_structure_v2(
                model,
                config_dict={"rules": rules},
                verbose=False,
            )
        )
        failures = [f"{check}: {detail}" for check, ok, detail in results if not ok]
        if failures:
            raise AssertionError("; ".join(failures))
        return [(name, True, "AlphaGenomeModel matches the declared structure rules")]
    except Exception:
        return [(name, False, traceback.format_exc())]


def check_model_forward_equivalence(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Preserve the old evaluator's forward/reference coverage."""
    checks = [
        ("forward_equivalence:default_forward", False, "Not executed due to earlier failure"),
        ("forward_equivalence:intermediate_activations", False, "Not executed due to earlier failure"),
        ("forward_equivalence:length_sweep", False, "Not executed due to earlier failure"),
        ("forward_equivalence:batch_isolation", False, "Not executed due to earlier failure"),
        ("forward_equivalence:determinism_state", False, "Not executed due to earlier failure"),
    ]

    try:
        candidate_module = _load_candidate_modules("run_inference.py")
        reference_module = _load_reference_modules("run_inference.py")
        candidate_cls = _load_class(candidate_module, "AlphaGenomeModel")
        reference_cls = _load_class(reference_module, "AlphaGenomeModel")
        torch.manual_seed(SEED)
        reference_model = reference_cls().to(DEVICE).eval()
        candidate_model = candidate_cls().to(DEVICE).eval()
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
    except Exception:
        detail = traceback.format_exc()
        return [(name, False, detail) for name, _, _ in checks]

    try:
        for sequences in (
            ["ACGT", "TGCA"],
            ["ACGTACGTA", "TGCATGCAT"],
            ["ACGTACGTACGTACGT", "TGCATGCATGCATGCA"],
        ):
            reference_input = reference_module.encode_dna(sequences).to(DEVICE)
            candidate_input = candidate_module.encode_dna(sequences).to(DEVICE)
            torch.testing.assert_close(candidate_input, reference_input, rtol=0.0, atol=0.0)
            with torch.no_grad():
                expected = reference_model(reference_input)
                actual = candidate_model(candidate_input)
            if not isinstance(actual, dict) or set(actual) != set(EXPECTED_TRACK_CHANNELS):
                raise AssertionError(f"forward output keys={sorted(actual) if isinstance(actual, dict) else type(actual)}")
            length = reference_input.shape[1]
            for track_name, channels in EXPECTED_TRACK_CHANNELS.items():
                expected_shape = (2, length, channels)
                value = actual[track_name]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise AssertionError(f"{track_name} shape={getattr(value, 'shape', None)}, expected={expected_shape}")
                if value.dtype != expected[track_name].dtype:
                    raise AssertionError(f"{track_name} dtype={value.dtype}, expected={expected[track_name].dtype}")
                if not torch.isfinite(value).all():
                    raise AssertionError(f"{track_name} contains NaN or Inf")
                if torch.any(value <= 0):
                    raise AssertionError(f"{track_name} must contain positive softplus outputs")
                torch.testing.assert_close(value, expected[track_name], rtol=RTOL, atol=ATOL)
        checks[0] = (checks[0][0], True, "default forward outputs match the reference")
    except Exception:
        checks[0] = (checks[0][0], False, traceback.format_exc())

    try:
        reference_modules = {
            "stem": reference_model.stem,
            "encoder": reference_model.encoder,
            "downsample": reference_model.downsample,
            "transformer.0": reference_model.transformer[0],
            "transformer.1": reference_model.transformer[1],
            "bottleneck_norm": reference_model.bottleneck_norm,
            "upsample_projection": reference_model.upsample_projection,
            "decoder": reference_model.decoder,
        }
        candidate_modules = {
            "stem": candidate_model.stem,
            "encoder": candidate_model.encoder,
            "downsample": candidate_model.downsample,
            "transformer.0": candidate_model.transformer[0],
            "transformer.1": candidate_model.transformer[1],
            "bottleneck_norm": candidate_model.bottleneck_norm,
            "upsample_projection": candidate_model.upsample_projection,
            "decoder": candidate_model.decoder,
        }
        for track_name in EXPECTED_TRACK_CHANNELS:
            reference_modules[f"output_heads.{track_name}"] = reference_model.output_heads[track_name]
            candidate_modules[f"output_heads.{track_name}"] = candidate_model.output_heads[track_name]

        reference_activations = {}
        candidate_activations = {}
        handles = []
        for stage_name, stage_module in reference_modules.items():
            handles.append(
                stage_module.register_forward_hook(
                    lambda _module, _inputs, output, name=stage_name: reference_activations.__setitem__(
                        name, output.detach().cpu()
                    )
                )
            )
        for stage_name, stage_module in candidate_modules.items():
            handles.append(
                stage_module.register_forward_hook(
                    lambda _module, _inputs, output, name=stage_name: candidate_activations.__setitem__(
                        name, output.detach().cpu()
                    )
                )
            )
        try:
            sequences = ["ACGTTGC", "TATACGA"]
            with torch.no_grad():
                reference_model(reference_module.encode_dna(sequences).to(DEVICE))
                candidate_model(candidate_module.encode_dna(sequences).to(DEVICE))
        finally:
            for handle in handles:
                handle.remove()
        if set(candidate_activations) != set(reference_activations):
            raise AssertionError("intermediate activation stages differ from reference")
        for stage_name, expected in reference_activations.items():
            actual = candidate_activations[stage_name]
            if tuple(actual.shape) != tuple(expected.shape):
                raise AssertionError(f"stage={stage_name} shape mismatch")
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        checks[1] = (checks[1][0], True, "intermediate activations match the reference")
    except Exception:
        checks[1] = (checks[1][0], False, traceback.format_exc())

    try:
        sequence_batches = [
            ["AAAA", "CGTA", "TTTT"],
            ["ACGTA", "TGCAT"],
            ["AAAAAAA", "CCCCCCC", "GGGGGGG", "TTTTTTT"],
            ["ACGTACGT", "TGCATGCA"],
            ["ACGTACGTACGTACG", "TGCATGCATGCATGC"],
            ["ACGTACGTACGTACGTACGTACGTACGTACG"],
        ]
        for sequences in sequence_batches:
            reference_input = reference_module.encode_dna(sequences).to(DEVICE)
            candidate_input = candidate_module.encode_dna(sequences).to(DEVICE)
            with torch.no_grad():
                expected = reference_model(reference_input)
                actual = candidate_model(candidate_input)
            for track_name in EXPECTED_TRACK_CHANNELS:
                torch.testing.assert_close(actual[track_name], expected[track_name], rtol=RTOL, atol=ATOL)
        checks[2] = (checks[2][0], True, "six-batch length sweep matches the reference")
    except Exception:
        checks[2] = (checks[2][0], False, traceback.format_exc())

    try:
        sequences = ["ACGTACGTACG", "TTTTGGGGCCC", "CGCGATATGCA"]
        batch_input = candidate_module.encode_dna(sequences).to(DEVICE)
        with torch.no_grad():
            batch_outputs = candidate_model(batch_input)
            for batch_index, sequence in enumerate(sequences):
                single_outputs = candidate_model(candidate_module.encode_dna(sequence).to(DEVICE))
                for track_name in EXPECTED_TRACK_CHANNELS:
                    torch.testing.assert_close(
                        batch_outputs[track_name][batch_index],
                        single_outputs[track_name][0],
                        rtol=RTOL,
                        atol=ATOL,
                    )
        checks[3] = (checks[3][0], True, "batched and single-sequence inference are isolated")
    except Exception:
        checks[3] = (checks[3][0], False, traceback.format_exc())

    try:
        if any(module.training for module in candidate_model.modules()):
            raise AssertionError("candidate model or child module remained in training mode")
        state_before = {
            name: value.detach().clone()
            for name, value in candidate_model.state_dict().items()
        }
        inputs = candidate_module.encode_dna(["GATTACAGATTACA", "CCCCAAAATTTTGG"]).to(DEVICE)
        repeated = []
        for _ in range(3):
            with torch.no_grad():
                repeated.append(candidate_model(inputs))
        for repeat_index in range(1, len(repeated)):
            for track_name in EXPECTED_TRACK_CHANNELS:
                if repeated[repeat_index][track_name].requires_grad:
                    raise AssertionError(f"{track_name} unexpectedly tracks gradients")
                torch.testing.assert_close(
                    repeated[repeat_index][track_name],
                    repeated[0][track_name],
                    rtol=0.0,
                    atol=0.0,
                )
        for name, expected in state_before.items():
            torch.testing.assert_close(candidate_model.state_dict()[name], expected, rtol=0.0, atol=0.0)
        checks[4] = (checks[4][0], True, "inference is deterministic and state-preserving")
    except Exception:
        checks[4] = (checks[4][0], False, traceback.format_exc())

    return checks


def check_checkpoint_loading(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Validate the raw/wrapped strict-loading contract from the original evaluator."""
    checks = [
        ("checkpoint_loading:strict_reference_layout", False, "Not executed due to earlier failure"),
        ("checkpoint_loading:raw_and_wrapped_cli", False, "Not executed due to earlier failure"),
        ("checkpoint_loading:reject_incompatible", False, "Not executed due to earlier failure"),
    ]
    try:
        candidate_module = _load_candidate_modules("run_inference.py")
        reference_module = _load_reference_modules("run_inference.py")
        candidate_cls = _load_class(candidate_module, "AlphaGenomeModel")
        reference_cls = _load_class(reference_module, "AlphaGenomeModel")
        torch.manual_seed(SEED)
        reference_model = reference_cls().cpu().eval()
        candidate_model = candidate_cls().cpu().eval()
        reference_state = reference_model.state_dict()
        if set(candidate_model.state_dict()) != set(reference_state):
            raise AssertionError("candidate state_dict keys differ from reference")
        candidate_model.load_state_dict(reference_state, strict=True)
        checks[0] = (checks[0][0], True, "reference state_dict loads with strict=True")
    except Exception:
        detail = traceback.format_exc()
        return [(checks[0][0], False, detail), (checks[1][0], False, detail), (checks[2][0], False, detail)]

    timeout = int(config.get("timeout", 90))
    sequence = str(config.get("sequence", "ACGTACGTACGTACGT"))
    script_path = (CANDIDATE_DIR / "run_inference.py").resolve()

    try:
        with tempfile.TemporaryDirectory(prefix="alphagenome_checkpoint_") as temporary:
            root = Path(temporary)
            raw_checkpoint = root / "weights.pt"
            wrapped_checkpoint = root / "wrapped.pt"
            torch.save(reference_state, raw_checkpoint)
            torch.save({"model_state_dict": reference_state}, wrapped_checkpoint)
            for label, checkpoint in (("raw", raw_checkpoint), ("wrapped", wrapped_checkpoint)):
                output_dir = root / f"{label}_outputs"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script_path),
                        "--sequence",
                        sequence,
                        "--checkpoint",
                        str(checkpoint),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cpu",
                    ],
                    cwd=str(script_path.parent),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{label} checkpoint CLI failed\nstdout={completed.stdout}\nstderr={completed.stderr}"
                    )
            checks[1] = (checks[1][0], True, "raw and model_state_dict checkpoints are accepted")
    except Exception:
        checks[1] = (checks[1][0], False, traceback.format_exc())

    try:
        with tempfile.TemporaryDirectory(prefix="alphagenome_bad_checkpoint_") as temporary:
            root = Path(temporary)
            invalid_state = dict(reference_state)
            invalid_state.pop(next(iter(invalid_state)))
            invalid_checkpoint = root / "invalid.pt"
            torch.save(invalid_state, invalid_checkpoint)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--sequence",
                    sequence,
                    "--checkpoint",
                    str(invalid_checkpoint),
                    "--output-dir",
                    str(root / "outputs"),
                    "--device",
                    "cpu",
                ],
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode == 0:
                raise AssertionError("incompatible checkpoint unexpectedly succeeded")
            checks[2] = (checks[2][0], True, "incompatible checkpoint is rejected")
    except Exception:
        checks[2] = (checks[2][0], False, traceback.format_exc())

    return checks


def check_infer_structure(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Reuse the shared inference AST check, then enforce AlphaGenome-only source constraints."""
    checks = [
        ("infer_structure:base_workflow", False, "Not executed due to earlier failure"),
        ("infer_structure:task_contract", False, "Not executed due to earlier failure"),
    ]

    try:
        base_results = _base_check_infer_structure(
            {
                "files": ["run_inference.py"],
                "check_config": {
                    "entry_function_names": ["main"],
                    "model_load_functions": ["load_checkpoint", "load_model", "load_weights"],
                    "model_load_attributes": ["load_state_dict"],
                    "input_parameter_names": ["sequence", "input", "dna"],
                    "input_load_functions": ["encode_dna"],
                    "input_load_attributes": [],
                    "infer_call_functions": ["main", "inference", "infer"],
                    "infer_call_attributes": ["forward"],
                    "output_attributes": ["save", "write", "write_text"],
                },
            },
            caller_globals=caller_globals,
            caller_locals=caller_locals,
        )
        failures = [f"{name}: {detail}" for name, ok, detail in base_results if not ok]
        if failures:
            raise AssertionError("; ".join(failures))
        checks[0] = (checks[0][0], True, "shared inference workflow check passed")
    except Exception:
        checks[0] = (checks[0][0], False, traceback.format_exc())

    try:
        script_path = (CANDIDATE_DIR / "run_inference.py").resolve()
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script_path))
        imported_roots = set()
        parser_flags = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_argument" and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        parser_flags.add(first.value)
        if "onescience" in imported_roots:
            raise AssertionError("candidate must not import onescience")
        required_flags = {"--sequence", "--checkpoint", "--output-dir", "--device"}
        missing_flags = sorted(required_flags - parser_flags)
        if missing_flags:
            raise AssertionError(f"missing required CLI flags: {missing_flags}")

        candidate_module = _load_candidate_modules("run_inference.py")
        for class_name in (
            "ResidualConvBlock",
            "AlphaGenomeTransformerBlock",
            "AlphaGenomeModel",
        ):
            model_class = _load_class(candidate_module, class_name)
            if not issubclass(model_class, nn.Module):
                raise AssertionError(f"{class_name} must inherit torch.nn.Module")

        checks[1] = (
            checks[1][0],
            True,
            "forbidden import, CLI flags, and nn.Module inheritance contract passed",
        )
    except Exception:
        checks[1] = (checks[1][0], False, traceback.format_exc())

    return checks


def check_infer_execution(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Run the real CLI and preserve the original six-track output oracle."""
    checks = [
        ("infer_execution:six_track_outputs", False, "Not executed due to earlier failure"),
        ("infer_execution:invalid_sequence", False, "Not executed due to earlier failure"),
        ("infer_execution:invalid_checkpoint", False, "Not executed due to earlier failure"),
    ]

    timeout = int(config.get("timeout", 90))
    sequence = str(config.get("sequence", "ACGTACGTACGTACGT"))
    script_path = (CANDIDATE_DIR / "run_inference.py").resolve()

    try:
        reference_module = _load_reference_modules("run_inference.py")
        reference_cls = _load_class(reference_module, "AlphaGenomeModel")
        torch.manual_seed(SEED)
        reference_model = reference_cls().cpu().eval()
        reference_state = reference_model.state_dict()
    except Exception:
        detail = traceback.format_exc()
        return [(name, False, detail) for name, _, _ in checks]

    with tempfile.TemporaryDirectory(prefix="alphagenome_cli_") as temporary:
        root = Path(temporary)
        raw_checkpoint = root / "weights.pt"
        wrapped_checkpoint = root / "wrapped.pt"
        invalid_checkpoint = root / "invalid.pt"
        torch.save(reference_state, raw_checkpoint)
        torch.save({"model_state_dict": reference_state}, wrapped_checkpoint)
        invalid_state = dict(reference_state)
        invalid_state.pop(next(iter(invalid_state)))
        torch.save(invalid_state, invalid_checkpoint)

        try:
            with torch.no_grad():
                expected_outputs = reference_model(reference_module.encode_dna(sequence))
            expected_files = {f"{name}.npy" for name in EXPECTED_TRACK_CHANNELS}
            for label, checkpoint in (("raw", raw_checkpoint), ("wrapped", wrapped_checkpoint)):
                output_dir = root / f"{label}_outputs"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script_path),
                        "--sequence",
                        sequence,
                        "--checkpoint",
                        str(checkpoint),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cpu",
                    ],
                    cwd=str(script_path.parent),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{label} CLI failed\nstdout={completed.stdout}\nstderr={completed.stderr}"
                    )
                actual_files = {path.name for path in output_dir.glob("*.npy")}
                if actual_files != expected_files:
                    raise AssertionError(
                        f"output files={sorted(actual_files)}, expected={sorted(expected_files)}"
                    )
                for track_name, channels in EXPECTED_TRACK_CHANNELS.items():
                    actual = np.load(output_dir / f"{track_name}.npy", allow_pickle=False)
                    expected = expected_outputs[track_name][0].detach().cpu().numpy()
                    expected_shape = (len(sequence), channels)
                    if actual.shape != expected_shape:
                        raise AssertionError(f"{track_name} shape={actual.shape}, expected={expected_shape}")
                    if actual.dtype != expected.dtype:
                        raise AssertionError(f"{track_name} dtype={actual.dtype}, expected={expected.dtype}")
                    if not np.isfinite(actual).all():
                        raise AssertionError(f"{track_name} contains NaN or Inf")
                    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
            checks[0] = (checks[0][0], True, "raw/wrapped CLI outputs match the six-track reference oracle")
        except Exception:
            checks[0] = (checks[0][0], False, traceback.format_exc())

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--sequence",
                    "ACGN",
                    "--checkpoint",
                    str(raw_checkpoint),
                    "--output-dir",
                    str(root / "invalid_sequence_outputs"),
                    "--device",
                    "cpu",
                ],
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode == 0:
                raise AssertionError("invalid DNA sequence unexpectedly succeeded")
            if not completed.stdout.strip() and not completed.stderr.strip():
                raise AssertionError("invalid DNA sequence failed without an error message")
            checks[1] = (checks[1][0], True, "invalid DNA sequence is rejected")
        except Exception:
            checks[1] = (checks[1][0], False, traceback.format_exc())

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--sequence",
                    sequence,
                    "--checkpoint",
                    str(invalid_checkpoint),
                    "--output-dir",
                    str(root / "invalid_checkpoint_outputs"),
                    "--device",
                    "cpu",
                ],
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode == 0:
                raise AssertionError("invalid checkpoint unexpectedly succeeded")
            if not completed.stdout.strip() and not completed.stderr.strip():
                raise AssertionError("invalid checkpoint failed without an error message")
            checks[2] = (checks[2][0], True, "invalid checkpoint is rejected")
        except Exception:
            checks[2] = (checks[2][0], False, traceback.format_exc())

    return checks
