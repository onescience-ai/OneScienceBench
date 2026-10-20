"""Task-specific evaluator methods for the standalone AlphaFold3-style case.

Reuse shared evaluator utilities whenever their contracts match the task. Keep
only AlphaFold3-specific prompt validation in this module.
"""

from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import torch
from torch import nn

# Shared evaluator configuration.
os.environ.setdefault(
    "candidate_env",
    "ALPHAFOLD3_SUBMISSION_DIR",
)
os.environ.setdefault(
    "candidate_dir_hints",
    "alphafold3_inference",
)
os.environ.setdefault(
    "standard_name",
    "alphafold3_inference",
)
# parallel_eval provides the authoritative candidate artifact directory.
_candidate_env_name = os.environ["candidate_env"]
_artifact_dir = os.environ.get("EVAL_ARTIFACT_DIR")

if (
    _artifact_dir
    and not os.environ.get(_candidate_env_name)
):
    os.environ[_candidate_env_name] = _artifact_dir


from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _initialize_parameters_with_constants,
    _load_candidate_modules,
    _load_class,
    _load_reference_modules,
    _move_tensors_to_device,
    check_files,
    check_infer_structure as _base_check_infer_structure,
    check_interface
    )
from base.model_structure_checker import check_model_structure_v2
from base.model_structure_checker import checkpoints_to_tuples


REQUIRED_CLASSES = {
    "TargetEmbedder",
    "PairFormerIteration",
    "DiffusionHead",
    "ConfidenceHead",
    "AlphaFold3",
}
REQUIRED_FUNCTIONS = {
    "sequence_from_json",
    "make_features",
    "write_pdb",
}
REQUIRED_FLAGS = {
    "--json-path",
    "--model-dir",
    "--output-dir",
    "--run-data-pipeline",
    "--run-inference",
    "--flash-attention-implementation",
    "--num-recycles",
    "--num-diffusion-samples",
    "--seed",
    "--device",
}
FORWARD_OUTPUT_KEYS = {
    "diffusion_samples",
    "atom_positions",
    "single_embeddings",
    "pair_embeddings",
    "predicted_lddt",
    "predicted_experimentally_resolved",
    "full_pae",
    "average_pae",
}
RTOL = 1e-6
ATOL = 1e-7


def check_method_correctness(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Validate hidden component APIs or prompt-required inference helpers."""
    scope = config.get("scope")

    if scope == "model_components":
        CHECKPOINT_NAMES = [
            "model_components:target_embedder",
            "model_components:pairformer",
            "model_components:diffusion_head",
            "model_components:confidence_head",
        ]
    elif scope == "inference_helpers":
        CHECKPOINT_NAMES = [
            "inference_helpers:sequence_from_json",
            "inference_helpers:make_features",
            "inference_helpers:write_pdb",
        ]
    else:
        return [
            (
                "method_correctness:scope",
                False,
                f"Unsupported scope: {scope}",
            )
        ]

    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    try:
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        reference_module = _load_reference_modules("alphafold3_inference.py")
    except Exception:
        error_detail = traceback.format_exc()
        return [(name, False, error_detail) for name in CHECKPOINT_NAMES]

    if scope == "model_components":
        current_index = 0
        try:
            candidate = candidate_module.TargetEmbedder(22, 16, 12).eval()
            reference = reference_module.TargetEmbedder(22, 16, 12).eval()
            _initialize_parameters_with_constants(candidate)
            _initialize_parameters_with_constants(reference)
            target = torch.linspace(-1.0, 1.0, steps=2 * 5 * 22).reshape(2, 5, 22)
            residue_index = torch.arange(5, dtype=torch.float32).repeat(2, 1)
            with torch.no_grad():
                candidate_output = candidate(target, residue_index)
                reference_output = reference(target, residue_index)
            if not isinstance(candidate_output, (tuple, list)) or len(candidate_output) != 2:
                raise AssertionError("TargetEmbedder must return (single, pair)")
            torch.testing.assert_close(candidate_output[0], reference_output[0], rtol=RTOL, atol=ATOL)
            torch.testing.assert_close(candidate_output[1], reference_output[1], rtol=RTOL, atol=ATOL)
            if tuple(candidate_output[0].shape) != (2, 5, 16):
                raise AssertionError(f"TargetEmbedder single shape={tuple(candidate_output[0].shape)}")
            if tuple(candidate_output[1].shape) != (2, 5, 5, 12):
                raise AssertionError(f"TargetEmbedder pair shape={tuple(candidate_output[1].shape)}")
            update_check(current_index, True, "TargetEmbedder hidden API and reference behavior passed")
        except Exception:
            update_check(current_index, False, traceback.format_exc())

        current_index = 1
        try:
            candidate = candidate_module.PairFormerIteration(16, 12).eval()
            reference = reference_module.PairFormerIteration(16, 12).eval()
            _initialize_parameters_with_constants(candidate)
            _initialize_parameters_with_constants(reference)
            single = torch.linspace(-1.0, 1.0, steps=2 * 5 * 16).reshape(2, 5, 16)
            pair = torch.linspace(-1.0, 1.0, steps=2 * 5 * 5 * 12).reshape(2, 5, 5, 12)
            with torch.no_grad():
                candidate_output = candidate(single, pair)
                reference_output = reference(single, pair)
            if not isinstance(candidate_output, (tuple, list)) or len(candidate_output) != 2:
                raise AssertionError("PairFormerIteration must return (single, pair)")
            torch.testing.assert_close(candidate_output[0], reference_output[0], rtol=RTOL, atol=ATOL)
            torch.testing.assert_close(candidate_output[1], reference_output[1], rtol=RTOL, atol=ATOL)
            if tuple(candidate_output[0].shape) != tuple(single.shape):
                raise AssertionError("PairFormerIteration must preserve single shape")
            if tuple(candidate_output[1].shape) != tuple(pair.shape):
                raise AssertionError("PairFormerIteration must preserve pair shape")
            update_check(current_index, True, "PairFormerIteration hidden API and reference behavior passed")
        except Exception:
            update_check(current_index, False, traceback.format_exc())

        current_index = 2
        try:
            candidate = candidate_module.DiffusionHead(16).eval()
            reference = reference_module.DiffusionHead(16).eval()
            _initialize_parameters_with_constants(candidate)
            _initialize_parameters_with_constants(reference)
            single = torch.linspace(-1.0, 1.0, steps=2 * 5 * 16).reshape(2, 5, 16)
            for num_samples in (1, 2, 4):
                for seed in (0, 7):
                    with torch.no_grad():
                        candidate_output = candidate(single, num_samples, seed)
                        reference_output = reference(single, num_samples, seed)
                    torch.testing.assert_close(candidate_output, reference_output, rtol=RTOL, atol=ATOL)
                    if tuple(candidate_output.shape) != (num_samples, 2, 5, 3):
                        raise AssertionError(f"DiffusionHead shape={tuple(candidate_output.shape)}")
            update_check(current_index, True, "DiffusionHead hidden API, sampling, and reference behavior passed")
        except Exception:
            update_check(current_index, False, traceback.format_exc())

        current_index = 3
        try:
            candidate = candidate_module.ConfidenceHead(16).eval()
            reference = reference_module.ConfidenceHead(16).eval()
            _initialize_parameters_with_constants(candidate)
            _initialize_parameters_with_constants(reference)
            single = torch.linspace(-1.0, 1.0, steps=2 * 5 * 16).reshape(2, 5, 16)
            with torch.no_grad():
                candidate_output = candidate(single)
                reference_output = reference(single)
            required = {
                "predicted_lddt",
                "predicted_experimentally_resolved",
                "full_pae",
                "average_pae",
            }
            if not isinstance(candidate_output, dict) or set(candidate_output) != required:
                raise AssertionError("ConfidenceHead output keys are invalid")
            for name in required:
                torch.testing.assert_close(candidate_output[name], reference_output[name], rtol=RTOL, atol=ATOL)
            for name in ("predicted_lddt", "predicted_experimentally_resolved"):
                if not torch.all((candidate_output[name] >= 0) & (candidate_output[name] <= 1)):
                    raise AssertionError(f"{name} must be in [0, 1]")
            if tuple(candidate_output["full_pae"].shape) != (2, 5, 5):
                raise AssertionError("ConfidenceHead full_pae shape is invalid")
            torch.testing.assert_close(
                candidate_output["average_pae"],
                candidate_output["full_pae"].mean(dim=(-1, -2)),
                rtol=RTOL,
                atol=ATOL,
            )
            update_check(current_index, True, "ConfidenceHead hidden API and reference behavior passed")
        except Exception:
            update_check(current_index, False, traceback.format_exc())

        return checks

    current_index = 0
    try:
        with tempfile.TemporaryDirectory(prefix="alphafold3_helpers_") as temporary:
            root = Path(temporary)
            fixtures = [
                ({"sequences": [{"protein": {"sequence": "ACDEFG"}}]}, "ACDEFG"),
                ({"sequences": [{"protein": {"sequence": "ACD"}}, {"protein": {"sequence": "EFG"}}]}, "ACDEFG"),
                ({"proteinChain": {"sequence": "MNPQRST"}}, "MNPQRST"),
                ({"sequences": [{"dna": {"sequence": "ACGT"}}]}, "ACGT"),
            ]
            for index, (payload, expected_text) in enumerate(fixtures):
                path = root / f"valid_{index}.json"
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                actual = candidate_module.sequence_from_json(path)
                expected = reference_module.sequence_from_json(path)
                if actual != expected_text or actual != expected:
                    raise AssertionError(f"sequence_from_json returned {actual!r}, expected {expected_text!r}")
            invalid = [
                {"sequences": []},
                {"sequences": [{"protein": {"sequence": ""}}]},
                {"sequences": [{"protein": {"sequence": "ACD1"}}]},
                {"sequences": [{"protein": {"id": ["A"]}}]},
                {"sequences": [{"dna": {"sequence": "ACDX"}}]},
            ]
            for index, payload in enumerate(invalid):
                path = root / f"invalid_{index}.json"
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                try:
                    candidate_module.sequence_from_json(path)
                except Exception:
                    pass
                else:
                    raise AssertionError(f"invalid JSON fixture {index} was accepted")
        update_check(current_index, True, "sequence_from_json parsing and validation passed")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 1
    try:
        for sequence in ("A", "ACDE", "ACDEFGHIK"):
            for num_recycles in (1, 2, 3):
                candidate_output = candidate_module.make_features(sequence, num_recycles)
                reference_output = reference_module.make_features(sequence, num_recycles)
                if not isinstance(candidate_output, dict) or set(candidate_output) != {"target_feat", "residue_index"}:
                    raise AssertionError("make_features must return exactly target_feat and residue_index")
                candidate_output = _move_tensors_to_device(candidate_output)
                reference_output = _move_tensors_to_device(reference_output)
                for name in ("target_feat", "residue_index"):
                    torch.testing.assert_close(candidate_output[name], reference_output[name], rtol=RTOL, atol=ATOL)
                length = len(sequence)
                if tuple(candidate_output["target_feat"].shape) != (1, length, num_recycles, 22):
                    raise AssertionError("target_feat shape is invalid")
                if tuple(candidate_output["residue_index"].shape) != (1, length, num_recycles):
                    raise AssertionError("residue_index shape is invalid")
        update_check(current_index, True, "make_features shapes and hidden reference encoding passed")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 2
    try:
        with tempfile.TemporaryDirectory(prefix="alphafold3_pdb_") as temporary:
            path = Path(temporary) / "sample.pdb"
            coordinates = torch.tensor(
                [[[0.0, 1.0, 2.0], [3.5, 4.5, 5.5], [-1.0, -2.0, -3.0]]],
                dtype=torch.float32,
            )
            candidate_module.write_pdb(coordinates, path)
            if not path.is_file():
                raise AssertionError("write_pdb did not create a file")
            lines = path.read_text(encoding="utf-8").splitlines()
            atom_lines = [line for line in lines if line.startswith("ATOM")]
            if len(atom_lines) != 3 or not lines or lines[-1].strip() != "END":
                raise AssertionError("write_pdb output protocol is invalid")
            for index, line in enumerate(atom_lines, 1):
                if int(line[6:11]) != index:
                    raise AssertionError("PDB serial numbers must be contiguous")
                values = torch.tensor([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
                if not torch.isfinite(values).all():
                    raise AssertionError("PDB coordinate contains NaN or Inf")
        update_check(current_index, True, "write_pdb produced finite ATOM records and an END terminator")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Check default structure and the original configurable-constructor hidden contract."""
    CHECKPOINT_NAMES = [
        "model_structure:required_components",
        "model_structure:default_pairformer_depth",
        "model_structure:representation_dimensions",
        "model_structure:configurable_constructor",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    try:
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        model_cls = _load_class(candidate_module, "AlphaFold3")
    except Exception:
        error_detail = traceback.format_exc()
        return [(name, False, error_detail) for name in CHECKPOINT_NAMES]

    default_model = None
    current_index = 0
    try:
        default_model = model_cls().to(DEVICE).eval()
        # Prefer the structural rules declared by the evaluation case so the
        # dataset JSON is the single source of truth for the expected
        # architecture (root layout, per-component submodules, and leaf layer
        # shapes). Fall back to the minimal component-count contract only when
        # a case ships no rules of its own.
        configured_rules = config.get("rules")
        if configured_rules:
            rules = {"rules": configured_rules}
        else:
            rules = {
                "rules": [
                    {"id": "target_embedder_count", "type": "module_count", "severity": "error", "params": {"selector": {"class_name": "TargetEmbedder"}, "count": 1}},
                    {"id": "pairformer_count", "type": "module_count", "severity": "error", "params": {"selector": {"class_name": "PairFormerIteration"}, "count": 2}},
                    {"id": "diffusion_head_count", "type": "module_count", "severity": "error", "params": {"selector": {"class_name": "DiffusionHead"}, "count": 1}},
                    {"id": "confidence_head_count", "type": "module_count", "severity": "error", "params": {"selector": {"class_name": "ConfidenceHead"}, "count": 1}},
                ]
            }
        structural_results = checkpoints_to_tuples(check_model_structure_v2(default_model, config_dict=rules, verbose=False))
        failures = [f"{name}: {detail}" for name, ok, detail in structural_results if not ok]
        if failures:
            raise AssertionError("; ".join(failures))
        update_check(current_index, True, "default AlphaFold3 matches the declared structural rules (components, submodules, and layer shapes)")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 1
    try:
        model = default_model if default_model is not None else model_cls().to(DEVICE).eval()
        rules = {
            "rules": [
                {"id": "default_pairformer_count", "type": "module_count", "severity": "error", "params": {"selector": {"class_name": "PairFormerIteration"}, "count": 2}}
            ]
        }
        structural_results = checkpoints_to_tuples(check_model_structure_v2(model, config_dict=rules, verbose=False))
        failures = [f"{name}: {detail}" for name, ok, detail in structural_results if not ok]
        if failures:
            raise AssertionError("; ".join(failures))
        update_check(current_index, True, "AlphaFold3 defaults to two PairFormerIteration blocks")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 2
    try:
        model = default_model if default_model is not None else model_cls().to(DEVICE).eval()
        num_recycles = int(config.get("num_recycles", 2))
        length = 6
        target_feat = torch.zeros((1, length, num_recycles, 22), device=DEVICE)
        token_index = torch.arange(length, device=DEVICE) % 22
        for recycle in range(num_recycles):
            target_feat[0, torch.arange(length, device=DEVICE), recycle, token_index] = 1.0
        residue_index = torch.arange(length, device=DEVICE).view(1, length, 1).expand(1, length, num_recycles)
        with torch.no_grad():
            output = model({"target_feat": target_feat, "residue_index": residue_index}, seed=int(config.get("seed", 7)))
        if tuple(output["single_embeddings"].shape) != (1, length, 32):
            raise AssertionError("single_embeddings must have width 32")
        if tuple(output["pair_embeddings"].shape) != (1, length, length, 32):
            raise AssertionError("pair_embeddings must have width 32")
        update_check(current_index, True, "default model exposes 32-wide single and pair representations")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 3
    try:
        num_recycles = int(config.get("num_recycles", 2))
        num_samples = int(config.get("num_diffusion_samples", 2))
        num_layers = int(config.get("num_pairformer_layers", 2))
        model = model_cls(
            num_recycles=num_recycles,
            num_diffusion_samples=num_samples,
            num_pairformer_layers=num_layers,
        ).to(DEVICE).eval()
        rules = {
            "rules": [
                {"id": "configured_pairformer_count", "type": "module_count", "severity": "error", "params": {"selector": {"class_name": "PairFormerIteration"}, "count": num_layers}}
            ]
        }
        structural_results = checkpoints_to_tuples(check_model_structure_v2(model, config_dict=rules, verbose=False))
        failures = [f"{name}: {detail}" for name, ok, detail in structural_results if not ok]
        if failures:
            raise AssertionError("; ".join(failures))
        update_check(current_index, True, "original configurable AlphaFold3 constructor hidden contract passed")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Validate strict loading independently of the configurable-constructor hidden check."""
    CHECKPOINT_NAMES = [
        "checkpoint_loading:strict_roundtrip",
        "checkpoint_loading:direct_state_dict",
        "checkpoint_loading:directory_model_pt",
        "checkpoint_loading:reject_incompatible",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    try:
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        model_cls = _load_class(candidate_module, "AlphaFold3")
        model = model_cls().cpu().eval()
        _initialize_parameters_with_constants(model)
        state_dict = model.state_dict()
    except Exception:
        error_detail = traceback.format_exc()
        return [(name, False, error_detail) for name in CHECKPOINT_NAMES]

    current_index = 0
    try:
        clone = model_cls().cpu().eval()
        missing, unexpected = clone.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise AssertionError(f"strict=True returned missing={missing}, unexpected={unexpected}")
        update_check(current_index, True, "default candidate state_dict round-trips with strict=True")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    num_recycles = int(config.get("num_recycles", 2))
    num_samples = int(config.get("num_diffusion_samples", 2))
    seed = int(config.get("seed", 7))
    timeout = int(config.get("timeout", 90))
    sequence = str(config.get("sequence", "ACDEFG"))

    with tempfile.TemporaryDirectory(prefix="alphafold3_checkpoint_") as temporary:
        root = Path(temporary)
        input_path = root / "input.json"
        input_path.write_text(
            json.dumps({"sequences": [{"protein": {"id": ["A"], "sequence": sequence}}]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        script_path = (CANDIDATE_DIR / "alphafold3_inference.py").resolve()
        if not script_path.is_file():
            error_detail = f"Candidate script not found: {script_path}"
            for index in range(1, 4):
                update_check(index, False, error_detail)
            return checks

        current_index = 1
        try:
            raw_checkpoint = root / "weights.pt"
            torch.save(state_dict, raw_checkpoint)
            command = [
                sys.executable, str(script_path), "--json-path", str(input_path), "--model-dir", str(raw_checkpoint),
                "--output-dir", str(root / "raw_output"), "--run-data-pipeline", "true", "--run-inference", "true",
                "--flash-attention-implementation", "xla", "--num-recycles", str(num_recycles),
                "--num-diffusion-samples", str(num_samples), "--seed", str(seed), "--device", "cpu",
            ]
            completed = subprocess.run(command, cwd=str(script_path.parent), text=True, capture_output=True, timeout=timeout, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"direct state_dict CLI failed\nstdout={completed.stdout}\nstderr={completed.stderr}")
            update_check(current_index, True, "--model-dir accepts a direct state_dict file")
        except Exception:
            update_check(current_index, False, traceback.format_exc())

        current_index = 2
        try:
            model_directory = root / "model_dir"
            model_directory.mkdir(exist_ok=True)
            torch.save({"model_state_dict": state_dict}, model_directory / "model.pt")
            command = [
                sys.executable, str(script_path), "--json-path", str(input_path), "--model-dir", str(model_directory),
                "--output-dir", str(root / "directory_output"), "--run-data-pipeline", "true", "--run-inference", "true",
                "--flash-attention-implementation", "xla", "--num-recycles", str(num_recycles),
                "--num-diffusion-samples", str(num_samples), "--seed", str(seed), "--device", "cpu",
            ]
            completed = subprocess.run(command, cwd=str(script_path.parent), text=True, capture_output=True, timeout=timeout, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"directory/model.pt CLI failed\nstdout={completed.stdout}\nstderr={completed.stderr}")
            update_check(current_index, True, "--model-dir accepts directory/model.pt with model_state_dict")
        except Exception:
            update_check(current_index, False, traceback.format_exc())

        current_index = 3
        try:
            invalid_state = dict(state_dict)
            invalid_state.pop(next(iter(invalid_state)))
            invalid_checkpoint = root / "invalid.pt"
            torch.save({"model_state_dict": invalid_state}, invalid_checkpoint)
            command = [
                sys.executable, str(script_path), "--json-path", str(input_path), "--model-dir", str(invalid_checkpoint),
                "--output-dir", str(root / "invalid_output"), "--run-data-pipeline", "true", "--run-inference", "true",
                "--flash-attention-implementation", "xla", "--num-recycles", str(num_recycles),
                "--num-diffusion-samples", str(num_samples), "--seed", str(seed), "--device", "cpu",
            ]
            completed = subprocess.run(command, cwd=str(script_path.parent), text=True, capture_output=True, timeout=timeout, check=False)
            if completed.returncode == 0:
                raise AssertionError("incompatible checkpoint unexpectedly succeeded")
            update_check(current_index, True, "incompatible checkpoint is rejected by strict loading")
        except Exception:
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_execution(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Validate forward behavior without depending on the configurable constructor."""
    CHECKPOINT_NAMES = [
        "model_forward:shape_and_value_contract",
        "model_forward:module_call_chain",
        "model_forward:seed_semantics",
        "model_forward:gradient_contract",
        "model_forward:state_integrity",
        "model_forward:batch_dimension",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    try:
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        model_cls = _load_class(candidate_module, "AlphaFold3")
        model = model_cls().to(DEVICE).eval()
        _initialize_parameters_with_constants(model)
    except Exception:
        error_detail = traceback.format_exc()
        return [(name, False, error_detail) for name in CHECKPOINT_NAMES]

    num_recycles = int(config.get("num_recycles", 2))
    num_samples = int(config.get("num_diffusion_samples", 2))
    seed = int(config.get("seed", 7))

    def make_synthetic_features(length: int, batch: int = 1):
        target_feat = torch.zeros((batch, length, num_recycles, 22), device=DEVICE)
        position = torch.arange(length, device=DEVICE)
        token_index = position % 22
        for batch_index in range(batch):
            for recycle in range(num_recycles):
                target_feat[batch_index, position, recycle, token_index] = 1.0
        residue_index = position.view(1, length, 1).expand(batch, length, num_recycles).clone()
        return {"target_feat": target_feat, "residue_index": residue_index}

    current_index = 0
    try:
        sequences = config.get("sequences") or ["ACDEFG"]
        if isinstance(sequences, str):
            sequences = [sequences]
        for sequence in sequences:
            length = len(str(sequence))
            output = model(make_synthetic_features(length), seed=seed)
            if not isinstance(output, dict) or set(output) != FORWARD_OUTPUT_KEYS:
                raise AssertionError("forward must return exactly the required output keys")
            expected_shapes = {
                "diffusion_samples": (num_samples, 1, length, 3),
                "atom_positions": (1, length, 3),
                "single_embeddings": (1, length, 32),
                "pair_embeddings": (1, length, length, 32),
                "predicted_lddt": (1, length),
                "predicted_experimentally_resolved": (1, length),
                "full_pae": (1, length, length),
                "average_pae": (1,),
            }
            for name, expected_shape in expected_shapes.items():
                value = output[name]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise AssertionError(f"{name} shape={getattr(value, 'shape', None)}, expected={expected_shape}")
                if not torch.isfinite(value).all():
                    raise AssertionError(f"{name} contains NaN or Inf")
            if not torch.equal(output["atom_positions"], output["diffusion_samples"][0]):
                raise AssertionError("atom_positions must equal diffusion_samples[0]")
            torch.testing.assert_close(output["average_pae"], output["full_pae"].mean(dim=(-1, -2)), rtol=RTOL, atol=ATOL)
            for name in ("predicted_lddt", "predicted_experimentally_resolved"):
                if not torch.all((output[name] >= 0) & (output[name] <= 1)):
                    raise AssertionError(f"{name} must be in [0, 1]")
        update_check(current_index, True, "forward output keys, shapes, ranges, and derived values are valid")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 1
    try:
        target_modules = [module for module in model.modules() if isinstance(module, candidate_module.TargetEmbedder)]
        pairformer_modules = [module for module in model.modules() if isinstance(module, candidate_module.PairFormerIteration)]
        diffusion_modules = [module for module in model.modules() if isinstance(module, candidate_module.DiffusionHead)]
        confidence_modules = [module for module in model.modules() if isinstance(module, candidate_module.ConfidenceHead)]
        if len(target_modules) != 1 or len(pairformer_modules) != 2 or len(diffusion_modules) != 1 or len(confidence_modules) != 1:
            raise AssertionError(
                f"module counts target={len(target_modules)}, pairformer={len(pairformer_modules)}, diffusion={len(diffusion_modules)}, confidence={len(confidence_modules)}"
            )
        call_order = []
        handles = [target_modules[0].register_forward_hook(lambda _m, _i, _o: call_order.append("target_embedder"))]
        for index, module in enumerate(pairformer_modules):
            handles.append(module.register_forward_hook(lambda _m, _i, _o, idx=index: call_order.append(f"pairformer_{idx}")))
        handles.append(diffusion_modules[0].register_forward_hook(lambda _m, _i, _o: call_order.append("diffusion_head")))
        handles.append(confidence_modules[0].register_forward_hook(lambda _m, _i, _o: call_order.append("confidence_head")))
        try:
            with torch.no_grad():
                model(make_synthetic_features(6), seed=seed)
        finally:
            for handle in handles:
                handle.remove()
        required_calls = {"target_embedder", "pairformer_0", "pairformer_1", "diffusion_head", "confidence_head"}
        missing_calls = sorted(required_calls - set(call_order))
        if missing_calls:
            raise AssertionError(f"AlphaFold3.forward did not execute required modules: {missing_calls}")
        target_position = call_order.index("target_embedder")
        pair_positions = [call_order.index("pairformer_0"), call_order.index("pairformer_1")]
        diffusion_position = call_order.index("diffusion_head")
        confidence_position = call_order.index("confidence_head")
        if target_position > min(pair_positions):
            raise AssertionError("TargetEmbedder must execute before PairFormerIteration")
        if max(pair_positions) > min(diffusion_position, confidence_position):
            raise AssertionError("PairFormerIteration must execute before DiffusionHead and ConfidenceHead")
        update_check(current_index, True, "forward executes the required component chain in stage order")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 2
    try:
        features = make_synthetic_features(9)
        with torch.no_grad():
            first = model(features, seed=7)
            repeated = model(features, seed=7)
            changed = model(features, seed=8)
        for name in FORWARD_OUTPUT_KEYS:
            torch.testing.assert_close(first[name], repeated[name], rtol=0.0, atol=0.0)
        if torch.equal(first["diffusion_samples"], changed["diffusion_samples"]):
            raise AssertionError("different seeds must change diffusion_samples")
        for name in ("predicted_lddt", "predicted_experimentally_resolved", "full_pae", "average_pae", "single_embeddings", "pair_embeddings"):
            torch.testing.assert_close(first[name], changed[name], rtol=0.0, atol=0.0)
        update_check(current_index, True, "same seed is reproducible and different seeds only change diffusion samples")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 3
    try:
        model.zero_grad(set_to_none=True)
        output = model(make_synthetic_features(6), seed=seed)
        objective = output["atom_positions"].square().mean() + output["predicted_lddt"].mean() + output["full_pae"].mean()
        if not objective.requires_grad:
            raise AssertionError("forward output lost gradients")
        objective.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError("forward produced no finite parameter gradients")
        model.zero_grad(set_to_none=True)
        update_check(current_index, True, "forward retains a finite gradient path to parameters")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 4
    try:
        model.eval()
        state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        features = make_synthetic_features(10)
        with torch.inference_mode():
            first = model(features, seed=17)
            repeated = model(features, seed=17)
        for name in FORWARD_OUTPUT_KEYS:
            torch.testing.assert_close(first[name], repeated[name], rtol=0.0, atol=0.0)
        for name, expected_value in state_before.items():
            torch.testing.assert_close(model.state_dict()[name], expected_value, rtol=0.0, atol=0.0)
        update_check(current_index, True, "eval/inference is deterministic and state-preserving")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 5
    try:
        with torch.no_grad():
            output = model(make_synthetic_features(6, batch=2), seed=seed)
        if tuple(output["diffusion_samples"].shape) != (num_samples, 2, 6, 3):
            raise AssertionError("batched diffusion shape is invalid")
        if tuple(output["single_embeddings"].shape) != (2, 6, 32):
            raise AssertionError("batched single_embeddings shape is invalid")
        if tuple(output["pair_embeddings"].shape) != (2, 6, 6, 32):
            raise AssertionError("batched pair_embeddings shape is invalid")
        update_check(current_index, True, "hidden batch-dimension contract passed")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Preserve the strong reference oracle while isolating configurable-constructor failures."""
    CHECKPOINT_NAMES = [
        "model_forward_equivalence:default_reference_oracle",
        "model_forward_equivalence:configuration_sweep",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    try:
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        reference_module = _load_reference_modules("alphafold3_inference.py")
        candidate_cls = _load_class(candidate_module, "AlphaFold3")
        reference_cls = _load_class(reference_module, "AlphaFold3")
    except Exception:
        error_detail = traceback.format_exc()
        return [(name, False, error_detail) for name in CHECKPOINT_NAMES]

    seed = int(config.get("seed", 11))
    sequence = str(config.get("sequence", "ACDEFG"))

    def synthetic_features(length: int, recycles: int):
        target_feat = torch.zeros((1, length, recycles, 22), device=DEVICE)
        position = torch.arange(length, device=DEVICE)
        token_index = position % 22
        for recycle in range(recycles):
            target_feat[0, position, recycle, token_index] = 1.0
        residue_index = position.view(1, length, 1).expand(1, length, recycles).clone()
        return {"target_feat": target_feat, "residue_index": residue_index}

    current_index = 0
    try:
        candidate_model = candidate_cls().to(DEVICE).eval()
        reference_model = reference_cls().to(DEVICE).eval()
        _initialize_parameters_with_constants(candidate_model)
        _initialize_parameters_with_constants(reference_model)
        recycles = int(config.get("num_recycles", 2))
        features = synthetic_features(len(sequence), recycles)
        with torch.no_grad():
            candidate_output = candidate_model({name: value.clone() for name, value in features.items()}, seed=seed)
            reference_output = reference_model({name: value.clone() for name, value in features.items()}, seed=seed)
        if not isinstance(candidate_output, dict) or set(candidate_output) != FORWARD_OUTPUT_KEYS:
            raise AssertionError("candidate forward output keys are invalid")
        if not isinstance(reference_output, dict) or set(reference_output) != FORWARD_OUTPUT_KEYS:
            raise AssertionError("reference forward output keys are invalid")
        for name in FORWARD_OUTPUT_KEYS:
            torch.testing.assert_close(candidate_output[name], reference_output[name], rtol=RTOL, atol=ATOL)
        update_check(current_index, True, "default candidate forward matches the trusted reference oracle")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 1
    try:
        configurations = [
            (1, 1, 1),
            (1, 3, 2),
            (2, 2, 2),
            (3, 4, 3),
        ]
        for test_recycles, test_samples, test_layers in configurations:
            candidate_model = candidate_cls(
                num_recycles=test_recycles,
                num_diffusion_samples=test_samples,
                num_pairformer_layers=test_layers,
            ).to(DEVICE).eval()
            reference_model = reference_cls(
                num_recycles=test_recycles,
                num_diffusion_samples=test_samples,
                num_pairformer_layers=test_layers,
            ).to(DEVICE).eval()
            _initialize_parameters_with_constants(candidate_model)
            _initialize_parameters_with_constants(reference_model)
            features = synthetic_features(len(sequence), test_recycles)
            with torch.no_grad():
                candidate_output = candidate_model({name: value.clone() for name, value in features.items()}, seed=seed)
                reference_output = reference_model({name: value.clone() for name, value in features.items()}, seed=seed)
            if tuple(candidate_output["diffusion_samples"].shape) != (test_samples, 1, len(sequence), 3):
                raise AssertionError(
                    f"configuration {(test_recycles, test_samples, test_layers)} did not control diffusion shape"
                )
            for name in FORWARD_OUTPUT_KEYS:
                torch.testing.assert_close(candidate_output[name], reference_output[name], rtol=RTOL, atol=ATOL)
        update_check(current_index, True, "original hidden configuration sweep matches the reference")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_structure(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Run shared AST workflow checks and task-specific source checks independently."""
    CHECKPOINT_NAMES = [
        "infer_structure:base_workflow",
        "infer_structure:task_contract",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        base_config = {
            "files": ["alphafold3_inference.py"],
            "check_config": {
                "entry_function_names": ["main"],
                "model_load_functions": ["load_model", "load_checkpoint", "load_weights"],
                "model_load_attributes": ["load_state_dict"],
                "input_parameter_names": ["json_path", "sequence", "features", "input"],
                "input_load_functions": ["sequence_from_json", "make_features"],
                "input_load_attributes": ["load", "read_text"],
                "infer_call_functions": ["main", "inference", "infer"],
                "infer_call_attributes": ["forward"],
                "output_attributes": ["write", "writerow", "write_text", "dump"],
            },
        }
        base_results = _base_check_infer_structure(base_config, caller_globals=caller_globals, caller_locals=caller_locals)
        failures = [f"{name}: {detail}" for name, ok, detail in base_results if not ok]
        if failures:
            raise AssertionError("; ".join(failures))
        update_check(current_index, True, "base workflow passed, including eval and no_grad/inference_mode")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    current_index = 1
    try:
        script_path = (CANDIDATE_DIR / "alphafold3_inference.py").resolve()
        if not script_path.is_file():
            raise FileNotFoundError(f"Candidate script not found: {script_path}")
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script_path))
        imported_roots = set()
        parser_flags = set()
        has_main_guard = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args:
                first_argument = node.args[0]
                if isinstance(first_argument, ast.Constant) and isinstance(first_argument.value, str):
                    parser_flags.add(first_argument.value)
            elif isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "__main__"
                ):
                    has_main_guard = True
        if "onescience" in imported_roots:
            raise AssertionError("candidate must not import onescience")
        missing_flags = sorted(REQUIRED_FLAGS - parser_flags)
        if missing_flags:
            raise AssertionError(f"missing required CLI flags: {missing_flags}")
        if not has_main_guard:
            raise AssertionError("missing if __name__ == '__main__' entrypoint guard")
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        for class_name in REQUIRED_CLASSES:
            model_class = _load_class(candidate_module, class_name)
            if not issubclass(model_class, nn.Module):
                raise AssertionError(f"{class_name} must inherit torch.nn.Module")
        for function_name in REQUIRED_FUNCTIONS:
            if not callable(getattr(candidate_module, function_name, None)):
                raise AssertionError(f"missing callable {function_name}")
        update_check(current_index, True, "task-specific flags, entrypoint, classes/functions, and forbidden-import contract passed")
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_execution(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Run the real CLI and verify positive integer argument validation."""
    CHECKPOINT_NAMES = [
        "infer_execution:cli",
        "infer_execution:positive_integer_validation",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        num_recycles = int(config.get("num_recycles", 2))
        num_samples = int(config.get("num_diffusion_samples", 2))
        seed = int(config.get("seed", 7))
        timeout = int(config.get("timeout", 90))
        sequence = str(config.get("sequence", "ACDEFG"))

        model_cls = _load_class(candidate_module, "AlphaFold3")
        model = model_cls().cpu().eval()
        _initialize_parameters_with_constants(model)

        with tempfile.TemporaryDirectory(
            prefix="alphafold3_infer_"
        ) as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "sequences": [
                            {
                                "protein": {
                                    "id": ["A"],
                                    "sequence": sequence,
                                }
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            checkpoint_path = root / "model_state.pt"
            torch.save(model.state_dict(), checkpoint_path)
            output_dir = root / "outputs"
            script_path = (CANDIDATE_DIR / "alphafold3_inference.py").resolve()

            command = [
                sys.executable,
                str(script_path),
                "--json-path",
                str(input_path),
                "--model-dir",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--run-data-pipeline",
                "true",
                "--run-inference",
                "true",
                "--flash-attention-implementation",
                "xla",
                "--num-recycles",
                str(num_recycles),
                "--num-diffusion-samples",
                str(num_samples),
                "--seed",
                str(seed),
                "--device",
                "cpu",
            ]
            completed = subprocess.run(
                command,
                cwd=str(script_path.parent),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Inference CLI failed\n"
                    f"stdout={completed.stdout}\n"
                    f"stderr={completed.stderr}"
                )
            if not output_dir.is_dir():
                raise AssertionError("CLI did not create --output-dir")
            update_check(
                current_index,
                True,
                "CLI inference completed successfully",
            )

            current_index += 1
            for bad_recycles, bad_samples, description in (
                (0, num_samples, "num_recycles"),
                (num_recycles, 0, "num_diffusion_samples"),
            ):
                command = [
                    sys.executable,
                    str(script_path),
                    "--json-path",
                    str(input_path),
                    "--model-dir",
                    str(checkpoint_path),
                    "--output-dir",
                    str(root / f"bad_{description}"),
                    "--run-data-pipeline",
                    "true",
                    "--run-inference",
                    "true",
                    "--flash-attention-implementation",
                    "xla",
                    "--num-recycles",
                    str(bad_recycles),
                    "--num-diffusion-samples",
                    str(bad_samples),
                    "--seed",
                    str(seed),
                    "--device",
                    "cpu",
                ]
                completed = subprocess.run(
                    command,
                    cwd=str(script_path.parent),
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
                if completed.returncode == 0:
                    raise AssertionError(
                        f"zero {description} unexpectedly succeeded"
                    )
            update_check(
                current_index,
                True,
                "zero num_recycles and zero num_diffusion_samples are rejected",
            )

    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_output_schema(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Validate the exact three-file CLI output schema."""
    CHECKPOINT_NAMES = [
        "infer_output_schema:files",
        "infer_output_schema:pdb",
        "infer_output_schema:confidence",
        "infer_output_schema:ranking",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        num_recycles = int(config.get("num_recycles", 2))
        num_samples = int(config.get("num_diffusion_samples", 2))
        seed = int(config.get("seed", 7))
        timeout = int(config.get("timeout", 90))
        sequence = str(config.get("sequence", "ACDEFG"))

        model_cls = _load_class(candidate_module, "AlphaFold3")
        model = model_cls().cpu().eval()
        _initialize_parameters_with_constants(model)

        with tempfile.TemporaryDirectory(
            prefix="alphafold3_schema_"
        ) as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "sequences": [
                            {
                                "protein": {
                                    "id": ["A"],
                                    "sequence": sequence,
                                }
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            checkpoint_path = root / "model_state.pt"
            torch.save(model.state_dict(), checkpoint_path)
            output_dir = root / "outputs"
            script_path = (CANDIDATE_DIR / "alphafold3_inference.py").resolve()
            command = [
                sys.executable,
                str(script_path),
                "--json-path",
                str(input_path),
                "--model-dir",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--run-data-pipeline",
                "true",
                "--run-inference",
                "true",
                "--flash-attention-implementation",
                "xla",
                "--num-recycles",
                str(num_recycles),
                "--num-diffusion-samples",
                str(num_samples),
                "--seed",
                str(seed),
                "--device",
                "cpu",
            ]
            completed = subprocess.run(
                command,
                cwd=str(script_path.parent),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Inference CLI failed\n"
                    f"stdout={completed.stdout}\n"
                    f"stderr={completed.stderr}"
                )

            actual_files = {
                path.name
                for path in output_dir.iterdir()
                if path.is_file()
            }
            expected_files = {
                "sample_0.pdb",
                "confidence.json",
                "ranking_scores.csv",
            }
            if actual_files != expected_files:
                raise AssertionError(
                    f"output files={sorted(actual_files)}, "
                    f"expected={sorted(expected_files)}"
                )
            update_check(
                current_index,
                True,
                "output directory contains exactly sample_0.pdb, confidence.json, and ranking_scores.csv",
            )

            current_index += 1
            lines = (output_dir / "sample_0.pdb").read_text(
                encoding="utf-8"
            ).splitlines()
            if not lines or lines[-1].strip() != "END":
                raise AssertionError("sample_0.pdb must end with END")
            update_check(
                current_index,
                True,
                "sample_0.pdb terminates with END",
            )

            current_index += 1
            confidence = json.loads(
                (output_dir / "confidence.json").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(confidence, dict) or set(confidence) != {
                "average_pae",
                "mean_plddt",
            }:
                raise AssertionError(
                    "confidence.json must contain exactly average_pae and mean_plddt"
                )
            update_check(
                current_index,
                True,
                "confidence.json contains exactly average_pae and mean_plddt",
            )

            current_index += 1
            with (output_dir / "ranking_scores.csv").open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames
            if fieldnames != ["rank", "score", "structure_file"]:
                raise AssertionError(
                    f"ranking columns={fieldnames}, "
                    "expected=['rank', 'score', 'structure_file']"
                )
            if len(rows) != 1:
                raise AssertionError(
                    "ranking_scores.csv must contain exactly one data row"
                )
            if int(rows[0]["rank"]) != 1:
                raise AssertionError("ranking rank must equal 1")
            if rows[0]["structure_file"] != "sample_0.pdb":
                raise AssertionError(
                    "ranking structure_file must equal sample_0.pdb"
                )
            update_check(
                current_index,
                True,
                "ranking_scores.csv has the required columns and single rank-1 sample",
            )

    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_output_validity(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Validate finite CLI artifact values and ranking-score consistency."""
    CHECKPOINT_NAMES = [
        "infer_output_validity:pdb_coordinates",
        "infer_output_validity:confidence_values",
        "infer_output_validity:ranking_score",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        num_recycles = int(config.get("num_recycles", 2))
        num_samples = int(config.get("num_diffusion_samples", 2))
        seed = int(config.get("seed", 7))
        timeout = int(config.get("timeout", 90))
        sequence = str(config.get("sequence", "ACDEFG"))

        model_cls = _load_class(candidate_module, "AlphaFold3")
        model = model_cls().cpu().eval()
        _initialize_parameters_with_constants(model)

        with tempfile.TemporaryDirectory(
            prefix="alphafold3_validity_"
        ) as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "sequences": [
                            {
                                "protein": {
                                    "id": ["A"],
                                    "sequence": sequence,
                                }
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            checkpoint_path = root / "model_state.pt"
            torch.save(model.state_dict(), checkpoint_path)
            output_dir = root / "outputs"
            script_path = (CANDIDATE_DIR / "alphafold3_inference.py").resolve()
            command = [
                sys.executable,
                str(script_path),
                "--json-path",
                str(input_path),
                "--model-dir",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--run-data-pipeline",
                "true",
                "--run-inference",
                "true",
                "--flash-attention-implementation",
                "xla",
                "--num-recycles",
                str(num_recycles),
                "--num-diffusion-samples",
                str(num_samples),
                "--seed",
                str(seed),
                "--device",
                "cpu",
            ]
            completed = subprocess.run(
                command,
                cwd=str(script_path.parent),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Inference CLI failed\n"
                    f"stdout={completed.stdout}\n"
                    f"stderr={completed.stderr}"
                )

            lines = (output_dir / "sample_0.pdb").read_text(
                encoding="utf-8"
            ).splitlines()
            coordinates = []
            for line in lines:
                if line.startswith("ATOM"):
                    coordinates.append(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ]
                    )
            if not coordinates:
                raise AssertionError("PDB contains no ATOM records")
            coordinate_tensor = torch.tensor(
                coordinates,
                dtype=torch.float32,
            )
            if tuple(coordinate_tensor.shape) != (len(sequence), 3):
                raise AssertionError(
                    f"PDB coordinate shape={tuple(coordinate_tensor.shape)}, "
                    f"expected={(len(sequence), 3)}"
                )
            if not torch.isfinite(coordinate_tensor).all():
                raise AssertionError("PDB contains NaN or Inf coordinates")
            update_check(
                current_index,
                True,
                "sample_0.pdb contains one finite xyz coordinate per residue",
            )

            current_index += 1
            confidence = json.loads(
                (output_dir / "confidence.json").read_text(
                    encoding="utf-8"
                )
            )
            average_pae = float(confidence["average_pae"])
            mean_plddt = float(confidence["mean_plddt"])
            values = torch.tensor([average_pae, mean_plddt])
            if not torch.isfinite(values).all():
                raise AssertionError("confidence values must be finite")
            if not 0.0 <= mean_plddt <= 1.0:
                raise AssertionError("mean_plddt must be in [0, 1]")
            update_check(
                current_index,
                True,
                "confidence values are finite and mean_plddt lies in [0, 1]",
            )

            current_index += 1
            with (output_dir / "ranking_scores.csv").open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 1:
                raise AssertionError(
                    "ranking_scores.csv must contain exactly one data row"
                )
            score = float(rows[0]["score"])
            if abs(score - mean_plddt) > ATOL:
                raise AssertionError(
                    f"ranking score={score}, mean_plddt={mean_plddt}"
                )
            update_check(
                current_index,
                True,
                "ranking score equals mean_plddt",
            )

    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(
    config: dict,
    caller_globals=None,
    caller_locals=None,
) -> list[tuple]:
    """Validate candidate CLI artifacts against trusted reference forward output."""
    CHECKPOINT_NAMES = [
        "infer_metric:pdb_oracle",
        "infer_metric:confidence_oracle",
        "infer_metric:ranking_oracle",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0

    try:
        candidate_module = _load_candidate_modules("alphafold3_inference.py")
        reference_module = _load_reference_modules("alphafold3_inference.py")
        num_recycles = int(config.get("num_recycles", 2))
        num_samples = int(config.get("num_diffusion_samples", 2))
        seed = int(config.get("seed", 7))
        timeout = int(config.get("timeout", 90))
        sequence = str(config.get("sequence", "ACDEFGHIK"))

        candidate_cls = _load_class(candidate_module, "AlphaFold3")
        reference_cls = _load_class(reference_module, "AlphaFold3")
        candidate_model = candidate_cls().cpu().eval()
        reference_model = reference_cls().cpu().eval()
        _initialize_parameters_with_constants(candidate_model)
        _initialize_parameters_with_constants(reference_model)

        with tempfile.TemporaryDirectory(
            prefix="alphafold3_oracle_"
        ) as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "sequences": [
                            {
                                "protein": {
                                    "id": ["A"],
                                    "sequence": sequence,
                                }
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            checkpoint_path = root / "model_state.pt"
            torch.save(candidate_model.state_dict(), checkpoint_path)
            output_dir = root / "outputs"
            script_path = (CANDIDATE_DIR / "alphafold3_inference.py").resolve()
            command = [
                sys.executable,
                str(script_path),
                "--json-path",
                str(input_path),
                "--model-dir",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--run-data-pipeline",
                "true",
                "--run-inference",
                "true",
                "--flash-attention-implementation",
                "xla",
                "--num-recycles",
                str(num_recycles),
                "--num-diffusion-samples",
                str(num_samples),
                "--seed",
                str(seed),
                "--device",
                "cpu",
            ]
            completed = subprocess.run(
                command,
                cwd=str(script_path.parent),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Inference CLI failed\n"
                    f"stdout={completed.stdout}\n"
                    f"stderr={completed.stderr}"
                )

            # reference_features = _move_tensors_to_device(
            #     reference_module.make_features(sequence, num_recycles)
            # )

            reference_features = reference_module.make_features(sequence,num_recycles,)
            # Keep every oracle tensor on the same device as the CLI execution.
            reference_features = {
                name: value.cpu() if isinstance(value, torch.Tensor) else value
                for name, value in reference_features.items()
                }
            with torch.no_grad():
                expected = reference_model(reference_features, seed=seed)

            lines = (output_dir / "sample_0.pdb").read_text(
                encoding="utf-8"
            ).splitlines()
            coordinates = []
            for line in lines:
                if line.startswith("ATOM"):
                    coordinates.append(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ]
                    )
            if not coordinates:
                raise AssertionError("PDB contains no ATOM records")
            coordinate_tensor = torch.tensor(
                coordinates,
                dtype=torch.float32,
            )
            torch.testing.assert_close(
                coordinate_tensor,
                expected["atom_positions"][0].detach().cpu(),
                rtol=0.0,
                atol=5e-4,
            )
            update_check(
                current_index,
                True,
                "sample_0.pdb coordinates match the reference atom_positions",
            )

            current_index += 1
            confidence = json.loads(
                (output_dir / "confidence.json").read_text(
                    encoding="utf-8"
                )
            )
            expected_average_pae = float(
                expected["average_pae"].mean().detach().cpu()
            )
            expected_mean_plddt = float(
                expected["predicted_lddt"].mean().detach().cpu()
            )
            if abs(
                float(confidence["average_pae"]) - expected_average_pae
            ) > ATOL:
                raise AssertionError(
                    "confidence average_pae differs from the reference"
                )
            if abs(
                float(confidence["mean_plddt"]) - expected_mean_plddt
            ) > ATOL:
                raise AssertionError(
                    "confidence mean_plddt differs from the reference"
                )
            update_check(
                current_index,
                True,
                "confidence.json values match the reference confidence outputs",
            )

            current_index += 1
            with (output_dir / "ranking_scores.csv").open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 1:
                raise AssertionError(
                    "ranking_scores.csv must contain exactly one data row"
                )
            if abs(
                float(rows[0]["score"]) - expected_mean_plddt
            ) > ATOL:
                raise AssertionError(
                    "ranking score differs from reference mean pLDDT"
                )
            update_check(
                current_index,
                True,
                "ranking score matches the reference mean pLDDT",
            )

    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks
