"""V2 adapter for the legacy MACE force-curriculum loss evaluator."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import torch

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


def _method_params(config: Dict) -> Dict:
    params = config.get("params", config)
    return params if isinstance(params, dict) else {}


class _ReferenceBatch:
    """Minimal attribute-and-item fixture matching the MACE loss contract."""

    def __init__(self, **values: torch.Tensor) -> None:
        self._values = values
        for name, value in values.items():
            setattr(self, name, value)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._values[name]


def _load_module(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_reference(dtype: torch.dtype = torch.float64) -> _ReferenceBatch:
    return _ReferenceBatch(
        ptr=torch.tensor([0, 2, 5], dtype=torch.long),
        weight=torch.tensor([1.5, 0.5], dtype=dtype),
        # Keep the per-configuration energy coefficient at its native default;
        # the hidden distinction under test is the documented force weighting.
        energy_weight=torch.ones(2, dtype=dtype),
        forces_weight=torch.tensor([0.25, 2.0], dtype=dtype),
        energy=torch.tensor([4.0, -2.0], dtype=dtype),
        forces=torch.tensor(
            [
                [0.2, -0.1, 0.3],
                [-0.4, 0.5, 0.1],
                [1.0, -0.2, 0.0],
                [0.3, 0.7, -0.5],
                [-0.1, 0.4, 0.9],
            ],
            dtype=dtype,
        ),
    )


def _prediction(
    *,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    energy = torch.tensor([3.25, -0.5], dtype=dtype)
    forces = torch.tensor(
        [
            [0.0, 0.1, 0.2],
            [-0.2, 0.3, -0.1],
            [0.4, -0.1, 0.2],
            [0.0, 0.2, -0.4],
            [0.3, 0.1, 0.5],
        ],
        dtype=dtype,
    )
    return {
        "energy": energy.requires_grad_(requires_grad),
        "forces": forces.requires_grad_(requires_grad),
    }


def _instantiate(
    loss_class: type[torch.nn.Module],
    constructor: Dict | None = None,
) -> torch.nn.Module:
    constructor = constructor or {}
    return loss_class(
        energy_weight=constructor.get("energy_weight", 1.75),
        forces_weight=constructor.get("forces_weight", 3.5),
        warmup_steps=constructor.get("warmup_steps", 8),
    ).to(dtype=torch.float64)


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["required_loss_file_is_a_regular_file"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = _method_params(config)
        required_files = params.get("required_files", ["mace_force_curriculum_loss.py"])
        candidate_file = (CANDIDATE_DIR / required_files[0]).resolve()
        candidate_file.relative_to(CANDIDATE_DIR)
        assert candidate_file.is_file(), (
            f"mace_force_curriculum_loss.py was not found at {candidate_file}"
        )
        update_check(current_index, True, "mace_force_curriculum_loss.py is a regular file")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "loss_class_is_present",
        "loss_class_subclasses_torch_module",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = _method_params(config)
        files = params.get("files", ["mace_force_curriculum_loss.py"])
        candidate_file = (CANDIDATE_DIR / files[0]).resolve()
        candidate_file.relative_to(CANDIDATE_DIR)
        module_name = params.get(
            "legacy_module_name", "mace_force_curriculum_loss_submission"
        )
        candidate_module = _load_module(candidate_file, module_name)
        required_interfaces = params.get(
            "required_interfaces", ["MACEForceCurriculumLoss"]
        )
        class_name = required_interfaces[0]
        loss_class = getattr(candidate_module, class_name, None)
        assert isinstance(loss_class, type), "MACEForceCurriculumLoss class is missing"
        update_check(current_index, True, "MACEForceCurriculumLoss class is present")
        current_index += 1
        assert issubclass(loss_class, torch.nn.Module), (
            "MACEForceCurriculumLoss must inherit torch.nn.Module"
        )
        update_check(current_index, True, "MACEForceCurriculumLoss inherits torch.nn.Module")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "constructor_has_required_parameters",
        "energy_weight_default_is_one",
        "forces_weight_default_is_one",
        "warmup_steps_default_is_one_hundred",
        "forward_has_required_parameters",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = _method_params(config)
        model_file = params.get("model_file", "mace_force_curriculum_loss.py")
        candidate_file = (CANDIDATE_DIR / model_file).resolve()
        candidate_file.relative_to(CANDIDATE_DIR)
        module_name = params.get(
            "legacy_module_name", "mace_force_curriculum_loss_submission"
        )
        candidate_module = _load_module(candidate_file, module_name)
        class_name = params.get("model_class", "MACEForceCurriculumLoss")
        loss_class = getattr(candidate_module, class_name, None)
        signature = inspect.signature(loss_class.__init__)
        required = set(
            params.get(
                "required_init_parameters",
                {"energy_weight", "forces_weight", "warmup_steps"},
            )
        )
        missing = required - set(signature.parameters)
        assert not missing, f"constructor missing parameters: {sorted(missing)}"
        update_check(current_index, True, "constructor has all required parameters")
        current_index += 1

        defaults = params.get(
            "constructor_defaults",
            {
                "energy_weight": 1.0,
                "forces_weight": 1.0,
                "warmup_steps": 100,
            },
        )
        assert signature.parameters["energy_weight"].default == defaults["energy_weight"]
        update_check(current_index, True, "energy_weight default is 1.0")
        current_index += 1
        assert signature.parameters["forces_weight"].default == defaults["forces_weight"]
        update_check(current_index, True, "forces_weight default is 1.0")
        current_index += 1
        assert signature.parameters["warmup_steps"].default == defaults["warmup_steps"]
        update_check(current_index, True, "warmup_steps default is 100")
        current_index += 1

        forward_signature = inspect.signature(loss_class.forward)
        forward_required = set(
            params.get(
                "required_forward_parameters",
                {"reference", "prediction", "step", "ddp"},
            )
        )
        missing = forward_required - set(forward_signature.parameters)
        assert not missing, f"forward missing parameters: {sorted(missing)}"
        update_check(current_index, True, "forward has all required parameters")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["candidate_source_has_no_forbidden_imports"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = _method_params(config)
        source_file = params.get("file", "mace_force_curriculum_loss.py")
        candidate_file = (CANDIDATE_DIR / source_file).resolve()
        candidate_file.relative_to(CANDIDATE_DIR)
        tree = ast.parse(candidate_file.read_text(encoding="utf-8"))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        forbidden_exact_imports = params.get(
            "forbidden_exact_imports",
            [
                "evaluate_mace_force_curriculum_loss",
                "mace_force_curriculum_loss_standard",
            ],
        )
        forbidden_prefixes = params.get(
            "forbidden_prefixes",
            [
                "evaluation.eval_datasets.codegen.evaluators",
                "evaluation.eval_datasets.codegen.standard",
            ],
        )
        assert not any(
            name in forbidden_exact_imports
            or any(name.startswith(prefix) for prefix in forbidden_prefixes)
            for name in imported_modules
        ), "candidate imports evaluator or standard code"
        update_check(current_index, True, "candidate source imports no evaluator or standard code")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "standard_negative_output_is_tensor",
        "standard_negative_output_is_scalar",
        "standard_negative_output_is_finite",
        "candidate_negative_output_is_tensor",
        "candidate_negative_output_is_scalar",
        "candidate_negative_output_is_finite",
        "standard_start_output_is_tensor",
        "standard_start_output_is_scalar",
        "standard_start_output_is_finite",
        "candidate_start_output_is_tensor",
        "candidate_start_output_is_scalar",
        "candidate_start_output_is_finite",
        "standard_quarter_output_is_tensor",
        "standard_quarter_output_is_scalar",
        "standard_quarter_output_is_finite",
        "candidate_quarter_output_is_tensor",
        "candidate_quarter_output_is_scalar",
        "candidate_quarter_output_is_finite",
        "standard_half_output_is_tensor",
        "standard_half_output_is_scalar",
        "standard_half_output_is_finite",
        "candidate_half_output_is_tensor",
        "candidate_half_output_is_scalar",
        "candidate_half_output_is_finite",
        "standard_complete_output_is_tensor",
        "standard_complete_output_is_scalar",
        "standard_complete_output_is_finite",
        "candidate_complete_output_is_tensor",
        "candidate_complete_output_is_scalar",
        "candidate_complete_output_is_finite",
        "standard_after_output_is_tensor",
        "standard_after_output_is_scalar",
        "standard_after_output_is_finite",
        "candidate_after_output_is_tensor",
        "candidate_after_output_is_scalar",
        "candidate_after_output_is_finite",
        "zero_warmup_is_rejected",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = _method_params(config)
        model_file = params.get("model_file", "mace_force_curriculum_loss.py")
        candidate_file = (CANDIDATE_DIR / model_file).resolve()
        candidate_file.relative_to(CANDIDATE_DIR)
        candidate_module = _load_module(
            candidate_file,
            params.get(
                "legacy_module_name", "mace_force_curriculum_loss_submission"
            ),
        )
        class_name = params.get("model_class", "MACEForceCurriculumLoss")
        candidate_class = getattr(candidate_module, class_name)
        reference_file = (REFERENCE_DIR / "mace_force_curriculum_loss.py").resolve()
        reference_file.relative_to(REFERENCE_DIR.resolve())
        reference_module = _load_module(
            reference_file, "mace_force_curriculum_loss_standard"
        )
        standard_class = getattr(reference_module, "MACEForceCurriculumLoss")
        constructor = params.get(
            "constructor",
            {
                "energy_weight": 1.75,
                "forces_weight": 3.5,
                "warmup_steps": 8,
            },
        )
        standard = _instantiate(standard_class, constructor)
        candidate = _instantiate(candidate_class, constructor)
        reference = _make_reference()
        for name, step in (
            ("negative", -3),
            ("start", 0),
            ("quarter", 2),
            ("half", torch.tensor(4)),
            ("complete", 8),
            ("after", 20),
        ):
            expected = standard(reference, _prediction(), step=step, ddp=False)
            assert isinstance(expected, torch.Tensor), "loss must be a tensor"
            update_check(
                current_index, True, f"standard {name} loss is a tensor"
            )
            current_index += 1
            assert expected.ndim == 0, (
                f"loss must be scalar, got shape {tuple(expected.shape)}"
            )
            update_check(
                current_index, True, f"standard {name} loss is scalar"
            )
            current_index += 1
            assert torch.isfinite(expected), "loss must be finite"
            update_check(
                current_index, True, f"standard {name} loss is finite"
            )
            current_index += 1

            actual = candidate(reference, _prediction(), step=step, ddp=False)
            assert isinstance(actual, torch.Tensor), "loss must be a tensor"
            update_check(
                current_index, True, f"candidate {name} loss is a tensor"
            )
            current_index += 1
            assert actual.ndim == 0, (
                f"loss must be scalar, got shape {tuple(actual.shape)}"
            )
            update_check(
                current_index, True, f"candidate {name} loss is scalar"
            )
            current_index += 1
            assert torch.isfinite(actual), "loss must be finite"
            update_check(
                current_index, True, f"candidate {name} loss is finite"
            )
            current_index += 1

        try:
            candidate_class(warmup_steps=0)
        except ValueError:
            update_check(current_index, True, "warmup_steps=0 raises ValueError")
            current_index += 1
        else:
            raise AssertionError("warmup_steps=0 must raise ValueError")
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_equivalence(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "standard_negative_output_is_tensor",
        "standard_negative_output_is_scalar",
        "standard_negative_output_is_finite",
        "candidate_negative_output_is_tensor",
        "candidate_negative_output_is_scalar",
        "candidate_negative_output_is_finite",
        "candidate_matches_standard_at_negative",
        "standard_start_output_is_tensor",
        "standard_start_output_is_scalar",
        "standard_start_output_is_finite",
        "candidate_start_output_is_tensor",
        "candidate_start_output_is_scalar",
        "candidate_start_output_is_finite",
        "candidate_matches_standard_at_start",
        "standard_quarter_output_is_tensor",
        "standard_quarter_output_is_scalar",
        "standard_quarter_output_is_finite",
        "candidate_quarter_output_is_tensor",
        "candidate_quarter_output_is_scalar",
        "candidate_quarter_output_is_finite",
        "candidate_matches_standard_at_quarter",
        "standard_half_output_is_tensor",
        "standard_half_output_is_scalar",
        "standard_half_output_is_finite",
        "candidate_half_output_is_tensor",
        "candidate_half_output_is_scalar",
        "candidate_half_output_is_finite",
        "candidate_matches_standard_at_half",
        "standard_complete_output_is_tensor",
        "standard_complete_output_is_scalar",
        "standard_complete_output_is_finite",
        "candidate_complete_output_is_tensor",
        "candidate_complete_output_is_scalar",
        "candidate_complete_output_is_finite",
        "candidate_matches_standard_at_complete",
        "standard_after_output_is_tensor",
        "standard_after_output_is_scalar",
        "standard_after_output_is_finite",
        "candidate_after_output_is_tensor",
        "candidate_after_output_is_scalar",
        "candidate_after_output_is_finite",
        "candidate_matches_standard_after_warmup",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = _method_params(config)
        model_file = params.get("model_file", "mace_force_curriculum_loss.py")
        candidate_file = (CANDIDATE_DIR / model_file).resolve()
        candidate_file.relative_to(CANDIDATE_DIR)
        candidate_module = _load_module(
            candidate_file,
            params.get(
                "legacy_module_name", "mace_force_curriculum_loss_submission"
            ),
        )
        candidate_class_name = params.get(
            "candidate_class", "MACEForceCurriculumLoss"
        )
        candidate_class = getattr(candidate_module, candidate_class_name)
        reference_file = (REFERENCE_DIR / "mace_force_curriculum_loss.py").resolve()
        reference_file.relative_to(REFERENCE_DIR.resolve())
        reference_module = _load_module(
            reference_file, "mace_force_curriculum_loss_standard"
        )
        reference_class_name = params.get(
            "reference_class",
            "mace_force_curriculum_loss_standard.MACEForceCurriculumLoss",
        ).rsplit(".", 1)[-1]
        standard_class = getattr(reference_module, reference_class_name)
        constructor = params.get(
            "constructor",
            {
                "energy_weight": 1.75,
                "forces_weight": 3.5,
                "warmup_steps": 8,
            },
        )
        standard = _instantiate(standard_class, constructor)
        candidate = _instantiate(candidate_class, constructor)
        reference = _make_reference()
        comparison = params.get(
            "comparison", {"rtol": 1e-7, "atol": 1e-10}
        )
        rtol = comparison.get("rtol", 1e-7)
        atol = comparison.get("atol", 1e-10)

        losses: dict[str, float] = {}
        max_error = 0.0
        for name, step in (
            ("negative", -3),
            ("start", 0),
            ("quarter", 2),
            ("half", torch.tensor(4)),
            ("complete", 8),
            ("after", 20),
        ):
            expected = standard(reference, _prediction(), step=step, ddp=False)
            assert isinstance(expected, torch.Tensor), "loss must be a tensor"
            update_check(
                current_index, True, f"standard {name} loss is a tensor"
            )
            current_index += 1
            assert expected.ndim == 0, (
                f"loss must be scalar, got shape {tuple(expected.shape)}"
            )
            update_check(
                current_index, True, f"standard {name} loss is scalar"
            )
            current_index += 1
            assert torch.isfinite(expected), "loss must be finite"
            update_check(
                current_index, True, f"standard {name} loss is finite"
            )
            current_index += 1

            actual = candidate(reference, _prediction(), step=step, ddp=False)
            assert isinstance(actual, torch.Tensor), "loss must be a tensor"
            update_check(
                current_index, True, f"candidate {name} loss is a tensor"
            )
            current_index += 1
            assert actual.ndim == 0, (
                f"loss must be scalar, got shape {tuple(actual.shape)}"
            )
            update_check(
                current_index, True, f"candidate {name} loss is scalar"
            )
            current_index += 1
            assert torch.isfinite(actual), "loss must be finite"
            update_check(
                current_index, True, f"candidate {name} loss is finite"
            )
            current_index += 1

            torch.testing.assert_close(
                actual, expected, rtol=rtol, atol=atol
            )
            update_check(
                current_index,
                True,
                f"candidate and standard losses match at {name}",
            )
            current_index += 1
            error = float(torch.abs(actual - expected))
            max_error = max(max_error, error)
            losses[name] = float(actual)
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_method_correctness(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "negative_loss_is_a_tensor",
        "negative_loss_is_scalar",
        "negative_loss_is_finite",
        "start_loss_is_a_tensor",
        "start_loss_is_scalar",
        "start_loss_is_finite",
        "quarter_loss_is_a_tensor",
        "quarter_loss_is_scalar",
        "quarter_loss_is_finite",
        "half_loss_is_a_tensor",
        "half_loss_is_scalar",
        "half_loss_is_finite",
        "complete_loss_is_a_tensor",
        "complete_loss_is_scalar",
        "complete_loss_is_finite",
        "after_loss_is_a_tensor",
        "after_loss_is_scalar",
        "after_loss_is_finite",
        "negative_step_clamps_to_start",
        "loss_increases_linearly_during_warmup",
        "loss_clamps_after_warmup",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = _method_params(config)
        model_file = params.get("model_file", "mace_force_curriculum_loss.py")
        candidate_file = (CANDIDATE_DIR / model_file).resolve()
        candidate_file.relative_to(CANDIDATE_DIR)
        candidate_module = _load_module(
            candidate_file,
            params.get(
                "legacy_module_name", "mace_force_curriculum_loss_submission"
            ),
        )
        class_name = params.get("loss", "MACEForceCurriculumLoss")
        candidate_class = getattr(candidate_module, class_name)
        constructor = params.get(
            "constructor",
            {
                "energy_weight": 1.75,
                "forces_weight": 3.5,
                "warmup_steps": 8,
            },
        )
        candidate = _instantiate(candidate_class, constructor)
        reference = _make_reference()

        losses: dict[str, float] = {}
        for name, step in (
            ("negative", -3),
            ("start", 0),
            ("quarter", 2),
            ("half", torch.tensor(4)),
            ("complete", 8),
            ("after", 20),
        ):
            actual = candidate(reference, _prediction(), step=step, ddp=False)
            assert isinstance(actual, torch.Tensor), "loss must be a tensor"
            update_check(current_index, True, f"{name} loss is a tensor")
            current_index += 1
            assert actual.ndim == 0, (
                f"loss must be scalar, got shape {tuple(actual.shape)}"
            )
            update_check(current_index, True, f"{name} loss is scalar")
            current_index += 1
            assert torch.isfinite(actual), "loss must be finite"
            update_check(current_index, True, f"{name} loss is finite")
            current_index += 1
            losses[name] = float(actual)

        assert losses["negative"] == losses["start"], (
            "negative steps must clamp to zero"
        )
        update_check(
            current_index, True, "negative and start losses are equal"
        )
        current_index += 1
        assert losses["start"] < losses["quarter"] < losses["half"] < losses["complete"], (
            "force contribution must increase linearly during warmup"
        )
        update_check(
            current_index, True, "loss increases through warmup steps"
        )
        current_index += 1
        assert losses["complete"] == losses["after"], (
            "force contribution must clamp after warmup"
        )
        update_check(
            current_index, True, "complete and post-warmup losses are equal"
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_parameter_update(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "gradient_loss_is_a_tensor",
        "gradient_loss_is_scalar",
        "gradient_loss_is_finite",
        "gradient_backward_executes",
        "energy_gradient_is_present_and_finite",
        "forces_gradient_is_present_and_finite",
        "energy_gradient_is_nonzero",
        "forces_gradient_is_nonzero",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        params = _method_params(config)
        model_file = params.get("model_file", "mace_force_curriculum_loss.py")
        candidate_file = (CANDIDATE_DIR / model_file).resolve()
        candidate_file.relative_to(CANDIDATE_DIR)
        candidate_module = _load_module(
            candidate_file,
            params.get(
                "legacy_module_name", "mace_force_curriculum_loss_submission"
            ),
        )
        class_name = params.get("loss_class", "MACEForceCurriculumLoss")
        candidate_class = getattr(candidate_module, class_name)
        constructor = params.get(
            "constructor",
            {
                "energy_weight": 1.75,
                "forces_weight": 3.5,
                "warmup_steps": 8,
            },
        )
        candidate = _instantiate(candidate_class, constructor)
        reference = _make_reference()
        prediction_params = params.get("prediction", {})
        gradient_prediction = _prediction(
            requires_grad=prediction_params.get("requires_grad", True)
        )
        step = params.get("step", 4)
        gradient_loss = candidate(
            reference, gradient_prediction, step=step, ddp=False
        )
        assert isinstance(gradient_loss, torch.Tensor), "loss must be a tensor"
        update_check(current_index, True, "gradient loss is a tensor")
        current_index += 1
        assert gradient_loss.ndim == 0, (
            f"loss must be scalar, got shape {tuple(gradient_loss.shape)}"
        )
        update_check(current_index, True, "gradient loss is scalar")
        current_index += 1
        assert torch.isfinite(gradient_loss), "loss must be finite"
        update_check(current_index, True, "gradient loss is finite")
        current_index += 1

        gradient_loss.backward()
        update_check(current_index, True, "gradient loss backward() completed")
        current_index += 1
        energy_gradient = gradient_prediction["energy"].grad
        forces_gradient = gradient_prediction["forces"].grad
        assert energy_gradient is not None and torch.isfinite(energy_gradient).all()
        update_check(
            current_index, True, "energy gradient is present and finite"
        )
        current_index += 1
        assert forces_gradient is not None and torch.isfinite(forces_gradient).all()
        update_check(
            current_index, True, "forces gradient is present and finite"
        )
        current_index += 1
        assert torch.count_nonzero(energy_gradient) > 0
        update_check(current_index, True, "energy gradient has nonzero elements")
        current_index += 1
        assert torch.count_nonzero(forces_gradient) > 0
        energy_gradient_norm = float(torch.linalg.vector_norm(energy_gradient))
        forces_gradient_norm = float(torch.linalg.vector_norm(forces_gradient))
        update_check(
            current_index,
            True,
            (
                "forces gradient has nonzero elements; "
                f"energy_gradient_norm={energy_gradient_norm}, "
                f"forces_gradient_norm={forces_gradient_norm}"
            ),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
