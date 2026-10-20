"""V2 adapter for the legacy GenScore training evaluator."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict

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

FILENAME, ENV_NAME = "genscore_train.py", "GENSCORE_TRAIN_SUBMISSION_DIR"
CTOR = {
    "ligand_features": 8,
    "pocket_features": 10,
    "hidden_size": 32,
    "radial_bins": 8,
    "num_layers": 2,
}
ATTRS = (
    "ligand_encoder",
    "pocket_encoder",
    "distance_expansion",
    "ligand_layers",
    "ligand_layers.0.edge_gate",
    "pocket_layers",
    "cross_projection",
    "score_head",
)
EXPECTED = {
    "file": ("file:genscore_train.py", "file:standard:train_config.json"),
    "data": (
        "data:graph_shapes",
        "data:mask_contract",
        "data:invalid_coordinates_rejected",
    ),
    "train_script": (
        "train_script:independent_from_onescience",
        "train_script:cli_flags",
        "train_script:adamw_backward",
        "train_script:gradient_clipping",
        "train_script:structured_artifacts",
    ),
    "model_structure": (
        "model_structure:nn_module",
        "model_structure:required_components",
        "model_structure:state_dict_contract",
        "model_structure:module_order",
        "model_structure:constructor_contract",
    ),
    "forward_executable": (
        "forward_executable:tensor",
        "forward_executable:shape",
        "forward_executable:finite",
        "forward_executable:deterministic",
    ),
    "forward_equivalence": ("forward_equivalence:forward_equivalence",),
    "traning": (
        "traning:short_fit",
        "traning:parameter_update",
        "traning:npz_fit",
    ),
    "artifacts": ("artifacts:checkpoint", "artifacts:metrics"),
}
_ADAPTER_STATE: Dict[str, Any] = {}

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def dotted(value, name):
    for part in name.split("."):
        value = getattr(value, part)
    return value

def source_info(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = any(
        (
            isinstance(node, ast.Import)
            and any(
                alias.name.split(".")[0] == "onescience" for alias in node.names
            )
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "onescience"
        )
        for node in ast.walk(tree)
    )
    return (
        source,
        imports,
        {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)},
    )

def fixture(torch):
    generator = torch.Generator().manual_seed(73)
    return (
        torch.randn(2, 5, 8, generator=generator),
        torch.randn(2, 5, 3, generator=generator),
        torch.randn(2, 7, 10, generator=generator),
        torch.randn(2, 7, 3, generator=generator),
        torch.ones(2, 5, dtype=torch.bool),
        torch.ones(2, 7, dtype=torch.bool),
    )

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "file:genscore_train.py",
        "file:standard:train_config.json",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        path = (
            CANDIDATE_DIR
            / Path(config.get("submission_filename", FILENAME)).name
        ).resolve()
        candidate = load_module("genscore_train_candidate", path)
        reference_path = (REFERENCE_DIR / FILENAME).resolve()
        reference = load_module("genscore_train_reference", reference_path)
        standard_config_path = (
            REFERENCE_DIR
            / Path(config.get("standard_config_path", "train_config.json")).name
        ).resolve()
        assets = json.loads(
            standard_config_path.read_text(encoding="utf-8")
        )["external_assets"]
        _ADAPTER_STATE.update(
            torch=torch,
            path=path,
            candidate=candidate,
            reference=reference,
        )
        ok, msg = True, str(path)
        update_check(current_index, ok, msg)
        current_index += 1
        ok = (
            assets.get("mount_required") is True
            and Path(assets["weights"]).exists()
            and Path(assets["dataset"]).exists()
        )
        msg = str(assets)
        update_check(current_index, ok, msg)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_method_correctness(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "data:graph_shapes",
        "data:mask_contract",
        "data:invalid_coordinates_rejected",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        candidate_path = (
            CANDIDATE_DIR
            / Path(
                config.get("candidate", {}).get("file", FILENAME)
            ).name
        ).resolve()
        module = load_module("genscore_train_candidate_data", candidate_path)
        data = module.mock_dataset()
        aligned = (
            data["ligand_features"].shape[:2]
            == data["ligand_coordinates"].shape[:2]
            and data["pocket_features"].shape[:2]
            == data["pocket_coordinates"].shape[:2]
        )
        update_check(
            current_index,
            aligned,
            f"ligand={tuple(data['ligand_features'].shape)}, "
            f"pocket={tuple(data['pocket_features'].shape)}",
        )
        current_index += 1
        update_check(
            current_index,
            data["ligand_mask"].dtype == torch.bool
            and data["pocket_mask"].dtype == torch.bool,
            "boolean masks",
        )
        current_index += 1
        rejected = False
        try:
            module.GenScoreModel()(
                data["ligand_features"][:2],
                data["ligand_coordinates"][:2, :, :2],
                data["pocket_features"][:2],
                data["pocket_coordinates"][:2],
            )
        except ValueError:
            rejected = True
        update_check(
            current_index,
            rejected,
            "invalid coordinates fail",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_train_structure(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "train_script:independent_from_onescience",
        "train_script:cli_flags",
        "train_script:adamw_backward",
        "train_script:gradient_clipping",
        "train_script:structured_artifacts",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        files = config.get("files", [FILENAME])
        path = (CANDIDATE_DIR / Path(files[0]).name).resolve()
        source, imports, names = source_info(path)
        flags = (
            "--dataset-npz",
            "--mock-data",
            "--output-dir",
            "--max-steps",
            "--seed",
        )
        update_check(
            current_index,
            not imports,
            "no onescience import",
        )
        current_index += 1
        update_check(
            current_index,
            all(flag in source for flag in flags),
            str(flags),
        )
        current_index += 1
        update_check(
            current_index,
            "AdamW" in names and "backward" in names,
            str(sorted(names)),
        )
        current_index += 1
        update_check(
            current_index,
            "clip_grad_norm_" in names,
            str(sorted(names)),
        )
        current_index += 1
        update_check(
            current_index,
            "metrics.jsonl" in source and "model_state_dict" in source,
            "structured outputs",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_structure(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "model_structure:nn_module",
        "model_structure:required_components",
        "model_structure:state_dict_contract",
        "model_structure:module_order",
        "model_structure:constructor_contract",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        candidate_file = (
            CANDIDATE_DIR
            / Path(config.get("candidate_file", FILENAME)).name
        ).resolve()
        reference_file = (
            REFERENCE_DIR
            / Path(config.get("reference_file", FILENAME)).name
        ).resolve()
        candidate = load_module(
            "genscore_train_candidate_structure", candidate_file
        )
        reference = load_module(
            "genscore_train_reference_structure", reference_file
        )
        model_class = config.get("model_class", "GenScoreModel")
        constructor_args = config.get("constructor_args", CTOR)
        model = getattr(candidate, model_class)(**constructor_args)
        reference_model = getattr(reference, model_class)(**constructor_args)
        result = model.load_state_dict(reference_model.state_dict(), strict=True)
        model.eval()
        reference_model.eval()
        _ADAPTER_STATE.update(
            torch=torch,
            candidate=candidate,
            reference=reference,
            model=model,
            reference_model=reference_model,
        )
        missing, mismatched = [], []
        for name in ATTRS:
            try:
                if (
                    type(dotted(model, name)).__name__
                    != type(dotted(reference_model, name)).__name__
                ):
                    mismatched.append(name)
            except AttributeError:
                missing.append(name)
        state, ref = model.state_dict(), reference_model.state_dict()
        children = [
            (name, type(module).__name__)
            for name, module in model.named_children()
        ]
        ref_children = [
            (name, type(module).__name__)
            for name, module in reference_model.named_children()
        ]
        signature = inspect.signature(
            getattr(candidate, model_class).__init__
        )
        update_check(
            current_index,
            isinstance(model, torch.nn.Module),
            type(model).__name__,
        )
        current_index += 1
        update_check(
            current_index,
            not missing and not mismatched,
            f"missing={missing}, mismatched={mismatched}",
        )
        current_index += 1
        update_check(
            current_index,
            list(state) == list(ref)
            and not result.missing_keys
            and not result.unexpected_keys,
            f"keys={len(state)}",
        )
        current_index += 1
        update_check(
            current_index,
            children == ref_children,
            f"candidate={children}, reference={ref_children}",
        )
        current_index += 1
        update_check(
            current_index,
            all(name in signature.parameters for name in CTOR),
            str(signature),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_executable(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "forward_executable:tensor",
        "forward_executable:shape",
        "forward_executable:finite",
        "forward_executable:deterministic",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        model_file = (
            CANDIDATE_DIR
            / Path(config.get("model_file", FILENAME)).name
        ).resolve()
        module = load_module("genscore_train_candidate_forward", model_file)
        model = getattr(
            module, config.get("model_class", "GenScoreModel")
        )(**CTOR)
        model.eval()
        args = fixture(torch)
        with torch.inference_mode():
            output, repeated = model(*args), model(*args)
        update_check(
            current_index,
            isinstance(output, torch.Tensor),
            str(type(output)),
        )
        current_index += 1
        update_check(
            current_index,
            tuple(output.shape) == (2,),
            str(tuple(output.shape)),
        )
        current_index += 1
        update_check(
            current_index,
            bool(torch.isfinite(output).all()),
            "finite",
        )
        current_index += 1
        update_check(
            current_index,
            torch.equal(output, repeated),
            "repeat equality",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_equivalence(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_equivalence:forward_equivalence"]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        candidate_file = (
            CANDIDATE_DIR
            / Path(
                config.get("candidate", {}).get("model_file", FILENAME)
            ).name
        ).resolve()
        reference_file = (
            REFERENCE_DIR
            / Path(
                config.get("reference", {}).get("model_file", FILENAME)
            ).name
        ).resolve()
        candidate_module = load_module(
            "genscore_train_candidate_equivalence", candidate_file
        )
        reference_module = load_module(
            "genscore_train_reference_equivalence", reference_file
        )
        model_class = config.get("candidate", {}).get(
            "model_class", "GenScoreModel"
        )
        reference_class = config.get("reference", {}).get(
            "model_class", "GenScoreModel"
        )
        model = getattr(candidate_module, model_class)(**CTOR)
        reference_model = getattr(reference_module, reference_class)(**CTOR)
        model.load_state_dict(reference_model.state_dict(), strict=True)
        model.eval()
        reference_model.eval()
        args = fixture(torch)
        with torch.inference_mode():
            candidate, reference = model(*args), reference_model(*args)
        update_check(
            current_index,
            bool(torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6)),
            f"max_diff={float((candidate - reference).abs().max())}",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_smoke_training(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "traning:short_fit",
        "traning:parameter_update",
        "traning:npz_fit",
    ]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        run_file = (
            CANDIDATE_DIR / Path(config.get("run_file", FILENAME)).name
        ).resolve()
        manager = tempfile.TemporaryDirectory(prefix="genscore_train_")
        output = Path(manager.name) / "output"
        _ADAPTER_STATE.update(manager=manager, output=output, torch=torch)
        process = subprocess.run(
            [
                sys.executable,
                str(run_file),
                "--mock-data",
                "--output-dir",
                str(output),
                "--max-steps",
                "5",
                "--batch-size",
                "4",
                "--seed",
                "41",
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        checkpoint = output / "genscore_train.pt"
        _ADAPTER_STATE.update(checkpoint=checkpoint)
        updated = False
        if checkpoint.is_file():
            try:
                payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=True
                )
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            torch.manual_seed(41)
            reference_file = (REFERENCE_DIR / FILENAME).resolve()
            reference = load_module(
                "genscore_train_reference_smoke", reference_file
            )
            initial = reference.GenScoreModel(**CTOR).state_dict()
            state = payload.get("model_state_dict", {})
            updated = set(state) == set(initial) and any(
                not torch.equal(initial[name], state[name]) for name in initial
            )
        import numpy as np

        root = Path(manager.name)
        reference_file = (REFERENCE_DIR / FILENAME).resolve()
        reference = load_module(
            "genscore_train_reference_smoke_dataset", reference_file
        )
        fixture_data = reference.mock_dataset()
        dataset = root / "complexes.npz"
        np.savez_compressed(
            dataset,
            **{name: value.numpy() for name, value in fixture_data.items()},
        )
        npz_output = root / "npz_output"
        npz_process = subprocess.run(
            [
                sys.executable,
                str(run_file),
                "--dataset-npz",
                str(dataset),
                "--output-dir",
                str(npz_output),
                "--max-steps",
                "1",
                "--batch-size",
                "2",
                "--seed",
                "41",
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        update_check(
            current_index,
            process.returncode == 0,
            f"returncode={process.returncode}, "
            f"stderr={process.stderr[-1000:]}",
        )
        current_index += 1
        update_check(current_index, updated, str(checkpoint))
        current_index += 1
        update_check(
            current_index,
            npz_process.returncode == 0
            and (npz_output / "metrics.jsonl").is_file()
            and (npz_output / "genscore_train.pt").is_file(),
            f"returncode={npz_process.returncode}, "
            f"stderr={npz_process.stderr[-1000:]}",
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_smoke_artifacts(
    config: Dict, caller_globals=None, caller_locals=None
) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["artifacts:checkpoint", "artifacts:metrics"]
    checks = [
        (name, False, "Not executed due to earlier failure")
        for name in CHECKPOINT_NAMES
    ]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        checkpoint = _ADAPTER_STATE["checkpoint"]
        output = _ADAPTER_STATE["output"]
        payload = {}
        if checkpoint.is_file():
            try:
                payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=True
                )
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "step",
            "config",
        }
        update_check(
            current_index,
            isinstance(payload, dict)
            and required.issubset(payload)
            and payload["step"] == 5,
            f"keys={sorted(payload)}",
        )
        current_index += 1
        metrics = output / "metrics.jsonl"
        rows = (
            [
                json.loads(line)
                for line in metrics.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if metrics.is_file()
            else []
        )
        update_check(
            current_index,
            len(rows) == 5
            and all(
                row.get("step") == index
                and math.isfinite(float(row.get("train_loss", math.nan)))
                for index, row in enumerate(rows, 1)
            ),
            str(rows),
        )
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
