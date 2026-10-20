"""V2 adapter for the NequIX energy-force evaluator."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict

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

DATA_PATH = os.environ.get(
    "NEQUIX_TRAIN_DATA_PATH",
    "/public/share/sugonhpcapp01/onestore/onedatasets/matchem/mptrj_smoke/data_00000.aselmdb",
)
DATA_SHA256 = "b32cfaec1fdefb3a823e0bd19b7fab32d3a0c325e644dd388c5cc75117b4143b"
EXPECTED_STATE_COUNT = 107
EXPECTED_STATE_NUMEL = 732710
EXPECTED_PARAM_COUNT = 38
FRAME_INDEX = 0
FRAME_NUM_ATOMS = 28
FRAME_ENERGY = -170.60565946
SEED = 456
MIN_CHANGED_PARAMS = 38

REQUIRED_FILES = (
    "nequix_energy_force_train.py",
    "train_nequix_from_scratch.py",
    "nequix_train.yaml",
    "nequix_arch_config.json",
    "portable_nequix/__init__.py",
)
PROHIBITED_IMPORTS = {"onescience", "nequip", "mace", "mattersim", "nequix"}

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def inspect_source(submission):
    prohibited = []
    for path in submission.rglob("*.py"):
        if "portable_nequix" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                prohibited.extend(alias.name for alias in node.names if alias.name.split(".")[0] in PROHIBITED_IMPORTS)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in PROHIBITED_IMPORTS:
                prohibited.append(node.module)
    return prohibited

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["required_files_regular", "prohibited_imports_absent"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        missing = [name for name in REQUIRED_FILES if not (submission / name).is_file()]
        update_check(current_index, not missing, json.dumps({"missing": missing}, sort_keys=True))
        current_index += 1
        prohibited = inspect_source(submission) if not missing else []
        update_check(
            current_index,
            not prohibited,
            json.dumps({"missing": missing, "prohibited_active_imports": prohibited}, sort_keys=True),
        )
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "data_sha256",
        "structures_count",
        "energy_shape",
        "forces_shape",
        "positions_shape",
        "energy_value",
        "forces_finite",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        module = load_module(
            "nequix_energy_force_train_submission",
            CANDIDATE_DIR / "nequix_energy_force_train.py",
        )
        structures = module.load_training_structures(str(DATA_PATH))
        batch = module.build_train_batch(str(DATA_PATH), FRAME_INDEX, "cpu")
        data = batch["data"]
        energy = batch["energy"]
        forces = batch["forces"]
        shapes = {"energy": list(energy.shape), "forces": list(forces.shape), "positions": list(data.positions.shape)}
        data_hash = sha256(DATA_PATH)
        update_check(current_index, data_hash == DATA_SHA256, json.dumps({"data_sha256": data_hash}, sort_keys=True))
        current_index += 1
        update_check(current_index, len(structures) >= FRAME_INDEX + 1, json.dumps({"structures": len(structures)}, sort_keys=True))
        current_index += 1
        update_check(current_index, shapes["energy"] == [1, 1], json.dumps({"energy_shape": shapes["energy"]}, sort_keys=True))
        current_index += 1
        update_check(current_index, shapes["forces"] == [FRAME_NUM_ATOMS, 3], json.dumps({"forces_shape": shapes["forces"]}, sort_keys=True))
        current_index += 1
        update_check(current_index, shapes["positions"] == [FRAME_NUM_ATOMS, 3], json.dumps({"positions_shape": shapes["positions"]}, sort_keys=True))
        current_index += 1
        energy_value = float(energy.item())
        update_check(
            current_index,
            abs(energy_value - FRAME_ENERGY) <= 1e-2,
            json.dumps({"energy": energy_value, "expected_energy": FRAME_ENERGY}, sort_keys=True),
        )
        current_index += 1
        forces_finite = bool(torch.isfinite(forces).all())
        update_check(current_index, forces_finite, json.dumps({"forces_finite": forces_finite}, sort_keys=True))
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "required_train_apis",
        "required_training_config_keys",
        "optimizer",
        "learning_rate",
        "max_steps",
        "structure_index",
        "seed",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        train_module = load_module(
            "nequix_energy_force_train_submission_train",
            submission / "train_nequix_from_scratch.py",
        )
        required = ("load_config", "trainable_named_parameters", "build_model", "build_trainer", "train_step", "train", "main")
        missing = [name for name in required if not callable(getattr(train_module, name, None))]
        update_check(current_index, not missing, json.dumps({"missing_apis": missing}, sort_keys=True))
        current_index += 1
        config_data = train_module.load_config(str(submission / "nequix_train.yaml"))
        training = config_data.get("training", {})
        required_keys = {"optimizer", "lr", "weight_decay", "energy_weight", "force_weight", "gradient_clip_norm", "max_steps", "structure_index", "seed"}
        missing_keys = sorted(required_keys - set(training))
        update_check(current_index, not missing_keys, json.dumps({"missing_config_keys": missing_keys}, sort_keys=True))
        current_index += 1
        optimizer_ok = not missing_keys and training["optimizer"] == "AdamW"
        update_check(current_index, optimizer_ok, json.dumps({"optimizer": training.get("optimizer")}, sort_keys=True))
        current_index += 1
        lr_ok = not missing_keys and float(training["lr"]) == 1e-3
        update_check(current_index, lr_ok, json.dumps({"lr": training.get("lr")}, sort_keys=True))
        current_index += 1
        max_steps_ok = not missing_keys and int(training["max_steps"]) == 1
        update_check(current_index, max_steps_ok, json.dumps({"max_steps": training.get("max_steps")}, sort_keys=True))
        current_index += 1
        structure_index_ok = not missing_keys and int(training["structure_index"]) == FRAME_INDEX
        update_check(current_index, structure_index_ok, json.dumps({"structure_index": training.get("structure_index")}, sort_keys=True))
        current_index += 1
        seed_ok = not missing_keys and int(training["seed"]) == SEED
        update_check(current_index, seed_ok, json.dumps({"seed": training.get("seed")}, sort_keys=True))
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_eval_script_contract(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["optional_eval_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        update_check(current_index, True, "optional evaluation script is not part of this task")
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "model_class",
        "module_order",
        "layer_count",
        "state_count",
        "state_numel",
        "parameter_count",
        "cutoff",
        "species_count",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = load_module(
            "nequix_energy_force_train_submission",
            submission / "nequix_energy_force_train.py",
        )
        train_module = load_module(
            "nequix_energy_force_train_submission_train",
            submission / "train_nequix_from_scratch.py",
        )
        candidate_config = train_module.load_config(str(submission / "nequix_train.yaml"))
        model = module.build_model(candidate_config, "cpu")
        order = list(model._modules)
        expected_order = ["layers", "readout"]
        state = model.state_dict()
        n_params = len(list(model.parameters()))
        n_layers = len(model.layers)
        state_numel = sum(v.numel() for v in state.values() if torch.is_tensor(v))
        update_check(current_index, type(model).__name__ == "NequixTorch", json.dumps({"model": type(model).__name__}, sort_keys=True))
        current_index += 1
        update_check(current_index, order == expected_order, json.dumps({"module_order": order}, sort_keys=True))
        current_index += 1
        update_check(current_index, n_layers == 4, json.dumps({"n_layers": n_layers}, sort_keys=True))
        current_index += 1
        update_check(current_index, len(state) == EXPECTED_STATE_COUNT, json.dumps({"state_count": len(state)}, sort_keys=True))
        current_index += 1
        update_check(current_index, state_numel == EXPECTED_STATE_NUMEL, json.dumps({"state_numel": state_numel}, sort_keys=True))
        current_index += 1
        update_check(current_index, n_params == EXPECTED_PARAM_COUNT, json.dumps({"param_count": n_params}, sort_keys=True))
        current_index += 1
        update_check(current_index, abs(float(model.cutoff) - 6.0) < 1e-9, json.dumps({"cutoff": float(model.cutoff)}, sort_keys=True))
        current_index += 1
        update_check(current_index, model.n_species == 89, json.dumps({"n_species": model.n_species}, sort_keys=True))
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["energy_shape", "forces_shape", "energy_finite", "forces_finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = load_module(
            "nequix_energy_force_train_submission",
            submission / "nequix_energy_force_train.py",
        )
        train_module = load_module(
            "nequix_energy_force_train_submission_train",
            submission / "train_nequix_from_scratch.py",
        )
        candidate_config = train_module.load_config(str(submission / "nequix_train.yaml"))
        model = module.build_model(candidate_config, "cpu")
        model.train()
        batch = module.build_train_batch(str(DATA_PATH), FRAME_INDEX, "cpu")
        out = module.predict_energy_forces(model, batch)
        shapes = {"energy": list(out["energy"].shape), "forces": list(out["forces"].shape)}
        update_check(current_index, shapes["energy"] == [1], json.dumps({"energy_shape": shapes["energy"]}, sort_keys=True))
        current_index += 1
        update_check(current_index, shapes["forces"] == [FRAME_NUM_ATOMS, 3], json.dumps({"forces_shape": shapes["forces"]}, sort_keys=True))
        current_index += 1
        energy_finite = bool(torch.isfinite(out["energy"]).all())
        update_check(current_index, energy_finite, json.dumps({"energy_finite": energy_finite}, sort_keys=True))
        current_index += 1
        forces_finite = bool(torch.isfinite(out["forces"]).all())
        update_check(current_index, forces_finite, json.dumps({"forces_finite": forces_finite}, sort_keys=True))
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["state_keys", "state_shapes", "state_values"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = load_module(
            "nequix_energy_force_train_submission",
            submission / "nequix_energy_force_train.py",
        )
        train_module = load_module(
            "nequix_energy_force_train_submission_train",
            submission / "train_nequix_from_scratch.py",
        )
        candidate_config = train_module.load_config(str(submission / "nequix_train.yaml"))
        candidate_model = module.build_model(candidate_config, "cpu")
        standard_root = REFERENCE_DIR
        standard = load_module(
            "nequix_energy_force_train_standard",
            standard_root / "nequix_energy_force_train.py",
        )
        ref_config = {"paths": {"arch_config": str(standard_root / "nequix_arch_config.json")}, "training": {"seed": SEED}}
        torch.manual_seed(SEED)
        ref_model = standard.build_model(ref_config, "cpu")
        ref_state = ref_model.state_dict()
        cand_state = candidate_model.state_dict()
        same_keys = set(ref_state) == set(cand_state)
        same_shapes = same_keys and all(ref_state[k].shape == cand_state[k].shape for k in ref_state)
        same_values = same_shapes and all(torch.equal(ref_state[k].cpu(), cand_state[k].cpu()) for k in ref_state)
        update_check(current_index, same_keys, json.dumps({"same_keys": same_keys}, sort_keys=True))
        current_index += 1
        update_check(current_index, same_shapes, json.dumps({"same_shapes": same_shapes}, sort_keys=True))
        current_index += 1
        update_check(current_index, same_values, json.dumps({"same_values": same_values}, sort_keys=True))
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "deterministic_initial_state",
        "finite_train_step_record",
        "optimizer_trainable_ids",
        "trainable_parameter_count",
        "changed_parameter_count",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = load_module(
            "nequix_energy_force_train_submission",
            submission / "nequix_energy_force_train.py",
        )
        train_module = load_module(
            "nequix_energy_force_train_submission_train",
            submission / "train_nequix_from_scratch.py",
        )
        candidate_config = train_module.load_config(str(submission / "nequix_train.yaml"))
        torch.manual_seed(SEED)
        model, optimizer = train_module.build_trainer(candidate_config, "cpu")
        torch.manual_seed(SEED)
        second_model = train_module.build_model(candidate_config, "cpu")
        deterministic_init = all(
            torch.equal(model.state_dict()[k].cpu(), second_model.state_dict()[k].cpu())
            for k in model.state_dict()
        )
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        trainable_ids = {id(parameter) for _, parameter in trainable}
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        before = {name: parameter.detach().clone() for name, parameter in trainable}
        batch = module.build_train_batch(str(DATA_PATH), int(candidate_config["training"]["structure_index"]), "cpu")
        record = train_module.train_step(model, optimizer, batch, candidate_config)
        changed = [name for name, parameter in trainable if not torch.equal(before[name], parameter.detach())]
        finite = all(torch.isfinite(torch.tensor(float(value))).item() for value in record.values())
        optimizer_matches = optimizer_ids == trainable_ids
        update_check(current_index, deterministic_init, json.dumps({"deterministic_initial_state": deterministic_init}, sort_keys=True))
        current_index += 1
        update_check(current_index, finite, json.dumps({"record": record}, sort_keys=True))
        current_index += 1
        update_check(current_index, optimizer_matches, json.dumps({"optimizer_matches_trainable": optimizer_matches}, sort_keys=True))
        current_index += 1
        update_check(current_index, len(trainable) == EXPECTED_PARAM_COUNT, json.dumps({"trainable": len(trainable)}, sort_keys=True))
        current_index += 1
        update_check(current_index, len(changed) >= MIN_CHANGED_PARAMS, json.dumps({"changed": len(changed), "min_changed": MIN_CHANGED_PARAMS}, sort_keys=True))
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "training_artifact_paths",
        "checkpoint_keys",
        "checkpoint_format",
        "checkpoint_data_sha256",
        "checkpoint_completed_steps",
        "checkpoint_trainable_parameters",
        "checkpoint_optimizer_state",
        "checkpoint_history",
        "strict_checkpoint_reload",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        train_module = load_module(
            "nequix_energy_force_train_submission_train",
            submission / "train_nequix_from_scratch.py",
        )
        with tempfile.TemporaryDirectory(prefix="nequix_train_eval_") as directory:
            config_path = submission / "nequix_train.yaml"
            train_module.train(str(config_path), directory, "cpu")
            output = Path(directory)
            required = ["nequix_trained.pt", "metrics.jsonl", "train.log"]
            missing = [name for name in required if not (output / name).is_file()]
            paths_ok = not missing
            update_check(current_index, paths_ok, f"missing artifacts: {missing}" if missing else "all artifacts present")
            current_index += 1
            payload = None
            expected = {"format", "model_name", "model", "optimizer", "completed_steps", "trainable_parameter_names", "data_sha256", "config", "history"}
            if not missing:
                payload = torch.load(output / "nequix_trained.pt", map_location="cpu", weights_only=True)
            keys_ok = payload is not None and expected.issubset(payload)
            update_check(current_index, keys_ok, json.dumps({"keys": sorted(payload) if payload is not None else []}, sort_keys=True))
            current_index += 1
            format_ok = keys_ok and payload["format"] == "nequix-energy-force-train-v1"
            update_check(current_index, format_ok, json.dumps({"format": payload.get("format") if payload is not None else None}, sort_keys=True))
            current_index += 1
            data_ok = keys_ok and payload["data_sha256"] == DATA_SHA256
            update_check(current_index, data_ok, json.dumps({"data_sha256": payload.get("data_sha256") if payload is not None else None}, sort_keys=True))
            current_index += 1
            steps_ok = keys_ok and payload["completed_steps"] == 1
            update_check(current_index, steps_ok, json.dumps({"completed_steps": payload.get("completed_steps") if payload is not None else None}, sort_keys=True))
            current_index += 1
            trainable_ok = keys_ok and len(payload["trainable_parameter_names"]) == EXPECTED_PARAM_COUNT
            update_check(current_index, trainable_ok, json.dumps({"trainable_count": len(payload["trainable_parameter_names"]) if payload is not None and "trainable_parameter_names" in payload else 0}, sort_keys=True))
            current_index += 1
            optimizer_ok = keys_ok and bool(payload["optimizer"])
            update_check(current_index, optimizer_ok, json.dumps({"optimizer_nonempty": optimizer_ok}, sort_keys=True))
            current_index += 1
            history_ok = keys_ok and len(payload["history"]) == 1
            update_check(current_index, history_ok, json.dumps({"history_length": len(payload["history"]) if payload is not None and "history" in payload else 0}, sort_keys=True))
            current_index += 1
            reload_ok = False
            reload_detail = {"strict_reload": False}
            if keys_ok:
                train_config = train_module.load_config(str(submission / "nequix_train.yaml"))
                fresh = train_module.build_model(payload["config"], "cpu")
                fresh.load_state_dict(payload["model"], strict=True)
                reload_ok = True
                reload_detail = {"strict_reload": True}
            update_check(current_index, reload_ok, json.dumps(reload_detail, sort_keys=True))
            current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())

    return checks
