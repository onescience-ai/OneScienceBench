"""V2 evaluator adapter for the TACE energy-force training task."""

from __future__ import annotations

import ast
import hashlib
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
import base_evaluator
from base_evaluator import _load_module_from_path, check_files, check_interface, check_parameter_update


CANDIDATE_DIR = base_evaluator.CANDIDATE_DIR
if not (CANDIDATE_DIR / "tace_energy_force_train.py").is_file() and (
    CANDIDATE_DIR / "tace_energy_force_train" / "tace_energy_force_train.py"
).is_file():
    CANDIDATE_DIR = CANDIDATE_DIR / "tace_energy_force_train"
base_evaluator.CANDIDATE_DIR = CANDIDATE_DIR

DATA_PATH = os.environ.get(
    "TACE_TRAIN_DATA_PATH",
    "/public/share/sugonhpcapp01/onestore/onedatasets/matchem/mace/DMC/solvent_xtb_train_200.xyz",
)
DATA_SHA256 = "e540bc7c026cfad063fc970406bb74c0f668007fc3c6d200801ecfd5f1ba600e"
EXPECTED_STATE_COUNT = 456
EXPECTED_STATE_NUMEL = 7105809
EXPECTED_PARAM_COUNT = 184
FRAME_INDEX = 3
FRAME_NUM_ATOMS = 24
FRAME_ENERGY = -1602.093208085042
SEED = 456
MIN_CHANGED_PARAMS = 150

PROHIBITED_IMPORTS = {"onescience", "nequip", "mace", "mattersim"}

def _strip_redundant_prefix(relative, root):
    relative = Path(relative)
    parts = relative.parts
    if len(parts) > 1 and parts[0] == root.name:
        relative = Path(*parts[1:])
    return relative

def _candidate_path(relative):
    relative = _strip_redundant_prefix(relative, CANDIDATE_DIR)
    path = (CANDIDATE_DIR / relative).resolve()
    path.relative_to(CANDIDATE_DIR)
    return path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def inspect_source(submission):
    prohibited = []
    for path in submission.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                prohibited.extend(alias.name for alias in node.names if alias.name.split(".")[0] in PROHIBITED_IMPORTS)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in PROHIBITED_IMPORTS:
                prohibited.append(node.module)
    return prohibited

def check_dataset_pipeline(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "data_sha256", "minimum_structure_count", "energy_shape",
        "forces_shape", "positions_shape", "expected_energy", "finite_forces",
        "forces_match_source_labels",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        candidate = config.get("candidate", config)
        data_path_env = candidate.get("data_path_env", "TACE_TRAIN_DATA_PATH")
        data_path = os.environ.get(data_path_env, candidate.get("default_data_path", DATA_PATH))
        frame_index = int(candidate.get("frame_index", FRAME_INDEX))
        device = candidate.get("device", "cpu")
        expected_sha = config.get("data_sha256", DATA_SHA256)
        structures_min_count = int(config.get("structures_min_count", FRAME_INDEX + 1))
        expected_energy_shape = list(config.get("energy_shape", [1, 1]))
        expected_forces_shape = list(config.get("forces_shape", [FRAME_NUM_ATOMS, 3]))
        expected_positions_shape = list(config.get("positions_shape", [FRAME_NUM_ATOMS, 3]))
        expected_energy = float(config.get("expected_energy", FRAME_ENERGY))
        energy_atol = float(config.get("energy_atol", 1e-3))
        module = _load_module_from_path("tace_energy_force_train_submission", _candidate_path("tace_energy_force_train.py"))
        structures = module.load_training_structures(str(data_path))
        batch = module.build_train_batch(str(data_path), frame_index, device)
        data = batch["data"]
        energy = batch["energy"]
        forces = batch["forces"]
        shapes = {"energy": list(energy.shape), "forces": list(forces.shape), "positions": list(data["positions"].shape)}
        detail = json.dumps({"data_sha256": sha256(data_path), "shapes": shapes, "energy": float(energy.item())}, sort_keys=True)
        update_check(current_index, sha256(data_path) == expected_sha, detail)
        current_index += 1
        update_check(current_index, len(structures) >= structures_min_count, detail)
        current_index += 1
        update_check(current_index, shapes["energy"] == expected_energy_shape, detail)
        current_index += 1
        update_check(current_index, shapes["forces"] == expected_forces_shape, detail)
        current_index += 1
        update_check(current_index, shapes["positions"] == expected_positions_shape, detail)
        current_index += 1
        update_check(current_index, abs(float(energy.item()) - expected_energy) <= energy_atol, detail)
        current_index += 1
        update_check(current_index, bool(torch.isfinite(forces).all()), detail)
        current_index += 1
        import ase.io
        source_atoms = ase.io.read(str(data_path), index=frame_index)
        source_forces = torch.as_tensor(source_atoms.arrays["forces_xtb"],
                                        dtype=forces.dtype, device=forces.device)
        update_check(current_index, forces.shape == source_forces.shape and
                     bool(torch.allclose(forces, source_forces, atol=1e-6, rtol=1e-6)), detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "required_training_config_keys",
        "optimizer", "learning_rate", "max_steps", "structure_index", "seed",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        config_name = config.get("config_name", "tace_train.yaml")
        interface_file = config.get("interface_file", "train_tace_from_scratch.py")
        train_module = _load_module_from_path("tace_energy_force_train_submission_train", _candidate_path(interface_file))
        train_config = train_module.load_config(str(_candidate_path(config_name)))
        training = train_config.get("training", {})
        required_keys = set(config.get("required_config_keys", {"optimizer", "lr", "weight_decay", "energy_weight", "force_weight", "gradient_clip_norm", "max_steps", "structure_index", "seed"}))
        missing_keys = sorted(required_keys - set(training))
        expected_values = config.get("expected_values", {"training.optimizer": "AdamW", "training.lr": 1e-3, "training.max_steps": 1, "training.structure_index": FRAME_INDEX, "training.seed": SEED})
        detail = json.dumps({"missing_config_keys": missing_keys, "optimizer": training.get("optimizer"), "lr": training.get("lr"), "max_steps": training.get("max_steps")}, sort_keys=True)
        update_check(current_index, not missing_keys, detail)
        current_index += 1
        update_check(current_index, training.get("optimizer") == expected_values.get("training.optimizer", "AdamW"), detail)
        current_index += 1
        update_check(current_index, float(training.get("lr")) == float(expected_values.get("training.lr", 1e-3)), detail)
        current_index += 1
        update_check(current_index, int(training.get("max_steps")) == int(expected_values.get("training.max_steps", 1)), detail)
        current_index += 1
        update_check(current_index, int(training.get("structure_index")) == int(expected_values.get("training.structure_index", FRAME_INDEX)), detail)
        current_index += 1
        update_check(current_index, int(training.get("seed")) == int(expected_values.get("training.seed", SEED)), detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "tensor_model_type", "tace_readout_type", "readout_module_order",
        "state_entry_count", "state_numel", "parameter_count", "atomic_numbers", "cutoff",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        module = _load_module_from_path("tace_energy_force_train_submission", _candidate_path("tace_energy_force_train.py"))
        model_config = {"paths": {"arch_config": str(_candidate_path("tace_arch_config.json"))},
                        "training": {"seed": int(config.get("seed", SEED))}}
        model = module.build_model(model_config, config.get("device", "cpu"))
        readout = model.readout_fn
        order = list(readout._modules)
        expected_order = list(config.get("readout_module_order", ["representation", "energy_readouts", "atomic_energy_layer", "scale_shift", "zbl"]))
        state = model.state_dict()
        n_params = len(list(model.parameters()))
        atomic_numbers = config.get("atomic_numbers", [*range(1, 84), 89, 90, 91, 92, 93, 94])
        expected_state_count = int(config.get("state_count", EXPECTED_STATE_COUNT))
        expected_state_numel = int(config.get("state_numel", EXPECTED_STATE_NUMEL))
        expected_param_count = int(config.get("parameter_count", EXPECTED_PARAM_COUNT))
        expected_cutoff = float(config.get("cutoff", 6.0))
        cutoff_atol = float(config.get("cutoff_atol", 1e-9))
        detail = json.dumps({"wrapper": type(model).__name__, "readout": type(readout).__name__, "module_order": order, "state_count": len(state), "param_count": n_params, "num_atomic_numbers": len(model.get_atomic_numbers()), "cutoff": model.get_cutoff()}, sort_keys=True)
        update_check(current_index, type(model).__name__ == config.get("model_class", "TensorModel"), detail)
        current_index += 1
        update_check(current_index, type(readout).__name__ == config.get("readout_class", "e3nnTACE"), detail)
        current_index += 1
        update_check(current_index, order == expected_order, detail)
        current_index += 1
        update_check(current_index, len(state) == expected_state_count, detail)
        current_index += 1
        update_check(current_index, sum(v.numel() for v in state.values() if torch.is_tensor(v)) == expected_state_numel, detail)
        current_index += 1
        update_check(current_index, n_params == expected_param_count, detail)
        current_index += 1
        update_check(current_index, model.get_atomic_numbers() == atomic_numbers, detail)
        current_index += 1
        update_check(current_index, abs(model.get_cutoff() - expected_cutoff) < cutoff_atol, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = ["forward_output_shapes", "finite_forward_outputs"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        module = _load_module_from_path("tace_energy_force_train_submission", _candidate_path("tace_energy_force_train.py"))
        model_config = {"paths": {"arch_config": str(_candidate_path("tace_arch_config.json"))},
                        "training": {"seed": int(config.get("seed", SEED))}}
        device = config.get("device", "cpu")
        frame_index = int(config.get("frame_index", FRAME_INDEX))
        data_path = os.environ.get(config.get("data_path_env", "TACE_TRAIN_DATA_PATH"), config.get("default_data_path", DATA_PATH))
        model = module.build_model(model_config, device)
        model.train()
        batch = module.build_train_batch(str(data_path), frame_index, device)
        out = module.predict_energy_forces(model, batch)
        shapes = {"energy": list(out["energy"].shape), "forces": list(out["forces"].shape)}
        expected_shapes = config.get("expected", {"energy_shape": [1], "forces_shape": [FRAME_NUM_ATOMS, 3]})
        expected_shapes = {"energy": list(expected_shapes.get("energy_shape", [1])), "forces": list(expected_shapes.get("forces_shape", [FRAME_NUM_ATOMS, 3]))}
        finite = bool(torch.isfinite(out["energy"]).all()) and bool(torch.isfinite(out["forces"]).all())
        detail = json.dumps({"shapes": shapes, "finite": bool(finite)}, sort_keys=True)
        update_check(current_index, shapes == expected_shapes, detail)
        current_index += 1
        update_check(current_index, finite, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check local-source independence or same-seed initialization."""
    task_rule = config.get("task_rule", "deterministic_initialization")
    if task_rule == "local_source":
        prohibited = inspect_source(CANDIDATE_DIR)
        return [("model:forbidden_imports_absent", not prohibited,
                 json.dumps({"prohibited_imports": prohibited}, sort_keys=True))]
    if task_rule != "deterministic_initialization":
        return [("model:task_rule", False, f"unknown task_rule: {task_rule}")]
    CHECKPOINT_NAMES = ["state_dict_keys_equal", "state_dict_shapes_equal", "state_dict_values_equal"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        module = _load_module_from_path("tace_energy_force_train_submission", _candidate_path("tace_energy_force_train.py"))
        model_config = {"paths": {"arch_config": str(_candidate_path("tace_arch_config.json"))},
                        "training": {"seed": int(config.get("seed", SEED))}}
        device = config.get("device", "cpu")
        seed = int(config.get("seed", SEED))
        torch.manual_seed(seed)
        first = module.build_model(model_config, device).state_dict()
        torch.manual_seed(seed)
        second = module.build_model(model_config, device).state_dict()
        same_keys = set(first) == set(second)
        same_shapes = same_keys and all(first[k].shape == second[k].shape for k in first)
        same_values = same_shapes and all(torch.equal(first[k].cpu(), second[k].cpu()) for k in first)
        detail = json.dumps({"same_keys": same_keys, "same_shapes": same_shapes, "same_values": same_values}, sort_keys=True)
        update_check(current_index, same_keys, detail)
        current_index += 1
        update_check(current_index, same_shapes, detail)
        current_index += 1
        update_check(current_index, same_values, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_train_loss(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the configured energy/force weights and both MSE terms."""
    try:
        tree = ast.parse(_candidate_path("train_tace_from_scratch.py").read_text(encoding="utf-8"))
        train_step = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == "train_step")
        mse_calls = [node for node in ast.walk(train_step) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "mse_loss"]
        train_module = _load_module_from_path("tace_train_loss", _candidate_path("train_tace_from_scratch.py"))
        training = train_module.load_config(str(_candidate_path("tace_train.yaml")))["training"]
        return [
            ("loss:energy_and_force_mse", len(mse_calls) >= 2, f"mse_loss calls={len(mse_calls)}"),
            ("loss:energy_weight", float(training["energy_weight"]) == 1.0, str(training["energy_weight"])),
            ("loss:force_weight", float(training["force_weight"]) == 5.0, str(training["force_weight"])),
        ]
    except Exception:
        return [("loss:contract", False, traceback.format_exc())]


def check_smoke_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "deterministic_initial_state", "finite_trainable_parameters",
        "optimizer_matches_trainable_parameters", "optimizer_is_adamw",
        "optimizer_hyperparameters", "trainable_parameter_count",
        "minimum_changed_parameters",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        module = _load_module_from_path("tace_energy_force_train_submission", _candidate_path("tace_energy_force_train.py"))
        train_module = _load_module_from_path("tace_energy_force_train_submission_train", _candidate_path("train_tace_from_scratch.py"))
        config_name = config.get("config_name", "tace_train.yaml")
        train_config = train_module.load_config(str(_candidate_path(config_name)))
        device = config.get("device", "cpu")
        seed = int(config.get("seed", SEED))
        torch.manual_seed(seed)
        model, optimizer = train_module.build_trainer(train_config, device)
        torch.manual_seed(seed)
        second_model = train_module.build_model(train_config, device)
        deterministic_init = all(torch.equal(model.state_dict()[k].cpu(), second_model.state_dict()[k].cpu()) for k in model.state_dict())
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        trainable_ids = {id(parameter) for _, parameter in trainable}
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        before = {name: parameter.detach().clone() for name, parameter in trainable}
        data_path = os.environ.get(config.get("data_path_env", "TACE_TRAIN_DATA_PATH"), config.get("default_data_path", DATA_PATH))
        structure_index = int(train_config["training"]["structure_index"])
        batch = module.build_train_batch(str(data_path), structure_index, device)
        record = train_module.train_step(model, optimizer, batch, train_config)
        changed = [name for name, parameter in trainable if not torch.equal(before[name], parameter.detach())]
        finite = all(bool(torch.isfinite(parameter).all()) for _, parameter in trainable)
        expected_param_count = int(config.get("expected_trainable_parameter_count", EXPECTED_PARAM_COUNT))
        minimum_changed = int(config.get("minimum_changed_parameters", MIN_CHANGED_PARAMS))
        detail = json.dumps({"record": record, "deterministic_initial_state": deterministic_init, "trainable": len(trainable), "changed": len(changed), "min_changed": minimum_changed, "optimizer_matches_trainable": optimizer_ids == trainable_ids}, sort_keys=True, default=str)
        update_check(current_index, deterministic_init, detail)
        current_index += 1
        update_check(current_index, finite, detail)
        current_index += 1
        update_check(current_index, optimizer_ids == trainable_ids, detail)
        current_index += 1
        update_check(current_index, isinstance(optimizer, torch.optim.AdamW),
                     type(optimizer).__name__)
        current_index += 1
        hyperparameters_ok = all(
            abs(float(group[key]) - expected) <= 1e-12
            for group in optimizer.param_groups
            for key, expected in (("lr", 1e-3), ("weight_decay", 1e-8))
        )
        update_check(current_index, hyperparameters_ok,
                     json.dumps([{"lr": group["lr"], "weight_decay": group["weight_decay"]}
                                 for group in optimizer.param_groups]))
        current_index += 1
        update_check(current_index, len(trainable) == expected_param_count, detail)
        current_index += 1
        update_check(current_index, len(changed) >= minimum_changed, detail)
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """
    Checkpoint implementation.
    """
    CHECKPOINT_NAMES = [
        "training_output_files", "checkpoint_keys", "checkpoint_format",
        "checkpoint_data_sha256", "checkpoint_completed_steps",
        "checkpoint_trainable_parameter_count", "checkpoint_optimizer_nonempty",
        "checkpoint_history_length", "strict_checkpoint_reload",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        train_module = _load_module_from_path("tace_energy_force_train_submission_train", _candidate_path("train_tace_from_scratch.py"))
        config_name = config.get("config_name", "tace_train.yaml")
        config_path = _candidate_path(config_name)
        device = config.get("device", "cpu")
        output_files = list(config.get("output_files", ["tace_trained.pt", "metrics.jsonl", "train.log"]))
        with tempfile.TemporaryDirectory(prefix=config.get("temporary_directory_prefix", "tace_train_eval_")) as directory:
            train_module.train(str(config_path), directory, device)
            output = Path(directory)
            missing = [name for name in output_files if not (output / name).is_file()]
            detail = f"missing artifacts: {missing}" if missing else json.dumps({"output_files": output_files}, sort_keys=True)
            update_check(current_index, not missing, detail)
            current_index += 1
            if not missing:
                checkpoint_name = config.get("checkpoint_name", "tace_trained.pt")
                payload = torch.load(output / checkpoint_name, map_location=config.get("map_location", "cpu"), weights_only=True)
                expected = set(config.get("required_checkpoint_keys", ["format", "model_name", "model", "optimizer", "completed_steps", "trainable_parameter_names", "data_sha256", "config", "history"]))
                data_sha = config.get("data_sha256", DATA_SHA256)
                expected_format = config.get("format", "tace-energy-force-train-v1")
                expected_steps = int(config.get("completed_steps", 1))
                expected_trainable = int(config.get("trainable_parameter_count", EXPECTED_PARAM_COUNT))
                expected_history = int(config.get("history_length", 1))
                detail = json.dumps({"keys": sorted(payload), "completed_steps": payload.get("completed_steps"), "trainable_count": len(payload.get("trainable_parameter_names", [])), "strict_reload": True}, sort_keys=True)
                update_check(current_index, expected.issubset(payload), detail)
                current_index += 1
                update_check(current_index, payload["format"] == expected_format, detail)
                current_index += 1
                update_check(current_index, payload["data_sha256"] == data_sha, detail)
                current_index += 1
                update_check(current_index, payload["completed_steps"] == expected_steps, detail)
                current_index += 1
                update_check(current_index, len(payload["trainable_parameter_names"]) == expected_trainable, detail)
                current_index += 1
                update_check(current_index, bool(payload["optimizer"]), detail)
                current_index += 1
                update_check(current_index, len(payload["history"]) == expected_history, detail)
                current_index += 1
                fresh = train_module.build_model(payload["config"], device)
                fresh.load_state_dict(payload["model"], strict=True)
                update_check(current_index, True, detail)
                current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
