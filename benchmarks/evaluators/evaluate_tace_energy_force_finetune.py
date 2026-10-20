'''V2 adapter for the TACE energy-force fine-tuning evaluator.'''
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
if not (CANDIDATE_DIR / "tace_energy_force_finetune.py").is_file() and (
    CANDIDATE_DIR / "tace_energy_force_finetune" / "tace_energy_force_finetune.py"
).is_file():
    CANDIDATE_DIR = CANDIDATE_DIR / "tace_energy_force_finetune"
base_evaluator.CANDIDATE_DIR = CANDIDATE_DIR
REFERENCE_DIR = base_evaluator.REFERENCE_DIR

MODEL_PATH = os.environ.get("TACE_FINETUNE_MODEL_PATH", "/public/share/sugonhpcapp01/onestore/onemodels/tace/TACE-OAM-7M.pt")
DATA_PATH = os.environ.get("TACE_FINETUNE_DATA_PATH", "/public/share/sugonhpcapp01/onestore/onedatasets/matchem/mace/DMC/solvent_xtb_train_200.xyz")
MODEL_SHA256 = "87f6b23c40a5b29c3b0d2520bca104d73c653866e82541919db94979c7f6511c"
DATA_SHA256 = "e540bc7c026cfad063fc970406bb74c0f668007fc3c6d200801ecfd5f1ba600e"
EXPECTED_STATE_COUNT = 456
EXPECTED_STATE_NUMEL = 7105809
EXPECTED_PARAM_COUNT = 184
FRAME_INDEX = 3
FRAME_NUM_ATOMS = 24
FRAME_ENERGY = -1602.093208085042
SEED = 456
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

def _reference_path(relative):
    relative = _strip_redundant_prefix(relative, REFERENCE_DIR)
    path = (REFERENCE_DIR / relative).resolve()
    path.relative_to(REFERENCE_DIR)
    return path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def inspect_source(submission):
    prohibited = []
    safe_load = False
    strict_load = False
    for path in submission.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                prohibited.extend(alias.name for alias in node.names if alias.name.split(".")[0] in PROHIBITED_IMPORTS)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in PROHIBITED_IMPORTS:
                prohibited.append(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                if node.func.attr == "load" and isinstance(keywords.get("weights_only"), ast.Constant) and keywords["weights_only"].value is True:
                    safe_load = True
                if node.func.attr == "load_state_dict" and isinstance(keywords.get("strict"), ast.Constant) and keywords["strict"].value is True:
                    strict_load = True
    return prohibited, safe_load, strict_load

def check_dataset_pipeline(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["data:data_sha256", "data:minimum_structure_count", "data:energy_shape", "data:forces_shape", "data:positions_shape", "data:energy_value", "data:finite_forces", "data:forces_match_source_labels"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        candidate = config.get("candidate", {})
        module_path = _candidate_path(candidate.get("file", "tace_energy_force_finetune.py"))
        module = _load_module_from_path("tace_energy_force_finetune_submission", module_path)
        data_path = config.get("data_path", DATA_PATH)
        structures = module.load_training_structures(str(data_path))
        batch = module.build_finetune_batch(str(data_path), FRAME_INDEX, "cpu")
        data = batch["data"]
        energy = batch["energy"]
        forces = batch["forces"]
        shapes = {"energy": list(energy.shape), "forces": list(forces.shape), "positions": list(data["positions"].shape)}
        detail = json.dumps({"data_sha256": sha256(data_path), "shapes": shapes, "energy": float(energy.item())}, sort_keys=True)
        update_check(current_index, sha256(data_path) == DATA_SHA256, detail)
        current_index += 1
        update_check(current_index, len(structures) >= FRAME_INDEX + 1, detail)
        current_index += 1
        update_check(current_index, shapes["energy"] == [1, 1], detail)
        current_index += 1
        update_check(current_index, shapes["forces"] == [FRAME_NUM_ATOMS, 3], detail)
        current_index += 1
        update_check(current_index, shapes["positions"] == [FRAME_NUM_ATOMS, 3], detail)
        current_index += 1
        update_check(current_index, abs(float(energy.item()) - FRAME_ENERGY) <= 1e-3, detail)
        current_index += 1
        update_check(current_index, bool(torch.isfinite(forces).all()), detail)
        current_index += 1
        import ase.io
        source_atoms = ase.io.read(str(data_path), index=FRAME_INDEX)
        source_forces = torch.as_tensor(source_atoms.arrays["forces_xtb"],
                                        dtype=forces.dtype, device=forces.device)
        update_check(current_index, forces.shape == source_forces.shape and
                     bool(torch.allclose(forces, source_forces, atol=1e-6, rtol=1e-6)), detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["train_script:required_training_keys", "train_script:optimizer", "train_script:learning_rate", "train_script:max_steps", "train_script:structure_index", "train_script:seed"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        module_path = _candidate_path(config.get("interface_file", "train_tace_finetune.py"))
        config_path = _candidate_path(config.get("config_name", "tace_finetune.yaml"))
        train_module = _load_module_from_path("tace_energy_force_finetune_submission_train", module_path)
        required_keys = set(config.get("required_config_keys", ("optimizer", "lr", "weight_decay", "energy_weight", "force_weight", "gradient_clip_norm", "max_steps", "structure_index", "seed")))
        loaded = train_module.load_config(str(config_path))
        training = loaded.get("training", {})
        missing_keys = sorted(required_keys - set(training))
        values = [not missing_keys, not missing_keys and training["optimizer"] == "AdamW", not missing_keys and float(training["lr"]) == 1e-3, not missing_keys and int(training["max_steps"]) == 1, not missing_keys and int(training["structure_index"]) == FRAME_INDEX, not missing_keys and int(training["seed"]) == SEED]
        messages = [json.dumps({"missing_config_keys": missing_keys}), json.dumps({"optimizer": training.get("optimizer")}), json.dumps({"lr": training.get("lr")}), json.dumps({"max_steps": training.get("max_steps")}), json.dumps({"structure_index": training.get("structure_index")}), json.dumps({"seed": training.get("seed")})]
        for passed, message in zip(values, messages):
            update_check(current_index, passed, message)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_checkpoint_file(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Validate the fixed external checkpoint before importing candidate code."""
    try:
        digest = sha256(MODEL_PATH)
        return [("checkpoint:sha256", digest == MODEL_SHA256, digest)]
    except Exception:
        return [("checkpoint:file", False, traceback.format_exc())]


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Verify the fixed input checkpoint and the candidate's strict restoration."""
    names = (
        "checkpoint:weights_only", "checkpoint:strict_load",
        "checkpoint:state_count", "checkpoint:state_numel",
        "checkpoint:state_keys", "checkpoint:state_values",
    )
    try:
        module = _load_module_from_path("tace_finetune_checkpoint", _candidate_path("tace_energy_force_finetune.py"))
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        _, safe_load, strict_load = inspect_source(CANDIDATE_DIR)
        loaded = model.state_dict()
        same_keys = set(loaded) == set(state)
        same_values = same_keys and all(torch.equal(loaded[k].cpu(), state[k].cpu()) for k in loaded)
        values = (
            safe_load, strict_load,
            len(state) == EXPECTED_STATE_COUNT,
            sum(v.numel() for v in state.values() if torch.is_tensor(v)) == EXPECTED_STATE_NUMEL,
            same_keys, same_values,
        )
        detail = json.dumps({"state_count": len(state),
                             "same_keys": same_keys, "same_values": same_values}, sort_keys=True)
        return [(name, bool(ok), detail) for name, ok in zip(names, values)]
    except Exception:
        return [(names[0], False, traceback.format_exc())]


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the Prompt's prohibition on external model imports."""
    if config.get("task_rule") != "local_source":
        return [("model:task_rule", False, "unknown task_rule")]
    prohibited, _, _ = inspect_source(CANDIDATE_DIR)
    return [("model:forbidden_imports_absent", not prohibited,
             json.dumps({"prohibited_imports": prohibited}, sort_keys=True))]


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["model_structure:model_class", "model_structure:readout_class", "model_structure:module_order", "model_structure:state_count", "model_structure:state_numel", "model_structure:parameter_count", "model_structure:state_keys", "model_structure:state_values"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        module_path = _candidate_path(config.get("candidate", {}).get("file", "tace_energy_force_finetune.py"))
        module = _load_module_from_path("tace_energy_force_finetune_submission_structure", module_path)
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        loaded = model.state_dict()
        same_keys = set(loaded) == set(state)
        same_values = same_keys and all(torch.equal(loaded[k].cpu(), state[k].cpu()) for k in loaded)
        readout = model.readout_fn
        order = list(readout._modules)
        expected_order = ["representation", "energy_readouts", "atomic_energy_layer", "scale_shift", "zbl"]
        detail = json.dumps({"wrapper": type(model).__name__, "readout": type(readout).__name__, "module_order": order, "state_count": len(state), "param_count": len(list(model.parameters())), "same_keys": same_keys, "same_values": same_values}, sort_keys=True)
        values = [type(model).__name__ == "TensorModel", type(readout).__name__ == "e3nnTACE", order == expected_order, len(state) == EXPECTED_STATE_COUNT, sum(v.numel() for v in state.values() if torch.is_tensor(v)) == EXPECTED_STATE_NUMEL, len(list(model.parameters())) == EXPECTED_PARAM_COUNT, same_keys, same_values]
        for passed in values:
            update_check(current_index, passed, detail)
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_executable:energy_shape", "forward_executable:forces_shape", "forward_executable:finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        module_path = _candidate_path(config.get("candidate", {}).get("file", "tace_energy_force_finetune.py"))
        module = _load_module_from_path("tace_energy_force_finetune_submission_forward", module_path)
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        model.train()
        batch = module.build_finetune_batch(str(DATA_PATH), FRAME_INDEX, "cpu")
        out = module.predict_energy_forces(model, batch)
        shapes = {"energy": list(out["energy"].shape), "forces": list(out["forces"].shape)}
        finite = bool(torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all())
        detail = json.dumps({"shapes": shapes, "finite": finite}, sort_keys=True)
        update_check(current_index, shapes["energy"] == [1], detail)
        current_index += 1
        update_check(current_index, shapes["forces"] == [FRAME_NUM_ATOMS, 3], detail)
        current_index += 1
        update_check(current_index, finite, detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_equivalence:energy", "forward_equivalence:forces"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        candidate_path = _candidate_path(config.get("candidate", {}).get("file", "tace_energy_force_finetune.py"))
        reference_path = _reference_path("tace_energy_force_finetune.py")
        # Both implementations use the same local package name. Isolate the
        # reference import so it cannot silently reuse candidate modules.
        candidate_packages = {key: value for key, value in sys.modules.items()
                              if key == "portable_tace" or key.startswith("portable_tace.")}
        for key in candidate_packages:
            sys.modules.pop(key, None)
        sys.path.insert(0, str(REFERENCE_DIR))
        try:
            standard = _load_module_from_path("tace_energy_force_finetune_standard", reference_path)
            torch.manual_seed(SEED)
            ref_model, _, _ = standard.load_model(str(MODEL_PATH), "cpu")
            ref_model.eval()
            ref_batch = standard.build_finetune_batch(str(DATA_PATH), FRAME_INDEX, "cpu")
            torch.manual_seed(SEED)
            reference = standard.predict_energy_forces(ref_model, ref_batch)
        finally:
            for key in list(sys.modules):
                if key == "portable_tace" or key.startswith("portable_tace."):
                    sys.modules.pop(key, None)
            sys.modules.update(candidate_packages)
            sys.path.remove(str(REFERENCE_DIR))
        module = _load_module_from_path("tace_energy_force_finetune_submission_equivalence", candidate_path)
        torch.manual_seed(SEED)
        cand_model, _, cand_state = module.load_model(str(MODEL_PATH), "cpu")
        cand_model.eval()
        cand_batch = module.build_finetune_batch(str(DATA_PATH), FRAME_INDEX, "cpu")
        prediction = module.predict_energy_forces(cand_model, cand_batch)
        errors = {name: float((prediction[name] - reference[name]).abs().max()) for name in ("energy", "forces")}
        detail = json.dumps({"max_abs_error": errors, "atol": 1e-5}, sort_keys=True)
        update_check(current_index, errors["energy"] <= 1e-5, detail)
        current_index += 1
        update_check(current_index, errors["forces"] <= 1e-5, detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_train_loss(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the configured energy/force weights and both MSE terms."""
    try:
        tree = ast.parse(_candidate_path("train_tace_finetune.py").read_text(encoding="utf-8"))
        train_step = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == "train_step")
        mse_calls = [node for node in ast.walk(train_step) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "mse_loss"]
        train_module = _load_module_from_path("tace_finetune_loss", _candidate_path("train_tace_finetune.py"))
        training = train_module.load_config(str(_candidate_path("tace_finetune.yaml")))["training"]
        return [
            ("loss:energy_and_force_mse", len(mse_calls) >= 2, f"mse_loss calls={len(mse_calls)}"),
            ("loss:energy_weight", float(training["energy_weight"]) == 1.0, str(training["energy_weight"])),
            ("loss:force_weight", float(training["force_weight"]) == 5.0, str(training["force_weight"])),
        ]
    except Exception:
        return [("loss:contract", False, traceback.format_exc())]


def check_smoke_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["training:finite_trainable_parameters", "training:optimizer_parameter_ids", "training:optimizer_is_adamw", "training:optimizer_hyperparameters", "training:trainable_parameter_count", "training:changed_parameter_count"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        train_path = _candidate_path(config.get("candidate_train_module", "train_tace_finetune.py"))
        model_path = _candidate_path(config.get("candidate_model_module", "tace_energy_force_finetune.py"))
        config_path = _candidate_path(config.get("config", "tace_finetune.yaml"))
        train_module = _load_module_from_path("tace_energy_force_finetune_submission_train_runtime", train_path)
        module = _load_module_from_path("tace_energy_force_finetune_submission_runtime", model_path)
        loaded = train_module.load_config(str(config_path))
        model, optimizer = train_module.build_trainer(loaded, "cpu")
        named = list(train_module.trainable_named_parameters(model))
        trainable_ids = {id(parameter) for _, parameter in named}
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        before = {name: parameter.detach().clone() for name, parameter in named}
        batch = module.build_finetune_batch(str(DATA_PATH), int(loaded["training"]["structure_index"]), "cpu")
        record = train_module.train_step(model, optimizer, batch, loaded)
        changed = [name for name, parameter in named if not torch.equal(before[name], parameter.detach())]
        finite = all(bool(torch.isfinite(parameter).all()) for _, parameter in named)
        detail = json.dumps({"record": record, "trainable_count": len(named), "changed": len(changed), "optimizer_matches_trainable": optimizer_ids == trainable_ids}, sort_keys=True, default=str)
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
        update_check(current_index, len(named) == EXPECTED_PARAM_COUNT, detail)
        current_index += 1
        update_check(current_index, len(changed) == EXPECTED_PARAM_COUNT, detail)
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["artifacts:required_files", "artifacts:required_keys", "artifacts:format", "artifacts:source_checkpoint_sha256", "artifacts:completed_steps", "artifacts:trainable_parameter_count", "artifacts:optimizer", "artifacts:history", "artifacts:strict_reload"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        train_path = _candidate_path(config.get("candidate_train_module", "train_tace_finetune.py"))
        config_path = _candidate_path(config.get("config_name", "tace_finetune.yaml"))
        train_module = _load_module_from_path("tace_energy_force_finetune_submission_train_artifacts", train_path)
        with tempfile.TemporaryDirectory(prefix="tace_finetune_eval_") as directory:
            train_module.train(str(config_path), directory, "cpu")
            output = Path(directory)
            required = ["tace_finetuned.pt", "metrics.jsonl", "train.log"]
            missing = [name for name in required if not (output / name).is_file()]
            payload = torch.load(output / "tace_finetuned.pt", map_location="cpu", weights_only=True) if not missing else {}
            expected = {"format", "model_name", "model", "optimizer", "completed_steps", "trainable_parameter_names", "source_checkpoint_sha256", "config", "history"}
            strict_reload = False
            try:
                if "model" in payload and "config" in payload:
                    fresh = train_module.build_model(payload["config"], "cpu")
                    fresh.load_state_dict(payload["model"], strict=True)
                    strict_reload = True
            except Exception:
                strict_reload = False
            values = [not missing, expected.issubset(payload), payload.get("format") == "tace-energy-force-finetune-v1", payload.get("source_checkpoint_sha256") == MODEL_SHA256, payload.get("completed_steps") == 1, len(payload.get("trainable_parameter_names", [])) == EXPECTED_PARAM_COUNT, bool(payload.get("optimizer")), len(payload.get("history", [])) == 1, strict_reload]
            detail = json.dumps({"missing": missing, "keys": sorted(payload), "completed_steps": payload.get("completed_steps"), "trainable_count": len(payload.get("trainable_parameter_names", []))}, sort_keys=True)
            for passed in values:
                update_check(current_index, passed, detail)
                current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks
