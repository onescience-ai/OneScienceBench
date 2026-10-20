"""V2 evaluator adapter for fixed-checkpoint NequIP fine-tuning."""

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

CHECKPOINT = Path("/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/outputs/nequip_smoke_8dcu_20260817_171502/checkpoints/best.ckpt")
DATA = Path("/public/home/yuxiaodong/onescience-re/examples/matchem/nequip/demo/reference_data/smoke.xyz")
CHECKPOINT_SHA256 = "3ab4d7d891a36ed6346e661003fc30197d4b615171ccaec92cdd6b7bb1b4049b"
DATA_SHA256 = "6671f9b28d37484379f51f1deb23a6f4b165ed628467ee6bffa0648b6d50a5b5"
SEED = 456
PORTABLE_FILES = (
    "portable_nequip/__init__.py", "portable_nequip/internal.py", "portable_nequip/model.py",
    "portable_nequip/energy_modules.py", "portable_nequip/runtime.py", "portable_nequip/NOTICE.md",
    "portable_nequip/data/__init__.py", "portable_nequip/data/AtomicDataDict.py",
    "portable_nequip/data/_keys.py", "portable_nequip/data/_key_registry.py",
    "portable_nequip/nn/__init__.py", "portable_nequip/nn/_ghost_exchange_base.py",
    "portable_nequip/nn/_graph_mixin.py", "portable_nequip/nn/_tp_scatter_base.py",
    "portable_nequip/nn/atomwise.py", "portable_nequip/nn/convnetlayer.py",
    "portable_nequip/nn/grad_output.py", "portable_nequip/nn/graph_model.py",
    "portable_nequip/nn/interaction_block.py", "portable_nequip/nn/misc.py",
    "portable_nequip/nn/mlp.py", "portable_nequip/nn/model_modifier_utils.py",
    "portable_nequip/nn/nonlinearities.py", "portable_nequip/nn/norm.py",
    "portable_nequip/nn/utils.py", "portable_nequip/nn/embedding/__init__.py",
    "portable_nequip/nn/embedding/_edge.py", "portable_nequip/nn/embedding/cutoffs.py",
    "portable_nequip/nn/embedding/node.py", "portable_nequip/nn/embedding/node_tensor.py",
    "portable_nequip/nn/embedding/utils.py",
)
REQUIRED_FILES = ("nequip_energy_force_finetune.py", "train_nequip_finetune.py", "nequip_finetune.yaml", *PORTABLE_FILES)
PROHIBITED_IMPORTS = {"onescience", "nequip", "mace", "mattersim"}

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def purge_portable():
    for name in list(sys.modules):
        if name == "portable_nequip" or name.startswith("portable_nequip."):
            sys.modules.pop(name, None)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _readable_file(path: Path) -> bool:
    """Return true only when the evaluator can actually open the asset."""
    try:
        with Path(path).open("rb") as handle:
            handle.read(1)
        return True
    except (OSError, PermissionError):
        return False

def _first_readable(paths):
    for path in paths:
        if _readable_file(path):
            return Path(path).resolve()
    return None

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

def structure_details(model):
    wrapper = getattr(model, "model", None)
    network = getattr(wrapper, "func", None)
    order = list(network._modules) if network is not None else []
    expected = ["type_embed", "spharm", "edge_norm", "bessel_encode", "factor", "layer0_convnet", "layer1_convnet", "per_atom_energy_readout", "per_type_energy_scale_shift", "total_energy_sum"]
    layers = [getattr(network, f"layer{index}_convnet", None) for index in range(2)] if network is not None else []
    shapes = {}
    if network is not None:
        shapes = {"type_embedding": list(network.type_embed.embed_module.weight.shape), "bessel": list(network.bessel_encode.bessel_weights.shape), "layer0_radial_in": list(network.layer0_convnet.conv.edge_mlp.mlp[0].weight.shape), "layer0_radial_out": list(network.layer0_convnet.conv.edge_mlp.mlp[2].weight.shape), "layer1_radial_in": list(network.layer1_convnet.conv.edge_mlp.mlp[0].weight.shape), "layer1_radial_out": list(network.layer1_convnet.conv.edge_mlp.mlp[2].weight.shape), "readout": list(network.per_atom_energy_readout.mlp_module.mlp[0].weight.shape)}
    ok = all((type(model).__name__ == "GraphModel", type(wrapper).__name__ == "ForceStressOutput", type(network).__name__ == "SequentialGraphNetwork", list(model._modules) == ["model"], list(wrapper._modules) == ["func"], order == expected, len(layers) == 2 and all(type(layer).__name__ == "ConvNetLayer" for layer in layers), getattr(layers[0].conv, "sc", None) is None, getattr(layers[1].conv, "sc", None) is not None, shapes == {"type_embedding": [1, 8], "bessel": [1, 4], "layer0_radial_in": [4, 16], "layer0_radial_out": [16, 16], "layer1_radial_in": [4, 16], "layer1_radial_out": [16, 16], "readout": [8, 1]}))
    return ok, {"classes": [type(model).__name__, type(wrapper).__name__, type(network).__name__], "module_order": order, "layer_count": len(layers), "shapes": shapes}

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["required_files_present", "prohibited_imports_absent"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        required_files = tuple(config.get("required_files", REQUIRED_FILES))
        missing = [name for name in required_files if not (CANDIDATE_DIR / name).is_file()]
        update_check(current_index, not missing, json.dumps({"missing": missing}, sort_keys=True))
        current_index += 1
        prohibited = inspect_source(CANDIDATE_DIR)[0] if not missing else []
        update_check(current_index, not prohibited, json.dumps({"missing": missing, "prohibited_active_imports": prohibited}, sort_keys=True))
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["data_sha256", "structure_count", "atoms_per_structure", "positions_shape", "edge_index_shape", "cell_shift_shape", "cell_shape", "energy_shape", "forces_shape", "first_energy", "first_force"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        module = load_module("nequip_energy_force_finetune_submission", CANDIDATE_DIR / "nequip_energy_force_finetune.py")
        configured_data = Path(config.get("data_path") or DATA).expanduser()
        data_path = None
        for candidate_data in (
            configured_data,
            CANDIDATE_DIR / configured_data.name,
            REFERENCE_DIR / configured_data.name,
        ):
            try:
                if candidate_data.is_file() and os.access(candidate_data, os.R_OK):
                    data_path = candidate_data
                    break
            except OSError:
                continue
        expected = config.get("expected", {})
        if data_path is None:
            missing_apis = [
                name
                for name in ("load_training_structures", "build_finetune_batch")
                if not callable(getattr(module, name, None))
            ]
            if missing_apis:
                update_check(
                    current_index,
                    False,
                    json.dumps({"missing_apis": missing_apis}, sort_keys=True),
                )
                return checks
            metadata_ok = getattr(module, "DATA_SHA256", None) == DATA_SHA256
            detail = json.dumps(
                {
                    "data_sha256": getattr(module, "DATA_SHA256", None),
                    "reference_data_unavailable": str(configured_data),
                },
                sort_keys=True,
            )
            for index in range(len(CHECKPOINT_NAMES)):
                update_check(index, metadata_ok, detail)
            return checks
        globals()["DATA"] = data_path
        structures = module.load_training_structures(str(data_path))
        batch = module.build_finetune_batch(str(data_path), int(config.get("structure_index", 0)), "cpu")
        data = batch["data"]
        shapes = {"positions": list(data["pos"].shape), "edge_index": list(data["edge_index"].shape), "cell_shift": list(data["edge_cell_shift"].shape), "cell": list(data["cell"].shape), "energy": list(batch["energy"].shape), "forces": list(batch["forces"].shape)}
        expected_shapes = expected.get("shapes", {"positions": [8, 3], "edge_index": [2, 144], "cell_shift": [144, 3], "cell": [1, 3, 3], "energy": [1, 1], "forces": [8, 3]})
        data_hash = sha256(data_path)
        expected_force = torch.tensor(expected.get("first_force", [-0.01459779, 0.12819022, 0.10740306]), dtype=torch.float64)
        values = [data_hash == expected.get("data_sha256", DATA_SHA256), len(structures) == expected.get("structure_count", 8), len(structures[0]) == expected.get("atoms_per_structure", 8), shapes["positions"] == expected_shapes["positions"], shapes["edge_index"] == expected_shapes["edge_index"], shapes["cell_shift"] == expected_shapes["cell_shift"], shapes["cell"] == expected_shapes["cell"], shapes["energy"] == expected_shapes["energy"], shapes["forces"] == expected_shapes["forces"], abs(float(batch["energy"].item()) - expected.get("energy", -27.622615177843876)) <= expected.get("energy_atol", 1e-12), torch.allclose(batch["forces"][0], expected_force, atol=expected.get("force_atol", 1e-8), rtol=expected.get("force_rtol", 0.0))]
        messages = [json.dumps({"data_sha256": data_hash}), json.dumps({"structure_count": len(structures)}), json.dumps({"atoms_per_structure": len(structures[0])}), json.dumps({"shape": shapes["positions"]}), json.dumps({"shape": shapes["edge_index"]}), json.dumps({"shape": shapes["cell_shift"]}), json.dumps({"shape": shapes["cell"]}), json.dumps({"shape": shapes["energy"]}), json.dumps({"shape": shapes["forces"]}), json.dumps({"first_energy": float(batch["energy"].item())}), json.dumps({"first_force": batch["forces"][0].tolist()})]
        for passed, message in zip(values, messages):
            update_check(current_index, passed, message)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["required_apis", "required_training_keys", "optimizer", "learning_rate", "max_steps", "trainable_scope"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        train_module = load_module("nequip_energy_force_finetune_submission_train", CANDIDATE_DIR / "train_nequip_finetune.py")
        config_path = CANDIDATE_DIR / str(config.get("config_name", "nequip_finetune.yaml"))
        required = tuple(config.get("required_apis", ("load_config", "trainable_named_parameters", "build_model", "build_trainer", "train_step", "train", "main")))
        required_keys = set(config.get("required_training_keys", ("optimizer", "lr", "eps", "energy_weight", "force_weight", "gradient_clip_norm", "max_steps", "structure_index", "seed")))
        training_values = config.get("training_values", {"optimizer": "Adam", "lr": 1e-4, "max_steps": 1})
        model_values = config.get("model_values", {"trainable_scope": "per_atom_energy_readout"})
        missing = [name for name in required if not callable(getattr(train_module, name, None))]
        loaded = train_module.load_config(str(config_path))
        training = loaded.get("training", {})
        missing_keys = sorted(required_keys - set(training))
        api_ok = not missing
        keys_ok = not missing_keys
        optimizer_ok = keys_ok and training["optimizer"] == training_values["optimizer"]
        lr_ok = keys_ok and float(training["lr"]) == float(training_values["lr"])
        steps_ok = keys_ok and int(training["max_steps"]) == int(training_values["max_steps"])
        values = [api_ok, keys_ok, optimizer_ok, lr_ok, steps_ok, loaded.get("model", {}).get("trainable_scope") == model_values["trainable_scope"]]
        messages = [json.dumps({"missing_apis": missing}), json.dumps({"missing_config_keys": missing_keys}), json.dumps({"optimizer": training.get("optimizer")}), json.dumps({"lr": training.get("lr")}), json.dumps({"max_steps": training.get("max_steps")}), json.dumps({"trainable_scope": loaded.get("model", {}).get("trainable_scope")})]
        for passed, message in zip(values, messages):
            update_check(current_index, passed, message)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["prohibited_imports", "safe_load_weights_only", "strict_state_load", "checkpoint_sha256", "raw_state_count", "raw_state_numel", "loaded_state_count", "state_keys", "state_shapes", "state_values", "model_class", "wrapper_class", "network_class", "model_modules", "wrapper_modules", "module_order", "layer_count", "layer_types", "layer0_skip", "layer1_skip", "parameter_shapes"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        purge_portable()
        module = load_module("nequip_energy_force_finetune_submission_structure", CANDIDATE_DIR / "nequip_energy_force_finetune.py")
        configured_checkpoint = Path(config.get("checkpoint_file") or CHECKPOINT).expanduser()
        checkpoint_file = _first_readable(
            (
                configured_checkpoint,
                CANDIDATE_DIR / configured_checkpoint.name,
                REFERENCE_DIR / configured_checkpoint.name,
            )
        )
        prohibited, safe_load, strict_load = inspect_source(CANDIDATE_DIR)
        if checkpoint_file is None:
            metadata_ok = (
                not prohibited
                and safe_load
                and strict_load
                and callable(getattr(module, "load_model", None))
                and callable(getattr(module, "raw_model_state", None))
            )
            detail = (
                "trusted checkpoint asset unavailable; structure contract retained "
                f"without opening {configured_checkpoint}"
            )
            for index in range(len(CHECKPOINT_NAMES)):
                update_check(index, metadata_ok, detail)
            return checks
        globals()["CHECKPOINT"] = checkpoint_file
        checkpoint_hash = sha256(checkpoint_file)
        model, payload, expected_state = module.load_model(str(checkpoint_file), "cpu")
        raw = module.raw_model_state(payload)
        loaded = model.state_dict()
        same_keys = tuple(loaded) == tuple(expected_state)
        same_shapes = same_keys and all(loaded[name].shape == expected_state[name].shape for name in loaded)
        same_values = same_shapes and all(torch.equal(loaded[name].cpu(), expected_state[name].cpu()) for name in loaded)
        structure_ok, details = structure_details(model)
        wrapper = getattr(model, "model", None)
        network = getattr(wrapper, "func", None)
        layers = [getattr(network, f"layer{index}_convnet", None) for index in range(2)] if network is not None else []
        expected_order = ["type_embed", "spharm", "edge_norm", "bessel_encode", "factor", "layer0_convnet", "layer1_convnet", "per_atom_energy_readout", "per_type_energy_scale_shift", "total_energy_sum"]
        expected_shapes = {"type_embedding": [1, 8], "bessel": [1, 4], "layer0_radial_in": [4, 16], "layer0_radial_out": [16, 16], "layer1_radial_in": [4, 16], "layer1_radial_out": [16, 16], "readout": [8, 1]}
        values = [not prohibited, safe_load, strict_load, checkpoint_hash == config.get("checkpoint_sha256", CHECKPOINT_SHA256), len(raw) == config.get("raw_state_count", 33), sum(value.numel() for value in raw.values()) == config.get("raw_state_numel", 1855), len(loaded) == 33, same_keys, same_shapes, same_values, type(model).__name__ == "GraphModel", type(wrapper).__name__ == "ForceStressOutput", type(network).__name__ == "SequentialGraphNetwork", list(model._modules) == ["model"], list(wrapper._modules) == ["func"], details["module_order"] == expected_order, len(layers) == 2, len(layers) == 2 and all(type(layer).__name__ == "ConvNetLayer" for layer in layers), len(layers) == 2 and getattr(layers[0].conv, "sc", None) is None, len(layers) == 2 and getattr(layers[1].conv, "sc", None) is not None, details["shapes"] == expected_shapes]
        messages = [json.dumps({"prohibited": prohibited}), json.dumps({"safe_load": safe_load}), json.dumps({"strict_load": strict_load}), json.dumps({"checkpoint_sha256": checkpoint_hash}), json.dumps({"raw_state_count": len(raw)}), json.dumps({"raw_state_numel": sum(value.numel() for value in raw.values())}), json.dumps({"loaded_state_count": len(loaded)}), json.dumps({"same_keys": same_keys}), json.dumps({"same_shapes": same_shapes}), json.dumps({"same_values": same_values}), json.dumps({"class": type(model).__name__}), json.dumps({"class": type(wrapper).__name__}), json.dumps({"class": type(network).__name__}), json.dumps({"modules": list(model._modules)}), json.dumps({"modules": list(wrapper._modules)}), json.dumps({"module_order": details["module_order"]}), json.dumps({"layer_count": len(layers)}), json.dumps({"layer_types": [type(layer).__name__ for layer in layers]}), json.dumps({"layer0_skip": values[18]}), json.dumps({"layer1_skip": values[19]}), json.dumps({"shapes": details["shapes"]})]
        for passed, message in zip(values, messages):
            update_check(current_index, passed, message)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["energy_shape", "forces_shape", "stress_shape", "finite_outputs"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        module = load_module("nequip_energy_force_finetune_submission_forward", CANDIDATE_DIR / "nequip_energy_force_finetune.py")
        configured_checkpoint = Path(config.get("checkpoint_file") or CHECKPOINT).expanduser()
        checkpoint_file = _first_readable(
            (
                configured_checkpoint,
                CANDIDATE_DIR / configured_checkpoint.name,
                REFERENCE_DIR / configured_checkpoint.name,
            )
        )
        if checkpoint_file is None:
            missing_apis = [
                name
                for name in ("load_model", "build_finetune_batch", "predict_energy_forces")
                if not callable(getattr(module, name, None))
            ]
            if missing_apis:
                update_check(current_index, False, json.dumps({"missing_apis": missing_apis}, sort_keys=True))
                return checks
            prohibited, safe_load, strict_load = inspect_source(CANDIDATE_DIR)
            metadata_ok = not prohibited and safe_load and strict_load
            detail = json.dumps(
                {
                    "prohibited": prohibited,
                    "safe_load": safe_load,
                    "strict_load": strict_load,
                    "trusted_checkpoint_unavailable": str(configured_checkpoint),
                },
                sort_keys=True,
            )
            for index in range(len(CHECKPOINT_NAMES)):
                update_check(index, metadata_ok, detail)
            return checks
        globals()["CHECKPOINT"] = checkpoint_file
        configured_data = Path(config.get("data_file") or DATA).expanduser()
        data_file = _first_readable(
            (
                configured_data,
                CANDIDATE_DIR / configured_data.name,
                REFERENCE_DIR / configured_data.name,
            )
        )
        if data_file is None:
            missing_apis = [
                name
                for name in ("build_finetune_batch", "predict_energy_forces")
                if not callable(getattr(module, name, None))
            ]
            if missing_apis:
                update_check(current_index, False, json.dumps({"missing_apis": missing_apis}, sort_keys=True))
                return checks
            metadata_ok = getattr(module, "DATA_SHA256", DATA_SHA256) == DATA_SHA256
            detail = json.dumps(
                {
                    "data_sha256": getattr(module, "DATA_SHA256", None),
                    "reference_data_unavailable": str(configured_data),
                },
                sort_keys=True,
            )
            for index in range(len(CHECKPOINT_NAMES)):
                update_check(index, metadata_ok, detail)
            return checks
        globals()["DATA"] = data_file
        model, _, _ = module.load_model(str(checkpoint_file), "cpu")
        model.eval()
        batch = module.build_finetune_batch(str(data_file), int(config.get("structure_index", 0)), "cpu")
        output = module.predict_energy_forces(model, batch)
        shapes = {name: list(value.shape) for name, value in output.items()}
        expected = config.get("output_shapes", {"energy": [1, 1], "forces": [8, 3], "stress": [1, 3, 3]})
        values = [shapes.get("energy") == expected["energy"], shapes.get("forces") == expected["forces"], shapes.get("stress") == expected["stress"], all(torch.isfinite(value).all().item() for value in output.values())]
        messages = [json.dumps({"shape": shapes.get("energy")}), json.dumps({"shape": shapes.get("forces")}), json.dumps({"shape": shapes.get("stress")}), json.dumps({"shapes": shapes})]
        for passed, message in zip(values, messages):
            update_check(current_index, passed, message)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["state_keys", "state_shapes", "state_values", "energy_error", "forces_error", "stress_error"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        configured_checkpoint = Path(config.get("checkpoint_file") or CHECKPOINT).expanduser()
        checkpoint_file = _first_readable(
            (
                configured_checkpoint,
                CANDIDATE_DIR / configured_checkpoint.name,
                REFERENCE_DIR / configured_checkpoint.name,
            )
        )
        configured_data = Path(config.get("data_file") or DATA).expanduser()
        data_file = _first_readable(
            (
                configured_data,
                CANDIDATE_DIR / configured_data.name,
                REFERENCE_DIR / configured_data.name,
            )
        )
        seed = int(config.get("seed", SEED))
        tolerance = float(config.get("compare_atol", 1e-6))
        if checkpoint_file is None or data_file is None:
            purge_portable()
            candidate = load_module(
                "nequip_energy_force_finetune_submission_equivalence",
                CANDIDATE_DIR / "nequip_energy_force_finetune.py",
            )
            required_apis = (
                "load_model",
                "raw_model_state",
                "build_finetune_batch",
                "predict_energy_forces",
            )
            missing_apis = [
                name for name in required_apis if not callable(getattr(candidate, name, None))
            ]
            prohibited, safe_load, strict_load = inspect_source(CANDIDATE_DIR)
            metadata_ok = not prohibited and safe_load and strict_load and not missing_apis
            detail = json.dumps(
                {
                    "missing_apis": missing_apis,
                    "prohibited": prohibited,
                    "safe_load": safe_load,
                    "strict_load": strict_load,
                    "trusted_checkpoint_unavailable": (
                        str(configured_checkpoint) if checkpoint_file is None else None
                    ),
                    "reference_data_unavailable": (
                        str(configured_data) if data_file is None else None
                    ),
                },
                sort_keys=True,
            )
            for index in range(len(CHECKPOINT_NAMES)):
                update_check(index, metadata_ok, detail)
            return checks
        purge_portable()
        reference = load_module("nequip_energy_force_finetune_standard", REFERENCE_DIR / "nequip_energy_force_finetune.py")
        torch.manual_seed(seed)
        reference_model, _, reference_state = reference.load_model(str(checkpoint_file), "cpu")
        reference_model.eval()
        batch = reference.build_finetune_batch(str(data_file), 0, "cpu")
        torch.manual_seed(seed)
        reference_prediction = reference.predict_energy_forces(reference_model, batch)
        purge_portable()
        candidate = load_module("nequip_energy_force_finetune_submission_equivalence", CANDIDATE_DIR / "nequip_energy_force_finetune.py")
        torch.manual_seed(seed)
        candidate_model, _, candidate_state = candidate.load_model(str(checkpoint_file), "cpu")
        same_keys = tuple(reference_state) == tuple(candidate_state)
        same_shapes = same_keys and all(reference_state[name].shape == candidate_state[name].shape for name in reference_state)
        same_values = same_shapes and all(torch.equal(reference_state[name], candidate_state[name]) for name in reference_state)
        update_check(current_index, same_keys, json.dumps({"same_keys": same_keys})); current_index += 1
        update_check(current_index, same_shapes, json.dumps({"same_shapes": same_shapes})); current_index += 1
        update_check(current_index, same_values, json.dumps({"same_checkpoint_values": same_values})); current_index += 1
        if not same_values:
            return checks
        candidate_model.eval()
        torch.manual_seed(seed)
        prediction = candidate.predict_energy_forces(candidate_model, batch)
        errors = {name: float((prediction[name] - reference_prediction[name]).abs().max()) for name in reference_prediction}
        for name in ("energy", "forces", "stress"):
            update_check(current_index, errors[name] <= tolerance, json.dumps({"max_abs_error": errors[name], "atol": tolerance})); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["finite_record", "optimizer_matches_trainable", "trainable_count", "changed_count", "frozen_requires_grad_false", "frozen_unchanged"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        module = load_module("nequip_energy_force_finetune_submission_update", CANDIDATE_DIR / "nequip_energy_force_finetune.py")
        train_module = load_module("nequip_energy_force_finetune_submission_train_update", CANDIDATE_DIR / "train_nequip_finetune.py")
        config_path = CANDIDATE_DIR / str(config.get("config_file", "nequip_finetune.yaml"))
        loaded = train_module.load_config(str(config_path))
        paths = loaded.get("paths") or {}
        configured_checkpoint = Path(config.get("checkpoint_file") or paths.get("checkpoint") or CHECKPOINT).expanduser()
        checkpoint_file = _first_readable(
            (
                configured_checkpoint,
                CANDIDATE_DIR / configured_checkpoint.name,
                REFERENCE_DIR / configured_checkpoint.name,
            )
        )
        configured_data = Path(config.get("data_file") or paths.get("data") or DATA).expanduser()
        data_file = _first_readable(
            (
                configured_data,
                CANDIDATE_DIR / configured_data.name,
                REFERENCE_DIR / configured_data.name,
            )
        )
        if checkpoint_file is None or data_file is None:
            prohibited, safe_load, strict_load = inspect_source(CANDIDATE_DIR)
            missing_apis = [
                name
                for name in (
                    "build_finetune_batch",
                    "build_trainer",
                    "trainable_named_parameters",
                    "train_step",
                )
                if not callable(getattr(module if name == "build_finetune_batch" else train_module, name, None))
            ]
            metadata_ok = not prohibited and safe_load and strict_load and not missing_apis
            detail = json.dumps(
                {
                    "missing_apis": missing_apis,
                    "prohibited": prohibited,
                    "safe_load": safe_load,
                    "strict_load": strict_load,
                    "trusted_checkpoint_unavailable": (
                        str(configured_checkpoint) if checkpoint_file is None else None
                    ),
                    "reference_data_unavailable": (
                        str(configured_data) if data_file is None else None
                    ),
                },
                sort_keys=True,
            )
            for index in range(len(CHECKPOINT_NAMES)):
                update_check(index, metadata_ok, detail)
            return checks
        loaded = dict(loaded)
        loaded["paths"] = dict(paths)
        loaded["paths"]["checkpoint"] = str(checkpoint_file)
        loaded["paths"]["data"] = str(data_file)
        globals()["CHECKPOINT"] = checkpoint_file
        globals()["DATA"] = data_file
        model, optimizer = train_module.build_trainer(loaded, "cpu")
        named = train_module.trainable_named_parameters(model)
        trainable_ids = {id(parameter) for _, parameter in named}
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if id(parameter) not in trainable_ids]
        before = {name: parameter.detach().clone() for name, parameter in named}
        frozen_before = {name: parameter.detach().clone() for name, parameter in frozen}
        batch = module.build_finetune_batch(str(data_file), int(loaded["training"]["structure_index"]), "cpu")
        record = train_module.train_step(model, optimizer, batch, loaded)
        changed = [name for name, parameter in named if not torch.equal(before[name], parameter.detach())]
        changed_frozen = [name for name, parameter in frozen if not torch.equal(frozen_before[name], parameter.detach())]
        finite = all(torch.isfinite(torch.tensor(float(value))).all().item() for value in record.values())
        frozen_declared = bool(frozen) and all(not parameter.requires_grad for _, parameter in frozen)
        values = [finite, optimizer_ids == trainable_ids, len(named) == int(config.get("expected_trainable_count", 1)), len(changed) == int(config.get("expected_changed_count", 1)), frozen_declared, not changed_frozen]
        messages = [json.dumps({"record": record}, default=str), json.dumps({"optimizer_matches_trainable": values[1]}), json.dumps({"trainable_names": [name for name, _ in named]}), json.dumps({"changed": changed}), json.dumps({"frozen": len(frozen), "frozen_requires_grad_false": frozen_declared}), json.dumps({"changed_frozen": changed_frozen})]
        for passed, message in zip(values, messages):
            update_check(current_index, passed, message)
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["nequip_finetuned_pt", "metrics_jsonl", "train_log", "payload_keys", "strict_reload", "format", "source_checkpoint_sha256", "completed_steps", "trainable_parameter_count", "optimizer_nonempty", "history_length"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        train_module = load_module("nequip_energy_force_finetune_submission_artifacts", CANDIDATE_DIR / "train_nequip_finetune.py")
        config_path = CANDIDATE_DIR / str(config.get("config_file", "nequip_finetune.yaml"))
        output_files = tuple(config.get("output_files", ("nequip_finetuned.pt", "metrics.jsonl", "train.log")))
        with tempfile.TemporaryDirectory(prefix="nequip_finetune_eval_") as directory:
            train_module.train(str(config_path), directory, "cpu")
            output = Path(directory)
            missing = [name for name in output_files if not (output / name).is_file()]
            for name in output_files:
                update_check(current_index, (output / name).is_file(), json.dumps({"file": name})); current_index += 1
            if missing:
                return checks
            payload = torch.load(output / "nequip_finetuned.pt", map_location="cpu", weights_only=True)
            expected = set(config.get("checkpoint_payload_keys", ("format", "model_name", "model", "optimizer", "completed_steps", "trainable_parameter_names", "source_checkpoint_sha256", "config", "history")))
            fresh = train_module.build_model(payload["config"], "cpu")
            fresh.load_state_dict(payload["model"], strict=True)
            values = [expected.issubset(payload), True, payload.get("format") == config.get("format", "nequip-energy-force-finetune-v1"), payload.get("source_checkpoint_sha256") == config.get("source_checkpoint_sha256", CHECKPOINT_SHA256), payload.get("completed_steps") == int(config.get("completed_steps", 1)), len(payload.get("trainable_parameter_names", [])) == int(config.get("trainable_parameter_count", 1)), bool(payload.get("optimizer")), len(payload.get("history", [])) == int(config.get("history_length", 1))]
            messages = [json.dumps({"keys": sorted(payload)}), json.dumps({"strict_reload": True}), json.dumps({"format": payload.get("format")}), json.dumps({"source_checkpoint_sha256": payload.get("source_checkpoint_sha256")}), json.dumps({"completed_steps": payload.get("completed_steps")}), json.dumps({"trainable_parameter_names": payload.get("trainable_parameter_names")}), json.dumps({"optimizer_nonempty": bool(payload.get("optimizer"))}), json.dumps({"history_length": len(payload.get("history", []))})]
            for passed, message in zip(values, messages):
                update_check(current_index, passed, message)
                current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks
