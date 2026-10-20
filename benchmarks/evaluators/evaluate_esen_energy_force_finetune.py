"""V2 adapter for the eSEN energy-force fine-tuning evaluator."""
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

CHECKPOINT = Path("/public/share/sugonhpcapp01/onestore/onemodels/eSEN/esen_30m_mptrj.pt")
JD_PATH = Path("/public/share/sugonhpcapp01/onestore/onemodels/UMA/checkpoint/Jd.pt")
DATA = Path("/public/share/sugonhpcapp01/onestore/onedatasets/matchem/esen/oxides/train.db")
CHECKPOINT_SHA256 = "f44aecdfbdb74575f8861c08ee6aa86939cce4571e73ac508666190cb67f064b"
JD_SHA256 = "b4059c45be246dcb6c49c545670b65c56550eb0c2e7a9c92b4b50a92d370dbe2"
STATE_COUNT = 394
STATE_NUMEL = 30087404
SEED = 20260820
REQUIRED_FILES = (
    "esen_energy_force_finetune.py", "train_esen_finetune.py", "esen_finetune.yaml",
    "portable_esen/__init__.py", "portable_esen/model.py", "portable_esen/esen_dens.py",
    "portable_esen/graph.py", "portable_esen/layer/esen/esen_block.py",
    "portable_esen/layer/esen/common/rotation.py", "portable_esen/layer/esen/common/so3.py",
    "portable_esen/layer/esen/nn/embedding.py", "portable_esen/layer/esen/nn/layer_norm.py",
    "portable_esen/layer/esen/nn/radial.py", "portable_esen/layer/esen/nn/so2_layers.py",
    "portable_esen/layer/esen/nn/so3_layers.py", "portable_esen/layer/esen/nn/activation.py",
)
PROHIBITED = {"onescience", "fairchem", "mattersim", "mace"}

def _candidate_path(relative):
    path = (CANDIDATE_DIR / Path(relative)).resolve()
    path.relative_to(CANDIDATE_DIR)
    return path

def _reference_path(relative):
    path = (REFERENCE_DIR / Path(relative)).resolve()
    path.relative_to(REFERENCE_DIR)
    return path

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _purge_portable():
    for name in list(sys.modules):
        if name == "portable_esen" or name.startswith("portable_esen."):
            sys.modules.pop(name, None)

def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["file:esen_energy_force_finetune.py", "file:train_esen_finetune.py", "file:esen_finetune.yaml", "file:portable_esen/__init__.py", "file:portable_esen/model.py", "file:portable_esen/esen_dens.py", "file:portable_esen/graph.py", "file:portable_esen/layer/esen/esen_block.py", "file:portable_esen/layer/esen/common/rotation.py", "file:portable_esen/layer/esen/common/so3.py", "file:portable_esen/layer/esen/nn/embedding.py", "file:portable_esen/layer/esen/nn/layer_norm.py", "file:portable_esen/layer/esen/nn/radial.py", "file:portable_esen/layer/esen/nn/so2_layers.py", "file:portable_esen/layer/esen/nn/so3_layers.py", "file:portable_esen/layer/esen/nn/activation.py"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        for relative in REQUIRED_FILES:
            path = (CANDIDATE_DIR / Path(relative)).resolve()
            path.relative_to(CANDIDATE_DIR)
            update_check(current_index, path.is_file(), str(path))
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["data:structure_count", "data:first_structure_length", "data:first_positions_finite", "data:energy_shape", "data:forces_shape", "data:atomic_numbers_shape", "data:pos_shape", "data:cell_shape"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        candidate_file = config.get("candidate", {}).get("file", "esen_energy_force_finetune.py")
        candidate_path = (CANDIDATE_DIR / Path(candidate_file)).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        spec = importlib.util.spec_from_file_location("esen_energy_force_finetune_submission", candidate_path)
        if spec is None or spec.loader is None:
            raise ImportError(candidate_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["esen_energy_force_finetune_submission"] = module
        spec.loader.exec_module(module)
        structures = module.load_training_structures(str(DATA))
        update_check(current_index, len(structures) == 238, f"count={len(structures)}"); current_index += 1
        first = structures[0]
        update_check(current_index, len(first) == 6, f"length={len(first)}"); current_index += 1
        update_check(current_index, bool(torch.isfinite(torch.as_tensor(first.get_positions())).all()), "finite positions"); current_index += 1
        batch = module.build_finetune_batch(str(DATA), 0, "cpu")
        data = batch["data"]
        actual = {"energy": list(batch["energy"].shape), "forces": list(batch["forces"].shape), "atomic_numbers": list(data["atomic_numbers"].shape), "pos": list(data["pos"].shape), "cell": list(data["cell"].shape)}
        expected = {"energy": [1], "forces": [6, 3], "atomic_numbers": [6], "pos": [6, 3], "cell": [1, 3, 3]}
        update_check(current_index, actual["energy"] == expected["energy"], str(actual["energy"])); current_index += 1
        update_check(current_index, actual["forces"] == expected["forces"], str(actual["forces"])); current_index += 1
        update_check(current_index, actual["atomic_numbers"] == expected["atomic_numbers"], str(actual["atomic_numbers"])); current_index += 1
        update_check(current_index, actual["pos"] == expected["pos"], str(actual["pos"])); current_index += 1
        update_check(current_index, actual["cell"] == expected["cell"], str(actual["cell"])); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["train:load_config", "train:required_interfaces", "train:optimizer", "train:lr", "train:max_steps"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        train_file = config.get("interface_file", "train_esen_finetune.py")
        train_path = (CANDIDATE_DIR / Path(train_file)).resolve()
        train_path.relative_to(CANDIDATE_DIR)
        spec = importlib.util.spec_from_file_location("esen_energy_force_finetune_submission_train", train_path)
        if spec is None or spec.loader is None:
            raise ImportError(train_path)
        train_module = importlib.util.module_from_spec(spec)
        sys.modules["esen_energy_force_finetune_submission_train"] = train_module
        spec.loader.exec_module(train_module)
        config_path = (CANDIDATE_DIR / Path(config.get("config_name", "esen_finetune.yaml"))).resolve()
        config_path.relative_to(CANDIDATE_DIR)
        loaded = train_module.load_config(str(config_path))
        update_check(current_index, isinstance(loaded, dict), str(type(loaded).__name__)); current_index += 1
        required = tuple(config.get("required_interfaces", ("load_config", "build_model", "build_trainer", "train_step", "train", "main")))
        missing_interfaces = [name for name in required if not callable(getattr(train_module, name, None))]
        update_check(current_index, not missing_interfaces, f"missing={missing_interfaces}"); current_index += 1
        training = loaded.get("training", {})
        update_check(current_index, training.get("optimizer") == "Adam", str(training.get("optimizer"))); current_index += 1
        update_check(current_index, float(training.get("lr")) == 1e-6, str(training.get("lr"))); current_index += 1
        update_check(current_index, int(training.get("max_steps")) == 1, str(training.get("max_steps"))); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["checkpoint:prohibited_imports", "checkpoint:weights_only", "checkpoint:strict", "checkpoint:state_count", "checkpoint:state_numel", "checkpoint:model_state_count", "checkpoint:checkpoint_hash", "checkpoint:jd_hash", "checkpoint:same_keys", "checkpoint:same_shapes", "checkpoint:same_values", "structure:backbone_class", "structure:head_class", "structure:block_count", "structure:backbone_attributes", "structure:model_order", "structure:backbone_order", "structure:block_order", "structure:head_order", "structure:sphere_shape", "structure:distance_shape", "structure:energy_shape"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        candidate_path = (CANDIDATE_DIR / Path(config.get("candidate_file", "esen_energy_force_finetune.py"))).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        prohibited, has_weights_only_true, has_strict_true = [], False, False
        for path in CANDIDATE_DIR.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    prohibited += [alias.name for alias in node.names if alias.name.split(".")[0] in PROHIBITED]
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in PROHIBITED:
                    prohibited.append(node.module)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                    if node.func.attr == "load" and isinstance(keywords.get("weights_only"), ast.Constant) and keywords["weights_only"].value is True:
                        has_weights_only_true = True
                    if node.func.attr == "load_state_dict" and isinstance(keywords.get("strict"), ast.Constant) and keywords["strict"].value is True:
                        has_strict_true = True
        update_check(current_index, not prohibited, str(prohibited)); current_index += 1
        update_check(current_index, has_weights_only_true, "weights_only=True"); current_index += 1
        update_check(current_index, has_strict_true, "strict=True"); current_index += 1
        spec = importlib.util.spec_from_file_location("esen_energy_force_finetune_submission_structure", candidate_path)
        if spec is None or spec.loader is None:
            raise ImportError(candidate_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["esen_energy_force_finetune_submission_structure"] = module
        spec.loader.exec_module(module)
        payload = module.load_checkpoint_payload(str(CHECKPOINT))
        state = payload["state_dict"]
        loaded = module.load_model(str(CHECKPOINT), device="cpu", jd_path=str(JD_PATH))
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        expected_state = {str(key)[len("module."):]: value for key, value in state.items()}
        model_state = model.state_dict()
        same_keys = tuple(expected_state) == tuple(model_state)
        same_shapes = same_keys and all(expected_state[name].shape == model_state[name].shape for name in expected_state)
        same_values = same_shapes and all(torch.equal(expected_state[name], model_state[name].cpu()) for name in expected_state)
        update_check(current_index, len(state) == STATE_COUNT, str(len(state))); current_index += 1
        update_check(current_index, sum(value.numel() for value in state.values()) == STATE_NUMEL, str(sum(value.numel() for value in state.values()))); current_index += 1
        update_check(current_index, len(model_state) == STATE_COUNT, str(len(model_state))); current_index += 1
        update_check(current_index, _file_sha256(CHECKPOINT) == CHECKPOINT_SHA256, _file_sha256(CHECKPOINT)); current_index += 1
        update_check(current_index, _file_sha256(JD_PATH) == JD_SHA256, _file_sha256(JD_PATH)); current_index += 1
        update_check(current_index, same_keys, str(same_keys)); current_index += 1
        update_check(current_index, same_shapes, str(same_shapes)); current_index += 1
        update_check(current_index, same_values, str(same_values)); current_index += 1
        backbone = model.backbone
        head = model.output_heads["mptrj"]
        model_order = list(model._modules)
        backbone_order = list(backbone._modules)
        required_order = ("SO3_grid", "sphere_embedding", "distance_expansion", "source_embedding", "target_embedding", "edge_degree_embedding", "blocks", "norm", "force_embedding")
        positions = [backbone_order.index(name) for name in required_order if name in backbone_order]
        backbone_order_ok = len(positions) == len(required_order) and positions == sorted(positions)
        blocks = list(backbone.blocks)
        block_order_ok = len(blocks) == 10 and all(tuple(block._modules) == ("norm_1", "edge_wise", "norm_2", "atom_wise") for block in blocks)
        with torch.no_grad():
            sphere_shape = list(backbone.sphere_embedding(torch.tensor([1, 8])).shape)
            distance_shape = list(backbone.distance_expansion(torch.tensor([0.5, 3.0, 5.5])).shape)
            energy_shape = list(head.energy_block(torch.ones(2, 128)).shape)
        update_check(current_index, type(backbone).__name__ == "eSEN_DeNS_Backbone", type(backbone).__name__); current_index += 1
        update_check(current_index, type(head).__name__ == "MLP_EFS_Head", type(head).__name__); current_index += 1
        update_check(current_index, len(blocks) == 10, str(len(blocks))); current_index += 1
        update_check(current_index, (backbone.lmax, backbone.mmax, backbone.sphere_channels, backbone.cutoff, backbone.direct_forces, backbone.regress_stress) == (3, 2, 128, 6.0, False, True), str((backbone.lmax, backbone.mmax, backbone.sphere_channels, backbone.cutoff, backbone.direct_forces, backbone.regress_stress))); current_index += 1
        update_check(current_index, model_order == ["backbone", "output_heads"], str(model_order)); current_index += 1
        update_check(current_index, backbone_order_ok, str(backbone_order)); current_index += 1
        update_check(current_index, block_order_ok, str([tuple(block._modules) for block in blocks])); current_index += 1
        update_check(current_index, list(head._modules) == ["energy_block", "denoising_linear"], str(list(head._modules))); current_index += 1
        update_check(current_index, sphere_shape == [2, 128], str(sphere_shape)); current_index += 1
        update_check(current_index, distance_shape == [3, 10], str(distance_shape)); current_index += 1
        update_check(current_index, energy_shape == [2, 1], str(energy_shape)); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward:energy_shape", "forward:forces_shape", "forward:stress_shape", "forward:energy_finite", "forward:forces_finite", "forward:stress_finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        candidate_path = (CANDIDATE_DIR / Path(config.get("candidate_file", "esen_energy_force_finetune.py"))).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        spec = importlib.util.spec_from_file_location("esen_energy_force_finetune_submission_forward", candidate_path)
        if spec is None or spec.loader is None:
            raise ImportError(candidate_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["esen_energy_force_finetune_submission_forward"] = module
        spec.loader.exec_module(module)
        torch.manual_seed(SEED)
        loaded = module.load_model(str(CHECKPOINT), device="cpu")
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model.eval()
        batch = module.build_finetune_batch(str(DATA), 0, "cpu")
        prediction = module.predict_energy_forces(model, batch)
        expected = {"energy": [1], "forces": [6, 3], "stress": [1, 3, 3]}
        for key in ("energy", "forces", "stress"):
            actual = list(prediction[key].shape)
            update_check(current_index, actual == expected[key], f"{key}={actual}"); current_index += 1
        for key in ("energy", "forces", "stress"):
            finite = bool(torch.isfinite(prediction[key]).all().item())
            update_check(current_index, finite, f"{key} finite={finite}"); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["equivalence:state_keys", "equivalence:state_shapes", "equivalence:state_values", "equivalence:energy", "equivalence:forces", "equivalence:stress"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        _purge_portable()
        standard = _load_module("esen_energy_force_finetune_standard", _reference_path("esen_energy_force_finetune.py"))
        torch.manual_seed(SEED)
        std_model = standard.load_model(str(CHECKPOINT), "cpu")[0]
        std_batch = standard.build_finetune_batch(str(DATA), 0, "cpu")
        torch.manual_seed(SEED)
        std_pred = standard.predict_energy_forces(std_model, std_batch)
        _purge_portable()
        torch.manual_seed(SEED)
        candidate_path = (CANDIDATE_DIR / Path(config.get("candidate", {}).get("file", "esen_energy_force_finetune.py"))).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        module = _load_module("esen_energy_force_finetune_submission_equivalence", candidate_path)
        loaded = module.load_model(str(CHECKPOINT), device="cpu")
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        batch = module.build_finetune_batch(str(DATA), 0, "cpu")
        standard_state = std_model.state_dict()
        candidate_state = model.state_dict()
        same_keys = tuple(standard_state) == tuple(candidate_state)
        same_shapes = same_keys and all(standard_state[name].shape == candidate_state[name].shape for name in standard_state)
        same_values = same_shapes and all(torch.equal(standard_state[name], candidate_state[name]) for name in standard_state)
        update_check(current_index, same_keys, str(same_keys)); current_index += 1
        update_check(current_index, same_shapes, str(same_shapes)); current_index += 1
        update_check(current_index, same_values, str(same_values)); current_index += 1
        torch.manual_seed(SEED)
        pred = module.predict_energy_forces(model, batch)
        errors = {key: float((pred[key] - std_pred[key]).abs().max()) for key in ("energy", "forces", "stress")}
        for key in ("energy", "forces", "stress"):
            update_check(current_index, errors[key] <= 1e-4, f"{key} max_abs_error={errors[key]}"); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["training:record_finite", "training:optimizer_ids", "training:changed_fraction", "training:frozen_present", "training:frozen_requires_grad_false", "training:no_frozen_changes"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        train_path = (CANDIDATE_DIR / Path(config.get("interface_file", "train_esen_finetune.py"))).resolve()
        train_path.relative_to(CANDIDATE_DIR)
        train_module = _load_module("esen_energy_force_finetune_submission_train_update", train_path)
        candidate_path = (CANDIDATE_DIR / Path(config.get("candidate_file", "esen_energy_force_finetune.py"))).resolve()
        candidate_path.relative_to(CANDIDATE_DIR)
        module = _load_module("esen_energy_force_finetune_submission_update", candidate_path)
        loaded_config = train_module.load_config(str((CANDIDATE_DIR / Path(config.get("config_name", "esen_finetune.yaml"))).resolve()))
        torch.manual_seed(int(loaded_config["training"]["seed"]))
        model, optimizer = train_module.build_trainer(loaded_config, "cpu")
        batch = module.build_finetune_batch(str(DATA), 0, "cpu")
        payload = module.load_checkpoint_payload(str(CHECKPOINT))
        selector = getattr(train_module, "trainable_named_parameters", None)
        names = selector(model) if callable(selector) else [(name, parameter) for name, parameter in model.named_parameters() if "denoising_linear" not in name]
        trainable_ids = {id(parameter) for _, parameter in names}
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        frozen = [(name, parameter) for name, parameter in model.named_parameters() if id(parameter) not in trainable_ids]
        before = {name: parameter.detach().clone() for name, parameter in names}
        frozen_before = {name: parameter.detach().clone() for name, parameter in frozen}
        record = train_module.train_step(model, optimizer, batch, payload, loaded_config)
        changed = [name for name, parameter in names if not torch.equal(before[name], parameter.detach())]
        changed_frozen = [name for name, parameter in frozen if not torch.equal(frozen_before[name], parameter.detach())]
        finite = all(value == value and abs(float(value)) < float("inf") for value in record.values())
        frozen_declared = bool(frozen) and all(not parameter.requires_grad for _, parameter in frozen)
        update_check(current_index, finite, str(record)); current_index += 1
        update_check(current_index, optimizer_ids == trainable_ids, str(optimizer_ids == trainable_ids)); current_index += 1
        update_check(current_index, len(changed) >= max(1, int(0.95 * len(names))), f"changed={len(changed)} trainable={len(names)}"); current_index += 1
        update_check(current_index, bool(frozen), str(len(frozen))); current_index += 1
        update_check(current_index, frozen_declared, str(frozen_declared)); current_index += 1
        update_check(current_index, not changed_frozen, str(changed_frozen)); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["artifacts:checkpoint_file", "artifacts:metrics_file", "artifacts:log_file", "artifacts:payload_keys", "artifacts:format", "artifacts:optimizer", "artifacts:completed_steps", "artifacts:history", "artifacts:strict_reload"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        train_path = (CANDIDATE_DIR / Path(config.get("interface_file", "train_esen_finetune.py"))).resolve()
        train_path.relative_to(CANDIDATE_DIR)
        train_module = _load_module("esen_energy_force_finetune_submission_train_smoke", train_path)
        config_path = (CANDIDATE_DIR / Path(config.get("config_name", "esen_finetune.yaml"))).resolve()
        config_path.relative_to(CANDIDATE_DIR)
        with tempfile.TemporaryDirectory(prefix="esen_finetune_eval_") as directory:
            train_module.train(str(config_path), directory, "cpu")
            required = ["esen_finetuned.pt", "metrics.jsonl", "train.log"]
            paths = {name: Path(directory) / name for name in required}
            for name in required:
                update_check(current_index, paths[name].is_file(), str(paths[name])); current_index += 1
            payload = torch.load(paths["esen_finetuned.pt"], map_location="cpu", weights_only=True)
            expected = {"format", "model_name", "model", "model_args", "optimizer", "completed_steps", "trainable_parameter_names", "config", "history"}
            update_check(current_index, expected.issubset(set(payload)), str(sorted(payload))); current_index += 1
            update_check(current_index, payload["format"] == "esen-energy-force-finetune-v1", str(payload.get("format"))); current_index += 1
            update_check(current_index, bool(payload["optimizer"]), str(bool(payload.get("optimizer")))); current_index += 1
            update_check(current_index, payload["completed_steps"] == 1, str(payload.get("completed_steps"))); current_index += 1
            update_check(current_index, bool(payload["history"]), str(bool(payload.get("history")))); current_index += 1
            fresh_model = train_module.build_model(payload["config"], "cpu")
            fresh_model.load_state_dict(payload["model"], strict=True)
            update_check(current_index, True, "strict reload succeeded"); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks
