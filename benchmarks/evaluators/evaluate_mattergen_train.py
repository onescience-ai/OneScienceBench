"""V2 evaluator adapter for the legacy MatterGen training evaluator."""
from pathlib import Path
import sys
import ast
import gc
import importlib.util
import inspect
import json
import math
import os
import tempfile
import traceback
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

REQUIRED_FILES = ("mattergen_denoiser.py", "train_mattergen.py", "mattergen_train.yaml", "portable_mattergen/__init__.py", "portable_mattergen/gemnet/gemnet-dT.json")
FORBIDDEN_IMPORT_ROOTS = {"onescience", "mattergen", "torch_geometric", "torch_scatter", "torch_sparse"}
OUTPUT_KEYS = ("pos", "cell", "atomic_numbers")
EXPECTED_STATE_ENTRY_COUNT = 281
EXPECTED_STATE_NUMEL = 53_737_595
FORWARD_ATOL = 1.0e-6
SEED = 20260818
_ADAPTER_STATE: dict[str, Any] = {}

def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _purge_loaded_modules() -> None:
    for name in list(sys.modules):
        if name in {"mattergen_denoiser", "train_mattergen"} or name.startswith("portable_mattergen") or name.startswith("mattergen_train_eval_loaded_"):
            sys.modules.pop(name, None)

def _load_implementation(root: Path, role: str):
    _purge_loaded_modules()
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    model_module = _load_module("mattergen_denoiser", root / "mattergen_denoiser.py")
    train_module = _load_module(f"mattergen_train_eval_loaded_{role}", root / "train_mattergen.py")
    return model_module, train_module

def _candidate_modules(context: dict[str, Any]):
    if "candidate_model_module" not in context:
        model_module, train_module = _load_implementation(CANDIDATE_DIR, "candidate")
        config = model_module.load_config(CANDIDATE_DIR / "mattergen_train.yaml")
        context.update({"candidate_model_module": model_module, "candidate_train_module": train_module, "candidate_config": config})
    return context["candidate_model_module"], context["candidate_train_module"], context["candidate_config"]

def _tensor_equal(left: Any, right: Any) -> bool:
    import torch
    return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and bool(torch.equal(left.detach().cpu(), right.detach().cpu()))

def _fixture_to_device(fixture: dict[str, Any], device: Any) -> dict[str, Any]:
    return {"batch": fixture["batch"].to(device), "timesteps": fixture["timesteps"].to(device), "targets": {key: value.to(device) for key, value in fixture["targets"].items()}, "atom_loss_mask": fixture["atom_loss_mask"].to(device)}

def _select_device(torch) -> Any:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_parameters_with_constants(model, weight_value: float = 0.002, bias_value: float = 0.0001, embedding_value: float = 0.001, norm_value: float = 1.0, attention_value: float = 0.001, temperature_value: float = 1.0, other_value: float = 0.001):
    import torch
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            name_lower = name.lower()
            if "bias" in name_lower:
                parameter.fill_(bias_value)
            elif any(token in name_lower for token in ("embedding", "embed", "token", "position", "pos_embed")):
                parameter.fill_(embedding_value)
            elif any(token in name_lower for token in ("norm", "ln", "layernorm")):
                parameter.fill_(norm_value)
            elif "temperature" in name_lower:
                parameter.fill_(temperature_value)
            elif parameter.ndim >= 2 and "weight" in name_lower:
                pattern = torch.arange(parameter.numel(), device=parameter.device, dtype=parameter.dtype).reshape(parameter.shape)
                pattern = pattern - pattern.mean()
                scale = pattern.abs().max().clamp_min(1.0)
                parameter.copy_(weight_value * pattern / scale)
            elif any(token in name_lower for token in ("attn", "attention", "q_proj", "k_proj", "v_proj", "query", "key", "value")):
                parameter.fill_(attention_value)
            else:
                parameter.fill_(other_value if parameter.ndim == 0 else embedding_value)
    return model

def _reset_forward_rng(torch) -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def _scan_sources(submission_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    python_files = sorted(submission_dir.rglob("*.py"))
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                        errors.append(f"{path.relative_to(submission_dir)}: forbidden import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module
            if imported and imported.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                errors.append(f"{path.relative_to(submission_dir)}: forbidden import {imported}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                normalized = node.value.replace("\\", "/").lower()
                if "benchmarks/standard" in normalized:
                    errors.append(f"{path.relative_to(submission_dir)}: standard source access is forbidden")
                if "benchmarks/evaluators" in normalized:
                    errors.append(f"{path.relative_to(submission_dir)}: evaluator source access is forbidden")
    return {"python_file_count": len(python_files), "errors": errors}

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["file:required:mattergen_denoiser.py", "file:required:train_mattergen.py", "file:required:mattergen_train.yaml", "file:required:portable_mattergen/__init__.py", "file:required:portable_mattergen/gemnet/gemnet-dT.json"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        for relative in REQUIRED_FILES:
            path = (CANDIDATE_DIR / relative).resolve()
            update_check(current_index, path.is_file(), str(path))
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["data:fixture_keys", "data:batch_keys", "data:target_keys", "data:batch_pos_shape", "data:batch_cell_shape", "data:batch_atomic_numbers_shape", "data:batch_num_atoms_shape", "data:target_pos_shape", "data:target_cell_shape", "data:target_atomic_numbers_shape", "data:pos_dtype", "data:cell_dtype", "data:atomic_numbers_dtype", "data:num_atoms_dtype", "data:timesteps_shape", "data:timesteps_dtype", "data:mask_shape", "data:mask_dtype", "data:num_atoms_values", "data:num_atoms_sum", "data:mask_values", "data:masked_atomic_numbers", "data:finite_fields", "data:positive_cell_determinants", "data:deterministic"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = caller_locals if isinstance(caller_locals, dict) else {}
        _model_module, train_module, _config = _candidate_modules(context)
        first = train_module.build_synthetic_batch(seed=SEED)
        second = train_module.build_synthetic_batch(seed=SEED)
        required = {"batch", "timesteps", "targets", "atom_loss_mask"}
        passed = isinstance(first, dict) and set(first) == required
        update_check(current_index, passed, f"fixture keys must be {sorted(required)}")
        current_index += 1
        if not passed:
            return checks
        batch = first["batch"]
        targets = first["targets"]
        expected_batch_shapes = {"pos": (7, 3), "cell": (2, 3, 3), "atomic_numbers": (7,), "num_atoms": (2,)}
        expected_target_shapes = {"pos": (7, 3), "cell": (2, 3, 3), "atomic_numbers": (7,)}
        passed = set(batch) == set(expected_batch_shapes); update_check(current_index, passed, "batch must contain exactly four crystal fields"); current_index += 1
        passed = set(targets) == set(expected_target_shapes); update_check(current_index, passed, "targets must contain exactly three denoising fields"); current_index += 1
        for key, shape in expected_batch_shapes.items():
            value = batch.get(key); update_check(current_index, isinstance(value, torch.Tensor) and tuple(value.shape) == shape, f"batch[{key}] shape must be {shape}"); current_index += 1
        for key, shape in expected_target_shapes.items():
            value = targets.get(key); update_check(current_index, isinstance(value, torch.Tensor) and tuple(value.shape) == shape, f"targets[{key}] shape must be {shape}"); current_index += 1
        update_check(current_index, batch["pos"].dtype == torch.float32, "pos must be float32"); current_index += 1
        update_check(current_index, batch["cell"].dtype == torch.float32, "cell must be float32"); current_index += 1
        update_check(current_index, batch["atomic_numbers"].dtype == torch.long, "atomic_numbers must be int64"); current_index += 1
        update_check(current_index, batch["num_atoms"].dtype == torch.long, "num_atoms must be int64"); current_index += 1
        update_check(current_index, tuple(first["timesteps"].shape) == (2,), "timesteps must have shape [2]"); current_index += 1
        update_check(current_index, first["timesteps"].dtype == torch.float32, "timesteps must be float32"); current_index += 1
        update_check(current_index, tuple(first["atom_loss_mask"].shape) == (7,), "atom_loss_mask must have shape [7]"); current_index += 1
        update_check(current_index, first["atom_loss_mask"].dtype == torch.bool, "atom_loss_mask must be bool"); current_index += 1
        update_check(current_index, batch["num_atoms"].tolist() == [4, 3], "num_atoms must be [4, 3]"); current_index += 1
        update_check(current_index, int(batch["num_atoms"].sum()) == 7, "num_atoms must sum to seven"); current_index += 1
        update_check(current_index, first["atom_loss_mask"].tolist() == [False, True, False, False, False, True, False], "fixed atom mask is incorrect"); current_index += 1
        update_check(current_index, batch["atomic_numbers"][first["atom_loss_mask"]].tolist() == [101, 101], "masked atom types must use class 101"); current_index += 1
        update_check(current_index, bool(torch.isfinite(batch["pos"]).all() and torch.isfinite(batch["cell"]).all()), "batch contains non-finite values"); current_index += 1
        update_check(current_index, bool((torch.linalg.det(batch["cell"]) > 0).all()), "cells must be invertible with positive determinant"); current_index += 1
        deterministic = all(_tensor_equal(first["batch"][key], second["batch"][key]) for key in expected_batch_shapes) and all(_tensor_equal(first["targets"][key], second["targets"][key]) for key in expected_target_shapes) and _tensor_equal(first["timesteps"], second["timesteps"]) and _tensor_equal(first["atom_loss_mask"], second["atom_loss_mask"])
        update_check(current_index, deterministic, "build_synthetic_batch must be deterministic for a fixed seed")
        current_index += 1
        _ADAPTER_STATE["fixture"] = first
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks

def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["train:required_functions", "train:source_scan", "train:cli_flags", "train:backward", "train:step", "train:Adam", "train:model.hidden_dim", "train:model.denoise_atom_types", "train:model.atom_type_diffusion", "train:gemnet.num_targets", "train:gemnet.cutoff", "train:gemnet.emb_size_atom", "train:gemnet.emb_size_edge", "train:gemnet.max_neighbors", "train:gemnet.max_cell_images_per_dim", "train:gemnet.num_blocks", "train:gemnet.regress_stress", "train:gemnet.otf_graph", "train:seed", "train:optimizer", "train:steps", "train:learning_rate", "train:gradient_clip_val", "train:loss_weights"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        context = caller_locals if isinstance(caller_locals, dict) else {}
        model_module, train_module, config_value = _candidate_modules(context)
        model_functions = ("load_config", "build_model")
        train_functions = ("build_synthetic_batch", "compute_materials_loss", "train_step", "train", "main")
        missing = [name for name in model_functions if not callable(getattr(model_module, name, None))] + [name for name in train_functions if not callable(getattr(train_module, name, None))]
        update_check(current_index, not missing, f"missing required functions: {missing}"); current_index += 1
        source_scan = _scan_sources(CANDIDATE_DIR)
        update_check(current_index, not source_scan["errors"], str(source_scan)); current_index += 1
        source = (CANDIDATE_DIR / "train_mattergen.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="train_mattergen.py")
        cli_flags: set[str] = set(); called_attributes: set[str] = set(); called_names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call): continue
            if isinstance(node.func, ast.Name): called_names.add(node.func.id)
            if isinstance(node.func, ast.Attribute):
                called_attributes.add(node.func.attr)
                if node.func.attr == "add_argument" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str): cli_flags.add(node.args[0].value)
        required_flags = {"--config", "--output-dir", "--device"}
        update_check(current_index, required_flags <= cli_flags, f"CLI flags must include {sorted(required_flags)}"); current_index += 1
        update_check(current_index, "backward" in called_attributes, "training script must call loss.backward()"); current_index += 1
        update_check(current_index, "step" in called_attributes, "training script must call optimizer.step()"); current_index += 1
        update_check(current_index, "Adam" in called_attributes or "Adam" in called_names, "training script must construct Adam"); current_index += 1
        expected_model = {"hidden_dim": 512, "denoise_atom_types": True, "atom_type_diffusion": "mask"}
        expected_gemnet = {"num_targets": 1, "cutoff": 7.0, "emb_size_atom": 512, "emb_size_edge": 512, "max_neighbors": 50, "max_cell_images_per_dim": 5, "num_blocks": 4, "regress_stress": True, "otf_graph": True}
        model_config = config_value.get("model", {}); gemnet_config = model_config.get("gemnet", {})
        for key, expected in expected_model.items():
            update_check(current_index, model_config.get(key) == expected, f"model.{key} must be {expected!r}"); current_index += 1
        for key, expected in expected_gemnet.items():
            update_check(current_index, gemnet_config.get(key) == expected, f"model.gemnet.{key} must be {expected!r}"); current_index += 1
        training = config_value.get("training", {})
        update_check(current_index, training.get("seed") == SEED, f"training.seed must be {SEED}"); current_index += 1
        update_check(current_index, training.get("optimizer") == "Adam", "training.optimizer must be Adam"); current_index += 1
        update_check(current_index, training.get("steps") == 1, "training.steps must be 1"); current_index += 1
        update_check(current_index, math.isclose(float(training.get("learning_rate", -1)), 1.0e-4), "training.learning_rate must be 1e-4"); current_index += 1
        update_check(current_index, math.isclose(float(training.get("gradient_clip_val", -1)), 0.5), "training.gradient_clip_val must be 0.5"); current_index += 1
        expected_weights = {"pos": 0.1, "cell": 1.0, "atomic_numbers": 1.0}
        update_check(current_index, training.get("loss_weights") == expected_weights, f"loss_weights must be {expected_weights}"); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_eval_script_contract(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["eval_script:optional_not_required"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        optional_paths = [path.name for path in CANDIDATE_DIR.glob("eval*.py") if path.name != "evaluate_mattergen_train.py"]
        observations = {"required": False, "status": "not required; training artifacts are inspected directly", "optional_eval_files": sorted(optional_paths)}
        update_check(current_index, True, json.dumps(observations, ensure_ascii=False, sort_keys=True))
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["model_structure:module", "model_structure:class", "model_structure:local_class", "model_structure:forward_signature", "model_structure:top_level_order", "model_structure:hidden_dim", "model_structure:property_embeddings", "model_structure:gemnet_components", "model_structure:graph_builder", "model_structure:block_counts", "model_structure:atom_shape", "model_structure:time_shape", "model_structure:atom_head_shape", "model_structure:state_entries", "model_structure:state_numel"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = caller_locals if isinstance(caller_locals, dict) else {}
        model_module, _train_module, config_value = _candidate_modules(context)
        torch.manual_seed(SEED)
        model = model_module.build_model(config_value)
        update_check(current_index, isinstance(model, torch.nn.Module), "build_model must return torch.nn.Module"); current_index += 1
        update_check(current_index, type(model).__name__ == "GemNetTDenoiser", "model class must be GemNetTDenoiser"); current_index += 1
        try:
            source_path = Path(inspect.getfile(type(model))).resolve(); local_class = source_path.is_relative_to(CANDIDATE_DIR.resolve())
        except Exception:
            source_path = Path("<unknown>"); local_class = False
        update_check(current_index, local_class, "GemNetTDenoiser must be implemented inside the submission"); current_index += 1
        forward_parameters = list(inspect.signature(model.forward).parameters)
        update_check(current_index, forward_parameters[:2] == ["batch", "timesteps"], "forward signature must start with (batch, timesteps)"); current_index += 1
        top_children = [name for name, _module in model.named_children()]
        update_check(current_index, top_children == ["gemnet", "noise_level_encoding", "property_embeddings", "fc_atom"], f"top-level module order must be ['gemnet', 'noise_level_encoding', 'property_embeddings', 'fc_atom']"); current_index += 1
        update_check(current_index, int(getattr(model, "hidden_dim", -1)) == 512, "hidden_dim must be 512"); current_index += 1
        update_check(current_index, isinstance(getattr(model, "property_embeddings", None), torch.nn.ModuleDict), "property_embeddings must be ModuleDict"); current_index += 1
        backbone = getattr(model, "gemnet", None)
        component_names = ("radial_basis", "cbf_basis3", "mlp_rbf_lattice", "mlp_rbf3", "mlp_cbf3", "mlp_rbf_h", "mlp_rbf_out", "atom_emb", "atom_latent_emb", "edge_emb", "int_blocks", "out_blocks", "lattice_out_blocks")
        missing_components = [name for name in component_names if not hasattr(backbone, name)]
        update_check(current_index, not missing_components, f"GemNet-T components missing: {missing_components}"); current_index += 1
        update_check(current_index, callable(getattr(backbone, "generate_interaction_graph", None)), "GemNet-T graph builder is missing"); current_index += 1
        block_counts = {"interaction": len(getattr(backbone, "int_blocks", [])), "output": len(getattr(backbone, "out_blocks", [])), "lattice": len(getattr(backbone, "lattice_out_blocks", []))}
        update_check(current_index, block_counts == {"interaction": 4, "output": 5, "lattice": 5}, f"incorrect GemNet-T block counts: {block_counts}"); current_index += 1
        embedding_shapes: dict[str, Any] = {}
        try:
            embedding_shapes["atom"] = list(backbone.atom_emb(torch.tensor([1, 6, 14, 101], dtype=torch.long)).shape)
            embedding_shapes["time"] = list(model.noise_level_encoding(torch.tensor([0.125, 0.875])).shape)
            embedding_shapes["atom_head"] = list(model.fc_atom(torch.zeros((3, 512), dtype=torch.float32)).shape)
        except Exception:
            pass
        update_check(current_index, embedding_shapes.get("atom") == [4, 512], "atom embedding shape must be [4, 512]"); current_index += 1
        update_check(current_index, embedding_shapes.get("time") == [2, 512], "time embedding shape must be [2, 512]"); current_index += 1
        update_check(current_index, embedding_shapes.get("atom_head") == [3, 101], "atom head shape must be [3, 101]"); current_index += 1
        state = model.state_dict(); state_numel = sum(value.numel() for value in state.values())
        update_check(current_index, len(state) == EXPECTED_STATE_ENTRY_COUNT, f"state_dict must contain {EXPECTED_STATE_ENTRY_COUNT} entries"); current_index += 1
        update_check(current_index, state_numel == EXPECTED_STATE_NUMEL, f"state_dict must contain {EXPECTED_STATE_NUMEL} elements"); current_index += 1
        del model, state
        gc.collect()
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward:pos_tensor", "forward:pos_shape", "forward:pos_floating", "forward:pos_finite", "forward:pos_grad", "forward:cell_tensor", "forward:cell_shape", "forward:cell_floating", "forward:cell_finite", "forward:cell_grad", "forward:atomic_numbers_tensor", "forward:atomic_numbers_shape", "forward:atomic_numbers_floating", "forward:atomic_numbers_finite", "forward:atomic_numbers_grad"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = caller_locals if isinstance(caller_locals, dict) else {}
        model_module, train_module, config_value = _candidate_modules(context)
        fixture = _ADAPTER_STATE.get("fixture") or train_module.build_synthetic_batch(seed=SEED)
        device = _select_device(torch)
        torch.manual_seed(SEED)
        model = model_module.build_model(config_value).to(device).train()
        moved = _fixture_to_device(fixture, device)
        output = model(moved["batch"], moved["timesteps"])
        shapes = {"pos": (7, 3), "cell": (2, 3, 3), "atomic_numbers": (7, 101)}
        for key in OUTPUT_KEYS:
            value = output.get(key) if hasattr(output, "get") else None
            expected_shape = shapes[key]
            update_check(current_index, isinstance(value, torch.Tensor), f"output[{key}] must be a tensor"); current_index += 1
            update_check(current_index, isinstance(value, torch.Tensor) and tuple(value.shape) == expected_shape, f"output[{key}] shape must be {expected_shape}"); current_index += 1
            update_check(current_index, isinstance(value, torch.Tensor) and value.is_floating_point(), f"output[{key}] must be floating point"); current_index += 1
            update_check(current_index, isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), f"output[{key}] contains NaN or Inf"); current_index += 1
            update_check(current_index, isinstance(value, torch.Tensor) and value.requires_grad, f"output[{key}] must retain gradients in training mode"); current_index += 1
        _ADAPTER_STATE["device"] = str(device)
        del model, moved, output
        if device.type == "cuda": torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_equivalence:same_state_layout", "forward_equivalence:identical_initialized_state", "forward_equivalence:pos", "forward_equivalence:cell", "forward_equivalence:atomic_numbers", "forward_equivalence:weighted_loss", "forward_equivalence:pos_loss", "forward_equivalence:cell_loss", "forward_equivalence:atomic_numbers_loss"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = caller_locals if isinstance(caller_locals, dict) else {}
        candidate_model_module, candidate_train_module, candidate_config = _candidate_modules(context)
        fixture = _ADAPTER_STATE.get("fixture") or candidate_train_module.build_synthetic_batch(seed=SEED)
        reference_model_module, reference_train_module = _load_implementation(REFERENCE_DIR, "standard")
        reference_config = reference_model_module.load_config(REFERENCE_DIR / "mattergen_train.yaml")
        torch.manual_seed(SEED); reference_model = reference_model_module.build_model(reference_config).cpu().train()
        torch.manual_seed(SEED); candidate_model = candidate_model_module.build_model(candidate_config).cpu().train()
        initialize_parameters_with_constants(reference_model); initialize_parameters_with_constants(candidate_model)
        reference_state = reference_model.state_dict(); candidate_state = candidate_model.state_dict()
        same_layout = set(reference_state) == set(candidate_state) and all(tuple(reference_state[name].shape) == tuple(candidate_state[name].shape) for name in reference_state.keys() & candidate_state.keys())
        identical_initialized_state = same_layout and all(torch.equal(reference_state[name], candidate_state[name]) for name in reference_state)
        update_check(current_index, same_layout, "candidate and standard state layouts differ"); current_index += 1
        update_check(current_index, identical_initialized_state, "shared deterministic initialization did not produce identical states"); current_index += 1
        if not same_layout or not identical_initialized_state: return checks
        device = _select_device(torch); moved = _fixture_to_device(fixture, device)
        candidate_model = candidate_model.to(device); _reset_forward_rng(torch); candidate_output_device = candidate_model(moved["batch"], moved["timesteps"]); candidate_output = {key: candidate_output_device[key].detach().cpu() for key in OUTPUT_KEYS}; candidate_model.cpu()
        reference_model = reference_model.to(device); _reset_forward_rng(torch); reference_output_device = reference_model(moved["batch"], moved["timesteps"]); reference_output = {key: reference_output_device[key].detach().cpu() for key in OUTPUT_KEYS}
        output_diffs: dict[str, Any] = {}
        for key in OUTPUT_KEYS:
            actual = candidate_output[key]; expected = reference_output[key]; max_abs = float((actual - expected).abs().max().item()); mae = float((actual - expected).abs().mean().item()); close = bool(torch.allclose(actual, expected, rtol=1.0e-6, atol=FORWARD_ATOL)); output_diffs[key] = {"max_abs_diff": max_abs, "mae": mae, "close": close}; update_check(current_index, close, f"{key} differs from standard (max_abs_diff={max_abs})"); current_index += 1
        cpu_fixture = _fixture_to_device(fixture, torch.device("cpu")); candidate_loss, candidate_components = candidate_train_module.compute_materials_loss(candidate_output, cpu_fixture["targets"], cpu_fixture["atom_loss_mask"], candidate_config["training"]["loss_weights"]); reference_loss, reference_components = reference_train_module.compute_materials_loss(reference_output, cpu_fixture["targets"], cpu_fixture["atom_loss_mask"], reference_config["training"]["loss_weights"])
        loss_diff = abs(float(candidate_loss) - float(reference_loss)); update_check(current_index, math.isclose(float(candidate_loss), float(reference_loss), rel_tol=1.0e-6, abs_tol=1.0e-6), f"weighted MaterialsLoss differs from standard by {loss_diff}"); current_index += 1
        for key in ("pos", "cell", "atomic_numbers"):
            actual_component = float(candidate_components[key]); expected_component = float(reference_components[key]); update_check(current_index, math.isclose(actual_component, expected_component, rel_tol=1.0e-6, abs_tol=1.0e-6), f"{key} loss differs from standard"); current_index += 1
        del candidate_model, reference_model, moved, reference_output_device
        if device.type == "cuda": torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["training:train_result", "training:payload", "training:model_state_layout", "training:step", "training:optimizer_state", "training:gemnet_update", "training:atom_head_update", "training:metrics_keys", "training:metrics_finite", "training:loss_positive", "training:gradient_positive", "training:parameters_with_grad"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = caller_locals if isinstance(caller_locals, dict) else {}
        model_module, train_module, config_value = _candidate_modules(context)
        training = config_value["training"]; torch.manual_seed(int(training["seed"])); baseline_model = model_module.build_model(config_value).cpu()
        parameter_names = [name for name, parameter in baseline_model.named_parameters() if parameter.requires_grad]; backbone_names = [name for name in parameter_names if name.startswith("gemnet.")][:8]; head_names = [name for name in parameter_names if name.startswith("fc_atom.")]; selected_names = backbone_names + head_names; baseline_state = {name: dict(baseline_model.named_parameters())[name].detach().cpu().clone() for name in selected_names}; baseline_keys = set(baseline_model.state_dict()); del baseline_model; gc.collect()
        temporary = tempfile.TemporaryDirectory(prefix="mattergen-train-eval-"); output_dir = Path(temporary.name); train_result = train_module.train(CANDIDATE_DIR / "mattergen_train.yaml", output_dir, device="auto"); checkpoint_path = output_dir / "checkpoint.pt"
        if not checkpoint_path.is_file(): raise FileNotFoundError(f"training did not create {checkpoint_path}")
        try: payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception: payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _ADAPTER_STATE.update({"training_temporary": temporary, "training_output_dir": output_dir, "checkpoint_payload": payload, "train_result": train_result})
        update_check(current_index, isinstance(train_result, dict), "train must return a dictionary"); current_index += 1
        update_check(current_index, isinstance(payload, dict), "checkpoint must be a dictionary"); current_index += 1
        trained_state = payload.get("model_state_dict", {}) if isinstance(payload, dict) else {}; optimizer_state = payload.get("optimizer_state_dict", {}) if isinstance(payload, dict) else {}; metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        update_check(current_index, set(trained_state) == baseline_keys, "checkpoint model_state_dict layout is not exact"); current_index += 1
        update_check(current_index, payload.get("step") == 1, "checkpoint step must be 1"); current_index += 1
        update_check(current_index, isinstance(optimizer_state, dict) and bool(optimizer_state.get("state")), "Adam optimizer state must be non-empty"); current_index += 1
        changed = {name: name in trained_state and not torch.equal(baseline_state[name], trained_state[name].detach().cpu()) for name in selected_names}; changed_backbone = [name for name in backbone_names if changed.get(name)]; changed_head = [name for name in head_names if changed.get(name)]
        update_check(current_index, bool(changed_backbone), "no inspected GemNet backbone parameter changed"); current_index += 1
        update_check(current_index, bool(changed_head), "no atom-head parameter changed"); current_index += 1
        required_metrics = {"step", "loss", "pos_loss", "cell_loss", "atomic_numbers_loss", "gradient_norm", "parameters_with_grad"}; update_check(current_index, isinstance(metrics, dict) and required_metrics <= set(metrics), f"metrics must contain {sorted(required_metrics)}"); current_index += 1
        if isinstance(metrics, dict): numeric_values = [metrics.get(key) for key in required_metrics - {"step"}]
        else: numeric_values = []
        update_check(current_index, all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in numeric_values), "training metrics must be finite numbers"); current_index += 1
        update_check(current_index, float(metrics.get("loss", 0.0)) > 0.0 if isinstance(metrics, dict) else False, "loss must be positive"); current_index += 1
        update_check(current_index, float(metrics.get("gradient_norm", 0.0)) > 0.0 if isinstance(metrics, dict) else False, "gradient norm must be positive"); current_index += 1
        update_check(current_index, int(metrics.get("parameters_with_grad", 0)) > 0 if isinstance(metrics, dict) else False, "at least one parameter must receive a gradient"); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["artifacts:checkpoint", "artifacts:metrics", "artifacts:log", "artifacts:log_line_count", "artifacts:metrics_equal_checkpoint", "artifacts:log_equal_checkpoint", "artifacts:state_entries", "artifacts:state_numel", "artifacts:config"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        output_dir: Path = _ADAPTER_STATE["training_output_dir"]; payload = _ADAPTER_STATE["checkpoint_payload"]
        expected_paths = {"checkpoint": output_dir / "checkpoint.pt", "metrics": output_dir / "metrics.json", "log": output_dir / "train.log"}
        for label, path in expected_paths.items():
            update_check(current_index, path.is_file() and path.stat().st_size > 0, f"missing or empty {label} artifact: {path.name}"); current_index += 1
        metrics_json: dict[str, Any] = {}; log_record: dict[str, Any] = {}
        if expected_paths["metrics"].is_file():
            with expected_paths["metrics"].open("r", encoding="utf-8") as handle: metrics_json = json.load(handle)
        if expected_paths["log"].is_file():
            lines = [line for line in expected_paths["log"].read_text(encoding="utf-8").splitlines() if line.strip()]
            update_check(current_index, len(lines) == 1, "train.log must contain exactly one JSON line"); current_index += 1
            if lines: log_record = json.loads(lines[0])
        else:
            update_check(current_index, False, "train.log must contain exactly one JSON line"); current_index += 1
        checkpoint_metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        update_check(current_index, metrics_json == checkpoint_metrics, "metrics.json must match checkpoint metrics"); current_index += 1
        update_check(current_index, log_record == checkpoint_metrics, "train.log record must match checkpoint metrics"); current_index += 1
        state = payload.get("model_state_dict", {}) if isinstance(payload, dict) else {}
        update_check(current_index, len(state) == EXPECTED_STATE_ENTRY_COUNT, f"checkpoint must contain {EXPECTED_STATE_ENTRY_COUNT} model entries"); current_index += 1
        update_check(current_index, sum(value.numel() for value in state.values()) == EXPECTED_STATE_NUMEL, f"checkpoint must contain {EXPECTED_STATE_NUMEL} model elements"); current_index += 1
        update_check(current_index, isinstance(payload.get("config"), dict), "checkpoint must preserve parsed config"); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks
