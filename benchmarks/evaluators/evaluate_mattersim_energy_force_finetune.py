"""V2 adapter for MatterSim energy-force fine-tuning evaluation."""
from pathlib import Path
import sys
import os
import traceback
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


def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check required submission files."""
    CHECKPOINT_NAMES = ["file:required_files"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        REQUIRED_FILES = (
            "mattersim_energy_force_finetune.py", "train_mattersim_finetune.py",
            "mattersim_finetune.yaml", "portable_mattersim/__init__.py",
            "portable_mattersim/model.py", "portable_mattersim/blocks.py",
            "portable_mattersim/layers.py", "portable_mattersim/embeddings.py",
            "portable_mattersim/scaling.py", "portable_mattersim/graph.py",
        )
        missing = [name for name in REQUIRED_FILES if not (CANDIDATE_DIR / name).is_file()]
        update_check(current_index, not missing, f"missing required files: {missing}" if missing else "all required files are present")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())
    return checks


def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check fixed data and graph contracts."""
    CHECKPOINT_NAMES = [
        "data:config_checkpoint_path", "data:config_dataset_path",
        "data:checkpoint_exists", "data:dataset_exists", "data:checkpoint_hash",
        "data:dataset_hash", "data:structure_count", "data:atom_counts",
        "data:batch_shapes", "data:finite_labels",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import importlib.util
        import torch
        INPUT_BATCH_SIZE = 1
        EXPECTED_STRUCTURE_COUNT = 10
        EXPECTED_ATOMS_PER_STRUCTURE = 192
        CHECKPOINT_PATH = Path(os.environ.get("MATTERSIM_CHECKPOINT_PATH", "/public/share/sugonhpcapp01/onestore/onemodels/mattersim/mattersim-v1.0.0-1M.pth")).expanduser()
        DATASET_PATH = Path(os.environ.get("MATTERSIM_FINETUNE_DATA_PATH", "/public/share/sugonhpcapp01/onestore/onedatasets/matchem/mattersim/high_level_water.xyz")).expanduser()
        EXPECTED_CHECKPOINT_SHA256 = "28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be"
        EXPECTED_DATASET_SHA256 = "46c7ad6375ca6596a39fc5c15c8a0a76e7ead90ec74189b370fae3fcc726e779"
        def load_implementation(root, role):
            for name in list(sys.modules):
                if name in {"mattersim_energy_force_finetune", "train_mattersim_finetune", "portable_mattersim"} or name.startswith("portable_mattersim.") or name.startswith("mattersim_finetune_eval_"):
                    sys.modules.pop(name, None)
            for entry in list(sys.path):
                try:
                    if Path(entry).name == "mattersim_energy_force_finetune": sys.path.remove(entry)
                except (OSError, TypeError, ValueError):
                    continue
            sys.path.insert(0, str(root))
            def load_module(module_name, path):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec is None or spec.loader is None: raise ImportError(f"Could not create import spec for {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
            return load_module("mattersim_energy_force_finetune", root / "mattersim_energy_force_finetune.py"), load_module(f"mattersim_finetune_eval_{role}_train", root / "train_mattersim_finetune.py")
        def sha256(path):
            import hashlib
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
            return digest.hexdigest()
        model_module, train_module = load_implementation(CANDIDATE_DIR, "candidate")
        candidate_config = train_module.load_config(CANDIDATE_DIR / "mattersim_finetune.yaml")
        checkpoint = Path(candidate_config.get("checkpoint", "")).expanduser().resolve()
        dataset = Path(candidate_config.get("train_data_path", "")).expanduser().resolve()
        ok = checkpoint == CHECKPOINT_PATH.resolve()
        update_check(current_index, ok, "config checkpoint path is not fixed" if not ok else "config checkpoint path is fixed"); current_index += 1
        ok = dataset == DATASET_PATH.resolve()
        update_check(current_index, ok, "config dataset path is not fixed" if not ok else "config dataset path is fixed"); current_index += 1
        ok = checkpoint.is_file()
        update_check(current_index, ok, "fixed checkpoint does not exist" if not ok else "fixed checkpoint exists"); current_index += 1
        ok = dataset.is_file()
        update_check(current_index, ok, "fixed dataset does not exist" if not ok else "fixed dataset exists"); current_index += 1
        if checkpoint.is_file():
            ok = sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA256
            msg = "checkpoint hash mismatch" if not ok else "checkpoint hash matches"
        else: ok, msg = True, "checkpoint hash not evaluated because checkpoint is absent"
        update_check(current_index, ok, msg); current_index += 1
        if dataset.is_file():
            ok = sha256(dataset) == EXPECTED_DATASET_SHA256
            msg = "dataset hash mismatch" if not ok else "dataset hash matches"
        else: ok, msg = True, "dataset hash not evaluated because dataset is absent"
        update_check(current_index, ok, msg); current_index += 1
        structures = model_module.load_training_structures(dataset)
        ok = len(structures) == EXPECTED_STRUCTURE_COUNT
        update_check(current_index, ok, "dataset must contain ten structures" if not ok else "dataset contains ten structures"); current_index += 1
        atom_counts = [len(atoms) for atoms in structures]
        ok = atom_counts == [EXPECTED_ATOMS_PER_STRUCTURE] * EXPECTED_STRUCTURE_COUNT
        update_check(current_index, ok, "each fixed structure must contain 192 atoms" if not ok else "all structures contain 192 atoms"); current_index += 1
        batch = model_module.build_finetune_batch(dataset, structure_index=0, device="cpu")
        graph = batch.get("graph", {})
        ok = isinstance(graph, dict) and tuple(batch["energy"].shape) == (INPUT_BATCH_SIZE,) and tuple(batch["forces"].shape) == (EXPECTED_ATOMS_PER_STRUCTURE, 3) and tuple(graph["atom_pos"].shape) == (EXPECTED_ATOMS_PER_STRUCTURE, 3) and tuple(graph["cell"].shape) == (INPUT_BATCH_SIZE, 3, 3) and graph["edge_index"].ndim == 2 and graph["edge_index"].shape[0] == 2 and graph["three_body_indices"].ndim == 2 and graph["three_body_indices"].shape[1] == 2
        update_check(current_index, ok, "batch graph or label shapes are incorrect" if not ok else "batch graph and label shapes are correct"); current_index += 1
        ok = bool(torch.isfinite(batch["energy"]).all() and torch.isfinite(batch["forces"]).all())
        update_check(current_index, ok, "energy and force labels must be finite" if not ok else "energy and force labels are finite"); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check source, public API, and training contract."""
    CHECKPOINT_NAMES = [
        "train_script:source_scan", "train_script:weights_only_load", "train_script:strict_load",
        "train_script:model_api", "train_script:training_api", "train_script:tuner_class",
        "train_script:train_signature", "train_script:training_config", "train_script:nn_module",
        "train_script:all_parameters_trainable",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import ast
        import builtins
        import importlib
        import importlib.util
        import inspect
        import torch
        REQUIRED_TUNER_CLASS = "MatterSimEnergyForceFineTuner"
        SEED = 20260819
        class SourceVisitor(ast.NodeVisitor):
            def __init__(self): self.errors, self.has_weights_only_load, self.has_strict_load = [], False, False
            def visit_Import(self, node):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root == "onescience" or alias.name.startswith("mattersim."): self.errors.append(f"forbidden import: {alias.name}")
                self.generic_visit(node)
            def visit_ImportFrom(self, node):
                module = node.module or ""; root = module.split(".", 1)[0]
                if root == "onescience" or module.startswith("mattersim."): self.errors.append(f"forbidden import: {module}")
                self.generic_visit(node)
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__": self.errors.append("dynamic __import__ is forbidden")
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "import_module": self.errors.append("dynamic import_module is forbidden")
                    if node.func.attr == "load": self.has_weights_only_load |= any(keyword.arg == "weights_only" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords)
                    if node.func.attr == "load_state_dict": self.has_strict_load |= any(keyword.arg == "strict" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords)
                self.generic_visit(node)
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    normalized = node.value.replace("\\", "/").lower()
                    if any(fragment in normalized for fragment in ("benchmarks/standard", "benchmarks/evaluators")): self.errors.append("evaluation implementation access is forbidden")
                self.generic_visit(node)
        def scan_sources(submission_dir):
            errors, weights_only, strict = [], False, False
            python_files = sorted(submission_dir.rglob("*.py"))
            if not python_files: errors.append("submission contains no Python source")
            for path in python_files:
                visitor = SourceVisitor()
                try: visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
                except SyntaxError as exc: errors.append(f"{path.relative_to(submission_dir)}: {exc}"); continue
                errors.extend(f"{path.relative_to(submission_dir)}: {error}" for error in visitor.errors)
                weights_only |= visitor.has_weights_only_load; strict |= visitor.has_strict_load
            return {"errors": errors, "weights_only_true": weights_only, "strict_true": strict, "python_files": len(python_files)}
        def load_implementation(root, role):
            for name in list(sys.modules):
                if name in {"mattersim_energy_force_finetune", "train_mattersim_finetune", "portable_mattersim"} or name.startswith("portable_mattersim.") or name.startswith("mattersim_finetune_eval_"): sys.modules.pop(name, None)
            for entry in list(sys.path):
                try:
                    if Path(entry).name == "mattersim_energy_force_finetune": sys.path.remove(entry)
                except (OSError, TypeError, ValueError): continue
            sys.path.insert(0, str(root))
            def load_module(module_name, path):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec is None or spec.loader is None: raise ImportError(f"Could not create import spec for {path}")
                module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
            return load_module("mattersim_energy_force_finetune", root / "mattersim_energy_force_finetune.py"), load_module(f"mattersim_finetune_eval_{role}_train", root / "train_mattersim_finetune.py")
        model_module, train_module = load_implementation(CANDIDATE_DIR, "candidate")
        scan = scan_sources(CANDIDATE_DIR)
        ok = not scan["errors"]
        update_check(current_index, ok, "; ".join(scan["errors"]) if not ok else "source scan passed"); current_index += 1
        ok = scan["weights_only_true"]
        update_check(current_index, ok, "torch.load(..., weights_only=True) is required" if not ok else "weights_only load found"); current_index += 1
        ok = scan["strict_true"]
        update_check(current_index, ok, "load_state_dict(..., strict=True) is required" if not ok else "strict state load found"); current_index += 1
        required_model_api = ("load_checkpoint_payload", "build_model", "load_pretrained_model", "load_training_structures", "build_finetune_batch", "predict_energy_forces", "compute_energy_force_loss")
        required_train_api = ("load_config", "build_model", "build_trainer", "train_step", "train", "main")
        ok = all(callable(getattr(model_module, name, None)) for name in required_model_api)
        update_check(current_index, ok, "model API is incomplete" if not ok else "model API is complete"); current_index += 1
        ok = all(callable(getattr(train_module, name, None)) for name in required_train_api)
        update_check(current_index, ok, "training API is incomplete" if not ok else "training API is complete"); current_index += 1
        tuner_cls = getattr(model_module, REQUIRED_TUNER_CLASS, None)
        ok = inspect.isclass(tuner_cls)
        update_check(current_index, ok, f"{REQUIRED_TUNER_CLASS} must be a class" if not ok else "fine-tuner class is present"); current_index += 1
        signature = inspect.signature(train_module.train)
        ok = all(name in signature.parameters for name in ("config", "output_dir", "device"))
        update_check(current_index, ok, "train signature is incorrect" if not ok else "train signature is correct"); current_index += 1
        candidate_config = train_module.load_config(CANDIDATE_DIR / "mattersim_finetune.yaml")
        training = candidate_config.get("training", {})
        expected_training = {"optimizer": "Adam", "lr": 2.0e-4, "huber_delta": 0.01, "force_loss_ratio": 1.0, "gradient_clip_norm": 1.0, "max_steps": 1, "structure_index": 0, "seed": SEED}
        ok = training == expected_training
        update_check(current_index, ok, "training configuration does not match the fixed contract" if not ok else "training configuration matches the fixed contract"); current_index += 1
        model = train_module.build_model(candidate_config, device="cpu")
        ok = isinstance(model, torch.nn.Module)
        update_check(current_index, ok, "build_model must return nn.Module" if not ok else "build_model returned nn.Module"); current_index += 1
        ok = all(parameter.requires_grad for parameter in model.parameters())
        update_check(current_index, ok, "full fine-tuning requires all parameters trainable" if not ok else "all parameters are trainable"); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks


def check_eval_script_contract(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Record that no standalone evaluation script is required."""
    CHECKPOINT_NAMES = ["eval_script:not_required"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        update_check(current_index, True, "no separate evaluation script is required")
        current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check the local M3GNet architecture."""
    CHECKPOINT_NAMES = [
        "model_structure:nn_module", "model_structure:model_class", "model_structure:child_order",
        "model_structure:main_blocks", "model_structure:block_modules", "model_structure:top_level_types",
        "model_structure:model_args", "model_structure:embedding_basis_shapes", "model_structure:state_envelope",
        "model_structure:local_model_source", "model_structure:public_model_class",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import importlib.util
        import inspect
        import torch
        SEED = 20260819; REQUIRED_MODEL_CLASS = "M3Gnet"
        EXPECTED_MODEL_ARGS = {"num_blocks": 3, "units": 128, "max_l": 4, "max_n": 4, "cutoff": 5.0, "max_z": 94, "threebody_cutoff": 4.0}
        EXPECTED_STATE_ENTRY_COUNT = 89; EXPECTED_STATE_NUMEL = 890_246
        def load_implementation(root, role):
            for name in list(sys.modules):
                if name in {"mattersim_energy_force_finetune", "train_mattersim_finetune", "portable_mattersim"} or name.startswith("portable_mattersim.") or name.startswith("mattersim_finetune_eval_"): sys.modules.pop(name, None)
            for entry in list(sys.path):
                try:
                    if Path(entry).name == "mattersim_energy_force_finetune": sys.path.remove(entry)
                except (OSError, TypeError, ValueError): continue
            sys.path.insert(0, str(root))
            def load_module(module_name, path):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec is None or spec.loader is None: raise ImportError(f"Could not create import spec for {path}")
                module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
            return load_module("mattersim_energy_force_finetune", root / "mattersim_energy_force_finetune.py"), load_module(f"mattersim_finetune_eval_{role}_train", root / "train_mattersim_finetune.py")
        def state_envelope(state): return len(state), sum(value.numel() for value in state.values())
        torch.manual_seed(SEED)
        model_module, train_module = load_implementation(CANDIDATE_DIR, "candidate")
        model = train_module.build_model(train_module.load_config(CANDIDATE_DIR / "mattersim_finetune.yaml"), device="cpu")
        ok = isinstance(model, torch.nn.Module)
        update_check(current_index, ok, "model must be nn.Module" if not ok else "model is nn.Module"); current_index += 1
        ok = type(model).__name__ == REQUIRED_MODEL_CLASS
        update_check(current_index, ok, "model class must be M3Gnet" if not ok else "model class is M3Gnet"); current_index += 1
        children = [name for name, _module in model.named_children()]
        expected_order = ("rbf", "sbf", "edge_encoder", "graph_conv", "final", "atom_embedding", "normalizer")
        present = [children.index(name) for name in expected_order if name in children]
        ok = len(present) == len(expected_order) and present == sorted(present)
        update_check(current_index, ok, "top-level M3GNet module order is incorrect" if not ok else "top-level M3GNet module order is correct"); current_index += 1
        blocks = list(getattr(model, "graph_conv", []))
        ok = len(blocks) == 3 and all(type(block).__name__ == "MainBlock" for block in blocks)
        update_check(current_index, ok, "model must contain three MainBlock modules" if not ok else "model contains three MainBlock modules"); current_index += 1
        block_names = ("three_body", "gated_mlp_edge", "edge_layer_edge", "gated_mlp_atom", "edge_layer_atom")
        ok = all(all(isinstance(getattr(block, name, None), torch.nn.Module) for name in block_names) for block in blocks)
        update_check(current_index, ok, "each MainBlock must implement three-body, edge, and atom updates" if not ok else "MainBlock modules contain required updates"); current_index += 1
        type_contract = {"rbf": "SmoothBesselBasis", "sbf": "SphericalBasisLayer", "edge_encoder": "MLP", "final": "GatedMLP", "atom_embedding": "MLP", "normalizer": "AtomScaling"}
        ok = all(type(getattr(model, name, None)).__name__ == expected for name, expected in type_contract.items())
        update_check(current_index, ok, "top-level module types are incorrect" if not ok else "top-level module types are correct"); current_index += 1
        ok = dict(getattr(model, "model_args", {})) == EXPECTED_MODEL_ARGS
        update_check(current_index, ok, "model_args are incorrect" if not ok else "model_args are correct"); current_index += 1
        ok = tuple(model.atom_embedding.mlp[0].linear.weight.shape) == (128, 95) and tuple(model.edge_encoder.mlp[0].linear.weight.shape) == (128, 4) and tuple(model.sbf.coef.shape) == (4, 9, 4) and "coef" in model.sbf._non_persistent_buffers_set
        update_check(current_index, ok, "embedding or basis shapes are incorrect" if not ok else "embedding and basis shapes are correct"); current_index += 1
        state_entries, state_numel = state_envelope(model.state_dict())
        ok = state_entries == EXPECTED_STATE_ENTRY_COUNT and state_numel == EXPECTED_STATE_NUMEL
        update_check(current_index, ok, "model state envelope is incorrect" if not ok else "model state envelope is correct"); current_index += 1
        model_source = Path(inspect.getfile(model.__class__)).resolve()
        ok = model_source.is_relative_to(CANDIDATE_DIR)
        update_check(current_index, ok, "scored model must be implemented inside the submission" if not ok else "scored model source is inside the submission"); current_index += 1
        ok = model_module.M3Gnet is type(model)
        update_check(current_index, ok, "public M3Gnet class and built model must match" if not ok else "public M3Gnet class matches built model"); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Execute energy-force prediction and loss construction."""
    CHECKPOINT_NAMES = [
        "forward_executable:output_keys", "forward_executable:energy_shape", "forward_executable:force_shape",
        "forward_executable:finite_outputs", "forward_executable:nonzero_outputs", "forward_executable:differentiable_loss",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import importlib.util
        import torch
        SEED = 20260819; INPUT_BATCH_SIZE = 1; EXPECTED_ATOMS_PER_STRUCTURE = 192
        DATASET_PATH = Path(os.environ.get("MATTERSIM_FINETUNE_DATA_PATH", "/public/share/sugonhpcapp01/onestore/onedatasets/matchem/mattersim/high_level_water.xyz")).expanduser()
        def load_implementation(root, role):
            for name in list(sys.modules):
                if name in {"mattersim_energy_force_finetune", "train_mattersim_finetune", "portable_mattersim"} or name.startswith("portable_mattersim.") or name.startswith("mattersim_finetune_eval_"): sys.modules.pop(name, None)
            for entry in list(sys.path):
                try:
                    if Path(entry).name == "mattersim_energy_force_finetune": sys.path.remove(entry)
                except (OSError, TypeError, ValueError): continue
            sys.path.insert(0, str(root))
            def load_module(module_name, path):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec is None or spec.loader is None: raise ImportError(f"Could not create import spec for {path}")
                module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
            return load_module("mattersim_energy_force_finetune", root / "mattersim_energy_force_finetune.py"), load_module(f"mattersim_finetune_eval_{role}_train", root / "train_mattersim_finetune.py")
        torch.manual_seed(SEED)
        model_module, train_module = load_implementation(CANDIDATE_DIR, "candidate")
        requested = os.environ.get("MATTERSIM_FINETUNE_EVAL_DEVICE", "auto")
        if requested == "auto": requested = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
        model = train_module.build_model(train_module.load_config(CANDIDATE_DIR / "mattersim_finetune.yaml"), device=device)
        batch = model_module.build_finetune_batch(DATASET_PATH, structure_index=0, device=device)
        output = model_module.predict_energy_forces(model, batch, create_graph=True)
        ok = isinstance(output, dict) and set(output) == {"energy", "forces"}
        update_check(current_index, ok, "forward output keys must be energy and forces" if not ok else "forward output keys are correct"); current_index += 1
        ok = tuple(output["energy"].shape) == (INPUT_BATCH_SIZE,)
        update_check(current_index, ok, "energy output shape must be [1]" if not ok else "energy output shape is [1]"); current_index += 1
        ok = tuple(output["forces"].shape) == (EXPECTED_ATOMS_PER_STRUCTURE, 3)
        update_check(current_index, ok, "force output shape must be [192,3]" if not ok else "force output shape is [192,3]"); current_index += 1
        finite = bool(torch.isfinite(output["energy"]).all() and torch.isfinite(output["forces"]).all())
        update_check(current_index, finite, "forward outputs must be finite" if not finite else "forward outputs are finite"); current_index += 1
        nontrivial = bool(torch.count_nonzero(output["energy"]).item() and torch.count_nonzero(output["forces"]).item())
        update_check(current_index, nontrivial, "forward outputs must not be all zero" if not nontrivial else "forward outputs are nonzero"); current_index += 1
        loss, components = model_module.compute_energy_force_loss(output, batch, 1.0, 0.01)
        ok = bool(torch.isfinite(loss)) and loss.requires_grad
        update_check(current_index, ok, "energy-force Huber loss must be finite and differentiable" if not ok else "energy-force Huber loss is finite and differentiable"); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks


def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Compare fixed-checkpoint CPU predictions with the reference."""
    CHECKPOINT_NAMES = ["forward_equivalence:state", "forward_equivalence:energy", "forward_equivalence:forces"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import importlib.util
        import torch
        SEED = 20260819; FORWARD_COMPARE_ATOL = 1.0e-6
        DATASET_PATH = Path(os.environ.get("MATTERSIM_FINETUNE_DATA_PATH", "/public/share/sugonhpcapp01/onestore/onedatasets/matchem/mattersim/high_level_water.xyz")).expanduser()
        def load_implementation(root, role):
            for name in list(sys.modules):
                if name in {"mattersim_energy_force_finetune", "train_mattersim_finetune", "portable_mattersim"} or name.startswith("portable_mattersim.") or name.startswith("mattersim_finetune_eval_"): sys.modules.pop(name, None)
            for entry in list(sys.path):
                try:
                    if Path(entry).name == "mattersim_energy_force_finetune": sys.path.remove(entry)
                except (OSError, TypeError, ValueError): continue
            sys.path.insert(0, str(root))
            def load_module(module_name, path):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec is None or spec.loader is None: raise ImportError(f"Could not create import spec for {path}")
                module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
            return load_module("mattersim_energy_force_finetune", root / "mattersim_energy_force_finetune.py"), load_module(f"mattersim_finetune_eval_{role}_train", root / "train_mattersim_finetune.py")
        candidate_model_module, candidate_train = load_implementation(CANDIDATE_DIR, "candidate")
        standard_model_module, standard_train = load_implementation(REFERENCE_DIR, "standard")
        device = torch.device("cpu")
        previous_threads = torch.get_num_threads(); previous_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
            candidate_config = candidate_train.load_config(CANDIDATE_DIR / "mattersim_finetune.yaml")
            torch.manual_seed(SEED)
            candidate_model = candidate_train.build_model(candidate_config, device=device)
            candidate_batch = candidate_model_module.build_finetune_batch(DATASET_PATH, 0, device)
            torch.manual_seed(SEED)
            standard_model = standard_train.build_model(standard_train.load_config(REFERENCE_DIR / "mattersim_finetune.yaml"), device=device)
            standard_batch = standard_model_module.build_finetune_batch(DATASET_PATH, 0, device)
            torch.manual_seed(SEED)
            standard_output = standard_model_module.predict_energy_forces(standard_model, standard_batch, create_graph=False)
            torch.manual_seed(SEED)
            candidate_output = candidate_model_module.predict_energy_forces(candidate_model, candidate_batch, create_graph=False)
        finally:
            torch.use_deterministic_algorithms(previous_deterministic); torch.set_num_threads(previous_threads)
        candidate_state = candidate_model.state_dict(); standard_state = standard_model.state_dict()
        state_keys = set(candidate_state) == set(standard_state)
        state_shapes = state_keys and all(candidate_state[name].shape == standard_state[name].shape for name in standard_state)
        state_values = state_shapes and all(torch.equal(candidate_state[name].detach().cpu(), standard_state[name].detach().cpu()) for name in standard_state)
        ok = state_keys and state_shapes and state_values
        update_check(current_index, ok, "candidate and standard must hold identical fixed checkpoint state" if not ok else "candidate and standard fixed checkpoint states are identical"); current_index += 1
        metrics = {}
        for key in ("energy", "forces"):
            actual = candidate_output[key]; expected = standard_output[key]
            same_shape = actual.shape == expected.shape
            max_abs = float((actual.detach() - expected.detach()).abs().max().cpu()) if same_shape else float("inf")
            metrics[f"{key}_max_abs_error"] = max_abs
            ok = same_shape and max_abs <= FORWARD_COMPARE_ATOL
            update_check(current_index, ok, f"{key} does not match standard forward" if not ok else f"{key} matches standard forward")
            current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks


def check_training(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Run and compare one full fine-tuning step."""
    CHECKPOINT_NAMES = [
        "training:candidate_result_mapping", "training:reference_result_mapping", "training:candidate_result_keys",
        "training:single_step_history", "training:finite_candidate_metrics", "training:loss_equivalence",
        "training:energy_loss_equivalence", "training:force_loss_equivalence", "training:grad_norm_equivalence",
        "training:parameter_names", "training:parameter_equivalence", "training:all_parameter_updates",
        "training:parameters_trainable",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import importlib.util
        import math
        import tempfile
        import torch
        SEED = 20260819; FORWARD_COMPARE_ATOL = 1.0e-6
        def load_implementation(root, role):
            for name in list(sys.modules):
                if name in {"mattersim_energy_force_finetune", "train_mattersim_finetune", "portable_mattersim"} or name.startswith("portable_mattersim.") or name.startswith("mattersim_finetune_eval_"): sys.modules.pop(name, None)
            for entry in list(sys.path):
                try:
                    if Path(entry).name == "mattersim_energy_force_finetune": sys.path.remove(entry)
                except (OSError, TypeError, ValueError): continue
            sys.path.insert(0, str(root))
            def load_module(module_name, path):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec is None or spec.loader is None: raise ImportError(f"Could not create import spec for {path}")
                module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
            return load_module("mattersim_energy_force_finetune", root / "mattersim_energy_force_finetune.py"), load_module(f"mattersim_finetune_eval_{role}_train", root / "train_mattersim_finetune.py")
        def finite_record(record):
            required = {"step", "loss", "energy_loss", "force_loss", "grad_norm"}
            return isinstance(record, dict) and required.issubset(record) and all(math.isfinite(float(record[key])) for key in required)
        context = caller_locals if isinstance(caller_locals, dict) else caller_globals if isinstance(caller_globals, dict) else {}
        _candidate_model_module, candidate_train = load_implementation(CANDIDATE_DIR, "candidate")
        _standard_model_module, standard_train = load_implementation(REFERENCE_DIR, "standard")
        requested = os.environ.get("MATTERSIM_FINETUNE_EVAL_DEVICE", "auto")
        if requested == "auto": requested = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
        candidate_config = candidate_train.load_config(CANDIDATE_DIR / "mattersim_finetune.yaml")
        standard_config = standard_train.load_config(REFERENCE_DIR / "mattersim_finetune.yaml")
        initial_model = candidate_train.build_model(candidate_config, device="cpu")
        initial_state = {name: value.detach().clone() for name, value in initial_model.named_parameters()}
        temporary = tempfile.TemporaryDirectory(prefix="mattersim-finetune-eval-")
        context["training_temporary"] = temporary
        work_root = Path(temporary.name)
        torch.manual_seed(SEED)
        candidate_result = candidate_train.train(candidate_config, work_root / "candidate", device=device)
        torch.manual_seed(SEED)
        standard_result = standard_train.train(standard_config, work_root / "standard", device=device)
        ok = isinstance(candidate_result, dict)
        update_check(current_index, ok, "candidate train must return a mapping" if not ok else "candidate train returned a mapping"); current_index += 1
        ok = isinstance(standard_result, dict)
        update_check(current_index, ok, "standard train must return a mapping" if not ok else "standard train returned a mapping"); current_index += 1
        required_keys = {"model", "optimizer", "history", "checkpoint_path", "metrics_path", "log_path"}
        ok = required_keys.issubset(candidate_result)
        update_check(current_index, ok, "candidate train result is incomplete" if not ok else "candidate train result contains required keys"); current_index += 1
        candidate_history = candidate_result.get("history", []); standard_history = standard_result.get("history", [])
        ok = len(candidate_history) == len(standard_history) == 1
        update_check(current_index, ok, "minimal run must contain one step" if not ok else "candidate and standard runs contain one step"); current_index += 1
        ok = bool(candidate_history) and finite_record(candidate_history[0])
        update_check(current_index, ok, "candidate metrics must be finite" if not ok else "candidate metrics are finite"); current_index += 1
        metric_diffs = {}
        if candidate_history and standard_history:
            for key in ("loss", "energy_loss", "force_loss", "grad_norm"):
                difference = abs(float(candidate_history[0][key]) - float(standard_history[0][key])); metric_diffs[key] = difference
                ok = difference <= FORWARD_COMPARE_ATOL
                update_check(current_index, ok, f"training metric {key} differs from standard" if not ok else f"training metric {key} matches standard")
                current_index += 1
        else:
            for key in ("loss", "energy_loss", "force_loss", "grad_norm"):
                update_check(current_index, False, f"training metric {key} was not available"); current_index += 1
        candidate_parameters = dict(candidate_result["model"].named_parameters()); standard_parameters = dict(standard_result["model"].named_parameters())
        names_equal = set(candidate_parameters) == set(standard_parameters) == set(initial_state)
        update_check(current_index, names_equal, "trained parameter names must match standard and initial model" if not names_equal else "trained parameter names match standard and initial model"); current_index += 1
        max_model_diff = 0.0; updated = {}
        if names_equal:
            for name, value in candidate_parameters.items():
                max_model_diff = max(max_model_diff, float((value.detach().cpu() - standard_parameters[name].detach().cpu()).abs().max()))
                updated[name] = float((value.detach().cpu() - initial_state[name]).abs().max())
        ok = max_model_diff <= FORWARD_COMPARE_ATOL
        update_check(current_index, ok, "trained parameters differ from standard" if not ok else "trained parameters match standard"); current_index += 1
        not_updated = sorted(name for name, difference in updated.items() if difference == 0.0)
        ok = bool(updated) and not not_updated
        update_check(current_index, ok, f"all full-fine-tuning parameter tensors must update: {not_updated}" if not ok else "all full-fine-tuning parameter tensors updated"); current_index += 1
        ok = all(parameter.requires_grad for parameter in candidate_parameters.values())
        update_check(current_index, ok, "all model parameters must remain trainable" if not ok else "all model parameters remain trainable"); current_index += 1
        context["training_result"] = candidate_result
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check persisted fine-tuning artifacts and strict reloadability."""
    CHECKPOINT_NAMES = [
        "artifacts:checkpoint_path", "artifacts:metrics_path", "artifacts:log_path", "artifacts:files",
        "artifacts:checkpoint_schema", "artifacts:checkpoint_identity", "artifacts:model_args",
        "artifacts:state_envelope", "artifacts:strict_reload", "artifacts:trainable_names",
        "artifacts:completed_steps", "artifacts:optimizer_state", "artifacts:history_jsonl",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import importlib.util
        import json
        import torch
        EXPECTED_MODEL_ARGS = {"num_blocks": 3, "units": 128, "max_l": 4, "max_n": 4, "cutoff": 5.0, "max_z": 94, "threebody_cutoff": 4.0}
        EXPECTED_STATE_ENTRY_COUNT = 89; EXPECTED_STATE_NUMEL = 890_246
        ARTIFACT_FORMAT = "mattersim-1m-energy-force-finetune-v1"; CHECKPOINT_NAME = "mattersim_finetuned.pth"; METRICS_NAME = "metrics.jsonl"; LOG_NAME = "train.log"
        def load_implementation(root, role):
            for name in list(sys.modules):
                if name in {"mattersim_energy_force_finetune", "train_mattersim_finetune", "portable_mattersim"} or name.startswith("portable_mattersim.") or name.startswith("mattersim_finetune_eval_"): sys.modules.pop(name, None)
            for entry in list(sys.path):
                try:
                    if Path(entry).name == "mattersim_energy_force_finetune": sys.path.remove(entry)
                except (OSError, TypeError, ValueError): continue
            sys.path.insert(0, str(root))
            def load_module(module_name, path):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec is None or spec.loader is None: raise ImportError(f"Could not create import spec for {path}")
                module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module; spec.loader.exec_module(module); return module
            return load_module("mattersim_energy_force_finetune", root / "mattersim_energy_force_finetune.py"), load_module(f"mattersim_finetune_eval_{role}_train", root / "train_mattersim_finetune.py")
        def state_envelope(state): return len(state), sum(value.numel() for value in state.values())
        def read_jsonl(path): return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        context = caller_locals if isinstance(caller_locals, dict) else caller_globals if isinstance(caller_globals, dict) else {}
        if "training_result" not in context or "training_temporary" not in context:
            check_training(config, caller_globals=caller_globals, caller_locals=context)
        model_module, _train_module = load_implementation(CANDIDATE_DIR, "candidate")
        result = context["training_result"]
        checkpoint = Path(result["checkpoint_path"]); metrics = Path(result["metrics_path"]); log = Path(result["log_path"])
        expected_root = Path(context["training_temporary"].name) / "candidate"
        ok = checkpoint.resolve() == (expected_root / CHECKPOINT_NAME).resolve()
        update_check(current_index, ok, "checkpoint path is incorrect" if not ok else "checkpoint path is correct"); current_index += 1
        ok = metrics.resolve() == (expected_root / METRICS_NAME).resolve()
        update_check(current_index, ok, "metrics path is incorrect" if not ok else "metrics path is correct"); current_index += 1
        ok = log.resolve() == (expected_root / LOG_NAME).resolve()
        update_check(current_index, ok, "log path is incorrect" if not ok else "log path is correct"); current_index += 1
        artifacts_exist = checkpoint.is_file() and metrics.is_file() and log.is_file()
        update_check(current_index, artifacts_exist, "required training artifacts are missing" if not artifacts_exist else "required training artifacts are present"); current_index += 1
        if not artifacts_exist:
            current_index = len(CHECKPOINT_NAMES)
        else:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            required_keys = {"format", "model_name", "model", "model_args", "optimizer", "completed_steps", "trainable_parameter_names", "config", "history"}
            ok = isinstance(payload, dict) and set(payload) == required_keys
            update_check(current_index, ok, "artifact checkpoint schema is not exact" if not ok else "artifact checkpoint schema is exact"); current_index += 1
            ok = payload.get("format") == ARTIFACT_FORMAT and payload.get("model_name") == "m3gnet"
            update_check(current_index, ok, "artifact identity is incorrect" if not ok else "artifact identity is correct"); current_index += 1
            ok = payload.get("model_args") == EXPECTED_MODEL_ARGS
            update_check(current_index, ok, "artifact model_args are incorrect" if not ok else "artifact model_args are correct"); current_index += 1
            state = payload.get("model", {}) if isinstance(payload, dict) else {}
            entries, numel = state_envelope(state) if isinstance(state, dict) else (0, 0)
            ok = entries == EXPECTED_STATE_ENTRY_COUNT and numel == EXPECTED_STATE_NUMEL
            update_check(current_index, ok, "artifact state envelope is incorrect" if not ok else "artifact state envelope is correct"); current_index += 1
            reload_model = model_module.build_model(EXPECTED_MODEL_ARGS, device="cpu")
            incompatible = reload_model.load_state_dict(state, strict=True)
            ok = not incompatible.missing_keys and not incompatible.unexpected_keys
            update_check(current_index, ok, "artifact must strict reload" if not ok else "artifact strict reload succeeded"); current_index += 1
            trainable_names = {name for name, parameter in reload_model.named_parameters() if parameter.requires_grad}
            ok = set(payload.get("trainable_parameter_names", [])) == trainable_names
            update_check(current_index, ok, "artifact trainable names are incorrect" if not ok else "artifact trainable names are correct"); current_index += 1
            ok = int(payload.get("completed_steps", -1)) == 1
            update_check(current_index, ok, "artifact completed_steps must be one" if not ok else "artifact completed_steps is one"); current_index += 1
            optimizer = payload.get("optimizer", {})
            ok = isinstance(optimizer, dict) and bool(optimizer.get("state"))
            update_check(current_index, ok, "artifact optimizer state must be non-empty" if not ok else "artifact optimizer state is non-empty"); current_index += 1
            metric_records = read_jsonl(metrics); log_records = read_jsonl(log); history = payload.get("history", [])
            ok = metric_records == log_records == history and len(history) == 1
            update_check(current_index, ok, "artifact logs must exactly match checkpoint history" if not ok else "artifact logs exactly match checkpoint history"); current_index += 1
    except Exception:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks
