"""V2 adapter for the legacy MatterSim inference evaluator."""
from __future__ import annotations

import ast
import builtins
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

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
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["required_mattersim_artifacts"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        config = config or {}
        REQUIRED_FILES = tuple(config.get("required_files", (
            "mattersim.py",
            "run_inference.py",
            "mattersim_inference.yaml",
            "portable_mattersim/__init__.py",
            "portable_mattersim/model.py",
            "portable_mattersim/blocks.py",
            "portable_mattersim/layers.py",
            "portable_mattersim/embeddings.py",
            "portable_mattersim/scaling.py",
            "portable_mattersim/graph.py",
        )))
        submission_dir = CANDIDATE_DIR
        missing = [
            relative
            for relative in REQUIRED_FILES
            if not (submission_dir / relative).is_file()
        ]
        ok = not missing
        update_check(current_index, ok, f"required={len(REQUIRED_FILES)} missing={missing}")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "checkpoint_exists",
        "checkpoint_sha256",
        "source_weights_only_load",
        "source_strict_load",
        "checkpoint_payload_contract",
        "checkpoint_state_envelope",
        "required_loader_functions",
        "restored_state_layout",
        "restored_state_values",
        "model_source_within_candidate",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        config = config or {}
        checkpoint_config = config.get("checkpoint", {})
        CHECKPOINT_PATH = Path(os.environ.get(
            checkpoint_config.get("path_environment_variable", "MATTERSIM_CHECKPOINT_PATH"),
            checkpoint_config.get(
                "default_path",
                "/public/share/sugonhpcapp01/onestore/onemodels/mattersim/"
                "mattersim-v1.0.0-1M.pth",
            ),
        )).expanduser()
        EXPECTED_CHECKPOINT_SHA256 = checkpoint_config.get(
            "sha256", "28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be"
        )
        EXPECTED_MODEL_ARGS = config.get("expected_model_args", {
            "num_blocks": 3,
            "units": 128,
            "max_l": 4,
            "max_n": 4,
            "cutoff": 5.0,
            "max_z": 94,
            "threebody_cutoff": 4.0,
        })
        EXPECTED_STATE_ENTRY_COUNT = config.get("expected_state_entry_count", 89)
        EXPECTED_STATE_NUMEL = config.get("expected_state_numel", 890_246)
        context = None
        for scope in (caller_locals, caller_globals):
            if isinstance(scope, dict) and isinstance(scope.get("context"), dict):
                context = scope["context"]
                break
        if context is None:
            context = caller_globals if isinstance(caller_globals, dict) else {}

        def _sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        class _SourceVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.errors: list[str] = []
                self.has_weights_only_load = False
                self.has_strict_load = False

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root == "onescience" or alias.name.startswith("mattersim."):
                        self.errors.append(f"forbidden import: {alias.name}")
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                module = node.module or ""
                root = module.split(".", 1)[0]
                if root == "onescience" or module.startswith("mattersim."):
                    self.errors.append(f"forbidden import: {module}")
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    self.errors.append("dynamic __import__ is forbidden")
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "import_module":
                        self.errors.append("dynamic import_module is forbidden")
                    if node.func.attr == "load":
                        self.has_weights_only_load |= any(
                            keyword.arg == "weights_only"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                            for keyword in node.keywords
                        )
                    if node.func.attr == "load_state_dict":
                        self.has_strict_load |= any(
                            keyword.arg == "strict"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                            for keyword in node.keywords
                        )
                self.generic_visit(node)

            def visit_Attribute(self, node: ast.Attribute) -> None:
                if (
                    node.attr == "modules"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "sys"
                ):
                    self.errors.append("sys.modules source aliases are forbidden")
                self.generic_visit(node)

            def visit_Constant(self, node: ast.Constant) -> None:
                if isinstance(node.value, str):
                    normalized = node.value.replace("\\", "/").lower()
                    if (
                        "benchmarks/standard" in normalized
                        or "benchmarks/evaluators" in normalized
                    ):
                        self.errors.append("evaluation implementation access is forbidden")
                self.generic_visit(node)

        def _scan_submission_sources(submission_dir: Path) -> tuple[bool, bool]:
            errors: list[str] = []
            has_weights_only_load = False
            has_strict_load = False
            python_files = sorted(submission_dir.rglob("*.py"))
            if not python_files:
                raise AssertionError("submission contains no Python source files")
            for path in python_files:
                visitor = _SourceVisitor()
                visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
                errors.extend(
                    f"{path.relative_to(submission_dir)}: {error}"
                    for error in visitor.errors
                )
                has_weights_only_load |= visitor.has_weights_only_load
                has_strict_load |= visitor.has_strict_load
            if errors:
                raise AssertionError("; ".join(errors))
            return has_weights_only_load, has_strict_load

        def _install_import_guard() -> None:
            if getattr(_install_import_guard, "installed", False):
                return
            original_import = builtins.__import__
            original_import_module = importlib.import_module

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name.split(".", 1)[0] == "onescience" or name.startswith("mattersim."):
                    raise ImportError(f"forbidden runtime import attempted: {name}")
                return original_import(name, globals, locals, fromlist, level)

            def guarded_import_module(name: str, package: str | None = None):
                if name.split(".", 1)[0] == "onescience" or name.startswith("mattersim."):
                    raise ImportError(f"forbidden runtime import attempted: {name}")
                return original_import_module(name, package)

            builtins.__import__ = guarded_import
            importlib.import_module = guarded_import_module
            _install_import_guard.installed = True

        def load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def _purge_mattersim_modules() -> None:
            for name in list(sys.modules):
                if (
                    name in {"mattersim", "run_inference", "portable_mattersim"}
                    or name.startswith("portable_mattersim.")
                    or name.startswith("mattersim_eval_loaded_")
                ):
                    sys.modules.pop(name, None)

        def load_implementation(root: Path, role: str):
            _purge_mattersim_modules()
            for entry in list(sys.path):
                try:
                    if Path(entry).name == "mattersim_inference":
                        sys.path.remove(entry)
                except (OSError, TypeError, ValueError):
                    continue
            sys.path.insert(0, str(root))
            model_module = load_module_from_path("mattersim", root / "mattersim.py")
            run_module = load_module_from_path(
                f"mattersim_eval_loaded_{role}_run", root / "run_inference.py"
            )
            return model_module, run_module

        def _state_envelope(state: dict[str, Any]) -> tuple[int, int]:
            return len(state), sum(value.numel() for value in state.values())

        diagnostics: list[str] = []
        checkpoint = CHECKPOINT_PATH.resolve()
        checkpoint_exists = checkpoint.is_file()
        diagnostics.append(f"checkpoint_exists={checkpoint_exists}")
        update_check(current_index, checkpoint_exists, diagnostics[-1])
        current_index += 1
        if checkpoint_exists:
            actual_hash = _sha256(checkpoint)
            diagnostics.append(f"sha256={actual_hash}")
            hash_ok = actual_hash == EXPECTED_CHECKPOINT_SHA256
            update_check(current_index, hash_ok, diagnostics[-1])
            current_index += 1

            has_weights_only_load, has_strict_load = _scan_submission_sources(CANDIDATE_DIR)
            diagnostics.append(f"weights_only_true={has_weights_only_load}")
            update_check(current_index, has_weights_only_load, diagnostics[-1])
            current_index += 1
            diagnostics.append(f"strict_true={has_strict_load}")
            update_check(current_index, has_strict_load, diagnostics[-1])
            current_index += 1

            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            required_top_keys = {
                "model_name", "model", "model_args", "optimizer", "ema", "scheduler",
                "last_epoch", "validation_metrics", "description",
            }
            payload_ok = (
                isinstance(payload, dict)
                and set(payload) == required_top_keys
                and payload.get("model_name") == "m3gnet"
                and payload.get("model_args") == EXPECTED_MODEL_ARGS
                and isinstance(payload.get("model"), dict)
            )
            update_check(current_index, payload_ok, f"payload_contract={payload_ok}")
            current_index += 1
            state = payload["model"] if payload_ok else {}
            entries, numel = _state_envelope(state)
            envelope_ok = entries == EXPECTED_STATE_ENTRY_COUNT and numel == EXPECTED_STATE_NUMEL
            diagnostics.append(f"state_entries={entries} state_numel={numel}")
            update_check(current_index, envelope_ok, diagnostics[-1])
            current_index += 1

            _install_import_guard()
            model_module, run_module = load_implementation(CANDIDATE_DIR, "candidate")
            required_functions = tuple(config.get("required_model_functions", (
                "load_checkpoint_payload", "build_model", "load_model", "evaluate_structure",
            )))
            functions_ok = all(callable(getattr(model_module, name, None)) for name in required_functions)
            diagnostics.append(f"loader_functions={functions_ok}")
            update_check(current_index, functions_ok, diagnostics[-1])
            current_index += 1
            if not functions_ok:
                context.update({"candidate_model_module": model_module, "candidate_run_module": run_module})
            else:
                model = model_module.load_model(checkpoint, device="cpu")
                loaded_state = model.state_dict()
                state_layout_ok = set(loaded_state) == set(state)
                update_check(current_index, state_layout_ok, f"state_layout_exact={state_layout_ok}")
                current_index += 1
                exact_values = state_layout_ok and all(
                    torch.equal(loaded_state[key].detach().cpu(), state[key].detach().cpu())
                    for key in state
                )
                update_check(current_index, exact_values, f"state_values_exact={exact_values}")
                current_index += 1
                model_source = Path(inspect.getfile(model.__class__)).resolve()
                local_model_ok = model_source.is_relative_to(CANDIDATE_DIR.resolve())
                update_check(current_index, local_model_ok, f"model_source={model_source}")
                current_index += 1
                context.update({
                    "candidate_model_module": model_module,
                    "candidate_run_module": run_module,
                    "candidate_model": model,
                    "checkpoint_state": state,
                })
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "model_class",
        "top_level_module_order",
        "graph_conv_blocks",
        "main_block_components",
        "top_level_module_types",
        "model_arguments",
        "basis_contract",
        "embedding_tensor_shapes",
        "forward_signature",
        "state_envelope",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import torch

        config = config or {}
        SEED = config.get("seed", 20260804)
        EXPECTED_MODEL_ARGS = config.get("expected_model_args", {
            "num_blocks": 3,
            "units": 128,
            "max_l": 4,
            "max_n": 4,
            "cutoff": 5.0,
            "max_z": 94,
            "threebody_cutoff": 4.0,
        })
        EXPECTED_STATE_ENTRY_COUNT = config.get("expected_state_entry_count", 89)
        EXPECTED_STATE_NUMEL = config.get("expected_state_numel", 890_246)
        context = None
        for scope in (caller_locals, caller_globals):
            if isinstance(scope, dict) and isinstance(scope.get("context"), dict):
                context = scope["context"]
                break
        if context is None:
            context = caller_globals if isinstance(caller_globals, dict) else {}

        def _state_envelope(state: dict[str, Any]) -> tuple[int, int]:
            return len(state), sum(value.numel() for value in state.values())

        torch.manual_seed(SEED)
        model = context["candidate_model"]
        class_ok = isinstance(model, torch.nn.Module) and type(model).__name__ == "M3Gnet"
        update_check(current_index, class_ok, f"class={type(model).__name__}")
        current_index += 1
        children = [name for name, _module in model.named_children()]
        required_order = (
            "rbf", "sbf", "edge_encoder", "graph_conv", "final", "atom_embedding", "normalizer",
        )
        positions = [children.index(name) for name in required_order if name in children]
        order_ok = len(positions) == len(required_order) and positions == sorted(positions)
        update_check(current_index, order_ok, f"children={children}")
        current_index += 1
        module_types = [type(module).__name__ for module in model.graph_conv]
        blocks_ok = len(model.graph_conv) == 3 and module_types == ["MainBlock"] * 3
        update_check(current_index, blocks_ok, f"graph_conv={module_types}")
        current_index += 1
        block_components_ok = all(
            all(isinstance(getattr(block, name, None), torch.nn.Module) for name in (
                "three_body", "gated_mlp_edge", "edge_layer_edge", "gated_mlp_atom", "edge_layer_atom",
            ))
            for block in model.graph_conv
        )
        update_check(current_index, block_components_ok, f"block_components={block_components_ok}")
        current_index += 1
        top_types_ok = (
            type(model.rbf).__name__ == "SmoothBesselBasis"
            and type(model.sbf).__name__ == "SphericalBasisLayer"
            and type(model.edge_encoder).__name__ == "MLP"
            and type(model.final).__name__ == "GatedMLP"
            and type(model.atom_embedding).__name__ == "MLP"
            and type(model.normalizer).__name__ == "AtomScaling"
        )
        update_check(current_index, top_types_ok, f"top_types={top_types_ok}")
        current_index += 1
        args_ok = dict(getattr(model, "model_args", {})) == EXPECTED_MODEL_ARGS
        update_check(current_index, args_ok, f"args={getattr(model, 'model_args', None)}")
        current_index += 1
        basis_ok = (
            int(getattr(model.rbf, "max_n", -1)) == 4
            and int(getattr(model.sbf, "max_n", -1)) == 4
            and int(getattr(model.sbf, "max_l", -1)) == 4
            and float(getattr(model.sbf, "cutoff", -1.0)) == 5.0
            and tuple(getattr(model.sbf, "coef").shape) == (4, 9, 4)
            and "coef" in getattr(model.sbf, "_non_persistent_buffers_set", set())
            and "sbf.coef" not in model.state_dict()
        )
        update_check(current_index, basis_ok, f"basis_ok={basis_ok}")
        current_index += 1
        embedding_ok = (
            tuple(model.atom_embedding.mlp[0].linear.weight.shape) == (128, 95)
            and tuple(model.edge_encoder.mlp[0].linear.weight.shape) == (128, 4)
            and tuple(model.normalizer.scale.shape) == (95,)
            and tuple(model.normalizer.shift.shape) == (95,)
        )
        update_check(current_index, embedding_ok, f"embedding_ok={embedding_ok}")
        current_index += 1
        forward_parameters = list(inspect.signature(model.forward).parameters)
        forward_ok = forward_parameters[:1] == ["input"]
        update_check(current_index, forward_ok, f"forward={inspect.signature(model.forward)}")
        current_index += 1
        state_entries, state_numel = _state_envelope(model.state_dict())
        state_ok = state_entries == EXPECTED_STATE_ENTRY_COUNT and state_numel == EXPECTED_STATE_NUMEL
        update_check(current_index, state_ok, f"state_entries={state_entries}; state_numel={state_numel}")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "required_interface_functions",
        "evaluate_signature",
        "ase_input_loading",
        "inference_config",
        "main_cli_flags",
        "serialized_prediction_records",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import numpy as np
        from ase import Atoms
        from ase.io import write

        config = config or {}
        INPUT_BATCH_SIZE = config.get("input_batch_size", 3)
        checkpoint_config = config.get("checkpoint", {})
        CHECKPOINT_PATH = Path(os.environ.get(
            checkpoint_config.get("path_environment_variable", "MATTERSIM_CHECKPOINT_PATH"),
            checkpoint_config.get(
                "default_path",
                "/public/share/sugonhpcapp01/onestore/onemodels/mattersim/"
                "mattersim-v1.0.0-1M.pth",
            ),
        )).expanduser()
        EXPECTED_MODEL_ARGS = config.get("expected_model_args", {
            "num_blocks": 3,
            "units": 128,
            "max_l": 4,
            "max_n": 4,
            "cutoff": 5.0,
            "max_z": 94,
            "threebody_cutoff": 4.0,
        })
        OUTPUT_KEYS = tuple(config.get("output_keys", ("energies", "forces", "stresses")))
        context = None
        for scope in (caller_locals, caller_globals):
            if isinstance(scope, dict) and isinstance(scope.get("context"), dict):
                context = scope["context"]
                break
        if context is None:
            context = caller_globals if isinstance(caller_globals, dict) else {}

        def _fixed_structures():
            from ase.build import bulk
            return [bulk("Si", "diamond", a=a) for a in (5.40, 5.43, 5.46)]

        run_module = context["candidate_run_module"]
        required_functions = tuple(config.get("required_run_functions", (
            "load_config", "load_structures", "evaluate", "save_predictions", "main",
        )))
        functions_ok = all(callable(getattr(run_module, name, None)) for name in required_functions)
        update_check(current_index, functions_ok, f"functions={functions_ok}")
        current_index += 1
        signature = inspect.signature(run_module.evaluate)
        signature_ok = list(signature.parameters)[:3] == ["structures", "checkpoint", "device"]
        update_check(current_index, signature_ok, f"evaluate_signature={signature}")
        current_index += 1
        structures = _fixed_structures()
        with tempfile.TemporaryDirectory(prefix=config.get("temporary_directory_prefix", "mattersim-interface-")) as temporary:
            temporary_root = Path(temporary)
            input_path = temporary_root / "structures.extxyz"
            output_path = temporary_root / "predictions.json"
            write(input_path, structures, format="extxyz")
            loaded_path = run_module.load_structures(input_path)
            loaded_atoms = run_module.load_structures(structures[0])
            loaded_sequence = run_module.load_structures(structures)
            run_module.save_predictions(output_path, {
                "energies": np.arange(INPUT_BATCH_SIZE, dtype=np.float64),
                "forces": [np.zeros((2, 3)) for _ in structures],
                "stresses": np.zeros((INPUT_BATCH_SIZE, 3, 3)),
            })
            records = json.loads(output_path.read_text(encoding="utf-8"))
        ase_ok = (
            len(loaded_path) == INPUT_BATCH_SIZE
            and len(loaded_atoms) == 1
            and len(loaded_sequence) == INPUT_BATCH_SIZE
            and all(isinstance(item, Atoms) for item in loaded_path + loaded_atoms + loaded_sequence)
        )
        update_check(current_index, ase_ok, f"ase_inputs={ase_ok}")
        current_index += 1
        loaded_config = run_module.load_config(CANDIDATE_DIR / "mattersim_inference.yaml")
        config_ok = (
            Path(loaded_config.get("checkpoint", "")).expanduser().resolve() == CHECKPOINT_PATH.resolve()
            and loaded_config.get("model") == EXPECTED_MODEL_ARGS
            and loaded_config.get("inference", {}).get("outputs") == list(OUTPUT_KEYS)
        )
        update_check(current_index, config_ok, f"config={config_ok}")
        current_index += 1
        main_source = inspect.getsource(run_module.main)
        cli_ok = all(flag in main_source for flag in ("--config", "--checkpoint", "--input", "--output", "--device"))
        update_check(current_index, cli_ok, f"cli={cli_ok}")
        current_index += 1
        output_ok = (
            isinstance(records, list)
            and len(records) == INPUT_BATCH_SIZE
            and all(
                set(record) == {"index", "energy", "forces", "stress"}
                and record["index"] == index
                and np.asarray(record["forces"]).shape == (2, 3)
                and np.asarray(record["stress"]).shape == (3, 3)
                for index, record in enumerate(records)
            )
        )
        update_check(current_index, output_ok, f"output_json={output_ok}")
        current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["output_keys", "output_shapes", "output_finite_values"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import numpy as np
        import torch

        config = config or {}
        INPUT_BATCH_SIZE = config.get("input_batch_size", 3)
        OUTPUT_KEYS = tuple(config.get("output_keys", ("energies", "forces", "stresses")))
        checkpoint_config = config.get("checkpoint", {})
        CHECKPOINT_PATH = Path(os.environ.get(
            checkpoint_config.get("path_environment_variable", "MATTERSIM_CHECKPOINT_PATH"),
            checkpoint_config.get(
                "default_path",
                "/public/share/sugonhpcapp01/onestore/onemodels/mattersim/"
                "mattersim-v1.0.0-1M.pth",
            ),
        )).expanduser()
        SEED = config.get("seed", 20260804)
        context = None
        for scope in (caller_locals, caller_globals):
            if isinstance(scope, dict) and isinstance(scope.get("context"), dict):
                context = scope["context"]
                break
        if context is None:
            context = caller_globals if isinstance(caller_globals, dict) else {}

        def _fixed_structures():
            from ase.build import bulk
            return [bulk("Si", "diamond", a=a) for a in (5.40, 5.43, 5.46)]

        device = os.environ.get(
            config.get("device_environment_variable", "MATTERSIM_EVAL_DEVICE"),
            config.get("device_default", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        torch.manual_seed(SEED)
        run_module = context["candidate_run_module"]
        result = run_module.evaluate(_fixed_structures(), checkpoint=CHECKPOINT_PATH.resolve(), device=device)
        keys_ok = isinstance(result, dict) and set(result) == set(OUTPUT_KEYS)
        detail = f"keys={sorted(result) if isinstance(result, dict) else type(result).__name__}"
        update_check(current_index, keys_ok, detail)
        current_index += 1
        if keys_ok:
            energies = np.asarray(result["energies"])
            stresses = np.asarray(result["stresses"])
            forces = list(result["forces"])
            shape_ok = (
                energies.shape == (INPUT_BATCH_SIZE,)
                and stresses.shape == (INPUT_BATCH_SIZE, 3, 3)
                and len(forces) == INPUT_BATCH_SIZE
                and all(np.asarray(value).shape == (2, 3) for value in forces)
            )
            finite_ok = (
                np.isfinite(energies).all()
                and np.isfinite(stresses).all()
                and all(np.isfinite(np.asarray(value)).all() for value in forces)
            )
            shape_detail = (
                f"energies={energies.shape} forces={[np.asarray(value).shape for value in forces]} "
                f"stresses={stresses.shape}"
            )
            finite_detail = f"finite={bool(finite_ok)}"
        else:
            shape_ok = False
            finite_ok = False
            shape_detail = detail
            finite_detail = detail
        update_check(current_index, bool(shape_ok), shape_detail)
        current_index += 1
        update_check(current_index, bool(finite_ok), finite_detail)
        current_index += 1
        context["candidate_output"] = result
        context["device"] = device
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "candidate_output_keys",
        "candidate_output_shapes",
        "candidate_output_finite_values",
        "reference_output_keys",
        "reference_output_shapes",
        "reference_output_finite_values",
        "energy_mae",
        "force_mae",
        "stress_mae",
        "energy_max_abs_error",
        "force_max_abs_error",
        "stress_max_abs_error",
        "energies_allclose",
        "stresses_allclose",
        "forces_allclose",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        import numpy as np
        import torch

        config = config or {}
        INPUT_BATCH_SIZE = config.get("input_batch_size", 3)
        OUTPUT_KEYS = tuple(config.get("output_keys", ("energies", "forces", "stresses")))
        checkpoint_config = config.get("checkpoint", {})
        CHECKPOINT_PATH = Path(os.environ.get(
            checkpoint_config.get("path_environment_variable", "MATTERSIM_CHECKPOINT_PATH"),
            checkpoint_config.get(
                "default_path",
                "/public/share/sugonhpcapp01/onestore/onemodels/mattersim/"
                "mattersim-v1.0.0-1M.pth",
            ),
        )).expanduser()
        SEED = config.get("seed", 20260804)
        FORWARD_COMPARE_ATOL = config.get("forward_compare_atol", 1.0e-6)
        context = None
        for scope in (caller_locals, caller_globals):
            if isinstance(scope, dict) and isinstance(scope.get("context"), dict):
                context = scope["context"]
                break
        if context is None:
            context = caller_globals if isinstance(caller_globals, dict) else {}

        def _fixed_structures():
            from ase.build import bulk
            return [bulk("Si", "diamond", a=a) for a in (5.40, 5.43, 5.46)]

        def load_module_from_path(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def _purge_mattersim_modules() -> None:
            for name in list(sys.modules):
                if (
                    name in {"mattersim", "run_inference", "portable_mattersim"}
                    or name.startswith("portable_mattersim.")
                    or name.startswith("mattersim_eval_loaded_")
                ):
                    sys.modules.pop(name, None)

        def load_implementation(root: Path, role: str):
            _purge_mattersim_modules()
            for entry in list(sys.path):
                try:
                    if Path(entry).name == "mattersim_inference":
                        sys.path.remove(entry)
                except (OSError, TypeError, ValueError):
                    continue
            sys.path.insert(0, str(root))
            model_module = load_module_from_path("mattersim", root / "mattersim.py")
            run_module = load_module_from_path(
                f"mattersim_eval_loaded_{role}_run", root / "run_inference.py"
            )
            return model_module, run_module

        candidate = context["candidate_output"]
        device = context["device"]
        torch.manual_seed(SEED)
        _model_module, reference_run = load_implementation(REFERENCE_DIR, "standard")
        reference = reference_run.evaluate(_fixed_structures(), checkpoint=CHECKPOINT_PATH.resolve(), device=device)

        candidate_keys_ok = isinstance(candidate, dict) and set(candidate) == set(OUTPUT_KEYS)
        candidate_detail = f"keys={sorted(candidate) if isinstance(candidate, dict) else type(candidate).__name__}"
        update_check(current_index, candidate_keys_ok, candidate_detail)
        current_index += 1
        if candidate_keys_ok:
            candidate_energies = np.asarray(candidate["energies"])
            candidate_stresses = np.asarray(candidate["stresses"])
            candidate_forces = list(candidate["forces"])
            candidate_shape_ok = (
                candidate_energies.shape == (INPUT_BATCH_SIZE,)
                and candidate_stresses.shape == (INPUT_BATCH_SIZE, 3, 3)
                and len(candidate_forces) == INPUT_BATCH_SIZE
                and all(np.asarray(value).shape == (2, 3) for value in candidate_forces)
            )
            candidate_finite_ok = (
                np.isfinite(candidate_energies).all()
                and np.isfinite(candidate_stresses).all()
                and all(np.isfinite(np.asarray(value)).all() for value in candidate_forces)
            )
            candidate_shape_detail = (
                f"energies={candidate_energies.shape} "
                f"forces={[np.asarray(value).shape for value in candidate_forces]} "
                f"stresses={candidate_stresses.shape}"
            )
            candidate_finite_detail = f"finite={bool(candidate_finite_ok)}"
        else:
            candidate_shape_ok = False
            candidate_finite_ok = False
            candidate_shape_detail = candidate_detail
            candidate_finite_detail = candidate_detail
        update_check(current_index, bool(candidate_shape_ok), candidate_shape_detail)
        current_index += 1
        update_check(current_index, bool(candidate_finite_ok), candidate_finite_detail)
        current_index += 1

        reference_keys_ok = isinstance(reference, dict) and set(reference) == set(OUTPUT_KEYS)
        reference_detail = f"keys={sorted(reference) if isinstance(reference, dict) else type(reference).__name__}"
        update_check(current_index, reference_keys_ok, reference_detail)
        current_index += 1
        if reference_keys_ok:
            reference_energies = np.asarray(reference["energies"])
            reference_stresses = np.asarray(reference["stresses"])
            reference_forces = list(reference["forces"])
            reference_shape_ok = (
                reference_energies.shape == (INPUT_BATCH_SIZE,)
                and reference_stresses.shape == (INPUT_BATCH_SIZE, 3, 3)
                and len(reference_forces) == INPUT_BATCH_SIZE
                and all(np.asarray(value).shape == (2, 3) for value in reference_forces)
            )
            reference_finite_ok = (
                np.isfinite(reference_energies).all()
                and np.isfinite(reference_stresses).all()
                and all(np.isfinite(np.asarray(value)).all() for value in reference_forces)
            )
            reference_shape_detail = (
                f"energies={reference_energies.shape} "
                f"forces={[np.asarray(value).shape for value in reference_forces]} "
                f"stresses={reference_stresses.shape}"
            )
            reference_finite_detail = f"finite={bool(reference_finite_ok)}"
        else:
            reference_shape_ok = False
            reference_finite_ok = False
            reference_shape_detail = reference_detail
            reference_finite_detail = reference_detail
        update_check(current_index, bool(reference_shape_ok), reference_shape_detail)
        current_index += 1
        update_check(current_index, bool(reference_finite_ok), reference_finite_detail)
        current_index += 1

        candidate_ok = candidate_keys_ok and candidate_shape_ok and candidate_finite_ok
        reference_ok = reference_keys_ok and reference_shape_ok and reference_finite_ok
        if candidate_ok and reference_ok:
            energy_diff = np.abs(
                np.asarray(candidate["energies"], dtype=np.float64)
                - np.asarray(reference["energies"], dtype=np.float64)
            )
            force_diff = np.concatenate([
                np.abs(
                    np.asarray(actual, dtype=np.float64)
                    - np.asarray(expected, dtype=np.float64)
                ).reshape(-1)
                for actual, expected in zip(candidate["forces"], reference["forces"], strict=True)
            ])
            stress_diff = np.abs(
                np.asarray(candidate["stresses"], dtype=np.float64)
                - np.asarray(reference["stresses"], dtype=np.float64)
            )
            metrics = {
                "energy_mae": float(energy_diff.mean()),
                "force_mae": float(force_diff.mean()),
                "stress_mae": float(stress_diff.mean()),
                "energy_max_abs_error": float(energy_diff.max()),
                "force_max_abs_error": float(force_diff.max()),
                "stress_max_abs_error": float(stress_diff.max()),
            }
            context["metrics"] = metrics
            for metric_name in (
                "energy_mae", "force_mae", "stress_mae",
                "energy_max_abs_error", "force_max_abs_error", "stress_max_abs_error",
            ):
                update_check(current_index, True, f"{metric_name}={metrics[metric_name]}")
                current_index += 1
            energies_ok = np.allclose(
                np.asarray(candidate["energies"], dtype=np.float64),
                np.asarray(reference["energies"], dtype=np.float64),
                rtol=1.0e-6,
                atol=FORWARD_COMPARE_ATOL,
            )
            stresses_ok = np.allclose(
                np.asarray(candidate["stresses"], dtype=np.float64),
                np.asarray(reference["stresses"], dtype=np.float64),
                rtol=1.0e-6,
                atol=FORWARD_COMPARE_ATOL,
            )
            forces_ok = all(
                np.allclose(
                    np.asarray(actual, dtype=np.float64),
                    np.asarray(expected, dtype=np.float64),
                    rtol=1.0e-6,
                    atol=FORWARD_COMPARE_ATOL,
                )
                for actual, expected in zip(candidate["forces"], reference["forces"], strict=True)
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            update_check(current_index, bool(energies_ok), f"atol={FORWARD_COMPARE_ATOL}; rtol=1e-6")
            current_index += 1
            update_check(current_index, bool(stresses_ok), f"atol={FORWARD_COMPARE_ATOL}; rtol=1e-6")
            current_index += 1
            update_check(current_index, bool(forces_ok), f"atol={FORWARD_COMPARE_ATOL}; rtol=1e-6")
            current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())

    return checks
