'''V2 adapter for MatterGen inference evaluator.'''
from __future__ import annotations

import ast
import gc
import hashlib
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


REQUIRED_FILES = (
    "mattergen_denoiser.py",
    "run_inference.py",
    "mattergen_inference.yaml",
    "portable_mattergen/__init__.py",
    "portable_mattergen/gemnet/gemnet-dT.json",
)
INPUT_BATCH_SIZE = 2
SEED = 20260804
FORWARD_COMPARE_ATOL = 1.0e-6
CONFIG_PATH = Path(os.environ.get("MATTERGEN_CONFIG_PATH", "/public/share/sugonhpcapp01/onestore/onemodels/mattergen/mattergen_base/config.yaml")).expanduser()
CHECKPOINT_PATH = Path(os.environ.get("MATTERGEN_CHECKPOINT_PATH", "/public/share/sugonhpcapp01/onestore/onemodels/mattergen/mattergen_base/checkpoints/last.ckpt")).expanduser()
EXPECTED_CONFIG_SHA256 = "01648f40a8227b01b1dbc1090bf0f8229bb3241845ae9ac57af96b31e595bda7"
EXPECTED_CHECKPOINT_SHA256 = "81668ee12afc1ee1b037f362420730de3460bfd2d36e547585fdcb911a3dfdef"
EXPECTED_GEMNET_CONFIG_SHA256 = "867604ed6a605649241cd94d9604f25226fb1e362f332b7d1a9ae40d3f747e90"
CHECKPOINT_PREFIX = "diffusion_module.model."
EXPECTED_STATE_ENTRY_COUNT = 281
EXPECTED_STATE_NUMEL = 53_737_595
OUTPUT_KEYS = ("pos", "cell", "atomic_numbers")
_ADAPTER_CONTEXT: dict[str, Any] = {}

def _ctx(caller_globals, caller_locals):
    for holder in (caller_locals, caller_globals):
        if isinstance(holder, dict) and isinstance(holder.get("_adapter_context"), dict):
            return holder["_adapter_context"]
    return _ADAPTER_CONTEXT

def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _purge_mattergen_modules() -> None:
    for name in list(sys.modules):
        if name in {"mattergen_denoiser", "run_inference"} or name.startswith("portable_mattergen") or name.startswith("mattergen_eval_loaded_"):
            sys.modules.pop(name, None)

def load_implementation(root: Path, role: str):
    _purge_mattergen_modules()
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    model_module = load_module_from_path("mattergen_denoiser", root / "mattergen_denoiser.py")
    run_module = load_module_from_path(f"mattergen_eval_loaded_{role}_run", root / "run_inference.py")
    return model_module, run_module

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

class _ForbiddenSourceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.has_torch_load = False
        self.has_strict_load = False
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".", 1)[0] in {"onescience", "mattergen"}:
                self.errors.append(f"forbidden import: {alias.name}")
        self.generic_visit(node)
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".", 1)[0] in {"onescience", "mattergen"}:
            self.errors.append(f"forbidden import: {node.module}")
        self.generic_visit(node)
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            self.errors.append("dynamic __import__ is forbidden")
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "import_module":
                self.errors.append("dynamic import_module is forbidden")
            if node.func.attr == "load" and isinstance(node.func.value, ast.Name):
                self.has_torch_load |= node.func.value.id == "torch"
            if node.func.attr == "load_state_dict":
                self.has_strict_load |= any(keyword.arg == "strict" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords)
        self.generic_visit(node)
    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            normalized = node.value.replace("\\", "/").lower()
            if "benchmarks/standard" in normalized or "benchmarks/evaluators" in normalized:
                self.errors.append("standard/evaluator source access is forbidden")
        self.generic_visit(node)

def _scan_submission_sources(submission_dir: Path) -> tuple[bool, bool]:
    combined = _ForbiddenSourceVisitor()
    python_files = sorted(submission_dir.rglob("*.py"))
    if not python_files:
        raise AssertionError("submission contains no Python source files")
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _ForbiddenSourceVisitor()
        visitor.visit(tree)
        combined.errors.extend(f"{path.relative_to(submission_dir)}: {error}" for error in visitor.errors)
        combined.has_torch_load |= visitor.has_torch_load
        combined.has_strict_load |= visitor.has_strict_load
    if combined.errors:
        raise AssertionError("; ".join(combined.errors))
    return combined.has_torch_load, combined.has_strict_load

def _load_checkpoint_state(torch) -> dict[str, Any]:
    try:
        payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise TypeError("fixed checkpoint must contain a state_dict mapping")
    raw_state = payload["state_dict"]
    if not raw_state or any(not key.startswith(CHECKPOINT_PREFIX) for key in raw_state):
        raise ValueError(f"all checkpoint keys must start with {CHECKPOINT_PREFIX!r}")
    return {key[len(CHECKPOINT_PREFIX):]: value for key, value in raw_state.items()}

def _build_fixed_batch(torch) -> tuple[dict[str, Any], Any]:
    batch = {
        "pos": torch.tensor([[0.05,0.10,0.15],[0.35,0.45,0.25],[0.70,0.20,0.55],[0.20,0.75,0.80],[0.10,0.20,0.30],[0.55,0.65,0.40],[0.85,0.15,0.70]], dtype=torch.float32),
        "cell": torch.tensor([[[5.1,0.0,0.0],[0.2,5.3,0.0],[0.1,0.3,5.5]],[[4.8,0.0,0.0],[0.4,5.0,0.0],[0.2,0.1,5.2]]], dtype=torch.float32),
        "atomic_numbers": torch.tensor([6,8,14,26,12,29,101], dtype=torch.long),
        "num_atoms": torch.tensor([4,3], dtype=torch.long),
    }
    return batch, torch.tensor([0.125,0.875], dtype=torch.float32)

def _module_attr(module: Any, names: tuple[str, ...]):
    import torch
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, torch.nn.Module):
            return name, value
    return None, None

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["file:mattergen_denoiser.py","file:run_inference.py","file:mattergen_inference.yaml","file:portable_mattergen/__init__.py","file:portable_mattergen/gemnet/gemnet-dT.json","checkpoint_loading:gemnet_scaling_config_sha256"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        required_files = tuple(config.get("required_files", REQUIRED_FILES))
        for relative in required_files:
            ok = (CANDIDATE_DIR / relative).is_file()
            update_check(current_index, ok, str(CANDIDATE_DIR / relative)); current_index += 1
        if len(required_files) < 5:
            for relative in REQUIRED_FILES[len(required_files):]:
                update_check(current_index, (CANDIDATE_DIR / relative).is_file(), str(CANDIDATE_DIR / relative)); current_index += 1
        gemnet_json = CANDIDATE_DIR / "portable_mattergen/gemnet/gemnet-dT.json"
        gemnet_hash = _sha256(gemnet_json)
        update_check(current_index, gemnet_hash == EXPECTED_GEMNET_CONFIG_SHA256, f"actual={gemnet_hash}"); current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_checkpoint_file(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["checkpoint_loading:fixed_files_exist","checkpoint_loading:config_sha256","checkpoint_loading:checkpoint_sha256"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        config_path = CONFIG_PATH.resolve(); checkpoint_path = CHECKPOINT_PATH.resolve()
        update_check(current_index, config_path.is_file() and checkpoint_path.is_file(), f"config={config_path} checkpoint={checkpoint_path}"); current_index += 1
        if not config_path.is_file() or not checkpoint_path.is_file(): return checks
        config_hash = _sha256(config_path); checkpoint_hash = _sha256(checkpoint_path)
        update_check(current_index, config_hash == EXPECTED_CONFIG_SHA256, f"actual={config_hash}"); current_index += 1
        update_check(current_index, checkpoint_hash == EXPECTED_CHECKPOINT_SHA256, f"actual={checkpoint_hash}"); current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["checkpoint_loading:uses_torch_load","checkpoint_loading:uses_strict_state_loading","checkpoint_loading:checkpoint_envelope","checkpoint_loading:state_layout_exact","checkpoint_loading:strict_full_state_loaded","checkpoint_loading:model_class_is_local"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        has_torch_load, has_strict_load = _scan_submission_sources(CANDIDATE_DIR)
        update_check(current_index, has_torch_load, "candidate source contains torch.load"); current_index += 1
        update_check(current_index, has_strict_load, "candidate source contains load_state_dict(..., strict=True)"); current_index += 1
        model_module, run_module = load_implementation(CANDIDATE_DIR, "candidate")
        required_model_functions = ("load_config","build_model","extract_model_state","load_checkpoint_payload","load_model")
        missing = [name for name in required_model_functions if not callable(getattr(model_module, name, None))]
        if missing: return checks
        candidate_config = model_module.load_config(CANDIDATE_DIR / "mattergen_inference.yaml")
        context = _ctx(caller_globals, caller_locals); context.update({"candidate_model_module": model_module,"candidate_run_module": run_module,"candidate_config": candidate_config})
        checkpoint_state = _load_checkpoint_state(torch)
        state_count = len(checkpoint_state); state_numel = sum(tensor.numel() for tensor in checkpoint_state.values())
        update_check(current_index, state_count == EXPECTED_STATE_ENTRY_COUNT and state_numel == EXPECTED_STATE_NUMEL, f"entries={state_count} numel={state_numel}"); current_index += 1
        probe_model = model_module.build_model(candidate_config); context["candidate_model"] = probe_model; context["candidate_probe_model"] = probe_model
        candidate_keys = set(probe_model.state_dict()); checkpoint_keys = set(checkpoint_state)
        update_check(current_index, candidate_keys == checkpoint_keys, "missing={} unexpected={}".format(sorted(checkpoint_keys-candidate_keys)[:10], sorted(candidate_keys-checkpoint_keys)[:10])); current_index += 1
        model = model_module.load_model(candidate_config, CHECKPOINT_PATH.resolve(), device="cpu")
        loaded_state = model.state_dict(); loaded_keys = set(loaded_state)
        exact_values = loaded_keys == checkpoint_keys and all(torch.equal(loaded_state[key].detach().cpu(), checkpoint_state[key].detach().cpu()) for key in checkpoint_keys)
        update_check(current_index, exact_values, f"loaded_entries={len(loaded_state)} loaded_numel={sum(value.numel() for value in loaded_state.values())}"); current_index += 1
        model_source = Path(inspect.getfile(model.__class__)).resolve()
        update_check(current_index, model_source.is_relative_to(CANDIDATE_DIR.resolve()), str(model_source)); current_index += 1
        context["candidate_model"] = model; context.pop("candidate_probe_model", None)
        del probe_model, checkpoint_state, loaded_state; gc.collect()
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["checkpoint_loading:model_loader_functions","evaluation_interface:required_functions","evaluation_interface:function_signatures"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        context = _ctx(caller_globals, caller_locals)
        model_module, run_module = load_implementation(CANDIDATE_DIR, "candidate")
        model_config = model_module.load_config(CANDIDATE_DIR / "mattergen_inference.yaml")
        context.update({"candidate_model_module": model_module,"candidate_run_module": run_module,"candidate_config": model_config})
        required_model_functions = ("load_config","build_model","extract_model_state","load_checkpoint_payload","load_model")
        missing_model = [name for name in required_model_functions if not callable(getattr(model_module, name, None))]
        update_check(current_index, not missing_model, f"missing={missing_model}"); current_index += 1
        required_run_functions = ("validate_inputs","evaluate","load_input_npz","save_output_npz","main")
        missing_run = [name for name in required_run_functions if not callable(getattr(run_module, name, None))]
        update_check(current_index, not missing_run, f"missing={missing_run}"); current_index += 1
        if missing_run: return checks
        evaluate_signature = inspect.signature(run_module.evaluate); load_model_signature = inspect.signature(model_module.load_model)
        ok = all(name in evaluate_signature.parameters for name in ("model","batch","timesteps","device")) and all(name in load_model_signature.parameters for name in ("config","checkpoint_path","device"))
        update_check(current_index, ok, f"evaluate={evaluate_signature} load_model={load_model_signature}"); current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["evaluation_interface:model_is_nn_module","evaluation_interface:model_class","evaluation_interface:forward_signature","evaluation_interface:top_level_components","evaluation_interface:top_level_order","evaluation_interface:gemnet_components","evaluation_interface:interaction_output_block_counts","evaluation_interface:embedding_shapes","evaluation_interface:atom_head_shape","evaluation_interface:parameter_count"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = _ctx(caller_globals, caller_locals); model = context.get("candidate_model")
        if model is None:
            model_module = context["candidate_model_module"]; model = model_module.build_model(context["candidate_config"]); context["candidate_model"] = model
        named_modules = dict(model.named_modules())
        update_check(current_index, isinstance(model, torch.nn.Module), f"class={type(model).__name__}"); current_index += 1
        update_check(current_index, type(model).__name__ == "GemNetTDenoiser", f"class={type(model).__name__}"); current_index += 1
        try:
            signature = inspect.signature(model.forward); forward_ok = list(signature.parameters)[:2] == ["batch","timesteps"]; forward_detail = str(signature)
        except Exception as exc:
            forward_ok = False; forward_detail = f"{type(exc).__name__}: {exc}"
        update_check(current_index, forward_ok, forward_detail); current_index += 1
        time_name, time_module = _module_attr(model, ("noise_level_encoding",)); backbone_name, backbone = _module_attr(model, ("gemnet",)); head_name, atom_head = _module_attr(model, ("fc_atom",))
        top_names = [name for name, _module in model.named_children()]
        top_ok = all(value is not None for value in (time_name, backbone_name, head_name)) and isinstance(getattr(model,"property_embeddings",None), torch.nn.ModuleDict) and int(getattr(model,"hidden_dim",-1)) == 512
        update_check(current_index, top_ok, f"children={top_names} time={time_name} backbone={backbone_name} head={head_name}"); current_index += 1
        required_order = ["gemnet","noise_level_encoding","property_embeddings","fc_atom"]; positions = [top_names.index(name) for name in required_order if name in top_names]; order_ok = len(positions) == 4 and positions == sorted(positions)
        update_check(current_index, order_ok, f"required={required_order} children={top_names}"); current_index += 1
        if backbone is not None:
            interaction_name, interaction_blocks = _module_attr(backbone,("int_blocks",)); output_name, output_blocks = _module_attr(backbone,("out_blocks",)); lattice_name, lattice_blocks = _module_attr(backbone,("lattice_out_blocks",))
            basis_names = {name: hasattr(backbone,name) for name in ("radial_basis","cbf_basis3","mlp_rbf_lattice","mlp_rbf3","mlp_cbf3","mlp_rbf_h","mlp_rbf_out","atom_emb","atom_latent_emb","edge_emb")}
            gemnet_ok = all(basis_names.values()) and interaction_blocks is not None and output_blocks is not None and lattice_blocks is not None and callable(getattr(backbone,"generate_interaction_graph",None))
            block_ok = interaction_blocks is not None and output_blocks is not None and lattice_blocks is not None and len(interaction_blocks) == 4 and len(output_blocks) == 5 and len(lattice_blocks) == 5
            gemnet_detail = f"backbone={backbone_name} basis={basis_names} interaction={interaction_name} output={output_name} lattice={lattice_name}"; block_detail = f"interaction={len(interaction_blocks) if interaction_blocks is not None else -1} output={len(output_blocks) if output_blocks is not None else -1} lattice={len(lattice_blocks) if lattice_blocks is not None else -1}"
        else:
            gemnet_ok = False; block_ok = False; gemnet_detail = "backbone module not found"; block_detail = gemnet_detail
        update_check(current_index, gemnet_ok, gemnet_detail); current_index += 1
        update_check(current_index, block_ok, block_detail); current_index += 1
        hidden_dim = int(context["candidate_config"].get("model",{}).get("hidden_dim",512)); embedding_ok = False; embedding_detail = "not probed"
        try:
            atom_shape = tuple(backbone.atom_emb(torch.tensor([1,6,14,101],dtype=torch.long)).shape); time_shape = tuple(time_module(torch.tensor([0.125,0.875])).shape) if time_module is not None else None; embedding_ok = atom_shape == (4,hidden_dim) and time_shape == (2,hidden_dim); embedding_detail = f"atom={atom_shape} time={time_shape} hidden_dim={hidden_dim}"
        except Exception as exc: embedding_detail = f"{type(exc).__name__}: {exc}"
        update_check(current_index, embedding_ok, embedding_detail); current_index += 1
        atom_head_ok = False; atom_head_detail = "not probed"
        try:
            head_shape = tuple(atom_head(torch.zeros((3,hidden_dim),dtype=torch.float32)).shape); atom_head_ok = head_shape == (3,101); atom_head_detail = f"head={head_shape} expected={(3,101)}"
        except Exception as exc: atom_head_detail = f"{type(exc).__name__}: {exc}"
        update_check(current_index, atom_head_ok, atom_head_detail); current_index += 1
        parameter_count = sum(parameter.numel() for parameter in model.parameters()); state_numel = sum(value.numel() for value in model.state_dict().values())
        update_check(current_index, state_numel == EXPECTED_STATE_NUMEL, f"params={parameter_count} state_numel={state_numel} expected_state_numel={EXPECTED_STATE_NUMEL} modules={len(named_modules)}"); current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["evaluation_interface:model_configuration","evaluation_interface:fixed_checkpoint_paths"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        context = _ctx(caller_globals, caller_locals); model_config = context.get("candidate_config")
        if model_config is None:
            model_module, run_module = load_implementation(CANDIDATE_DIR,"candidate"); model_config = model_module.load_config(CANDIDATE_DIR / "mattergen_inference.yaml"); context.update({"candidate_model_module":model_module,"candidate_run_module":run_module,"candidate_config":model_config})
        expected_model = {"hidden_dim":512,"denoise_atom_types":True,"atom_type_diffusion":"mask"}; expected_gemnet = {"num_targets":1,"cutoff":7.0,"emb_size_atom":512,"emb_size_edge":512,"max_neighbors":50,"max_cell_images_per_dim":5,"num_blocks":4,"regress_stress":True,"otf_graph":True}; model_values = model_config.get("model",{}); gemnet_values = model_values.get("gemnet",{}); ok = all(model_values.get(key) == value for key,value in expected_model.items()) and all(gemnet_values.get(key) == value for key,value in expected_gemnet.items())
        update_check(current_index, ok, f"model={model_values}"); current_index += 1
        checkpoint_config = model_config.get("checkpoint",{}); configured_checkpoint = Path(str(checkpoint_config.get("path",""))).expanduser(); configured_source_config = Path(str(checkpoint_config.get("config_path",""))).expanduser(); ok = configured_checkpoint.resolve() == CHECKPOINT_PATH.resolve() and configured_source_config.resolve() == CONFIG_PATH.resolve()
        update_check(current_index, ok, f"config={configured_source_config} checkpoint={configured_checkpoint}"); current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["evaluation_interface:cli_contract","evaluation_interface:inference_mode_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        context = _ctx(caller_globals, caller_locals); run_module = context.get("candidate_run_module")
        if run_module is None:
            model_module, run_module = load_implementation(CANDIDATE_DIR,"candidate"); context["candidate_model_module"] = model_module; context["candidate_run_module"] = run_module
        run_source = Path(run_module.__file__).read_text(encoding="utf-8"); run_tree = ast.parse(run_source, filename=str(run_module.__file__)); cli_flags = set(); called_attributes = set()
        for node in ast.walk(run_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called_attributes.add(node.func.attr)
                if node.func.attr == "add_argument" and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str): cli_flags.add(node.args[0].value)
        expected_cli_flags = {"--config","--checkpoint","--input","--output","--device"}
        update_check(current_index, expected_cli_flags <= cli_flags, f"flags={sorted(cli_flags)}"); current_index += 1
        update_check(current_index, "eval" in called_attributes and bool({"no_grad","inference_mode"} & called_attributes), f"calls={sorted(called_attributes)}"); current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_method_correctness(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["evaluation_interface:fixed_batch_contract","evaluation_interface:npz_input_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import numpy as np
        import torch
        context = _ctx(caller_globals, caller_locals); torch.manual_seed(SEED); batch, timesteps = _build_fixed_batch(torch); batch_ok = batch["cell"].shape[0] == INPUT_BATCH_SIZE and batch["pos"].shape == (7,3) and timesteps.shape == (INPUT_BATCH_SIZE,)
        update_check(current_index, batch_ok, f"batch_size={batch['cell'].shape[0]} atoms={batch['pos'].shape[0]} keys={sorted(batch)}"); current_index += 1
        context.update({"batch":batch,"timesteps":timesteps,"device":torch.device("cpu")}); run_module = context.get("candidate_run_module")
        if run_module is None:
            model_module, run_module = load_implementation(CANDIDATE_DIR,"candidate"); context.update({"candidate_model_module":model_module,"candidate_run_module":run_module})
        with tempfile.TemporaryDirectory(prefix="mattergen-inference-interface-") as temporary:
            root = Path(temporary); input_path = root / "input.npz"; np.savez(input_path,pos=batch["pos"].numpy(),cell=batch["cell"].numpy(),atomic_numbers=batch["atomic_numbers"].numpy(),num_atoms=batch["num_atoms"].numpy(),timesteps=timesteps.numpy())
            loaded_batch, loaded_timesteps = run_module.load_input_npz(input_path); input_ok = all(torch.equal(loaded_batch[key],batch[key]) for key in batch) and torch.equal(loaded_timesteps,timesteps)
        update_check(current_index, input_ok, str(input_path)); current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["evaluation_interface:forward_call_contract","evaluation_interface:forward_output_shapes"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = _ctx(caller_globals, caller_locals); model = context["candidate_model"]; run_module = context["candidate_run_module"]; batch = context["batch"]; timesteps = context["timesteps"]; device = context["device"]
        source = inspect.getsource(run_module.evaluate); tree = ast.parse(source); model_called = False; guard_called = False
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                if isinstance(node.func,ast.Name) and node.func.id == "model": model_called = True
                if isinstance(node.func,ast.Attribute):
                    if isinstance(node.func.value,ast.Name) and node.func.value.id == "model": model_called = True
                    guard_called |= node.func.attr in {"no_grad","inference_mode"}
        update_check(current_index, model_called and guard_called, f"model_called={model_called} inference_guard={guard_called}"); current_index += 1
        with torch.inference_mode(): output = run_module.evaluate(model,batch,timesteps,device=device)
        context["candidate_output"] = {key: output[key].detach().cpu() for key in OUTPUT_KEYS} if isinstance(output,dict) and all(isinstance(output.get(key),torch.Tensor) for key in OUTPUT_KEYS) else None
        expected_shapes = {"pos":(7,3),"cell":(INPUT_BATCH_SIZE,3,3),"atomic_numbers":(7,101)}; shapes_ok = isinstance(output,dict) and all(isinstance(output.get(key),torch.Tensor) and tuple(output[key].shape) == shape and output[key].is_floating_point() and output[key].device.type == "cpu" for key,shape in expected_shapes.items())
        update_check(current_index, shapes_ok, " ".join(f"{key}={tuple(output[key].shape) if isinstance(output,dict) and hasattr(output.get(key),'shape') else type(output.get(key))}" for key in OUTPUT_KEYS)); current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_output_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["evaluation_interface:forward_output_keys","evaluation_interface:forward_output_finite","evaluation_interface:npz_output_contract"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import numpy as np
        import torch
        context = _ctx(caller_globals, caller_locals); output = context.get("candidate_output"); run_module = context.get("candidate_run_module")
        if output is None:
            with torch.inference_mode(): raw = run_module.evaluate(context["candidate_model"],context["batch"],context["timesteps"],device=context["device"])
            output = {key: raw[key].detach().cpu() for key in OUTPUT_KEYS}; context["candidate_output"] = output
        keys_ok = isinstance(output,dict) and set(output) == set(OUTPUT_KEYS); update_check(current_index, keys_ok, f"keys={sorted(output) if isinstance(output,dict) else type(output).__name__}"); current_index += 1
        finite_ok = keys_ok and all(isinstance(output.get(key),torch.Tensor) and bool(torch.isfinite(output[key]).all()) for key in OUTPUT_KEYS); update_check(current_index, finite_ok, " ".join(f"{key}={bool(torch.isfinite(output[key]).all()) if keys_ok and isinstance(output[key],torch.Tensor) else False}" for key in OUTPUT_KEYS)); current_index += 1
        with tempfile.TemporaryDirectory(prefix="mattergen-inference-interface-") as temporary:
            output_path = Path(temporary) / "output.npz"; run_module.save_output_npz(output_path,output)
            expected_shapes = {"pos":(7,3),"cell":(2,3,3),"atomic_numbers":(7,101)}
            with np.load(output_path,allow_pickle=False) as arrays: output_ok = set(arrays.files) == set(OUTPUT_KEYS) and all(arrays[key].shape == expected_shapes[key] and np.isfinite(arrays[key]).all() for key in OUTPUT_KEYS)
        update_check(current_index, output_ok, str(output_path)); current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks

def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["metric:pos_oracle","metric:cell_oracle","metric:atomic_numbers_oracle"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str): checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        context = _ctx(caller_globals, caller_locals); reference_model_module, reference_run_module = load_implementation(REFERENCE_DIR,"standard"); reference_config = reference_model_module.load_config(REFERENCE_DIR / "mattergen_inference.yaml"); reference_model = reference_model_module.load_model(reference_config,CHECKPOINT_PATH.resolve(),device="cpu"); reference_output = reference_run_module.evaluate(reference_model,context["batch"],context["timesteps"],device=context["device"]); reference_model.cpu(); candidate_output = context["candidate_output"]
        metrics = {}
        for key in OUTPUT_KEYS:
            expected = reference_output[key].detach().cpu(); actual = candidate_output[key]; same_shape = tuple(actual.shape) == tuple(expected.shape); mae = float((actual-expected).abs().mean().item()) if same_shape else float("inf"); max_abs_diff = float((actual-expected).abs().max().item()) if same_shape else float("inf"); close = same_shape and bool(torch.allclose(actual,expected,rtol=1.0e-6,atol=FORWARD_COMPARE_ATOL)); metrics[f"{key}_mae"] = mae; update_check(current_index, close, f"mae={mae} max_abs_diff={max_abs_diff} atol={FORWARD_COMPARE_ATOL}"); current_index += 1
        context["metrics"] = metrics; del reference_model, reference_output
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES): update_check(current_index, False, traceback.format_exc())
    return checks
