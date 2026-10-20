"""V2 adapter for the standalone NequIX fixed-checkpoint inference evaluator."""

from __future__ import annotations

from pathlib import Path
import sys
import ast
import hashlib
import importlib.util
import json
import os
import traceback
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


def _candidate_path(path: str | Path) -> Path:
    """Resolve paths rooted at either the artifact directory or its parent."""
    path = Path(path)
    if path.is_absolute():
        return path
    direct = CANDIDATE_DIR / path
    if direct.exists():
        return direct
    if path.parts and path.parts[0] == CANDIDATE_DIR.name:
        return CANDIDATE_DIR.joinpath(*path.parts[1:])
    return direct

MODEL_PATH = os.environ.get(
    "NEQUIX_INFERENCE_MODEL_PATH",
    "/public/share/sugonhpcapp01/onestore/onemodels/NequIX/nequix-mp-1.pt",
)
DATA_PATH = os.environ.get(
    "NEQUIX_INFERENCE_DATA_PATH",
    "/public/share/sugonhpcapp01/onestore/onedatasets/matchem/mptrj_smoke/data_00000.aselmdb",
)
MODEL_SHA256 = "c6f54523bb73537994bc932f0a1a55c06b1d643e582e9c9ab20d00c35546791c"
DATA_SHA256 = "b32cfaec1fdefb3a823e0bd19b7fab32d3a0c325e644dd388c5cc75117b4143b"
EXPECTED_STATE_COUNT = 107
EXPECTED_STATE_NUMEL = 732710
EXPECTED_PARAM_COUNT = 38
EXPECTED_ATOMIC_NUMBERS = list(range(1, 84)) + [89, 90, 91, 92, 93, 94]
REF_NUM_ATOMS = 28
REF_ENERGY = -170.49705505371094
REF_FORCES = [[1.3364478945732117e-07, 0.0005347970873117447, -0.1422138214111328],[-1.431908458471298e-07, 0.0005343826487660408, 0.1421908140182495],[-1.2713694275134912e-07, -0.0005360981449484825, -0.1422053575515747],[1.1990778148174286e-08, -0.0005349693819880486, 0.14221841096878052],[-2.669548848643899e-07, -0.00968366488814354, 0.010525792837142944],[-5.6690289795824356e-08, -0.009684372693300247, -0.010528385639190674],[-4.886533133685589e-08, 0.009683892130851746, 0.010529249906539917],[1.2107162206120847e-07, 0.009682904928922653, -0.010527253150939941],[-6.705522537231445e-08, 0.005860884208232164, -0.005731863435357809],[-4.423782229423523e-09, 0.005860256031155586, 0.005727042444050312],[-1.431908458471298e-07, -0.005859747529029846, -0.005731203593313694],[-4.074536263942718e-08, -0.0058609540574252605, 0.005728244781494141],[3.387712155245026e-08, -0.0329960472881794, 0.0133693041279912],[1.1274863709331847e-07, -0.03299330919981003, -0.013371221721172333],[4.2782630771398544e-08, 0.032992392778396606, 0.013371078297495842],[1.6469857655465603e-07, 0.032994020730257034, -0.013369519263505936],[-1.084986323007797e-07, 0.03672581911087036, 0.015066704712808132],[8.265487849712372e-08, 0.03672276809811592, -0.015069331973791122],[1.4784021118430246e-07, -0.036721017211675644, 0.01506915781646967],[5.920122703173547e-08, -0.03672335296869278, -0.015068503096699715],[1.3504194384950097e-07, 0.0044353557750582695, 0.10970360785722733],[1.9554863683879375e-07, 0.004435777198523283, -0.10969489812850952],[9.872019290924072e-08, -0.004435214679688215, 0.1097038984298706],[-3.306486178189516e-07, -0.0044356402941048145, -0.10970404744148254],[1.7847924027591944e-08, -0.019763752818107605, 0.0038284333422780037],[1.0703444530690831e-07, -0.019763141870498657, -0.003813760355114937],[-7.847160077290027e-08, 0.019764456897974014, 0.003828110173344612],[-3.541936166584492e-08, 0.01976354792714119, -0.0038306722417473793]]
REF_STRESS = [[0.0012042548041790724, -7.824133702039049e-12, 3.647646629190149e-09],[-7.824133702039049e-12, -0.004582214169204235, -7.934428225553347e-09],[3.647646629190149e-09, -7.934428225553347e-09, 0.010338063351809978]]
ATOL_ENERGY = 1.0e-3
ATOL_FORCE = 1.0e-3
ATOL_STRESS = 1.0e-3
REQUIRED_FILES = ("nequix_inference/nequix_inference.py", "nequix_inference/run_inference.py", "nequix_inference/nequix_inference.yaml", "nequix_inference/portable_nequix/__init__.py")
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
    safe_load = False
    strict_load = False
    for path in submission.rglob("*.py"):
        if "portable_nequix" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                prohibited.extend(alias.name for alias in node.names if alias.name.split(".")[0] in PROHIBITED_IMPORTS)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in PROHIBITED_IMPORTS:
                prohibited.append(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                if node.func.attr == "load_state_dict" and isinstance(keywords.get("strict"), ast.Constant) and keywords["strict"].value is True:
                    strict_load = True
    return prohibited, safe_load, strict_load

def _load_atoms():
    import ase.db
    db = ase.db.connect(str(DATA_PATH), readonly=True, use_lock_file=False)
    return list(db.select())[0].toatoms()

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["required_file_nequix_inference", "required_file_run_inference", "required_file_config", "required_file_portable_init", "prohibited_active_imports"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        submission = CANDIDATE_DIR
        missing = [name for name in REQUIRED_FILES if not _candidate_path(name).is_file()]
        detail = json.dumps({"missing": missing}, sort_keys=True)
        update_check(current_index, _candidate_path(REQUIRED_FILES[0]).is_file(), detail); current_index += 1
        update_check(current_index, _candidate_path(REQUIRED_FILES[1]).is_file(), detail); current_index += 1
        update_check(current_index, _candidate_path(REQUIRED_FILES[2]).is_file(), detail); current_index += 1
        update_check(current_index, _candidate_path(REQUIRED_FILES[3]).is_file(), detail); current_index += 1
        prohibited = inspect_source(submission)[0] if not missing else []
        detail = json.dumps({"missing": missing, "prohibited_active_imports": prohibited}, sort_keys=True)
        update_check(current_index, not prohibited, detail); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["no_prohibited_active_imports", "strict_load_state_dict", "model_sha256", "data_sha256", "state_entry_count", "state_tensor_numel", "restored_state_keys", "restored_state_values"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = load_module("nequix_inference_submission", _candidate_path("nequix_inference/nequix_inference.py"))
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        prohibited, _safe_load, strict_load = inspect_source(submission)
        model_hash = sha256(MODEL_PATH); data_hash = sha256(DATA_PATH)
        state_keys = list(state); state_numel = sum(v.numel() for v in state.values() if torch.is_tensor(v))
        loaded = model.state_dict(); same_keys = set(loaded) == set(state)
        same_values = same_keys and all(torch.equal(loaded[k].cpu(), state[k].cpu()) for k in loaded)
        detail = json.dumps({"model_sha256": model_hash, "data_sha256": data_hash, "strict_load": strict_load, "state_count": len(state_keys), "state_numel": state_numel, "same_keys": same_keys, "same_values": same_values}, sort_keys=True)
        update_check(current_index, not prohibited, detail); current_index += 1
        update_check(current_index, strict_load, detail); current_index += 1
        update_check(current_index, model_hash == MODEL_SHA256, detail); current_index += 1
        update_check(current_index, data_hash == DATA_SHA256, detail); current_index += 1
        update_check(current_index, len(state_keys) == EXPECTED_STATE_COUNT, detail); current_index += 1
        update_check(current_index, state_numel == EXPECTED_STATE_NUMEL, detail); current_index += 1
        update_check(current_index, same_keys, detail); current_index += 1
        update_check(current_index, same_values, detail); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["model_class_name", "top_level_module_order", "layer_count", "first_layer_module_order", "parameter_tensor_count", "cutoff", "n_species"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = load_module("nequix_inference_submission", _candidate_path("nequix_inference/nequix_inference.py"))
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        order = list(model._modules); expected_order = ["layers", "readout"]; n_layers = len(model.layers)
        layer0_order = list(model.layers[0]._modules); expected_layer0 = ["linear_1", "tp", "sort", "radial_mlp", "linear_2", "skip", "layer_norm", "gate"]
        n_params = len(list(model.parameters())); cutoff = float(model.cutoff)
        detail = json.dumps({"model": type(model).__name__, "module_order": order, "n_layers": n_layers, "layer0_order": layer0_order, "param_count": n_params, "cutoff": cutoff, "n_species": model.n_species}, sort_keys=True)
        update_check(current_index, type(model).__name__ == "NequixTorch", detail); current_index += 1
        update_check(current_index, order == expected_order, detail); current_index += 1
        update_check(current_index, n_layers == 4, detail); current_index += 1
        update_check(current_index, layer0_order == expected_layer0, detail); current_index += 1
        update_check(current_index, n_params == EXPECTED_PARAM_COUNT, detail); current_index += 1
        update_check(current_index, abs(cutoff - 6.0) < 1e-9, detail); current_index += 1
        update_check(current_index, model.n_species == len(EXPECTED_ATOMIC_NUMBERS), detail); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_interface(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["required_callables", "evaluate_structure_parameters", "required_output_keys", "energy_shape", "forces_shape", "stress_shape", "energy_finite", "forces_finite", "stress_finite"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = load_module("nequix_inference_submission", _candidate_path("nequix_inference/nequix_inference.py"))
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        atoms = _load_atoms()
        missing = [name for name in ("load_model", "evaluate_structure", "load_checkpoint_payload", "raw_model_state") if not callable(getattr(module, name, None))]
        detail = json.dumps({"missing_apis": missing}, sort_keys=True)
        update_check(current_index, not missing, detail); current_index += 1
        import inspect as _inspect
        sig = _inspect.signature(module.evaluate_structure); params = set(sig.parameters)
        out = module.evaluate_structure(model, atoms, "cpu"); keys = set(out)
        energy = out["energy"]; forces = out["forces"]; stress = out["stress"]
        detail = json.dumps({"missing_apis": missing, "output_keys": sorted(keys), "energy_shape": list(energy.shape), "forces_shape": list(forces.shape), "stress_shape": list(stress.shape)}, sort_keys=True)
        update_check(current_index, {"model", "atoms"} <= params, detail); current_index += 1
        update_check(current_index, {"energy", "forces", "stress"} <= keys, detail); current_index += 1
        update_check(current_index, list(energy.shape) == [1], detail); current_index += 1
        update_check(current_index, list(forces.shape) == [REF_NUM_ATOMS, 3], detail); current_index += 1
        update_check(current_index, list(stress.shape) == [1, 3, 3], detail); current_index += 1
        update_check(current_index, bool(torch.isfinite(energy).all()), detail); current_index += 1
        update_check(current_index, bool(torch.isfinite(forces).all()), detail); current_index += 1
        update_check(current_index, bool(torch.isfinite(stress).all()), detail); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["forward_outputs_finite", "forward_output_shapes"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = load_module("nequix_inference_submission", _candidate_path("nequix_inference/nequix_inference.py"))
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        atoms = _load_atoms(); out = module.evaluate_structure(model, atoms, "cpu")
        finite = all(torch.isfinite(v).all().item() for v in (out["energy"], out["forces"], out["stress"]))
        shapes = {"energy": list(out["energy"].shape), "forces": list(out["forces"].shape), "stress": list(out["stress"].shape)}
        detail = json.dumps({"shapes": shapes, "finite": finite}, sort_keys=True)
        update_check(current_index, finite, detail); current_index += 1
        update_check(current_index, shapes == {"energy": [1], "forces": [REF_NUM_ATOMS, 3], "stress": [1, 3, 3]}, detail); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ["energy_oracle", "forces_oracle", "stress_oracle"]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = load_module("nequix_inference_submission", _candidate_path("nequix_inference/nequix_inference.py"))
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        atoms = _load_atoms(); out = module.evaluate_structure(model, atoms, "cpu")
        energy_err = abs(float(out["energy"].reshape(-1)[0]) - REF_ENERGY)
        forces = out["forces"].detach().cpu().float(); ref_forces = torch.tensor(REF_FORCES, dtype=torch.float32); force_err = float((forces - ref_forces).abs().max())
        stress = out["stress"].detach().cpu().float(); ref_stress = torch.tensor(REF_STRESS, dtype=torch.float32); stress_err = float((stress - ref_stress).abs().max())
        detail = json.dumps({"energy_abs_error": energy_err, "force_abs_error": force_err, "stress_abs_error": stress_err, "atol": {"energy": ATOL_ENERGY, "force": ATOL_FORCE, "stress": ATOL_STRESS}}, sort_keys=True)
        update_check(current_index, energy_err <= ATOL_ENERGY, detail); current_index += 1
        update_check(current_index, force_err <= ATOL_FORCE, detail); current_index += 1
        update_check(current_index, stress_err <= ATOL_STRESS, detail); current_index += 1
    except Exception as e:
        update_check(current_index, False, traceback.format_exc())
    return checks
