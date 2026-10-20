"""V2 adapter for the TACE inference legacy evaluator."""
import ast
import hashlib
import json
import os
import subprocess
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
from base_evaluator import _load_module_from_path, check_files, check_interface, check_config_content


CANDIDATE_DIR = base_evaluator.CANDIDATE_DIR
if not (CANDIDATE_DIR / "tace_inference.py").is_file() and (
    CANDIDATE_DIR / "tace_inference" / "tace_inference.py"
).is_file():
    CANDIDATE_DIR = CANDIDATE_DIR / "tace_inference"
base_evaluator.CANDIDATE_DIR = CANDIDATE_DIR


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
    "TACE_INFERENCE_MODEL_PATH",
    "/public/share/sugonhpcapp01/onestore/onemodels/tace/TACE-OAM-7M.pt",
)
DATA_PATH = os.environ.get(
    "TACE_INFERENCE_DATA_PATH",
    "/public/share/sugonhpcapp01/onestore/onedatasets/matchem/TACE/BaTiO3.xyz",
)
MODEL_SHA256 = "87f6b23c40a5b29c3b0d2520bca104d73c653866e82541919db94979c7f6511c"
DATA_SHA256 = "688c5486100968538f6b146ab3fc69736f97124ac797e5635650e8105a6a0de9"
EXPECTED_STATE_COUNT = 456
EXPECTED_STATE_NUMEL = 7105809
EXPECTED_ATOMIC_NUMBERS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 89, 90, 91, 92, 93, 94,
]
REF_ENERGY = -1080.5152587890625
REF_FORCES = [
    [-0.05026010051369667, -0.01835726946592331, -0.13968107104301453],
    [0.07090035080909729, -0.07679210603237152, -0.02453170344233513],
    [0.04508743807673454, -0.053827475756406784, 0.058414168655872345],
]
REF_STRESS = [
    [-0.010844347067177296, 1.5563826991638052e-06, 0.0002339275524718687],
    [1.5563826991638052e-06, -0.010819287039339542, 8.044990681810305e-05],
    [0.0002339275524718687, 8.044990681810305e-05, -0.009753160178661346],
]
REF_NUM_ATOMS = 135
ATOL_ENERGY = 1.0e-3
ATOL_FORCE = 1.0e-3
ATOL_STRESS = 1.0e-3
PROHIBITED_IMPORTS = {"onescience", "nequip", "mace", "mattersim"}


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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_data_availability(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Verify the fixed 59-structure inference input before model checks."""
    try:
        import ase.io
        digest = sha256(DATA_PATH)
        structures = ase.io.read(str(DATA_PATH), index=":")
        return [
            ("data:sha256", digest == DATA_SHA256, digest),
            ("data:structure_count", len(structures) == 59, str(len(structures))),
        ]
    except Exception:
        return [("data:availability", False, traceback.format_exc())]


def check_checkpoint_file(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Validate the external checkpoint; Base's file check lacks SHA validation."""
    try:
        digest = sha256(MODEL_PATH)
        return [("checkpoint:sha256", digest == MODEL_SHA256, digest)]
    except Exception:
        return [("checkpoint:file", False, traceback.format_exc())]


def check_checkpoint_loading(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Check safe deserialization and complete strict state restoration."""
    names = (
        "checkpoint:weights_only", "checkpoint:strict_load",
        "checkpoint:payload_keys", "checkpoint:state_count",
        "checkpoint:state_numel", "checkpoint:state_keys", "checkpoint:state_values",
    )
    try:
        module = _load_module_from_path("tace_inference_checkpoint", _candidate_path("tace_inference.py"))
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        _, safe_load, strict_load = inspect_source(CANDIDATE_DIR)
        loaded = model.state_dict()
        same_keys = set(loaded) == set(state)
        same_values = same_keys and all(torch.equal(loaded[k].cpu(), state[k].cpu()) for k in loaded)
        required = {"cfg", "state_dict", "statistics", "target_property", "embedding_property"}
        values = (
            safe_load, strict_load, required.issubset(payload),
            len(state) == EXPECTED_STATE_COUNT,
            sum(v.numel() for v in state.values() if torch.is_tensor(v)) == EXPECTED_STATE_NUMEL,
            same_keys, same_values,
        )
        detail = json.dumps({"state_count": len(state), "same_keys": same_keys,
                             "same_values": same_values}, sort_keys=True)
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
    CHECKPOINT_NAMES = [
        "model_structure:wrapper_class",
        "model_structure:readout_class",
        "model_structure:module_order",
        "model_structure:atomic_numbers",
        "model_structure:cutoff",
        "model_structure:target_property",
        "model_structure:num_layers",
        "model_structure:num_channel",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = _load_module_from_path("tace_inference_submission", _candidate_path("tace_inference/tace_inference.py"))
        import ase.io
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        ctx = {"model": model, "payload": payload, "state": state}
        ctx["atoms"] = ase.io.read(str(DATA_PATH), index=0)
        model = ctx["model"]
        readout = model.readout_fn
        order = list(readout._modules)
        expected_order = ["representation", "energy_readouts", "atomic_energy_layer", "scale_shift", "zbl"]
        atomic_numbers = model.get_atomic_numbers()
        cutoff = model.get_cutoff()
        target = model.get_target_property()
        num_layers = readout.num_layers
        num_channel = readout.num_channel
        ok = type(model).__name__ == "TensorModel"
        update_check(current_index, ok, type(model).__name__)
        current_index += 1
        ok = type(readout).__name__ == "e3nnTACE"
        update_check(current_index, ok, type(readout).__name__)
        current_index += 1
        ok = order == expected_order
        update_check(current_index, ok, str(order))
        current_index += 1
        ok = atomic_numbers == EXPECTED_ATOMIC_NUMBERS
        update_check(current_index, ok, str(atomic_numbers))
        current_index += 1
        ok = abs(cutoff - 6.0) < 1e-9
        update_check(current_index, ok, str(cutoff))
        current_index += 1
        ok = set(target) == {"energy", "forces", "stress"}
        update_check(current_index, ok, str(target))
        current_index += 1
        ok = num_layers == 5
        update_check(current_index, ok, str(num_layers))
        current_index += 1
        ok = num_channel == 128
        update_check(current_index, ok, str(num_channel))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_output_schema(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Validate tensor keys and shapes on a real ASE structure."""
    names = (
        "infer_output:keys", "infer_output:energy_shape", "infer_output:forces_shape",
        "infer_output:stress_shape",
    )
    try:
        import ase.io
        module = _load_module_from_path("tace_inference_schema", _candidate_path("tace_inference.py"))
        model, _, _ = module.load_model(str(MODEL_PATH), "cpu")
        atoms = ase.io.read(str(DATA_PATH), index=0)
        out = module.evaluate_structure(model, atoms, "cpu")
        keys_ok = isinstance(out, dict) and {"energy", "forces", "stress"} <= set(out)
        if not keys_ok:
            return [(names[0], False, f"keys={list(out) if isinstance(out, dict) else type(out).__name__}")]
        values = (
            keys_ok, torch.is_tensor(out["energy"]) and list(out["energy"].shape) == [1],
            torch.is_tensor(out["forces"]) and list(out["forces"].shape) == [len(atoms), 3],
            torch.is_tensor(out["stress"]) and list(out["stress"].shape) == [1, 3, 3],
        )
        detail = json.dumps({k: list(out[k].shape) for k in ("energy", "forces", "stress")},
                            sort_keys=True)
        return [(name, bool(ok), detail) for name, ok in zip(names, values)]
    except Exception:
        return [(names[0], False, traceback.format_exc())]

def check_infer_output_validity(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Verify that energy, force and stress are finite tensors."""
    try:
        import ase.io
        module = _load_module_from_path("tace_inference_validity", _candidate_path("tace_inference.py"))
        model, _, _ = module.load_model(str(MODEL_PATH), "cpu")
        atoms = ase.io.read(str(DATA_PATH), index=0)
        out = module.evaluate_structure(model, atoms, "cpu")
        return [(f"infer_output:{key}_finite",
                 torch.is_tensor(out.get(key)) and bool(torch.isfinite(out[key]).all()),
                 str(type(out.get(key)).__name__))
                for key in ("energy", "forces", "stress")]
    except Exception:
        return [("infer_output:finite", False, traceback.format_exc())]

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "forward_executable:finite",
        "forward_executable:shapes",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = _load_module_from_path("tace_inference_submission", _candidate_path("tace_inference/tace_inference.py"))
        import ase.io
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        atoms = ase.io.read(str(DATA_PATH), index=0)
        ctx = {"model": model, "payload": payload, "state": state, "atoms": atoms}
        atoms = ctx["atoms"]
        model = ctx["model"]
        out = module.evaluate_structure(model, atoms, "cpu")
        finite = all(torch.isfinite(v).all().item() for v in (out["energy"], out["forces"], out["stress"]))
        shapes = {"energy": list(out["energy"].shape), "forces": list(out["forces"].shape), "stress": list(out["stress"].shape)}
        ok = finite
        update_check(current_index, ok, json.dumps({"finite": finite}, sort_keys=True))
        current_index += 1
        ok = shapes == {"energy": [1], "forces": [REF_NUM_ATOMS, 3], "stress": [1, 3, 3]}
        update_check(current_index, ok, json.dumps({"shapes": shapes}, sort_keys=True))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks


def check_infer_execution(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Run the public CLI on two genuine structures and inspect its artifact."""
    try:
        import ase.io
        with tempfile.TemporaryDirectory(prefix="tace_infer_eval_") as directory:
            source = Path(directory) / "input.xyz"
            output = Path(directory) / "predictions.json"
            ase.io.write(str(source), ase.io.read(str(DATA_PATH), index="0:2"))
            command = [sys.executable, str(_candidate_path("run_inference.py")),
                       "--config", str(_candidate_path("tace_inference.yaml")),
                       "--checkpoint", str(MODEL_PATH), "--input", str(source),
                       "--output", str(output), "--device", "cpu"]
            run = subprocess.run(command, cwd=str(CANDIDATE_DIR), capture_output=True,
                                 text=True, timeout=600, check=False)
            if run.returncode:
                return [("infer_execution:cli", False,
                         f"exit={run.returncode}; stderr={run.stderr[-4000:]}")]
            exists = output.is_file()
            records = json.loads(output.read_text(encoding="utf-8")) if exists else None
            valid = isinstance(records, list) and len(records) == 2
            return [("infer_execution:output_file", exists, str(output)),
                    ("infer_execution:record_count", valid,
                     f"records={len(records) if isinstance(records, list) else None}")]
    except Exception:
        return [("infer_execution:cli", False, traceback.format_exc())]


def check_evaluate_oracle(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = [
        "fixed_input_oracle:energy",
        "fixed_input_oracle:forces",
        "fixed_input_oracle:stress",
    ]
    checks = [(name, False, "Not executed due to earlier failure") for name in CHECKPOINT_NAMES]

    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)

    current_index = 0
    try:
        submission = CANDIDATE_DIR
        module = _load_module_from_path("tace_inference_submission", _candidate_path("tace_inference/tace_inference.py"))
        import ase.io
        model, payload, state = module.load_model(str(MODEL_PATH), "cpu")
        atoms = ase.io.read(str(DATA_PATH), index=0)
        ctx = {"model": model, "payload": payload, "state": state, "atoms": atoms}
        atoms = ctx["atoms"]
        model = ctx["model"]
        out = module.evaluate_structure(model, atoms, "cpu")
        energy_err = abs(float(out["energy"].reshape(-1)[0]) - REF_ENERGY)
        forces = out["forces"].detach().cpu().float()
        ref_forces = torch.tensor(REF_FORCES, dtype=torch.float32)
        force_err = float((forces[:3] - ref_forces).abs().max())
        stress = out["stress"].detach().cpu().float()
        ref_stress = torch.tensor(REF_STRESS, dtype=torch.float32)
        stress_err = float((stress - ref_stress).abs().max())
        ok = energy_err <= ATOL_ENERGY
        update_check(current_index, ok, json.dumps({"energy_abs_error": energy_err, "atol": ATOL_ENERGY}, sort_keys=True))
        current_index += 1
        ok = force_err <= ATOL_FORCE
        update_check(current_index, ok, json.dumps({"force_abs_error": force_err, "atol": ATOL_FORCE}, sort_keys=True))
        current_index += 1
        ok = stress_err <= ATOL_STRESS
        update_check(current_index, ok, json.dumps({"stress_abs_error": stress_err, "atol": ATOL_STRESS}, sort_keys=True))
        current_index += 1
    except Exception as e:
        if current_index < len(CHECKPOINT_NAMES):
            update_check(current_index, False, traceback.format_exc())

    return checks
