"""GeoMPNN checks for the currently approved, small-scale acceptance scope.

Base and submissions remain read-only. CLI smoke runs validate the original
configuration before applying an explicitly recorded process-local budget.
PASS does not certify the complete paper experiment or metric mathematics.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import runpy
import subprocess
import sys
import tempfile
import traceback
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

_EVALUATOR_DIR = Path(__file__).resolve().parent
for _path in (_EVALUATOR_DIR, _EVALUATOR_DIR / "base"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from base import base_evaluator
from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _resolve_args,
    check_parameter_update,
    check_train_structure,
)
from base.model_structure_checker import ModelStructureChecker

_FIELDS = ("x-velocity", "y-velocity", "pressure", "turbulent_viscosity")
_RUNTIME_ROOT = (
    Path(os.environ.get("EVAL_RESULT_DIR", tempfile.gettempdir())).resolve()
    / "geompnn-smoke"
    / hashlib.sha256(str(CANDIDATE_DIR.resolve()).encode()).hexdigest()[:16]
)


def check_interface(config: dict, caller_globals=None, caller_locals=None):
    """Reuse Base for the named model; scripts retain their CLI interface."""
    if config.get("required_interfaces"):
        with patch.object(sys, "dont_write_bytecode", True):
            return base_evaluator.check_interface(config, caller_globals, caller_locals)
    checks = []
    for file_name in config["files"]:
        tree = ast.parse((CANDIDATE_DIR / file_name).read_text(encoding="utf-8-sig"))
        # Do not require train_one_repeat / main / a newly invented function.
        guards = [node for node in ast.walk(tree)
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                  and isinstance(node.test.left, ast.Name)
                  and node.test.left.id == "__name__"
                  and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
                  and len(node.test.comparators) == 1
                  and isinstance(node.test.comparators[0], ast.Constant)
                  and node.test.comparators[0].value == "__main__"]
        top_level_calls = any(
            isinstance(child, ast.Call)
            for node in tree.body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                     ast.Import, ast.ImportFrom))
            for child in ast.walk(node)
        )
        checks.append((f"{config['phase']}:{file_name}_cli", bool(guards) or top_level_calls,
                       "CLI guard or top-level script; actual behavior is tested separately"))
    return checks


@patch.object(sys, "dont_write_bytecode", True)
def check_model_structure(config: dict):
    """Base construction and rule engine; JSON supplies every structure rule."""
    checks = [("model_structure:build_model_ok", False, "Not executed")]
    try:
        with patch.object(sys, "dont_write_bytecode", True):
            model_module = _load_candidate_modules(config["model_file"])
        model_cls = _load_class(model_module, config["model_class"])
        candidate_config = _load_candidate_config(config["config_name"])
        try:
            model_args = _resolve_args(config["model_args"], {"config": candidate_config})
            inspect.signature(model_cls).bind(**model_args)
        except (KeyError, TypeError) as exc:
            return [("model_structure:build_model_ok", False,
                     f"UNRESOLVED: configured constructor adapter needs review: {exc}")]
        model = model_cls(**model_args).to(DEVICE).eval()
        checks[0] = ("model_structure:build_model_ok", True, type(model).__name__)
        checks.append(("model_structure:trainable", any(p.requires_grad and p.numel()
                                                       for p in model.parameters()),
                       "Model has trainable parameters"))
        for rule in config["rules"]:
            checker = ModelStructureChecker()
            checker.add_rules_from_rules([rule])
            results = checker.check(model)
            # 'any' allows equivalent organization and does not impose the
            # reference's registration order, attribute names or extra depth.
            passed = (any(result.passed for result in results)
                      if rule.get("match") == "any"
                      else bool(results) and all(result.passed for result in results))
            checks.append((f"model_structure:{rule['id']}", passed,
                           "; ".join(result.message for result in results)))
    except Exception:
        checks.append(("model_structure:execution", False, traceback.format_exc()))
    return checks


@patch.object(sys, "dont_write_bytecode", True)
def check_model_forward_executable(config: dict):
    """Base forward flow adapted to graph objects and the correct out_shape key."""
    checks = []
    try:
        with patch.object(sys, "dont_write_bytecode", True):
            model_module = _load_candidate_modules(config["model_file"])
        model_cls = _load_class(model_module, config["model_class"])
        candidate_config = _load_candidate_config(config["config_name"])
        try:
            base_args = _resolve_args(config["model_args"], {"config": candidate_config})
            inspect.signature(model_cls).bind(**base_args)
        except (KeyError, TypeError) as exc:
            return [("model_forward:adapter", False,
                     f"UNRESOLVED: constructor contract requires review: {exc}")]

        with torch.random.fork_rng(devices=[DEVICE.index or 0] if DEVICE.type == "cuda" else []):
            torch.manual_seed(int(config["seed"]))
            points = int(config["volume_points"])
            surface_points = int(config["surface_points"])
            angle = torch.arange(surface_points, device=DEVICE) * (2 * torch.pi / surface_points)
            surface_pos = torch.stack((0.5 + 0.5 * torch.cos(angle),
                                       0.04 * torch.sin(angle)), dim=-1)
            volume_pos = torch.rand(points, 2, device=DEVICE)
            tail = torch.tensor([1.0, 0.0], device=DEVICE)
            inlet = torch.tensor([30.0, 2.0], device=DEVICE)
            pair = {}
            for name, pos in (("volume", volume_pos), ("surface", surface_pos)):
                coords = {"nose": pos, "tail": pos - tail}
                pair[name] = SimpleNamespace(
                    pos=pos, x=inlet.expand(len(pos), 2).clone(),
                    y=torch.zeros(len(pos), 4, device=DEVICE),
                    surf=torch.full((len(pos),), name == "surface", device=DEVICE),
                    inlet_vel=inlet.expand(len(pos), 2).clone(),
                    surface_normal=(torch.stack((torch.cos(angle), torch.sin(angle)), -1)
                                    if name == "surface" else torch.zeros_like(pos)),
                    coords=coords,
                    dist={"nose": pos.norm(dim=1, keepdim=True),
                          "tail": (pos - tail).norm(dim=1, keepdim=True),
                          "closest": torch.cdist(pos, surface_pos).amin(1, keepdim=True)},
                )
            for field in config["fields"]:
                args = dict(base_args, field=field)
                model = model_cls(**args).to(DEVICE).eval()
                with torch.no_grad():
                    output = model(pair)
                is_tensor = isinstance(output, torch.Tensor)
                shape = list(output.shape) if is_tensor else None
                checks.append((f"model_forward:{field}:shape",
                               is_tensor and shape == config["out_shape"], f"shape={shape}"))
                checks.append((f"model_forward:{field}:finite",
                               is_tensor and bool(torch.isfinite(output).all()),
                               "All output elements finite"))
    except (AttributeError, KeyError, TypeError) as exc:
        checks.append(("model_forward:adapter", False,
                       f"UNRESOLVED: graph-interface adapter requires review: {exc}"))
    except Exception:
        checks.append(("model_forward:execution", False, traceback.format_exc()))
    return checks


def check_infer_structure(config: dict, caller_globals=None, caller_locals=None):
    """Keep Base's seven checks; recognize ModuleDict indexing as a model call."""
    checks = base_evaluator.check_infer_structure(config, caller_globals, caller_locals)
    # Only patch Base's missing callable-subscript case, not its other results.
    indexed_call = False
    for file_name in config["files"]:
        tree = ast.parse((CANDIDATE_DIR / file_name).read_text(encoding="utf-8-sig"))
        indexed_call |= any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id.lower() in config.get("model_container_names", ["models"])
            for node in ast.walk(tree)
        )
    return [(name, passed or indexed_call, detail + "; callable model container found")
            if name == "inference_start:has_inference" and indexed_call
            else (name, passed, detail) for name, passed, detail in checks]


def _runtime_evidence(config: dict, stage: str):
    """Shared immutable-run provenance for training, artifacts and inference."""
    root = Path(config.get("work_dir", _RUNTIME_ROOT)).resolve()
    candidate = CANDIDATE_DIR.resolve()
    if root == candidate or candidate in root.parents:
        raise ValueError("Runtime output must not be inside the frozen submission")
    record = json.loads((root / f"{stage}_latest.json").read_text(encoding="utf-8"))
    if record["evaluator_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Stale evidence: evaluator changed")
    if record["base_sha256"] != hashlib.sha256(Path(base_evaluator.__file__).read_bytes()).hexdigest():
        raise ValueError("Stale evidence: BaseEvaluator changed")
    for file_name, digest in record["source_hashes"].items():
        if hashlib.sha256((candidate / file_name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Stale evidence: {file_name} changed")
    run_dir = Path(record["run_dir"]).resolve()
    if root not in run_dir.parents:
        raise ValueError("Evidence run directory escaped the runtime root")
    return record


def check_files(config: dict, caller_globals=None, caller_locals=None):
    """Use Base file checks; bind smoke artifacts to this training run."""
    if config.get("phase") != "smoke_train":
        return base_evaluator.check_files(config, caller_globals, caller_locals)

    trace = _runtime_evidence(config, "train")
    run_dir = Path(trace["run_dir"]).resolve()
    paths = [Path(path).resolve() for path in trace.get("checkpoint_saves", [])]
    if not paths:
        return [("smoke_train.checkpoint", False,
                 "The current training run did not save a checkpoint")]
    if any(run_dir not in path.parents for path in paths):
        return [("smoke_train.checkpoint", False,
                 "Checkpoint path escaped the current training run directory")]
    checks = base_evaluator.check_files(dict(config, required_files=list(map(str, paths))),
                                        caller_globals, caller_locals)
    return [(f"smoke_train.checkpoint:{index}", passed, detail)
            for index, (_, passed, detail) in enumerate(checks)]


def _execute_cli(config: dict, stage: str, run_dir: str):
    """Isolated CLI observer; never creates a substitute training/inference step."""
    run_path = Path(run_dir).resolve()
    run_file = config["run_file"]
    module = _load_candidate_modules(run_file)
    trace = {"optimizer_steps": [], "forward": [], "checkpoint_loads": [],
             "checkpoint_saves": [], "state_reloads": []}
    original_call = torch.nn.Module._call_impl
    original_init = torch.optim.Optimizer.__init__
    original_load = torch.load
    original_save = torch.save
    original_reload = torch.nn.Module.load_state_dict

    def observed_call(model, *args, **kwargs):
        output = original_call(model, *args, **kwargs)
        if model.__class__.__name__ == config.get("model_class", "GeoMPNN"):
            pair = args[0] if args else None
            volume = pair.get("volume") if isinstance(pair, dict) else None
            expected_points = len(volume.pos) if hasattr(volume, "pos") else None
            trace["forward"].append({
                "field": str(getattr(model, "field", "")),
                "training": bool(model.training), "grad_enabled": torch.is_grad_enabled(),
                "finite": isinstance(output, torch.Tensor) and bool(torch.isfinite(output).all()),
                "shape": list(output.shape) if isinstance(output, torch.Tensor) else None,
                "expected_points": expected_points,
            })
        return output

    def observed_init(optimizer, *args, **kwargs):
        original_init(optimizer, *args, **kwargs)
        original_step = optimizer.step

        def observed_step(instance, *step_args, **step_kwargs):
            if stage == "train" and len(trace["optimizer_steps"]) >= (
                int(config["smoke_config"]["epochs"])
                * int(config["smoke_config"]["max_train_simulations"])
                * len(config["fields"])
            ):
                raise RuntimeError("UNRESOLVED: candidate did not honor the configured smoke budget")
            parameters = [p for group in instance.param_groups for p in group["params"]
                          if p.requires_grad]
            before = [p.detach().clone() for p in parameters]
            grads = [p.grad for p in parameters if p.grad is not None]
            gradients_ok = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
            result = original_step(*step_args, **step_kwargs)
            trace["optimizer_steps"].append({
                "optimizer": type(instance).__name__,
                "gradients_finite": gradients_ok,
                "changed": any(not torch.equal(a, p.detach()) for a, p in zip(before, parameters)),
                "parameters_finite": all(bool(torch.isfinite(p).all()) for p in parameters),
            })
            return result

        optimizer.step = MethodType(observed_step, optimizer)

    def observed_load(path, *args, **kwargs):
        result = original_load(path, *args, **kwargs)
        if isinstance(path, (str, Path)):
            trace["checkpoint_loads"].append(str(Path(path).resolve()))
        return result

    def observed_reload(model, state_dict, *args, **kwargs):
        result = original_reload(model, state_dict, *args, **kwargs)
        bound = inspect.signature(original_reload).bind(model, state_dict, *args, **kwargs)
        trace["state_reloads"].append({"strict": bool(bound.arguments.get("strict", True))})
        return result

    def observed_save(obj, path, *args, **kwargs):
        result = original_save(obj, path, *args, **kwargs)
        if isinstance(path, (str, Path)):
            trace["checkpoint_saves"].append(str(Path(path).resolve()))
        return result

    original_argv, original_cwd = sys.argv[:], Path.cwd()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.nn.Module, "_call_impl", observed_call))
            stack.enter_context(patch.object(torch.optim.Optimizer, "__init__", observed_init))
            stack.enter_context(patch.object(torch, "load", observed_load))
            stack.enter_context(patch.object(torch, "save", observed_save))
            stack.enter_context(patch.object(torch.nn.Module, "load_state_dict", observed_reload))
            config_path = (CANDIDATE_DIR / config["config_name"]).resolve()
            output = run_path / "trainings"
            if stage == "train":
                adapter = config["budget_adapter"]
                loader = getattr(module, adapter["config_loader"], None)
                prepare = getattr(module, adapter["dataset_loader"], None)
                if not callable(loader) or not callable(prepare):
                    raise RuntimeError("UNRESOLVED: candidate CLI needs a different budget adapter")
                # Validate the ORIGINAL 600-epoch configuration, before overrides.
                original_config = loader(str(config_path))
                if not isinstance(original_config, dict):
                    raise TypeError("Original candidate configuration must be a mapping")
                runtime_config = deepcopy(original_config)
                keys = adapter["epochs_key"].split(".")
                section = runtime_config
                for key in keys[:-1]:
                    section = section[key]
                original_epochs = section[keys[-1]]
                section[keys[-1]] = int(config["smoke_config"]["epochs"])
                max_cases = int(config["smoke_config"]["max_train_simulations"])
                if section[keys[-1]] <= 0 or max_cases <= 0:
                    raise ValueError("Smoke budget must be positive")
                if "limit" not in inspect.signature(prepare).parameters:
                    raise RuntimeError("UNRESOLVED: dataset loader has no supported limit parameter")

                def limited_config(path, *args, **kwargs):
                    # Preserve validation on every CLI config load; do not swallow
                    # validation failures or alter the original dictionary.
                    validated = loader(path, *args, **kwargs)
                    if validated != original_config:
                        raise ValueError("Candidate config changed during smoke")
                    trace["cli_config_loads"] = trace.get("cli_config_loads", 0) + 1
                    return deepcopy(runtime_config)

                def limited_dataset(*args, **kwargs):
                    kwargs["limit"] = max_cases
                    result = prepare(*args, **kwargs)
                    graphs = result[0]
                    if not 0 < len(graphs) <= max_cases:
                        raise RuntimeError("Candidate data loader did not honor smoke budget")
                    trace["loaded_train_simulations"] = len(graphs)
                    return result

                for function, replacement in ((loader, limited_config), (prepare, limited_dataset)):
                    owner = importlib.import_module(function.__module__)
                    if Path(inspect.getfile(function)).resolve().parent != CANDIDATE_DIR.resolve():
                        raise RuntimeError("UNRESOLVED: budget hook is not from this submission")
                    # runpy executes the unmodified script and its __main__ guard.
                    # Imported helper bindings are patched only within this child.
                    if owner is module:
                        raise RuntimeError("UNRESOLVED: inline CLI helpers need a different budget adapter")
                    stack.enter_context(patch.object(owner, function.__name__, replacement))
                trace["budget"] = {"original_epochs": original_epochs,
                                   "epochs": section[keys[-1]], "max_train_simulations": max_cases,
                                   "repeat_index": int(config["repeat_index"]),
                                   "original_config_validated": True}
                argv = [run_file, "--config", str(config_path), "--output", str(output),
                        "--repeat-index", str(config["repeat_index"])]
            else:
                checkpoint = Path(config["checkpoint"]).resolve()
                argv = [run_file, "--config", str(config_path), "--checkpoint", str(checkpoint),
                        "--split", config["split"], "--max-simulations",
                        str(config["max_simulations"]), "--output-dir", str(output / "predictions")]
                trace["expected_checkpoint"] = str(checkpoint)
            sys.argv = argv
            os.chdir(run_path)
            runpy.run_path(str((CANDIDATE_DIR / run_file).resolve()), run_name="__main__")
            if stage == "train" and ("loaded_train_simulations" not in trace
                                     or not trace.get("cli_config_loads")):
                raise RuntimeError("UNRESOLVED: configured budget hooks were not executed")
        trace["completed"] = True
    except SystemExit as exc:
        trace["completed"] = exc.code in (None, 0) and (
            stage != "train" or ("loaded_train_simulations" in trace
                                  and bool(trace.get("cli_config_loads"))))
        if not trace["completed"]:
            trace["error"] = f"CLI exited with status {exc.code}"
    except Exception:
        trace["completed"] = False
        trace["error"] = traceback.format_exc()
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)
    return trace


def _run_cli(config: dict, stage: str):
    """Launch one isolated original CLI and save bounded, source-bound evidence."""
    root = Path(config.get("work_dir", _RUNTIME_ROOT)).resolve()
    candidate = CANDIDATE_DIR.resolve()
    if root == candidate or candidate in root.parents:
        raise ValueError("Runtime output must not be inside the frozen submission")
    root.mkdir(parents=True, exist_ok=True)
    # Never accept evidence left behind by an earlier failed/timeout invocation.
    marker = root / f"{stage}_latest.json"
    if marker.exists():
        marker.unlink()
    if stage == "train" and (root / "infer_latest.json").exists():
        (root / "infer_latest.json").unlink()
    run_dir = Path(tempfile.mkdtemp(prefix=f"{stage}-", dir=root))
    result_path = run_dir / "trace.json"
    payload = dict(config)
    if stage == "infer":
        trained = _runtime_evidence(config, "train")
        checkpoints = [Path(path) for path in trained.get("checkpoint_saves", [])]
        if len(checkpoints) != 1:
            raise ValueError(f"UNRESOLVED: CLI checkpoint routing needs review; found {len(checkpoints)} saved paths")
        output = (Path(trained["run_dir"]) / "trainings").resolve()
        if output not in checkpoints[0].resolve().parents:
            raise ValueError("Checkpoint was not saved under the required trainings/ directory")
        payload["checkpoint"] = str(checkpoints[0])
    source_hashes = {
        str(path.relative_to(candidate)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in candidate.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    }
    driver = (
        "import importlib.util,json,pathlib,sys; "
        "p=json.load(sys.stdin); "
        "s=importlib.util.spec_from_file_location('geompnn_cli_observer',p['evaluator']); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "r=m._execute_cli(p['config'],p['stage'],p['run_dir']); "
        "pathlib.Path(p['result']).write_text(json.dumps(r),encoding='utf-8')"
    )
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, \
            (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [sys.executable, "-B", "-c", driver],
            input=json.dumps({"evaluator": str(Path(__file__).resolve()),
                              "config": payload, "stage": stage, "run_dir": str(run_dir),
                              "result": str(result_path)}),
            text=True, stdout=stdout, stderr=stderr, cwd=run_dir,
            timeout=float(config["timeout_seconds"]), check=False,
        )
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"CLI observer failed; see {run_dir / 'stderr.log'}")
    trace = json.loads(result_path.read_text(encoding="utf-8"))
    trace.update(run_dir=str(run_dir), source_hashes=source_hashes,
                 evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 base_sha256=hashlib.sha256(Path(base_evaluator.__file__).read_bytes()).hexdigest())
    marker.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace


def check_smoke_training(config: dict):
    """Base smoke flow with original CLI, process-local budget and observed steps."""
    trace = _run_cli(config, "train")
    steps = trace["optimizer_steps"]
    fields = {event["field"] for event in trace["forward"] if event["training"]}
    return [
        ("training:smoke_test", bool(trace.get("completed")), trace.get("error", trace["run_dir"])),
        ("training:budget", bool(trace.get("budget", {}).get("original_config_validated"))
         and 0 < trace.get("loaded_train_simulations", 0) <= config["smoke_config"]["max_train_simulations"],
         str(trace.get("budget", {}))),
        ("training:candidate_forward", set(config["fields"]) <= fields,
         f"Observed training fields: {sorted(fields)}"),
        ("training:actual_parameter_update", bool(steps) and all(
            item["gradients_finite"] and item["changed"] and item["parameters_finite"] for item in steps),
         f"Observed optimizer steps: {steps}"),
    ]


def check_infer_execution(config: dict):
    """Base inference execution flow adapted to CLI and real NPZ artifacts."""
    trace = _run_cli(config, "infer")
    expected = trace.get("expected_checkpoint")
    forwards = trace["forward"]
    return [
        ("inference:execution", bool(trace.get("completed")), trace.get("error", trace["run_dir"])),
        ("inference:checkpoint_reload", expected in trace["checkpoint_loads"]
         and bool(trace["state_reloads"]) and all(item["strict"] for item in trace["state_reloads"]),
         f"loads={trace['checkpoint_loads']}; state_reloads={trace['state_reloads']}"),
        ("inference:forward", bool(forwards) and set(config["fields"]) <= {
            event["field"] for event in forwards} and all(
                event["finite"] and event["expected_points"] is not None
                and event["shape"] == [event["expected_points"], 1] for event in forwards),
         f"Observed {len(forwards)} GeoMPNN calls with graph-aligned scalar outputs"),
        ("inference:eval_no_grad", bool(forwards) and all(
            not event["training"] and not event["grad_enabled"] for event in forwards),
         "Candidate inference ran in eval mode with gradients disabled"),
    ]


def check_infer_output_schema(config: dict):
    """Base output-schema intent adapted to the actual NPZ field dictionary."""
    trace = _runtime_evidence(config, "infer")
    output = Path(trace["run_dir"]) / "trainings" / "predictions"
    files = sorted(output.glob("*.npz"))
    checks = [("infer_output_schema:files", len(files) == config["max_simulations"],
               f"prediction files={len(files)}")]
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            required = set(config["fields"]) | {"position", "surface"}
            checks.append((f"infer_output_schema:{path.stem}:keys", required <= keys,
                           f"keys={sorted(keys)}"))
            if required <= keys:
                points = len(data["position"])
                checks.append((f"infer_output_schema:{path.stem}:shape",
                               data["position"].shape == (points, 2) and points > 0
                               and data["surface"].shape == (points,)
                               and all(data[field].shape == (points,) for field in config["fields"]),
                               f"points={points}; field shapes align"))
    return checks


def check_infer_output_validity(config: dict):
    """Validate finite real output, deliberately not metric mathematics."""
    trace = _runtime_evidence(config, "infer")
    files = sorted((Path(trace["run_dir"]) / "trainings" / "predictions").glob("*.npz"))
    checks = [("infer_output_validity:nonempty", bool(files), "Prediction files exist")]
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            for field in config["fields"]:
                checks.append((f"infer_output_validity:{path.stem}:{field}",
                               field in data and data[field].size > 0
                               and bool(np.isfinite(data[field]).all()),
                               "Finite nonempty predictions; no accuracy claim"))
    return checks
