"""Spectral Refiner checks for rule_eval; Base and submission sources stay unchanged.

This core profile validates the public model/build/CLI/artifact contracts. It is
not a Table 2 accuracy oracle or a validation of the H^-1 mathematics.
"""
import inspect
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch
from base import base_evaluator
from base.base_evaluator import (
    CANDIDATE_DIR,
    DEVICE,
    _load_candidate_config,
    _load_candidate_modules,
    _load_class,
    _resolve_args,
    check_infer_structure,
    check_interface,
    check_parameter_update,
)
from base.model_structure_checker import ModelStructureChecker, ModuleInfoExtractor

# One isolated CLI run is shared by training, artifacts and inference checks.
# Do not seed this directory with submitted checkpoints or historical results.
_SMOKE_RUN = {}


def _printed_metric_leaf(text, metric, phase):
    """Require a finite printed numeric value, not just a metric name."""
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = rf"(?<![\w])[\"']?{re.escape(metric)}[\"']?\s*(?:[:=]|\s+)\s*(?:tensor\(\s*)?({number})"
    match = re.search(pattern, text)
    value = float(match.group(1)) if match else None
    passed = value is not None and math.isfinite(value) and value >= 0
    return (f"{phase}:printed_metric:{metric}", passed,
            f"Printed finite nonnegative numeric {metric}: {value!r}")


def _build_model(config):
    """Use the Prompt's public builder, without fixing SFNO constructor names."""
    model_module = _load_candidate_modules(config["model_file"])
    model_cls = _load_class(model_module, config["model_class"])
    candidate_config = _load_candidate_config(config["config_name"])
    builder_module = _load_candidate_modules(config["builder_file"])
    builder = getattr(builder_module, config["builder_method"])
    args = _resolve_args(config["builder_args"], {"config": candidate_config})
    signature = inspect.signature(builder)
    try:
        signature.bind(**args)
    except TypeError:
        # The Prompt fixes build_model's name, not its config parameter name.
        signature.bind(candidate_config)
        model = builder(candidate_config)
    else:
        model = builder(**args)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model must return an nn.Module")
    if not isinstance(model, model_cls):
        # Base imports model.py as candidate_model; train.py normally imports
        # that same file as model. Python then creates two distinct SFNO class
        # objects. Compare the defining source and public class in the MRO,
        # rather than rejecting this ordinary multi-file implementation.
        declared_source = Path(inspect.getfile(model_cls)).resolve()
        same_source = any(
            ancestor.__name__ == model_cls.__name__
            and Path(inspect.getfile(ancestor)).resolve() == declared_source
            for ancestor in type(model).__mro__
        )
        if not same_source:
            raise TypeError("build_model must return the submission's SFNO nn.Module")
    return model.to(DEVICE).eval()


def check_model_structure(config, caller_globals=None, caller_locals=None):
    """Base model-construction/rules flow, adapted to the public build_model."""
    checks = [("model_structure:build_model_ok", False, "Not executed")]
    try:
        model = _build_model(config)
        checks[0] = (checks[0][0], True, "Public build_model constructed SFNO")
        checker = ModelStructureChecker()
        checker.add_rules_from_rules(config["rules"])
        root = ModuleInfoExtractor(model).extract()
        for rule in checker.rules:
            selected = checker._get_modules_for_rule(root, rule, None)
            if rule.rule_type == "module_count":
                expected = rule.params.get("count")
                checks.append((f"model_structure:{rule.id}",
                               expected is not None and len(selected) == expected,
                               f"Expected count={expected}; actual={len(selected)}"))
                continue
            for info in selected:
                passed, message = checker.evaluate_rule(rule, info)
                name = f"model_structure:{rule.id}"
                if info.module_path:
                    name += f":{info.module_path}"
                checks.append((name, passed, message))
        checks.append((
            "model_structure:trainable_parameters",
            any(p.requires_grad and p.numel() > 0 for p in model.parameters()),
            "SFNO has registered trainable parameters; full paper topology is not certified",
        ))
    except Exception:
        checks.append(("model_structure:execution", False, traceback.format_exc()))
    return checks


def check_model_forward_executable(config, caller_globals=None, caller_locals=None):
    """Base forward flow; fix out_shape key and use the public model builder."""
    names = (
        "forward_executable:forward_run",
        "forward_executable:output_shape",
        "forward_executable:finite_output",
    )
    checks = [(name, False, "Not executed") for name in names]
    index = 0
    try:
        model = _build_model(config)
        inputs = torch.randn(*config["input_shape"], device=DEVICE)
        # Positional input avoids requiring the standard's internal 'v' name.
        args = _resolve_args(config.get("forward_args", {}), {})
        with torch.no_grad():
            output = model(inputs, **args)
        checks[0] = (names[0], True, "Candidate SFNO forward executed")
        index = 1
        expected = config["out_shape"]
        shape_ok = (
            isinstance(output, torch.Tensor) and output.ndim == len(expected)
            and all(size > 0 and (wanted is None or size == wanted)
                    for size, wanted in zip(output.shape, expected))
        )
        checks[1] = (names[1], shape_ok, f"Actual={getattr(output, 'shape', None)}; expected={config['out_shape']}")
        index = 2
        finite = isinstance(output, torch.Tensor) and bool(torch.isfinite(output).all())
        checks[2] = (names[2], finite, "Forward output must contain only finite values")
    except Exception:
        checks[index] = (names[index], False, traceback.format_exc())
    return checks


def check_train_structure(config, caller_globals=None, caller_locals=None):
    """Keep Base AST checks, omit script forms not required by the Prompt.

    A main-named function, DataLoader and an explicit .train() call are not
    mandatory public interfaces. The actual CLI is exercised separately.
    """
    excluded = {
        "train_start:has_main", "train_start:has_dataloader",
        "train_start:has_train_mode",
    }
    return [leaf for leaf in base_evaluator.check_train_structure(config)
            if leaf[0] not in excluded]


def check_smoke_training(config, caller_globals=None, caller_locals=None):
    """Adapt Base smoke config validation/execution to the required CLI."""
    names = ("smoke_training:config", "smoke_training:execution")
    checks = [(name, False, "Not executed") for name in names]
    checks += [(f"smoke_training:printed_metric:{name}", False, "Not executed")
               for name in config["printed_metrics"]]
    _SMOKE_RUN.clear()
    index = 0
    try:
        _load_candidate_config(config["config_name"])
        checks[0] = (names[0], True, "Candidate YAML is a mapping")
        index = 1
        work = Path(tempfile.mkdtemp(prefix="spectral-refiner-smoke-"))
        project = work / "submission"
        _SMOKE_RUN.update(work=work, project=project, trained=False, inferred=False)
        shutil.copytree(
            CANDIDATE_DIR, project,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", "checkpoints", "results",
            ),
        )
        command = [
            sys.executable, str(project / config["run_file"]),
            "--config", str(project / config["config_name"]), "--smoke",
        ]
        log_path = work / "train.stdout.log"
        error_path = work / "train.stderr.log"
        with log_path.open("w", encoding="utf-8") as stdout, error_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.run(
                command, cwd=project, stdout=stdout, stderr=stderr,
                timeout=config.get("timeout_seconds", 600), check=False,
            )
        checks[1] = (
            names[1], process.returncode == 0,
            f"exit={process.returncode}; stdout={log_path}; stderr={error_path}; work={project}",
        )
        if process.returncode != 0:
            return checks
        _SMOKE_RUN["trained"] = True
        index = 2
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for metric_index, name in enumerate(config["printed_metrics"], start=2):
            checks[metric_index] = _printed_metric_leaf(text, name, "smoke_training")
    except Exception:
        checks[index] = (checks[index][0], False, traceback.format_exc())
    return checks


def check_files(config, caller_globals=None, caller_locals=None):
    """Use Base unchanged for source files and current-run smoke artifacts."""
    if config.get("phase") != "smoke_train":
        return base_evaluator.check_files(config, caller_globals, caller_locals)
    if not _SMOKE_RUN.get("trained"):
        return [("smoke_train.artifacts", False, "No successful current smoke training")]
    project = _SMOKE_RUN["project"].resolve()
    relative_files = config["required_files"]
    paths = [(project / relative).resolve() for relative in relative_files]
    if any(project not in path.parents for path in paths):
        return [("smoke_train.artifacts", False,
                 "Artifact path escaped the current training workspace")]
    bound_files = list(map(str, paths))
    checks = base_evaluator.check_files(dict(config, required_files=bound_files),
                                        caller_globals, caller_locals)
    return [(f"{config['phase']}.{relative}", passed, detail)
            for relative, (_, passed, detail) in zip(relative_files, checks)]


def check_infer_execution(config, caller_globals=None, caller_locals=None):
    """Base execution flow adapted to eval.py CLI and training-run workspace."""
    names = ("inference_execution:checkpoint_input",
             "inference_execution:execution")
    checks = [(name, False, "Not executed") for name in names]
    checks += [(f"inference_execution:printed_metric:{name}", False, "Not executed")
               for name in config["printed_metrics"]]
    index = 0
    try:
        if not _SMOKE_RUN.get("trained"):
            raise RuntimeError("Current smoke training must succeed before inference")
        project = _SMOKE_RUN["project"]
        missing = [name for name in config["checkpoint_files"]
                   if not (project / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Current training checkpoints missing: {missing}")
        checks[0] = (names[0], True, "Uses checkpoints from this smoke run")
        index = 1
        _SMOKE_RUN["inferred"] = False
        # Remove the training metrics in the temporary copy only. Evaluation
        # must produce its own results, not pass on stale training output.
        result_path = project / config["output_path"]
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable, str(project / config["run_file"]),
            "--config", str(project / config["config_name"]), "--smoke",
        ]
        log_path = _SMOKE_RUN["work"] / "eval.stdout.log"
        error_path = _SMOKE_RUN["work"] / "eval.stderr.log"
        with log_path.open("w", encoding="utf-8") as stdout, error_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.run(
                command, cwd=project, stdout=stdout, stderr=stderr,
                timeout=config.get("timeout_seconds", 600), check=False,
            )
        checks[1] = (
            names[1], process.returncode == 0,
            f"exit={process.returncode}; stdout={log_path}; stderr={error_path}",
        )
        if process.returncode != 0:
            return checks
        _SMOKE_RUN["inferred"] = True
        index = 2
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for metric_index, name in enumerate(config["printed_metrics"], start=2):
            checks[metric_index] = _printed_metric_leaf(text, name, "inference_execution")
    except Exception:
        checks[index] = (checks[index][0], False, traceback.format_exc())
    return checks


def check_infer_output_schema(config, caller_globals=None, caller_locals=None):
    """Reuse Base parsing; bind to current run and require the exact JSON file."""
    try:
        if not _SMOKE_RUN.get("inferred"):
            raise RuntimeError("No successful current inference run")
        path = _SMOKE_RUN["project"] / config["output_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Required JSON missing: {path}")
        # Do not accept Base's sibling .pkl fallback for the Prompt's JSON.
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result, dict) or not result:
            raise TypeError("metrics.json must be a nonempty JSON object")
        bound = dict(config, output_path=str(path))
        return base_evaluator.check_infer_output_schema(bound)
    except Exception:
        return [("inference_output:json_contract", False, traceback.format_exc())]


def check_infer_output_validity(config, caller_globals=None, caller_locals=None):
    """Validate metric values, not their scientific formulas or paper accuracy."""
    checks = []
    try:
        if not _SMOKE_RUN.get("inferred"):
            raise RuntimeError("No successful current inference run")
        path = _SMOKE_RUN["project"] / config["output_path"]
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        for key in config["required_metrics"]:
            value = result.get(key)
            valid = (
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0
            )
            checks.append((
                f"inference_output:metric:{key}", valid,
                f"{key} must be finite, numeric and nonnegative; actual={value!r}",
            ))
    except Exception:
        checks.append(("inference_output:metric_load", False, traceback.format_exc()))
    return checks
