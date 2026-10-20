#!/usr/bin/env python3
"""Run the ordered rule checks declared by an evaluation case."""

from __future__ import annotations

import os
import argparse
import importlib.util
import inspect
import json
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATOR_DIR = (SCRIPT_DIR.parent / "benchmarks" / "evaluators").resolve()
BASE_EVALUATOR_DIR = (SCRIPT_DIR.parent / "benchmarks" / "evaluators" / "base").resolve()
RESULT_FILENAME = "rule_eval.json"


class RuleEvalError(ValueError):
    """Raised when the case or evaluator cannot be used."""


def _safe(value: Any) -> Any:
    """Convert evaluator values into JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(item) for item in value]
    for converter_name in ("item", "tolist"):
        converter = getattr(value, converter_name, None)
        if callable(converter):
            try:
                converted = converter()
            except Exception:
                continue
            if converted is not value:
                return _safe(converted)
    return repr(value)


def load_case(path: str | Path) -> dict[str, Any]:
    """Load and validate one evaluation case JSON object."""

    case_path = Path(path).expanduser()
    try:
        value = json.loads(case_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuleEvalError(f"case JSON does not exist: {case_path}") from exc
    except OSError as exc:
        raise RuleEvalError(f"cannot read case JSON {case_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuleEvalError(f"invalid case JSON {case_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuleEvalError("case JSON root must be an object")
    return value


def _evaluation_script_name(case: Mapping[str, Any]) -> str:
    dynamic = case.get("dynamic_eval")
    if isinstance(dynamic, Mapping) and dynamic.get("evaluation_script") is not None:
        value = dynamic.get("evaluation_script")
    else:
        value = case.get("evaluation_script")
    if not isinstance(value, str) or not value.strip():
        raise RuleEvalError("case JSON must define a non-empty evaluation_script")
    return value.strip()


def resolve_evaluation_script(case: Mapping[str, Any], evaluator_dir: str | Path = BASE_EVALUATOR_DIR) -> Path:
    """Resolve ``evaluation_script`` below the repository evaluator directory."""

    name = _evaluation_script_name(case)
    if name != "base_evaluator.py":
        evaluator_dir = EVALUATOR_DIR
    root = Path(evaluator_dir).expanduser().resolve()
    requested = Path(name)
    # The case stores a file name, not an arbitrary path.  Restricting it to
    # the configured directory also prevents accidental path traversal.
    if requested.is_absolute() or requested.name != name:
        raise RuleEvalError(
            f"evaluation_script must be a file name in {root}: {name}"
        )
    script_path = (root / requested).resolve()
    if script_path.parent != root:
        raise RuleEvalError(
            f"evaluation_script must be located in {root}: {name}"
        )
    if not script_path.is_file():
        raise RuleEvalError(f"evaluation script does not exist: {script_path}")
    return script_path


def load_evaluation_module(path: str | Path) -> Any:
    """Import an evaluator module from a resolved file path."""

    script_path = Path(path).expanduser().resolve()
    if not script_path.is_file():
        raise RuleEvalError(f"evaluation script does not exist: {script_path}")
    module_name = f"rule_eval_{script_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuleEvalError(f"cannot load evaluation script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    module_dir = str(script_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuleEvalError(
            f"evaluation script import failed: {type(exc).__name__}: {exc}"
        ) from exc
    return module


def get_checks(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read the ordered check definitions from a case JSON object."""

    checks: Any = case.get("checks")
    if checks is None:
        dynamic = case.get("dynamic_eval")
        if isinstance(dynamic, Mapping):
            checks = dynamic.get("checks")
    if checks is None and isinstance(case.get("rule_eval"), Mapping):
        checks = case["rule_eval"].get("checks")
    if not isinstance(checks, list):
        raise RuleEvalError("case JSON checks must be an array")
    normalized: list[dict[str, Any]] = []
    for index, check in enumerate(checks):
        if not isinstance(check, Mapping):
            raise RuleEvalError(f"checks[{index}] must be an object")
        normalized.append(dict(check))
    return normalized


def _method_names(check: Mapping[str, Any]) -> list[str]:
    """Return method names, accepting the historical ``methods`` spelling."""

    value = check.get("method", check.get("methods"))
    if isinstance(value, str):
        names = [value.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        names = [item.strip() for item in value if isinstance(item, str)]
    else:
        names = []
    names = [name for name in names if name]
    if not names:
        raise RuleEvalError("each check must define a non-empty method")
    return names


def _find_method(module: Any, name: str) -> Any:
    """Find a callable by a dotted attribute name in the evaluator module."""

    value: Any = module
    for part in name.split("."):
        if not part or not hasattr(value, part):
            raise RuleEvalError(f"method {name!r} was not found in evaluation script")
        value = getattr(value, part)
    if inspect.isclass(value):
        value = value()
    if not callable(value):
        raise RuleEvalError(f"method {name!r} is not callable")
    return value


def _params_call(method: Any, params: Any, context: Mapping[str, Any]) -> Any:
    """Call a check method using its JSON ``params`` configuration.

    Mappings are keyword arguments, sequences are positional arguments, and
    scalar values are passed as one positional argument.  ``args``/``kwargs``
    are supported for methods that need both forms.  Runtime context is added
    only when the method explicitly declares the corresponding parameter.
    """

    if params is None:
        params = {}
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    plain_mapping = isinstance(params, Mapping) and "args" not in params and "kwargs" not in params
    signature = inspect.signature(method)

    # The evaluator template exposes checks as ``check(config, ... )``.  Pass
    # the complete JSON params mapping as that config object; do not expand it
    # into unrelated keyword arguments such as ``required_files``.
    first_parameter = next(iter(signature.parameters.values()), None)
    config_style = (
        plain_mapping
        and first_parameter is not None
        and first_parameter.name in {"config", "params", "configuration"}
        and first_parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    if config_style:
        args = [params]
        plain_mapping = False
    elif isinstance(params, Mapping):
        raw_args = params.get("args", ())
        if isinstance(raw_args, (str, bytes, bytearray)) or not isinstance(raw_args, Sequence):
            raise RuleEvalError("params.args must be an array")
        args = list(raw_args)
        raw_kwargs = params.get("kwargs", {})
        if not isinstance(raw_kwargs, Mapping):
            raise RuleEvalError("params.kwargs must be an object")
        kwargs = dict(raw_kwargs)
        if plain_mapping:
            kwargs = dict(params)
    elif isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
        args = list(params)
    else:
        args = [params]

    positional_parameters = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ][:len(args)]
    for key, value in context.items():
        if key not in kwargs and key in signature.parameters and key not in positional_parameters:
            kwargs[key] = value

    # Custom checks may intentionally accept the execution context through
    # ``**kwargs`` instead of naming each namespace parameter explicitly.
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD
           for parameter in signature.parameters.values()):
        for key, value in context.items():
            if key not in positional_parameters:
                kwargs.setdefault(key, value)

    try:
        signature.bind(*args, **kwargs)
    except TypeError:
        # A method declared as ``check(params)`` should also be able to receive
        # a mapping as one configuration object when keyword binding does not
        # match its signature.
        if plain_mapping and not args and len(signature.parameters) == 1:
            return method(params)
        raise
    return method(*args, **kwargs)


def _normalize_results(value: Any) -> list[list[Any]]:
    """Validate and JSON-normalize a method's ``(info, ok, detail)`` list."""

    if not isinstance(value, list):
        raise RuleEvalError(
            "check method must return a list of (check_info, bool, detail) tuples"
        )
    normalized: list[list[Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise RuleEvalError(
                f"check result at index {index} must contain check_info, bool, and detail"
            )
        check_info, passed, detail = item
        if not isinstance(passed, bool):
            raise RuleEvalError(
                f"check result at index {index} has a non-boolean status"
            )
        normalized.append([_safe(check_info), passed, _safe(detail)])
    return normalized


def _dependency_names(check: Mapping[str, Any]) -> list[str]:
    """Return the preceding check IDs declared by ``check``.

    ``depdends`` is the case-file spelling used by the evaluation datasets.
    The correctly-spelled aliases are retained for older cases, but all
    variants have the same contract: an array containing check ``id`` values.
    """

    value: Any = None
    for field in ("depends", "depends_on", "depdends_on"):
        if field in check:
            value = check[field]
            break
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuleEvalError("check depdends must be an array of check IDs")
    if not all(isinstance(item, str) for item in value):
        raise RuleEvalError("check depdends must contain only string IDs")
    return [item.strip() for item in value if item.strip()]


def _check_passed(result: Mapping[str, Any]) -> bool:
    """Return whether a completed check has at least one all-true result."""

    results = result.get("results")
    if not isinstance(results, list) or not results:
        return False
    return all(
        isinstance(item, (list, tuple))
        and len(item) == 3
        and item[1] is True
        for item in results
    )


def _result_identifiers(result: Mapping[str, Any]) -> set[str]:
    """Return the check ID exposed by a prior result.

    Dependencies deliberately do not match ``check_type``, display name, or
    method name: those fields are not unique in a case and are not the
    dependency contract.
    """

    value = result.get("id")
    return {value} if isinstance(value, str) and value else set()


def _dependency_failure(
    check: Mapping[str, Any],
    names: list[str],
    previous_results: Sequence[Mapping[str, Any]],
) -> str | None:
    """Explain why a check cannot run because a preceding check failed."""

    for dependency in names:
        matches = [
            result
            for result in previous_results
            if dependency in _result_identifiers(result)
        ]
        if not matches:
            return f"dependency check {dependency!r} was not found among preceding checks"
        if not _check_passed(matches[-1]):
            return f"dependency check {dependency!r} did not pass"
    return None


def _run_one_check(
    module: Any,
    check: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    artifact_path: Path,
    previous_results: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    names = _method_names(check)
    check_id = check.get("id")
    if check_id is not None and not isinstance(check_id, str):
        raise RuleEvalError("check id must be a string when provided")
    check_id = check_id.strip() if isinstance(check_id, str) else None
    check_type = check.get("type", None) or check.get("check_type", None) or  names[0]
    phase = check.get("phase", "")

    dependencies = _dependency_names(check)
    dependency_reason = _dependency_failure(check, dependencies, previous_results)
    if dependency_reason is not None:
        return {
            "id": check_id,
            "check_type": check_type,
            "phase": phase,
            "method": names[0] if len(names) == 1 else names,
            "depends": dependencies,
            "results": [[check_type, False, dependency_reason]],
            "score": 0.0,
            "summary": {"total": 1, "passed": 0, "failed": 1},
        }

    raw_params = check.get("params", {})
    if isinstance(raw_params, Mapping):
        params: Any = dict(raw_params)
        if check.get("phase") is not None:
            params.setdefault("phase", check["phase"])
    else:
        params = raw_params
    all_results: list[list[Any]] = []
    errors: list[str] = []
    # These namespaces are supplied to every check invocation.  The
    # ``caller_*`` names match the evaluator template; the short aliases make
    # the same context available to custom check functions that declare
    # ``locals``/``globals`` parameters.
    caller_globals = globals()
    caller_locals = locals()
    context = {
        # Check functions in the evaluator template use ``config`` as their
        # first argument.  Keep ``params`` as an alias for custom checks.
        "config": params,
        "artifact_path": str(artifact_path),
        "submission_dir": str(artifact_path),
        "case": case,
        "check": check,
        "params": params,
        "caller_globals": caller_globals,
        "caller_locals": caller_locals,
        "globals": caller_globals,
        "locals": caller_locals,
    }
    for name in names:
        try:
            value = _params_call(_find_method(module, name), params, context)
            all_results.extend(_normalize_results(value))
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            all_results.append([name, False, str(exc)])

    total = len(all_results)
    passed = sum(item[1] is True for item in all_results)
    result: dict[str, Any] = {
        "id": check_id,
        "check_type": check_type,
        "phase": phase,
        "method": names[0] if len(names) == 1 else names,
        "depends": dependencies,
        "results": all_results,
        "score": passed / total if total else 0.0,
        "summary": {"total": total, "passed": passed, "failed": total - passed},
    }
    if errors:
        result["errors"] = errors
    return result


def _case_info(case: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("id", "task", "eval_type", "source_family", "source_variant")
    return {field: _safe(case[field]) for field in fields if field in case}

def build_candidate_config(
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Build candidate runtime configuration from case JSON.

    Example:
        id = functional_attention_V1
        stardard_dir_name = functional_attention_repro

    Produces:
        candidate_env =
            FUNCTIONAL_ATTENTION_SUBMISSION_DIR

        candidate_dir_hints =
            ["functional_attention", "functional_attention_V1"]

        stardard_name =
            functional_attention_repro
    """

    case_id = case.get("id")
    stardard_dir_name = case.get("stardard_dir_name")

    if not isinstance(case_id, str) or not case_id.strip():
        raise RuleEvalError(
            "case JSON must define a non-empty string 'id'"
        )

    if (
        not isinstance(stardard_dir_name, str)
        or not stardard_dir_name.strip()
    ):
        raise RuleEvalError(
            "case JSON must define a non-empty string "
            "'stardard_dir_name'"
        )

    case_id = case_id.strip()
    stardard_dir_name = stardard_dir_name.strip()

    # functional_attention_V1 -> functional_attention
    if "_" in case_id:
        candidate_name = case_id.rsplit("_", 1)[0]
    else:
        candidate_name = case_id

    # functional_attention -> FUNCTIONAL_ATTENTION
    candidate_env_prefix = candidate_name.upper()

    candidate_env = (
        f"{candidate_env_prefix}_SUBMISSION_DIR"
    )

    candidate_dir_hints = [
        candidate_name,
        case_id,
    ]

    return {
        "candidate_env": candidate_env,
        "candidate_dir_hints": candidate_dir_hints,
        "stardard_name": stardard_dir_name,
    }


def _matches_candidate_hint(path_value: str, hints: Sequence[str]) -> bool:
    """Return whether a path is a candidate directory named by ``hints``."""

    if not path_value:
        return False
    name = Path(path_value).expanduser().name
    return any(
        name == hint or name.startswith(f"{hint}_V")
        for hint in hints
    )


def _configure_candidate_environment(
    candidate_config: Mapping[str, Any], artifact: Path
) -> None:
    """Remove inherited candidate paths and bind the current artifact."""

    raw_hints = candidate_config.get("candidate_dir_hints", ())
    hints = [str(value) for value in raw_hints]

    # rule_eval may inherit variables exported for an earlier evaluation.
    # Remove every candidate-directory binding before exposing this case.
    for name in list(os.environ):
        if name.endswith("_SUBMISSION_DIR") or name in candidate_config:
            os.environ.pop(name, None)

    python_path_entries = [
        entry
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry and not _matches_candidate_hint(entry, hints)
    ]
    artifact_text = str(artifact)
    python_path_entries.insert(0, artifact_text)
    os.environ["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    # PYTHONPATH is copied into sys.path only when Python starts, so keep the
    # live interpreter in sync before importing the evaluator module.
    sys.path[:] = [
        entry
        for entry in sys.path
        if not _matches_candidate_hint(entry, hints)
        and entry != artifact_text
    ]
    sys.path.insert(0, artifact_text)

    for key, value in candidate_config.items():
        os.environ[key] = str(value)
    os.environ[str(candidate_config["candidate_env"])] = artifact_text


def run_case(case_path: str | Path, artifact_path: str | Path, eval_output_dir: str | Path) -> dict[str, Any]:
    """Run all checks and write ``<eval_output_dir>/rule_eval.json``."""

    case_file = Path(case_path).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    output_dir = Path(eval_output_dir).expanduser().resolve()
    case = load_case(case_file)
    if not artifact.is_dir():
        raise RuleEvalError(f"artifact path is not a directory: {artifact}")

    candidate_config = build_candidate_config(case)
    _configure_candidate_environment(candidate_config, artifact)

    script_path = resolve_evaluation_script(case)
    module = load_evaluation_module(script_path)
    checks = get_checks(case)
    # Checks are intentionally evaluated in declaration order: a dependency
    # may only refer to a check that has already produced a result.
    check_results: list[dict[str, Any]] = []
    for check in checks:
        check_results.append(
            _run_one_check(
                module,
                check,
                case=case,
                artifact_path=artifact,
                previous_results=check_results,
            )
        )
    result: dict[str, Any] = {
        "case_info": _case_info(case),
        "id": _safe(case.get("id")),
        "task": _safe(case.get("task")),
        "eval_type": _safe(case.get("eval_type")),
        "case_path": str(case_file),
        "artifact_path": str(artifact),
        "evaluation_script": _evaluation_script_name(case),
        "evaluation_script_path": str(script_path),
        "execution_status": "finished",
        "checks": check_results,
        "passed": bool(check_results) and all(item["score"] == 1.0 for item in check_results),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / RESULT_FILENAME
    result["result_path"] = str(output_path)
    output_path.write_text(
        json.dumps(_safe(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _failure_result(
    case_path: str | Path,
    message: str,
    *,
    case: Mapping[str, Any] | None = None,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    case = case or {}
    result = {
        "case_info": _case_info(case),
        "id": _safe(case.get("id")),
        "task": _safe(case.get("task")),
        "eval_type": _safe(case.get("eval_type")),
        "case_path": str(Path(case_path).expanduser().resolve()),
        "artifact_path": (
            str(Path(artifact_path).expanduser().resolve())
            if artifact_path is not None
            else None
        ),
        "execution_status": "failed",
        "passed": False,
        "checks": [],
        "error": message,
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-path", required=True, type=str)
    parser.add_argument("--artifact-path", required=True, type=str)
    parser.add_argument("--eval-output-dir", required=True, type=str)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    case: dict[str, Any] = {}
    try:
        case = load_case(args.case_path)
    except (RuleEvalError, OSError, TypeError, ValueError):
        pass
    try:
        result = run_case(args.case_path, args.artifact_path, args.eval_output_dir)
    except (RuleEvalError, OSError, TypeError, ValueError) as exc:
        result = _failure_result(
            args.case_path,
            str(exc),
            case=case,
            artifact_path=args.artifact_path,
        )
        output_path = Path(args.eval_output_dir).expanduser().resolve() / RESULT_FILENAME
        result["result_path"] = str(output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(_safe(result), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as write_exc:
            print(f"failed to write result: {write_exc}", file=sys.stderr)
            return 2
        print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
