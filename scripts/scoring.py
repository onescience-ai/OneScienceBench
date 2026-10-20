#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


class DynamicEvalScoreError(ValueError):
    pass


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DynamicEvalScoreError(f"input JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DynamicEvalScoreError(
            f"input JSON is invalid at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise DynamicEvalScoreError("input JSON root must be an object")
    return data


def parse_weights_json(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None

    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DynamicEvalScoreError(
            f"--weights-json must be a JSON object: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise DynamicEvalScoreError("--weights-json must be a JSON object")
    return validate_weights(data)


def validate_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for check_type, value in weights.items():
        if not isinstance(check_type, str) or not check_type.strip():
            raise DynamicEvalScoreError("weight keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DynamicEvalScoreError(f"weight for {check_type!r} must be a number")

        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise DynamicEvalScoreError(
                f"weight for {check_type!r} must be finite and greater than zero"
            )
        normalized[check_type] = weight
    return normalized


def _parse_check_item(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Parse one rule_eval check and score it by its result pass ratio."""
    check_id = item.get("id")
    if not isinstance(check_id, str) or not check_id.strip():
        raise DynamicEvalScoreError(f"checks[{index}].id must be a non-empty string")

    check_type = item.get("check_type")
    if not isinstance(check_type, str) or not check_type.strip():
        raise DynamicEvalScoreError(
            f"checks[{index}].check_type must be a non-empty string"
        )

    raw_results = item.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise DynamicEvalScoreError(
            f"checks[{index}].results must be a non-empty array"
        )

    passed = 0
    failed_results: list[Any] = []
    for result_index, result_item in enumerate(raw_results):
        if not isinstance(result_item, (list, tuple)) or len(result_item) < 2:
            raise DynamicEvalScoreError(
                f"checks[{index}].results[{result_index}] must contain a status"
            )
        status = result_item[1]
        if not isinstance(status, bool):
            raise DynamicEvalScoreError(
                f"checks[{index}].results[{result_index}][1] must be a boolean"
            )
        if status:
            passed += 1
        else:
            failed_results.append(
                result_item[2] if len(result_item) > 2 else result_item
            )

    return {
        "id": check_id,
        "check_type": check_type.strip(),
        "phase": item.get("phase"),
        "passed_results": passed,
        "total_results": len(raw_results),
        "score": passed / len(raw_results),
        "failed_results": failed_results,
    }


def calculate_check_scores(
    result: Mapping[str, Any],
    weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks = result.get("checks")
    if not isinstance(checks, list) or not checks:
        raise DynamicEvalScoreError("checks must be a non-empty array")

    parsed_checks: list[dict[str, Any]] = []
    failed_checks: list[dict[str, Any]] = []

    for index, item in enumerate(checks):
        if not isinstance(item, dict):
            raise DynamicEvalScoreError(f"checks[{index}] must be an object")

        parsed = _parse_check_item(item, index)
        parsed_checks.append(parsed)
        if parsed["failed_results"]:
            failed_checks.append(
                {
                    "id": parsed["id"],
                    "check_type": parsed["check_type"],
                    "phase": parsed["phase"],
                    "failed_results": parsed["failed_results"],
                }
            )

    check_ids = [parsed["id"] for parsed in parsed_checks]
    check_types = [parsed["check_type"] for parsed in parsed_checks]
    if weights is None:
        raw_weights = {check_id: 1.0 for check_id in check_ids}
        weight_for = lambda parsed: raw_weights[parsed["id"]]
        weight_source = "equal"
    else:
        raw_weights = validate_weights(weights)
        observed_ids = set(check_ids)
        observed_types = set(check_types)
        configured = set(raw_weights)
        if observed_ids <= configured:
            weight_for = lambda parsed: raw_weights[parsed["id"]]
            missing = sorted(observed_ids - configured)
            unknown = sorted(configured - observed_ids)
            weight_source = "configured_by_id"
        elif observed_types <= configured:
            weight_for = lambda parsed: raw_weights[parsed["check_type"]]
            missing = sorted(observed_types - configured)
            unknown = sorted(configured - observed_types)
            weight_source = "configured_by_check_type"
        else:
            missing = sorted(observed_ids - configured)
            unknown = sorted(configured - observed_ids)
            details: list[str] = []
            if missing:
                details.append(f"missing check ids: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown check ids: {', '.join(unknown)}")
            raise DynamicEvalScoreError("invalid weights; " + "; ".join(details))

    total_weight = sum(weight_for(parsed) for parsed in parsed_checks)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise DynamicEvalScoreError("sum of check weights must be greater than zero")

    check_scores: list[dict[str, Any]] = []
    overall_score = 0.0

    for parsed in parsed_checks:
        score = parsed["score"]
        normalized_weight = weight_for(parsed) / total_weight
        weighted_score = normalized_weight * score
        overall_score += weighted_score
        check_scores.append(
            {
                "id": parsed["id"],
                "check_type": parsed["check_type"],
                "phase": parsed["phase"],
                "passed_results": parsed["passed_results"],
                "total_results": parsed["total_results"],
                "score": round(score, 6),
                "weight": round(normalized_weight, 6),
                "weighted_score": round(weighted_score, 6),
            }
        )

    evaluator_passed = result.get("passed")
    if "passed" in result and not isinstance(evaluator_passed, bool):
        raise DynamicEvalScoreError("top-level passed must be a boolean when provided")

    checks_passed = math.isclose(overall_score, 1.0, rel_tol=0.0, abs_tol=1e-12)
    passed = checks_passed and evaluator_passed is not False

    return {
        "passed": passed,
        "evaluator_passed": evaluator_passed,
        "pass_threshold": 1.0,
        "overall_score": round(overall_score, 6),
        "weight_source": weight_source,
        "total_checks": len(checks),
        "passed_checks": sum(
            math.isclose(parsed["score"], 1.0, rel_tol=0.0, abs_tol=1e-12)
            for parsed in parsed_checks
        ),
        "failed_checks": failed_checks,
        "check_scores": check_scores,
    }


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score each evaluation check by its results pass ratio."
    )
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument(
        "--weights-json",
        help=(
            "Optional JSON object mapping every observed check id or check_type to a weight, "
            'for example \'{"file": 1, "model": 2}\'.'
        ),
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = calculate_check_scores(
            load_json_object(args.input_json),
            weights=parse_weights_json(args.weights_json),
        )
        if args.output_json is not None:
            write_json(args.output_json, result)
    except (DynamicEvalScoreError, OSError) as exc:
        print(
            json.dumps(
                {"status": "failed", "passed": False, "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
