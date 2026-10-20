#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator


TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
IDENTITY_FIELDS = ("agent_id", "agent_version", "base_model")
OPENSCIENCE_AGENT_IDS = {"openscience", "open-science", "open_science"}


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{description} does not exist: {path}") from exc
    except OSError as exc:
        raise ValueError(f"unable to read {description} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {description} {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{description} JSON root must be an object: {path}")
    return data


def load_case_id(case_path: Path) -> str:
    case_data = load_json_object(case_path, "case file")
    case_id = case_data.get("id") or case_data.get("task") or case_path.stem
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case id must be a non-empty string")
    return case_id.strip()


def iter_log_records(log_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        log_file = log_path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"agent stdout log does not exist: {log_path}") from exc
    except OSError as exc:
        raise ValueError(f"unable to read agent stdout log {log_path}: {exc}") from exc

    with log_file:
        for line_number, line in enumerate(log_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield line_number, record


def parse_usage(usage: Any, line_number: int) -> int:
    if not isinstance(usage, dict):
        raise ValueError(f"usage on log line {line_number} must be an object")

    total = 0
    for field in TOKEN_FIELDS:
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"usage.{field} on log line {line_number} must be a non-negative integer"
            )
        total += value
    return total


def parse_identity_message(text: str) -> dict[str, str] | None:
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(message, dict):
        return None

    identity: dict[str, str] = {}
    for field in IDENTITY_FIELDS:
        value = message.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
        identity[field] = value.strip()

    logged_case_id = message.get("case_id")
    if isinstance(logged_case_id, str) and logged_case_id.strip():
        identity["case_id"] = logged_case_id.strip()
    return identity


def parse_agent_identity(record: dict[str, Any]) -> dict[str, str] | None:
    text: Any = None
    if record.get("type") == "item.completed":
        item = record.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
    elif record.get("type") == "text":
        part = record.get("part")
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")

    if not isinstance(text, str):
        return None
    return parse_identity_message(text)


def fallback_agent_identity(eval_output_dir: Path) -> dict[str, str] | None:
    identity: dict[str, str] = {}
    for field, env_name in (
        ("agent_id", "EVAL_AGENT_ID"),
        ("agent_version", "EVAL_AGENT_VERSION"),
        ("base_model", "EVAL_BASE_MODEL"),
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            identity[field] = value

    env_case_id = os.environ.get("EVAL_CASE_ID", "").strip()
    if env_case_id:
        identity["case_id"] = env_case_id

    if not all(identity.get(field) for field in IDENTITY_FIELDS):
        result_path = eval_output_dir / "agent_result.json"
        if result_path.is_file():
            try:
                result = load_json_object(result_path, "agent result")
            except ValueError:
                result = {}
            for field in IDENTITY_FIELDS:
                value = result.get(field)
                if field not in identity and isinstance(value, str) and value.strip():
                    identity[field] = value.strip()
            case_id = result.get("case_id")
            if "case_id" not in identity and isinstance(case_id, str) and case_id.strip():
                identity["case_id"] = case_id.strip()

    if all(identity.get(field) for field in IDENTITY_FIELDS):
        return identity
    return None


def parse_openscience_tokens(tokens: Any, line_number: int) -> int:
    if not isinstance(tokens, dict):
        raise ValueError(f"tokens on log line {line_number} must be an object")

    def total(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"tokens on log line {line_number} must contain non-negative integers")
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"tokens on log line {line_number} must contain non-negative integers")
            return value
        if isinstance(value, dict):
            return sum(total(item) for item in value.values())
        if value is None:
            return 0
        raise ValueError(f"tokens on log line {line_number} must contain non-negative integers")

    return total(tokens)


def collect_log_data(
    log_path: Path, fallback_identity: dict[str, str] | None = None
) -> tuple[int, dict[str, str]]:
    cost_token = 0
    usage_records = 0
    identity: dict[str, str] | None = None

    for line_number, record in iter_log_records(log_path):
        if record.get("type") == "turn.completed" and "usage" in record:
            cost_token += parse_usage(record["usage"], line_number)
            usage_records += 1

        parsed_identity = parse_agent_identity(record)
        if parsed_identity is not None:
            identity = parsed_identity

    if usage_records == 0:
        raise ValueError(f"no turn.completed usage record found in {log_path}")
    if identity is None:
        identity = fallback_identity
    if identity is None:
        raise ValueError(
            f"no agent identity found in {log_path}, environment, or agent_result.json"
        )
    return cost_token, identity


def collect_openscience_log_data(log_path: Path) -> tuple[int, dict[str, str]]:
    cost_token = 0
    records = 0
    for line_number, record in iter_log_records(log_path):
        if record.get("type") == "done" and "tokens" in record:
            cost_token += parse_openscience_tokens(record["tokens"], line_number)
            records += 1
    if records == 0:
        raise ValueError(f"no OpenScience done.tokens record found in {log_path}")
    return cost_token, {
        "agent_id": os.environ.get("EVAL_AGENT_ID", "openscience") or "openscience",
        "agent_version": os.environ.get("EVAL_AGENT_VERSION", "unknown") or "unknown",
        "base_model": os.environ.get("EVAL_BASE_MODEL", "unknown") or "unknown",
    }


def build_result(case_path: Path, eval_output_dir: Path) -> dict[str, Any]:
    case_id = load_case_id(case_path)
    log_path = eval_output_dir / "logs" / "agent_stdout.log"
    agent_id = os.environ.get("EVAL_AGENT_ID", "").strip().lower()
    if agent_id in OPENSCIENCE_AGENT_IDS:
        cost_token, identity = collect_openscience_log_data(log_path)
    else:
        identity_fallback = fallback_agent_identity(eval_output_dir)
        cost_token, identity = collect_log_data(log_path, identity_fallback)

    logged_case_id = identity.get("case_id")
    if logged_case_id is not None and logged_case_id != case_id:
        raise ValueError(
            f"case id mismatch: case file has {case_id!r}, log has {logged_case_id!r}"
        )

    return {
        "case_id": case_id,
        "agent_id": identity["agent_id"],
        "agent_version": identity["agent_version"],
        "base_model": identity["base_model"],
        "cost_token": cost_token,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Codex token usage from an evaluation agent stdout log."
    )
    parser.add_argument("--case-path", required=True, type=Path)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--eval-output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_result(args.case_path, args.eval_output_dir)
        args.eval_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.eval_output_dir / "token_count.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
