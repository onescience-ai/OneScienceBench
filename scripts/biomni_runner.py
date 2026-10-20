#!/usr/bin/env python3
"""Expose Biomni A1 through parallel_eval.py's CLI-agent contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from numbers import Integral
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


logger = logging.getLogger("biomni_runner")


def _setup_logging() -> None:
    level = os.environ.get("BIOMNI_RUNNER_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    if not root.handlers:
        root.addHandler(handler)
    logging.getLogger("httpx2").setLevel(logging.WARNING)


DEFAULT_AGENT_CONFIG: dict[str, Any] = {
    "data_path": "./data",
    "source": "Custom",
    "base_url_env": "BASE_URL",
    "api_key_env": "API_KEY",
    "skip_data_lake_download": False,
    "timeout_seconds": 600,
    # Streaming keeps long generations active through proxies that enforce a
    # read timeout on requests which produce no bytes for too long.
    "stream": True,
}


@dataclass(frozen=True)
class BiomniConfig:
    """Validated runtime configuration for one Biomni invocation."""

    data_path: str
    source: str
    base_url_env: str
    api_key_env: str
    skip_data_lake_download: bool
    timeout_seconds: int
    stream: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BiomniConfig":
        values = {
            key: getattr(args, key) if getattr(args, key) is not None else value
            for key, value in DEFAULT_AGENT_CONFIG.items()
        }
        # Permit deployment-specific configuration without putting credentials
        # or endpoint details in an evaluation profile.
        if getattr(args, "stream", None) is None:
            raw_stream = os.environ.get("BIOMNI_STREAM")
            if raw_stream is not None:
                normalized = raw_stream.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    values["stream"] = True
                elif normalized in {"0", "false", "no", "off"}:
                    values["stream"] = False
                else:
                    raise ValueError(
                        "BIOMNI_STREAM must be one of: 1, 0, true, false, yes, no, on, off"
                    )
        config = cls(**values)
        if config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if not config.data_path.strip():
            raise ValueError("data_path must not be empty")
        if not config.source.strip():
            raise ValueError("source must not be empty")
        if not config.base_url_env.strip():
            raise ValueError("base_url_env must not be empty")
        if not config.api_key_env.strip():
            raise ValueError("api_key_env must not be empty")
        return config

    def base_url(self) -> str:
        value = os.environ.get(self.base_url_env, "").strip()
        if not value:
            raise RuntimeError(f"Biomni base URL is missing; set {self.base_url_env}")
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise RuntimeError(f"Biomni API key is missing; set {self.api_key_env}")
        return value


def _biomni_version() -> str:
    try:
        return importlib.metadata.version("biomni")
    except importlib.metadata.PackageNotFoundError:
        try:
            from biomni.version import __version__
        except ImportError as exc:
            raise RuntimeError(
                "Biomni is not installed for this Python executable"
            ) from exc
        return str(__version__)


def _load_schema(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read output schema {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"output schema must be a JSON object: {path}")
    return value


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    candidates = [value.strip()]
    candidates.extend(
        reversed(
            re.findall(
                r"<solution>\s*(.*?)\s*</solution>",
                value,
                re.DOTALL | re.IGNORECASE,
            )
        )
    )
    candidates.extend(
        reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.DOTALL))
    )
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def _coerce_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, Integral):
        return max(0, int(value))
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return 0
    return converted if converted >= 0 else 0


def _first_usage_value(mapping: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in mapping:
            value = _coerce_nonnegative_int(mapping[key])
            if value:
                return value
    return 0


def _nested_mapping(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _extract_usage_mapping(value: Any) -> dict[str, int]:
    """Normalize common OpenAI/LangChain usage metadata shapes."""
    if not isinstance(value, Mapping):
        return {field: 0 for field in _TOKEN_FIELDS}

    candidates: list[Mapping[str, Any]] = [value]
    for key in ("token_usage", "usage", "usage_metadata"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)

    result = {field: 0 for field in _TOKEN_FIELDS}
    for candidate in candidates:
        input_details = _nested_mapping(candidate, "input_token_details", "prompt_tokens_details")
        output_details = _nested_mapping(candidate, "output_token_details", "completion_tokens_details")
        cache_details = _nested_mapping(candidate, "cache", "cache_details")
        current = {
            "input_tokens": _first_usage_value(candidate, "input_tokens", "prompt_tokens", "input", "prompt"),
            "cached_input_tokens": _first_usage_value(
                candidate,
                "cached_input_tokens",
                "cache_read_input_tokens",
                "cache_read",
                "cache_read_tokens",
                "cached_tokens",
            ) or _first_usage_value(
                input_details,
                "cache_read",
                "cached_tokens",
                "cached_input_tokens",
                "read",
            ) or _first_usage_value(
                _nested_mapping(input_details, "cache"),
                "read",
                "cache_read",
                "cached_tokens",
            ) or _first_usage_value(cache_details, "read", "cache_read"),
            "cache_write_input_tokens": _first_usage_value(
                candidate,
                "cache_write_input_tokens",
                "cache_creation_input_tokens",
                "cache_write",
                "cache_creation",
                "cache_write_tokens",
                "cache_creation_tokens",
            ) or _first_usage_value(
                input_details,
                "cache_write",
                "cache_creation",
                "cache_write_input_tokens",
                "cache_creation_input_tokens",
                "cache_write_tokens",
                "cache_creation_tokens",
                "write",
            ) or _first_usage_value(
                _nested_mapping(input_details, "cache"),
                "write",
                "cache_write",
                "cache_creation",
                "write_tokens",
                "cache_write_tokens",
                "creation_tokens",
                "cache_creation_tokens",
            ) or _first_usage_value(
                cache_details,
                "write",
                "cache_write",
                "write_tokens",
                "cache_write_tokens",
                "creation_tokens",
                "cache_creation_tokens",
            ),
            "output_tokens": _first_usage_value(candidate, "output_tokens", "completion_tokens", "output", "completion"),
            "reasoning_output_tokens": _first_usage_value(
                candidate,
                "reasoning_output_tokens",
                "reasoning_tokens",
                "reasoning",
            ) or _first_usage_value(
                output_details,
                "reasoning",
                "reasoning_tokens",
                "reasoning_output_tokens",
            ),
        }
        for field in _TOKEN_FIELDS:
            result[field] = max(result[field], current[field])
    return result


def _extract_usage_from_message(message: Any) -> dict[str, int]:
    """Read usage from LangChain messages and OpenAI-compatible response dicts."""
    sources: list[Any] = []
    if isinstance(message, Mapping):
        sources.extend(message.get(name) for name in ("usage_metadata", "response_metadata", "additional_kwargs", "usage"))
        sources.append(message)
    else:
        for name in ("usage_metadata", "response_metadata", "additional_kwargs", "usage"):
            sources.append(getattr(message, name, None))

    result = {field: 0 for field in _TOKEN_FIELDS}
    for source in sources:
        current = _extract_usage_mapping(source)
        for field in _TOKEN_FIELDS:
            result[field] = max(result[field], current[field])
    return result


def _usage_total(usage: Mapping[str, Any]) -> int:
    return sum(_coerce_nonnegative_int(usage.get(field)) for field in _TOKEN_FIELDS)


def _nonnegative_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


def _is_transient_provider_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "overloaded",
            "try again later",
            "temporarily unavailable",
            "rate limit",
            "too many requests",
            "service unavailable",
            "server busy",
        )
    )


def _is_overload_error(exc: BaseException) -> bool:
    """Return True for capacity/rate-limit errors that warrant long retries."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status in {408, 429, 503}:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "overloaded",
            "try again later",
            "rate limit",
            "too many requests",
            "service unavailable",
            "server busy",
        )
    )


class _SingleActionLLM:
    """Enforce A1's one-action-per-turn protocol at the provider boundary.

    A1 routes ``<think>`` back to its generation node.  The bounded no-action
    counter below prevents a provider that never emits an executable action
    from consuming the graph's entire recursion budget.
    """

    def __init__(
        self,
        llm: Any,
        *,
        stream: bool = True,
        provider_retries: int | None = None,
        provider_retry_backoff: float | None = None,
        max_consecutive_no_action: int | None = None,
    ) -> None:
        self._llm = llm
        self._stream = stream
        self._usage = {field: 0 for field in _TOKEN_FIELDS}
        self._provider_retries = (
            _nonnegative_env_int("BIOMNI_PROVIDER_RETRIES", 2)
            if provider_retries is None
            else provider_retries
        )
        self._provider_retry_backoff = (
            _nonnegative_env_float("BIOMNI_PROVIDER_RETRY_BACKOFF_SECONDS", 2.0)
            if provider_retry_backoff is None
            else provider_retry_backoff
        )
        self._overload_retries = (
            _nonnegative_env_int("BIOMNI_OVERLOAD_RETRIES", 6)
        )
        self._overload_retry_backoff = _nonnegative_env_float(
            "BIOMNI_OVERLOAD_RETRY_BACKOFF_SECONDS", 10.0
        )
        self._max_consecutive_no_action = (
            _nonnegative_env_int("BIOMNI_MAX_CONSECUTIVE_NO_ACTIONS", 10)
            if max_consecutive_no_action is None
            else max_consecutive_no_action
        )
        self._consecutive_no_action = 0
        if self._provider_retries < 0:
            raise ValueError("provider_retries must be non-negative")
        if self._provider_retry_backoff < 0:
            raise ValueError("provider_retry_backoff must be non-negative")
        if self._max_consecutive_no_action < 0:
            raise ValueError("max_consecutive_no_action must be non-negative")
        if stream and hasattr(llm, "stream_usage"):
            try:
                # Custom OpenAI-compatible endpoints do not enable this by
                # default, so request the final usage chunk explicitly.
                llm.stream_usage = True
            except (AttributeError, TypeError, ValueError):
                logger.debug("underlying LLM does not allow enabling stream usage", exc_info=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    @property
    def token_usage(self) -> dict[str, int]:
        """Return usage accumulated across all provider calls made by Biomni."""
        return dict(self._usage)

    def _add_usage(self, usage: Mapping[str, int]) -> None:
        for field in _TOKEN_FIELDS:
            self._usage[field] += usage.get(field, 0)

    def _call_with_retries(self, operation: Callable[[], Any]) -> Any:
        overload = False
        if self._overload_retries > self._provider_retries:
            max_attempts = self._overload_retries + 1
            overload = True
        else:
            max_attempts = self._provider_retries + 1
        for attempt in range(max_attempts):
            try:
                return operation()
            except Exception as exc:
                if attempt >= max_attempts - 1 or not _is_transient_provider_error(exc):
                    raise
                if overload and _is_overload_error(exc):
                    delay = min(self._overload_retry_backoff * (2**attempt), 60.0)
                else:
                    delay = min(self._provider_retry_backoff * (2**attempt), 30.0)
                logger.warning(
                    "transient provider error (%s); retrying request %d/%d in %.1fs",
                    exc,
                    attempt + 1,
                    max_attempts - 1,
                    delay,
                )
                if delay:
                    time.sleep(delay)

    def _stream_response(self, *args: Any, **kwargs: Any) -> tuple[Any, list[Any]]:
        response = None
        usage_chunks: list[Any] = []
        chunks = iter(self._llm.stream(*args, **kwargs))
        try:
            for chunk in chunks:
                usage_chunks.append(chunk)
                response = chunk if response is None else response + chunk
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()
        if response is None:
            raise RuntimeError("streaming LLM returned no response chunks")
        return response, usage_chunks

    @staticmethod
    def _stream_usage(chunks: Sequence[Any]) -> dict[str, int]:
        """Collapse chunk metadata without double-counting cumulative snapshots.

        OpenAI-compatible streams normally expose usage only on the final chunk,
        while some providers expose the same cumulative snapshot more than once.
        Taking the per-field maximum handles both forms and avoids multiplying a
        single response's usage by the number of content chunks.
        """
        usage = {field: 0 for field in _TOKEN_FIELDS}
        for chunk in chunks:
            current = _extract_usage_from_message(chunk)
            for field in _TOKEN_FIELDS:
                usage[field] = max(usage[field], current[field])
        return usage

    @staticmethod
    def _replace_content(response: Any, content: str) -> Any:
        try:
            return response.model_copy(update={"content": content})
        except (AttributeError, TypeError, ValueError):
            try:
                response.content = content
            except (AttributeError, TypeError):
                logger.warning("could not replace provider response content; returning original response")
            return response

    @staticmethod
    def _is_agent_generation_request(args: tuple[Any, ...]) -> bool:
        """Distinguish A1 turns from auxiliary plain-text LLM calls.

        Biomni's tool retriever uses the same LLM wrapper but asks for plain
        ``TOOLS: [...]``/``LIBRARIES: [...]`` text in a single HumanMessage.
        Only the main A1 graph has a SystemMessage containing the XML action
        protocol; applying the no-action guard to retrieval would reject valid
        retrieval output before the agent starts.
        """
        if not args:
            # Keep direct/unit-test calls on the guarded path.
            return True
        messages = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
        if not messages:
            return True
        return any(getattr(message, "type", None) == "system" for message in messages)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        if self._stream:
            response, usage_chunks = self._call_with_retries(
                lambda: self._stream_response(*args, **kwargs)
            )
            self._add_usage(self._stream_usage(usage_chunks))
        else:
            response = self._call_with_retries(lambda: self._llm.invoke(*args, **kwargs))
            self._add_usage(_extract_usage_from_message(response))
        content = response.content
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, Mapping):
                    part = block.get("text") or block.get("content")
                    if isinstance(part, str):
                        text_parts.append(part)
            content = "".join(text_parts)
            if content:
                response = self._replace_content(response, content)
        if not isinstance(content, str):
            return response
        if not self._is_agent_generation_request(args):
            return response
        first = re.search(
            r"<(execute|solution)>.*?</\1>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if first:
            self._consecutive_no_action = 0
            return self._replace_content(response, content[:first.end()])
        if not content.strip():
            raise RuntimeError("LLM returned an empty response")
        # Biomni already understands fenced code blocks and incomplete action
        # tags. Preserve those forms so an intended execution turn is not
        # converted into a final answer by this transport wrapper.
        if "```" in content or re.search(
            r"<(?:execute|solution|think|observation)\b", content, re.IGNORECASE
        ):
            # Explicit action tags and fenced code are handled by A1's parser.
            # A think/observation-only response is deliberately not counted as
            # an action below, because it still routes A1 back to ``generate``.
            if "```" in content or re.search(r"<(?:execute|solution)\b", content, re.IGNORECASE):
                self._consecutive_no_action = 0
            else:
                self._consecutive_no_action += 1
                self._raise_after_no_action_limit(content)
            return response
        self._consecutive_no_action += 1
        self._raise_after_no_action_limit(content)
        # A plain response is usually a model planning/reasoning turn whose
        # required XML tag was omitted. Wrapping it as <solution> terminates
        # A1 before it can execute the plan. <think> keeps the message in the
        # generate path, where A1 asks the provider to emit the next action.
        logger.warning("provider response had no action tag; normalized as <think>")
        return self._replace_content(response, f"<think>\n{content}\n</think>")

    def _raise_after_no_action_limit(self, content: str) -> None:
        """Stop A1 when the provider keeps producing turns it cannot execute.

        A1 routes ``<think>`` back to ``generate``.  If the provider never
        follows that reasoning with an action, the graph can consume its whole
        recursion budget without invoking a single tool.  Failing closed here
        preserves the distinction between an incomplete agent run and a final
        solution while avoiding an unbounded provider/token loop.
        """
        if self._consecutive_no_action <= self._max_consecutive_no_action:
            return
        preview = re.sub(r"\s+", " ", content).strip()[:160]
        logger.error(
            "aborting after %d consecutive provider responses without an action",
            self._consecutive_no_action,
        )
        raise RuntimeError(
            "LLM produced "
            f"{self._consecutive_no_action} consecutive responses without "
            "an <execute> action or <solution>; aborting to avoid an "
            f"unbounded A1 loop (last response={preview!r})"
        )


def _patch_empty_observations() -> None:
    """Ensure biomni execution tools never return empty stdout.

    When a bash script or Python snippet runs successfully but produces no
    stdout, biomni wraps the empty result in ``<observation></observation>``.
    Some LLMs misinterpret the empty observation as "tool unavailable" and
    abort the task.  Patching the tool functions referenced by
    ``biomni.agent.a1`` to return a clear success message prevents this.
    """
    import biomni.agent.a1 as _a1

    _EMPTY_MSG = "[Command executed successfully with no stdout output.]"

    for _fn_name in ("run_bash_script", "run_r_code", "run_python_repl"):
        _orig = getattr(_a1, _fn_name, None)
        if _orig is None or getattr(_orig, "_biomni_runner_patched", False):
            continue

        def _make_wrapper(orig: Any = _orig) -> Any:
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                result = orig(*args, **kwargs)
                if not result or not str(result).strip():
                    return _EMPTY_MSG
                return result

            _wrapper._biomni_runner_patched = True  # type: ignore[attr-defined]
            return _wrapper

        setattr(_a1, _fn_name, _make_wrapper())


def _schema_instruction(schema: Mapping[str, Any] | None) -> str:
    if schema is None:
        return ""
    return (
        "\n\nReturn only one JSON object matching this JSON Schema, without Markdown:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _token_usage(value: Any) -> dict[str, int]:
    return _extract_usage_mapping(value)


def _token_usage_with_total(value: Any) -> dict[str, int]:
    usage = _token_usage(value)
    usage["total_tokens"] = _usage_total(usage)
    return usage


def _emit_result_events(result: dict[str, Any]) -> None:
    message = dict(result)
    message.setdefault("agent_id", os.environ.get("EVAL_AGENT_ID", "biomni"))
    message.setdefault("agent_version", os.environ.get("EVAL_AGENT_VERSION", "unknown"))
    message.setdefault("base_model", os.environ.get("EVAL_BASE_MODEL", "unknown"))
    print(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": json.dumps(message, ensure_ascii=False),
                },
            },
            ensure_ascii=False,
        )
    )
    print(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": _token_usage(message.get("token_usage")),
            },
            ensure_ascii=False,
        )
    )


def _run_agent(agent: Any, prompt: str) -> Any:
    """Run the Biomni agent, logging start/finish/exception with timing."""
    started = time.monotonic()
    prompt_preview = re.sub(r"\s+", " ", prompt).strip()[:200]
    logger.info("agent.go starting; prompt preview=%r", prompt_preview)
    try:
        result = agent.go(prompt)
        logger.info("agent.go finished in %.2fs", time.monotonic() - started)
        return result
    except Exception:
        logger.exception("agent.go failed after %.2fs", time.monotonic() - started)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--data-path")
    parser.add_argument("--source")
    parser.add_argument("--base-url-env")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Biomni request timeout in seconds",
    )
    parser.add_argument(
        "--skip-data-lake-download",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="request streaming responses from the OpenAI-compatible endpoint (default: enabled)",
    )
    parser.add_argument("--commercial-mode", action="store_true")
    # parallel_eval adds these options for structured extraction and grading.
    parser.add_argument("-s", "--sandbox")
    parser.add_argument("--output-schema")
    parser.add_argument("prompt", nargs="?")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _setup_logging()
    args = build_parser().parse_args(argv)
    logger.info("invoked; argv=%r", list(argv) if argv is not None else sys.argv[1:])
    if args.version:
        try:
            version = _biomni_version()
            print(f"biomni-cli {version}")
            logger.info("version=%s", version)
        except RuntimeError as exc:
            logger.error("version lookup failed: %s", exc)
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    if not args.prompt:
        logger.error("no prompt provided")
        print("Biomni runner requires a prompt", file=sys.stderr)
        return 2

    overall_start = time.monotonic()
    try:
        config = BiomniConfig.from_args(args)
        logger.info(
            "config resolved: data_path=%s source=%s base_url_env=%s timeout=%ss skip_data_lake=%s stream=%s",
            config.data_path,
            config.source,
            config.base_url_env,
            config.timeout_seconds,
            config.skip_data_lake_download,
            config.stream,
        )
        base_url = config.base_url()
        api_key = config.api_key()
        logger.info("credentials resolved from env vars %s / %s", config.base_url_env, config.api_key_env)

        t0 = time.monotonic()
        from biomni.agent import A1

        _patch_empty_observations()
        logger.info("biomni version=%s imported in %.2fs", _biomni_version(), time.monotonic() - t0)

        kwargs: dict[str, Any] = {
            "path": config.data_path,
            "llm": args.model,
            "source": config.source,
            "base_url": base_url,
            "api_key": api_key,
            "timeout_seconds": config.timeout_seconds,
        }
        if config.skip_data_lake_download:
            kwargs["expected_data_lake_files"] = []
        if args.commercial_mode:
            kwargs["commercial_mode"] = True
        schema = _load_schema(args.output_schema)
        logger.info("output schema loaded: %s", "yes" if schema else "none")

        t1 = time.monotonic()
        agent = A1(**{key: value for key, value in kwargs.items() if value is not None})
        logger.info("A1 agent constructed in %.2fs", time.monotonic() - t1)
        logger.info("LLM response mode: %s", "streaming" if config.stream else "non-streaming")
        llm_wrapper = _SingleActionLLM(agent.llm, stream=config.stream)
        agent.llm = llm_wrapper
        _, final_output = _run_agent(
            agent,
            args.prompt + _schema_instruction(schema),
        )
        t2 = time.monotonic()
        parsed = _parse_json_object(final_output)
        logger.info(
            "output parsed in %.2fs; json_object=%s",
            time.monotonic() - t2,
            parsed is not None,
        )
        if parsed is None:
            print(str(final_output))
            logger.error("agent returned no JSON object; total elapsed %.2fs", time.monotonic() - overall_start)
            return 1
        else:
            provider_usage = llm_wrapper.token_usage
            parsed_usage = _token_usage(parsed.get("token_usage"))
            if _usage_total(provider_usage):
                parsed["token_usage"] = _token_usage_with_total(provider_usage)
                logger.info("using provider token usage: %s", parsed["token_usage"])
            else:
                # Some providers omit usage metadata. Keep any non-zero usage
                # supplied by the agent instead of replacing it with zeros.
                parsed["token_usage"] = _token_usage_with_total(parsed_usage)
                logger.info("provider returned no token usage; retaining parsed usage: %s", parsed["token_usage"])
            _emit_result_events(parsed)
            logger.info("result emitted; total elapsed %.2fs", time.monotonic() - overall_start)
        return 0
    except Exception as exc:
        logger.exception("runner failed after %.2fs: %s: %s", time.monotonic() - overall_start, type(exc).__name__, exc)
        print(f"Biomni runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
