"""One NVIDIA chat completion over urllib, mapped to an explicit D4 status (ADR 0005).

Retries only on 429, 5xx, timeout and network errors (at most ``max_retries`` retries).
Deterministic outcomes are never retried. Error bodies, headers and the key never reach an
exception or a return value, and no exception is chained (``from exc``), so raw server text
cannot survive in a traceback. ``status`` "OK" here means only "HTTP 200 and not truncated":
empty or malformed content is classified later by ``prompt.parse_answer``.
"""

from __future__ import annotations

import http.client
import json
import random
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Final, NamedTuple

__all__ = ["BASE_URL", "CallResult", "FatalCallError", "chat", "open_no_redirect"]

BASE_URL: Final = "https://integrate.api.nvidia.com/v1"
_MAX_BACKOFF_S: Final = 30.0
_MAX_RESPONSE_BYTES: Final = 4 * 1024 * 1024
_FATAL: Final = frozenset({"MODEL_UNAVAILABLE", "AUTH_ERROR", "BAD_REQUEST"})
_RETRYABLE: Final = frozenset(
    {"RATE_LIMITED", "SERVER_ERROR", "TIMEOUT", "NETWORK_ERROR"}
)


class CallResult(NamedTuple):
    status: str
    http_status: int | None
    content: str | None
    finish_reason: str | None
    model_reported: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int
    reasoning_chars: int
    attempts: int


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a 3xx: it would replay the Authorization header elsewhere."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_NO_REDIRECT_OPENER: Final = urllib.request.build_opener(_NoRedirect)


def open_no_redirect(request: urllib.request.Request, timeout: float) -> Any:
    """Default opener: a 3xx surfaces as HTTPError (mapped to BAD_REQUEST, fatal)."""
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


class FatalCallError(RuntimeError):
    """The run must stop: ``status`` is MODEL_UNAVAILABLE, AUTH_ERROR or BAD_REQUEST."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


def _error_result(status: str, http_status: int | None, latency_ms: int) -> CallResult:
    return CallResult(
        status, http_status, None, None, None, None, None, latency_ms, 0, 1
    )


def _status_for_http(code: int) -> str:
    if code == 429:
        return "RATE_LIMITED"
    if code >= 500:
        return "SERVER_ERROR"
    if code == 404:
        return "MODEL_UNAVAILABLE"
    if code in (401, 403):
        return "AUTH_ERROR"
    return "BAD_REQUEST"


def _opt_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_success(raw: bytes, latency_ms: int) -> CallResult:
    try:
        body = json.loads(raw.decode("utf-8"))
        choice = body["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError, TypeError, RecursionError):
        return _error_result("INVALID_JSON", 200, latency_ms)
    if not isinstance(message, dict):
        return _error_result("INVALID_JSON", 200, latency_ms)
    content = message.get("content")
    finish = choice.get("finish_reason")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    model = body.get("model")
    return CallResult(
        status="TRUNCATED" if finish == "length" else "OK",
        http_status=200,
        content=content if isinstance(content, str) else None,
        finish_reason=finish if isinstance(finish, str) else None,
        model_reported=model if isinstance(model, str) else None,
        prompt_tokens=_opt_int(usage.get("prompt_tokens")),
        completion_tokens=_opt_int(usage.get("completion_tokens")),
        latency_ms=latency_ms,
        reasoning_chars=len(reasoning) if isinstance(reasoning, str) else 0,
        attempts=1,
    )


def _retry_after(headers: Any) -> float:
    try:
        return max(0.0, float(headers.get("Retry-After", 0)))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _one_attempt(
    request: urllib.request.Request,
    timeout_s: float,
    opener: Callable[..., Any],
) -> tuple[CallResult, float]:
    """Run one HTTP request; never raises for transport or HTTP errors."""
    start = time.monotonic()
    retry_after = 0.0
    status = "NETWORK_ERROR"
    http_status: int | None = None
    raw = b""
    try:
        with opener(request, timeout=timeout_s) as resp:
            http_status = int(resp.status)
            raw = resp.read(_MAX_RESPONSE_BYTES)
        status = "OK" if http_status == 200 else "BAD_REQUEST"
    except urllib.error.HTTPError as err:
        http_status = err.code
        status = _status_for_http(err.code)
        retry_after = _retry_after(err.headers)
        err.close()
    except urllib.error.URLError as err:
        timed_out = isinstance(err.reason, (TimeoutError, socket.timeout))
        status = "TIMEOUT" if timed_out else "NETWORK_ERROR"
    except TimeoutError:
        status = "TIMEOUT"
    except (OSError, http.client.HTTPException):
        status = "NETWORK_ERROR"
    latency_ms = int((time.monotonic() - start) * 1000)
    if status == "OK":
        return _parse_success(raw, latency_ms), 0.0
    return _error_result(status, http_status, latency_ms), retry_after


def _sum_optional(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else a + b


def chat(
    messages: list[dict[str, str]],
    model: str,
    *,
    key: str,
    enable_thinking: bool,
    max_tokens: int,
    timeout_s: float = 60.0,
    max_retries: int = 3,
    before_attempt: Callable[[], None],
    after_attempt: Callable[[CallResult], None],
    opener: Callable[..., Any] = open_no_redirect,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> CallResult:
    """Return the final result; latency and tokens are summed over the attempts made.

    ``before_attempt`` runs before every attempt (the quota guard raises there) and
    ``after_attempt`` receives each attempt's own result. Raises ``FatalCallError`` after
    ``after_attempt`` for 404, 401/403 and other 4xx.
    """
    jitter = rng if rng is not None else random.Random()
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    total_latency = 0
    prompt_total: int | None = None
    completion_total: int | None = None
    for attempt in range(1, max_retries + 2):
        before_attempt()
        result, retry_after = _one_attempt(request, timeout_s, opener)
        result = result._replace(attempts=attempt)
        after_attempt(result)
        total_latency += result.latency_ms
        prompt_total = _sum_optional(prompt_total, result.prompt_tokens)
        completion_total = _sum_optional(completion_total, result.completion_tokens)
        if result.status in _FATAL:
            raise FatalCallError(result.status)
        if result.status not in _RETRYABLE or attempt > max_retries:
            break
        backoff = min(_MAX_BACKOFF_S, 2.0 * 2 ** (attempt - 1))
        backoff *= jitter.uniform(0.8, 1.2)
        sleep(min(_MAX_BACKOFF_S, max(backoff, retry_after)))
    return result._replace(
        latency_ms=total_latency,
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
    )
