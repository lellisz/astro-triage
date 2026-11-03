"""Proofs for ``llm.client`` (ADR 0005 D4 status table): fake opener, no network, no real sleep."""

import io
import json
import logging
import random
import urllib.error
from email.message import Message
from typing import Any, Self

import pytest

from astro_triage.llm.client import CallResult, FatalCallError, chat

KEY = "FAKE-KEY-123"
MESSAGES = [{"role": "user", "content": "hi"}]


class Resp:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]


def ok_body(
    finish: str = "stop",
    content: str | None = '{"label":"STAR","confidence":9}',
    **extra: Any,
) -> Resp:
    message: dict[str, Any] = {"content": content, **extra}
    body = {
        "model": "nvidia/fake",
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7},
    }
    return Resp(json.dumps(body).encode())


def http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    body = io.BytesIO(f'{{"error": "bad key {KEY}"}}'.encode())
    return urllib.error.HTTPError("https://x/y", code, "msg " + KEY, headers, body)


class Opener:
    """Plays a script; the last item repeats forever."""

    def __init__(self, *script: object, log: list[str] | None = None) -> None:
        self.script = list(script)
        self.calls = 0
        self.requests: list[Any] = []
        self.log = log

    def __call__(self, request: Any, timeout: float) -> Any:
        self.requests.append(request)
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if self.log is not None:
            self.log.append("open")
        if isinstance(item, BaseException):
            raise item
        return item


def run(
    opener: Opener,
    *,
    before: Any = None,
    after: Any = None,
    sleeps: list[float] | None = None,
    **kw: Any,
) -> CallResult:
    sleeps = [] if sleeps is None else sleeps
    return chat(
        MESSAGES,
        "nvidia/fake",
        key=KEY,
        enable_thinking=False,
        max_tokens=64,
        before_attempt=before or (lambda: None),
        after_attempt=after or (lambda r: None),
        opener=opener,
        sleep=sleeps.append,
        rng=random.Random(1234),
        **kw,
    )


def test_200_stop_is_ok_with_content_and_usage() -> None:
    res = run(Opener(ok_body(reasoning_content="think")))
    assert res.status == "OK"
    assert res.http_status == 200
    assert res.content == '{"label":"STAR","confidence":9}'
    assert res.finish_reason == "stop"
    assert res.model_reported == "nvidia/fake"
    assert (res.prompt_tokens, res.completion_tokens) == (100, 7)
    assert res.reasoning_chars == 5
    assert res.attempts == 1


def test_200_length_is_truncated_and_not_retried() -> None:
    op = Opener(ok_body(finish="length"))
    res = run(op)
    assert res.status == "TRUNCATED"
    assert op.calls == 1


@pytest.mark.parametrize(
    "body",
    [b"not json", b"{}", b'{"choices": []}', b'{"choices": [{"message": 3}]}', b"\xff"],
)
def test_malformed_200_is_invalid_json_not_retried(body: bytes) -> None:
    op = Opener(Resp(body))
    res = run(op)
    assert res.status == "INVALID_JSON"
    assert res.http_status == 200
    assert op.calls == 1


def test_200_with_null_content_is_returned_for_the_parser() -> None:
    res = run(Opener(ok_body(content=None)))
    assert (res.status, res.content) == ("OK", None)


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (400, "BAD_REQUEST"),
        (422, "BAD_REQUEST"),
        (401, "AUTH_ERROR"),
        (403, "AUTH_ERROR"),
        (404, "MODEL_UNAVAILABLE"),
    ],
)
def test_deterministic_http_errors_are_fatal_and_not_retried(
    code: int, status: str
) -> None:
    op = Opener(http_error(code))
    seen: list[CallResult] = []
    with pytest.raises(FatalCallError) as info:
        run(op, after=seen.append)
    assert info.value.status == status
    assert op.calls == 1
    assert [r.status for r in seen] == [status]  # after_attempt ran before the raise
    assert seen[0].http_status == code


@pytest.mark.parametrize(
    ("script", "status", "http"),
    [
        (http_error(429), "RATE_LIMITED", 429),
        (http_error(500), "SERVER_ERROR", 500),
        (http_error(503), "SERVER_ERROR", 503),
        (TimeoutError(), "TIMEOUT", None),
        (urllib.error.URLError(TimeoutError()), "TIMEOUT", None),
        (urllib.error.URLError("dns down"), "NETWORK_ERROR", None),
        (ConnectionResetError(), "NETWORK_ERROR", None),
    ],
)
def test_retryable_failures_exhaust_at_four_attempts(
    script: BaseException, status: str, http: int | None
) -> None:
    op = Opener(script)
    sleeps: list[float] = []
    after: list[CallResult] = []
    res = run(op, sleeps=sleeps, after=after.append)
    assert (res.status, res.http_status) == (status, http)
    assert res.attempts == 4
    assert op.calls == 4
    assert len(after) == 4
    assert len(sleeps) == 3  # no sleep after the last attempt
    assert all(0 < s <= 30 for s in sleeps)


def test_backoff_grows_with_jitter_bounds_and_is_capped_at_30() -> None:
    sleeps: list[float] = []
    run(Opener(http_error(503)), sleeps=sleeps, max_retries=7)
    assert len(sleeps) == 7
    for i, s in enumerate(sleeps):
        base = min(30.0, 2.0 * 2**i)
        assert base * 0.8 - 1e-9 <= s <= min(30.0, base * 1.2) + 1e-9
    assert max(sleeps) <= 30.0


def test_huge_retry_after_is_capped_and_small_one_is_a_floor() -> None:
    sleeps: list[float] = []
    run(Opener(http_error(429, "99999")), sleeps=sleeps)
    assert sleeps == [30.0, 30.0, 30.0]
    sleeps = []
    run(Opener(http_error(429, "10")), sleeps=sleeps)
    assert all(s >= 10 for s in sleeps)
    sleeps = []
    run(Opener(http_error(429, "garbage")), sleeps=sleeps)
    assert len(sleeps) == 3 and all(0 < s <= 30 for s in sleeps)


def test_retry_then_success_sums_usage_and_counts_attempts() -> None:
    op = Opener(http_error(429), TimeoutError(), ok_body())
    res = run(op)
    assert res.status == "OK"
    assert res.attempts == 3
    assert (res.prompt_tokens, res.completion_tokens) == (100, 7)


def test_request_shape() -> None:
    op = Opener(ok_body())
    run(op)
    req = op.requests[0]
    assert req.full_url.endswith("/chat/completions")
    assert req.get_method() == "POST"
    payload = json.loads(req.data)
    assert payload["model"] == "nvidia/fake"
    assert payload["temperature"] == 0
    assert payload["stream"] is False
    assert payload["max_tokens"] == 64
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_before_attempt_runs_before_every_attempt() -> None:
    log: list[str] = []
    op = Opener(http_error(503), http_error(503), ok_body(), log=log)
    run(op, before=lambda: log.append("before"))
    assert log == ["before", "open"] * 3


def test_exception_in_before_attempt_prevents_the_call() -> None:
    op = Opener(ok_body())

    def boom() -> None:
        raise RuntimeError("quota")

    with pytest.raises(RuntimeError, match="quota"):
        run(op, before=boom)
    assert op.calls == 0


def test_exception_in_before_attempt_on_a_retry_stops_further_calls() -> None:
    op = Opener(http_error(503))
    n = {"c": 0}

    def before() -> None:
        n["c"] += 1
        if n["c"] == 2:
            raise RuntimeError("quota")

    with pytest.raises(RuntimeError):
        run(op, before=before)
    assert op.calls == 1


def _no_key(*texts: object) -> None:
    for t in texts:
        assert KEY not in str(t)
        assert KEY not in repr(t)


@pytest.mark.parametrize(
    "script",
    [
        http_error(429),
        http_error(503),
        urllib.error.URLError(f"failed Bearer {KEY}"),
        ConnectionError(f"reset {KEY}"),
        Resp(f"not json {KEY}".encode()),
    ],
)
def test_key_never_reaches_result_or_logs(
    script: object, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        res = run(Opener(script))
    _no_key(res)
    assert KEY not in caplog.text


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_key_never_reaches_fatal_exception_or_its_chain(
    code: int, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG), pytest.raises(FatalCallError) as info:
        run(Opener(http_error(code)))
    exc = info.value
    _no_key(exc, exc.args)
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert KEY not in caplog.text
