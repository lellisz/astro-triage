"""Regression for security finding S1 (ADR 0005 D8): a 3xx must never replay Authorization.

Two loopback servers on ephemeral ports: A answers with a 3xx pointing at B, B records every
request it receives. Loopback only, fake key, no sleep.
"""

import inspect
import json
import logging
import random
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

import pytest

from astro_triage.llm import client as client_mod
from astro_triage.llm import runner as runner_mod
from astro_triage.llm.client import (
    CallResult,
    FatalCallError,
    chat,
    open_no_redirect,
)

KEY = "FAKE-KEY-123"
LOCATION_MARK = "LOCATION-MARK-9f3a"
MESSAGES = [{"role": "user", "content": "hi"}]
REDIRECT_CODES = [301, 302, 303, 307, 308]
OK_BODY = {
    "model": "nvidia/fake",
    "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


class Recorded:
    def __init__(self, method: str, path: str, auth: str | None) -> None:
        self.method = method
        self.path = path
        self.auth = auth


class FakeServer:
    """One loopback server; ``mode`` is ``"ok"`` or a 3xx code, ``location`` its target."""

    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self.mode: int | str = "ok"
        self.location = ""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                outer.requests.append(
                    Recorded(self.command, self.path, self.headers.get("Authorization"))
                )
                if outer.mode == "ok":
                    payload = json.dumps(OK_BODY).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                else:
                    payload = b"moved"
                    self.send_response(int(outer.mode))
                    self.send_header("Location", outer.location)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)

            do_POST = do_GET = do_PUT = _handle

            def log_message(self, *args: Any) -> None:
                return None

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def servers() -> Iterator[tuple[FakeServer, FakeServer]]:
    a, b = FakeServer(), FakeServer()
    a.location = f"{b.base}/v1/{LOCATION_MARK}"
    try:
        yield a, b
    finally:
        a.close()
        b.close()


def make_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )


def run_chat(**kw: Any) -> CallResult:
    return chat(
        MESSAGES,
        "nvidia/fake",
        key=KEY,
        enable_thinking=False,
        max_tokens=8,
        timeout_s=5.0,
        rng=random.Random(1),
        **kw,
    )


@pytest.mark.parametrize("code", REDIRECT_CODES)
def test_open_no_redirect_raises_httperror_and_b_sees_nothing(
    servers: tuple[FakeServer, FakeServer], code: int
) -> None:
    a, b = servers
    a.mode = code
    with pytest.raises(urllib.error.HTTPError) as info:
        open_no_redirect(make_request(f"{a.base}/v1/chat/completions"), 5.0)
    info.value.close()
    assert info.value.code == code
    assert b.requests == []
    assert len(a.requests) == 1
    assert a.requests[0].method == "POST"
    assert a.requests[0].auth == f"Bearer {KEY}"


@pytest.mark.parametrize("code", REDIRECT_CODES)
def test_chat_default_opener_maps_3xx_to_fatal_bad_request(
    servers: tuple[FakeServer, FakeServer],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    code: int,
) -> None:
    a, b = servers
    a.mode = code
    monkeypatch.setattr(client_mod, "BASE_URL", f"{a.base}/v1")
    seen: list[CallResult] = []
    sleeps: list[float] = []
    caplog.set_level(logging.DEBUG)
    with pytest.raises(FatalCallError) as info:
        run_chat(
            before_attempt=lambda: None,
            after_attempt=seen.append,
            sleep=sleeps.append,
        )
    exc = info.value
    assert exc.status == "BAD_REQUEST"
    assert exc.__cause__ is None
    assert len(seen) == 1
    assert seen[0].status == "BAD_REQUEST"
    assert seen[0].http_status == code
    assert seen[0].attempts == 1
    assert sleeps == []
    assert len(a.requests) == 1
    assert b.requests == []
    visible = " ".join([str(exc), repr(exc), repr(seen[0]), caplog.text])
    assert KEY not in visible
    assert LOCATION_MARK not in visible
    assert b.base not in visible


def test_runner_and_chat_default_to_open_no_redirect() -> None:
    assert inspect.signature(chat).parameters["opener"].default is open_no_redirect
    assert (
        inspect.signature(runner_mod.run).parameters["opener"].default
        is open_no_redirect
    )


def test_open_no_redirect_does_not_alter_200(
    servers: tuple[FakeServer, FakeServer],
) -> None:
    a, b = servers
    with open_no_redirect(make_request(f"{a.base}/v1/chat/completions"), 5.0) as resp:
        assert resp.status == 200
        assert json.loads(resp.read())["model"] == "nvidia/fake"
    assert b.requests == []


def test_chat_default_opener_reads_200_from_server(
    servers: tuple[FakeServer, FakeServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, _ = servers
    monkeypatch.setattr(client_mod, "BASE_URL", f"{a.base}/v1")
    res = run_chat(before_attempt=lambda: None, after_attempt=lambda r: None)
    assert (res.status, res.content, res.attempts) == ("OK", "hello", 1)


def test_injected_opener_still_used() -> None:
    class Resp:
        status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self, n: int = -1) -> bytes:
            return json.dumps(OK_BODY).encode()

    calls: list[Any] = []

    def opener(request: Any, timeout: float) -> Resp:
        calls.append(request)
        return Resp()

    res = run_chat(
        before_attempt=lambda: None, after_attempt=lambda r: None, opener=opener
    )
    assert res.status == "OK"
    assert res.content == "hello"
    assert len(calls) == 1
