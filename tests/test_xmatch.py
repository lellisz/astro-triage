"""Offline tests for astro_triage.ingest.xmatch (stubbed ``requests``, no network)."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Self

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pytest
import requests

from astro_triage.ingest import xmatch
from astro_triage.ingest.xmatch import (
    IngestError,
    QueryError,
    RetryExhaustedError,
    TruncatedResultError,
    read_xmatch_csv,
    xmatch_to_file,
)

BIG_ID = 506711614938966016  # 19 digits, not exact as float64
GAIA_ID = 4295806720138422400  # 19 digits, fits int64
HEADER = "angDist,Source,RA_ICRS,Plx,sid,cls\n"
GOOD_BODY = (
    HEADER
    + f"0.5,{GAIA_ID},10.1,1.25,S{BIG_ID},STAR\n"
    + f"1.5,{GAIA_ID + 1},10.2,,S{BIG_ID + 1},QSO\n"
    + f"2.5,,10.3,0.5,S{BIG_ID + 2},GALAXY\n"
)
UPLOAD_MARKER = "UPLOAD_ROW_MARKER_8f3a"
REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- stubs
class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        body: str | bytes = GOOD_BODY,
        *,
        fail_midstream: bool = False,
    ) -> None:
        self.status_code = status
        self._body = body.encode() if isinstance(body, str) else body
        self.fail_midstream = fail_midstream

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", "replace")

    def iter_content(self, chunk_size: int = 1) -> Any:
        yield self._body[: len(self._body) // 2]
        if self.fail_midstream:
            raise requests.exceptions.ChunkedEncodingError("connection broken")
        yield self._body[len(self._body) // 2 :]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakePost:
    """Replace ``requests.post``: pop one outcome per call (response or exception)."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        upload = kwargs["files"]["cat1"][1]
        self.calls.append({"url": url, "upload_bytes": upload.read(), **kwargs})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ChunkedResponse(FakeResponse):
    """Honour ``chunk_size`` like the real ``requests`` stream (fixed-size blocks)."""

    def iter_content(self, chunk_size: int = 1) -> Any:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(xmatch.time, "sleep", recorded.append)
    return recorded


def install(monkeypatch: pytest.MonkeyPatch, *outcomes: Any) -> FakePost:
    post = FakePost(list(outcomes))
    monkeypatch.setattr(xmatch.requests, "post", post)
    return post


@pytest.fixture
def upload(tmp_path: Path) -> Path:
    path = tmp_path / "upload.csv"
    path.write_text(f"sid,ra,dec,cls\nS1,1.0,2.0,{UPLOAD_MARKER}\n", encoding="utf-8")
    return path


@pytest.fixture
def log() -> logging.Logger:
    return logging.getLogger("test_xmatch")


def run(
    upload: Path,
    out: Path,
    log: logging.Logger,
    **overrides: Any,
) -> xmatch.XMatchStats:
    args: dict[str, Any] = {
        "catalog": "vizier:I/355/gaiadr3",
        "radius_arcsec": 5.0,
        "selection": "all",
        "ra_col": "ra",
        "dec_col": "dec",
        "base_delay_s": 1.0,
        "max_retries": 3,
        "logger": log,
    }
    args.update(overrides)
    return xmatch_to_file(upload, out, **args)


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


# ------------------------------------------------------------------- request shape
def test_request_parameters_url_and_multipart(
    monkeypatch: pytest.MonkeyPatch, upload: Path, tmp_path: Path, log: logging.Logger
) -> None:
    post = install(monkeypatch, FakeResponse())
    stats = run(upload, tmp_path / "out.csv", log, radius_arcsec=2.5, timeout_s=77)
    (call,) = post.calls
    assert call["url"].startswith("https://")
    assert call["url"] == xmatch.XMATCH_URL
    assert call["data"] == {
        "request": "xmatch",
        "distMaxArcsec": "2.5",
        "selection": "all",
        "RESPONSEFORMAT": "csv",
        "cat2": "vizier:I/355/gaiadr3",
        "colRA1": "ra",
        "colDec1": "dec",
    }
    name, mime = call["files"]["cat1"][0], call["files"]["cat1"][2]
    assert name == "upload.csv" and mime == "text/csv"
    assert UPLOAD_MARKER.encode() in call["upload_bytes"]
    assert call["timeout"] == (30.0, 77)
    assert call["stream"] is True
    prepared = requests.Request(
        "POST",
        call["url"],
        data=call["data"],
        files={"cat1": ("upload.csv", call["upload_bytes"], "text/csv")},
    ).prepare()
    assert prepared.headers["Content-Type"].startswith("multipart/form-data")
    assert stats.attempts == 1 and stats.rows == 3
    assert (tmp_path / "out.csv").read_text(encoding="utf-8") == GOOD_BODY


def test_radius_is_formatted_without_trailing_zeros(
    monkeypatch: pytest.MonkeyPatch, upload: Path, tmp_path: Path, log: logging.Logger
) -> None:
    post = install(monkeypatch, FakeResponse())
    run(upload, tmp_path / "o.csv", log, radius_arcsec=5.0, selection="best")
    assert post.calls[0]["data"]["distMaxArcsec"] == "5"
    assert post.calls[0]["data"]["selection"] == "best"


@pytest.mark.parametrize(
    "overrides",
    [
        {"catalog": "  "},
        {"radius_arcsec": 0},
        {"radius_arcsec": -1},
        {"radius_arcsec": 180.5},
        {"selection": "nearest"},
        {"max_retries": -1},
        {"base_delay_s": -0.1},
    ],
)
def test_invalid_arguments_raise_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
    overrides: dict[str, Any],
) -> None:
    post = install(monkeypatch, FakeResponse())
    with pytest.raises(ValueError):
        run(upload, tmp_path / "o.csv", log, **overrides)
    assert post.calls == []


def test_missing_upload_is_query_error_without_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, log: logging.Logger
) -> None:
    post = install(monkeypatch, FakeResponse())
    with pytest.raises(QueryError) as info:
        run(tmp_path / "nope.csv", tmp_path / "o.csv", log)
    assert "nope.csv" in str(info.value) and str(tmp_path) not in str(info.value)
    assert post.calls == []


# ---------------------------------------------------------------- retry / failure
@pytest.mark.parametrize(
    "failure",
    [
        requests.Timeout("read timed out"),
        requests.ConnectionError("reset by peer"),
        FakeResponse(429, "slow down"),
        FakeResponse(500, "oops"),
        FakeResponse(502, "bad gateway"),
        FakeResponse(503, "unavailable"),
        FakeResponse(504, "gateway timeout"),
    ],
    ids=["timeout", "connerr", "429", "500", "502", "503", "504"],
)
def test_transient_failure_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
    failure: Any,
) -> None:
    post = install(monkeypatch, failure, failure, FakeResponse())
    stats = run(upload, tmp_path / "o.csv", log, base_delay_s=4.0)
    assert len(post.calls) == 3 and stats.attempts == 3
    assert len(sleeps) == 2
    # Exponential backoff with jitter in [0.5, 1.0].
    assert 2.0 <= sleeps[0] <= 4.0
    assert 4.0 <= sleeps[1] <= 8.0
    assert (tmp_path / "o.csv").read_text(encoding="utf-8") == GOOD_BODY
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "failure",
    [requests.Timeout("t"), requests.ConnectionError("c"), FakeResponse(503, "x")],
    ids=["timeout", "connerr", "503"],
)
def test_retries_exhausted_raises_ingest_error_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
    failure: Any,
) -> None:
    post = install(monkeypatch, failure)
    with pytest.raises(RetryExhaustedError) as info:
        run(upload, tmp_path / "o.csv", log, max_retries=2)
    assert isinstance(info.value, IngestError)
    assert len(post.calls) == 3
    assert len(sleeps) == 2  # no sleep after the last attempt
    assert not (tmp_path / "o.csv").exists()
    assert leftovers(tmp_path) == []


def test_max_retries_zero_means_single_attempt(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    post = install(monkeypatch, FakeResponse(503, "x"))
    with pytest.raises(RetryExhaustedError):
        run(upload, tmp_path / "o.csv", log, max_retries=0)
    assert len(post.calls) == 1 and sleeps == []


def test_backoff_delay_is_capped(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    install(monkeypatch, FakeResponse(503, "x"))
    with pytest.raises(RetryExhaustedError):
        run(upload, tmp_path / "o.csv", log, max_retries=3, base_delay_s=1000.0)
    assert sleeps and max(sleeps) <= 120.0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_4xx_fails_immediately_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
    status: int,
) -> None:
    post = install(monkeypatch, FakeResponse(status, "unknown catalogue"))
    with pytest.raises(QueryError) as info:
        run(upload, tmp_path / "o.csv", log)
    assert isinstance(info.value, IngestError)
    assert not isinstance(info.value, RetryExhaustedError)
    assert str(status) in str(info.value)
    assert len(post.calls) == 1 and sleeps == []
    assert leftovers(tmp_path) == []
    assert not (tmp_path / "o.csv").exists()


def test_http_200_with_non_csv_body_is_query_error(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    post = install(monkeypatch, FakeResponse(200, "<html>maintenance</html>"))
    with pytest.raises(QueryError, match="not the expected CSV"):
        run(upload, tmp_path / "o.csv", log)
    assert len(post.calls) == 1 and sleeps == []
    assert leftovers(tmp_path) == [] and not (tmp_path / "o.csv").exists()


def test_midstream_failure_is_retried_and_leaves_no_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    install(monkeypatch, FakeResponse(fail_midstream=True))
    with pytest.raises(RetryExhaustedError):
        run(upload, tmp_path / "o.csv", log, max_retries=1)
    assert leftovers(tmp_path) == [] and not (tmp_path / "o.csv").exists()


# --------------------------------------------------------------------- truncation
def test_output_at_row_limit_raises_truncated_error(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    monkeypatch.setattr(xmatch, "XMATCH_MAX_ROWS", 3)
    out = tmp_path / "o.csv"
    out.write_text("previous good answer\n", encoding="utf-8")
    post = install(monkeypatch, FakeResponse(200, GOOD_BODY))  # exactly 3 data rows
    with pytest.raises(TruncatedResultError) as info:
        run(upload, out, log)
    assert isinstance(info.value, IngestError)
    assert "truncation" in str(info.value)
    assert len(post.calls) == 1 and sleeps == []  # not retried
    assert out.read_text(encoding="utf-8") == "previous good answer\n"
    assert leftovers(tmp_path) == []


def test_output_below_row_limit_is_accepted(
    monkeypatch: pytest.MonkeyPatch, upload: Path, tmp_path: Path, log: logging.Logger
) -> None:
    monkeypatch.setattr(xmatch, "XMATCH_MAX_ROWS", 4)
    install(monkeypatch, FakeResponse(200, GOOD_BODY))
    assert run(upload, tmp_path / "o.csv", log).rows == 3


def _data_rows_body(n_rows: int, header: bytes, final_newline: bool) -> bytes:
    body = header + b"".join(b"1234567\n" for _ in range(n_rows))  # 8-byte rows
    return body if final_newline else body[:-1]


def _count(tmp_path: Path, body: bytes) -> tuple[int, int]:
    return xmatch._stream_to(ChunkedResponse(200, body), str(tmp_path / "s.tmp"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("body", "rows"),
    [
        (b"", 0),
        (b"angDist", 0),
        (b"angDist\n", 0),
        (b"angDist\na", 1),
        (b"angDist\na\n", 1),
        (b"angDist\na\nb", 2),
        (b"angDist\na\nb\n", 2),
    ],
    ids=["empty", "header-nonl", "header-nl", "1-nonl", "1-nl", "2-nonl", "2-nl"],
)
def test_stream_counts_data_rows_small_bodies(
    tmp_path: Path, body: bytes, rows: int
) -> None:
    assert _count(tmp_path, body) == (len(body), rows)


@pytest.mark.parametrize("final_newline", [True, False], ids=["nl", "nonl"])
@pytest.mark.parametrize(
    ("header", "boundary_index"),
    [
        # 8-byte header + 131071 rows of 8 bytes: a newline is the LAST byte of block 1.
        (b"angDist\n", xmatch._CHUNK_BYTES - 1),
        # 9-byte header: the newline is the FIRST byte of block 2.
        (b"angDist_\n", xmatch._CHUNK_BYTES),
    ],
    ids=["newline-ends-block", "newline-starts-block"],
)
def test_stream_counts_rows_across_block_boundary(
    tmp_path: Path, header: bytes, boundary_index: int, final_newline: bool
) -> None:
    n_rows = 131_071 + 5
    body = _data_rows_body(n_rows, header, final_newline)
    assert len(body) > xmatch._CHUNK_BYTES
    assert body[boundary_index] == ord("\n")  # the layout really hits the boundary
    assert _count(tmp_path, body) == (len(body), n_rows)


def test_large_body_row_count_end_to_end(
    monkeypatch: pytest.MonkeyPatch, upload: Path, tmp_path: Path, log: logging.Logger
) -> None:
    body = _data_rows_body(131_076, b"angDist\n", final_newline=False)
    install(monkeypatch, ChunkedResponse(200, body))
    stats = run(upload, tmp_path / "o.csv", log)
    assert stats.rows == 131_076 and stats.bytes_received == len(body)
    assert (tmp_path / "o.csv").read_bytes() == body


def test_truncation_at_exact_limit_without_final_newline(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    monkeypatch.setattr(xmatch, "XMATCH_MAX_ROWS", 3)
    out = tmp_path / "o.csv"
    out.write_text("previous good answer\n", encoding="utf-8")
    install(
        monkeypatch, FakeResponse(200, "angDist\na\nb\nc")
    )  # 3 rows, no final newline
    with pytest.raises(TruncatedResultError):
        run(upload, out, log)
    assert out.read_text(encoding="utf-8") == "previous good answer\n"
    assert leftovers(tmp_path) == []
    install(monkeypatch, FakeResponse(200, "angDist\na\nb"))  # 2 rows: accepted
    assert run(upload, out, log).rows == 2
    assert out.read_text(encoding="utf-8") == "angDist\na\nb"


# ------------------------------------------------------------------- path masking
def test_mask_paths_windows_and_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_home = tmp_path / "fakehome"
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    assert xmatch._mask_paths(r"C:\Users\x\y.csv") == "<path>"
    assert xmatch._mask_paths("C:/a/b") == "<path>"
    assert xmatch._mask_paths(str(fake_home)) == "<path>"
    masked = xmatch._mask_paths(f"open {fake_home}/x.csv failed")
    assert str(fake_home) not in masked and "<path>" in masked
    assert (
        xmatch._mask_paths("read timed out (HTTP 503)") == "read timed out (HTTP 503)"
    )


def test_safe_detail_redacts_secrets_and_masks_paths() -> None:
    out = xmatch._safe_detail("token=abc123 failed at C:\\Users\\x\\y.csv")
    assert "abc123" not in out and "Users" not in out
    assert out == "token=*** failed at <path>"
    assert xmatch._safe_detail("plain  text\nhere") == "plain text here"


def test_timeout_message_path_is_not_logged_nor_in_exhausted_error(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    caplog.set_level(logging.DEBUG)
    secret_path = r"C:\Users\someone\secret_dir\ca.pem"
    install(monkeypatch, requests.Timeout(f"timed out reading {secret_path}"))
    with pytest.raises(RetryExhaustedError) as info:
        run(upload, tmp_path / "o.csv", log, max_retries=1)
    text = "\n".join(r.getMessage() for r in caplog.records) + str(info.value)
    assert "secret_dir" not in text and "someone" not in text
    assert "<path>" in str(info.value)
    assert sum("transient, retry" in r.getMessage() for r in caplog.records) == 1
    assert sum("giving up" in r.getMessage() for r in caplog.records) == 1


def test_row_limit_constant_is_two_million() -> None:
    assert xmatch.XMATCH_MAX_ROWS == 2_000_000


# ------------------------------------------------------------------ reading ids
def test_read_csv_keeps_19_digit_ids_exact_and_missing_as_null(tmp_path: Path) -> None:
    path = tmp_path / "answer.csv"
    path.write_text(GOOD_BODY, encoding="utf-8")
    table = read_xmatch_csv(path)
    assert table.schema.field("Source").type == pa.int64()
    assert table.schema.field("sid").type == pa.string()
    assert table.schema.field("cls").type == pa.string()
    assert table["Source"].to_pylist() == [GAIA_ID, GAIA_ID + 1, None]
    assert table["sid"].to_pylist() == [f"S{BIG_ID + i}" for i in range(3)]
    assert table["Plx"].null_count == 1
    assert table["Source"].null_count == 1
    assert float(GAIA_ID + 1) != GAIA_ID + 1  # a float path would have broken this


def test_all_digit_sid_is_still_read_as_text(tmp_path: Path) -> None:
    path = tmp_path / "answer.csv"
    path.write_text(f"angDist,Source,sid,cls\n1.0,{GAIA_ID},{BIG_ID + 1},STAR\n")
    table = read_xmatch_csv(path)
    assert table["sid"].to_pylist() == [str(BIG_ID + 1)]


def test_xmatch_returns_arrow_backed_frame_with_na(
    monkeypatch: pytest.MonkeyPatch, upload: Path, log: logging.Logger
) -> None:
    install(monkeypatch, FakeResponse())
    frame = xmatch.xmatch(
        upload,
        catalog="vizier:I/355/gaiadr3",
        radius_arcsec=5,
        selection="all",
        ra_col="ra",
        dec_col="dec",
        logger=log,
    )
    assert frame["Source"].dtype == pd.ArrowDtype(pa.int64())
    assert frame["Source"].iloc[0] == GAIA_ID
    assert frame["Source"].iloc[1] == GAIA_ID + 1
    assert frame["Source"].isna().tolist() == [False, False, True]
    assert frame["Plx"].isna().tolist() == [False, True, False]
    assert frame["sid"].iloc[0] == f"S{BIG_ID}"
    assert frame.isna().sum().sum() == 2


# -------------------------------------------------------------- atomic write
def test_failed_replace_keeps_old_file_and_removes_tmp(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    out = tmp_path / "o.csv"
    out.write_text("old content\n", encoding="utf-8")
    install(monkeypatch, FakeResponse())

    def boom(src: object, dst: object) -> None:
        raise OSError("disk exploded")

    monkeypatch.setattr(xmatch.os, "replace", boom)
    with pytest.raises(OSError, match="disk exploded"):
        run(upload, out, log)
    assert out.read_text(encoding="utf-8") == "old content\n"
    assert leftovers(tmp_path) == []


def test_midstream_failure_does_not_corrupt_previous_output(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    out = tmp_path / "o.csv"
    out.write_text("old content\n", encoding="utf-8")
    install(monkeypatch, FakeResponse(fail_midstream=True))
    with pytest.raises(RetryExhaustedError):
        run(upload, out, log, max_retries=0)
    assert out.read_text(encoding="utf-8") == "old content\n"


def test_success_replaces_previous_output_atomically(
    monkeypatch: pytest.MonkeyPatch, upload: Path, tmp_path: Path, log: logging.Logger
) -> None:
    out = tmp_path / "sub" / "o.csv"
    out.parent.mkdir()
    out.write_text("old\n", encoding="utf-8")
    install(monkeypatch, FakeResponse())
    run(upload, out, log)
    assert out.read_text(encoding="utf-8") == GOOD_BODY
    assert leftovers(out.parent) == []


# ------------------------------------------------------------------------ logging
def test_logs_never_contain_upload_body_or_absolute_paths(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
    upload: Path,
    tmp_path: Path,
    log: logging.Logger,
) -> None:
    caplog.set_level(logging.DEBUG)
    # Realistic message: requests/urllib3 errors often embed a local absolute path.
    post = install(
        monkeypatch,
        requests.Timeout(f"read timed out while using {tmp_path / 'cert.pem'}"),
        FakeResponse(503, "x"),
        FakeResponse(),
    )
    run(upload, tmp_path / "ok.csv", log)
    post.outcomes = [FakeResponse(400, "bad request")]
    with pytest.raises(QueryError):
        run(upload, tmp_path / "bad.csv", log)
    post.outcomes = [FakeResponse(503, "x")]
    with pytest.raises(RetryExhaustedError):
        run(upload, tmp_path / "gone.csv", log, max_retries=1)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert caplog.records, "expected log records"
    assert UPLOAD_MARKER not in text
    assert str(tmp_path) not in text
    assert str(tmp_path).replace("\\", "/") not in text
    assert "attempt=" in text


def test_default_logger_writes_nothing_when_logger_injected(
    monkeypatch: pytest.MonkeyPatch, upload: Path, tmp_path: Path, log: logging.Logger
) -> None:
    # Guard for the tests themselves: they must never touch the repo ingest log.
    before = (
        (REPO_ROOT / "logs" / "ingest.log").stat().st_size
        if (REPO_ROOT / "logs" / "ingest.log").exists()
        else None
    )
    install(monkeypatch, FakeResponse())
    run(upload, tmp_path / "o.csv", log)
    after = (
        (REPO_ROOT / "logs" / "ingest.log").stat().st_size
        if (REPO_ROOT / "logs" / "ingest.log").exists()
        else None
    )
    assert before == after


# ----------------------------------------------- sid prefix rule (batch script)
def _load_batch() -> Any:
    spec = importlib.util.spec_from_file_location(
        "xmatch_batch_under_test", REPO_ROOT / "scripts" / "xmatch_batch.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("as_int", [False, True], ids=["string-ids", "int64-ids"])
def test_batch_sid_has_non_digit_prefix_and_keeps_all_digits(
    tmp_path: Path, as_int: bool
) -> None:
    batch = _load_batch()
    ids = [BIG_ID, BIG_ID + 1, BIG_ID + 2]
    candidates = pa.table(
        {
            "specobjid": pa.array(ids if as_int else [str(i) for i in ids]),
            "plug_ra": [1.0, 2.0, 3.0],
            "plug_dec": [4.0, 5.0, 6.0],
            "class": ["STAR", "QSO", "GALAXY"],
        }
    )
    path = tmp_path / "cand.parquet"
    pq.write_table(candidates, path)
    sample = batch.sample_positions(path, 3, seed=1)
    assert batch.SID_PREFIX == "S"
    assert list(sample.columns) == ["sid", "ra", "dec", "cls"]
    assert all(s.startswith("S") for s in sample["sid"])
    assert not sample["sid"].str.isdigit().any()
    assert sorted(sample["sid"]) == [f"S{i}" for i in ids]

    upload_path = tmp_path / "up" / "upload.csv"
    batch.write_upload_csv(sample, upload_path)
    assert [p.name for p in upload_path.parent.iterdir()] == ["upload.csv"]
    # Read the way the XMatch service would: every column typed from the text.
    table = pacsv.read_csv(upload_path)
    assert table.schema.field("sid").type == pa.string()
    assert sorted(table["sid"].to_pylist()) == [f"S{i}" for i in ids]
    assert table.schema.field("ra").type == pa.float64()


def test_batch_sample_rejects_n_larger_than_candidates(tmp_path: Path) -> None:
    batch = _load_batch()
    path = tmp_path / "cand.parquet"
    pq.write_table(
        pa.table(
            {
                "specobjid": ["1"],
                "plug_ra": [1.0],
                "plug_dec": [2.0],
                "class": ["STAR"],
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="exceeds"):
        batch.sample_positions(path, 2, seed=1)
