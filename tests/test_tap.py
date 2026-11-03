"""Offline tests for astro_triage.ingest.tap (fake client, fake sleep, no network)."""

from __future__ import annotations

import inspect
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.table import MaskedColumn, Table

from astro_triage.ingest import tap
from astro_triage.ingest.tap import (
    CredentialError,
    QueryError,
    QuotaExceededError,
    RetryExhaustedError,
    login_interactive,
    query_cache_key,
    run_adql,
)

QUERY = "SELECT source_id, ra FROM gaiadr3.gaia_source WHERE ra < 1"


# --------------------------------------------------------------------------- helpers
class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class HttpError(Exception):
    """Exception carrying a `response.status_code`, like requests' HTTPError."""

    def __init__(self, status: int, text: str = "boom") -> None:
        super().__init__(text)
        self.response = _Response(status)


class _Job:
    def __init__(self, table: Table) -> None:
        self._table = table

    def get_results(self) -> Table:
        return self._table


class FakeClient:
    """Plays back `script`: an exception instance is raised, a Table is returned."""

    def __init__(self, script: list[object]) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, str]] = []

    def _next(self, mode: str, query: str) -> _Job:
        self.calls.append((mode, query))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, Table)
        return _Job(item)

    def launch_job(self, query: str) -> _Job:
        return self._next("sync", query)

    def launch_job_async(self, query: str) -> _Job:
        return self._next("async", query)


def simple_table() -> Table:
    return Table(
        {"source_id": np.array([1, 2, 3], dtype=np.int64), "ra": [0.1, 0.2, 0.3]}
    )


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"test.tap.{path}")
    logger.propagate = False  # keep the repo's logs/ and root handlers untouched
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def read_log(logger: logging.Logger) -> str:
    for h in logger.handlers:
        h.flush()
    return Path(logger.handlers[0].baseFilename).read_text(encoding="utf-8")  # type: ignore[attr-defined]


def run(client: FakeClient, tmp_path: Path, sleeps: Sleeps | None = None, **kw):
    kw.setdefault("logger", make_logger(tmp_path / "t.log"))
    return run_adql(
        kw.pop("query", QUERY),
        cache_dir=tmp_path / "cache",
        client=client,
        sleep=sleeps if sleeps is not None else Sleeps(),
        **kw,
    )


# --------------------------------------------------------------------------- (1) cache
def test_cache_second_call_skips_client_and_returns_equal_frame(tmp_path):
    first_client = FakeClient([simple_table()])
    first = run(first_client, tmp_path)
    assert len(first_client.calls) == 1

    exploding = FakeClient([AssertionError("client must not be called on cache hit")])
    reformatted = "  SELECT source_id,\n   ra\tFROM gaiadr3.gaia_source   WHERE ra < 1 "
    second = run(exploding, tmp_path, query=reformatted)

    assert exploding.calls == []
    pd.testing.assert_frame_equal(first, second)


def test_cache_key_depends_on_release_and_normalizes_whitespace():
    assert query_cache_key(QUERY, "gaiadr3") != query_cache_key(QUERY, "gaiadr2")
    assert query_cache_key("SELECT  1\n FROM t") == query_cache_key("SELECT 1 FROM t")
    assert query_cache_key("SELECT 1") != query_cache_key("SELECT 2")


def test_cache_miss_for_other_release_calls_client_again(tmp_path):
    run(FakeClient([simple_table()]), tmp_path, release="gaiadr3")
    other = FakeClient([simple_table()])
    run(other, tmp_path, release="gaiadr2")
    assert len(other.calls) == 1
    assert len(list((tmp_path / "cache").glob("*.parquet"))) == 2


def test_cache_written_atomically_leaves_no_tmp(tmp_path):
    run(FakeClient([simple_table()]), tmp_path)
    cache = tmp_path / "cache"
    assert list(cache.glob("*.tmp")) == []
    assert [p.name for p in cache.iterdir()] == [f"{query_cache_key(QUERY)}.parquet"]


def test_cache_write_failure_cleans_tmp_and_still_returns_data(tmp_path, monkeypatch):
    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    df = run(FakeClient([simple_table()]), tmp_path)
    assert len(df) == 3
    assert list((tmp_path / "cache").glob("*")) == []


def test_corrupt_cache_is_refetched(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{query_cache_key(QUERY)}.parquet").write_bytes(b"not parquet")
    client = FakeClient([simple_table()])
    df = run(client, tmp_path)
    assert len(client.calls) == 1
    assert len(df) == 3


def test_asynchronous_flag_selects_launcher(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    c_async = FakeClient([simple_table()])
    run(c_async, tmp_path / "a", asynchronous=True)
    c_sync = FakeClient([simple_table()])
    run(c_sync, tmp_path / "b", asynchronous=False)
    assert c_async.calls[0][0] == "async"
    assert c_sync.calls[0][0] == "sync"


def test_empty_query_and_negative_params_rejected(tmp_path):
    client = FakeClient([simple_table()])
    with pytest.raises(QueryError):
        run(client, tmp_path, query="   \n ")
    with pytest.raises(ValueError):
        run(client, tmp_path, max_retries=-1)
    assert client.calls == []


# --------------------------------------------------------------------------- (2) retry
def test_two_503_then_success_backoff_within_bounds(tmp_path):
    client = FakeClient([HttpError(503), HttpError(503), simple_table()])
    sleeps = Sleeps()
    df = run(client, tmp_path, sleeps, base_delay_s=2.0, max_delay_s=120.0)
    assert len(df) == 3
    assert len(client.calls) == 3
    assert len(sleeps.delays) == 2
    d1, d2 = sleeps.delays
    assert 2.0 * 0.5 <= d1 <= 2.0
    assert 4.0 * 0.5 <= d2 <= 4.0


def test_backoff_is_capped_by_max_delay(tmp_path):
    client = FakeClient([HttpError(503)])
    sleeps = Sleeps()
    with pytest.raises(RetryExhaustedError):
        run(
            client, tmp_path, sleeps, max_retries=5, base_delay_s=10.0, max_delay_s=15.0
        )
    assert len(sleeps.delays) == 5
    assert all(0 <= d <= 15.0 for d in sleeps.delays)
    # base*2^(n-1) = 10, 20, 40...: from the 2nd retry on, the cap (15) binds.
    assert sleeps.delays[0] <= 10.0
    assert all(7.5 <= d <= 15.0 for d in sleeps.delays[1:])


@pytest.mark.parametrize(
    "error",
    [
        HttpError(429, "too many requests"),
        TimeoutError("read timed out"),
        ConnectionError("connection reset"),
        Exception("HTTP Error 502: Bad Gateway"),
    ],
    ids=["http429", "timeout", "connection", "status-in-text"],
)
def test_transient_errors_are_retried(tmp_path, error):
    client = FakeClient([error, simple_table()])
    sleeps = Sleeps()
    df = run(client, tmp_path, sleeps)
    assert len(df) == 3
    assert len(client.calls) == 2
    assert len(sleeps.delays) == 1


# --------------------------------------------------------------------------- (3) exhaustion
@pytest.mark.parametrize("max_retries", [0, 1, 3])
def test_exhaustion_makes_max_retries_plus_one_calls(tmp_path, max_retries):
    client = FakeClient([HttpError(503)])
    sleeps = Sleeps()
    with pytest.raises(RetryExhaustedError):
        run(client, tmp_path, sleeps, max_retries=max_retries)
    assert len(client.calls) == max_retries + 1
    assert len(sleeps.delays) == max_retries
    assert (
        list((tmp_path / "cache").glob("*.parquet")) == []
        if (tmp_path / "cache").exists()
        else True
    )


# --------------------------------------------------------------------------- (4) no retry
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (Exception("ADQL syntax error near 'FORM'"), QueryError),
        (HttpError(400, "bad request"), QueryError),
        (HttpError(404, "not here"), QueryError),
        (Exception("HTTP Error 404: Not Found"), QueryError),
        (ValueError("something unknown"), QueryError),
        (Exception("DB quota exceeded for user"), QuotaExceededError),
        (HttpError(401, "nope"), CredentialError),
        (HttpError(403, "forbidden"), CredentialError),
        (Exception("Error 403: forbidden"), CredentialError),
    ],
    ids=[
        "adql",
        "400",
        "404",
        "404-text",
        "unknown",
        "quota",
        "401",
        "403",
        "403-text",
    ],
)
def test_permanent_errors_are_not_retried(tmp_path, error, expected):
    client = FakeClient([error, simple_table()])
    sleeps = Sleeps()
    with pytest.raises(expected):
        run(client, tmp_path, sleeps)
    assert len(client.calls) == 1
    assert sleeps.delays == []


def test_quota_wins_over_5xx_status(tmp_path):
    client = FakeClient([HttpError(500, "DB quota exceeded")])
    with pytest.raises(QuotaExceededError):
        run(client, tmp_path)
    assert len(client.calls) == 1


# --------------------------------------------------------------------------- (5) probe
def test_5xx_without_text_marker_is_retried(tmp_path):
    # Current behaviour: a bare 5xx has no ADQL marker, so it is transient.
    client = FakeClient([HttpError(500, "Internal Server Error"), simple_table()])
    run(client, tmp_path)
    assert len(client.calls) == 2


def test_5xx_with_syntax_error_text_is_not_retried(tmp_path):
    # Current behaviour: text markers beat the status code, so a permanent ADQL error that
    # the server reports as 500 fails fast (good).
    client = FakeClient([HttpError(500, "ADQL syntax error at line 1")])
    with pytest.raises(QueryError):
        run(client, tmp_path)
    assert len(client.calls) == 1


def test_transient_5xx_whose_text_mentions_adql_is_retried(tmp_path):
    # The status code is checked first: a 503 whose body merely mentions "ADQL" is the
    # server describing itself, not an invalid query.
    client = FakeClient(
        [HttpError(503, "ADQL service temporarily unavailable"), simple_table()]
    )
    sleeps = Sleeps()
    df = run(client, tmp_path, sleeps)
    assert len(client.calls) == 2
    assert len(sleeps.delays) == 1
    assert len(df) == 3


def test_persistent_503_mentioning_adql_exhausts_retries(tmp_path):
    client = FakeClient([HttpError(503, "ADQL service temporarily unavailable")])
    sleeps = Sleeps()
    with pytest.raises(RetryExhaustedError):
        run(client, tmp_path, sleeps, max_retries=2)
    assert len(client.calls) == 3
    assert len(sleeps.delays) == 2


@pytest.mark.parametrize(
    ("status", "text", "expected"),
    [
        (503, "ADQL authentication service busy", "transient"),
        (429, "too many requests", "transient"),
        (429, "authentication rate limit", "transient"),
        (500, "ADQL syntax error near FROM", "permanent"),
        (400, "authentication required", "auth"),
        (401, "nope", "auth"),
        (403, "nope", "auth"),
        (404, "no such resource", "permanent"),
    ],
)
def test_classify_checks_status_before_text(status, text, expected):
    kind, got_status = tap._classify(HttpError(status, text))
    assert (kind, got_status) == (expected, status)


@pytest.mark.parametrize(
    "exc", [PermissionError("denied"), FileNotFoundError("no such file")]
)
def test_local_filesystem_errors_are_permanent_without_retry_or_sleep(tmp_path, exc):
    client = FakeClient([exc])
    sleeps = Sleeps()
    with pytest.raises(QueryError):
        run(client, tmp_path, sleeps)
    assert len(client.calls) == 1
    assert sleeps.delays == []


@pytest.mark.parametrize("exc", [ConnectionResetError("reset"), TimeoutError("slow")])
def test_connection_and_timeout_errors_are_retried(tmp_path, exc):
    client = FakeClient([exc, simple_table()])
    sleeps = Sleeps()
    run(client, tmp_path, sleeps)
    assert len(client.calls) == 2
    assert len(sleeps.delays) == 1


# --------------------------------------------------------------------------- (6) missing values
def masked_table() -> Table:
    return Table(
        {
            "f": MaskedColumn([1.5, 2.5, 3.5], mask=[False, True, False]),
            "i_masked": MaskedColumn(
                np.array([1, 2, 3], dtype=np.int64), mask=[True, False, False]
            ),
            "i_plain": np.array([7, 8, 9], dtype=np.int64),
            "b_masked": MaskedColumn(
                np.array([True, False, True]), mask=[False, False, True]
            ),
            "s": MaskedColumn(["a", "b", "c"], mask=[False, False, True]),
            "f_nan": np.array([np.nan, 1.0, 2.0]),
        }
    )


def test_missing_value_contract(tmp_path):
    df = run(FakeClient([masked_table()]), tmp_path)

    assert df["f"].dtype == np.float64
    assert np.isnan(df["f"].iloc[1]) and df["f"].iloc[0] == 1.5

    assert str(df["i_masked"].dtype) == "Int64"
    assert df["i_masked"].isna().tolist() == [True, False, False]
    assert df["i_masked"].iloc[1] == 2
    assert df["i_masked"].iloc[0] is pd.NA

    assert df["i_plain"].dtype == np.int64

    assert str(df["b_masked"].dtype) == "boolean"
    assert df["b_masked"].isna().tolist() == [False, False, True]

    assert df["s"].dtype == object
    assert df["s"].iloc[2] is None
    assert df["s"].iloc[:2].tolist() == ["a", "b"]

    assert df["f_nan"].isna().tolist() == [True, False, False]


def test_no_masked_objects_leak(tmp_path):
    df = run(FakeClient([masked_table()]), tmp_path)
    for name in df.columns:
        col = df[name]
        assert not isinstance(col, MaskedColumn)
        assert not isinstance(col.values, np.ma.MaskedArray)
        assert not any(
            isinstance(v, (np.ma.MaskedArray, np.ma.core.MaskedConstant)) for v in col
        )


def test_missing_values_survive_cache_roundtrip(tmp_path):
    first = run(FakeClient([masked_table()]), tmp_path)
    second = run(FakeClient([AssertionError("no call")]), tmp_path)
    for name in first.columns:
        assert second[name].isna().tolist() == first[name].isna().tolist()
    assert second["i_masked"].dtype == first["i_masked"].dtype
    assert str(second["b_masked"].dtype) == "boolean"
    assert np.isnan(second["f"].iloc[1])


def test_missing_string_is_none_also_on_cache_hit(tmp_path):
    run(FakeClient([masked_table()]), tmp_path)
    second = run(FakeClient([AssertionError("no call")]), tmp_path)
    assert second["s"].iloc[2] is None


def test_empty_result_table(tmp_path):
    table = Table({"a": np.array([], dtype=np.int64), "b": np.array([], dtype=float)})
    df = run(FakeClient([table]), tmp_path)
    assert len(df) == 0
    assert list(df.columns) == ["a", "b"]


# --------------------------------------------------------------------------- (7) log
def test_attempts_logged_without_query_text(tmp_path):
    logger = make_logger(tmp_path / "ingest.log")
    secret_query = "SELECT hidden_column_zzz FROM gaiadr3.gaia_source"
    client = FakeClient([HttpError(503), simple_table()])
    run_adql(
        secret_query,
        cache_dir=tmp_path / "cache",
        client=client,
        sleep=Sleeps(),
        logger=logger,
    )
    text = read_log(logger)
    assert "attempt=1/6" in text and "attempt=2/6" in text
    assert "retry in" in text and "ok rows=3" in text
    assert "hidden_column_zzz" not in text
    assert "gaia_source" not in text
    assert query_cache_key(secret_query)[:12] in text


def test_log_on_cache_hit_has_no_query_text(tmp_path):
    logger = make_logger(tmp_path / "ingest.log")
    q = "SELECT hidden_column_zzz FROM t"
    for _ in range(2):
        run_adql(
            q,
            cache_dir=tmp_path / "cache",
            client=FakeClient([simple_table()]),
            sleep=Sleeps(),
            logger=logger,
        )
    text = read_log(logger)
    assert "cache hit" in text
    assert "hidden_column_zzz" not in text


def test_secrets_redacted_in_log_and_exception(tmp_path):
    logger = make_logger(tmp_path / "ingest.log")
    err = Exception("request failed password=abc token=abc cookie=abc")
    client = FakeClient([err])
    with pytest.raises(QueryError) as info:
        run_adql(
            QUERY,
            cache_dir=tmp_path / "cache",
            client=client,
            sleep=Sleeps(),
            logger=logger,
        )
    for text in (read_log(logger), str(info.value)):
        for key in ("password", "token", "cookie"):
            assert f"{key}=abc" not in text
            assert f"{key}=***" in text


def test_sanitize_redacts_colon_and_header_forms():
    out = tap._sanitize("Authorization: Bearer abc.def JSESSIONID=XYZ passwd: s3cret")
    assert "abc.def" not in out and "XYZ" not in out and "s3cret" not in out


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        ("Authorization: Bearer abc.def", "abc.def"),
        ("retry with Bearer tok.en-9/x=", "tok.en-9"),
        ("{'password': 'S3CRET'}", "S3CRET"),
        ('{"token": "T0K"}', "T0K"),
        ("GET https://x/y?key=AIzaSECRET", "AIzaSECRET"),
        ("x-goog-api-key: AIzaSECRET", "AIzaSECRET"),
    ],
)
def test_sanitize_leaves_no_literal_secret(message, secret):
    out = tap._sanitize(message)
    assert secret not in out
    assert "***" in out


def test_sanitize_replaces_query_text_with_placeholder():
    q = "SELECT hidden_column_zzz\n  FROM t"
    flat = " ".join(q.split())
    for message in (f"bad [{q}] end", f"bad [{flat}] end"):
        out = tap._sanitize(message, q)
        assert "hidden_column_zzz" not in out
        assert "<query>" in out


def test_default_logger_writes_to_logs_ingest_log_path():
    # Only inspects the configured path; never creates handlers or files.
    assert tap._LOG_PATH.name == "ingest.log"
    assert tap._LOG_PATH.parent.name == "logs"


def test_probe_error_message_echoing_the_query_reaches_the_log(tmp_path):
    # Servers often echo the query in the error body; it must not reach the log.
    logger = make_logger(tmp_path / "ingest.log")
    q = "SELECT hidden_column_zzz FROM t"
    err = Exception(f"Error 500: could not run [{q}]")
    with pytest.raises(RetryExhaustedError):
        run_adql(
            q,
            cache_dir=tmp_path / "cache",
            client=FakeClient([err]),
            sleep=Sleeps(),
            logger=logger,
            max_retries=0,
        )
    assert "hidden_column_zzz" not in read_log(logger)


# --------------------------------------------------------------------------- (8) login
def test_login_interactive_takes_no_parameters():
    sig = inspect.signature(login_interactive)
    assert list(sig.parameters) == []


def _install_fake_gaia(monkeypatch, login):
    gaia = types.SimpleNamespace(login=login)
    module = types.ModuleType("astroquery.gaia")
    module.Gaia = gaia  # type: ignore[attr-defined]
    if "astroquery" not in sys.modules:
        monkeypatch.setitem(sys.modules, "astroquery", types.ModuleType("astroquery"))
    monkeypatch.setitem(sys.modules, "astroquery.gaia", module)


def test_login_failure_raises_credential_error_without_leaking(monkeypatch):
    def failing_login():
        raise RuntimeError("bad login password=abc")

    _install_fake_gaia(monkeypatch, failing_login)
    with pytest.raises(CredentialError) as info:
        login_interactive()
    assert "abc" not in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_login_success_returns_none(monkeypatch):
    called = []
    _install_fake_gaia(monkeypatch, lambda: called.append(1))
    assert login_interactive() is None
    assert called == [1]
