"""Gaia Archive ADQL/TAP client with a local Parquet cache, retry and an ingest log.

Missing-value contract of the returned DataFrame (no astropy masks ever leak out):
  * float columns: masked or NaN entries become NaN (pandas' own missing marker for floats);
  * integer and boolean columns: plain numpy dtype when nothing is missing, pandas nullable
    dtype (Int64, UInt8, boolean, ...) with pd.NA when something is masked;
  * string columns: object dtype with None for missing entries.
ADR 0001 decision 3 ("NULL on disk, never NaN") is enforced by the Parquet writer that
consumes this frame, not here.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import random
import re
import socket
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from astropy.table import MaskedColumn, Table

__all__ = [
    "CredentialError",
    "IngestError",
    "QueryError",
    "QuotaExceededError",
    "RetryExhaustedError",
    "login_interactive",
    "query_cache_key",
    "run_adql",
]

_LOG_PATH = Path(__file__).resolve().parents[3] / "logs" / "ingest.log"
_LOGGER_NAME = "astro_triage.ingest.tap"
_handler_lock = threading.Lock()

_TRANSIENT_STATUSES = frozenset({429}) | frozenset(range(500, 600))
_STATUS_RE = re.compile(
    r"(?:\bHTTP Error|\bError|\bstatus(?: code)?)\D{0,3}(\d{3})\b", re.IGNORECASE
)
_SECRET_RE = re.compile(
    r"""(?P<k>(?<![\w-])['"]?(?:password|passwd|pwd|token|authorization|cookie|jsessionid"""
    r"""|secret|x-goog-api-key|api[_-]?key|key)['"]?\s*[=:]\s*)"""
    r"""(?:(?:bearer|basic)\s+)?(?:'[^']*'|"[^"]*"|\S+)""",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bbearer\s+[\w.~+/=-]+", re.IGNORECASE)
_NETWORK_ERRNOS = frozenset(
    getattr(errno, name)
    for name in (
        "ETIMEDOUT",
        "ECONNRESET",
        "ECONNREFUSED",
        "ECONNABORTED",
        "ENETUNREACH",
        "ENETDOWN",
        "ENETRESET",
        "EHOSTUNREACH",
        "EHOSTDOWN",
        "EPIPE",
    )
    if hasattr(errno, name)
)
_QUOTA_MARKERS = ("quota exceeded", "db quota")
_AUTH_MARKERS = (
    "unauthorized",
    "authentication",
    "credential",
    "login failed",
    "bad password",
)
_ADQL_MARKERS = (
    "adql",
    "syntax error",
    "parsing",
    "unexpected token",
    "unknown table",
    "unknown column",
    "not found in",
    "was not found",
    "incorrect",
)
_MAX_LOGGED_MESSAGE = 300


class IngestError(Exception):
    """Base class for every error raised by this module."""


class QueryError(IngestError):
    """Permanent failure (invalid ADQL, HTTP 4xx other than 429, unknown error). Not retried."""


class QuotaExceededError(IngestError):
    """The Gaia Archive user quota is exhausted. Not retried."""


class CredentialError(IngestError):
    """Missing or rejected credentials. Not retried."""


class RetryExhaustedError(IngestError):
    """Every attempt failed with a transient error."""


class _Job(Protocol):
    def get_results(self) -> Table: ...


class TapClient(Protocol):
    """Subset of astroquery.gaia.Gaia used here; inject a fake in tests."""

    def launch_job(self, query: str) -> _Job: ...

    def launch_job_async(self, query: str) -> _Job: ...


class _AstroqueryClient:
    def __init__(self) -> None:
        from astroquery.gaia import Gaia

        # The anonymous default of 50 rows would silently truncate every result.
        Gaia.ROW_LIMIT = -1
        self._gaia = Gaia

    def launch_job(self, query: str) -> _Job:
        return self._gaia.launch_job(query)

    def launch_job_async(self, query: str) -> _Job:
        return self._gaia.launch_job_async(query)


def query_cache_key(query: str, release: str = "gaiadr3") -> str:
    normalized = " ".join(query.split())
    return hashlib.sha256(f"{release}\n{normalized}".encode()).hexdigest()


def _default_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    with _handler_lock:
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(_LOG_PATH, encoding="utf-8", delay=True)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    return logger


def _sanitize(message: str, query: str | None = None) -> str:
    """Flatten, drop the query text and secrets, and truncate a message for logs."""
    if query is not None and query.strip():
        flat_query = " ".join(query.split())
        message = message.replace(query, "<query>")
        message = " ".join(message.split()).replace(flat_query, "<query>")
    flat = " ".join(message.split())
    flat = _SECRET_RE.sub(r"\g<k>***", flat)
    return _BEARER_RE.sub("Bearer ***", flat)[:_MAX_LOGGED_MESSAGE]


def _is_network_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, socket.gaierror)):
        return True
    return isinstance(exc, OSError) and exc.errno in _NETWORK_ERRNOS


def _status_of(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    match = _STATUS_RE.search(str(exc))
    return int(match.group(1)) if match else None


def _classify(exc: BaseException) -> tuple[str, int | None]:
    """Return ("quota" | "auth" | "permanent" | "transient", http status)."""
    text = str(exc).lower()
    status = _status_of(exc)
    if any(m in text for m in _QUOTA_MARKERS):
        return "quota", status
    if status in (401, 403):
        return "auth", status
    if status in _TRANSIENT_STATUSES:
        # Only an unambiguous ADQL syntax error is permanent behind a 5xx; a bare "adql" or
        # "authentication" in the body of a 503/429 is just the server describing itself.
        if status != 429 and (
            "syntax error" in text
            or ("incorrect" in text and "adql" in text and "pars" in text)
        ):
            return "permanent", status
        return "transient", status
    if status is not None:
        if any(m in text for m in _AUTH_MARKERS):
            return "auth", status
        return "permanent", status
    if any(m in text for m in _AUTH_MARKERS):
        return "auth", None
    if any(m in text for m in _ADQL_MARKERS):
        return "permanent", None
    if _is_network_error(exc) or "timed out" in text:
        return "transient", None
    return "permanent", None


def _none_for_missing_text(df: pd.DataFrame) -> pd.DataFrame:
    for name in df.columns:
        col = df[name]
        if col.dtype == object or isinstance(col.dtype, pd.StringDtype):
            df[name] = pd.Series(
                [
                    None if v is pd.NA or (isinstance(v, float) and np.isnan(v)) else v
                    for v in col.tolist()
                ],
                index=col.index,
                dtype=object,
            )
    return df


def _to_dataframe(table: Table) -> pd.DataFrame:
    columns: dict[str, pd.Series | pd.api.extensions.ExtensionArray | np.ndarray] = {}
    for name in table.colnames:
        col = table[name]
        data = np.ma.getdata(col)
        mask = (
            np.ma.getmaskarray(col)
            if isinstance(col, MaskedColumn)
            else np.zeros(len(col), bool)
        )
        kind = data.dtype.kind
        if kind in "fiub":
            data = data.astype(data.dtype.newbyteorder("="), copy=True)
        if kind == "f":
            data[mask] = np.nan
            columns[name] = data
        elif kind in "iu":
            columns[name] = pd.arrays.IntegerArray(data, mask) if mask.any() else data
        elif kind == "b":
            columns[name] = pd.arrays.BooleanArray(data, mask) if mask.any() else data
        else:
            values = [
                None
                if m
                else (v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
                for v, m in zip(data.tolist(), mask.tolist(), strict=True)
            ]
            columns[name] = pd.Series(values, dtype=object)
    return pd.DataFrame(columns)


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def run_adql(
    query: str,
    *,
    cache_dir: str | Path,
    release: str = "gaiadr3",
    asynchronous: bool = True,
    max_retries: int = 5,
    base_delay_s: float = 2.0,
    max_delay_s: float = 120.0,
    logger: logging.Logger | None = None,
    client: TapClient | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Run an ADQL query, served from `cache_dir/<sha256>.parquet` when present.

    `max_retries` is the number of retries after the first attempt (max_retries + 1 calls).
    Raises QuotaExceededError, CredentialError, QueryError (permanent) or RetryExhaustedError.
    """
    if not query.strip():
        raise QueryError("Empty ADQL query.")
    if max_retries < 0 or base_delay_s < 0 or max_delay_s < 0:
        raise ValueError("max_retries, base_delay_s and max_delay_s must be >= 0.")

    log = logger if logger is not None else _default_logger()
    key = query_cache_key(query, release)
    short = key[:12]
    cache_path = Path(cache_dir) / f"{key}.parquet"

    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
        except (OSError, ValueError) as exc:
            log.warning(
                "query=%s cache unreadable, refetching: %s",
                short,
                _sanitize(str(exc), query),
            )
        else:
            log.info("query=%s cache hit rows=%d", short, len(cached))
            return _none_for_missing_text(cached)

    tap = client if client is not None else _AstroqueryClient()
    total = max_retries + 1
    for attempt in range(1, total + 1):
        try:
            launch = tap.launch_job_async if asynchronous else tap.launch_job
            table = launch(query).get_results()
            df = _to_dataframe(table)
        except Exception as exc:
            kind, status = _classify(exc)
            detail = (
                f"{type(exc).__name__} status={status} {_sanitize(str(exc), query)}"
            )
            if kind == "quota":
                log.error(
                    "query=%s attempt=%d/%d permanent quota: %s",
                    short,
                    attempt,
                    total,
                    detail,
                )
                raise QuotaExceededError(
                    "Gaia Archive quota exceeded. Delete old jobs and user tables at "
                    "https://gea.esac.esa.int/archive/ (Jobs tab), or wait for the quota to "
                    "reset, then run again."
                ) from exc
            if kind == "auth":
                log.error(
                    "query=%s attempt=%d/%d permanent auth: %s",
                    short,
                    attempt,
                    total,
                    detail,
                )
                raise CredentialError(
                    "Gaia Archive rejected the credentials or the request needs a login. "
                    "Call login_interactive() and type your user and password in the terminal."
                ) from exc
            if kind == "permanent":
                log.error(
                    "query=%s attempt=%d/%d permanent error: %s",
                    short,
                    attempt,
                    total,
                    detail,
                )
                raise QueryError(
                    f"Gaia TAP query failed without retry: {detail}"
                ) from exc
            if attempt == total:
                log.error(
                    "query=%s attempt=%d/%d transient, giving up: %s",
                    short,
                    attempt,
                    total,
                    detail,
                )
                raise RetryExhaustedError(
                    f"Gaia TAP still failing after {total} attempts: {detail}"
                ) from exc
            delay = min(
                max_delay_s, base_delay_s * 2 ** (attempt - 1)
            ) * random.uniform(0.5, 1.0)
            log.warning(
                "query=%s attempt=%d/%d transient error, retry in %.1fs: %s",
                short,
                attempt,
                total,
                delay,
                detail,
            )
            sleep(delay)
            continue

        log.info("query=%s attempt=%d/%d ok rows=%d", short, attempt, total, len(df))
        try:
            _write_cache(df, cache_path)
        except (OSError, ValueError) as exc:
            log.warning(
                "query=%s cache write failed: %s", short, _sanitize(str(exc), query)
            )
        return df
    raise AssertionError("unreachable")  # pragma: no cover


def login_interactive() -> None:
    """Start Gaia.login(); astroquery asks for user and password in the terminal."""
    from astroquery.gaia import Gaia

    try:
        Gaia.login()
    except Exception as exc:
        raise CredentialError(
            f"Gaia login failed ({type(exc).__name__}). Check the user name and try again; "
            "the password is only typed in the terminal prompt."
        ) from exc
