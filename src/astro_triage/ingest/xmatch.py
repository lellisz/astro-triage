"""CDS XMatch client: upload a CSV of positions, receive the pairs with a VizieR catalogue.

The raw CSV answer is streamed to a temporary file and moved into place with ``os.replace``,
so a failed or truncated transfer never leaves a partial file under the final name. Integer
identifiers (``Source``, ``sid``) are read as int64 or string and never pass through float.
The service itself parses an all-digit upload column as double, so callers must send
19-digit ids with a non-digit prefix.
Errors reuse the ``IngestError`` hierarchy of ``tap``: 4xx (except 429) is permanent
(``QueryError``), timeouts, connection resets, 429 and 5xx are retried with exponential
backoff (``RetryExhaustedError`` when every attempt fails).

The uploaded body and local absolute paths are never logged.
"""

from __future__ import annotations

import logging
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import requests

from astro_triage.ingest.tap import (
    IngestError,
    QueryError,
    RetryExhaustedError,
    _default_logger,
    _sanitize,
)

__all__ = [
    "XMATCH_MAX_ROWS",
    "XMATCH_URL",
    "IngestError",
    "QueryError",
    "RetryExhaustedError",
    "TruncatedResultError",
    "XMatchStats",
    "read_xmatch_csv",
    "xmatch",
    "xmatch_to_file",
]

# https is mandatory: the http:// endpoint resets the connection (R-09).
XMATCH_URL: Final = "https://cdsxmatch.cds.unistra.fr/xmatch/api/v1/sync"
XMATCH_MAX_ROWS: Final = 2_000_000
_MAX_RADIUS_ARCSEC: Final = 180.0
_SELECTIONS: Final = frozenset({"best", "all"})
_CONNECT_TIMEOUT_S: Final = 30.0
_CHUNK_BYTES: Final = 1024 * 1024
_MAX_DELAY_S: Final = 120.0
_MAX_ARROW_BLOCK: Final = 2**31 - 1
_WINDOWS_PATH: Final = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|]*")
_ID_COLUMN_TYPES: Final[dict[str, pa.DataType]] = {
    "Source": pa.int64(),
    "sid": pa.string(),
    "cls": pa.string(),
}


class TruncatedResultError(IngestError):
    """The service output reached the 2,000,000-row truncation limit; the result is incomplete."""


@dataclass(frozen=True)
class XMatchStats:
    rows: int
    bytes_received: int
    attempts: int


def _validate(catalog: str, radius_arcsec: float, selection: str) -> None:
    if not catalog.strip():
        raise ValueError("catalog must not be empty")
    if not 0 < radius_arcsec <= _MAX_RADIUS_ARCSEC:
        raise ValueError(f"radius_arcsec must be in (0, {_MAX_RADIUS_ARCSEC:g}]")
    if selection not in _SELECTIONS:
        raise ValueError(f"selection must be one of {sorted(_SELECTIONS)}")


def _mask_paths(text: str) -> str:
    """Replace the home directory and Windows absolute paths with ``<path>``."""
    home = str(Path.home())
    if home:
        text = text.replace(home, "<path>")
    return _WINDOWS_PATH.sub("<path>", text)


def _safe_detail(text: str) -> str:
    return _mask_paths(_sanitize(text))


def _stream_to(response: requests.Response, tmp_name: str) -> tuple[int, int]:
    """Write the body to ``tmp_name``; return (bytes, data rows excluding the header)."""
    total = 0
    lines = 0
    last = b"\n"
    with open(tmp_name, "wb") as handle:
        for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
            if not chunk:
                continue
            handle.write(chunk)
            total += len(chunk)
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        lines += 1
    return total, max(lines - 1, 0)


def _check_header(tmp_name: str) -> None:
    with open(tmp_name, "rb") as handle:
        head = handle.read(_CHUNK_BYTES).split(b"\n", 1)[0].decode("utf-8", "replace")
    if "angDist" not in head:
        raise QueryError(
            "CDS XMatch answered HTTP 200 but the body is not the expected CSV: "
            + _sanitize(head)
        )


def xmatch_to_file(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    catalog: str,
    radius_arcsec: float,
    selection: str,
    ra_col: str,
    dec_col: str,
    timeout_s: float = 900,
    max_retries: int = 4,
    base_delay_s: float = 5.0,
    logger: logging.Logger | None = None,
) -> XMatchStats:
    """Upload ``csv_path`` and write the raw CSV answer to ``out_path`` (temp file + replace).

    ``max_retries`` is the number of retries after the first attempt. Raises ``QueryError``
    (permanent), ``RetryExhaustedError`` or ``TruncatedResultError``.
    """
    _validate(catalog, radius_arcsec, selection)
    if max_retries < 0 or base_delay_s < 0:
        raise ValueError("max_retries and base_delay_s must be >= 0")
    source = Path(csv_path)
    if not source.is_file():
        raise QueryError(f"upload file not found: {source.name}")
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    log = logger if logger is not None else _default_logger()
    params = {
        "request": "xmatch",
        "distMaxArcsec": f"{radius_arcsec:g}",
        "selection": selection,
        "RESPONSEFORMAT": "csv",
        "cat2": catalog,
        "colRA1": ra_col,
        "colDec1": dec_col,
    }
    label = f"xmatch cat2={catalog} file={source.name}"
    total = max_retries + 1
    for attempt in range(1, total + 1):
        fd, tmp_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp"
        )
        os.close(fd)
        try:
            with source.open("rb") as upload:
                response = requests.post(
                    XMATCH_URL,
                    data=params,
                    files={"cat1": (source.name, upload, "text/csv")},
                    timeout=(_CONNECT_TIMEOUT_S, timeout_s),
                    stream=True,
                )
            with response:
                status = response.status_code
                if status == 429 or status >= 500:
                    raise _Transient(f"HTTP {status}")
                if status >= 400:
                    detail = _sanitize(response.text)
                    log.error("%s permanent HTTP %d: %s", label, status, detail)
                    raise QueryError(
                        f"CDS XMatch rejected the request (HTTP {status}): {detail}"
                    )
                size, rows = _stream_to(response, tmp_name)
            _check_header(tmp_name)
            if rows >= XMATCH_MAX_ROWS:
                log.error("%s output truncated rows=%d", label, rows)
                raise TruncatedResultError(
                    f"CDS XMatch output has {rows} rows, at the {XMATCH_MAX_ROWS}-row "
                    "truncation limit; split the upload into smaller batches or use "
                    "selection=best"
                )
            os.replace(tmp_name, target)
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            _Transient,
        ) as exc:
            Path(tmp_name).unlink(missing_ok=True)
            detail = f"{type(exc).__name__} {_safe_detail(str(exc))}"
            if attempt == total:
                log.error(
                    "%s attempt=%d/%d giving up: %s", label, attempt, total, detail
                )
                raise RetryExhaustedError(
                    f"CDS XMatch still failing after {total} attempts: {detail}"
                ) from exc
            delay = min(
                _MAX_DELAY_S, base_delay_s * 2 ** (attempt - 1)
            ) * random.uniform(0.5, 1.0)
            log.warning(
                "%s attempt=%d/%d transient, retry in %.1fs: %s",
                label,
                attempt,
                total,
                delay,
                detail,
            )
            time.sleep(delay)
            continue
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        log.info(
            "%s attempt=%d/%d ok rows=%d bytes=%d params=%s",
            label,
            attempt,
            total,
            rows,
            size,
            params,
        )
        return XMatchStats(rows=rows, bytes_received=size, attempts=attempt)
    raise AssertionError("unreachable")  # pragma: no cover


class _Transient(Exception):
    """Internal marker for a retryable HTTP status."""


def read_xmatch_csv(path: str | Path) -> pa.Table:
    """Read a raw XMatch CSV; ``Source`` is int64, ``sid`` and ``cls`` are strings.

    The whole file is one parse block, so column types are inferred globally and empty
    values become NULL.
    """
    source = Path(path)
    block = min(source.stat().st_size + 1, _MAX_ARROW_BLOCK)
    return pacsv.read_csv(
        source,
        read_options=pacsv.ReadOptions(block_size=max(block, 1 << 20)),
        convert_options=pacsv.ConvertOptions(column_types=_ID_COLUMN_TYPES),
    )


def xmatch(
    csv_path: str | Path,
    *,
    catalog: str,
    radius_arcsec: float,
    selection: str,
    ra_col: str,
    dec_col: str,
    timeout_s: float = 900,
    max_retries: int = 4,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Cross-match the positions in ``csv_path`` and return the pairs as a DataFrame.

    Nullable Arrow-backed dtypes keep integer ids exact and missing values as NA, never NaN.
    """
    with tempfile.TemporaryDirectory(prefix="xmatch_") as workdir:
        raw = Path(workdir) / "answer.csv"
        xmatch_to_file(
            csv_path,
            raw,
            catalog=catalog,
            radius_arcsec=radius_arcsec,
            selection=selection,
            ra_col=ra_col,
            dec_col=dec_col,
            timeout_s=timeout_s,
            max_retries=max_retries,
            logger=logger,
        )
        return read_xmatch_csv(raw).to_pandas(types_mapper=pd.ArrowDtype)
