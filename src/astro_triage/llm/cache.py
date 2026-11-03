"""Append-only JSONL result cache with resume and a single-writer lock (ADR 0005 D5.4-5).

One line per terminal result, flushed and fsynced per line. Existing lines are never rewritten.
A line damaged by a crash mid-write is skipped with a warning. The lock file is created
exclusively and is never stolen automatically: after a hard kill the user deletes it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

__all__ = ["TERMINAL", "CacheLockedError", "ResultCache", "cache_key", "load_records"]

log = logging.getLogger(__name__)

TERMINAL: Final[frozenset[str]] = frozenset(
    {"OK", "UNKNOWN", "TRUNCATED", "EMPTY_CONTENT", "INVALID_JSON", "SCHEMA_VIOLATION"}
)


class CacheLockedError(RuntimeError):
    """Another writer holds the cache lock (or a previous run was killed)."""


def cache_key(
    model: str,
    prompt_version: str,
    arm: str,
    enable_thinking: bool,
    max_tokens: int,
    filled_prompt_sha256: str,
    source_id: int,
) -> str:
    parts = (
        model,
        prompt_version,
        arm,
        str(int(enable_thinking)),
        str(int(max_tokens)),
        filled_prompt_sha256,
        str(int(source_id)),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _parse_lines(raw: bytes) -> tuple[dict[str, dict[str, Any]], bool]:
    """Last record per key, and whether the file lacks a final newline."""
    records: dict[str, dict[str, Any]] = {}
    needs_newline = bool(raw) and not raw.endswith(b"\n")
    for number, line in enumerate(
        raw.decode("utf-8", errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            log.warning("cache line %d is not valid JSON, ignored", number)
            continue
        key = record.get("key") if isinstance(record, dict) else None
        if not isinstance(key, str):
            log.warning("cache line %d has no key, ignored", number)
            continue
        records[key] = record
    return records, needs_newline


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    """Read-only view of a cache file: last record per key; no lock, never writes."""
    if not path.exists():
        return {}
    return _parse_lines(path.read_bytes())[0]


class ResultCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock_path = path.with_name(path.name + ".lock")
        self._records: dict[str, dict[str, Any]] = {}
        self._needs_newline = False
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise CacheLockedError(
                f"cache is locked by another writer; if no run is active, "
                f"delete {self._lock_path.name}"
            ) from None
        os.close(fd)
        try:
            self._load()
        except BaseException:
            self.close()
            raise

    def _load(self) -> None:
        if not self._path.exists():
            return
        self._records, self._needs_newline = _parse_lines(self._path.read_bytes())

    def get(self, key: str) -> dict[str, Any] | None:
        """Last record stored for ``key``."""
        return self._records.get(key)

    def is_terminal(self, key: str) -> bool:
        record = self._records.get(key)
        return record is not None and record.get("status") in TERMINAL

    def put(self, key: str, record: Mapping[str, object]) -> None:
        if self._closed:
            raise ValueError("cache is closed")
        entry: dict[str, Any] = {"key": key, **record}
        if "source_id" in entry:
            entry["source_id"] = int(entry["source_id"])
        line = json.dumps(entry, ensure_ascii=True, separators=(",", ":"))
        with self._path.open("ab") as fh:
            if self._needs_newline:
                fh.write(b"\n")
                self._needs_newline = False
            fh.write(line.encode("ascii") + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._records[key] = entry

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
