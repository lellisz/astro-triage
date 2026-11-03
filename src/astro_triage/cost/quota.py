"""Pacing and per-run hard caps checked before every HTTP attempt, plus a JSONL ledger (ADR 0005 D5).

Single process, sequential requests. The guard never lets an attempt start if a cap would be
crossed; the ledger accepts only an allowlist of keys so that headers, request bodies and raw
error text cannot be written by mistake.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

__all__ = ["Ledger", "QuotaExceeded", "QuotaGuard"]

# Conservative prompt-size estimate used until the first response reports real usage.
_INITIAL_PROMPT_TOKENS: Final = 2_000
LEDGER_KEYS: Final = frozenset(
    {
        "ts_utc",
        "run_id",
        "model",
        "prompt_version",
        "arm",
        "enable_thinking",
        "source_id",
        "attempt",
        "status",
        "http_status",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "reasoning_chars",
    }
)


class QuotaExceeded(RuntimeError):
    """An attempt was not made because a per-run cap would be exceeded."""


class QuotaGuard:
    def __init__(
        self,
        *,
        rpm_cap: float,
        max_calls: int,
        max_tokens_total: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm_cap <= 0 or max_calls < 0 or max_tokens_total < 0:
            raise ValueError("rpm_cap must be > 0 and the caps must be >= 0")
        self._interval = 60.0 / rpm_cap
        self._max_calls = max_calls
        self._max_tokens_total = max_tokens_total
        self._clock = clock
        self._sleep = sleep
        self._last_start: float | None = None
        self._max_prompt_seen: int | None = None
        self.calls_done = 0
        self.tokens_done = 0

    def before_attempt(self, max_tokens_request: int) -> None:
        """Raise ``QuotaExceeded`` if a cap would be crossed, else pace and count the attempt."""
        if self.calls_done + 1 > self._max_calls:
            raise QuotaExceeded("max_calls would be exceeded")
        est_prompt = (
            _INITIAL_PROMPT_TOKENS
            if self._max_prompt_seen is None
            else self._max_prompt_seen
        )
        if self.tokens_done + est_prompt + max_tokens_request > self._max_tokens_total:
            raise QuotaExceeded("max_tokens_total would be exceeded")
        if self._last_start is not None:
            wait = self._interval - (self._clock() - self._last_start)
            if wait > 0:
                self._sleep(wait)
        self._last_start = self._clock()
        self.calls_done += 1

    def record(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        """Add the usage of one attempt; ``None`` (error responses) counts as 0 tokens."""
        self.tokens_done += (prompt_tokens or 0) + (completion_tokens or 0)
        if prompt_tokens is not None:
            # The conservative initial estimate is replaced by the max actually seen.
            seen = self._max_prompt_seen
            self._max_prompt_seen = (
                prompt_tokens if seen is None else max(seen, prompt_tokens)
            )


class Ledger:
    """Append-only JSONL, one line per HTTP attempt, flushed and fsynced."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, entry: Mapping[str, object]) -> None:
        extra = set(entry) - LEDGER_KEYS
        if extra:
            raise ValueError(f"ledger keys outside the allowlist: {sorted(extra)}")
        line = json.dumps(dict(entry), ensure_ascii=True, separators=(",", ":"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
