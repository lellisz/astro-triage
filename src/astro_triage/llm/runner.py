"""Run Layer 2 for one release fold, model and arm (ADR 0005 D4-D6).

The loop is sequential and resumable: terminal cache keys are skipped, every attempt is written
to the ledger by ``chat``'s ``after_attempt`` hook, and the run stops (after recording) on a
quota stop, a fatal API status, three consecutive retryable failures or a failed drift canary.
Every routed object stays in the denominator: ``report["pending"]`` counts the ones with no
terminal record, and ``consolidate`` refuses while any exists.
"""

from __future__ import annotations

import random
import time
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from astro_triage.cost.quota import Ledger, QuotaExceeded, QuotaGuard
from astro_triage.llm.cache import ResultCache
from astro_triage.llm.client import CallResult, FatalCallError, chat, open_no_redirect
from astro_triage.llm.consolidate import IncompleteRunError, consolidate
from astro_triage.llm.prompt import PROMPT_VERSION, ParseError, parse_answer
from astro_triage.llm.routing import (
    DEFAULT_SEED,
    MANIFEST_NAME,
    ManifestMismatchError,
    max_tokens_for,
    request_for,
    select_routed,
)

__all__ = [
    "IncompleteRunError",
    "ManifestMismatchError",
    "consolidate",
    "run",
    "select_routed",
]

CANARY_N: Final = 20
CANARY_MIN_AGREE: Final = 18
MAX_CONSECUTIVE_RETRYABLE: Final = 3
_RETRYABLE: Final = frozenset(
    {"RATE_LIMITED", "SERVER_ERROR", "TIMEOUT", "NETWORK_ERROR"}
)
_ANSWERED: Final = frozenset({"OK", "UNKNOWN"})


def _classify(res: CallResult) -> tuple[str, str | None, int | None]:
    """Map a call result to (D4 status, label, raw confidence)."""
    if res.status != "OK":
        return res.status, None, None
    if res.finish_reason != "stop":
        # content_filter, None, tool_calls...: the answer cannot be trusted, and it is
        # deterministic, so it is a terminal schema failure instead of an endless retry.
        return "SCHEMA_VIOLATION", None, None
    try:
        answer = parse_answer(res.content)
    except ParseError as exc:
        return exc.status, None, None
    status = "UNKNOWN" if answer.label == "UNKNOWN" else "OK"
    return status, answer.label, answer.confidence


def _utc_ms(now: Callable[[], datetime]) -> str:
    return now().astimezone(UTC).isoformat(timespec="milliseconds")


def run(
    release_dir: Path,
    fold: Literal["val", "test"],
    model: str,
    arm: str,
    enable_thinking: bool,
    guard: QuotaGuard,
    cache: ResultCache,
    ledger: Ledger,
    key: str,
    sample: int | None = None,
    canary: bool = True,
    *,
    run_id: str | None = None,
    opener: Callable[..., Any] = open_no_redirect,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Run the canary then the loop; return the run report (``stop_reason`` is None when not stopped)."""
    rid = run_id if run_id is not None else uuid.uuid4().hex[:12]
    manifest = release_dir / "layer2" / MANIFEST_NAME
    max_tokens = max_tokens_for(enable_thinking)
    rows = select_routed(release_dir, fold, manifest, sample)
    counts: Counter[str] = Counter()
    report: dict[str, Any] = {
        "run_id": rid,
        "fold": fold,
        "model": model,
        "arm": arm,
        "enable_thinking": enable_thinking,
        "prompt_version": PROMPT_VERSION,
        "routed": len(rows),
        "cached_skipped": 0,
        "calls_made": 0,
        "model_mismatch": 0,
        "canary": {"status": "disabled"},
        "stop_reason": None,
    }
    current: list[int] = [0]

    def call(messages: list[dict[str, str]]) -> CallResult:
        def after(res: CallResult) -> None:
            guard.record(res.prompt_tokens, res.completion_tokens)
            ledger.append(
                {
                    "ts_utc": _utc_ms(now),
                    "run_id": rid,
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "arm": arm,
                    "enable_thinking": enable_thinking,
                    "source_id": current[0],
                    "attempt": res.attempts,
                    "status": res.status,
                    "http_status": res.http_status,
                    "latency_ms": res.latency_ms,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "finish_reason": res.finish_reason,
                    "reasoning_chars": res.reasoning_chars,
                }
            )

        return chat(
            messages,
            model,
            key=key,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            before_attempt=lambda: guard.before_attempt(max_tokens),
            after_attempt=after,
            opener=opener,
            sleep=sleep,
            rng=rng,
        )

    def run_canary() -> str | None:
        val = (
            rows
            if fold == "val" and sample is None
            else select_routed(release_dir, "val", manifest, None)
        )
        cached: list[tuple[pd.Series, str]] = []
        for _, row in val.iterrows():
            _, ckey = request_for(row, model, arm, enable_thinking)
            rec = cache.get(ckey)
            if rec is not None and rec.get("status") in _ANSWERED:
                cached.append((row, str(rec["label_l2"])))
        if len(cached) < CANARY_N:
            report["canary"] = {"status": "skipped", "cached_val": len(cached)}
            return None
        order = np.random.default_rng(DEFAULT_SEED).permutation(len(cached))
        tally = {"agree": 0, "disagree": 0, "inconclusive": 0}
        # Written before the first call so that a fatal stop or a quota stop mid-canary
        # still leaves the partial counts in the report.
        report["canary"] = {"status": "inconclusive", **tally, "n": CANARY_N}
        for i in order[:CANARY_N]:
            row, label = cached[int(i)]
            current[0] = int(row["source_id"])
            messages, _ = request_for(row, model, arm, enable_thinking)
            res = call(messages)
            report["calls_made"] += 1
            if res.status in _RETRYABLE:
                tally["inconclusive"] += 1  # transport failure, not model behaviour
            else:
                status, new_label, _ = _classify(res)
                same = status in _ANSWERED and new_label == label
                tally["agree" if same else "disagree"] += 1
            report["canary"] = {**report["canary"], **tally}
        if tally["disagree"] > CANARY_N - CANARY_MIN_AGREE:
            report["canary"]["status"] = "drift"
            return "DRIFT"
        if tally["agree"] + tally["disagree"] < CANARY_MIN_AGREE:
            return "CANARY_INCONCLUSIVE"
        report["canary"]["status"] = "pass"
        return None

    stop: str | None = None
    try:
        if canary:
            stop = run_canary()
        consecutive = 0
        for _, row in rows.iterrows() if stop is None else iter(()):
            messages, ckey = request_for(row, model, arm, enable_thinking)
            if cache.is_terminal(ckey):
                report["cached_skipped"] += 1
                continue
            sid = int(row["source_id"])
            current[0] = sid
            try:
                res = call(messages)
            except FatalCallError as exc:
                cache.put(ckey, {"source_id": sid, "status": exc.status})
                raise
            report["calls_made"] += 1
            status, label, confidence = _classify(res)
            counts[status] += 1
            if res.model_reported not in (None, model):
                report["model_mismatch"] += 1
            cache.put(
                ckey,
                {
                    "source_id": sid,
                    "status": status,
                    "label_l2": label,
                    "confidence_raw": confidence,
                    "model_reported": res.model_reported,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "latency_ms": res.latency_ms,
                    "attempts": res.attempts,
                    "finish_reason": res.finish_reason,
                    "reasoning_chars": res.reasoning_chars,
                    "called_utc": _utc_ms(now),
                },
            )
            consecutive = consecutive + 1 if status in _RETRYABLE else 0
            if consecutive >= MAX_CONSECUTIVE_RETRYABLE:
                stop = "CONSECUTIVE_RETRYABLE"
                break
    except QuotaExceeded:
        stop = "QUOTA_STOP"
    except FatalCallError as exc:
        stop = exc.status
    report["stop_reason"] = stop
    report["status_counts"] = dict(sorted(counts.items()))
    report["calls_done"] = guard.calls_done
    report["tokens_done"] = guard.tokens_done
    report["pending"] = sum(
        not cache.is_terminal(request_for(row, model, arm, enable_thinking)[1])
        for _, row in rows.iterrows()
    )
    return report
