"""Proofs for ``cost.quota``: pacing, hard caps and the ledger allowlist (ADR 0005 D5)."""

import itertools
import json
from pathlib import Path

import pytest

from astro_triage.cost.quota import Ledger, QuotaExceeded, QuotaGuard


class FakeTime:
    """A clock that only moves when told to or when the guard sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_guard(ft: FakeTime, **over: float) -> QuotaGuard:
    args = {"rpm_cap": 30, "max_calls": 100, "max_tokens_total": 10**9}
    args.update(over)
    return QuotaGuard(**args, clock=ft.clock, sleep=ft.sleep)  # type: ignore[arg-type]


def test_attempts_are_spaced_at_least_interval_apart() -> None:
    ft = FakeTime()
    guard = make_guard(ft, rpm_cap=30)  # interval 2 s
    starts = []
    for _ in range(5):
        guard.before_attempt(100)
        starts.append(ft.now)
        ft.now += 0.5  # the request itself takes 0.5 s
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert all(g >= 60 / 30 - 1e-9 for g in gaps)
    assert ft.sleeps == pytest.approx([1.5] * 4)
    assert guard.calls_done == 5


def test_no_sleep_when_enough_time_already_passed_and_first_call_is_free() -> None:
    ft = FakeTime()
    guard = make_guard(ft, rpm_cap=60)
    guard.before_attempt(10)
    assert ft.sleeps == []
    ft.now += 5.0
    guard.before_attempt(10)
    assert ft.sleeps == []


def test_third_attempt_raises_with_max_calls_two() -> None:
    ft = FakeTime()
    guard = make_guard(ft, max_calls=2)
    guard.before_attempt(10)
    guard.before_attempt(10)
    n_sleeps = len(ft.sleeps)
    with pytest.raises(QuotaExceeded):
        guard.before_attempt(10)
    assert guard.calls_done == 2
    assert len(ft.sleeps) == n_sleeps  # it raised before pacing


def test_zero_max_calls_never_allows_an_attempt() -> None:
    with pytest.raises(QuotaExceeded):
        make_guard(FakeTime(), max_calls=0).before_attempt(1)


def test_token_cap_raises_before_crossing_using_initial_estimate() -> None:
    ft = FakeTime()
    guard = make_guard(ft, max_tokens_total=2_500)
    with pytest.raises(QuotaExceeded):  # 0 + 2000 (estimate) + 1000 > 2500
        guard.before_attempt(1000)
    assert guard.calls_done == 0


def test_token_cap_uses_observed_usage_and_never_crosses() -> None:
    ft = FakeTime()
    guard = make_guard(ft, max_tokens_total=5_000)
    guard.before_attempt(1000)  # 0 + 2000 + 1000 <= 5000
    guard.record(1500, 800)
    guard.before_attempt(1000)  # 2300 + 1500 + 1000 = 4800 <= 5000
    guard.record(1500, 900)
    assert guard.tokens_done == 4700
    with pytest.raises(QuotaExceeded):  # 4700 + 1500 + 1000 > 5000
        guard.before_attempt(1000)
    assert guard.tokens_done <= 5_000
    assert guard.calls_done == 2


def test_record_counts_none_as_zero() -> None:
    guard = make_guard(FakeTime())
    guard.record(None, None)
    assert guard.tokens_done == 0
    guard.record(10, None)
    assert guard.tokens_done == 10


@pytest.mark.parametrize(
    "bad",
    [{"rpm_cap": 0}, {"rpm_cap": -1}, {"max_calls": -1}, {"max_tokens_total": -1}],
)
def test_invalid_caps_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        make_guard(FakeTime(), **bad)


def entry(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ts_utc": "2026-10-27T00:00:00Z",
        "run_id": "r1",
        "model": "m",
        "attempt": 1,
        "status": "OK",
        "source_id": 4295806720338829312,
    }
    base.update(over)
    return base


def test_ledger_appends_one_json_line_per_entry(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append(entry())
    ledger.append(entry(attempt=2, status="RATE_LIMITED", http_status=429))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["attempt"] for ln in lines] == [1, 2]
    assert json.loads(lines[0])["source_id"] == 4295806720338829312


@pytest.mark.parametrize(
    "key", ["authorization", "headers", "request_body", "content", "error_text", "key"]
)
def test_ledger_rejects_key_outside_allowlist(tmp_path: Path, key: str) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append(entry())
    before = path.read_bytes()
    with pytest.raises(ValueError, match="allowlist"):
        ledger.append(entry(**{key: "Bearer FAKE-KEY-123"}))
    assert path.read_bytes() == before  # nothing partial was written
