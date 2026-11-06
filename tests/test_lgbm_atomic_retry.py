"""`lgbm._atomic`: Windows-only retry of PermissionError on os.replace."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from astro_triage.models import lgbm

REAL_REPLACE = os.replace


def _setup(monkeypatch: pytest.MonkeyPatch, platform: str, fail_times: int | None):
    calls: list[int] = []
    sleeps: list[float] = []

    def fake(src, dst):
        calls.append(1)
        if fail_times is None or len(calls) <= fail_times:
            raise PermissionError("locked")
        REAL_REPLACE(src, dst)

    monkeypatch.setattr(lgbm.sys, "platform", platform)
    monkeypatch.setattr(lgbm.os, "replace", fake)
    monkeypatch.setattr(lgbm.time, "sleep", sleeps.append)
    return calls, sleeps


def _write(tmp: str) -> None:
    Path(tmp).write_bytes(b"new")


def test_retries_then_succeeds_on_windows(tmp_path: Path, monkeypatch) -> None:
    calls, sleeps = _setup(monkeypatch, "win32", 2)
    target = tmp_path / "f.bin"
    lgbm._atomic(target, _write)
    assert target.read_bytes() == b"new" and len(calls) == 3
    assert sleeps == sorted(sleeps) and len(sleeps) == 2 and sleeps[0] >= 0.05
    assert not list(tmp_path.glob("*.tmp"))


def test_always_failing_reraises_and_cleans_tmp(tmp_path: Path, monkeypatch) -> None:
    calls, sleeps = _setup(monkeypatch, "win32", None)
    with pytest.raises(PermissionError):
        lgbm._atomic(tmp_path / "f.bin", _write)
    assert len(calls) == 5 and len(sleeps) == 4 and max(sleeps) <= 0.4 + 1e-9
    assert not list(tmp_path.glob("*.tmp"))


def test_no_retry_off_windows(tmp_path: Path, monkeypatch) -> None:
    calls, sleeps = _setup(monkeypatch, "linux", None)
    with pytest.raises(PermissionError):
        lgbm._atomic(tmp_path / "f.bin", _write)
    assert len(calls) == 1 and not sleeps
