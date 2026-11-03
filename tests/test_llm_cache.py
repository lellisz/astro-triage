"""Proofs for ``llm.cache``: append-only JSONL, crash recovery and the single-writer lock."""

import logging
from pathlib import Path

import pytest

from astro_triage.llm.cache import TERMINAL, CacheLockedError, ResultCache, cache_key


def rec(i: int, status: str = "OK") -> dict[str, object]:
    return {"source_id": 4295806720338829312 + i, "status": status, "n": i}


def fill3(path: Path) -> None:
    with ResultCache(path) as cache:
        for i in (1, 2, 3):
            cache.put(f"k{i}", rec(i))


def test_roundtrip_and_big_source_id_stays_int(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    fill3(path)
    with ResultCache(path) as cache:
        got = cache.get("k2")
        assert got is not None
        assert got["source_id"] == 4295806720338829312 + 2
        assert isinstance(got["source_id"], int)
        assert cache.get("absent") is None
        assert cache.is_terminal("k1")
        assert not cache.is_terminal("absent")


def test_truncated_last_line_is_dropped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "c.jsonl"
    fill3(path)
    raw = path.read_bytes()
    path.write_bytes(raw[:-12])  # cut in the middle of the third line
    with caplog.at_level(logging.WARNING):
        cache = ResultCache(path)
    try:
        assert cache.get("k1") is not None
        assert cache.get("k2") is not None
        assert cache.get("k3") is None
    finally:
        cache.close()
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_put_after_file_without_final_newline_does_not_merge_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "c.jsonl"
    fill3(path)
    path.write_bytes(path.read_bytes()[:-12])  # torn third line, no "\n"
    with ResultCache(path) as cache:
        cache.put("k4", rec(4))
    with ResultCache(path) as cache:
        assert cache.get("k1") is not None
        assert cache.get("k2") is not None
        assert cache.get("k4") == {"key": "k4", **rec(4)}
    lines = path.read_bytes().splitlines()
    assert lines[-1].startswith(b'{"key":"k4"')  # own line, not glued to the debris


def test_put_after_complete_record_missing_only_the_newline(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    fill3(path)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with ResultCache(path) as cache:
        cache.put("k4", rec(4))
    with ResultCache(path) as cache:
        assert all(cache.get(f"k{i}") is not None for i in (1, 2, 3, 4))


def test_damaged_line_in_the_middle_is_ignored_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "c.jsonl"
    with ResultCache(path) as cache:
        cache.put("k1", rec(1))
    with path.open("ab") as fh:
        fh.write(b'{"key":"kx","sta\xff\xfe garbage\n[1,2]\n{"no_key":1}\n')
    with ResultCache(path) as cache:
        cache.put("k3", rec(3))
    with caplog.at_level(logging.WARNING), ResultCache(path) as cache:
        assert cache.get("k1") is not None
        assert cache.get("k3") is not None
        assert cache.get("kx") is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) >= 3


def test_last_record_for_a_key_wins_and_retryable_is_not_terminal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "c.jsonl"
    with ResultCache(path) as cache:
        cache.put("a", rec(1, "RATE_LIMITED"))
        assert not cache.is_terminal("a")
        assert cache.get("a") is not None
        cache.put("a", rec(1, "OK"))
        assert cache.is_terminal("a")
    with ResultCache(path) as cache:
        assert cache.is_terminal("a")


@pytest.mark.parametrize(
    "status",
    [
        "RATE_LIMITED",
        "SERVER_ERROR",
        "TIMEOUT",
        "NETWORK_ERROR",
        "AUTH_ERROR",
        "BAD_REQUEST",
        "MODEL_UNAVAILABLE",
        "",
    ],
)
def test_non_terminal_statuses(tmp_path: Path, status: str) -> None:
    with ResultCache(tmp_path / "c.jsonl") as cache:
        cache.put("a", rec(1, status))
        assert not cache.is_terminal("a")


@pytest.mark.parametrize("status", sorted(TERMINAL))
def test_terminal_statuses(tmp_path: Path, status: str) -> None:
    with ResultCache(tmp_path / "c.jsonl") as cache:
        cache.put("a", rec(1, status))
        assert cache.is_terminal("a")


def test_second_writer_fails_and_lock_is_released_on_close(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    first = ResultCache(path)
    with pytest.raises(CacheLockedError):
        ResultCache(path)
    assert (tmp_path / "c.jsonl.lock").exists()  # the failed attempt did not steal it
    first.close()
    first.close()  # idempotent
    assert not (tmp_path / "c.jsonl.lock").exists()
    ResultCache(path).close()


def test_lock_is_released_when_the_body_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    with pytest.raises(RuntimeError, match="boom"), ResultCache(path):
        raise RuntimeError("boom")
    assert not (tmp_path / "c.jsonl.lock").exists()
    ResultCache(path).close()


def test_put_after_close_raises(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "c.jsonl")
    cache.close()
    with pytest.raises(ValueError):
        cache.put("a", rec(1))


def test_stale_lock_is_not_stolen_automatically(tmp_path: Path) -> None:
    (tmp_path / "c.jsonl.lock").write_bytes(b"")
    with pytest.raises(CacheLockedError, match="c.jsonl.lock"):
        ResultCache(tmp_path / "c.jsonl")
    assert (tmp_path / "c.jsonl.lock").exists()


def test_existing_lines_are_never_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    fill3(path)
    before = path.read_bytes()
    with ResultCache(path) as cache:
        cache.put("k4", rec(4))
    assert path.read_bytes().startswith(before)


def test_cache_key_depends_on_every_field() -> None:
    base = ("m", "p1", "blind", False, 512, "ab" * 32, 7)
    k0 = cache_key(*base)
    assert k0 == cache_key(*base)
    assert len(k0) == 64
    variants = [
        ("m2", "p1", "blind", False, 512, "ab" * 32, 7),
        ("m", "p2", "blind", False, 512, "ab" * 32, 7),
        ("m", "p1", "with_l1", False, 512, "ab" * 32, 7),
        ("m", "p1", "blind", True, 512, "ab" * 32, 7),
        ("m", "p1", "blind", False, 513, "ab" * 32, 7),
        ("m", "p1", "blind", False, 512, "cd" * 32, 7),
        ("m", "p1", "blind", False, 512, "ab" * 32, 8),
    ]
    assert len({k0, *(cache_key(*v) for v in variants)}) == 8
