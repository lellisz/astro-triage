"""open_test_set guard order and write_marker (spec S11, ADR 0006 section 5.2): no test read before the gates."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from dl_eval_kit import frozen_env

from astro_triage.evaluation import dl_eval as E
from astro_triage.evaluation.runner import EvalError

SENTINEL = object()


@pytest.fixture
def spy(monkeypatch):
    """load_test_set replaced by a recorder; _TEST_OPENED reset and restored after the test."""
    calls: list = []
    monkeypatch.setattr(E, "load_test_set", lambda out: calls.append(out) or SENTINEL)
    monkeypatch.setitem(E._TEST_OPENED, "done", False)
    return calls


def test_success_returns_loader_result_and_freeze_and_locks(tmp_path, spy):
    out, dl, lg, doc = frozen_env(tmp_path)
    ts, freeze = E.open_test_set(out, dl, lg, " approved by owner ")
    assert ts is SENTINEL and freeze == doc and spy == [out]
    assert E._TEST_OPENED["done"] is True


def test_second_open_in_same_process_refused(tmp_path, spy):
    out, dl, lg, _ = frozen_env(tmp_path)
    E.open_test_set(out, dl, lg, "ok")
    with pytest.raises(EvalError, match="already opened in this process"):
        E.open_test_set(out, dl, lg, "ok")
    assert len(spy) == 1


@pytest.mark.parametrize("approval", ["", " ", "\n\t "])
def test_empty_approval_refused_first(tmp_path, spy, approval):
    out, dl, lg, _ = frozen_env(tmp_path)
    (dl / E.MARKER_NAME).write_text("{}")  # would trip the marker guard
    (dl / E.FREEZE_NAME).unlink()  # would trip verify_freeze
    E._TEST_OPENED["done"] = True  # would trip the process lock
    with pytest.raises(EvalError, match="approval"):
        E.open_test_set(out, dl, lg, approval)
    assert spy == []


def test_marker_checked_before_verify_and_lock(tmp_path, spy):
    out, dl, lg, _ = frozen_env(tmp_path)
    (dl / E.MARKER_NAME).write_text("{}")
    (dl / E.FREEZE_NAME).unlink()
    E._TEST_OPENED["done"] = True
    with pytest.raises(EvalError, match="test already opened"):
        E.open_test_set(out, dl, lg, "ok")
    assert spy == []


def test_verify_checked_before_process_lock(tmp_path, spy):
    out, dl, lg, _ = frozen_env(tmp_path)
    (dl / E.FREEZE_NAME).unlink()
    E._TEST_OPENED["done"] = True
    with pytest.raises(EvalError, match="val_freeze.json missing"):
        E.open_test_set(out, dl, lg, "ok")
    assert spy == []


def test_refusals_do_not_burn_the_process_lock(tmp_path, spy):
    out, dl, lg, _ = frozen_env(tmp_path)
    with pytest.raises(EvalError):
        E.open_test_set(out, dl, lg, " ")
    (dl / E.MARKER_NAME).write_text("{}")
    with pytest.raises(EvalError):
        E.open_test_set(out, dl, lg, "ok")
    (dl / E.MARKER_NAME).unlink()
    labels = out / "labels.parquet"
    labels.write_bytes(labels.read_bytes() + b"x")
    with pytest.raises(EvalError, match="labels sha256"):
        E.open_test_set(out, dl, lg, "ok")
    assert E._TEST_OPENED["done"] is False and spy == []


def test_tampered_prediction_never_reaches_load_test_set(tmp_path, spy):
    out, dl, lg, _ = frozen_env(tmp_path)
    p = E._paths(dl, lg)["ftt_full_s2"]
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(EvalError, match="sha256 differs"):
        E.open_test_set(out, dl, lg, "ok")
    assert spy == []


def test_pilot_dir_never_opens(tmp_path, spy):
    out, dl, lg, _ = frozen_env(tmp_path / "pilot")
    with pytest.raises(EvalError, match="pilot"):
        E.open_test_set(out, dl, lg, "ok")
    assert spy == []


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from _keys(v)


def test_marker_content_has_no_metrics_nor_test_key(tmp_path):
    path = E.write_marker(tmp_path, freeze_sha="f" * 64, git_sha="abc123")
    assert path == tmp_path / E.MARKER_NAME == tmp_path / "test_opened.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert set(doc) == {
        "schema_name",
        "schema_version",
        "opened_at",
        "git_sha",
        "freeze_sha256",
    }
    assert (
        doc["schema_name"] == E.MARKER_SCHEMA == "astro_triage.dl_test_opened"
    )  # "test" in a VALUE is fine
    assert (
        doc["schema_version"] == 1
        and doc["git_sha"] == "abc123"
        and doc["freeze_sha256"] == "f" * 64
    )
    assert datetime.fromisoformat(doc["opened_at"]).utcoffset().total_seconds() == 0
    assert not any("test" in k.lower() for k in _keys(doc))


def test_marker_refuses_overwrite(tmp_path):
    first = E.write_marker(tmp_path, freeze_sha="a", git_sha="g1")
    before = first.read_bytes()
    with pytest.raises(EvalError, match="test_opened.json already exists"):
        E.write_marker(tmp_path, freeze_sha="b", git_sha="g2")
    assert first.read_bytes() == before


def test_marker_makes_open_test_set_refuse(tmp_path, spy):
    out, dl, lg, _ = frozen_env(tmp_path)
    E.write_marker(dl, freeze_sha="a", git_sha="g")
    with pytest.raises(EvalError, match="test already opened"):
        E.open_test_set(out, dl, lg, "ok")
    assert spy == []


def test_marker_written_through_atomic_temp_file(tmp_path, monkeypatch):
    seen = {}
    real = E._atomic

    def watch(path, write):
        def inner(tmp):
            seen["tmp"], seen["final_existed"] = tmp, path.exists()
            write(tmp)

        real(path, inner)

    monkeypatch.setattr(E, "_atomic", watch)
    path = E.write_marker(tmp_path, freeze_sha="a", git_sha="g")
    assert seen["tmp"] != str(path) and seen["final_existed"] is False
    assert [p.name for p in tmp_path.iterdir()] == [E.MARKER_NAME]


def test_marker_failed_write_leaves_nothing(tmp_path, monkeypatch):
    real = E._atomic

    def failing(path, write):
        def inner(tmp):
            write(tmp)
            raise OSError("disk full")

        real(path, inner)

    monkeypatch.setattr(E, "_atomic", failing)
    with pytest.raises(OSError, match="disk full"):
        E.write_marker(tmp_path, freeze_sha="a", git_sha="g")
    assert list(tmp_path.iterdir()) == []
