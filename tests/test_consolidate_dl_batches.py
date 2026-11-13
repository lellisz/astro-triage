"""scripts/consolidate_dl_batches.py: dry run, idempotence, incremental merge, refusals write nothing."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import consolidate_dl_batches as C

ENV = {"torch": "2.8", "cuda": "12.6", "gpu": "T4", "first_seen": "2026-11-05"}


def batch(root: Path, name: str, ids: list[str], env=ENV, **over) -> Path:
    b = root / name
    (b / "models").mkdir(parents=True)
    fits = {}
    for i in ids:
        data = f"pred-{i}".encode()
        (b / f"predictions_{i}_v0.parquet").write_bytes(data)
        (b / "models" / f"{i}.pt").write_bytes(b"w" + data)
        fits[i] = {"prediction_sha256": hashlib.sha256(data).hexdigest(), "seed": i}
    rep = {"environment": {"kaggle": dict(env)}, "fits": fits}
    rep.update(over)
    (b / "train_report_dl.json").write_text(json.dumps(rep))
    (b / "environment.json").write_text('{"from": "batch"}')
    return b


def tree(p: Path) -> dict[str, bytes]:
    return {x.relative_to(p).as_posix(): x.read_bytes() for x in sorted(p.rglob("*")) if x.is_file()}


def run(capsys, *batches: Path, dest: Path, apply: bool = True) -> tuple[int, str, str]:
    argv = [a for b in batches for a in ("--batch", str(b))] + ["--dest", str(dest)]
    rc = C.main(argv + (["--apply"] if apply else []))
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_dry_run_writes_nothing(tmp_path, capsys):
    b, dest = batch(tmp_path, "a", ["s0"]), tmp_path / "dest"
    rc, out, _ = run(capsys, b, dest=dest, apply=False)
    assert rc == 0 and "dry run" in out and not dest.exists()


def test_apply_empty_dest_copies_and_keeps_environment(tmp_path, capsys):
    b, dest = batch(tmp_path, "a", ["s1", "s0"]), tmp_path / "dest"
    dest.mkdir()
    (dest / "environment.json").write_text('{"from": "dest"}')
    rc, out, _ = run(capsys, b, dest=dest)
    assert rc == 0 and "OK (2 of 2)" in out
    assert (dest / "environment.json").read_text() == '{"from": "dest"}'
    assert (dest / "models" / "s0.pt").read_bytes() == b"wpred-s0"
    rep = json.loads((dest / "train_report_dl.json").read_text())
    assert list(rep["fits"]) == ["s0", "s1"] and rep["environment"]["kaggle"] == ENV
    d2 = tmp_path / "dest2"  # no destination environment.json: none is created
    assert run(capsys, b, dest=d2)[0] == 0 and not (d2 / "environment.json").exists()


def test_reapply_is_idempotent(tmp_path, capsys):
    b, dest = batch(tmp_path, "a", ["s0", "s1"]), tmp_path / "dest"
    run(capsys, b, dest=dest)
    before = tree(dest)
    rc, out, _ = run(capsys, b, dest=dest)
    assert rc == 0 and "copied 0 files" in out
    assert tree(dest) == before


def test_incremental_adds_only_new_batch_and_merges_environment(tmp_path, capsys):
    a = batch(tmp_path, "a", ["s0"])
    b = batch(tmp_path, "b", ["s1"], env={**ENV, "first_seen": "2026-11-02"})
    dest = tmp_path / "dest"
    run(capsys, a, dest=dest)
    rp = dest / "train_report_dl.json"
    rep = json.loads(rp.read_text())
    rp.write_text(json.dumps({**rep, "environment": {**rep["environment"], "local": {"torch": "cpu"}}}))
    rc, out, _ = run(capsys, a, b, dest=dest)
    assert rc == 0 and "copied 2 files" in out
    new = json.loads(rp.read_text())
    assert list(new["fits"]) == ["s0", "s1"]
    assert new["environment"]["local"] == {"torch": "cpu"}
    assert new["environment"]["kaggle"]["first_seen"] == "2026-11-02"


def _bad_conflict(tmp, dest):
    a = batch(tmp, "a", ["s0"])
    (dest / "models").mkdir(parents=True)
    (dest / "models" / "s0.pt").write_bytes(b"different")
    return [a]


def _bad_dup_model(tmp, dest):
    a, b = batch(tmp, "a", ["s0"]), batch(tmp, "b", ["s0"])
    rp = b / "train_report_dl.json"
    rp.write_text(rp.read_text().replace('"seed": "s0"', '"seed": "other"'))
    return [a, b]


def _bad_env(tmp, dest):
    return [batch(tmp, "a", ["s0"]), batch(tmp, "b", ["s1"], env={**ENV, "gpu": "P100"})]


def _bad_sha(tmp, dest):
    a = batch(tmp, "a", ["s0"])
    (a / "predictions_s0_v0.parquet").write_bytes(b"tampered")
    return [a]


def _bad_no_report(tmp, dest):
    a = batch(tmp, "a", ["s0"])
    (a / "train_report_dl.json").unlink()
    return [a]
def _bad_json(tmp, dest):
    a = batch(tmp, "a", ["s0"])
    (a / "train_report_dl.json").write_text("{not json")
    return [a]


def _bad_missing_pred(tmp, dest):
    a = batch(tmp, "a", ["s0", "s1"])
    (a / "predictions_s1_v0.parquet").unlink()
    return [a]


@pytest.mark.parametrize(
    "mk",
    [_bad_conflict, _bad_dup_model, _bad_env, _bad_sha, _bad_no_report, lambda t, d: [t / "nope"], _bad_json, _bad_missing_pred],
)
def test_refusals_exit_2_and_write_nothing(tmp_path, capsys, mk):
    dest = tmp_path / "dest"
    batches = mk(tmp_path, dest)
    snap = lambda: tree(dest) if dest.exists() else None
    before = snap()
    rc, _, err = run(capsys, *batches, dest=dest)
    assert rc == 2 and "refused" in err
    assert snap() == before


def test_io_failure_mid_apply_leaves_no_report_nor_tmp_and_retry_completes(tmp_path, capsys, monkeypatch):
    b, dest = batch(tmp_path, "a", ["s0", "s1"]), tmp_path / "dest"
    real, calls = shutil.copy2, []

    def flaky(*a, **k):
        calls.append(a)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(*a, **k)

    with monkeypatch.context() as mp:
        mp.setattr(C.shutil, "copy2", flaky)
        rc, _, err = run(capsys, b, dest=dest)
    assert rc == 2 and "failed while writing" in err
    assert not (dest / "train_report_dl.json").exists() and not list(dest.rglob("*.tmp"))
    rc, out, _ = run(capsys, b, dest=dest)
    assert rc == 0 and "OK (2 of 2)" in out and (dest / "train_report_dl.json").exists()


def test_corrupted_destination_prediction_is_refused_not_ignored(tmp_path, capsys):
    b, dest = batch(tmp_path, "a", ["s0"]), tmp_path / "dest"
    run(capsys, b, dest=dest)
    (dest / "predictions_s0_v0.parquet").write_bytes(b"corrupt")
    rc, _, err = run(capsys, b, dest=dest)
    assert rc == 2 and "conflict" in err
