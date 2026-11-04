"""scripts/compute_delta.py: guards, atomic write and exit codes (ADR 0006 section 5)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from test_dl_compare import make_release, random_data

from astro_triage.evaluation.runner import EvalError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compute_delta as mod

GOOD_DOC = {"ok": 1.0}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake repo root: data/ lives in tmp_path, the real data/ is never touched."""
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    out = tmp_path / "data" / "rel"
    (out / mod.SEEDS_DIRNAME).mkdir(parents=True)
    return out


def leftovers(out: Path) -> list[str]:
    return sorted(p.name for p in (out / mod.SEEDS_DIRNAME).iterdir())


def test_full_run_writes_valid_json_and_no_tmp(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    pix, cls, scores = random_data(per_band=25)
    out, sd, _ = make_release(tmp_path / "data" / "rel", pix, cls, scores)
    assert mod.main(["--outdir", str(out)]) == 0
    target = sd / "delta.json"
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["fold"] == "val" and "prediction_sha256" in doc
    assert not [p for p in sd.iterdir() if p.name.endswith(".tmp")]
    assert f"wrote {target}" in capsys.readouterr().out


def test_refuses_existing_delta_json(repo, monkeypatch, capsys):
    target = repo / mod.SEEDS_DIRNAME / mod.DELTA_NAME
    target.write_text("KEEP", encoding="utf-8")
    monkeypatch.setattr(
        mod, "compute_delta", lambda *a: pytest.fail("must not compute")
    )
    assert mod.main(["--outdir", str(repo)]) == 1
    assert target.read_text(encoding="utf-8") == "KEEP"
    assert "refusing to overwrite" in capsys.readouterr().err


def test_refuses_outdir_outside_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / mod.SEEDS_DIRNAME).mkdir(parents=True)
    monkeypatch.setattr(
        mod, "compute_delta", lambda *a: pytest.fail("must not compute")
    )
    assert mod.main(["--outdir", str(elsewhere)]) == 1
    assert "must be inside data/" in capsys.readouterr().err
    assert list((elsewhere / mod.SEEDS_DIRNAME).iterdir()) == []
    # traversal that resolves outside data/
    assert mod.main(["--outdir", str(tmp_path / "data" / ".." / "elsewhere")]) == 1


@pytest.mark.parametrize(
    "exc",
    [EvalError("boom"), OSError("disk"), ValueError("bad")],
    ids=lambda e: type(e).__name__,
)
def test_exit_1_on_expected_errors_and_nothing_written(repo, monkeypatch, capsys, exc):
    def raiser(*a):
        raise exc

    monkeypatch.setattr(mod, "compute_delta", raiser)
    assert mod.main(["--outdir", str(repo)]) == 1
    assert type(exc).__name__ in capsys.readouterr().err
    assert leftovers(repo) == []


def test_nan_in_doc_is_refused_by_allow_nan_false(repo, monkeypatch):
    monkeypatch.setattr(mod, "compute_delta", lambda *a: {"x": float("nan")})
    assert mod.main(["--outdir", str(repo)]) == 1
    assert leftovers(repo) == []


def test_write_is_atomic_failure_in_link_leaves_no_target_no_tmp(repo, monkeypatch):
    monkeypatch.setattr(mod, "compute_delta", lambda *a: GOOD_DOC)

    def boom(*a, **k):
        raise OSError("link failed")

    monkeypatch.setattr(os, "link", boom)
    assert mod.main(["--outdir", str(repo)]) == 1
    assert leftovers(repo) == []


def test_write_success_creates_complete_json_only(repo, monkeypatch):
    monkeypatch.setattr(mod, "compute_delta", lambda *a: GOOD_DOC)
    assert mod.main(["--outdir", str(repo)]) == 0
    assert leftovers(repo) == [mod.DELTA_NAME]
    assert (
        json.loads((repo / mod.SEEDS_DIRNAME / mod.DELTA_NAME).read_text("utf-8"))
        == GOOD_DOC
    )


def test_missing_predictions_exit_1_without_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    pix, cls, scores = random_data(per_band=25)
    out, sd, _ = make_release(tmp_path / "data" / "rel", pix, cls, scores)
    (sd / "predictions_lgbm_noind_s4_v0.parquet").unlink()
    assert mod.main(["--outdir", str(out)]) == 1
    assert not (sd / mod.DELTA_NAME).exists()
