"""train_dl.py --freeze / --ensemble / mode flags and the pi_src check; fakes only, no torch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dl_refs import FRACS, write_lgbm_ref
from test_train_dl_guards import PRED, fake, repo, run  # noqa: F401

from astro_triage.evaluation.runner import EvalError
from astro_triage.models import dl_session
from astro_triage.models.lgbm import TrainError

mod = sys.modules["train_dl"]
SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SEED0 = "predictions_lgbm_full_s0_v0.parquet"


def rel(repo: Path) -> Path:  # noqa: F811
    return repo / "data" / "rel"
def freeze(repo: Path, *extra: str) -> int:  # noqa: F811
    return mod.main(["--freeze", "--outdir", str(rel(repo)), *extra])


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def build(*a):
        calls.append(a)
        return {"approval_text": a[3], "sha": "x"}

    monkeypatch.setattr(dl_session, "build_freeze", build)
    return calls


@pytest.mark.parametrize("extra", [[], ["--approval-text", "  "]])
def test_freeze_without_approval_exits_1_before_build(repo, spy, extra, capsys) -> None:  # noqa: F811
    assert freeze(repo, *extra) == 1
    assert "approval" in capsys.readouterr().err and spy == []
    assert not (rel(repo) / "dl" / "val_freeze.json").exists()


def test_freeze_writes_once_and_existing_target_exits_before_build(repo, spy, capsys) -> None:  # noqa: F811
    dl = rel(repo) / "dl"
    dl.mkdir()
    assert freeze(repo, "--approval-text", "ok") == 0
    assert json.loads((dl / "val_freeze.json").read_text())["approval_text"] == "ok"
    before = (dl / "val_freeze.json").read_bytes()
    assert freeze(repo, "--approval-text", "again") == 1
    assert "refusing to overwrite" in capsys.readouterr().err and len(spy) == 1
    assert (dl / "val_freeze.json").read_bytes() == before
    assert [p.name for p in dl.iterdir()] == ["val_freeze.json"]


def test_write_once_is_atomic_link_refuses_existing_and_rejects_nan(tmp_path, monkeypatch) -> None:
    links = []
    real = os.link
    monkeypatch.setattr(dl_session.os, "link", lambda s, d: (links.append((s, d)), real(s, d))[1])
    target = tmp_path / "v.json"
    dl_session.write_once(target, {"a": 1})
    assert len(links) == 1 and Path(links[0][0]) != target
    with pytest.raises(TrainError, match="refusing to overwrite"):
        dl_session.write_once(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}
    with pytest.raises(ValueError):
        dl_session.write_once(tmp_path / "n.json", {"a": float("nan")})
    assert [p.name for p in tmp_path.iterdir()] == ["v.json"]


def test_build_freeze_eval_error_becomes_freeze_refused(repo, monkeypatch, capsys) -> None:  # noqa: F811
    def boom(*_a):
        raise EvalError("delta.json missing")

    monkeypatch.setattr(dl_session, "build_freeze", boom)
    (rel(repo) / "dl").mkdir()
    assert freeze(repo, "--approval-text", "ok") == 1
    assert "freeze refused: delta.json missing" in capsys.readouterr().err
    assert not (rel(repo) / "dl" / "val_freeze.json").exists()


def test_freeze_does_not_import_torch_or_dl_train(tmp_path) -> None:
    root = str(tmp_path)
    code = (
        "import sys, pathlib\n"
        f"sys.path[:0] = [{str(SRC)!r}, {str(SCRIPTS)!r}]\n"
        "import train_dl as m\n"
        "from astro_triage.models import dl_session as s\n"
        f"m.REPO_ROOT = pathlib.Path({root!r})\n"
        "s.build_freeze = lambda *a: {'a': 1}\n"
        f"d = pathlib.Path({root!r}) / 'data' / 'rel' / 'dl'; d.mkdir(parents=True)\n"
        "rc = m.main(['--freeze', '--outdir', str(d.parent), '--approval-text', 'ok'])\n"
        "print(rc, 'torch' in sys.modules, 'astro_triage.models.dl_train' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split()[-3:] == ["0", "False", "False"], out.stderr


@pytest.mark.parametrize("extra", [["--freeze", "--approval-text", "ok"], ["--ensemble"]])
def test_lgbm_seeds_outside_data_exits_1(repo, spy, monkeypatch, extra, capsys) -> None:  # noqa: F811
    monkeypatch.setattr(mod, "SEEDS_DIRNAME", "../../outside")
    assert mod.main(["--outdir", str(rel(repo)), *extra]) == 1
    assert "data/" in capsys.readouterr().err and spy == []


def ens_setup(repo, monkeypatch):  # noqa: F811
    dl = rel(repo) / "dl"
    dl.mkdir()
    (dl / "environment.json").write_text(json.dumps({"local": {"torch": "9.9", "cuda": None, "gpu": "cpu"}}))
    built: list = []

    def build(outdir, dl_dir, lgbm_dir, net, k):
        built.append((net, k))
        (dl_dir / f"predictions_ens_{net}_full_s{k}_v0.parquet").write_bytes(f"{net}{k}".encode())
        return {"model_id": f"ens_{net}_full_s{k}", "weight": 0.5}

    monkeypatch.setattr(dl_session, "build_ensemble", build)
    monkeypatch.setattr(dl_session, "val_purity_summary", lambda o, p: {"n_val": 3, "purity": 0.9})
    return dl, built


def test_ensemble_records_entries_skips_built_and_respects_arch(repo, fake, monkeypatch, capsys) -> None:  # noqa: F811
    dl, built = ens_setup(repo, monkeypatch)
    out = ["--ensemble", "--seeds", "0-1", "--outdir", str(rel(repo))]
    assert mod.main(out) == 0
    assert sorted(built) == [("ftt", 0), ("ftt", 1), ("mlp", 0), ("mlp", 1)]
    text = (dl / mod.REPORT_NAME).read_text()
    e = json.loads(text)["fits"]["ens_mlp_full_s1"]
    assert (e["platform"], e["torch_version"], e["val_purity"]) == ("local", "9.9", {"n_val": 3, "purity": 0.9})
    assert len(e["prediction_sha256"]) == 64 and "test" not in text.lower()
    assert mod.main(out) == 0 and len(built) == 4
    assert "already built, skipped" in capsys.readouterr().err
    (dl / mod.REPORT_NAME).unlink()
    built.clear()
    assert mod.main([*out[:1], "--arch", "ftt", "--seeds", "1", *out[3:]]) == 0 and built == [("ftt", 1)]


def test_mode_flag_conflicts(repo, fake, capsys) -> None:  # noqa: F811
    base = ["--outdir", str(rel(repo))]
    assert mod.main([*base, "--ensemble", "--freeze", "--approval-text", "x"]) == 1
    assert "mutually exclusive" in capsys.readouterr().err
    assert mod.main(base) == 1
    err = capsys.readouterr().err
    assert "--arch is required" in err and len(err.strip().splitlines()) == 1


def test_check_pi_src_match_ok_and_divergence_deletes_outputs(tmp_path) -> None:
    ref = write_lgbm_ref(tmp_path, "full", 0, FRACS)
    out = tmp_path / "out.bin"
    out.write_bytes(b"x")
    dl_session.check_pi_src(dict(FRACS), ref, "m", [out])
    assert out.exists()
    with pytest.raises(TrainError, match="differ"):
        dl_session.check_pi_src({**FRACS, "STAR": 0.5 + 1e-6, "QSO": 0.2 - 1e-6}, ref, "m", [out])
    assert not out.exists()


@pytest.mark.parametrize("kind", ["missing", "garbage", "nokey"])
def test_check_pi_src_unreadable_reference_deletes_outputs(tmp_path, kind) -> None:
    ref = tmp_path / "ref.parquet"
    if kind == "garbage":
        ref.write_bytes(b"not parquet")
    if kind == "nokey":
        pq.write_table(pa.table({"a": [1]}), ref)
    out = tmp_path / "out.bin"
    out.write_bytes(b"x")
    with pytest.raises(TrainError, match="cannot compare"):
        dl_session.check_pi_src(dict(FRACS), ref, "m", [out])
    assert not out.exists()


def test_run_divergent_or_missing_reference_discards_prediction(repo, fake, capsys) -> None:  # noqa: F811
    write_lgbm_ref(rel(repo), "full", 0, {"STAR": 0.4, "GALAXY": 0.4, "QSO": 0.2})
    assert run(repo) == 1 and "differ" in capsys.readouterr().err
    assert not (rel(repo) / "dl" / PRED).exists()
    (rel(repo) / "lgbm_seeds" / SEED0).unlink()
    assert run(repo) == 1 and "cannot compare" in capsys.readouterr().err
    assert not (rel(repo) / "dl" / PRED).exists()


@pytest.mark.parametrize("flag", ["--train-subset", "--plan-c-subsample"])
def test_pi_src_not_checked_with_subset(repo, fake, flag) -> None:  # noqa: F811
    (rel(repo) / "lgbm_seeds" / SEED0).unlink()
    assert run(repo, flag, "5") == 0


def test_t2_failure_is_reported_before_pi_src_check(repo, fake, capsys) -> None:  # noqa: F811
    fake.beats = False
    (rel(repo) / "lgbm_seeds" / SEED0).unlink()
    assert run(repo) == 1 and "T2" in capsys.readouterr().err
    fits = json.loads((rel(repo) / "dl" / mod.REPORT_NAME).read_text())["fits"]
    assert fits["mlp_full_s0"]["status"] == "failed_T2_beats_prior"
