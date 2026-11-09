"""train_dl guards, second batch: freeze/subset conflict, ensemble without folds, SHA skips, discard paths, JSON I/O."""

from __future__ import annotations

import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dl_refs import FRACS, write_lgbm_ref
from test_train_dl_guards import PRED, Fake, fake, repo, run  # noqa: F401
from test_train_dl_modes import ens_setup, rel

from astro_triage.evaluation.runner import EvalError
from astro_triage.models import dl_session
from astro_triage.models.lgbm import TrainError

mod = sys.modules["train_dl"]


@pytest.mark.parametrize("flag", ["--train-subset", "--plan-c-subsample"])
def test_freeze_with_subset_flag_refused_before_any_work(repo, flag, monkeypatch, capsys):  # noqa: F811
    calls = []
    monkeypatch.setattr(mod, "run_freeze", lambda *a: calls.append(a))
    monkeypatch.setattr(mod, "load_folds", lambda _o: calls.append("folds"))
    before = sorted(p.name for p in rel(repo).iterdir())
    assert mod.main(["--freeze", "--approval-text", "ok", flag, "5", "--outdir", str(rel(repo))]) == 1
    err = capsys.readouterr().err
    assert "not allowed with subset flags" in err and len(err.strip().splitlines()) == 1
    assert calls == [] and sorted(p.name for p in rel(repo).iterdir()) == before


def test_ensemble_never_loads_folds(repo, fake, monkeypatch):  # noqa: F811
    ens_setup(repo, monkeypatch)
    loads = []
    monkeypatch.setattr(mod, "load_folds", lambda o: loads.append(o) or pytest.fail("load_folds called"))
    assert mod.main(["--ensemble", "--seeds", "0", "--outdir", str(rel(repo))]) == 0
    assert loads == [] and fake.calls == []


def test_ensemble_skips_by_sha_not_by_report_presence(repo, fake, monkeypatch, capsys):  # noqa: F811
    dl, built = ens_setup(repo, monkeypatch)
    argv = ["--ensemble", "--arch", "mlp", "--seeds", "0", "--outdir", str(rel(repo))]
    pred = dl / "predictions_ens_mlp_full_s0_v0.parquet"
    assert mod.main(argv) == 0 and len(built) == 1
    pred.write_bytes(b"another sha")  # the report still holds the old sha for this id
    capsys.readouterr()
    assert mod.main(argv) == 0 and len(built) == 2
    assert "skipped" not in capsys.readouterr().err
    pred.unlink()  # report entry without a file
    assert mod.main(argv) == 0 and len(built) == 3
    assert mod.main(argv) == 0 and len(built) == 3  # unchanged file: skipped
    assert "already built, skipped" in capsys.readouterr().err


def test_training_skips_by_sha_not_by_report_presence(repo, fake, capsys):  # noqa: F811
    dl = rel(repo) / "dl"
    assert run(repo) == 0 and len(fake.calls) == 1
    (dl / PRED).write_bytes(b"q")  # file no longer the one recorded in the report
    assert run(repo) == 0 and len(fake.calls) == 2
    capsys.readouterr()
    assert run(repo) == 0 and len(fake.calls) == 2
    assert "already trained, skipped" in capsys.readouterr().err


def test_ensemble_purity_failure_deletes_parquet_and_manifest(repo, fake, monkeypatch, capsys):  # noqa: F811
    dl, _ = ens_setup(repo, monkeypatch)
    pred = dl / "predictions_ens_mlp_full_s0_v0.parquet"
    manifest = pred.with_name(f"{pred.stem}.manifest.json")

    def build(outdir, dl_dir, lgbm_dir, net, k):
        pred.write_bytes(b"e")
        manifest.write_text("{}")
        return {"model_id": f"ens_{net}_full_s{k}", "weight": 0.5}

    def boom(o, p):
        raise EvalError("val labels broken")

    monkeypatch.setattr(dl_session, "build_ensemble", build)
    monkeypatch.setattr(dl_session, "val_purity_summary", boom)
    assert mod.main(["--ensemble", "--arch", "mlp", "--seeds", "0", "--outdir", str(rel(repo))]) == 1
    assert "val purity failed" in capsys.readouterr().err
    assert not pred.exists() and not manifest.exists()
    assert "ens_mlp_full_s0" not in dl_session.fits_on_disk(dl / mod.REPORT_NAME)


def test_trained_model_purity_failure_deletes_all_five_outputs(repo, monkeypatch, capsys):  # noqa: F811
    class Writes(Fake):
        def train_one(self, spec, folds, outdir, ckpt_dir, **kw):
            res = super().train_one(spec, folds, outdir, ckpt_dir, **kw)
            (outdir / "models").mkdir(exist_ok=True)
            for f in mod._output_files(outdir, spec.model_id)[1:]:
                f.write_bytes(b"x")
            return res

    f = Writes()
    monkeypatch.setitem(sys.modules, "astro_triage.models.dl_train", f)
    import astro_triage.models as pkg

    monkeypatch.setattr(pkg, "dl_train", f, raising=False)

    def boom(o, p):
        raise EvalError("no val ids")

    monkeypatch.setattr(dl_session, "val_purity_summary", boom)
    assert run(repo) == 1
    err = capsys.readouterr().err
    assert "val purity failed" in err and "no val ids" in err
    dl = rel(repo) / "dl"
    outputs = mod._output_files(dl, "mlp_full_s0")
    assert len(outputs) == 5 and not any(p.exists() for p in outputs)
    assert "mlp_full_s0" not in dl_session.fits_on_disk(dl / mod.REPORT_NAME)


def test_purity_or_discard_lists_files_it_could_not_delete(tmp_path, monkeypatch):
    a = tmp_path / "a"
    monkeypatch.setattr(dl_session, "val_purity_summary", lambda o, p: (_ for _ in ()).throw(EvalError("bad")))
    monkeypatch.setattr(dl_session, "unlink_all", lambda paths: list(paths))
    with pytest.raises(TrainError, match="val purity failed, outputs discarded: bad; could not delete: .*a"):
        dl_session.purity_or_discard(tmp_path, a, "m", [a])


def ref_with(tmp_path, raw):
    path = tmp_path / "ref.parquet"
    pq.write_table(pa.table({"a": [1]}).replace_schema_metadata({"train_class_fractions": raw}), path)
    return path


@pytest.mark.parametrize("raw", ["null", '{"STAR": null, "GALAXY": 0.3, "QSO": 0.2}', "[1]", '{"STAR": "x", "GALAXY": 1, "QSO": 1}'])
def test_check_pi_src_null_or_bad_reference_metadata_deletes_outputs(tmp_path, raw):
    out = tmp_path / "out.bin"
    out.write_bytes(b"x")
    with pytest.raises(TrainError, match="cannot compare"):
        dl_session.check_pi_src(dict(FRACS), ref_with(tmp_path, raw), "m", [out])
    assert not out.exists()


@pytest.mark.parametrize("fractions", [None, ["STAR"], {"STAR": None, "GALAXY": 0.3, "QSO": 0.2}, {"STAR": 0.5}])
def test_check_pi_src_bad_dl_fractions_deletes_outputs(tmp_path, fractions):
    ref = write_lgbm_ref(tmp_path, "full", 0, FRACS)
    out = tmp_path / "out.bin"
    out.write_bytes(b"x")
    with pytest.raises(TrainError, match="cannot compare"):
        dl_session.check_pi_src(fractions, ref, "m", [out])
    assert not out.exists()


@pytest.mark.parametrize(("d", "ok"), [(5e-10, True), (-5e-10, True), (2e-9, False), (-2e-9, False)])
def test_check_pi_src_fraction_tolerance(tmp_path, d, ok):
    ref = write_lgbm_ref(tmp_path, "full", 0, FRACS)
    out = tmp_path / "out.bin"
    out.write_bytes(b"x")
    got = {**FRACS, "GALAXY": FRACS["GALAXY"] + d}
    if ok:
        dl_session.check_pi_src(got, ref, "m", [out])
        assert out.exists()
    else:
        with pytest.raises(TrainError, match="differ"):
            dl_session.check_pi_src(got, ref, "m", [out])
        assert not out.exists()


def test_run_with_null_reference_fractions_exits_1_not_traceback(repo, fake, capsys):  # noqa: F811
    write_lgbm_ref(rel(repo), "full", 0, {"STAR": None, "GALAXY": None, "QSO": None})
    assert run(repo) == 1
    assert "cannot compare" in capsys.readouterr().err
    assert not (rel(repo) / "dl" / PRED).exists()


@pytest.mark.parametrize("doc", [{"a": {"b": {"Test_x": 1}}}, {"a": [{"TEST": 1}]}, {"my_test": 1}])
def test_write_json_refuses_test_keys_at_any_level(tmp_path, doc):
    with pytest.raises(TrainError, match="test-derived key"):
        dl_session.write_json(tmp_path / "r.json", doc)
    assert list(tmp_path.iterdir()) == []


def test_write_json_allows_test_in_values(tmp_path):
    dl_session.write_json(tmp_path / "r.json", {"note": "latest test", "l": ["test"]})
    assert json.loads((tmp_path / "r.json").read_text())["note"] == "latest test"


def test_run_freeze_malformed_delta_is_one_line_train_error(tmp_path, monkeypatch):
    from test_dl_val_freeze import make_env

    out, dl, lg = make_env(tmp_path)
    (lg / "delta.json").write_text('{"prediction_sha256": ["a"]}')
    with pytest.raises(TrainError, match="freeze refused") as ei:
        dl_session.run_freeze(out, dl, lg, "approved")
    assert "\n" not in str(ei.value) and not (dl / "val_freeze.json").exists()
