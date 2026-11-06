"""scripts/train_dl.py end to end on a tiny synthetic release (CPU, 2 epochs); T2 and directory layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dl_refs import fractions_of, write_lgbm_ref
from test_dl_train_one import make_folds

from astro_triage.models import dl_train, lgbm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_dl as mod

PRED = "predictions_mlp_full_s0_v0.parquet"
FAST = [
    "--arch",
    "mlp",
    "--features",
    "full",
    "--seeds",
    "0",
    "--max-epochs",
    "2",
    "--patience",
    "5",
]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lgbm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_folds", lambda _o: make_folds(labelled_test=True))
    (tmp_path / "data" / "rel").mkdir(parents=True)
    write_lgbm_ref(
        tmp_path / "data" / "rel", "full", 0, fractions_of(make_folds()["train"].labels)
    )
    return tmp_path


def argv(repo: Path, *extra: str) -> list[str]:
    return ["--outdir", str(repo / "data" / "rel"), *FAST, *extra]


def outputs(dl: Path) -> list[Path]:
    return [
        dl / PRED,
        dl / PRED.replace(".parquet", ".manifest.json"),
        dl / "models" / "mlp_full_s0.pt",
        dl / "models" / "model_mlp_full_s0.manifest.json",
        dl / "models" / "mlp_full_s0.preproc.pkl",
    ]


def force_t2_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    real = dl_train.train_one

    def fake(*a, **k):
        return {**real(*a, **k), "beats_prior_logloss": False}

    monkeypatch.setattr(dl_train, "train_one", fake)


def test_full_run_outputs_report_without_test_key_and_idempotent(repo, capsys) -> None:
    dl = repo / "data" / "rel" / "dl"
    assert mod.main(argv(repo)) == 0
    assert json.loads(capsys.readouterr().out) == {"trained": ["mlp_full_s0"]}
    for p in [*outputs(dl), dl / "environment.json", dl / "train_report_dl.json"]:
        assert p.exists(), p
    text = (dl / "train_report_dl.json").read_text(encoding="utf-8")
    assert "test" not in text.lower()
    assert json.loads(text)["fits"]["mlp_full_s0"]["prediction_sha256"]
    assert "local" in json.loads((dl / "environment.json").read_text(encoding="utf-8"))
    before = (dl / PRED).read_bytes()
    assert mod.main(argv(repo)) == 0
    cap = capsys.readouterr()
    assert (
        json.loads(cap.out) == {"trained": []} and "already trained, skipped" in cap.err
    )
    assert (dl / PRED).read_bytes() == before


def test_write_json_refuses_test_key(tmp_path) -> None:
    with pytest.raises(lgbm.TrainError, match="test-derived"):
        mod._write_json(tmp_path / "r.json", {"fits": {"m": {"test_logloss": 1.0}}})
    assert not (tmp_path / "r.json").exists()


@pytest.mark.parametrize(
    ("flag", "dirname"),
    [("--train-subset", "dl/pilot"), ("--plan-c-subsample", "dl_sub90")],
)
def test_subset_directories(repo, flag, dirname) -> None:
    assert mod.main(argv(repo, flag, "90")) == 0
    rel = repo / "data" / "rel"
    assert (rel / dirname / PRED).exists()
    assert not (rel / "dl" / PRED).exists()
    assert (rel / "dl" / "environment.json").exists()


def test_t2_failure_outside_pilot_discards_outputs_and_exits_1(
    repo, monkeypatch, capsys
) -> None:
    force_t2_fail(monkeypatch)
    dl = repo / "data" / "rel" / "dl"
    assert mod.main(argv(repo)) == 1
    assert "T2" in capsys.readouterr().err
    assert [p for p in outputs(dl) if p.exists()] == []
    report = json.loads((dl / "train_report_dl.json").read_text(encoding="utf-8"))
    entry = report["fits"]["mlp_full_s0"]
    assert entry["status"] == "failed_T2_beats_prior"
    assert "prediction_file" not in entry and "model_file" not in entry


def test_t2_failure_in_pilot_keeps_outputs(repo, monkeypatch) -> None:
    force_t2_fail(monkeypatch)
    pilot = repo / "data" / "rel" / "dl" / "pilot"
    assert mod.main(argv(repo, "--train-subset", "90")) == 0
    assert all(p.exists() for p in outputs(pilot))
