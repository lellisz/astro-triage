"""Seed override of run_model and scripts/train_lgbm_seeds.py (ADR 0006 3.8, 15); offline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from test_lgbm import FAST_PARAMS, copy_inputs, make_dataset

from astro_triage.models import lgbm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_lgbm_seeds as mod

SEED = lgbm.SEED
FULL = ["--features", "full"]
SD = "lgbm_seeds/"
REPORT = "train_report_lgbm_seeds.json"
P1 = "predictions_lgbm_full_s1_v0"


@pytest.fixture(autouse=True)
def tiny_lgbm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lgbm, "PARAMS", FAST_PARAMS)
    monkeypatch.setattr(lgbm, "MAX_TREES", 60)
    monkeypatch.setattr(lgbm, "PATIENCE", 5)
    monkeypatch.setattr(lgbm, "REPO_ROOT", tmp_path)


@pytest.fixture(scope="module")
def src(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("seedsrc") / "ds")


def _meta(path: Path) -> dict[bytes, bytes]:
    md = pq.read_schema(path).metadata
    return {k: v for k, v in md.items() if k not in {b"created_utc", b"git_sha"}}


def _query(outdir: Path, rel: str) -> str:
    return json.loads((outdir / rel).read_text("utf-8"))["query_text"]


def _tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def _no_abs(blob: str, tmp: Path) -> None:
    blob = blob.replace("\\\\", "\\")
    assert str(tmp) not in blob and tmp.as_posix() not in blob
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", blob)


def _run(rel: Path | str, *args: str) -> int:
    return mod.main(["--outdir", str(rel), *args])


def _fits(rel: Path) -> dict:
    return json.loads((rel / SD / REPORT).read_text("utf-8"))["fits"]


def _fit(src: Path, tmp: Path, name: str, **kw: object):
    out = copy_inputs(src, tmp / "data" / name)
    return out, lgbm.run_model(lgbm.MODEL_FULL, lgbm.load_folds(out), out, **kw)


def test_default_call_equals_explicit_defaults(src, tmp_path) -> None:
    a, ra = _fit(src, tmp_path, "a")
    b, rb = _fit(src, tmp_path, "b", seed=SEED, model_id="lgbm_full")
    pa, pb = (x / "predictions_lgbm_full_v0.parquet" for x in (a, b))
    assert pq.read_table(pa).equals(pq.read_table(pb)) and _meta(pa) == _meta(pb)
    ma, mb = (x / "models/lgbm_full.txt" for x in (a, b))
    assert ma.read_bytes() == mb.read_bytes()
    assert ra["best_iteration"] == rb["best_iteration"]
    for m in ("predictions_lgbm_full_v0", "models/model_lgbm_full"):
        q = _query(a, f"{m}.manifest.json")  # golden values from the module constants
        assert f"params={json.dumps(lgbm.PARAMS, sort_keys=True)};" in q
        assert f"seed={lgbm.SEED};" in q


def test_seed_override_reaches_manifests_and_boosters(src, tmp_path) -> None:
    before = dict(lgbm.PARAMS)
    a, _ = _fit(src, tmp_path, "s0", seed=SEED, model_id="lgbm_full_s0")
    b, _ = _fit(src, tmp_path, "s1", seed=SEED + 1, model_id="lgbm_full_s1")
    assert lgbm.PARAMS == before
    for m in (P1, "models/model_lgbm_full_s1"):
        q = _query(b, f"{m}.manifest.json")
        assert str(SEED) not in q
        assert f"seed={SEED + 1};" in q and f'"seed": {SEED + 1}' in q
        assert q.count(str(SEED + 1)) == 2
    ta = (a / "models/lgbm_full_s0.txt").read_text()
    tb = (b / "models/lgbm_full_s1.txt").read_text()
    assert f"[seed: {SEED}]" in ta and f"[seed: {SEED + 1}]" in tb
    assert ta.split("end of trees")[0] != tb.split("end of trees")[0]


def test_seed_fits_go_to_subdir_old_artefacts_untouched(src, tmp_path, capsys) -> None:
    rel = copy_inputs(src, tmp_path / "data" / "rel")
    (rel / "models").mkdir()
    for i, n in enumerate(["predictions_lgbm_full_v0.parquet", "models/lgbm_full.txt"]):
        (rel / n).write_bytes(f"PREVIOUS-{i}".encode())
    before = _tree(rel)
    assert _run(rel, "--features", "full", "full", "--seeds", "0-1") == 0  # N-01 dedupe
    assert {k: v for k, v in _tree(rel).items() if k in before} == before
    for k in (0, 1):
        assert (rel / SD / f"models/lgbm_full_s{k}.txt").is_file()
        meta = _meta(rel / SD / f"predictions_lgbm_full_s{k}_v0.parquet")
        assert meta[b"model_id"] == f"lgbm_full_s{k}".encode()
    done = re.findall(r"(?m)^(\w+): seed \d+, wall", capsys.readouterr().err)
    assert done == sorted(_fits(rel)) == ["lgbm_full_s0", "lgbm_full_s1"]  # once each


def test_report_merges_has_no_test_key_and_no_abs_path(src, tmp_path) -> None:
    rel = copy_inputs(src, tmp_path / "data" / "rel")
    assert _run(rel, *FULL, "--seeds", "0-1") == 0
    first = _fits(rel)
    assert _run(rel, "--features", "noind", "--seeds", "3") == 0
    text, fits = (rel / SD / REPORT).read_text("utf-8"), _fits(rel)
    assert set(fits) == {"lgbm_full_s0", "lgbm_full_s1", "lgbm_noind_s3"}
    assert fits["lgbm_full_s0"] == first["lgbm_full_s0"]
    assert fits["lgbm_noind_s3"]["seed"] == SEED + 3
    assert not [k for k in re.findall(r'"([^"]*)":', text) if "test" in k.lower()]
    for blob in [text, *(p.read_text("utf-8") for p in rel.rglob("*.manifest.json"))]:
        _no_abs(blob, tmp_path)


@pytest.mark.parametrize(
    "existing",
    [
        f"{P1}.parquet",
        "models/lgbm_full_s1.txt",
        f"{P1}.manifest.json",
        "models/model_lgbm_full_s1.manifest.json",
        REPORT,
    ],
)
def test_existing_target_refuses(src, tmp_path, existing) -> None:
    rel = copy_inputs(src, tmp_path / "data" / "rel")
    (rel / SD / "models").mkdir(parents=True)
    path = rel / SD / existing
    keep = b'{"fits": {"lgbm_full_s1": {}}}'  # valid JSON for the report case
    path.write_bytes(keep)
    before = _tree(rel)
    assert _run(rel, *FULL, "--seeds", "0-2") != 0
    assert _tree(rel) == before  # not even the s0 fit that precedes s1 was written
    assert path.read_bytes() == keep


def test_outdir_outside_data_refused(src, tmp_path, monkeypatch, capsys) -> None:
    sibling = copy_inputs(src, tmp_path / "data_other" / "rel")
    elsewhere = copy_inputs(src, tmp_path / "other" / "elsewhere")
    (tmp_path / "data").mkdir()
    before = {d: _tree(d) for d in (sibling, elsewhere)}
    monkeypatch.chdir(tmp_path / "data")
    for out in (str(sibling), "../other/elsewhere", "../data_other/rel"):
        assert _run(out, *FULL, "--seeds", "0") != 0
        err = capsys.readouterr().err
        assert "must be inside" in err and Path(out).as_posix() in err
        if not Path(out).is_absolute():  # typed form only, never the resolved path
            _no_abs(err, tmp_path)
            assert str(Path(out).resolve()) not in err
    assert {d: _tree(d) for d in before} == before


def test_measure_one(src, tmp_path, capsys) -> None:
    rel = copy_inputs(src, tmp_path / "data" / "rel")
    for extra in (FULL, ["--seeds", "0"]):
        assert _run(rel, "--measure-one", *extra) != 0
        assert not (rel / "lgbm_seeds").exists()
    assert _run(rel, "--measure-one") == 0
    status = json.loads(capsys.readouterr().out)["status"]
    assert "--outdir data/rel --features full --seeds 1-4" in status
    assert "--outdir data/rel --features noind --seeds 0-4" in status
    _no_abs(status, tmp_path)
    preds = [p.name for p in (rel / "lgbm_seeds").glob("predictions_*.parquet")]
    assert preds == ["predictions_lgbm_full_s0_v0.parquet"]


def test_report_write_is_atomic_and_parse_seeds(tmp_path) -> None:
    path = tmp_path / REPORT
    mod._write_report(path, {"lgbm_full_s0": {"seed": SEED}})
    good = path.read_bytes()
    with pytest.raises(TypeError):
        mod._write_report(path, {"lgbm_full_s1": object()})
    with pytest.raises(lgbm.TrainError):  # recursive guard, also inside lists
        mod._write_report(path, {"a": [{"x": {"Test_acc": 1}}]})
    assert path.read_bytes() == good
    assert mod.parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert mod.parse_seeds("0,2,3") == [0, 2, 3]
    for bad in ("", "a-b", "0-5", "4-0", "1,1", "7"):
        with pytest.raises(lgbm.TrainError):
            mod.parse_seeds(bad)
