"""restore_seed_outputs of kaggle/run_kaggle.py: merge, orphans, zip-slip, atomicity.

Synthetic files under tmp_path only. Swap the module with KAGGLE_RUNNER_UNDER_TEST
(mutation checks).
"""

import importlib.util
import json
import os
import zipfile
from pathlib import Path

import pytest

DEFAULT = Path(__file__).resolve().parents[1] / "kaggle" / "run_kaggle.py"
REPORT = "train_report_lgbm_seeds.json"


@pytest.fixture
def rk(tmp_path, monkeypatch):
    path = Path(os.environ.get("KAGGLE_RUNNER_UNDER_TEST", DEFAULT))
    spec = importlib.util.spec_from_file_location("rk_restore_ut", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ds = tmp_path / "working/repo/data/interim/full_v0"
    ds.mkdir(parents=True)
    (tmp_path / "input").mkdir()
    monkeypatch.setattr(mod, "INPUT", str(tmp_path / "input"))
    monkeypatch.setattr(mod, "DATASET_DIR", str(ds))
    monkeypatch.setattr(mod, "SEEDS_STATE", {})
    mod.logs = []
    monkeypatch.setattr(mod, "log", mod.logs.append)
    mod.target = ds / "lgbm_seeds"
    mod.root = tmp_path
    return mod


def fit(tag):
    return {"wall_seconds": tag}


def put(root, rel, text="x"):
    path = Path(root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def src_dir(rk, name, fits, extra=None):
    base = rk.root / "input" / name / "lgbm_seeds"
    put(base, REPORT, json.dumps({"fits": fits}) if isinstance(fits, dict) else fits)
    for rel, text in (extra or {}).items():
        put(base, rel, text)


def src_zip(rk, name, fits, extra=None):
    path = rk.root / "input" / name / "astro_triage_outputs.zip"
    path.parent.mkdir(parents=True)
    raw = json.dumps({"fits": fits}) if isinstance(fits, dict) else fits
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"lgbm_seeds/{REPORT}", raw)
        for rel, text in (extra or {}).items():
            z.writestr(f"lgbm_seeds/{rel}", text)


def merged(rk):
    return json.loads((rk.target / REPORT).read_text())["fits"]


def art(i):
    return {f"models/{i}.txt": i, f"predictions_{i}_v1.parquet": i}


# fmt: off
@pytest.mark.parametrize("make", [src_dir, src_zip])
def test_disjoint_sources_merge_and_first_wins(rk, make):
    make(rk, "a", {"lgbm_full_s0": fit("A0"), "lgbm_full_s1": fit("A1")}, {**art("lgbm_full_s0"), **art("lgbm_full_s1")})
    make(rk, "b", {"lgbm_full_s1": fit("B1"), "lgbm_noind_s0": fit("B2")}, {**art("lgbm_noind_s0"), "models/lgbm_full_s1.txt": "OTHER"})
    rk.restore_seed_outputs()
    assert merged(rk) == {"lgbm_full_s0": fit("A0"), "lgbm_full_s1": fit("A1"), "lgbm_noind_s0": fit("B2")}
    assert (rk.target / "models/lgbm_full_s1.txt").read_text() == "lgbm_full_s1"  # no overwrite
    assert (rk.target / "models/lgbm_noind_s0.txt").is_file()
    assert rk.SEEDS_STATE["fits_skipped_as_restored"] == ["lgbm_full_s0", "lgbm_full_s1", "lgbm_noind_s0"]
    assert not list(rk.target.rglob("*.tmp")) and not list(rk.target.glob(".report.*"))


def test_existing_target_report_and_files_win_over_sources(rk):
    put(rk.target, REPORT, json.dumps({"fits": {"lgbm_full_s0": fit("LOCAL")}}))
    put(rk.target, "models/lgbm_full_s0.txt", "LOCAL")
    src_dir(rk, "a", {"lgbm_full_s0": fit("SRC"), "lgbm_full_s2": fit("S2")}, {**art("lgbm_full_s0"), **art("lgbm_full_s2")})
    rk.restore_seed_outputs()
    assert merged(rk) == {"lgbm_full_s0": fit("LOCAL"), "lgbm_full_s2": fit("S2")}
    assert (rk.target / "models/lgbm_full_s0.txt").read_text() == "LOCAL"


@pytest.mark.parametrize("make", [src_dir, src_zip])
def test_orphan_artefacts_fail_early_naming_sorted_ids(rk, make):
    extra = {**art("lgbm_noind_s3"), "models/model_lgbm_full_s2.manifest.json": "{}", **art("lgbm_full_s0")}
    make(rk, "a", {"lgbm_full_s0": fit(0)}, extra)
    with pytest.raises(SystemExit) as exc:
        rk.restore_seed_outputs()
    assert "['lgbm_full_s2', 'lgbm_noind_s3']" in str(exc.value)
    assert "fits_skipped_as_restored" not in rk.SEEDS_STATE


@pytest.mark.parametrize("raw", ["{not json", '{"fits": []}', "{}", "[1]", '{"fits": 3}'])
@pytest.mark.parametrize("make", [src_dir, src_zip])
def test_malformed_source_report_is_skipped_and_its_artefacts_are_orphans(rk, make, raw):
    src_dir(rk, "a", {"lgbm_full_s0": fit(0)}, art("lgbm_full_s0"))
    make(rk, "b", raw, {"models/lgbm_noind_s1.txt": "x"})
    with pytest.raises(SystemExit) as exc:
        rk.restore_seed_outputs()
    assert "['lgbm_noind_s1']" in str(exc.value)
    assert any("ignoring unreadable report" in line for line in rk.logs)
    assert merged(rk) == {"lgbm_full_s0": fit(0)}


def test_zip_slip_and_filtered_names_never_escape_or_land(rk):
    names = ["../x", "/abs", "lgbm_seeds/../x", "lgbm_seeds/../../evil", "lgbm_seeds//x", "lgbm_seeds/.hid", "lgbm_seeds/models/.d"]
    path = rk.root / "input/a/astro_triage_outputs.zip"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as z:
        for name in [*names, "lgbm_seeds/ok.txt"]:
            z.writestr(name, "boom")
    before = set(rk.root.rglob("*"))
    rk.restore_seed_outputs()
    new = {p.name for p in rk.root.rglob("*") if p not in before}
    assert not new & {"x", "evil", "abs", ".hid", ".d"}
    assert (rk.target / "ok.txt").read_text() == "boom"


def test_dir_source_skips_dotfiles_and_tmp(rk):
    src_dir(rk, "a", {"lgbm_full_s0": fit(0)}, {".hidden": "h", "models/z.tmp": "t", "keep.bin": "k"})
    rk.restore_seed_outputs()
    assert sorted(p.name for p in rk.target.rglob("*") if p.is_file()) == ["keep.bin", REPORT]


def test_atomic_write_failure_leaves_no_partial_report(rk, monkeypatch):
    src_dir(rk, "a", {"lgbm_full_s0": fit(0)})
    monkeypatch.setattr(rk.os, "replace", lambda *a: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        rk.restore_seed_outputs()
    assert not (rk.target / REPORT).exists() and not list(rk.target.glob(".report.*"))
    put(rk.target, REPORT, '{"fits": {"old": 1}}')
    with pytest.raises(ZeroDivisionError):
        rk.restore_seed_outputs()
    assert (rk.target / REPORT).read_text() == '{"fits": {"old": 1}}'
    assert not list(rk.target.glob(".report.*"))


def test_remaining_fits_skip_merged_and_trainer_would_not_clash(rk):
    done = ["lgbm_full_s0", "lgbm_full_s1", "lgbm_noind_s0", "lgbm_noind_s4"]
    src_dir(rk, "a", {k: fit(0) for k in done}, {i: "x" for k in done for i in art(k)})
    rk.restore_seed_outputs()
    rest = rk.remaining_fits({"fits": merged(rk)})
    assert rest == {"full": [2, 3, 4], "noind": [1, 2, 3]}
    ids = [f"lgbm_{n}_s{k}" for n, ks in rest.items() for k in ks]
    assert not set(ids) & set(merged(rk))
    for mid in ids:  # the names train_lgbm_seeds.run() refuses to overwrite
        assert not (rk.target / "models" / f"{mid}.txt").exists()
        assert not (rk.target / "models" / f"model_{mid}.manifest.json").exists()
        assert not list(rk.target.glob(f"predictions_{mid}_*"))
# fmt: on
