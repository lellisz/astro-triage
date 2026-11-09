# ruff: noqa: F811
"""kaggle/run_kaggle_dl.py: restore_outputs (dir and zip sources) and skipping finished systems."""

import json
import zipfile
from pathlib import Path

import pytest
from kaggle_dl_kit import MODE, REPORT, put, rk, run_main, stage_world, write_report  # noqa: F401


# fmt: off
def fit(tag, sha="h"):
    return {"prediction_sha256": sha, "tag": tag}


def art(mid):
    return {f"predictions_{mid}_v0.parquet": mid, f"models/{mid}.pt": mid, f"models/{mid}.preproc.pkl": mid}


def src_dir(rk, name, fits, extra=None):
    base = rk.root / "kaggle/input" / name / "full_v0/dl"
    write_report(base, fits)
    for rel, text in (extra or {}).items():
        put(base / rel, text)


def src_zip(rk, name, fits, extra=None, report=None):
    path = rk.root / "kaggle/input" / name / "astro_triage_dl_outputs.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"dl/{REPORT}", report if report is not None else json.dumps({"fits": fits}))
        for rel, text in (extra or {}).items():
            z.writestr(f"dl/{rel}", text)


def merged(rk):
    return json.loads((rk.dl / REPORT).read_text())["fits"]


@pytest.mark.parametrize("make", [src_dir, src_zip])
def test_restore_merges_first_wins_and_never_overwrites(rk, make):
    make(rk, "a", {"mlp_full_s0": fit("A0")}, art("mlp_full_s0"))
    make(rk, "b", {"mlp_full_s0": fit("B0"), "ftt_full_s0": fit("B1")}, {**art("ftt_full_s0"), "models/mlp_full_s0.pt": "OTHER"})
    rk.restore_outputs()
    assert merged(rk) == {"mlp_full_s0": fit("A0"), "ftt_full_s0": fit("B1")}
    assert (rk.dl / "models/mlp_full_s0.pt").read_text() == "mlp_full_s0"
    assert (rk.dl / "predictions_ftt_full_s0_v0.parquet").exists()


def test_existing_local_file_and_report_win_over_sources(rk):
    write_report(rk.dl, {"mlp_full_s0": fit("LOCAL")})
    put(rk.dl / "models/mlp_full_s0.pt", "LOCAL")
    src_dir(rk, "a", {"mlp_full_s0": fit("SRC")}, art("mlp_full_s0"))
    rk.restore_outputs()
    assert merged(rk)["mlp_full_s0"]["tag"] == "LOCAL"
    assert (rk.dl / "models/mlp_full_s0.pt").read_text() == "LOCAL"


def test_pilot_is_ignored_in_both_sources(rk):
    src_dir(rk, "a", {"mlp_full_s0": fit("A")}, {**art("mlp_full_s0"), "pilot/predictions_zzz_v0.parquet": "p", f"pilot/{REPORT}": "{}"})
    src_zip(rk, "b", {}, {"pilot/models/zzz.pt": "p", "pilot/predictions_zzz_v0.parquet": "p"})
    rk.restore_outputs()
    assert not (rk.dl / "pilot").exists()


def test_ckpt_and_environment_restored_and_tmp_dotfiles_skipped(rk):
    src_dir(rk, "a", {}, {"ckpt/mlp_full_s1/last.pt": "c", "environment.json": "{}", "x.tmp": "t", ".hidden": "h"})
    rk.restore_outputs()
    assert (rk.dl / "ckpt/mlp_full_s1/last.pt").read_text() == "c"
    assert (rk.dl / "environment.json").exists()
    assert not (rk.dl / "x.tmp").exists() and not (rk.dl / ".hidden").exists()


def test_artifacts_without_report_entry_fail(rk):
    src_dir(rk, "a", {"mlp_full_s0": fit("A")}, {**art("mlp_full_s0"), **art("ftt_noind_s2")})
    with pytest.raises(SystemExit, match=r"ftt_noind_s2"):
        rk.restore_outputs()


def test_orphan_manifest_and_model_prefix_detected(rk):
    src_dir(rk, "a", {}, {"models/model_ftt_full_s4.manifest.json": "m"})
    with pytest.raises(SystemExit, match=r"ftt_full_s4"):
        rk.restore_outputs()


def test_unreadable_report_is_ignored_not_fatal(rk):
    src_zip(rk, "a", {}, None, report="{broken")
    src_dir(rk, "b", {"mlp_full_s0": fit("B")}, art("mlp_full_s0"))
    rk.restore_outputs()
    assert merged(rk) == {"mlp_full_s0": fit("B")}


def test_zip_slip_member_not_extracted(rk):
    path = rk.root / "kaggle/input/a/astro_triage_dl_outputs.zip"
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("dl/../../evil.txt", "e")
    rk.restore_outputs()
    assert not (rk.root / "evil.txt").exists() and not list(rk.root.rglob("evil.txt"))


def test_zip_named_variants_are_found(rk):
    src_zip(rk, "a", {"mlp_full_s0": fit("A")}, art("mlp_full_s0"))
    p = rk.root / "kaggle/input/a/astro_triage_dl_outputs.zip"
    p.rename(p.with_name("astro_triage_dl_outputs (1).zip"))
    rk.restore_outputs()
    assert "mlp_full_s0" in merged(rk)


def test_finished_requires_hash_and_parquet(rk):
    write_report(rk.dl, {"a_s0": fit("x"), "b_s0": {"prediction_sha256": ""}, "c_s0": {}, "d_s0": fit("d")})
    put(rk.dl / "predictions_a_s0_v0.parquet")
    put(rk.dl / "predictions_b_s0_v0.parquet")
    assert rk.finished_ids(str(rk.dl)) == {"a_s0"}  # d has a hash but no parquet
    assert rk.finished_ids(str(rk.dl / "none")) == set()


def test_unreadable_local_report_in_finished_ids_fails(rk):
    put(rk.dl / REPORT, "{nope")
    with pytest.raises(SystemExit, match="cannot read"):
        rk.finished_ids(str(rk.dl))


def test_finished_systems_leave_the_seeds_list_end_to_end(rk):
    stage_world(rk)
    fits = {}
    for mid in ("ftt_full_s0", "ftt_full_s1", "mlp_full_s1"):
        fits[mid] = fit(mid)
    art_files = {a: "x" for m in fits for a in art(m)}
    src_zip(rk, "prev", fits, art_files)
    assert run_main(rk) == 0
    seeds = {}
    for c in rk.calls:
        if any(str(p).endswith("train_dl.py") for p in c):
            seeds[(c[c.index("--arch") + 1], c[c.index("--features") + 1])] = c[c.index("--seeds") + 1]
    assert ("ftt", "full") not in seeds  # fully done
    assert seeds[("mlp", "full")] == "0"
    assert seeds[("ftt", "noind")] == "0,1" and seeds[("mlp", "noind")] == "0,1"


def test_train_units_pure(rk):
    units = rk.train_units(MODE, {"mlp_noind_s0", "mlp_noind_s1", "ftt_full_s1"})
    assert units == [("ftt", "full", [0]), ("mlp", "full", [0, 1]), ("ftt", "noind", [0, 1])]


def test_restore_failure_still_produces_zip(rk):
    stage_world(rk)
    src_dir(rk, "a", {}, art("ftt_noind_s2"))
    assert "ftt_noind_s2" in str(run_main(rk))
    assert Path(rk.OUT_ZIP).exists()
