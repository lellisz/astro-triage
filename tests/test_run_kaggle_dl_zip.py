# ruff: noqa: F811
"""kaggle/run_kaggle_dl.py: output zip (finally, members, refusals, pilot batch)."""

import json
import zipfile

import pytest
from kaggle_dl_kit import MODE, put, rk, run_main, stage_world, write_report, zip_names  # noqa: F401


# fmt: off
def seed_dl(rk, done=("mlp_full_s0",), unfinished=("mlp_full_s1",)):
    fits = {}
    for mid in done:
        put(rk.dl / f"predictions_{mid}_v0.parquet")
        put(rk.dl / f"models/{mid}.pt")
        put(rk.dl / f"ckpt/{mid}/last.pt")
        fits[mid] = {"prediction_sha256": "h"}
    for mid in unfinished:
        put(rk.dl / f"ckpt/{mid}/last.pt")
    write_report(rk.dl, fits)


def test_zip_built_in_finally_on_training_failure(rk):
    stage_world(rk)
    rk.handlers["train"] = lambda cmd: (seed_dl(rk), 7)[1]
    assert "exited with code 7" in str(run_main(rk))
    names = zip_names(rk)
    assert "dl/predictions_mlp_full_s0_v0.parquet" in names  # partial work is kept
    assert "dl/train_report_dl.json" in names and "run_log.txt" in names


def test_zip_members_unfinished_ckpt_only_and_log(rk):
    stage_world(rk)
    rk.handlers["train"] = lambda cmd: (seed_dl(rk), 0)[1]
    put(rk.work / "environment.json", "{}")
    assert run_main(rk) == 0
    names = zip_names(rk)
    assert {"dl/models/mlp_full_s0.pt", "dl/train_report_dl.json", "environment.json", "run_log.txt"} <= names
    assert "dl/ckpt/mlp_full_s1/last.pt" in names
    assert "dl/ckpt/mlp_full_s0/last.pt" not in names  # finished -> ckpt dropped


def test_zip_skips_tmp_dotfiles_and_pilot(rk):
    stage_world(rk)
    seed_dl(rk)
    put(rk.dl / ".report.x.tmp"); put(rk.dl / "x.tmp"); put(rk.dl / "pilot/predictions_a_v0.parquet")
    names = {a for _p, a in rk.zip_members(MODE)}
    assert not any(".tmp" in n or "pilot" in n for n in names)


def test_working_dir_cleaned_except_zip_and_log(rk):
    stage_world(rk)
    rk.handlers["train"] = lambda cmd: (seed_dl(rk), 0)[1]
    run_main(rk)
    assert sorted(p.name for p in rk.work.iterdir()) == ["astro_triage_dl_outputs.zip", "run_log.txt"]


@pytest.mark.parametrize("bad", ["labels.parquet", "features.parquet", "lgbm_seeds/predictions_lgbm_full_s0_v0.parquet",
                                 "lgbm_seeds/sub/x.json"])
def test_zip_refuses_banned_members(rk, bad):
    seed_dl(rk)
    put(rk.dl / bad)
    with pytest.raises(SystemExit, match=r"\[zip\] refusing to package"):
        rk.package_outputs(MODE)
    assert not (rk.work / "astro_triage_dl_outputs.zip").exists()


def test_dataset_level_lgbm_seeds_and_inputs_never_packaged(rk):
    seed_dl(rk)
    ds = rk.dl.parent
    put(ds / "lgbm_seeds/predictions_lgbm_full_s0_v0.parquet")
    put(ds / "labels.parquet")
    put(rk.work / "run_log.txt", "log")
    rk.package_outputs(MODE)
    assert not any("lgbm_seeds" in n or "labels" in n for n in zip_names(rk))


def test_zip_refuses_banned_name_via_runner_failure_path(rk):
    stage_world(rk)
    rk.handlers["train"] = lambda cmd: (seed_dl(rk), put(rk.dl / "features.parquet"), 0)[2]
    assert "refusing to package" in str(run_main(rk))


def test_pilot_batch_ships_only_pilot_report_env_and_log(rk):
    mode = {**MODE, "train_subset": 1000}
    seed_dl(rk)
    put(rk.dl / "pilot/train_report_dl.json", "{}")
    put(rk.dl / "pilot/predictions_a_v0.parquet")
    put(rk.dl / "pilot/models/a.pt")
    put(rk.work / "environment.json", "{}")
    put(rk.work / "run_log.txt", "log")
    rk.package_outputs(mode)
    assert zip_names(rk) == {"dl/pilot/train_report_dl.json", "environment.json", "run_log.txt"}


def test_pilot_batch_without_report_still_zips_log(rk):
    put(rk.work / "run_log.txt", "log")
    rk.package_outputs({**MODE, "train_subset": 5})
    assert zip_names(rk) == {"run_log.txt"}


def test_zip_is_valid_and_parquet_stored(rk):
    seed_dl(rk)
    put(rk.work / "run_log.txt", "log")
    rk.package_outputs(MODE)
    with zipfile.ZipFile(rk.OUT_ZIP) as z:
        assert z.testzip() is None
        assert z.getinfo("dl/predictions_mlp_full_s0_v0.parquet").compress_type == zipfile.ZIP_STORED
        assert json.loads(z.read("dl/train_report_dl.json"))["fits"]
