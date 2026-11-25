"""Tests of the Gráficos aggregations on a small in-memory release (no files)."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_charts as C
from astro_triage.dashboard.data_galaxies import error_rate_table, wilson_interval
from astro_triage.dashboard.data_galaxies_z import Z_LOW, error_rate_table_z
from astro_triage.dashboard.data_release import CLASSES

MODEL = "lgbm_full"


def _release(n: int = 300, classes: tuple[str, ...] = CLASSES) -> D.Release:
    rng = np.random.default_rng(7)
    true = np.array([classes[i % len(classes)] for i in range(n)])
    p = rng.dirichlet(np.ones(3), size=n)
    raw = rng.dirichlet(np.ones(3), size=n)
    test = pd.DataFrame(
        {
            "true": true,
            "healpix_l5": np.arange(n) % 5,
            "phot_g_mean_mag": rng.uniform(9.0, 23.5, size=n),
            "sdss_z": rng.uniform(-0.01, 3.5, size=n),
        }
    )
    for j, c in enumerate(CLASSES):
        test[f"p_{MODEL}_{c}"] = p[:, j]
        test[f"praw_{MODEL}_{c}"] = raw[:, j]
    test[f"pred_{MODEL}"] = np.array(CLASSES)[p.argmax(axis=1)]
    test[f"conf_{MODEL}"] = p.max(axis=1)
    test[f"err_{MODEL}"] = test["true"] != test[f"pred_{MODEL}"]
    return D.Release("t", D.REPO_ROOT, {"meta": {}}, {}, pd.DataFrame(), test, 0)


def test_edges_are_the_declared_grids():
    assert len(C.G_EDGES) == 13
    assert C.G_EDGES[0] == 10.0 and C.G_EDGES[-1] == 22.0
    assert all(b - a == 1.0 for a, b in zip(C.G_EDGES, C.G_EDGES[1:]))
    assert C.QSO_Z_EDGES == (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def test_calibration_bins_match_calibration_points_and_hits_reproduce_observed():
    rel = _release()
    bins = C.calibration_bins(rel, MODEL)
    ref = D.calibration_points(rel, MODEL, C.CALIBRATION_BINS)
    assert set(bins) == set(CLASSES)
    for c in CLASSES:
        got = bins[c]
        assert got["mean_pred"].tolist() == ref[c]["mean_pred"].tolist()
        assert got["observed"].tolist() == ref[c]["observed"].tolist()
        assert got["n"].tolist() == ref[c]["n"].tolist()
        assert "raw_mean_pred" in got and "raw_observed" in got
        assert got["hits"].dtype.kind == "i"
        assert (got["hits"] / got["n"]).tolist() == got["observed"].tolist()
        for _, r in got.iterrows():
            lo, hi = wilson_interval(int(r["hits"]), int(r["n"]))
            assert (r["ci_lo"], r["ci_hi"]) == (lo, hi)
            assert r["ci_lo"] <= r["observed"] <= r["ci_hi"]


def test_calibration_bins_without_raw_probabilities_keep_the_calibrated_columns():
    rel = _release()
    rel = D.Release(
        "t",
        D.REPO_ROOT,
        {"meta": {}},
        {},
        pd.DataFrame(),
        rel.test.drop(columns=[f"praw_{MODEL}_{c}" for c in CLASSES]),
        0,
    )
    got = C.calibration_bins(rel, MODEL)["STAR"]
    assert "raw_mean_pred" not in got and "hits" in got


def test_calibration_bins_respect_n_bins():
    got = C.calibration_bins(_release(), MODEL, n_bins=4)
    assert all(len(df) == 4 for df in got.values())


def test_confusion_cells_cover_nine_cells_and_sum_to_the_test_fold():
    rel = _release()
    cells = C.confusion_cells(rel, MODEL)
    assert list(zip(cells["true"], cells["pred"])) == [
        (t, p) for t in CLASSES for p in CLASSES
    ]
    assert int(cells["n"].sum()) == len(rel.test)
    for t, grp in cells.groupby("true", sort=False):
        assert (grp["row_n"] == (rel.test["true"] == t).sum()).all()
        assert grp["share"].sum() == pytest.approx(1.0)
        assert ((grp["ci_lo"] <= grp["share"]) & (grp["share"] <= grp["ci_hi"])).all()
    ref = D.confusion(rel, MODEL)
    for _, r in cells.iterrows():
        assert r["n"] == ref.loc[r["true"], r["pred"]]


def test_confusion_cells_of_an_empty_class_have_nan_share():
    rel = _release(classes=("STAR", "GALAXY"))
    cells = C.confusion_cells(rel, MODEL)
    qso = cells[cells["true"] == "QSO"]
    assert len(cells) == 9 and (qso["row_n"] == 0).all() and (qso["n"] == 0).all()
    assert qso["share"].isna().all()
    assert int(cells["n"].sum()) == len(rel.test)


def test_error_by_g_equals_error_rate_table_for_each_class():
    rel = _release()
    got = C.error_by(rel, MODEL, "G")
    assert list(got) == list(CLASSES)
    for c in CLASSES:
        pd.testing.assert_frame_equal(
            got[c], error_rate_table(rel, MODEL, c, "G", C.G_EDGES)
        )


def test_error_by_g_adds_a_row_without_measure_when_g_is_missing():
    rel = _release()
    rel.test.loc[rel.test.index[:6], "phot_g_mean_mag"] = math.nan
    got = C.error_by(rel, MODEL, "G")
    for c in CLASSES:
        last = got[c].iloc[-1]
        assert last["label"] == "sem medida"
        assert last["n"] == rel.test.loc[rel.test["true"] == c, "phot_g_mean_mag"].isna().sum()
        assert math.isnan(last["lo"]) and math.isnan(last["hi"])
        assert int(got[c]["n"].sum()) == int((rel.test["true"] == c).sum())


def test_error_by_g_has_no_missing_row_when_every_g_is_measured():
    got = C.error_by(_release(), MODEL, "G")
    assert all("sem medida" not in df["label"].tolist() for df in got.values())


def test_error_by_g_of_an_empty_class_has_zero_counts_and_nan_rates():
    got = C.error_by(_release(classes=("STAR", "GALAXY")), MODEL, "G")["QSO"]
    assert (got["n"] == 0).all() and (got["errors"] == 0).all()
    assert got["rate"].isna().all() and got["ci_lo"].isna().all()


def test_error_by_z_has_the_low_band_for_galaxies_and_the_qso_grid():
    rel = _release()
    gal_idx = rel.test.index[rel.test["true"] == "GALAXY"][:5]
    rel.test.loc[gal_idx, "sdss_z"] = [-0.005, 0.0, Z_LOW, Z_LOW / 2, -1.0]
    got = C.error_by(rel, MODEL, "z")
    assert list(got) == ["GALAXY", "QSO"]
    pd.testing.assert_frame_equal(got["GALAXY"], error_rate_table_z(rel, MODEL))
    pd.testing.assert_frame_equal(
        got["QSO"], error_rate_table(rel, MODEL, "QSO", "z", C.QSO_Z_EDGES)
    )
    low = got["GALAXY"].iloc[0]
    galaxies = rel.test[rel.test["true"] == "GALAXY"]
    assert low["hi"] == Z_LOW and low["n"] == int((galaxies["sdss_z"] <= Z_LOW).sum())
    assert low["n"] >= 5
    assert int(got["GALAXY"]["n"].sum()) == len(galaxies)


def test_error_by_z_of_an_empty_class_is_all_zero():
    got = C.error_by(_release(classes=("STAR", "GALAXY")), MODEL, "z")
    assert (got["QSO"]["n"] == 0).all()
    assert got["GALAXY"]["n"].sum() > 0


def test_module_has_no_forbidden_imports():
    src = Path(C.__file__).read_text(encoding="utf-8")
    for banned in ("streamlit", "data_cached", "views_", "dl_"):
        assert banned not in src


def test_g_edges_match_the_evaluation_histogram_of_full_v0():
    root = os.environ.get("ASTRO_DATA_ROOT", "")
    path = Path(root) / "interim" / "full_v0" / "eval" / "results.json"
    if not root or not path.is_file():
        pytest.skip("ASTRO_DATA_ROOT sem interim/full_v0/eval/results.json")
    results = json.loads(path.read_text(encoding="utf-8"))
    assert list(C.G_EDGES) == results["histograms"]["g_mag_edges"]
