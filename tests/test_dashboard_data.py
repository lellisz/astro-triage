"""Tests for the pure functions of the dashboard data layer (synthetic data, no files)."""

import math

import numpy as np
import pandas as pd
import pytest

from astro_triage.dashboard import data as D


def _release(true, pred, conf) -> D.Release:
    test = pd.DataFrame(
        {
            "true": true,
            "pred_lgbm_full": pred,
            "conf_lgbm_full": conf,
            "healpix_l5": np.arange(len(true)) % 2,
        }
    )
    test["err_lgbm_full"] = test["true"] != test["pred_lgbm_full"]
    return D.Release("t", D.REPO_ROOT, {"meta": {}}, {}, pd.DataFrame(), test, 0)


def test_wilson_contains_point_and_handles_empty():
    lo, hi = D.wilson(5, 50)
    assert lo < 0.1 < hi
    assert all(math.isnan(v) for v in D.wilson(0, 0))
    assert D.wilson(0, 10)[0] == 0.0 and D.wilson(10, 10)[1] == 1.0


def test_mollweide_poles_and_centre():
    x, y = D.mollweide(np.array([0.0, 0.0, 0.0]), np.array([0.0, 90.0, -90.0]))
    assert x[0] == pytest.approx(0.0) and y[0] == pytest.approx(0.0)
    assert y[1] == pytest.approx(math.sqrt(2), abs=1e-6)
    assert y[2] == pytest.approx(-math.sqrt(2), abs=1e-6)


def test_mollweide_longitude_increases_to_the_left():
    x, _ = D.mollweide(np.array([10.0, 350.0]), np.array([0.0, 0.0]))
    assert x[0] < 0 < x[1]


def test_confidence_histogram_counts_every_row_once():
    rel = _release(
        ["STAR", "QSO", "GALAXY", "STAR"],
        ["STAR", "STAR", "GALAXY", "STAR"],
        [0.3, 0.55, 0.995, 1.0],
    )
    hist = D.confidence_histogram(rel)
    assert hist["n"].sum() == 4
    assert hist["n_wrong"].sum() == 1


def test_abstention_curve_threshold_sets_rows_aside():
    rel = _release(["STAR", "QSO", "STAR"], ["STAR", "STAR", "STAR"], [0.99, 0.6, 0.95])
    row = D.abstention_curve(rel, thresholds=np.array([0.9])).iloc[0]
    assert row["coverage"] == pytest.approx(2 / 3)
    assert row["accuracy_kept"] == 1.0
    assert row["errors_set_aside"] == 1 and row["errors_total"] == 1


def test_confusion_and_error_types_follow_class_names():
    rel = _release(["STAR", "STAR", "QSO"], ["STAR", "QSO", "QSO"], [0.9, 0.7, 0.9])
    conf = D.confusion(rel)
    assert list(conf.index) == list(D.CLASSES)
    assert conf.loc["STAR", "QSO"] == 1
    errs = D.errors_by_type(rel)
    assert errs.iloc[0]["true"] == "STAR" and errs.iloc[0]["pred"] == "QSO"
