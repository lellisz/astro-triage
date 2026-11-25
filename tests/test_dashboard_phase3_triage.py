"""Phase 3 triage: data_triage.triage_summary, charts_triage."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from astro_triage.dashboard import data_triage as DT


def _curve():
    return pd.DataFrame(
        {
            "threshold": [0.34, 0.5, 0.7, 0.9, 0.99],
            "coverage": [1.0, 0.9, 0.8, 0.6, 0.2],
            "accuracy_kept": [0.90, 0.93, 0.95, 0.97, 0.99],
            "n_set_aside": [0, 100, 2000, 40000, 80000],
            "errors_set_aside": [0, 10, 300, 1000, 1500],
            "errors_total": [1600] * 5,
        }
    )


def test_summary_nearest_row_and_decide_all():
    s = DT.triage_summary(_curve(), 0.88)
    assert s["threshold"] == 0.9
    assert s["coverage"] == 0.6 and s["accuracy_kept"] == 0.97
    assert s["n_set_aside"] == 40000 and s["errors_set_aside"] == 1000
    assert s["errors_total"] == 1600
    assert s["decide_all_threshold"] == 0.34  # the minimum threshold
    assert s["decide_all_coverage"] == 1.0 and s["decide_all_accuracy"] == 0.90


@pytest.mark.parametrize("thr, expected", [(-5.0, 0.34), (7.0, 0.99)])
def test_summary_threshold_out_of_range_clamps_to_edge(thr, expected):
    assert DT.triage_summary(_curve(), thr)["threshold"] == expected


@pytest.mark.parametrize(
    "curve",
    [
        pd.DataFrame(),
        None,
        pd.DataFrame({"coverage": [1.0]}),
        pd.DataFrame({"threshold": [math.nan]}),
    ],
)
def test_summary_unusable_curve_is_none(curve):
    assert DT.triage_summary(curve, 0.9) is None


def test_summary_missing_values_become_nan():
    curve = pd.DataFrame({"threshold": [0.4, 0.9], "coverage": [1.0, None]})
    s = DT.triage_summary(curve, 0.9)
    assert s["threshold"] == 0.9
    assert math.isnan(s["coverage"]) and math.isnan(s["accuracy_kept"])
    assert math.isnan(s["errors_total"])
    assert s["decide_all_coverage"] == 1.0 and math.isnan(s["decide_all_accuracy"])


def test_summary_single_row_both_points_coincide():
    s = DT.triage_summary(_curve().iloc[[2]], 0.9)
    assert s["threshold"] == s["decide_all_threshold"] == 0.7


# ---- axis_floor ----






# ---- triage_curve ----














