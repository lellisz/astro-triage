"""Phase 3 triage: data_triage.triage_summary, charts_triage."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from astro_triage.dashboard import charts_triage as CT
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
def test_axis_floor_default_when_all_respect_it():
    assert CT.axis_floor(pd.Series([0.96, 0.99, 1.0])) == 0.96


@pytest.mark.parametrize(
    "values, expected", [([0.9534, 0.99], 0.95), ([0.8999, 1.0], 0.89), ([0.5], 0.5)]
)
def test_axis_floor_rounds_down_to_two_decimals(values, expected):
    assert CT.axis_floor(pd.Series(values)) == pytest.approx(expected)


def test_axis_floor_empty_or_nan_uses_default():
    assert CT.axis_floor(pd.Series([], dtype=float)) == 0.96
    assert CT.axis_floor(pd.Series([math.nan])) == 0.96


# ---- triage_curve ----
@pytest.fixture(scope="module")
def fig():
    c = _curve()
    return CT.triage_curve(c, DT.triage_summary(c, 0.9))


def test_curve_names_no_legend_and_serializes(fig):
    assert all(t.name and "trace" not in t.name.lower() for t in fig.data)
    assert "Curva de abstenção" in [t.name for t in fig.data]
    assert not fig.layout.showlegend
    assert fig.to_json()


def test_curve_axes(fig):
    x = fig.layout.xaxis
    assert x.tickformat == ".0%" and x.dtick == 0.2 and tuple(x.range) == (0, 1)


def test_curve_floor_in_range_and_note():
    c = _curve()
    c.loc[0, "accuracy_kept"] = 0.8123  # floor must drop to 0.81
    f = CT.triage_curve(c, DT.triage_summary(c, 0.9))
    assert f.layout.yaxis.range[0] == pytest.approx(0.81)
    notes = [a.text for a in f.layout.annotations if "começa em" in a.text]
    assert len(notes) == 1 and "81%" in notes[0] and "96%" not in notes[0]


def test_curve_y_title_short(fig):
    assert fig.layout.yaxis.title.text == "Acerto entre os decididos"
    assert fig.layout.yaxis.range[0] == 0.96 or any(
        "começa em" in a.text for a in fig.layout.annotations
    )


def test_curve_two_point_labels(fig):
    texts = [a.text for a in fig.layout.annotations]
    assert sum(t.startswith("Com limiar") for t in texts) == 1
    assert sum(t.startswith("Decidir tudo") for t in texts) == 1
    assert not any("p.p." in t or "acerto global" in t for t in texts)


def test_curve_markers_filled_vs_open(fig):
    by = {t.name: t for t in fig.data}
    assert by["Limiar"].marker.symbol == "circle"
    assert by["Decidir quase tudo"].marker.symbol == "circle-open"


def test_curve_hatched_region_and_annotation(fig):
    fills = [t for t in fig.data if t.fill == "toself"]
    assert len(fills) == 1 and fills[0].fillpattern.shape
    assert any(a.text == "vai para espectro" for a in fig.layout.annotations)
