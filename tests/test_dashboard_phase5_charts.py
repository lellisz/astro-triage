"""Phase 5 galaxies charts: rate_dots, rate_bars and galaxy_scatter (synthetic data)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from astro_triage.dashboard import charts_galaxies as cg
from astro_triage.dashboard import charts_galaxies_rates as cr
from astro_triage.dashboard import charts_galaxies_scatter as cs
from astro_triage.dashboard import data_galaxies as dg
from astro_triage.dashboard import data_galaxies_z as dz

FLOOR = 30


def _table(spec):
    """spec: list of (label, n, errors)."""
    return pd.DataFrame(
        [
            dg.rate_row(lab, 0, 1, pd.Series([True] * k + [False] * (n - k)))
            for lab, n, k in spec
        ]
    )


@pytest.fixture
def dots():
    by_g = _table([("a", 200, 20), ("b", 10, 3), ("c", 60, 0)])
    by_z = _table([("d", 500, 50), ("e", 12, 2)])
    return cr.rate_dots(by_g, by_z, 0.1, FLOOR)


def _by_symbol(fig, symbol):
    return [t for t in fig.data if getattr(t.marker, "symbol", None) == symbol]


# (4) rate_dots
def test_dots_traces_have_explicit_names_without_trace(dots):
    assert dots.data
    for t in dots.data:
        assert t.name and "trace" not in t.name.lower()


def test_dots_have_no_legend(dots):
    assert dots.layout.showlegend is not True


def test_dots_axis_is_log_with_shared_range_and_tickformat(dots):
    y = dots.layout.yaxis
    assert y.type == "log"
    assert tuple(y.range) == cg.RATE_LOG_RANGE
    assert y.tickformat


def test_dots_dashed_mean_line(dots):
    dashes = [s.line.dash for s in dots.layout.shapes]
    assert "dash" in dashes
    assert [s.y0 for s in dots.layout.shapes] == [0.1]


def test_dots_filled_circle_for_enough_n_open_for_few(dots):
    filled = _by_symbol(dots, "circle")
    few = _by_symbol(dots, "circle-open")
    assert [list(t.y) for t in filled] == [[pytest.approx(0.1), pytest.approx(0.1)]]
    assert len(few) == 1 and sorted(few[0].customdata[:, 2]) == [10, 12]


def test_dots_zero_error_is_x_symbol(dots):
    xs = _by_symbol(dots, "x")
    assert len(xs) == 1 and len(xs[0].x) == 1


def test_dots_n_in_tick_text(dots):
    texts = dots.layout.xaxis.ticktext
    assert any("n=200" in t for t in texts) and any("n=500" in t for t in texts)


def test_dots_serialise_to_json(dots):
    assert dots.to_json()


def test_dots_with_only_zero_error_rows_does_not_crash():
    g = _table([("a", 50, 0)])
    fig = cr.rate_dots(g, _table([("b", 50, 0)]), 0.0, FLOOR)
    assert fig.to_json() and not _by_symbol(fig, "circle")


# (5) rate_bars
def test_bars_few_group_has_n_in_label_and_open_circle():
    tab = _table([("2 parâmetros", 500, 25), ("5 parâmetros", 4, 1)])
    fig = cg.rate_bars(tab, FLOOR)
    assert any("n=4" in x for x in fig.data[0].x)
    open_ = _by_symbol(fig, "circle-open")
    assert len(open_) == 1 and list(open_[0].x) == [fig.data[0].x[1]]
    assert fig.layout.yaxis.type == "log"


def test_bars_no_open_circle_when_all_groups_large():
    fig = cg.rate_bars(_table([("a", 500, 25), ("b", 300, 9)]), FLOOR)
    assert not _by_symbol(fig, "circle-open")


def test_bars_empty_group_is_labelled_without_objects():
    fig = cg.rate_bars(_table([("a", 500, 25), ("b", 0, 0)]), FLOOR)
    assert any("sem objetos" in x for x in fig.data[0].x)
    assert fig.to_json()


# (6) galaxy_scatter
def _scatter_df():
    rng = np.random.default_rng(1)
    n = 60
    z = np.concatenate(
        [[-0.4, -0.01, 0.0, 5e-4, 1e-3, 2e-4], rng.uniform(0.01, 1, n - 6)]
    )
    pred = np.where(rng.random(n) < 0.5, "QSO", "STAR")
    err = np.zeros(n, bool)
    err[[0, 1, 2, 3, 4, 6, 7, 8, 9]] = True
    pred[~err] = "GALAXY"
    return pd.DataFrame(
        {"z": z, "G": rng.uniform(17, 21, n), "error": err, "pred": pred,
         "confidence": rng.random(n)}
    )  # fmt: skip


def test_scatter_no_clip_pile_at_z_low():
    fig = cs.galaxy_scatter(_scatter_df())
    right = [t for t in fig.data if t.xaxis == "x2"]
    xs = np.concatenate([np.asarray(t.x, float) for t in right if len(t.x)])
    assert (xs[np.isfinite(xs)] > dz.Z_LOW).all()  # NaN = gap between contour segments
    left = [t for t in fig.data if t.xaxis == "x"]
    lx = np.concatenate([np.asarray(t.x, float) for t in left if len(t.x)])
    assert len(set(lx.round(12))) == len(lx)  # jittered, not stacked


def test_scatter_hover_keeps_real_z_for_low_band():
    df = _scatter_df()
    fig = cs.galaxy_scatter(df)
    low = [t for t in fig.data if t.xaxis == "x" and t.customdata is not None]
    zs = sorted(np.concatenate([t.customdata[:, 0].astype(float) for t in low]))
    assert zs == sorted(df[(df["z"] <= dz.Z_LOW) & df["error"]]["z"])
    assert min(zs) < 0 and "customdata[0]" in low[0].hovertemplate


def test_scatter_symbols_quasar_open_star_diamond_rest_small_grey():
    fig = cs.galaxy_scatter(_scatter_df())
    symbols = {t.marker.symbol for t in fig.data if t.marker.symbol}
    assert {"circle-open", "diamond"} <= symbols
    grey = [t for t in fig.data if t.name == "classificada certo"]
    assert grey and all(t.marker.size <= 5 for t in grey)


def test_scatter_axes_magnitude_reversed_z_log():
    fig = cs.galaxy_scatter(_scatter_df())
    assert fig.layout.yaxis.autorange == "reversed"
    assert fig.layout.xaxis2.type == "log"


def test_scatter_is_deterministic():
    df = _scatter_df()
    assert cs.galaxy_scatter(df).to_json() == cs.galaxy_scatter(df).to_json()


def test_scatter_without_errors_or_low_z_does_not_crash():
    df = _scatter_df().assign(error=False, pred="GALAXY", z=0.5)
    assert cs.galaxy_scatter(df).to_json()
    assert not math.isnan(cs.STRIP_HALF_WIDTH)
