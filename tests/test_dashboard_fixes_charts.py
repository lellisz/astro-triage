"""Dashboard phase-1 fixes: zero-error rows, rate bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from astro_triage.dashboard import charts


# 6. zero_error_rows / rate_bars
def _table():
    return pd.DataFrame(
        {
            "label": ["a", "b", "c", "d"],
            "n": [50, 4, 0, 10],
            "errors": [0, 4, 0, 1],
            "rate": [0.0, 1.0, np.nan, 0.1],
            "ci_lo": [0.0, 0.5, np.nan, 0.02],
            "ci_hi": [0.07, 1.0, np.nan, 0.4],
        }
    )


def test_zero_error_rows_only_n_positive_and_no_errors():
    out = charts.zero_error_rows(_table())
    assert list(out["label"]) == ["a"]


def test_zero_error_rows_empty_table():
    empty = _table().iloc[0:0]
    assert charts.zero_error_rows(empty).empty


def test_rate_bars_zero_and_all_error_rows_share_log_range():
    t = _table().iloc[:2]
    fig = charts.rate_bars(t)
    assert fig.layout.yaxis.type == "log"
    assert tuple(fig.layout.yaxis.range) == charts.RATE_LOG_RANGE
    markers = [tr for tr in fig.data if getattr(tr, "mode", None) == "markers+text"]
    assert len(markers) == 1
    assert list(markers[0].y) == [10 ** charts.RATE_LOG_RANGE[0]]
    assert 10 ** charts.RATE_LOG_RANGE[0] == min(charts.RATE_TICKS)
