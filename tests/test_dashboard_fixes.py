"""Dashboard phase-1 fixes: floor sentence, paired ticks, calibration legend, limits, thresholds."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astro_triage.dashboard import charts, limits_pt

ROOT = Path(__file__).resolve().parents[1]


# 2. paired_dtick / paired_diffs
@pytest.mark.parametrize("span", [1e-9, 0.0005, 0.003, 0.0081, 0.04, 0.1, 0.4, 5.0])
def test_paired_dtick_valid_and_bounded(span):
    d = charts.paired_dtick(-span / 2, span / 2)
    assert d >= 0.001
    assert d in charts._DTICKS
    if span / charts._DTICKS[-1] <= 8:
        assert span / d <= 8 + 1e-9


@pytest.mark.parametrize(
    "lo,hi", [(0.0, 0.0), (1.0, -1.0), (math.nan, 1.0), (0.0, math.nan)]
)
def test_paired_dtick_degenerate_span(lo, hi):
    assert charts.paired_dtick(lo, hi) == 0.001


def test_paired_diffs_ticks_distinct_with_three_decimals():
    diffs = pd.DataFrame(
        {
            "pair": ["A-B", "A-C"],
            "point": [0.0012, -0.004],
            "lo": [-0.0003, -0.009],
            "hi": [0.0031, 0.0009],
        }
    )
    fig = charts.paired_diffs(diffs)
    xa = fig.layout.xaxis
    assert xa.tickformat == "+.3f"
    lo, hi = min(0.0, diffs["lo"].min()), max(0.0, diffs["hi"].max())
    ticks = np.arange(xa.tick0, hi + xa.dtick, xa.dtick)
    ticks = np.concatenate(
        [-np.arange(xa.dtick, -lo + xa.dtick, xa.dtick)[::-1], ticks]
    )
    labels = [format(t, "+.3f") for t in ticks]
    assert len(labels) == len(set(labels))


# 3. calibration
def _calib(raw: bool) -> dict[str, pd.DataFrame]:
    base = {"mean_pred": [0.1, 0.5, 0.9], "observed": [0.12, 0.45, 0.88]}
    if raw:
        base |= {"raw_mean_pred": [0.05, 0.4, 0.95], "raw_observed": [0.12, 0.45, 0.88]}
    return {c: pd.DataFrame(base) for c in ("STAR", "GALAXY", "QSO")}


def test_calibration_series_names():
    fig = charts.calibration(_calib(True))
    names = {t.name for t in fig.data}
    assert all(t.name and "trace" not in t.name.lower() for t in fig.data)
    assert {"Antes da calibração", "Depois (temperatura)", "Perfeita"} <= names


def test_calibration_without_raw_columns_drops_before_series():
    fig = charts.calibration(_calib(False))
    names = {t.name for t in fig.data}
    assert "Antes da calibração" not in names
    assert {"Depois (temperatura)", "Perfeita"} <= names
    assert all(t.name and "trace" not in t.name.lower() for t in fig.data)


# 4. limits_pt
def test_limits_keys_are_nonempty_strings():
    assert limits_pt.LIMITS_PT
    assert all(isinstance(k, str) and k.strip() for k in limits_pt.LIMITS_PT)


def test_translate_known_and_unknown():
    key = next(iter(limits_pt.LIMITS_PT))
    assert limits_pt.translate(key) == (limits_pt.LIMITS_PT[key], True)
    assert limits_pt.translate("never seen text") == ("never seen text", False)


@pytest.mark.parametrize("release", ["full_v0", "dataset_100k_v0"])
def test_every_limitation_in_results_json_is_translated(release):
    path = ROOT / "data" / "interim" / release / "eval" / "results.json"
    if not path.exists():
        pytest.skip(f"{path} missing")
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh).get("limitations", [])
    missing = [t[:80] for t in items if not limits_pt.translate(t)[1]]
    assert not missing, missing
