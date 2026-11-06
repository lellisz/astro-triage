"""Phase 5 galaxies: Wilson interval, low-z band table and its sentence (synthetic data)."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from astro_triage.dashboard import data_galaxies as dg
from astro_triage.dashboard import data_galaxies_z as dz
from astro_triage.dashboard.fmt import CI_LEVEL


def _rel(zs, errors=None, extra_other=2):
    """Galaxy rows with the given z values, plus non-galaxy rows that must be ignored."""
    errors = errors if errors is not None else [False] * len(zs)
    rows = [
        {"true": "GALAXY", "sdss_z": z, "err_m": e, "pred_m": "QSO" if e else "GALAXY"}
        for z, e in zip(zs, errors, strict=True)
    ]
    rows += [
        {"true": "STAR", "sdss_z": 0.0, "err_m": True, "pred_m": "QSO"}
    ] * extra_other
    return SimpleNamespace(test=pd.DataFrame(rows))


# (1) Wilson
def test_wilson_zero_errors_known_values():
    lo, hi = dg.wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)  # float noise, not exactly 0
    assert hi == pytest.approx(0.2775, abs=2e-4)


def test_wilson_all_errors_known_values():
    lo, hi = dg.wilson_interval(10, 10)
    assert hi == 1.0
    assert lo == pytest.approx(0.7225, abs=2e-4)


def test_wilson_half_is_symmetric_around_half():
    lo, hi = dg.wilson_interval(5, 10)
    assert lo + hi == pytest.approx(1.0)
    assert lo == pytest.approx(0.2366, abs=2e-4)


def test_wilson_empty_sample_is_nan_without_exception():
    lo, hi = dg.wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


@pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (3, 7), (1, 1000), (999, 1000)])
def test_wilson_brackets_the_point_estimate(k, n):
    lo, hi = dg.wilson_interval(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_z_ci_derives_from_ci_level():
    assert CI_LEVEL == 0.95
    assert dg.Z_CI == pytest.approx(1.95996, abs=1e-4)


def test_wider_z_gives_wider_interval():
    narrow = dg.wilson_interval(5, 50, 1.0)
    wide = dg.wilson_interval(5, 50, 2.58)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


# (2) table by z
ZS = [-0.5, -0.001, 0.0, 1e-3, 0.0010001, 0.04, 0.05, 0.3, 2.0, math.nan]


def test_table_starts_with_low_band_and_has_literal_label():
    tab = dz.error_rate_table_z(_rel(ZS), "m")
    assert tab["label"].iloc[0] == "z ≤ 0,001"


def test_table_partitions_without_double_counting():
    rel = _rel(ZS)
    tab = dz.error_rate_table_z(rel, "m")
    assert tab["n"].sum() == len(ZS)  # non-galaxy rows excluded, NaN counted once
    assert tab["errors"].sum() == 0


def test_table_low_band_includes_negatives_and_boundary_only():
    tab = dz.error_rate_table_z(_rel(ZS), "m").set_index("label")
    # -0.5, -0.001, 0.0 and z == Z_LOW are inside; 0.0010001 is outside
    assert tab.loc["z ≤ 0,001", "n"] == 4


def test_table_value_just_above_z_low_is_not_in_low_band():
    tab = dz.error_rate_table_z(_rel([dz.Z_LOW, dz.Z_LOW * 1.0001]), "m")
    assert tab["n"].iloc[0] == 1
    assert tab["n"].iloc[1:].sum() == 1


def test_table_nan_goes_to_sem_medida_only():
    tab = dz.error_rate_table_z(_rel([0.2, math.nan, math.nan]), "m")
    row = tab[tab["label"] == "sem medida"]
    assert len(row) == 1 and int(row["n"].iloc[0]) == 2
    assert tab["n"].iloc[0] == 0


def test_table_without_nan_has_no_sem_medida_row():
    tab = dz.error_rate_table_z(_rel([0.2, 0.3]), "m")
    assert "sem medida" not in set(tab["label"])


def test_table_counts_errors_per_band():
    rel = _rel([0.0, 0.0, 0.2], [True, False, True])
    tab = dz.error_rate_table_z(rel, "m")
    assert (tab["n"].iloc[0], tab["errors"].iloc[0]) == (2, 1)
    assert tab["errors"].sum() == 2


# (3) sentence
def _summary(zs, errs):
    return dz.z_low_summary(_rel(zs, errs), "m")


def test_summary_counts_n_negatives_rate():
    s = _summary([-1.0, -0.2, 0.0, 0.5], [True, False, False, True])
    assert (s.n, s.negatives, s.errors) == (3, 2, 1)
    assert s.rate == pytest.approx(1 / 3)
    assert s.share_of_errors == pytest.approx(0.5)
    assert s.share_of_galaxies == pytest.approx(0.75)


def test_summary_empty_band_is_nan_not_exception():
    s = _summary([0.5, 0.6], [True, False])
    assert s.n == 0 and math.isnan(s.rate)
    assert "PROVISÓRIO" not in dz.z_low_sentence(s)


def test_sentence_states_n_negatives_and_pt_rate():
    s = _summary([-1.0, 0.0, 0.0, 0.0, 0.5], [True, False, False, False, False])
    text = dz.z_low_sentence(s)
    assert "4 galáxias" in text
    assert "inclui 1 com z negativo" in text
    assert "25,0%" in text and "25.0" not in text


def test_sentence_without_negatives_omits_parenthesis():
    text = dz.z_low_sentence(_summary([0.0, 0.0], [True, False]))
    assert "(inclui" not in text and "negativo" not in text
    assert "z ≤ 0,001:" in text


def test_sentence_has_no_status_seal_and_avoids_cause_claims():
    text = dz.z_low_sentence(_summary([-1.0, 0.0], [True, False]))
    assert "PROVISÓRIO" not in text
    assert not re.search(r"parcela grande|estrela|ruído", text, re.IGNORECASE)
