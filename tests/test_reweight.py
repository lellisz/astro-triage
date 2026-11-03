"""Specification tests for ``eval.reweight.reweight`` (ADR 0003 test cases 1-7)."""

import numpy as np
import pytest

from eval.reweight import EmptyClassError, reweight

GRID = {
    "P0": {"star": 1 / 3, "qso": 1 / 3, "galaxy": 1 / 3},
    "P1": {"star": 0.997871, "qso": 0.001996, "galaxy": 0.000133},
    "P2": {"star": 0.998813, "qso": 0.000989, "galaxy": 0.000198},
    "P3": {"star": 0.996830, "qso": 0.002640, "galaxy": 0.000530},
}


def test_basic_weights_match_prior_over_class_counts():
    y = ["star", "star", "star", "qso"]
    w = reweight(y, {"star": 0.5, "qso": 0.5})
    assert w.dtype == np.float64
    assert w.shape == (4,)
    np.testing.assert_allclose(w, [1 / 6, 1 / 6, 1 / 6, 1 / 2], atol=1e-12)
    assert abs(w.sum() - 1.0) < 1e-12


def test_numpy_input_is_accepted_and_not_mutated():
    y = np.array(["star", "star", "qso"])
    before = y.copy()
    w = reweight(y, {"star": 0.5, "qso": 0.5})
    np.testing.assert_array_equal(y, before)
    np.testing.assert_allclose(w, [0.25, 0.25, 0.5], atol=1e-12)


def test_empty_class_with_positive_prior_raises_empty_class_error():
    with pytest.raises(EmptyClassError, match="qso"):
        reweight(["star", "star"], {"star": 0.998, "qso": 0.002})


def test_zero_prior_class_gets_zero_weight():
    w = reweight(["star", "galaxy", "qso"], {"star": 0.5, "qso": 0.5, "galaxy": 0.0})
    np.testing.assert_allclose(w, [0.5, 0.0, 0.5], atol=1e-12)


def test_absent_class_with_zero_prior_is_allowed():
    w = reweight(["star", "star"], {"star": 1.0, "qso": 0.0})
    np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-12)


def test_unknown_label_raises_value_error_and_is_not_empty_class_error():
    prior = {"star": 0.9, "qso": 0.05, "galaxy": 0.05}
    with pytest.raises(ValueError) as excinfo:
        reweight(["star", "wd"], prior)
    assert not isinstance(excinfo.value, EmptyClassError)


def test_prior_within_tolerance_is_renormalised():
    w = reweight(
        ["star", "qso"],
        {"star": 0.5, "qso": 0.4999995},  # sums to 0.9999995
    )
    assert abs(w.sum() - 1.0) < 1e-12


def test_prior_summing_to_point_nine_raises_value_error():
    with pytest.raises(ValueError):
        reweight(["star", "qso"], {"star": 0.45, "qso": 0.45})


def test_prior_with_negative_entry_raises_value_error():
    with pytest.raises(ValueError):
        reweight(["star", "qso"], {"star": 1.2, "qso": -0.2})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_prior_with_nan_or_inf_raises_value_error(bad):
    with pytest.raises(ValueError):
        reweight(["star", "qso"], {"star": 0.5, "qso": bad})


def test_empty_y_true_raises_value_error():
    with pytest.raises(ValueError):
        reweight([], {"star": 0.5, "qso": 0.5})


def test_per_class_weight_sums_equal_prior():
    y = np.array(["star"] * 7 + ["qso"] * 2 + ["galaxy"] * 1)
    prior = GRID["P1"]
    w = reweight(y, prior)
    total = sum(prior.values())
    for c, p in prior.items():
        assert w[y == c].sum() == pytest.approx(p / total, abs=1e-12)


def test_balanced_regime_equals_uniform_prior():
    y = ["star", "star", "star", "qso", "galaxy", "galaxy"]
    w = reweight(y, {"star": 1 / 3, "qso": 1 / 3, "galaxy": 1 / 3})
    np.testing.assert_allclose(
        w, [1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 6, 1 / 6], atol=1e-12
    )


@pytest.mark.parametrize("grid_id", sorted(GRID))
def test_weighted_completeness_equals_unweighted_for_every_grid_prior(grid_id):
    y = np.array(["star"] * 6 + ["qso"] * 4 + ["galaxy"] * 5)
    pred = np.array(
        ["star", "star", "star", "star", "qso", "galaxy"]
        + ["qso", "qso", "qso", "star"]
        + ["galaxy", "galaxy", "qso", "star", "galaxy"]
    )
    w = reweight(y, GRID[grid_id])
    for c in ("star", "qso", "galaxy"):
        in_c = y == c
        unweighted = np.mean(pred[in_c] == c)
        weighted = w[in_c & (pred == c)].sum() / w[in_c].sum()
        assert weighted == pytest.approx(unweighted, abs=1e-12)


def test_realistic_qso_purity_collapses_under_star_dominated_prior():
    y = np.array(["star"] * 1000 + ["qso"] * 1000)
    pred = np.array(["qso"] * 10 + ["star"] * 990 + ["qso"] * 950 + ["star"] * 50)
    # Unweighted purity: 950 / 960 (sanity check of the fixture, not of reweight).
    assert np.mean(y[pred == "qso"] == "qso") == pytest.approx(950 / 960)

    w = reweight(y, {"star": 0.998, "qso": 0.002})
    is_pred_qso = pred == "qso"
    purity = w[is_pred_qso & (y == "qso")].sum() / w[is_pred_qso].sum()
    assert purity == pytest.approx(0.1599, abs=1e-4)
