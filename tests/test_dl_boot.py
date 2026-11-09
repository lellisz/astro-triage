"""dl_boot (S12, ADR 0006 sections 4.2/4.4): purity from class sums, shared replicates, interval, memory."""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest

from astro_triage.evaluation import dl_boot as B
from astro_triage.evaluation.runner import EvalError
from eval import metrics as M

PI = np.array([0.5, 0.3, 0.2])  # STAR, GALAXY, QSO


def test_purity_hand_value():
    n, cnt = np.array([10.0, 5.0, 4.0]), np.array([2.0, 1.0, 2.0])
    w = PI / n
    want = w[2] * 2 / (w @ cnt)
    assert B.purity(n, cnt, PI, 2) == pytest.approx(want)


def test_absent_contaminant_adds_zero_g6():
    n, cnt = np.array([10.0, 0.0, 4.0]), np.array([2.0, 3.0, 2.0])  # GALAXY absent
    w = np.array([0.5 / 10, 0.0, 0.2 / 4])
    assert B.purity(n, cnt, PI, 2) == pytest.approx(w[2] * 2 / (w[0] * 2 + w[2] * 2))
    assert np.isfinite(B.purity(n, cnt, PI, 2))


def test_absent_target_class_and_empty_selection_are_invalid():
    cnt = np.array([2.0, 1.0, 0.0])
    assert math.isnan(B.purity(np.array([10.0, 5.0, 0.0]), cnt, PI, 2))  # n_c = 0
    assert math.isnan(B.purity(np.array([10.0, 5.0, 4.0]), np.zeros(3), PI, 2))


def test_interval_percentiles_and_na_threshold():
    lo, hi, frac = B.interval(np.arange(101, dtype=float))
    assert (lo, hi, frac) == (2.5, 97.5, 0.0)
    d = np.arange(200, dtype=float)
    d[:3] = np.nan  # 1.5 % invalid
    assert math.isnan(B.interval(d)[0])
    assert math.isnan(B.interval(np.full(5, np.nan))[0])


def test_interval_exactly_one_percent_invalid_is_not_na():
    """Spec S12: n/a only when the invalid share is > 1 %; 10 of 1000 is exactly 1 %."""
    for n, k in ((200, 2), (1000, 10)):
        d = np.arange(n, dtype=float)
        d[:k] = np.nan
        assert np.isfinite(B.interval(d)[0]), (n, k, B.interval(d)[2])


def test_seed_mean_diff_propagates_nan():
    a = [np.array([1.0, np.nan]), np.array([3.0, 3.0])]
    r = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    out = B.seed_mean_diff(a, r)
    assert out[0] == 1.0 and math.isnan(out[1])


def test_block_labels_and_class_codes():
    got = B.block_labels(np.array([0, 15, 16, 32]), np.array(["A", "B", "C", "A"]))
    assert got.tolist() == [0, 1, 6, 8]
    assert B.class_codes(np.array(["QSO", "STAR"])).tolist() == [2, 0]
    with pytest.raises(EvalError):
        B.class_codes(np.array(["QSO", "?"]))


def _toy(n=600, seed=1):
    rng = np.random.default_rng(seed)
    pix = rng.integers(0, 60, n).astype(np.int64)
    band = np.array(["A", "B", "C"])[pix % 3]
    code = rng.integers(0, 3, n).astype(np.int64)
    return code, B.block_labels(pix, band), band


def test_replicates_shared_between_systems_and_independent_of_their_number(monkeypatch):
    code, block, band = _toy()
    sel = B.make_selection(code, code >= 1)
    other = B.make_selection(code, code == 2)
    pri = {"P2": PI}
    draws = []
    orig = M.bootstrap_pixel_draw
    monkeypatch.setattr(
        M,
        "bootstrap_pixel_draw",
        lambda p, b, r: draws.append(orig(p, b, r)) or draws[-1],
    )
    two = B.replicate_purities(
        code, block, band, {"a": (sel, 2), "b": (other, 2)}, pri, 6, 7
    )
    assert len(draws) == 6  # one draw per replicate, whatever the number of systems
    rng = np.random.default_rng(7)
    for got in draws:
        assert np.array_equal(got, orig(block, band, rng))
    one = B.replicate_purities(code, block, band, {"b": (other, 2)}, pri, 6, 7)
    assert np.array_equal(two[("b", "P2")], one[("b", "P2")], equal_nan=True)
    same = B.replicate_purities(
        code, block, band, {"a": (sel, 2), "c": (sel, 2)}, pri, 6, 7
    )
    assert np.array_equal(same[("a", "P2")], same[("c", "P2")], equal_nan=True)


def test_no_b_by_n_matrix_peak_memory():
    n, n_boot = 500_000, 40  # a B x n float matrix would be 160 MB on its own
    rng = np.random.default_rng(0)
    pix = rng.integers(0, 3000, n).astype(np.int64)
    band = np.array(["A", "B", "C"])[pix % 3]
    code = rng.integers(0, 3, n).astype(np.int64)
    sel = B.make_selection(code, rng.random(n) < 0.1)
    tracemalloc.start()
    out = B.replicate_purities(
        code, B.block_labels(pix, band), band, {"s": (sel, 2)}, {"P2": PI}, n_boot, 3
    )
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert np.isfinite(out[("s", "P2")]).all()
    assert peak < 90 * 2**20, peak / 2**20
