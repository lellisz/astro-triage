"""Unit tests for mix and fit_weight in models/ensemble.py (ADR 0006 section 3); build tests: test_ensemble_build.py."""

from __future__ import annotations

import numpy as np

from astro_triage.models import ensemble as E
from astro_triage.models.lgbm import log_loss


def dirichlet(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet([1.0, 1.0, 1.0], size=n)


def test_mix_rows_sum_to_one_and_convex():
    a, b = dirichlet(1, 20), dirichlet(2, 20)
    m = E.mix(a, b, 0.3)
    assert np.allclose(m.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(m, 0.3 * a + 0.7 * b)
    assert np.allclose(E.mix(a, b, 1.0), a) and np.allclose(E.mix(a, b, 0.0), b)


def test_hand_case_unique_minimum_at_0_7():
    # 7 rows where only lgbm is right (loss log w), 3 where only the net is (log 1-w): argmax 7 log w + 3 log(1-w)
    pl, pn = np.tile([1.0, 0, 0], (10, 1)), np.tile([0, 1.0, 0], (10, 1))
    y = np.array([0] * 7 + [1] * 3)
    assert E.fit_weight(pl, pn, y) == 0.7


def test_identical_inputs_give_one_and_perfect_net_gives_zero():
    p, y = dirichlet(3, 40), np.random.default_rng(3).integers(0, 3, 40)
    assert E.fit_weight(p, p.copy(), y) == 1.0
    assert E.fit_weight(np.full((40, 3), 1 / 3), np.eye(3)[y], y) == 0.0


def test_tie_within_tolerance_goes_to_larger_weight():
    p, y = dirichlet(4, 40), np.random.default_rng(4).integers(0, 3, 40)
    q = p.copy()
    q[np.arange(40), y] += 1e-13  # net slightly better on the true class, by far less than TIE_ABS_TOL

    def loss(w: float) -> float:
        return log_loss(E._log_mix(p, q, w), y)

    gap = loss(1.0) - min(loss(w) for w in E.W_GRID if w < 1.0)
    assert 0.0 < gap < E.TIE_ABS_TOL  # some w < 1 is strictly lower in loss, yet ...
    assert E.fit_weight(p, q, y) == 1.0  # ... the tie goes to the larger weight
    q2 = p.copy()
    q2[np.arange(40), y] += 1e-9  # control: a gap above the tolerance moves the weight
    assert E.fit_weight(p, q2, y) < 1.0

