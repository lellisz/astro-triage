"""Hand-computed proofs for ``eval.metrics`` (ADR 0003, D4).

Class order is STAR, GALAXY, QSO; confusion matrices are rows = true, columns = predicted.
sklearn is used only as a cross-check, never as the sole oracle.
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from eval import metrics as M
from eval.metrics import CLASSES
from eval.reweight import EmptyClassError, reweight

S, G, Q = "STAR", "GALAXY", "QSO"
GRID = {
    "P0": {S: 1 / 3, G: 1 / 3, Q: 1 / 3},
    "P1": {S: 0.997871, Q: 0.001996, G: 0.000133},
    "P2": {S: 0.998813, Q: 0.000989, G: 0.000198},
    "P3": {S: 0.996830, Q: 0.002640, G: 0.000530},
}

# Confusion of 12 objects (rows true, columns predicted: STAR, GALAXY, QSO):
#   STAR   : 4 1 1   (6)
#   GALAXY : 1 2 0   (3)
#   QSO    : 0 1 2   (3)
CM12 = np.array([[4, 1, 1], [1, 2, 0], [0, 1, 2]], dtype=float)


def make12() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt, yp = [], []
    for i, ct in enumerate(CLASSES):
        for j, cp in enumerate(CLASSES):
            yt += [ct] * int(CM12[i, j])
            yp += [cp] * int(CM12[i, j])
    return np.array(yt), np.array(yp), np.arange(12)


def onehot_proba(pred: np.ndarray, conf: float = 0.7) -> np.ndarray:
    p = np.full((len(pred), 3), (1 - conf) / 2)
    for i, c in enumerate(pred):
        p[i, CLASSES.index(c)] = conf
    return p


# ------------------------------------------------------------------ (1) confusion etc.


def test_confusion_matrix_12_objects():
    yt, yp, _ = make12()
    assert np.array_equal(M.confusion_matrix(yt, yp), CM12)


def test_confusion_matrix_weighted():
    # Prior STAR .9, GALAXY .05, QSO .05 -> w = .9/6 = .15 (STAR), .05/3 = 1/60 (others).
    # Row STAR = [.6, .15, .15]; row GALAXY = [1, 2, 0]/60; row QSO = [0, 1, 2]/60.
    yt, yp, _ = make12()
    w = reweight(yt, {S: 0.9, G: 0.05, Q: 0.05})
    cm = M.confusion_matrix(yt, yp, w)
    expected = np.array([[0.6, 0.15, 0.15], [1 / 60, 2 / 60, 0], [0, 1 / 60, 2 / 60]])
    assert np.allclose(cm, expected, atol=1e-12)
    assert np.isclose(cm.sum(), 1.0, atol=1e-12)


def test_class_metrics_unweighted_hand_values():
    # Column sums = predicted counts: STAR 5, GALAXY 4, QSO 3.
    # purity: STAR 4/5, GALAXY 2/4, QSO 2/3; completeness: 4/6, 2/3, 2/3.
    # F1 = 2TP/(2TP+FP+FN): STAR 8/(8+1+2) = 8/11, GALAXY 4/(4+2+1) = 4/7, QSO 4/(4+1+1) = 2/3.
    yt, yp, pix = make12()
    r = M.class_metrics(yt, yp, pix, None, min_objects=1, min_pixels=1)
    assert r[S]["purity"] == pytest.approx(4 / 5)
    assert r[G]["purity"] == pytest.approx(1 / 2)
    assert r[Q]["purity"] == pytest.approx(2 / 3)
    for c in CLASSES:
        assert r[c]["completeness"] == pytest.approx(4 / 6 if c == S else 2 / 3)
    assert r[S]["f1"] == pytest.approx(8 / 11)
    assert r[G]["f1"] == pytest.approx(4 / 7)
    assert r[Q]["f1"] == pytest.approx(2 / 3)
    assert M.macro_f1(r) == pytest.approx((8 / 11 + 4 / 7 + 2 / 3) / 3)
    # no weights -> never flagged low-count (flag is a REALISTIC-regime concept)
    assert all(r[c]["purity_flag"] is None for c in CLASSES)


def test_class_metrics_f1_matches_sklearn_cross_check():
    yt, yp, pix = make12()
    r = M.class_metrics(yt, yp, pix, None, min_objects=1, min_pixels=1)
    sk = f1_score(yt, yp, labels=list(CLASSES), average=None)
    assert [r[c]["f1"] for c in CLASSES] == pytest.approx(list(sk))
    w = reweight(yt, {S: 0.9, G: 0.05, Q: 0.05})
    rw = M.class_metrics(yt, yp, pix, w, min_objects=1, min_pixels=1)
    skw = f1_score(yt, yp, labels=list(CLASSES), average=None, sample_weight=w)
    assert [rw[c]["f1"] for c in CLASSES] == pytest.approx(list(skw))


def test_realistic_purity_hand_values_and_flags():
    # Weighted columns (see test_confusion_matrix_weighted):
    #   STAR col = .6 + 1/60 + 0       -> purity .6 / (.6 + 1/60) = 36/37
    #   GALAXY col = .15 + 2/60 + 1/60 = .2   -> purity (1/30) / .2 = 1/6
    #   QSO col = .15 + 0 + 2/60 = 11/60      -> purity (1/30) / (11/60) = 2/11
    yt, yp, pix = make12()
    w = reweight(yt, {S: 0.9, G: 0.05, Q: 0.05})
    r = M.class_metrics(
        yt, yp, pix, w, min_objects=1, min_pixels=1, flag_low_count=True
    )
    assert r[S]["purity"] == pytest.approx(36 / 37)
    assert r[G]["purity"] == pytest.approx(1 / 6)
    assert r[Q]["purity"] == pytest.approx(2 / 11)
    assert r[G]["f1"] == pytest.approx(2 * (1 / 6) * (2 / 3) / (1 / 6 + 2 / 3))
    # Raw contaminant counts into GALAXY: 1 STAR, 1 QSO -> total 2 < 10 -> low-count.
    assert r[G]["contaminants"] == {S: 1, Q: 1}
    assert r[G]["purity_flag"] == "low-count"
    # Into STAR: 1 GALAXY, 0 QSO. Zero count -> rate is the rule-of-three bound 3/n_QSO = 1.
    assert r[S]["contaminants"] == {G: 1, Q: 0}
    assert r[S]["contamination"][Q] == {"count": 0, "rate": 1.0, "upper_bound": True}
    assert r[S]["contamination"][G] == {"count": 1, "rate": 1 / 3, "upper_bound": False}


def _into_galaxy(n_star: int, n_qso: int, n_qso_total: int = 20):
    """20 STAR (n_star predicted GALAXY), n_qso_total QSO (n_qso predicted GALAXY), 20 GALAXY."""
    yt = [S] * 20 + [Q] * n_qso_total + [G] * 20
    yp = [G] * n_star + [S] * (20 - n_star)
    yp += [G] * n_qso + [Q] * (n_qso_total - n_qso) + [G] * 20
    return np.array(yt), np.array(yp)


def _galaxy_entry(
    n_star: int, n_qso: int, n_qso_total: int = 20, flag_low_count: bool = True
) -> dict:
    yt, yp = _into_galaxy(n_star, n_qso, n_qso_total)
    prior = {S: 0.4, G: 0.2, Q: 0.4} if n_qso_total else {S: 0.5, G: 0.5, Q: 0.0}
    w = reweight(yt, prior)
    r = M.class_metrics(
        yt,
        yp,
        np.arange(len(yt)),
        w,
        min_objects=1,
        min_pixels=1,
        flag_low_count=flag_low_count,
    )
    return r[G]


def test_low_count_flag_only_in_realistic_regime():
    # Same few contaminants (total 2 < 10): the caller states the regime.
    yt, yp = _into_galaxy(1, 1)
    pix = np.arange(len(yt))
    # BALANCED: uniform weights, flag off (default) -> never flagged.
    wb = M.regime_weights(yt)
    rb = M.class_metrics(yt, yp, pix, wb, min_objects=1, min_pixels=1)
    assert rb[G]["contaminants"] == {S: 1, Q: 1}
    assert rb[G]["purity_flag"] is None
    assert _galaxy_entry(1, 1, flag_low_count=False)["purity_flag"] is None
    # REALISTIC: flag on -> flagged.
    wr = M.regime_weights(yt, {S: 0.4, G: 0.2, Q: 0.4})
    rr = M.class_metrics(
        yt, yp, pix, wr, min_objects=1, min_pixels=1, flag_low_count=True
    )
    assert rr[G]["purity_flag"] == "low-count"
    # Flag on but contaminant total >= 10: not flagged.
    assert _galaxy_entry(6, 4)["purity_flag"] is None


def test_low_count_uses_total_contaminants():
    e = _galaxy_entry(5, 4)  # total 9 across two classes -> flagged
    assert e["contaminants"] == {S: 5, Q: 4} and e["purity_flag"] == "low-count"
    e = _galaxy_entry(6, 4)  # total 10 is NOT < 10
    assert e["contaminants"] == {S: 6, Q: 4} and e["purity_flag"] is None
    # Contaminant class QSO absent from the test, total 12 >= 10: no flag.
    e = _galaxy_entry(12, 0, 0)
    assert e["contaminants"] == {S: 12, Q: 0} and e["purity_flag"] is None
    # Same with total 9: flagged by the sum, not by the absent class.
    assert _galaxy_entry(9, 0, 0)["purity_flag"] == "low-count"
    # Total 0: flagged; rule-of-three bound only for the contaminant class with n_k > 0.
    e = _galaxy_entry(0, 0, 0)
    assert e["purity_flag"] == "low-count"
    assert e["contamination"][S] == {"count": 0, "rate": 3 / 20, "upper_bound": True}
    assert np.isnan(e["contamination"][Q]["rate"])
    assert e["contamination"][Q]["upper_bound"] is False


def test_adr_case7_regime_effect_and_no_low_count_at_10():
    # ADR 0003 test case 7: 1000 stars (10 predicted QSO), 1000 QSO (950 predicted QSO).
    # Unweighted QSO purity 950/960; prior STAR .998, QSO .002 -> .0019 / (.0019 + .00998).
    yt = np.array([S] * 1000 + [Q] * 1000)
    yp = np.array([S] * 990 + [Q] * 10 + [Q] * 950 + [S] * 50)
    pix = np.arange(2000) % 50
    r0 = M.class_metrics(yt, yp, pix, None)
    assert r0[Q]["purity"] == pytest.approx(950 / 960)
    w = reweight(yt, {S: 0.998, G: 0.0, Q: 0.002})
    r1 = M.class_metrics(yt, yp, pix, w, flag_low_count=True)
    assert r1[Q]["purity"] == pytest.approx(0.1599, abs=1e-4)
    # completeness is identical across regimes
    assert r1[Q]["completeness"] == pytest.approx(r0[Q]["completeness"])
    assert r1[Q]["contaminants"][S] == 10  # exactly 10 is NOT < 10


def test_completeness_invariant_under_reweighting_and_weights_sum_to_one():
    yt, yp, pix = make12()
    base = M.class_metrics(yt, yp, pix, None, min_objects=1, min_pixels=1)
    for prior in GRID.values():
        w = M.regime_weights(yt, prior)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        r = M.class_metrics(yt, yp, pix, w, min_objects=1, min_pixels=1)
        for c in CLASSES:
            assert r[c]["completeness"] == pytest.approx(
                base[c]["completeness"], abs=1e-12
            )
    wb = M.regime_weights(yt)  # BALANCED == uniform prior
    assert np.allclose(wb, reweight(yt, M.uniform_prior()))
    assert wb.sum() == pytest.approx(1.0, abs=1e-12)


def test_class_metrics_edges_absent_class_and_floor():
    # absent class: no exception, nan with reason below_floor
    yt = np.array([S, S, Q])
    r = M.class_metrics(yt, yt, np.array([0, 1, 2]), None, min_objects=1, min_pixels=1)
    assert r[G]["n"] == 0 and r[G]["status"] == "below_floor"
    assert np.isnan(r[G]["purity"]) and r[G]["reason"] == "below_floor"
    assert r[S]["purity"] == 1.0
    # Zero contaminants into STAR: n_QSO = 1 -> bound 3/1; GALAXY absent -> rate nan.
    assert r[S]["contamination"][Q]["upper_bound"] is True
    assert r[S]["contamination"][Q]["rate"] == pytest.approx(3.0)
    assert np.isnan(r[S]["contamination"][G]["rate"])

    # floor boundary: 100 objects in 20 pixels passes; 99 objects or 19 pixels fails.
    def build(n: int, npix: int):
        y = np.array([c for c in CLASSES for _ in range(n)])
        pix = np.concatenate([np.arange(n) % npix for _ in CLASSES])
        return y, pix

    y, pix = build(100, 20)
    assert all(M.class_metrics(y, y, pix)[c]["status"] == "ok" for c in CLASSES)
    y, pix = build(100, 19)
    assert all(
        M.class_metrics(y, y, pix)[c]["status"] == "below_floor" for c in CLASSES
    )
    y, pix = build(99, 20)
    assert all(np.isnan(M.class_metrics(y, y, pix)[c]["f1"]) for c in CLASSES)


def test_no_predictions_gives_undefined_purity():
    yt = np.array([S, G, Q])
    yp = np.array([S, S, S])  # nothing predicted GALAXY or QSO
    r = M.class_metrics(yt, yp, np.arange(3), None, min_objects=1, min_pixels=1)
    assert np.isnan(r[G]["purity"]) and np.isnan(r[G]["f1"])
    assert r[G]["completeness"] == 0.0


def test_regime_weights_absent_class_raises():
    with pytest.raises(EmptyClassError):
        M.regime_weights(np.array([S, Q]))


def test_predict_argmax():
    p = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1]])
    assert M.predict_argmax(p).tolist() == [Q, S, G]


# --------------------------------------------------------------------------- (2) PR-AUC

# 4 objects, QSO is the class under test; true labels [QSO, STAR, QSO, GALAXY] and QSO scores
# [.9, .8, .7, .1]. Other columns split the remaining mass evenly.
Y4 = np.array([Q, S, Q, G])
Q4 = np.array([0.9, 0.8, 0.7, 0.1])
P4 = np.column_stack([(1 - Q4) / 2, (1 - Q4) / 2, Q4])


def test_pr_auc_hand_value_unweighted():
    # Thresholds .9: P=1, R=.5; .8: P=.5, R=.5; .7: P=2/3, R=1.
    # AP = sum (R_n - R_{n-1}) P_n = .5*1 + 0*.5 + .5*(2/3) = 5/6.
    assert M.pr_auc(Y4, P4)[Q].value == pytest.approx(5 / 6)


def test_pr_auc_hand_value_weighted():
    # w = [1, 2, 1, 1]: positives weigh 1 + 1 = 2.
    # .9: tp=1, fp=0 -> P=1, R=.5; .8: fp=2 -> P=1/3, R=.5; .7: tp=2, fp=2 -> P=.5, R=1.
    # AP = .5*1 + .5*.5 = .75 (differs from the unweighted 5/6).
    w = np.array([1.0, 2.0, 1.0, 1.0])
    assert M.pr_auc(Y4, P4, w)[Q].value == pytest.approx(0.75)


def test_pr_auc_edges_absent_and_floor():
    y = np.array([S, S, Q])
    p = np.array([[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8]])
    r = M.pr_auc(y, p)
    assert np.isnan(r[G].value) and r[G].reason == "absent_class"
    assert r[Q].value == pytest.approx(1.0)  # perfect ranking
    rf = M.pr_auc(y, p, pixel=np.arange(3))  # default floor: n >= 100 fails
    assert rf[Q].reason == "below_floor" and np.isnan(rf[Q].value)


def test_pr_auc_weight_scale_invariance():
    w = np.array([1.0, 2.0, 1.0, 1.0])
    a = M.pr_auc(Y4, P4, w)[Q].value
    b = M.pr_auc(Y4, P4, w / w.sum())[Q].value
    assert a == pytest.approx(b)


# ------------------------------------------------------------------------------ (3) ECE


def test_ece_equal_width_hand_value_unweighted():
    # 2 equal-width bins: [0, .5) = {.1 (outcome 0)}, [.5, 1] = {.9, .8, .7 (outcomes 1, 0, 1)}.
    # bin 0: conf .1, freq 0, gap .1, share 1/4. bin 1: conf .8, freq 2/3, gap 2/15, share 3/4.
    # ECE = .025 + .1 = .125.
    value, curve = M.ece(Y4, P4, n_bins=2, binning="equal_width")[Q]
    assert value.value == pytest.approx(0.125)
    assert curve["count"].tolist() == [1, 3]
    assert curve["mean_conf"] == pytest.approx([0.1, 0.8])
    assert curve["freq"] == pytest.approx([0.0, 2 / 3])


def test_ece_equal_width_hand_value_weighted():
    # w = [1, 2, 1, 1] on objects [.9, .8, .7, .1]. bin 0: weight 1, conf .1, freq 0.
    # bin 1: weights 1, 2, 1 (total 4): conf (.9 + 1.6 + .7)/4 = .8, freq (1 + 0 + 1)/4 = .5.
    # ECE = (1/5)*.1 + (4/5)*.3 = .26.
    w = np.array([1.0, 2.0, 1.0, 1.0])
    value, curve = M.ece(Y4, P4, w, n_bins=2, binning="equal_width")[Q]
    assert value.value == pytest.approx(0.26)
    assert curve["freq"] == pytest.approx([0.0, 0.5])
    assert curve["count"].tolist() == [1, 3]  # counts stay raw


def test_ece_equal_mass_hand_value_unweighted():
    # Q-class confidences [.9, .8, .7, .1], outcomes [1, 0, 1, 0]; 2 equal-mass bins.
    # Sorted: .1, .7, .8, .9 (mass 1 each, W = 4); mid-masses .5, 1.5, 2.5, 3.5 -> /4*2 =
    # .25, .75, 1.25, 1.75 -> bins 0, 0, 1, 1.
    # bin 0 {.1 (0), .7 (1)}: conf .4, freq .5, gap .1, share 1/2.
    # bin 1 {.8 (0), .9 (1)}: conf .85, freq .5, gap .35, share 1/2.
    # ECE = .05 + .175 = .225.
    value, curve = M.ece(Y4, P4, n_bins=2)[Q]
    assert curve["count"].tolist() == [2, 2]
    assert curve["mean_conf"] == pytest.approx([0.4, 0.85])
    assert curve["freq"] == pytest.approx([0.5, 0.5])
    assert value.value == pytest.approx(0.225)


def test_ece_equal_mass_hand_value_weighted():
    # w = [1, 2, 1, 1] on objects [.9, .8, .7, .1]. Sorted by conf: .1 (w 1), .7 (1), .8 (2),
    # .9 (1); W = 5. Mid-masses .5, 1.5, 3, 4.5 -> /5*2 = .2, .6, 1.2, 1.8 -> bins 0, 0, 1, 1.
    # bin 0 weight 2: conf (.1 + .7)/2 = .4, freq (0 + 1)/2 = .5, gap .1.
    # bin 1 weight 3: conf (1.6 + .9)/3 = 5/6, freq (0*2 + 1)/3 = 1/3, gap .5.
    # ECE = (2/5)*.1 + (3/5)*.5 = .34.
    w = np.array([1.0, 2.0, 1.0, 1.0])
    value, curve = M.ece(Y4, P4, w, n_bins=2)[Q]
    assert curve["weight"] == pytest.approx([2.0, 3.0])
    assert curve["mean_conf"] == pytest.approx([0.4, 5 / 6])
    assert curve["freq"] == pytest.approx([0.5, 1 / 3])
    assert value.value == pytest.approx(0.34)


def test_ece_equal_mass_ties_are_not_split_and_cause_no_nan():
    # conf .2 x6 (outcomes 1, 0, 0, 0, 0, 0) and .8 x2 (outcomes 1, 1); 2 bins, W = 8.
    # Tied groups: .2 mid-mass 3 -> 3/8*2 = .75 -> bin 0; .8 mid-mass 7 -> 1.75 -> bin 1.
    # bin 0: conf .2, freq 1/6, gap 1/30, share 6/8 -> .025; bin 1: conf .8, freq 1, gap .2,
    # share 2/8 -> .05. ECE = .075. Ties stay together even though 6 != 8/2.
    y = np.array([Q, S, S, S, S, S, Q, Q])
    conf = np.array([0.2] * 6 + [0.8] * 2)
    proba = np.column_stack([np.zeros(8), np.zeros(8), conf])
    value, curve = M.ece(y, proba, n_bins=2)[Q]
    assert curve["count"].tolist() == [6, 2]
    assert value.value == pytest.approx(0.075)
    # All confidences identical: one populated bin, the others empty (nan, count 0).
    proba_tied = np.column_stack([np.zeros(8), np.zeros(8), np.full(8, 0.5)])
    value, curve = M.ece(y, proba_tied, n_bins=4)[Q]
    assert curve["count"].sum() == 8 and (curve["count"] > 0).sum() == 1
    assert np.isnan(curve["freq"]).sum() == 3
    assert value.value == pytest.approx(abs(3 / 8 - 0.5))  # freq 3/8, conf .5
    # every distinct confidence lands in exactly one bin (no value split across bins)
    c = np.random.default_rng(0).integers(0, 5, 200) / 5.0
    idx = M._bin_index(c, np.ones(200), 15, "equal_mass")
    assert all(np.unique(idx[c == v]).size == 1 for v in np.unique(c))


def test_ece_equal_mass_bins_have_equal_mass():
    rng = np.random.default_rng(1)
    n, n_bins = 1000, 15
    conf = rng.random(n)
    outcome = rng.random(n) < conf
    curve = M.reliability_curve(conf, outcome, n_bins=n_bins)
    assert curve["count"].sum() == n
    assert curve["count"].max() - curve["count"].min() <= 1  # 66 or 67 each
    w = rng.random(n) + 0.5  # weights in [.5, 1.5)
    curve_w = M.reliability_curve(conf, outcome, w, n_bins)
    target = w.sum() / n_bins
    assert np.all(np.abs(curve_w["weight"] - target) < w.max())  # within one object
    assert curve_w["weight"].sum() == pytest.approx(w.sum())


def test_ece_invalid_binning_raises():
    with pytest.raises(ValueError):
        M.ece(Y4, P4, binning="quantile")


def test_ece_bin_edges_and_perfect_calibration():
    # Equal width: 0.5 is in the upper bin (edges [lo, hi)), 1.0 in the last, 0.0 in the first.
    curve = M.reliability_curve(
        np.array([0.0, 0.5, 1.0]), np.array([0, 1, 1]), n_bins=2, binning="equal_width"
    )
    assert curve["count"].tolist() == [1, 2]
    # Perfect calibration: 10 objects at conf .7 with 7 positives -> ECE 0, default 15 bins.
    n = 10
    y = np.array([Q] * 7 + [S] * 3)
    q = np.full(n, 0.7)
    proba = np.column_stack([np.full(n, 0.2), np.full(n, 0.1), q])
    assert M.ece(y, proba)[Q][0].value == pytest.approx(0.0, abs=1e-12)
    assert len(M.ece(y, proba)[Q][1]["count"]) == 15


def test_ece_edges_absent_class_and_floor():
    r = M.ece(Y4, P4, n_bins=2)
    assert r[S][0].reason is None and np.isfinite(r[S][0].value)  # STAR is present
    y = np.array([S, S, Q])
    p = np.array([[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8]])
    assert M.ece(y, p)[G][0].reason == "absent_class"
    assert M.ece(y, p, pixel=np.arange(3))[Q][0].reason == "below_floor"


def test_ece_zero_prior_class_is_absent_like_pr_auc():
    # GALAXY is present (2 objects) but its prior is 0 -> total weight 0 -> absent_class.
    y = np.array([S, S, G, G, Q, Q])
    p = np.array(
        [[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1],
         [0.3, 0.5, 0.2], [0.1, 0.1, 0.8], [0.2, 0.2, 0.6]]
    )  # fmt: skip
    w = M.regime_weights(y, {S: 0.5, G: 0.0, Q: 0.5})
    assert w[y == G].sum() == 0
    r = M.ece(y, p, w, n_bins=2)
    assert np.isnan(r[G][0].value) and r[G][0].reason == "absent_class"
    assert M.pr_auc(y, p, w)[G].reason == "absent_class"  # same rule as pr_auc
    assert r[S][0].reason is None and np.isfinite(r[S][0].value)
    # Without the zero-weight the same class is evaluated normally.
    assert M.ece(y, p, n_bins=2)[G][0].reason is None


# ---------------------------------------------------------------------- (5) bootstrap


def test_pixel_draw_stays_within_band_and_is_deterministic():
    pix = np.arange(6)
    band = np.array([0, 0, 0, 1, 1, 1])
    idx = M.bootstrap_pixel_draw(pix, band, np.random.default_rng(1))
    assert len(idx) == 6
    assert (band[idx] == 0).sum() == 3 and (band[idx] == 1).sum() == 3
    idx2 = M.bootstrap_pixel_draw(pix, band, np.random.default_rng(1))
    assert np.array_equal(idx, idx2)


def test_pixel_draw_keeps_whole_pixels_together():
    # Objects 0 and 1 share pixel 0: they are always drawn together (same multiplicity).
    pix = np.array([0, 0, 1, 2])
    band = np.zeros(4, dtype=int)
    for seed in range(20):
        idx = M.bootstrap_pixel_draw(pix, band, np.random.default_rng(seed))
        assert (idx == 0).sum() == (idx == 1).sum()


def test_replicate_weights_are_recomputed():
    # Objects: o0 STAR->STAR, o1 STAR->QSO, o2 QSO->QSO. Replicate duplicates o2: idx = [0, 1, 2, 2].
    # Resampled n: STAR 2, QSO 2. Prior STAR .5, QSO .5 (GALAXY 0) -> every object weighs .25.
    # QSO purity = (2 * .25) / (2 * .25 + .25) = 2/3.
    # (Stale weights from the full sample would give w_QSO = .5 per object, w_STAR = .25:
    #  purity = 1.0 / (1.0 + .25) = 0.8.)
    y = np.array([S, S, Q])
    proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
    prior = {S: 0.5, G: 0.0, Q: 0.5}
    res = M.replicate_metrics(y, proba, np.array([0, 1, 2, 2]), prior)
    assert res is not None
    assert res["purity"][Q] == pytest.approx(2 / 3)
    assert res["completeness"][Q] == pytest.approx(1.0)
    # class with positive prior absent from the replicate -> invalid (None)
    assert M.replicate_metrics(y, proba, np.array([0, 0]), prior) is None


def _dataset(n_pix: int, errors_mod: int | None):
    """n_pix pixels (half in band 0, half in band 1), one STAR, GALAXY, QSO each.

    Predictions are correct except QSO -> STAR on pixels with ``pixel % errors_mod == 0``
    (all QSO -> STAR when ``errors_mod`` is 1; none when ``None``).
    """
    y, pred, pix, band = [], [], [], []
    for p in range(n_pix):
        for c in CLASSES:
            y.append(c)
            wrong = c == Q and errors_mod is not None and p % errors_mod == 0
            pred.append(S if wrong else c)
            pix.append(p)
            band.append(0 if p < n_pix // 2 else 1)
    return np.array(y), np.array(pred), np.array(pix), np.array(band)


BOOT_KW = {"min_objects": 1, "min_pixels": 1}


def test_bootstrap_hand_values_on_degenerate_data():
    # Every pixel: STAR->STAR, GALAXY->GALAXY, QSO->STAR. Uniform weights in every replicate:
    # STAR purity = w/(w+w) = .5, completeness 1, F1 2*.5*1/1.5 = 2/3; GALAXY all 1;
    # QSO completeness 0 and QSO purity undefined (never predicted) -> invalid replicates.
    y, pred, pix, band = _dataset(10, 1)
    res = M.cluster_bootstrap(
        y, {"a": onehot_proba(pred)}, pix, band, n_boot=50, **BOOT_KW
    )
    a = res["models"]["a"]
    for key, val in [("purity", 0.5), ("completeness", 1.0), ("f1", 2 / 3)]:
        cell = a[key][S]
        assert (cell["point"], cell["lo"], cell["hi"]) == pytest.approx((val, val, val))
    assert a["purity"][G]["point"] == pytest.approx(1.0)
    assert a["completeness"][Q]["lo"] == a["completeness"][Q]["hi"] == 0.0
    assert (
        np.isnan(a["purity"][Q]["lo"])
        and a["purity"][Q]["reason"] == "too_many_invalid"
    )
    assert a["purity"][Q]["n_invalid"] == 50


def test_bootstrap_deterministic_same_seed_same_ci():
    y, pred, pix, band = _dataset(20, 3)
    pr = {"a": onehot_proba(pred)}
    r1 = M.cluster_bootstrap(y, pr, pix, band, n_boot=100, **BOOT_KW)
    r2 = M.cluster_bootstrap(y, pr, pix, band, n_boot=100, **BOOT_KW)
    assert r1 == r2
    r3 = M.cluster_bootstrap(y, pr, pix, band, n_boot=100, seed=1, **BOOT_KW)
    c1 = r1["models"]["a"]["completeness"][Q]
    c3 = r3["models"]["a"]["completeness"][Q]
    assert (c1["lo"], c1["hi"]) != (c3["lo"], c3["hi"])
    assert (
        c1["lo"] < c1["point"] < c1["hi"]
    )  # QSO completeness 13/20 = .65 with real spread
    assert c1["point"] == pytest.approx(13 / 20)  # pixels 0,3,...,18 wrong: 7 of 20


def test_bootstrap_pairing_identical_models_have_zero_width_diff():
    # Same predictions -> per-replicate differences are exactly 0, so the paired CI is [0, 0]
    # even though each model's own CI is wide. An unpaired CI could not be zero-width.
    y, pred, pix, band = _dataset(20, 3)
    pr = {"a": onehot_proba(pred), "b": onehot_proba(pred)}
    res = M.cluster_bootstrap(y, pr, pix, band, n_boot=100, **BOOT_KW)
    own = res["models"]["a"]["completeness"][Q]
    assert own["hi"] - own["lo"] > 0.1
    d = res["diff"][("b", "a")]["completeness"][Q]
    assert (d["point"], d["lo"], d["hi"]) == (0.0, 0.0, 0.0)


def test_bootstrap_diff_is_second_minus_first_and_paired():
    # a errs on QSO at pixels % 3 == 0 (7 of 20), b never errs: completeness diff is b - a > 0.
    y, pred_a, pix, band = _dataset(20, 3)
    _, pred_b, _, _ = _dataset(20, None)
    pr = {"a": onehot_proba(pred_a), "b": onehot_proba(pred_b)}
    res = M.cluster_bootstrap(y, pr, pix, band, n_boot=100, **BOOT_KW)
    d = res["diff"][("b", "a")]["completeness"][Q]
    assert d["point"] == pytest.approx(1.0 - 13 / 20)
    assert d["lo"] > 0  # b is always >= a on any resample, strictly better on most
    b_cell = res["models"]["b"]["completeness"][Q]
    assert (b_cell["lo"], b_cell["hi"]) == (1.0, 1.0)


def test_bootstrap_rare_class_missing_marks_metric_invalid():
    # 20 pixels in one band; GALAXY only in pixel 0. P(GALAXY absent in a replicate)
    # = (19/20)^20 = 0.358 > 1% -> every metric is n/a (replicates are invalid as a whole).
    y, pix = [], []
    for p in range(20):
        y += [S, Q]
        pix += [p, p]
    y.append(G)
    pix.append(0)
    y, pix = np.array(y), np.array(pix)
    band = np.zeros(len(y), dtype=int)
    pred = y.copy()
    res = M.cluster_bootstrap(
        y, {"a": onehot_proba(pred)}, pix, band, n_boot=300, **BOOT_KW
    )
    cell = res["models"]["a"]["purity"][S]
    assert cell["reason"] == "too_many_invalid" and np.isnan(cell["lo"])
    assert 0.25 < cell["n_invalid"] / 300 < 0.47


def test_bootstrap_below_floor_and_input_validation():
    y, pred, pix, band = _dataset(10, 1)
    res = M.cluster_bootstrap(y, {"a": onehot_proba(pred)}, pix, band, n_boot=10)
    assert (
        res["models"]["a"]["purity"][S]["reason"] == "below_floor"
    )  # default floor: n=10
    bad_band = band.copy()
    bad_band[0] = 1 - bad_band[0]  # pixel 0 now spans two bands
    with pytest.raises(ValueError):
        M.cluster_bootstrap(y, {"a": onehot_proba(pred)}, pix, bad_band, n_boot=10)
