"""Phase 1 metric harness (ADR 0003, D4): pure numpy functions, no I/O.

Conventions
-----------
* Classes are ``CLASSES = ("STAR", "GALAXY", "QSO")``; ``proba`` columns follow that order.
* ``y_true`` / ``y_pred`` are arrays of those strings, ``pixel`` holds the level-5 HEALPix
  pixel of each object, ``sample_weight`` (optional) holds importance weights.
* Weights come from :func:`eval.reweight.reweight` (see :func:`regime_weights`); the
  BALANCED regime is the uniform prior, the REALISTIC regime a grid prior.
* A per-class metric is ``nan`` with reason ``"below_floor"`` unless the class has at least
  ``MIN_OBJECTS`` objects in at least ``MIN_PIXELS`` distinct pixels.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
from sklearn.metrics import average_precision_score

from eval.reweight import EmptyClassError, reweight

CLASSES: tuple[str, ...] = ("STAR", "GALAXY", "QSO")
MIN_OBJECTS = 100
MIN_PIXELS = 20
LOW_COUNT = 10
N_BINS = 15
N_BOOT = 1000
SEED = 20261025
MAX_INVALID_FRACTION = 0.01
BOOT_METRICS: tuple[str, ...] = ("purity", "completeness", "f1", "pr_auc")


class MetricValue(NamedTuple):
    """A metric value and, when it is ``nan``, the reason."""

    value: float
    reason: str | None


# --------------------------------------------------------------------------- helpers


def uniform_prior(classes: Sequence[str] = CLASSES) -> dict[str, float]:
    """Return the uniform prior over ``classes`` (the BALANCED regime, P0)."""
    return {c: 1.0 / len(classes) for c in classes}


def regime_weights(
    y_true: np.ndarray, prior: Mapping[str, float] | None = None
) -> np.ndarray:
    """Weights for a regime: ``prior=None`` is BALANCED (uniform), else REALISTIC.

    Raises ``EmptyClassError`` if a class with positive prior is absent from ``y_true``.
    """
    return reweight(y_true, uniform_prior() if prior is None else prior)


def predict_argmax(proba: np.ndarray) -> np.ndarray:
    """Argmax decision rule: map each row of ``proba`` to a class string."""
    return np.asarray(CLASSES)[np.argmax(np.asarray(proba), axis=1)]


def floor_status(
    y_true: np.ndarray,
    pixel: np.ndarray,
    min_objects: int = MIN_OBJECTS,
    min_pixels: int = MIN_PIXELS,
) -> dict[str, tuple[int, int, bool]]:
    """Per class: ``(n_objects, n_distinct_pixels, passes_floor)``."""
    y = np.asarray(y_true)
    pix = np.asarray(pixel)
    out: dict[str, tuple[int, int, bool]] = {}
    for c in CLASSES:
        mask = y == c
        n = int(mask.sum())
        npix = int(np.unique(pix[mask]).size)
        out[c] = (n, npix, n >= min_objects and npix >= min_pixels)
    return out


def _weights(y_true: np.ndarray, sample_weight: np.ndarray | None) -> np.ndarray:
    if sample_weight is None:
        return np.ones(len(y_true), dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.shape != (len(y_true),):
        raise ValueError("sample_weight must have shape (n,)")
    return w


# ------------------------------------------------------------------- (1) confusion etc.


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
) -> np.ndarray:
    """3x3 (weighted) confusion matrix; rows = true class, columns = predicted class."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    w = _weights(yt, sample_weight)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.float64)
    for i, ct in enumerate(CLASSES):
        for j, cp in enumerate(CLASSES):
            cm[i, j] = w[(yt == ct) & (yp == cp)].sum()
    return cm


def class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pixel: np.ndarray,
    sample_weight: np.ndarray | None = None,
    min_objects: int = MIN_OBJECTS,
    min_pixels: int = MIN_PIXELS,
    flag_low_count: bool = False,
) -> dict[str, dict]:
    """Per-class purity, completeness and F1 for a fixed decision rule.

    ``sample_weight=None`` gives raw counts; pass :func:`regime_weights` output for the
    BALANCED or REALISTIC regime. Each class entry holds ``n``, ``pixels``, ``status``
    (``"ok"`` or ``"below_floor"``), ``purity``, ``completeness``, ``f1`` (``nan`` when
    below floor or undefined, with ``reason``), ``contaminants`` (raw count of true-other
    objects predicted as this class, per other class) and ``contamination`` (per other
    class: ``count``, ``rate = count / n_k`` or the rule-of-three bound ``3 / n_k`` when
    the count is 0, ``upper_bound`` flag). ``purity_flag`` is ``"low-count"`` only when
    the caller sets ``flag_low_count=True`` (REALISTIC regime; ADR 0003 limits the flag to
    REALISTIC purity, the default never flags) and the SUM of the raw contaminant counts over the other classes is
    < 10 (ADR 0003, total reading); a contaminant class absent from the test set adds 0 and
    does not trigger the flag by itself.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    w = _weights(yt, sample_weight)
    floor = floor_status(yt, pixel, min_objects, min_pixels)
    out: dict[str, dict] = {}
    for c in CLASSES:
        n, npix, ok = floor[c]
        pred_c = yp == c
        true_c = yt == c
        contaminants = {k: int((pred_c & (yt == k)).sum()) for k in CLASSES if k != c}
        contamination = {}
        for k, cnt in contaminants.items():
            n_k = int((yt == k).sum())
            if n_k == 0:
                contamination[k] = {"count": cnt, "rate": np.nan, "upper_bound": False}
            elif cnt == 0:
                contamination[k] = {"count": 0, "rate": 3.0 / n_k, "upper_bound": True}
            else:
                contamination[k] = {
                    "count": cnt,
                    "rate": cnt / n_k,
                    "upper_bound": False,
                }
        entry: dict = {
            "n": n,
            "pixels": npix,
            "status": "ok" if ok else "below_floor",
            "purity": np.nan,
            "completeness": np.nan,
            "f1": np.nan,
            "reason": None if ok else "below_floor",
            "contaminants": contaminants,
            "contamination": contamination,
            "purity_flag": None,
        }
        if ok:
            tp = w[pred_c & true_c].sum()
            pred_w = w[pred_c].sum()
            true_w = w[true_c].sum()
            purity = tp / pred_w if pred_w > 0 else np.nan
            completeness = tp / true_w if true_w > 0 else np.nan
            entry["purity"] = float(purity)
            entry["completeness"] = float(completeness)
            if np.isnan(purity) or np.isnan(completeness):
                entry["reason"] = "undefined"
            elif purity + completeness > 0:
                entry["f1"] = float(2 * purity * completeness / (purity + completeness))
            else:
                entry["f1"] = 0.0
            if flag_low_count and sum(contaminants.values()) < LOW_COUNT:
                entry["purity_flag"] = "low-count"
        out[c] = entry
    return out


def macro_f1(per_class: Mapping[str, Mapping]) -> float:
    """Unweighted mean of the per-class F1; ``nan`` if any class F1 is ``nan``."""
    vals = np.array([per_class[c]["f1"] for c in CLASSES], dtype=np.float64)
    return float(vals.mean())


# ------------------------------------------------------------------------- (2) PR-AUC


def pr_auc(
    y_true: np.ndarray,
    proba: np.ndarray,
    sample_weight: np.ndarray | None = None,
    pixel: np.ndarray | None = None,
    min_objects: int = MIN_OBJECTS,
    min_pixels: int = MIN_PIXELS,
) -> dict[str, MetricValue]:
    """One-vs-rest (weighted) average precision per class.

    Absent classes give ``nan`` (``"absent_class"``); if ``pixel`` is given the
    reporting floor is applied (``"below_floor"``).
    """
    yt = np.asarray(y_true)
    p = np.asarray(proba, dtype=np.float64)
    w = None if sample_weight is None else _weights(yt, sample_weight)
    floor = None if pixel is None else floor_status(yt, pixel, min_objects, min_pixels)
    out: dict[str, MetricValue] = {}
    for i, c in enumerate(CLASSES):
        positives = yt == c
        if not positives.any() or (w is not None and w[positives].sum() <= 0):
            out[c] = MetricValue(np.nan, "absent_class")
        elif floor is not None and not floor[c][2]:
            out[c] = MetricValue(np.nan, "below_floor")
        else:
            out[c] = MetricValue(
                float(average_precision_score(positives, p[:, i], sample_weight=w)),
                None,
            )
    return out


# --------------------------------------------------------------------------- (3) ECE


BINNINGS: tuple[str, ...] = ("equal_mass", "equal_width")


def _bin_index(c: np.ndarray, w: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    """Bin index of each confidence.

    ``equal_width``: ``floor(c * n_bins)`` over [0, 1] (1.0 falls in the last bin).

    ``equal_mass``: weighted quantile bins. Each distinct confidence value carries the
    total weight of its ties and goes to bin ``floor(m / W * n_bins)``, where ``m`` is the
    cumulative weight up to the middle of that value's mass and ``W`` the total weight.
    Tie rule: equal confidences are never split across bins (so a bin can exceed or fall
    short of ``W / n_bins`` by less than one tied group); bins left without any object
    (heavy ties, or fewer distinct values than ``n_bins``) stay in the output with
    ``count = 0`` and ``nan`` statistics, and no division by zero occurs. If ``W <= 0``
    everything goes to bin 0.
    """
    if binning == "equal_width":
        return np.minimum((c * n_bins).astype(int), n_bins - 1)
    if binning != "equal_mass":
        raise ValueError(f"binning must be one of {BINNINGS}, got {binning!r}")
    uniq, inv = np.unique(c, return_inverse=True)
    inv = inv.reshape(-1)
    mass = np.bincount(inv, weights=w, minlength=uniq.size)
    total = mass.sum()
    if total <= 0:
        return np.zeros(len(c), dtype=int)
    mid = np.cumsum(mass) - mass / 2.0
    bin_u = np.minimum((mid / total * n_bins).astype(int), n_bins - 1)
    return bin_u[inv]


def reliability_curve(
    conf: np.ndarray,
    outcome: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_bins: int = N_BINS,
    binning: str = "equal_mass",
) -> dict[str, np.ndarray]:
    """Reliability points over ``n_bins`` bins (``equal_mass`` default, or ``equal_width``).

    Returns arrays of length ``n_bins``: ``mean_conf`` and ``freq`` (weighted, ``nan``
    for empty bins), ``count`` (raw objects) and ``weight`` (summed weight per bin).
    See :func:`_bin_index` for the binning and tie rules.
    """
    c = np.asarray(conf, dtype=np.float64)
    o = np.asarray(outcome, dtype=np.float64)
    w = _weights(c, sample_weight)
    idx = _bin_index(c, w, n_bins, binning)
    mean_conf = np.full(n_bins, np.nan)
    freq = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)
    weight = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        count[b] = int(m.sum())
        weight[b] = w[m].sum()
        if count[b] > 0 and weight[b] > 0:
            mean_conf[b] = np.average(c[m], weights=w[m])
            freq[b] = np.average(o[m], weights=w[m])
    return {"mean_conf": mean_conf, "freq": freq, "count": count, "weight": weight}


def ece(
    y_true: np.ndarray,
    proba: np.ndarray,
    sample_weight: np.ndarray | None = None,
    pixel: np.ndarray | None = None,
    n_bins: int = N_BINS,
    min_objects: int = MIN_OBJECTS,
    min_pixels: int = MIN_PIXELS,
    binning: str = "equal_mass",
) -> dict[str, tuple[MetricValue, dict[str, np.ndarray]]]:
    """One-vs-rest ECE per class with its reliability points.

    ECE = sum_b (weight_b / weight_total) * |freq_b - mean_conf_b| over the bins of
    ``binning`` (15 equal-mass bins by default, ADR 0003; ``"equal_width"`` optional).
    Absent / below-floor classes give ``nan`` and still return the curve when computable.
    """
    yt = np.asarray(y_true)
    p = np.asarray(proba, dtype=np.float64)
    w = _weights(yt, sample_weight)
    floor = None if pixel is None else floor_status(yt, pixel, min_objects, min_pixels)
    out: dict[str, tuple[MetricValue, dict[str, np.ndarray]]] = {}
    for i, c in enumerate(CLASSES):
        outcome = yt == c
        curve = reliability_curve(p[:, i], outcome, w, n_bins, binning)
        if not outcome.any() or w[outcome].sum() <= 0:
            out[c] = (MetricValue(np.nan, "absent_class"), curve)
        elif floor is not None and not floor[c][2]:
            out[c] = (MetricValue(np.nan, "below_floor"), curve)
        else:
            ok = ~np.isnan(curve["freq"])
            gap = np.abs(curve["freq"][ok] - curve["mean_conf"][ok])
            value = float((curve["weight"][ok] / curve["weight"][ok].sum() * gap).sum())
            out[c] = (MetricValue(value, None), curve)
    return out


# ------------------------------------------------------------------ (5) cluster bootstrap


def bootstrap_pixel_draw(
    pixel: np.ndarray, lat_band: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw one replicate: resample pixels with replacement within each latitude band.

    Returns the object indices of the drawn pixels (a pixel drawn twice appears twice).
    Bands are visited in sorted order, so a fixed ``rng`` state gives a fixed draw.
    """
    pix = np.asarray(pixel)
    band = np.asarray(lat_band)
    parts: list[np.ndarray] = []
    for b in np.unique(band):
        in_band = np.flatnonzero(band == b)
        upix, inv = np.unique(pix[in_band], return_inverse=True)
        members = [in_band[inv.reshape(-1) == k] for k in range(upix.size)]
        drawn = rng.integers(0, upix.size, size=upix.size)
        parts.extend(members[k] for k in drawn)
    return np.concatenate(parts)


def replicate_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    idx: np.ndarray,
    prior: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]] | None:
    """Metrics of one replicate with weights recomputed on the resampled labels.

    Returns ``None`` (invalid replicate) when a class with positive prior is absent.
    The reporting floor is not applied here (it is judged on the full test set).
    """
    y = np.asarray(y_true)[idx]
    p = np.asarray(proba)[idx]
    try:
        w = regime_weights(y, prior)
    except EmptyClassError:
        return None
    pred = predict_argmax(p)
    # pixel is irrelevant to the values; floor is disabled with zero thresholds.
    cm = class_metrics(y, pred, np.zeros(len(y), dtype=np.int64), w, 0, 0)
    ap = pr_auc(y, p, w)
    out: dict[str, dict[str, float]] = {m: {} for m in BOOT_METRICS}
    for c in CLASSES:
        out["purity"][c] = cm[c]["purity"]
        out["completeness"][c] = cm[c]["completeness"]
        out["f1"][c] = cm[c]["f1"]
        out["pr_auc"][c] = ap[c].value
    return out


def _interval(samples: np.ndarray, n_boot: int) -> tuple[float, float, int, str | None]:
    finite = np.isfinite(samples)
    n_invalid = int((~finite).sum())
    if n_invalid > MAX_INVALID_FRACTION * n_boot:
        return np.nan, np.nan, n_invalid, "too_many_invalid"
    lo, hi = np.percentile(samples[finite], [2.5, 97.5])
    return float(lo), float(hi), n_invalid, None


def cluster_bootstrap(
    y_true: np.ndarray,
    probas: Mapping[str, np.ndarray],
    pixel: np.ndarray,
    lat_band: np.ndarray,
    prior: Mapping[str, float] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    min_objects: int = MIN_OBJECTS,
    min_pixels: int = MIN_PIXELS,
) -> dict:
    """Pixel-cluster bootstrap within latitude bands for one or more models.

    ``lat_band`` gives the band of each object (constant within a pixel). All models are
    evaluated on the same replicates; weights are recomputed per replicate. Result:
    ``{"models": {name: {metric: {class: cell}}}, "diff": {(b, a): {metric: {class:
    cell}}}}`` where a cell is ``{point, lo, hi, n_invalid, reason}`` (95% percentile
    interval) and ``diff`` is model ``b`` minus model ``a`` for every ordered pair of
    names in input order (``b`` after ``a``), paired replicate by replicate. A replicate
    is invalid for a metric if it is non-finite (class with positive prior absent, or an
    undefined ratio); more than 1% invalid -> ``nan`` / ``"too_many_invalid"``. Classes
    below the floor on the full sample give ``nan`` / ``"below_floor"``.
    """
    y = np.asarray(y_true)
    pix = np.asarray(pixel)
    band = np.asarray(lat_band)
    for u in np.unique(pix):
        if np.unique(band[pix == u]).size != 1:
            raise ValueError(f"pixel {u} appears in more than one latitude band")
    names = list(probas)
    floor = floor_status(y, pix, min_objects, min_pixels)

    point = {}
    w_full = regime_weights(y, prior)
    for name in names:
        p = np.asarray(probas[name])
        cm = class_metrics(y, predict_argmax(p), pix, w_full, min_objects, min_pixels)
        ap = pr_auc(y, p, w_full, pix, min_objects, min_pixels)
        point[name] = {
            "purity": {c: cm[c]["purity"] for c in CLASSES},
            "completeness": {c: cm[c]["completeness"] for c in CLASSES},
            "f1": {c: cm[c]["f1"] for c in CLASSES},
            "pr_auc": {c: ap[c].value for c in CLASSES},
        }

    rng = np.random.default_rng(seed)
    draws = {
        name: {m: {c: np.full(n_boot, np.nan) for c in CLASSES} for m in BOOT_METRICS}
        for name in names
    }
    for r in range(n_boot):
        idx = bootstrap_pixel_draw(pix, band, rng)
        for name in names:
            res = replicate_metrics(y, probas[name], idx, prior)
            if res is None:
                continue
            for m in BOOT_METRICS:
                for c in CLASSES:
                    draws[name][m][c][r] = res[m][c]

    def cell(pt: float, samples: np.ndarray, c: str) -> dict:
        if not floor[c][2]:
            return {
                "point": np.nan,
                "lo": np.nan,
                "hi": np.nan,
                "n_invalid": 0,
                "reason": "below_floor",
            }
        lo, hi, n_inv, reason = _interval(samples, n_boot)
        return {"point": pt, "lo": lo, "hi": hi, "n_invalid": n_inv, "reason": reason}

    models = {
        name: {
            m: {c: cell(point[name][m][c], draws[name][m][c], c) for c in CLASSES}
            for m in BOOT_METRICS
        }
        for name in names
    }
    diff = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diff[(b, a)] = {
                m: {
                    c: cell(
                        point[b][m][c] - point[a][m][c],
                        draws[b][m][c] - draws[a][m][c],
                        c,
                    )
                    for c in CLASSES
                }
                for m in BOOT_METRICS
            }
    return {"models": models, "diff": diff}
