"""Overview numbers: headline, error types, confusion, calibration, abstention, model facts."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from astro_triage.dashboard.data_release import (
    CLASSES,
    Release,
    wilson,
)

CONFIDENCE_EDGES: Final = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001)


@dataclass(frozen=True)
class Headline:
    model: str
    n_test: int
    n_errors: int
    accuracy: float
    hits_per_100: int
    errors_per_100: int
    star_qso_error_share: float  # STAR<->QSO errors over all errors
    recall: dict[str, float]  # per true class
    precision: dict[str, float]  # per predicted class
    class_mix_train: dict[str, float]
    dsc_coverage: float
    n_test_pixels: int
    n_pixels_with_error: int


def headline(rel: Release, model: str = "lgbm_full") -> Headline:
    t = rel.test
    err = t[f"err_{model}"]
    n, n_err = len(t), int(err.sum())
    pred = t[f"pred_{model}"]
    recall = {c: float((~err[t["true"] == c]).mean()) for c in CLASSES}
    precision = {
        c: float((~err[pred == c]).mean()) if (pred == c).any() else math.nan
        for c in CLASSES
    }
    pair = ((t["true"] == "STAR") & (pred == "QSO")) | (
        (t["true"] == "QSO") & (pred == "STAR")
    )
    acc = 1 - n_err / n
    px = t.groupby("healpix_l5")[f"err_{model}"].any()
    fractions = rel.train_report["models"][0]["train_class_fractions"]
    return Headline(
        model=model,
        n_test=n,
        n_errors=n_err,
        accuracy=acc,
        hits_per_100=round(acc * 100),
        errors_per_100=100 - round(acc * 100),
        star_qso_error_share=float((err & pair).sum() / n_err) if n_err else math.nan,
        recall=recall,
        precision=precision,
        class_mix_train={c: float(fractions[c]) for c in CLASSES},
        dsc_coverage=float(t["dsc_available"].mean()),
        n_test_pixels=int(px.size),
        n_pixels_with_error=int(px.sum()),
    )


def errors_by_type(rel: Release, model: str = "lgbm_full") -> pd.DataFrame:
    """The off-diagonal cells as rows (true, pred, n, share), largest first."""
    t = rel.test[rel.test[f"err_{model}"]]
    out = t.groupby(["true", f"pred_{model}"]).size().rename("n").reset_index()
    out = out.rename(columns={f"pred_{model}": "pred"}).sort_values(
        "n", ascending=False
    )
    out["share"] = out["n"] / out["n"].sum() if len(out) else math.nan
    return out.reset_index(drop=True)


def confusion(rel: Release, model: str = "lgbm_full") -> pd.DataFrame:
    """Counts with rows = true class and columns = predicted class, in CLASSES order."""
    t = rel.test
    table = pd.crosstab(t["true"], t[f"pred_{model}"]).reindex(
        index=CLASSES, columns=CLASSES
    )
    return table.fillna(0).astype(int)


def reliability_rows(p: np.ndarray, hit: np.ndarray, n_bins: int) -> list[dict]:
    order = np.argsort(p, kind="stable")
    return [
        {
            "mean_pred": float(p[g].mean()),
            "observed": float(hit[g].mean()),
            "n": len(g),
        }
        for g in np.array_split(order, n_bins)
        if len(g)
    ]


def calibration_points(
    rel: Release, model: str = "lgbm_full", n_bins: int = 10
) -> dict[str, pd.DataFrame]:
    """One-vs-rest reliability points per class, equal-mass bins (mean_pred, observed, n).

    Also ``raw_mean_pred`` and ``raw_observed``: the same bins built on the uncalibrated
    probabilities (``p_raw_*``), i.e. the curve before temperature scaling.
    """
    t = rel.test
    out: dict[str, pd.DataFrame] = {}
    for c in CLASSES:
        hit = (t["true"] == c).to_numpy()
        cal = pd.DataFrame(
            reliability_rows(t[f"p_{model}_{c}"].to_numpy(dtype=float), hit, n_bins)
        )
        if f"praw_{model}_{c}" not in t:
            out[c] = cal
            continue
        raw = pd.DataFrame(
            reliability_rows(t[f"praw_{model}_{c}"].to_numpy(dtype=float), hit, n_bins)
        )
        out[c] = cal.assign(
            raw_mean_pred=raw["mean_pred"], raw_observed=raw["observed"]
        )
    return out


def confidence_histogram(
    rel: Release, model: str = "lgbm_full", edges: tuple[float, ...] = CONFIDENCE_EDGES
) -> pd.DataFrame:
    """Per confidence bin: objects, wrong ones, error share and a Wilson interval."""
    t = rel.test
    conf, wrong = t[f"conf_{model}"].to_numpy(), t[f"err_{model}"].to_numpy()
    rows = []
    for lo, hi in itertools.pairwise(edges):
        sel = (conf >= lo) & (conf < hi)
        n, k = int(sel.sum()), int(wrong[sel].sum())
        low, high = wilson(k, n)
        rows.append(
            {
                "lo": lo,
                "hi": min(hi, 1.0),
                "n": n,
                "n_wrong": k,
                "error_share": k / n if n else math.nan,
                "ci_lo": low,
                "ci_hi": high,
            }
        )
    return pd.DataFrame(rows)


def abstention_curve(
    rel: Release, model: str = "lgbm_full", thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """For each confidence threshold: coverage kept, accuracy on the kept rows, errors set aside."""
    grid = np.round(np.arange(0.40, 1.0, 0.01), 2) if thresholds is None else thresholds
    t = rel.test
    conf, wrong = t[f"conf_{model}"].to_numpy(), t[f"err_{model}"].to_numpy()
    total_err = int(wrong.sum())
    rows = []
    for thr in grid:
        keep = conf >= thr
        k = int(keep.sum())
        rows.append(
            {
                "threshold": float(thr),
                "coverage": k / len(conf),
                "accuracy_kept": float(1 - wrong[keep].mean()) if k else math.nan,
                "n_set_aside": int((~keep).sum()),
                "errors_set_aside": int(wrong[~keep].sum()),
                "errors_total": total_err,
            }
        )
    return pd.DataFrame(rows)
