"""Triage summary: what the abstention curve says at one threshold and at the lowest grid threshold."""

from __future__ import annotations

import math

import pandas as pd

_ROW_KEYS = (
    "coverage",
    "accuracy_kept",
    "n_set_aside",
    "errors_set_aside",
    "errors_total",
)


def _float(value: float | None) -> float:
    if value is None:
        return math.nan
    return float(value)


def triage_summary(curve: pd.DataFrame, threshold: float) -> dict | None:
    """Curve row nearest to ``threshold`` plus the lowest-threshold point (not 100% coverage).

    Returns None when the curve has no usable row. Missing values come back as NaN.
    """
    if curve is None or curve.empty or "threshold" not in curve.columns:
        return None
    valid = curve.dropna(subset=["threshold"])
    if valid.empty:
        return None
    nearest = valid.iloc[(valid["threshold"] - threshold).abs().argmin()]
    lowest = valid.iloc[valid["threshold"].to_numpy().argmin()]
    out: dict = {"threshold": _float(nearest["threshold"])}
    out |= {k: _float(nearest.get(k)) for k in _ROW_KEYS}
    out["decide_all_threshold"] = _float(lowest["threshold"])
    out["decide_all_coverage"] = _float(lowest.get("coverage"))
    out["decide_all_accuracy"] = _float(lowest.get("accuracy_kept"))
    return out
