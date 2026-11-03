"""Routing threshold, verbal-confidence calibration and layer combination (ADR 0005, D2/D4).

Conventions that matter:
- Classes are mapped by NAME (``p_cal_star`` ...), never by position: the prediction parquet
  orders its columns star, qso, galaxy, unlike the display order STAR, GALAXY, QSO.
- Every threshold is fitted on the val fold only. Callers pass val rows; ``select_val``
  refuses anything else so a test row can never reach a fit.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

CLASSES: Final[tuple[str, ...]] = ("STAR", "GALAXY", "QSO")
ABSTAIN: Final = "ABSTAIN"
MIN_OK_ANSWERS: Final = 100  # below this the verbal confidence stays uncalibrated
DEFAULT_GRID: Final = np.round(np.arange(0.50, 0.99 + 1e-9, 0.01), 2)


def score_columns() -> list[str]:
    return [f"p_cal_{c.lower()}" for c in CLASSES]


def select_val(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ``fold == "val"`` rows; the test fold must never feed a threshold."""
    if "fold" not in df.columns:
        raise ValueError("a 'fold' column is required to guarantee val-only fitting")
    val = df[df["fold"].astype(str) == "val"]
    if val.empty:
        raise ValueError("no val rows: thresholds are fitted on the val fold only")
    return val


def l1_confidence(pred: pd.DataFrame) -> pd.Series:
    """Layer 1 confidence: max of the three calibrated probabilities, read by name."""
    cols = score_columns()
    missing = [c for c in cols if c not in pred.columns]
    if missing:
        raise ValueError(f"prediction columns missing: {missing}")
    return pred[cols].max(axis=1).rename("l1_conf")


def choose_threshold(
    conf: np.ndarray,
    correct: np.ndarray,
    a_keep: float,
    r_max: float,
    grid: np.ndarray = DEFAULT_GRID,
) -> dict[str, float | bool]:
    """ADR D2.2: smallest tau with kept accuracy >= a_keep, capped by the routed fraction r_max.

    Objects with ``conf >= tau`` are kept by Layer 1. If the routed fraction at that tau
    exceeds ``r_max``, tau is lowered to the largest grid value with routed fraction <=
    ``r_max`` and ``conflict`` is True (kept accuracy then falls below ``a_keep``).
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if conf.shape != correct.shape or conf.ndim != 1 or conf.size == 0:
        raise ValueError(
            "conf and correct must be non-empty 1-D arrays of equal length"
        )
    if np.isnan(conf).any():
        raise ValueError("conf contains NaN")
    grid = np.sort(np.asarray(grid, dtype=float))

    def routed_fraction(tau: float) -> float:
        return float(np.mean(conf < tau))

    def kept_accuracy(tau: float) -> float:
        kept = conf >= tau
        return float(correct[kept].mean()) if kept.any() else float("nan")

    # NaN (nothing kept) compares False, so an empty kept set never satisfies a_keep.
    ok = [t for t in grid if kept_accuracy(float(t)) >= a_keep]
    if not ok:
        raise ValueError(f"no grid value reaches kept accuracy >= {a_keep}")
    tau_acc = float(ok[0])
    tau = tau_acc
    conflict = routed_fraction(tau_acc) > r_max
    if conflict:
        fits = [t for t in grid if routed_fraction(float(t)) <= r_max]
        if not fits:
            raise ValueError(f"no grid value routes a fraction <= {r_max}")
        tau = float(fits[-1])
    return {
        "tau": tau,
        "tau_from_a_keep": tau_acc,
        "conflict": conflict,
        "routed_fraction": routed_fraction(tau),
        "accuracy_kept": kept_accuracy(tau),
        "n": float(conf.size),
    }


def fit_verbal_calibration(
    raw: np.ndarray, correct: np.ndarray, n_bins: int = 5
) -> dict[str, list[float | None]] | None:
    """Equal-mass histogram binning of verbalized confidence (ADR D4.5).

    ``raw`` and ``correct`` are the val answers with status OK. Returns None when fewer
    than ``MIN_OK_ANSWERS`` exist. Edges come from quantiles and are de-duplicated, so
    identical raw scores never produce empty duplicate bins.
    """
    raw = np.asarray(raw, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if raw.shape != correct.shape or raw.ndim != 1:
        raise ValueError("raw and correct must be 1-D arrays of equal length")
    if raw.size < MIN_OK_ANSWERS or np.isnan(raw).any():
        return None
    inner = np.unique(np.quantile(raw, np.linspace(0, 1, n_bins + 1)[1:-1]))
    idx = np.searchsorted(inner, raw, side="right")
    values: list[float | None] = []
    for b in range(inner.size + 1):
        in_bin = idx == b
        values.append(float(correct[in_bin].mean()) if in_bin.any() else None)
    return {"edges": [float(e) for e in inner], "values": values}


def apply_verbal_calibration(
    raw: np.ndarray, calib: Mapping[str, list[float | None]] | None
) -> np.ndarray:
    """Map raw verbal confidence to ``s_l2``; NaN everywhere when uncalibrated."""
    raw = np.asarray(raw, dtype=float)
    if calib is None:
        return np.full(raw.shape, np.nan)
    values = np.array(
        [np.nan if v is None else v for v in calib["values"]], dtype=float
    )
    idx = np.searchsorted(np.asarray(calib["edges"], dtype=float), raw, side="right")
    out = values[idx]
    out[np.isnan(raw)] = np.nan
    return out


def combine(
    l1_label: np.ndarray,
    l1_conf: np.ndarray,
    l2_status: np.ndarray,
    l2_label: np.ndarray,
    s_l2: np.ndarray,
    tau: float,
    tau2: float | None,
    policy: Literal["abstain_on_fail", "fallback_l1"],
) -> np.ndarray:
    """ADR D2.4: Layer 1 label when confident, else Layer 2 if trusted, else ABSTAIN/L1."""
    if policy not in ("abstain_on_fail", "fallback_l1"):
        raise ValueError(f"unknown policy: {policy!r}")
    l1_label = np.asarray(l1_label, dtype=object)
    l2_label = np.asarray(l2_label, dtype=object)
    routed = np.asarray(l1_conf, dtype=float) < tau
    trusted = (np.asarray(l2_status, dtype=object) == "OK") & np.isin(l2_label, CLASSES)
    if tau2 is not None:
        # NaN s_l2 compares False: an unscored answer is not trusted once tau2 applies.
        trusted &= np.asarray(s_l2, dtype=float) >= tau2
    fallback = l1_label if policy == "fallback_l1" else ABSTAIN
    return np.where(routed, np.where(trusted, l2_label, fallback), l1_label).astype(
        object
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_routing_manifest(path: Path, payload: Mapping[str, object]) -> None:
    """Write the frozen manifest through a temp file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
