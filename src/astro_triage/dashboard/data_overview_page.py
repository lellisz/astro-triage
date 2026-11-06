"""Inputs of the Visão geral page, gathered from the artifacts. No Streamlit import.

Nothing here is typed by hand: every field is read or computed from the release.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from astro_triage.dashboard import data as D
from astro_triage.dashboard.data_release import CLASSES, Release

PRIOR = "P2"
# A control parameter of the triage (confidence threshold), not a result.
TRIAGE_THRESHOLD = 0.90


@dataclass(frozen=True)
class PageData:
    n_pairs: int | None
    n_test: int
    n_errors: int
    n_pixels: int
    accuracy: float
    class_mix: dict[str, float]
    qso_hits: int
    qso_total: int
    prior: str = PRIOR
    shipped: dict | None = None
    matched: dict | None = None
    validation: dict | None = None
    directions: list[tuple[str, str, int]] = field(default_factory=list)
    triage: dict | None = None
    threshold: float = TRIAGE_THRESHOLD


def total_pairs(train_report: dict) -> int | None:
    """Matched pairs of the release (train + val + test rows); None when the report lacks them."""
    per_fold = train_report.get("n_per_fold")
    if not isinstance(per_fold, dict) or not per_fold:
        return None
    try:
        return sum(int(v) for v in per_fold.values())
    except (TypeError, ValueError):
        return None


def class_mix_of(meta: dict) -> dict[str, float]:
    """Share of each class among the test objects; empty when ``n_by_class`` is absent."""
    counts = meta.get("n_by_class")
    if not isinstance(counts, dict):
        return {}
    total = sum(int(counts.get(c, 0)) for c in CLASSES)
    if total <= 0:
        return {}
    return {c: int(counts.get(c, 0)) / total for c in CLASSES}


def _point(points: list[dict], key: str) -> dict | None:
    return next((p for p in points if p["key"] == key), None)


def usable_triage(summary: dict | None) -> dict | None:
    """The triage summary, or None when it lacks a number the page prints."""
    if summary is None:
        return None
    keys = (
        "coverage",
        "accuracy_kept",
        "n_set_aside",
        "errors_set_aside",
        "errors_total",
    )
    if any(math.isnan(summary[k]) for k in keys):
        return None
    return summary


def load_page(
    rel: Release, h: D.Headline, points: list[dict], curve: pd.DataFrame, model: str
) -> PageData:
    """Collect everything the page prints; ``h``, ``points`` and ``curve`` come from the cache."""
    conf = D.confusion(rel, model)
    directions = D.error_directions(rel, model)
    rows = [(str(r.true), str(r.pred), int(r.n)) for r in directions.itertuples()]
    return PageData(
        n_pairs=total_pairs(rel.train_report),
        n_test=h.n_test,
        n_errors=h.n_errors,
        n_pixels=h.n_test_pixels,
        accuracy=h.accuracy,
        class_mix=class_mix_of(rel.meta),
        qso_hits=int(conf.loc["QSO", "QSO"]),
        qso_total=int(conf.loc["QSO"].sum()),
        shipped=_point(points, "as_shipped"),
        matched=_point(points, "prior_matched"),
        validation=_point(points, "validation"),
        directions=rows,
        triage=usable_triage(D.triage_summary(curve, TRIAGE_THRESHOLD)),
    )
