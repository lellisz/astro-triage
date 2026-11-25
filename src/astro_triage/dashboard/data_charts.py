"""Aggregations of the test fold for the Gráficos page (LightGBM; the floor is the exporter's job)."""

from __future__ import annotations

import math
from typing import Final, Literal

import pandas as pd

from astro_triage.dashboard.data_galaxies import error_rate_table, wilson_interval
from astro_triage.dashboard.data_galaxies_z import error_rate_table_z
from astro_triage.dashboard.data_overview import calibration_points, confusion
from astro_triage.dashboard.data_release import CLASSES, Release

# Same edges as ``G_BINS`` of the evaluation (``results["histograms"]["g_mag_edges"]``).
G_EDGES: Final[tuple[float, ...]] = tuple(float(g) for g in range(10, 23))
QSO_Z_EDGES: Final[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
CALIBRATION_BINS: Final = 10


def calibration_bins(
    rel: Release, model: str, n_bins: int = CALIBRATION_BINS
) -> dict[str, pd.DataFrame]:
    """Per class: the reliability points plus the hit count and its Wilson interval."""
    out: dict[str, pd.DataFrame] = {}
    for c, points in calibration_points(rel, model, n_bins).items():
        hits = (points["observed"] * points["n"]).round().astype(int)
        bounds = [
            wilson_interval(int(k), int(n)) for k, n in zip(hits, points["n"])
        ]
        out[c] = points.assign(
            hits=hits,
            ci_lo=[lo for lo, _ in bounds],
            ci_hi=[hi for _, hi in bounds],
        )
    return out


def error_by(
    rel: Release, model: str, by: Literal["G", "z"]
) -> dict[str, pd.DataFrame]:
    """Error-rate tables per true class by G (all classes) or by z (GALAXY and QSO)."""
    if by == "G":
        return {c: error_rate_table(rel, model, c, "G", G_EDGES) for c in CLASSES}
    return {
        "GALAXY": error_rate_table_z(rel, model),
        "QSO": error_rate_table(rel, model, "QSO", "z", QSO_Z_EDGES),
    }


def confusion_cells(rel: Release, model: str) -> pd.DataFrame:
    """The 3 x 3 confusion counts as nine rows with the share of the true class."""
    counts = confusion(rel, model)
    rows = []
    for true in CLASSES:
        row_n = int(counts.loc[true].sum())
        for pred in CLASSES:
            n = int(counts.loc[true, pred])
            lo, hi = wilson_interval(n, row_n)
            rows.append(
                {
                    "true": true,
                    "pred": pred,
                    "n": n,
                    "row_n": row_n,
                    "share": n / row_n if row_n else math.nan,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return pd.DataFrame(rows)
