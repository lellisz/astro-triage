"""Aggregations of the test fold for the Gráficos page (LightGBM; the floor is the exporter's job)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from astro_triage.dashboard.data_galaxies import error_rate_table, wilson_interval
from astro_triage.dashboard.data_galaxies_z import error_rate_table_z
from astro_triage.dashboard.data_overview import calibration_points, confusion
from astro_triage.dashboard.data_release import CLASSES, Release

# Same edges as ``G_BINS`` of the evaluation (``results["histograms"]["g_mag_edges"]``).
G_EDGES: Final[tuple[float, ...]] = tuple(float(g) for g in range(10, 23))
QSO_Z_EDGES: Final[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
CALIBRATION_BINS: Final = 10
# Colour-magnitude grid as (first edge, last edge, step); fixed in code, never taken from the data.
CMD_G: Final = (12.0, 21.0, 0.5)
CMD_C: Final = (-0.5, 3.0, 0.25)


@dataclass(frozen=True, eq=False)
class CmdGrid:
    """Every cell of the colour-magnitude grid plus the two totals left off it.

    ``cells`` has one row per cell (``gi`` along G, ``ci`` along BP-RP; G-major order) with
    ``n``, ``errors``, ``rate`` and the Wilson interval; the floor is not applied here.
    """

    cells: pd.DataFrame
    n_missing: int
    n_outside: int


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


def _edges(first: float, last: float, step: float) -> np.ndarray:
    return np.linspace(first, last, round((last - first) / step) + 1)


def cmd_edges() -> tuple[np.ndarray, np.ndarray]:
    """Edges of the map along G (19) and along BP-RP (15), from the constants only."""
    return _edges(*CMD_G), _edges(*CMD_C)


def cmd_cells(rel: Release, model: str) -> CmdGrid:
    """Test objects and errors per cell of the fixed grid; each cell is half-open [lo, hi).

    ``n_missing`` counts objects without G or BP-RP; ``n_outside`` counts the measured ones
    that fall off the grid. Both stay out of the cells. No smoothing, no data-derived edge.
    """
    g_edges, c_edges = cmd_edges()
    n_g, n_c = len(g_edges) - 1, len(c_edges) - 1
    g = rel.test["phot_g_mean_mag"].to_numpy(dtype=float)
    c = rel.test["bp_rp"].to_numpy(dtype=float)
    err = rel.test[f"err_{model}"].to_numpy(dtype=bool)
    missing = np.isnan(g) | np.isnan(c)
    with np.errstate(invalid="ignore"):
        inside = (
            (g >= g_edges[0]) & (g < g_edges[-1]) & (c >= c_edges[0]) & (c < c_edges[-1])
        )
    inside &= ~missing
    gi = np.searchsorted(g_edges, g[inside], side="right") - 1
    ci = np.searchsorted(c_edges, c[inside], side="right") - 1
    flat = gi * n_c + ci
    n = np.bincount(flat, minlength=n_g * n_c)
    errors = np.bincount(flat, weights=err[inside], minlength=n_g * n_c).astype(int)
    bounds = [wilson_interval(int(k), int(m)) for k, m in zip(errors, n)]
    cells = pd.DataFrame(
        {
            "gi": np.repeat(np.arange(n_g), n_c),
            "ci": np.tile(np.arange(n_c), n_g),
            "n": n,
            "errors": errors,
            "rate": np.divide(
                errors, n, out=np.full(len(n), math.nan), where=n > 0
            ),
            "ci_lo": [lo for lo, _ in bounds],
            "ci_hi": [hi for _, hi in bounds],
        }
    )
    return CmdGrid(
        cells=cells,
        n_missing=int(missing.sum()),
        n_outside=int((~missing & ~inside).sum()),
    )
