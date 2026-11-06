"""Galaxy tab data for the low-redshift band and the overall galaxy error figures."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from astro_triage.dashboard import fmt
from astro_triage.dashboard.data_galaxies import (
    GALAXY_Z_EDGES,
    bin_label,
    rate_row,
)
from astro_triage.dashboard.data_release import SEED, Release

# Upper limit of the separate low-z band. The dashboard used to clip z to this value
# (``clip(lower=1e-3)``), which piled every smaller or negative z on the same point.
Z_LOW: Final = 1e-3

# Cap on the correctly classified galaxies (outside the z <= Z_LOW band) sent to the browser.
MAX_SCATTER_POINTS: Final = 5000


def z_low_label() -> str:
    return f"z ≤ {fmt.pt_num(Z_LOW, 3)}"


@dataclass(frozen=True)
class GalaxyTotals:
    n: int
    errors: int
    to_qso: int
    to_star: int

    @property
    def rate(self) -> float:
        return self.errors / self.n if self.n else math.nan


@dataclass(frozen=True)
class ZLowSummary:
    n: int
    negatives: int
    errors: int
    rate: float
    share_of_errors: float
    share_of_galaxies: float


def _galaxies(rel: Release) -> pd.DataFrame:
    return rel.test[rel.test["true"] == "GALAXY"]


def galaxy_totals(rel: Release, model: str) -> GalaxyTotals:
    t = _galaxies(rel)
    err = t[f"err_{model}"].astype(bool)
    pred = t[f"pred_{model}"]
    to_qso = int((err & (pred == "QSO")).sum())
    errors = int(err.sum())
    return GalaxyTotals(len(t), errors, to_qso, errors - to_qso)


def z_low_summary(rel: Release, model: str) -> ZLowSummary:
    """Counts of the ``z <= Z_LOW`` band (NaN z never falls in it)."""
    t = _galaxies(rel)
    low = t[t["sdss_z"] <= Z_LOW]
    n, k = len(low), int(low[f"err_{model}"].sum())
    total = galaxy_totals(rel, model)
    return ZLowSummary(
        n=n,
        negatives=int((low["sdss_z"] < 0).sum()),
        errors=k,
        rate=k / n if n else math.nan,
        share_of_errors=k / total.errors if total.errors else math.nan,
        share_of_galaxies=n / total.n if total.n else math.nan,
    )


def z_low_sentence(s: ZLowSummary) -> str:
    """One line for the separate band; states what was done, never a physical cause."""
    neg = f" (inclui {fmt.pt_int(s.negatives)} com z negativo)" if s.negatives else ""
    return (
        f"{z_low_label()}{neg}: agrupados só para caber no eixo log; "
        f"{fmt.pt_int(s.n)} galáxias, {fmt.pt_pct(s.rate)} classificadas errado."
    )


def scatter_sample(
    df: pd.DataFrame, max_points: int = MAX_SCATTER_POINTS, seed: int = SEED
) -> pd.DataFrame:
    """Every error plus a seeded, z-stratified sample of the correct galaxies.

    The ``z <= Z_LOW`` band is always whole. ``max_points`` caps the other correct galaxies,
    split across the z bands (``Z_LOW`` and ``GALAXY_Z_EDGES``) in proportion to their size.
    """
    ok = ~df["error"].to_numpy(dtype=bool)
    low = df["z"].to_numpy() <= Z_LOW
    mask = ~ok | low
    pos = np.flatnonzero(ok & ~low)
    band = np.searchsorted(
        (Z_LOW, *GALAXY_Z_EDGES), df["z"].to_numpy()[pos], side="right"
    )
    budget = max(0, max_points - int((ok & low).sum()))
    rng = np.random.default_rng(seed)
    for b in np.unique(band):
        members = pos[band == b]
        quota = min(len(members), budget * len(members) // len(pos))
        mask[rng.choice(members, size=quota, replace=False)] = True
    return df[mask].reset_index(drop=True)


def error_rate_table_z(rel: Release, model: str) -> pd.DataFrame:
    """Error rate by z with the ``z <= Z_LOW`` band first, then the usual bands above it."""
    t = _galaxies(rel)
    err = f"err_{model}"
    low = t[t["sdss_z"] <= Z_LOW]
    rows = [rate_row(z_low_label(), -math.inf, Z_LOW, low[err])]
    for lo, hi in itertools.pairwise((Z_LOW, *GALAXY_Z_EDGES, math.inf)):
        sub = t[(t["sdss_z"] > Z_LOW) & (t["sdss_z"] >= lo) & (t["sdss_z"] < hi)]
        rows.append(
            rate_row(bin_label(lo, hi, exclusive_lo=lo == Z_LOW), lo, hi, sub[err])
        )
    missing = t[t["sdss_z"].isna()]
    if len(missing):
        rows.append(rate_row("sem medida", math.nan, math.nan, missing[err]))
    return pd.DataFrame(rows)
