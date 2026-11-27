"""Galáxias page, simple mode: decisions by code (title, extreme bands) and the four bars.

Pure: no streamlit. Every figure comes from the error-rate tables and the galaxy totals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import pandas as pd

from astro_triage.dashboard import fmt, theme
from astro_triage.dashboard.constants import ERROR_HIGH

# Rates below this share are written with two decimals.
SMALL_RATE: Final = 0.01
# What each extreme band is, in words that do not read z as a distance.
BRIGHTEST: Final = "mais brilhantes"
NEAREST: Final = "redshift quase nulo"
FARTHEST: Final = "maior redshift"


@dataclass(frozen=True)
class Band:
    """One band of an error-rate table, ready to show."""

    name: str
    rate: float
    n: int
    errors: int
    ci_lo: float


def rate_color(rate: float) -> str:
    return theme.ACCENT if rate > ERROR_HIGH else theme.SOFT


def rate_text(rate: float) -> str:
    if math.isnan(rate):
        return "—"
    return fmt.pt_pct(rate, 2 if rate < SMALL_RATE else 1)


def _with_prefix(prefix: str, label: str) -> str:
    return label if label[:1].isalpha() else f"{prefix} {label}"


def _usable(table: pd.DataFrame) -> pd.DataFrame:
    """Bands with objects and a defined position (drops the 'sem medida' row)."""
    return table[(table["n"] > 0) & table["lo"].notna()]


def _band(row: pd.Series, prefix: str) -> Band:
    return Band(
        _with_prefix(prefix, str(row["label"])),
        float(row["rate"]),
        int(row["n"]),
        int(row["errors"]),
        float(row["ci_lo"]),
    )


def extreme_bands(
    by_g: pd.DataFrame, by_z: pd.DataFrame, floor: int
) -> list[tuple[str, Band]]:
    """(description, band) for the brightest, nearest and most distant galaxies, by position.

    The extreme is picked first; one with fewer than ``floor`` objects (the reporting floor)
    is dropped rather than replaced by a neighbour.
    """
    picks: list[tuple[str, Band]] = []
    g, z = _usable(by_g), _usable(by_z)
    if len(g):
        picks.append((BRIGHTEST, _band(g.loc[g["lo"].idxmin()], "G")))
    if len(z):
        picks.append((NEAREST, _band(z.loc[z["lo"].idxmin()], "z")))
        picks.append((FARTHEST, _band(z.loc[z["lo"].idxmax()], "z")))
    return [(what, band) for what, band in picks if band.n >= floor]
