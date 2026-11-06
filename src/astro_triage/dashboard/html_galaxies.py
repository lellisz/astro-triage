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
from astro_triage.dashboard.data_galaxies_z import GalaxyTotals
from astro_triage.dashboard.html_kit import h_bar, section

# Rates below this share are written with two decimals.
SMALL_RATE: Final = 0.01

TITLE_SPIKE: Final = "Galáxias: o erro dispara nas pontas"
TITLE_NEUTRAL: Final = "Galáxias: onde o modelo erra"
HOW_SIMPLE: Final = "cada barra é a parte das galáxias que o modelo erra."
WRAP: Final = f"max-width:{theme.READING_WIDTH_PX}px"


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
        picks.append(("As mais brilhantes", _band(g.loc[g["lo"].idxmin()], "G")))
    if len(z):
        picks.append(("As mais próximas", _band(z.loc[z["lo"].idxmin()], "z")))
        picks.append(("As mais distantes", _band(z.loc[z["lo"].idxmax()], "z")))
    return [(what, band) for what, band in picks if band.n >= floor]


def spikes_at_the_ends(total_rate: float, extremes: list[tuple[str, Band]]) -> bool:
    """True only if the Wilson lower bound of every extreme band is above the overall rate."""
    lows = [b.ci_lo for _, b in extremes]
    return (
        bool(lows)
        and not math.isnan(total_rate)
        and not any(math.isnan(v) for v in lows)
        and min(lows) > total_rate
    )


def lead_text(tot: GalaxyTotals, spike: bool) -> str:
    text = (
        f"Em {fmt.pt_int(tot.n)} galáxias de teste o modelo errou "
        f"{fmt.pt_int(tot.errors)} ({fmt.pt_pct(tot.rate)}): "
        f"{fmt.pt_int(tot.to_qso)} viraram quasar e {fmt.pt_int(tot.to_star)} viraram estrela."
    )
    return f"{text} Mas o erro não é parelho." if spike else text


def simple_html(
    tot: GalaxyTotals, by_g: pd.DataFrame, by_z: pd.DataFrame, floor: int
) -> str:
    extremes = extreme_bands(by_g, by_z, floor)
    spike = spikes_at_the_ends(tot.rate, extremes)
    title = TITLE_SPIKE if spike else TITLE_NEUTRAL
    bars = [("Todas as galáxias", tot.rate)] + [
        (f"{what} ({band.name})", band.rate) for what, band in extremes
    ]
    rows = "".join(
        h_bar(name, 0.0 if math.isnan(rate) else rate, rate_text(rate), rate_color(rate))
        for name, rate in bars
    )
    return (
        section(title, lead_text(tot, spike), HOW_SIMPLE)
        + f"<div style='{WRAP}'>{rows}</div>"
    )
