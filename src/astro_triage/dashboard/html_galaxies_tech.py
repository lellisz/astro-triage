"""Galáxias page, technical mode: error rate per band on a logarithmic 0,1-100% scale.

Pure: no streamlit. Each row shows n, errors, the rate as a point at its log position and
the Wilson interval as a grey segment; the point takes the highlight colour above ERROR_HIGH.
"""

from __future__ import annotations

import math
from html import escape
from typing import Final

import pandas as pd

from astro_triage.dashboard import fmt, theme
from astro_triage.dashboard.constants import ERROR_HIGH
from astro_triage.dashboard.data_galaxies_z import GalaxyTotals
from astro_triage.dashboard.html_galaxies import (
    TITLE_NEUTRAL,
    TITLE_SPIKE,
    extreme_bands,
    lead_text,
    rate_color,
    rate_text,
    spikes_at_the_ends,
)
from astro_triage.dashboard.html_kit import PERCENT, axis, dot, section, span

LOG_LO_PCT: Final = 0.1
LOG_HI_PCT: Final = 100.0
HOW_TECH: Final = (
    "cada ponto é uma faixa. Quanto mais à direita, maior a taxa de erro. A escala é "
    "logarítmica (cada marca é 10 vezes a anterior), e a linha cinza é a incerteza (Wilson)."
)
_LO_EXP: Final = round(math.log10(LOG_LO_PCT))
_HI_EXP: Final = round(math.log10(LOG_HI_PCT))
_DECADES: Final = tuple(10.0**e for e in range(_LO_EXP, _HI_EXP + 1))


def log_pos(pct: float) -> float:
    """Position in 0-100 of a rate in percent on the log axis; clamped, 0 for rates <= 0."""
    if math.isnan(pct) or pct <= 0:
        return 0.0
    span_log = math.log10(LOG_HI_PCT) - math.log10(LOG_LO_PCT)
    pos = (math.log10(pct) - math.log10(LOG_LO_PCT)) / span_log * PERCENT
    return min(PERCENT, max(0.0, pos))


def log_axis() -> str:
    return axis([(log_pos(v), fmt.pt_g(v) + "%") for v in _DECADES])


def _grid() -> str:
    return "".join(
        f"<div style='position:absolute;top:0;bottom:0;left:{log_pos(v):.2f}%;"
        f"width:1px;background:{theme.GRID}'></div>"
        for v in _DECADES
    )


def _marks(row: pd.Series) -> str:
    if not row["n"]:
        return ""
    rate = float(row["rate"])
    lo, hi = float(row["ci_lo"]) * PERCENT, float(row["ci_hi"]) * PERCENT
    below = rate * PERCENT <= LOG_LO_PCT
    return span(log_pos(lo), log_pos(hi)) + dot(
        log_pos(rate * PERCENT), hollow=below, color=rate_color(rate)
    )


def band_row(row: pd.Series) -> str:
    rate = float(row["rate"]) if row["n"] else math.nan
    return (
        f"<div class='at-rowg'><div>{escape(str(row['label']))}"
        f"<div class='at-cap'>n = {fmt.pt_int(row['n'])}</div></div>"
        f"<div class='at-plot' style='background:none'>{_grid()}{_marks(row)}</div>"
        f"<div class='at-num'>{fmt.pt_int(row['errors'])}</div>"
        f"<div class='at-num at-strong'>{rate_text(rate)}</div></div>"
    )


def band_table(title: str, table: pd.DataFrame) -> str:
    """Titled log table; the 'sem medida' row is left out (it has no position)."""
    rows = "".join(band_row(r) for _, r in table[table["lo"].notna()].iterrows())
    head = (
        f"<div class='at-rowg' style='border-top:none;min-height:30px'><div></div>"
        f"{log_axis()}<div class='at-hdr at-num'>erros</div><div class='at-hdr at-num'>taxa</div></div>"
    )
    return (
        f"<div class='at-strong' style='margin:24px 0 6px'>{escape(title)}</div>"
        f"<div class='at-ov'><div class='at-in'>{head}{rows}</div></div>"
    )


def missing_note(by_g: pd.DataFrame) -> str:
    gone = by_g[by_g["lo"].isna()]
    n = int(gone["n"].sum()) if len(gone) else 0
    note = f"Azul = taxa acima de {fmt.pt_pct(ERROR_HIGH, 0)}."
    if n:
        note += (
            f" Faltam {fmt.pt_int(n)} galáxias sem magnitude medida, "
            "que não entram na parte de brilho."
        )
    return f"<p class='at-cap' style='margin:14px 0 0'>{escape(note)}</p>"


def tech_html(
    tot: GalaxyTotals, by_g: pd.DataFrame, by_z: pd.DataFrame, floor: int
) -> str:
    spike = spikes_at_the_ends(tot.rate, extreme_bands(by_g, by_z, floor))
    title = TITLE_SPIKE if spike else TITLE_NEUTRAL
    return (
        section(title, lead_text(tot, spike), HOW_TECH)
        + band_table("Por brilho (magnitude G; menor = mais brilhante)", by_g)
        + band_table("Por distância (redshift z; maior = mais distante)", by_z)
        + missing_note(by_g)
    )
