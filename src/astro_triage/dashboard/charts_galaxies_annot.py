"""Shared axis, hover, bands and separator pieces of the galaxy rate charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt, theme
from astro_triage.dashboard.charts_annot import note, rgba
from astro_triage.dashboard.charts_base import DASH, TEXT

RATE_LOG_RANGE = (-3.0, 0.05)  # log10 of the shared y limits, 0.1% to ~112%
RATE_TICKS = (0.001, 0.01, 0.1, 1.0)
CAP_WIDTH = 5  # px, cap on each interval whisker
BAND_ALPHA = 0.045

RATE_HOVER = (
    "<b>%{customdata[0]}</b><br>%{customdata[1]:,d} erros em %{customdata[2]:,d} objetos"
    "<br>taxa de erro: %{customdata[5]:.2%}"
    f"<br>{fmt.pt_ci_label()} (Wilson): %{{customdata[3]:.2%}} a %{{customdata[4]:.2%}}"
    "<extra></extra>"
)


def rate_axis(**extra: object) -> dict[str, object]:
    """Log y axis shared by every rate panel; ticks read 0,1% 1% 10% 100%."""
    return {
        "type": "log",
        "range": list(RATE_LOG_RANGE),
        "tickvals": list(RATE_TICKS),
        "ticktext": [fmt.pt_pct(v, 1 if v < 0.01 else 0) for v in RATE_TICKS],
        "tickformat": ".1%",
        "ticks": "outside",
        **extra,
    }


def interval_bars(
    rate: pd.Series, lo: pd.Series, hi: pd.Series, thickness: float = 1.25
) -> dict[str, object]:
    """Asymmetric error bars with caps for Wilson intervals."""
    return {
        "type": "data",
        "symmetric": False,
        "array": (hi - rate).fillna(0),
        "arrayminus": (rate - lo).fillna(0),
        "color": DASH,
        "thickness": thickness,
        "width": CAP_WIDTH,
    }


def rate_customdata(sub: pd.DataFrame) -> np.ndarray:
    """label, errors, n, ci_lo, ci_hi, rate: the columns ``RATE_HOVER`` reads."""
    return np.stack(
        [
            sub["label"],
            sub["errors"],
            sub["n"],
            sub["ci_lo"],
            sub["ci_hi"],
            sub["rate"],
        ],
        axis=-1,
    )


def band_trace(slots: list[float]) -> go.Bar:
    """Faint full-height bands on an overlay axis (``yaxis2``), one per slot given."""
    return go.Bar(
        name="faixas alternadas",
        x=slots,
        y=[1] * len(slots),
        width=1,
        marker={"color": rgba(TEXT, BAND_ALPHA), "line": {"width": 0}},
        yaxis="y2",
        hoverinfo="skip",
    )


def overlay_axis() -> dict[str, object]:
    return {"overlaying": "y", "range": [0, 1], "visible": False, "fixedrange": True}


def separator(fig: go.Figure, x: float, text: str) -> None:
    """Dotted vertical rule between blocks with a direct label above it."""
    fig.add_trace(
        go.Scatter(
            name="separador dos blocos",
            x=[x, x],
            y=[0, 1],
            yaxis="y2",
            mode="lines",
            line={"color": theme.DASH, "width": 1, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    note(fig, x, 1, text, xref="x", yref="paper", yshift=8)
