"""Error-pair bars of the overview: errors per class pair with both directions and share."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt
from astro_triage.dashboard.charts_base import (
    ACCENT,
    FONT_LABEL,
    SANS,
    SOFT,
    TEXT,
    _base,
    rgba,
)

_RIGHT_MARGIN_BARS = 40
_HEADROOM = 1.15
_REST_OPACITY = 0.55
_ROW_PX = 44


def _share(row: pd.Series, digits: int) -> str:
    return "sem dado" if math.isnan(row["share"]) else fmt.pt_pct(row["share"], digits)


def pair_text(row: pd.Series) -> str:
    """'6.399 + 2.278 = 8.677 · 82%' (larger direction first)."""
    first, second = sorted((int(row["n_ab"]), int(row["n_ba"])), reverse=True)
    n = fmt.pt_int(row["n"])
    return f"{fmt.pt_int(first)} + {fmt.pt_int(second)} = {n} · {_share(row, 0)}"


def _pair_hover(row: pd.Series) -> str:
    both = f"{fmt.pt_int(row['n_ab'])} e {fmt.pt_int(row['n_ba'])}"
    return (
        f"{row['pair']}<br>erros neste par: {fmt.pt_int(row['n'])}"
        f"<br>nos dois sentidos: {both}<br>conta: {pair_text(row)}"
        f"<br>participação no total: {_share(row, 1)}"
    )


def error_bars(pairs: pd.DataFrame) -> go.Figure:
    """Horizontal bars of errors per class pair, sorted by count; the largest is highlighted."""
    pairs = pairs.sort_values("n", ascending=False, kind="stable")
    colors = [
        ACCENT if i == 0 else rgba(SOFT, _REST_OPACITY) for i in range(len(pairs))
    ]
    top = float(pairs["n"].max()) if len(pairs) else 1.0
    fig = _base(
        _ROW_PX * len(pairs) + 24,
        grid="",
        margin={"l": 8, "r": _RIGHT_MARGIN_BARS, "t": 8, "b": 8},
        yaxis={
            "autorange": "reversed",
            "categoryorder": "array",
            "categoryarray": list(pairs["pair"]),
        },
        xaxis={"visible": False, "range": [0, max(top, 1) * _HEADROOM]},
    )
    fig.add_trace(
        go.Bar(
            x=pairs["n"],
            y=pairs["pair"],
            orientation="h",
            name="Erros",
            width=0.42,
            marker={"color": colors},
            text=[_share(r, 0) for _, r in pairs.iterrows()],
            textposition="outside",
            textfont={"family": SANS, "color": TEXT, "size": FONT_LABEL},
            cliponaxis=False,
            hovertext=[_pair_hover(r) for _, r in pairs.iterrows()],
            hoverinfo="text",
        )
    )
    return fig
