"""Layer 2 figure: macro-F1 per system with intervals."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt
from astro_triage.dashboard.charts_base import (
    ACCENT,
    NEUTRAL,
    TEXT,
    TEXT_MUTED,
    _base,
)

CAP_SYMBOL = "line-ns"  # vertical tick closing each end of the interval


def layer2_f1(systems: pd.DataFrame) -> go.Figure:
    """Macro-F1 per Layer 2 system with intervals. ``systems``: unit, macro_f1(_lo, _hi, _mark).

    Systems below the reporting floor have no point to draw and are left off the chart.
    Each system is a capped interval plus a labelled point; the label is the F1 value itself.
    """
    shown = systems[systems["macro_f1_mark"] == ""]
    fig = _base(
        56 * max(len(shown), 1) + 80,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        xaxis={
            "tickformat": ".2f",
            "ticks": "outside",
            "title": {
                "text": f"F1 macro, com {fmt.pt_ci_label()} (bootstrap)",
                "font": {"color": TEXT_MUTED},
            },
        },
    )
    for r in shown.itertuples():
        fig.add_trace(
            go.Scatter(
                name=f"{r.unit}: intervalo",
                x=[r.macro_f1_lo, r.macro_f1_hi],
                y=[r.unit] * 2,
                mode="lines+markers",
                line={"color": NEUTRAL, "width": 2},
                marker={"symbol": CAP_SYMBOL, "size": 10, "color": NEUTRAL},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                name=f"{r.unit}: F1 macro",
                x=[r.macro_f1],
                y=[r.unit],
                mode="markers+text",
                marker={"size": 14, "color": ACCENT, "symbol": "circle"},
                text=[fmt.pt_num(r.macro_f1, 2)],
                textposition="top center",
                textfont={"color": TEXT},
                cliponaxis=False,
                hovertemplate=f"<b>{r.unit}</b><br>F1 macro: {fmt.pt_num(r.macro_f1)}<br>"
                f"{fmt.pt_ci_label()}: {fmt.pt_num(r.macro_f1_lo)} a "
                f"{fmt.pt_num(r.macro_f1_hi)}<extra></extra>",
            )
        )
    fig.update_yaxes(autorange="reversed", ticks="outside", showgrid=False)
    return fig
