"""Paired-difference figure of the overview: intervals around a zero reference."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt
from astro_triage.dashboard.charts_annotate import (
    direct_label,
    interval_row,
    ref_line,
    value_label,
)
from astro_triage.dashboard.charts_base import ACCENT, DASH, FONT_ANNOT, TEXT, _base

_DTICKS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def paired_dtick(lo: float, hi: float) -> float:
    """Smallest tick step in ``_DTICKS`` that gives at most 8 ticks over ``lo``..``hi``.

    A fixed step (never below 0.001, the precision of the ``+.3f`` labels) keeps two ticks
    from rendering the same label.
    """
    span = hi - lo
    if not math.isfinite(span) or span <= 0:
        return _DTICKS[0]
    return next((d for d in _DTICKS if span / d <= 8), _DTICKS[-1])


def paired_diffs(diffs: pd.DataFrame) -> go.Figure:
    """Paired differences with intervals and a zero line. ``diffs``: pair, point, lo, hi."""
    fig = _base(290, margin={"l": 8, "r": 16, "t": 62, "b": 8})
    ref_line(fig, x=0, dash="dot", color=DASH)
    direct_label(
        fig, 0, 1, "sem diferença", yref="paper", xanchor="center", yanchor="bottom",
        color=TEXT, size=FONT_ANNOT, yshift=12,
    )  # fmt: skip
    for r in diffs.itertuples():
        interval_row(fig, r.lo, r.hi, r.pair)
        fig.add_trace(
            go.Scatter(
                x=[r.point],
                y=[r.pair],
                name=f"Diferença: {r.pair}",
                **value_label([fmt.pt_num(r.point, signed=True)]),
                marker={"size": 13, "color": ACCENT},
                hovertemplate=f"{r.pair}<br>{fmt.pt_num(r.point, signed=True)} "
                f"({fmt.pt_ci_label()}: {fmt.pt_num(r.lo, signed=True)} a "
                f"{fmt.pt_num(r.hi, signed=True)})<extra></extra>",
            )
        )
    fig.update_yaxes(autorange="reversed")
    lo, hi = min(0.0, float(diffs["lo"].min())), max(0.0, float(diffs["hi"].max()))
    fig.update_xaxes(
        tickformat="+.3f", dtick=paired_dtick(lo, hi), tick0=0, zeroline=False
    )
    return fig
