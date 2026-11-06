"""Overview figures: waffle, recall, error types, macro-F1, paired differences, importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt
from astro_triage.dashboard.charts_annotate import (
    category_axis,
    interval_row,
    value_label,
)
from astro_triage.dashboard.charts_base import (
    ACCENT,
    CLASS_PT,
    ERROR,
    FONT_LABEL,
    GLYPH,
    MODEL_PT,
    MODEL_STYLE,
    SANS,
    SOFT,
    SYMBOL,
    TEXT_SOFT,
    _base,
    label,
)
from astro_triage.dashboard.charts_overview_paired import (  # noqa: F401
    _DTICKS,
    paired_diffs,
    paired_dtick,
)


def waffle(hits: int) -> go.Figure:
    """100 dots: ``hits`` neutral, the rest in ``ERROR``, filled row by row from the top."""
    idx = np.arange(100)
    x, y = idx % 10, 9 - idx // 10
    colors = [SOFT if i < hits else ERROR for i in idx]
    fig = _base(300)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"size": 20, "color": colors},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(visible=False, range=[-0.7, 9.7])
    fig.update_yaxes(visible=False, range=[-0.7, 9.7], scaleanchor="x")
    return fig


def recall_dots(rows: pd.DataFrame) -> go.Figure:
    """Recall per true class with Wilson intervals. ``rows``: cls, rate, ci_lo, ci_hi, n."""
    fig = _base(260, xaxis={"range": [0.8, 1.0], "tickformat": ".0%"})
    for r in rows.itertuples():
        interval_row(fig, r.ci_lo, r.ci_hi, label(r.cls))
        fig.add_trace(
            go.Scatter(
                x=[r.rate],
                y=[label(r.cls)],
                name=f"Acerto: {label(r.cls)}",
                marker={"size": 14, "color": ACCENT, "symbol": SYMBOL[r.cls]},
                hovertemplate=f"{label(r.cls)}<br>acerta {fmt.pt_pct(r.rate)} de "
                f"{fmt.pt_int(r.n)} objetos<br>{fmt.pt_ci_label()}: {fmt.pt_pct(r.ci_lo)}–"
                f"{fmt.pt_pct(r.ci_hi)}<extra></extra>",
                **value_label([fmt.pt_pct(r.rate)]),
            )
        )
    fig.update_yaxes(autorange="reversed")
    return fig


def error_types(rows: pd.DataFrame) -> go.Figure:
    """Horizontal bars of the off-diagonal cells, largest first. ``rows``: true, pred, n."""
    rows = rows.sort_values("n", ascending=False, kind="stable")
    rows = rows.assign(share=rows["n"] / rows["n"].sum())
    names = [f"{GLYPH[r.true]} → {GLYPH[r.pred]}" for r in rows.itertuples()]
    fig = _base(
        40 * len(rows) + 50,
        grid="",
        xaxis={"visible": False, "range": [0, float(rows["n"].max()) * 1.3]},
        yaxis=category_axis(names),
    )
    fig.add_trace(
        go.Bar(
            x=rows["n"],
            y=names,
            orientation="h",
            name="Erros por direção",
            width=0.5,
            marker={"color": ERROR},
            text=[
                f"{fmt.pt_int(r.n)} · {fmt.pt_pct(r.share, 0)}"
                for r in rows.itertuples()
            ],
            textposition="outside",
            textfont={"family": SANS, "size": FONT_LABEL, "color": TEXT_SOFT},
            cliponaxis=False,
            hovertext=[
                f"{CLASS_PT[r.true]} classificada como {CLASS_PT[r.pred]}<br>"
                f"{fmt.pt_int(r.n)} objetos · {fmt.pt_pct(r.share, 1)} dos erros"
                for r in rows.itertuples()
            ],
            hoverinfo="text",
        )
    )
    return fig


def f1_with_ci(metrics: pd.DataFrame, models: list[str]) -> go.Figure:
    """Macro-F1 per model with bootstrap intervals. ``metrics``: model, point, lo, hi."""
    fig = _base(290, xaxis={"tickformat": ".2f"})
    for m in models:
        r = metrics[metrics["model"] == m].iloc[0]
        interval_row(fig, r.lo, r.hi, MODEL_PT[m])
        fig.add_trace(
            go.Scatter(
                x=[r.point],
                y=[MODEL_PT[m]],
                name=f"F1 macro: {MODEL_PT[m]}",
                **value_label([fmt.pt_num(r.point)]),
                marker={"size": 13, **MODEL_STYLE[m]},
                hovertemplate=f"{MODEL_PT[m]}<br>F1 macro {fmt.pt_num(r.point)}<br>"
                f"{fmt.pt_ci_label()}: {fmt.pt_num(r.lo)}–{fmt.pt_num(r.hi)}<extra></extra>",
            )
        )
    fig.update_yaxes(autorange="reversed")
    return fig


def importance(df: pd.DataFrame) -> go.Figure:
    """Gain share per feature, horizontal bars in violet, largest first."""
    df = df.sort_values("share", ascending=False, kind="stable")
    fig = _base(
        32 * len(df) + 40,
        grid="x",
        xaxis={"tickformat": ".0%", "range": [0, max([*df["share"], 0.01]) * 1.18]},
        yaxis=category_axis(list(df["feature"])),
    )
    fig.add_trace(
        go.Bar(
            x=df["share"],
            y=df["feature"],
            orientation="h",
            name="Participação no ganho",
            width=0.55,
            marker={"color": ACCENT},
            text=[fmt.pt_pct(v, 1) for v in df["share"]],
            textposition="outside",
            textfont={"family": SANS, "size": FONT_LABEL, "color": TEXT_SOFT},
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%} do ganho<extra></extra>",
        )
    )
    return fig
