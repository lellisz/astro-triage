"""Overview figures: macro-F1, paired differences, importance."""

from __future__ import annotations

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
    FONT_LABEL,
    MODEL_PT,
    MODEL_STYLE,
    SANS,
    TEXT_SOFT,
    _base,
)
from astro_triage.dashboard.charts_overview_paired import (  # noqa: F401
    _DTICKS,
    paired_diffs,
    paired_dtick,
)


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
