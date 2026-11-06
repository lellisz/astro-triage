"""Triage figure: accuracy among the objects the model decides, against the share it decides."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt, theme
from astro_triage.dashboard.charts_annot import rgba
from astro_triage.dashboard.charts_annotate import direct_label
from astro_triage.dashboard.charts_base import (
    ACCENT,
    FONT_ANNOT,
    FONT_LABEL,
    FONT_TICK,
    SOFT,
    TEXT,
    TEXT_MUTED,
    _base,
)

DEFAULT_FLOOR = 0.96
_TOP = 1.0
_TRANSPARENT = "rgba(0,0,0,0)"
_ARROW_PX = 44
_HATCH_ALPHA = 0.35
_HATCH_SOLIDITY = 0.08
_CURVE_PX = 3
_BOTTOM_PX = 100
_NOTE_DROP = 72
_NOTE_GAP = 4
_LABEL_BOX = {"bgcolor": theme.BG, "borderpad": 3}


def axis_floor(accuracy: pd.Series) -> float:
    """0.96 when every value respects it, else the data minimum rounded down to 2 decimals."""
    values = accuracy.dropna()
    if values.empty:
        return DEFAULT_FLOOR
    low = float(values.min())
    if low >= DEFAULT_FLOOR:
        return DEFAULT_FLOOR
    return max(math.floor(low * 100 + 1e-9) / 100, 0.0)


def _dtick(floor: float) -> float:
    span = _TOP - floor
    for limit, tick in ((0.08, 0.01), (0.2, 0.02), (0.5, 0.05)):
        if span <= limit:
            return tick
    return 0.1


def _point(
    x: float, y: float, name: str, filled: bool, color: str, size: int
) -> go.Scatter:
    return go.Scatter(
        x=[x],
        y=[y],
        name=name,
        mode="markers",
        marker={
            "size": size,
            "symbol": "circle" if filled else "circle-open",
            "color": color,
            "line": {"color": color, "width": 2},
        },
        cliponaxis=False,
        hoverinfo="skip",
        showlegend=False,
    )


def triage_curve(curve: pd.DataFrame, summary: dict) -> go.Figure:
    floor = axis_floor(curve["accuracy_kept"])
    fig = _base(340, margin={"l": 72, "r": 40, "t": 44, "b": _BOTTOM_PX})
    cov, acc = summary["coverage"], summary["accuracy_kept"]
    all_cov, all_acc = summary["decide_all_coverage"], summary["decide_all_accuracy"]
    fig.add_trace(
        go.Scatter(
            x=[cov, _TOP, _TOP, cov],
            y=[floor, floor, _TOP, _TOP],
            name="Parte enviada ao espectro",
            mode="lines",
            fill="toself",
            fillpattern={
                "shape": "/",
                "fgcolor": rgba(theme.DASH, _HATCH_ALPHA),
                "bgcolor": _TRANSPARENT,
                "size": 14,
                "solidity": _HATCH_SOLIDITY,
            },
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve["coverage"],
            y=curve["accuracy_kept"],
            name="Curva de abstenção",
            mode="lines",
            line={"color": SOFT, "width": _CURVE_PX},
            customdata=curve["threshold"],
            hovertemplate="limiar %{customdata:.2f}<br>decide %{x:.1%}<br>acerta %{y:.1%}"
            "<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(_point(cov, acc, "Limiar", True, ACCENT, 17))
    fig.add_trace(_point(all_cov, all_acc, "Decidir quase tudo", False, SOFT, 13))
    direct_label(
        fig, cov, acc,
        f"Com limiar {fmt.pt_num(summary['threshold'], 2)}:<br>decide {fmt.pt_pct(cov)} "
        f"e acerta {fmt.pt_pct(acc)}",
        xanchor="right", arrow=(-_ARROW_PX, _ARROW_PX), color=TEXT, size=FONT_LABEL,
    )  # fmt: skip
    direct_label(
        fig, all_cov, all_acc, f"Decidir tudo: acerta {fmt.pt_pct(all_acc)}",
        xanchor="right", arrow=(-_ARROW_PX, -_ARROW_PX // 2), color=TEXT, size=FONT_LABEL,
    )  # fmt: skip
    direct_label(
        fig, (cov + _TOP) / 2, 1, "vai para espectro", yref="paper", xanchor="center",
        yanchor="bottom", yshift=_NOTE_GAP, size=FONT_ANNOT,
    )  # fmt: skip
    fig.layout.annotations[-1].update(**_LABEL_BOX)
    fig.update_xaxes(
        title={"text": "Parte decidida pelo modelo"},
        range=[0, _TOP],
        tickformat=".0%",
        dtick=0.2,
    )
    fig.update_yaxes(
        title={"text": "Acerto entre os decididos"},
        range=[floor, _TOP],
        tickformat=".0%",
        dtick=_dtick(floor),
    )
    fig.add_annotation(
        x=0, y=0, xref="paper", yref="paper", xanchor="left", yanchor="top",
        yshift=-_NOTE_DROP, showarrow=False,
        text=f"O eixo vertical começa em {fmt.pt_pct(floor, 0)}, não em zero.",
        font={"family": theme.FONT_SANS, "size": FONT_TICK, "color": TEXT_MUTED},
    )  # fmt: skip
    return fig
