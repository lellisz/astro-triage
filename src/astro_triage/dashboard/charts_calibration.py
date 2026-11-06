"""Calibration figures: confidence histogram, reliability curves, abstention."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt
from astro_triage.dashboard.charts_annotate import direct_label
from astro_triage.dashboard.charts_base import (
    ACCENT,
    ERROR,
    FONT_LABEL,
    NEUTRAL,
    SANS,
    SYMBOL,
    TEXT,
    TEXT_SOFT,
    _base,
    label,
)

_RATE_COLS = {"error_share", "ci_lo", "ci_hi"}


def _wrong_hover(hist: pd.DataFrame) -> list[str]:
    """Per bin: errors out of objects, plus the error rate with its interval when present."""
    rows = []
    for b in hist.itertuples():
        text = f"erros: {fmt.pt_int(b.n_wrong)} de {fmt.pt_int(b.n)}"
        if _RATE_COLS <= set(hist.columns) and b.n:
            text += "<br>taxa de erro: " + fmt.pt_pct_interval(
                b.error_share, b.ci_lo, b.ci_hi
            )
        rows.append(f"confiança {b.lo:g}–{b.hi:g}<br>{text}")
    return rows


def confidence_hist(hist: pd.DataFrame) -> go.Figure:
    """Stacked bars per confidence bin: right (violet) and wrong (``ERROR`` colour).

    A text-only series prints the object count of each bin above its bar.
    """
    names = [f"{b.lo:g}–{b.hi:g}" for b in hist.itertuples()]
    fig = _base(360, barmode="stack", yaxis={"type": "log"}, hovermode="x")
    fig.add_trace(
        go.Bar(
            x=names,
            y=hist["n"] - hist["n_wrong"],
            marker={"color": ACCENT},
            width=0.62,
            name="acertos",
            hovertemplate="confiança %{x}<br>acertos: %{y:,d}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=names,
            y=hist["n_wrong"],
            marker={"color": ERROR},
            width=0.62,
            name="erros",
            hovertext=_wrong_hover(hist),
            hoverinfo="text",
        )
    )
    shown = hist[hist["n"] > 0]
    fig.add_trace(
        go.Scatter(
            x=[f"{b.lo:g}–{b.hi:g}" for b in shown.itertuples()],
            y=shown["n"],
            mode="text",
            name="objetos por faixa",
            text=[fmt.pt_int(n) for n in shown["n"]],
            textposition="top center",
            textfont={"family": SANS, "size": FONT_LABEL, "color": TEXT_SOFT},
            cliponaxis=False,
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title={"text": "confiança máxima do modelo"})
    fig.update_yaxes(title={"text": "objetos (escala log)"}, tickformat=",d")
    return fig


def calibration(points: dict[str, pd.DataFrame]) -> go.Figure:
    """One-vs-rest reliability curves before and after calibration, one symbol per class.

    The legend has one entry per series (not per class); the class shows in the hover,
    together with the number of objects in the bin when the data carry it.
    """
    fig = _base(430, xaxis={"range": [0, 1]}, yaxis={"range": [0, 1]})
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfeita",
            hoverinfo="skip",
            line={"color": NEUTRAL, "width": 1, "dash": "dash"},
        )
    )
    direct_label(
        fig, 0.36, 0.55, "calibração perfeita", xanchor="right", yanchor="bottom",
        yshift=4, color=TEXT,
    )  # fmt: skip
    series = (
        ("Antes da calibração", "raw_mean_pred", "raw_observed", NEUTRAL, "dash"),
        ("Depois (temperatura)", "mean_pred", "observed", ACCENT, "solid"),
    )
    for name, x_col, y_col, color, dash in series:
        for i, (cls, df) in enumerate(points.items()):
            if x_col not in df:
                continue
            counts = df["n"] if "n" in df else pd.Series([None] * len(df))
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode="lines+markers",
                    line={"color": color, "width": 2, "dash": dash},
                    marker={"color": color, "size": 11, "symbol": SYMBOL[cls]},
                    name=name,
                    legendgroup=name,
                    showlegend=i == 0,
                    customdata=counts.map(
                        lambda n: (
                            ""
                            if n is None
                            else f"<br>objetos na faixa: {fmt.pt_int(n)}"
                        )
                    ),
                    hovertemplate=f"{name}<br>{label(cls)}<br>previsto %{{x:.2f}}"
                    "<br>observado %{y:.2f}%{customdata}<extra></extra>",
                )
            )
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "y": 1.1,
            "font": {"family": SANS, "size": FONT_LABEL, "color": TEXT},
        },
    )
    fig.update_xaxes(
        title={"text": "probabilidade prevista"}, dtick=0.2, tickformat=".1f"
    )
    fig.update_yaxes(
        title={"text": "frequência observada"}, dtick=0.2, tickformat=".1f"
    )
    return fig
