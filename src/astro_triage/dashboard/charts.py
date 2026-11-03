"""Plotly figures for the dashboard, in the editorial dark style.

Design rules (same as the approved prototype): no colour per class (classes are told apart by
symbol), ice-blue is reserved for errors, everything else is neutral greys.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BG = "#0E1013"
HAIRLINE = "#232831"
TEXT = "#E8EAEE"
TEXT_SOFT = "#D2D8E3"
TEXT_MUTED = "#9AA3B5"
ERROR = "#8EC5FF"
NEUTRAL = "#6B7487"

SYMBOL = {"STAR": "circle", "GALAXY": "square", "QSO": "triangle-up"}
GLYPH = {"STAR": "●", "GALAXY": "■", "QSO": "▲"}
CLASS_PT = {"STAR": "Estrela", "GALAXY": "Galáxia", "QSO": "Quasar"}
MODEL_PT = {
    "lgbm_full": "LightGBM completo",
    "lgbm_noind": "LightGBM sem indicadores de astrometria",
    "dsc_allosmod": "Gaia DSC (allosmod)",
}
MODEL_STYLE = {
    "lgbm_full": {"color": TEXT, "symbol": "circle"},
    "lgbm_noind": {"color": TEXT_MUTED, "symbol": "square"},
    "dsc_allosmod": {"color": NEUTRAL, "symbol": "diamond"},
}


def label(cls: str) -> str:
    return f"{GLYPH[cls]} {CLASS_PT[cls]}"


def _base(height: int, **layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"family": "Inter, sans-serif", "color": TEXT_SOFT, "size": 13},
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        showlegend=False,
        hoverlabel={
            "bgcolor": "#161A20",
            "bordercolor": HAIRLINE,
            "font": {"color": TEXT},
        },
        **layout,
    )
    fig.update_xaxes(
        gridcolor=HAIRLINE,
        zerolinecolor=HAIRLINE,
        linecolor=HAIRLINE,
        tickcolor=HAIRLINE,
    )
    fig.update_yaxes(
        gridcolor=HAIRLINE,
        zerolinecolor=HAIRLINE,
        linecolor=HAIRLINE,
        tickcolor=HAIRLINE,
    )
    return fig


def waffle(hits: int) -> go.Figure:
    """100 dots: ``hits`` neutral, the rest ice-blue, filled row by row from the top."""
    idx = np.arange(100)
    x, y = idx % 10, 9 - idx // 10
    colors = [TEXT_MUTED if i < hits else ERROR for i in idx]
    fig = _base(300)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"size": 17, "color": colors},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(visible=False, range=[-0.7, 9.7])
    fig.update_yaxes(visible=False, range=[-0.7, 9.7], scaleanchor="x")
    return fig


def recall_dots(rows: pd.DataFrame) -> go.Figure:
    """Recall per true class with Wilson intervals. ``rows``: cls, rate, ci_lo, ci_hi, n."""
    fig = _base(210, xaxis={"range": [0.8, 1.0], "tickformat": ".0%"})
    for r in rows.itertuples():
        fig.add_trace(
            go.Scatter(
                x=[r.ci_lo, r.ci_hi],
                y=[label(r.cls)] * 2,
                mode="lines",
                line={"color": NEUTRAL, "width": 2},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r.rate],
                y=[label(r.cls)],
                mode="markers",
                marker={"size": 13, "color": TEXT, "symbol": SYMBOL[r.cls]},
                hovertemplate=f"{label(r.cls)}<br>acerta {r.rate:.1%} de {r.n:,} objetos"
                f"<br>IC 95%: {r.ci_lo:.1%}–{r.ci_hi:.1%}<extra></extra>".replace(
                    ",", "."
                ),
            )
        )
    fig.update_yaxes(autorange="reversed")
    return fig


def error_types(rows: pd.DataFrame) -> go.Figure:
    """Horizontal bars of the off-diagonal cells. ``rows``: true, pred, n."""
    names = [
        f"{GLYPH[t]} → {GLYPH[p]}"
        for t, p in zip(rows["true"], rows["pred"], strict=True)
    ]
    fig = _base(30 * len(rows) + 40, xaxis={"showgrid": False})
    fig.add_trace(
        go.Bar(
            x=rows["n"],
            y=names,
            orientation="h",
            marker={"color": ERROR},
            text=rows["n"],
            textposition="outside",
            textfont={"color": TEXT_SOFT},
            customdata=np.stack([rows["true"], rows["pred"]], axis=-1),
            hovertemplate="%{customdata[0]} classificado como %{customdata[1]}: %{x}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(visible=False)
    return fig


def f1_with_ci(metrics: pd.DataFrame, models: list[str]) -> go.Figure:
    """Macro-F1 per model with bootstrap intervals. ``metrics``: model, point, lo, hi."""
    fig = _base(230, xaxis={"tickformat": ".2f"})
    for m in models:
        r = metrics[metrics["model"] == m].iloc[0]
        style = MODEL_STYLE[m]
        fig.add_trace(
            go.Scatter(
                x=[r.lo, r.hi],
                y=[MODEL_PT[m]] * 2,
                mode="lines",
                line={"color": NEUTRAL, "width": 2},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r.point],
                y=[MODEL_PT[m]],
                mode="markers",
                marker={"size": 13, "color": style["color"], "symbol": style["symbol"]},
                hovertemplate=f"{MODEL_PT[m]}<br>F1 macro {r.point:.3f}<br>IC 95%: "
                f"{r.lo:.3f}–{r.hi:.3f}<extra></extra>",
            )
        )
    fig.update_yaxes(autorange="reversed")
    return fig


def paired_diffs(diffs: pd.DataFrame) -> go.Figure:
    """Paired differences with intervals and a zero line. ``diffs``: pair, point, lo, hi."""
    fig = _base(190)
    fig.add_vline(x=0, line={"color": TEXT_MUTED, "width": 1, "dash": "dot"})
    for r in diffs.itertuples():
        fig.add_trace(
            go.Scatter(
                x=[r.lo, r.hi],
                y=[r.pair] * 2,
                mode="lines",
                line={"color": NEUTRAL, "width": 2},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r.point],
                y=[r.pair],
                mode="markers",
                marker={"size": 12, "color": TEXT},
                hovertemplate=f"{r.pair}<br>{r.point:+.3f} (IC 95%: {r.lo:+.3f} a {r.hi:+.3f})"
                "<extra></extra>",
            )
        )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat="+.2f", zeroline=False)
    return fig


def confidence_hist(hist: pd.DataFrame) -> go.Figure:
    """Stacked bars per confidence bin: right (grey) and wrong (ice-blue)."""
    names = [f"{b.lo:g}–{b.hi:g}" for b in hist.itertuples()]
    fig = _base(300, barmode="stack", yaxis={"type": "log"})
    fig.add_trace(
        go.Bar(
            x=names,
            y=hist["n"] - hist["n_wrong"],
            marker={"color": NEUTRAL},
            name="acertos",
            hovertemplate="confiança %{x}<br>acertos: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=names,
            y=hist["n_wrong"],
            marker={"color": ERROR},
            name="erros",
            hovertemplate="confiança %{x}<br>erros: %{y}<extra></extra>",
        )
    )
    fig.update_xaxes(
        title={"text": "confiança máxima do modelo", "font": {"color": TEXT_MUTED}}
    )
    fig.update_yaxes(
        title={"text": "objetos (escala log)", "font": {"color": TEXT_MUTED}}
    )
    return fig


def calibration(points: dict[str, pd.DataFrame]) -> go.Figure:
    """One-vs-rest reliability curves, one marker symbol per class."""
    fig = _base(320, xaxis={"range": [0, 1]}, yaxis={"range": [0, 1]})
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            hoverinfo="skip",
            line={"color": TEXT_MUTED, "width": 1, "dash": "dot"},
        )
    )
    for cls, df in points.items():
        fig.add_trace(
            go.Scatter(
                x=df["mean_pred"],
                y=df["observed"],
                mode="lines+markers",
                line={"color": TEXT_SOFT, "width": 1},
                marker={"color": TEXT_SOFT, "size": 9, "symbol": SYMBOL[cls]},
                name=label(cls),
                hovertemplate=f"{label(cls)}<br>previsto %{{x:.2f}}<br>observado %{{y:.2f}}"
                "<extra></extra>",
            )
        )
    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h", "y": 1.08, "font": {"color": TEXT_SOFT}},
    )
    fig.update_xaxes(
        title={"text": "probabilidade prevista", "font": {"color": TEXT_MUTED}}
    )
    fig.update_yaxes(
        title={"text": "frequência observada", "font": {"color": TEXT_MUTED}}
    )
    return fig


def abstention(curve: pd.DataFrame, marked: float) -> go.Figure:
    """Accuracy on the kept rows against coverage, with one threshold highlighted."""
    fig = _base(300, xaxis={"tickformat": ".0%"}, yaxis={"tickformat": ".1%"})
    fig.add_trace(
        go.Scatter(
            x=curve["coverage"],
            y=curve["accuracy_kept"],
            mode="lines",
            line={"color": TEXT_SOFT, "width": 2},
            customdata=curve["threshold"],
            hovertemplate="limiar %{customdata:.2f}<br>cobertura %{x:.1%}<br>acerto %{y:.1%}"
            "<extra></extra>",
        )
    )
    row = curve.iloc[(curve["threshold"] - marked).abs().argmin()]
    fig.add_trace(
        go.Scatter(
            x=[row.coverage],
            y=[row.accuracy_kept],
            mode="markers",
            marker={"size": 12, "color": ERROR},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(
        title={
            "text": "parte dos objetos que o sistema decide",
            "font": {"color": TEXT_MUTED},
        }
    )
    fig.update_yaxes(
        title={"text": "acerto entre os decididos", "font": {"color": TEXT_MUTED}}
    )
    return fig


def importance(df: pd.DataFrame) -> go.Figure:
    """Gain share per feature, horizontal bars in neutral grey."""
    fig = _base(26 * len(df) + 30, xaxis={"tickformat": ".0%", "showgrid": False})
    fig.add_trace(
        go.Bar(
            x=df["share"],
            y=df["feature"],
            orientation="h",
            marker={"color": NEUTRAL},
            hovertemplate="%{y}: %{x:.1%} do ganho<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _mollweide_outline() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(-math.pi / 2, math.pi / 2, 181)
    ex, ey = 2 * math.sqrt(2) * np.cos(t), math.sqrt(2) * np.sin(t)
    return np.concatenate([ex, -ex[::-1]]), np.concatenate([ey, ey[::-1]])


def sky_map(
    pixels: pd.DataFrame,
    points: pd.DataFrame | None,
    pixel_stats: pd.DataFrame,
    floor: int,
) -> go.Figure:
    """Mollweide map: pixel centres per fold, test objects (errors in ice-blue)."""
    fig = _base(520, xaxis={"visible": False, "range": [-2.95, 2.95]})
    fig.update_yaxes(visible=False, range=[-1.5, 1.5], scaleanchor="x")
    ox, oy = _mollweide_outline()
    fig.add_trace(
        go.Scatter(
            x=ox,
            y=oy,
            mode="lines",
            line={"color": HAIRLINE, "width": 1},
            hoverinfo="skip",
        )
    )
    shades = {"train": "#2A303B", "val": "#4A5263", "test": "#9AA3B5"}
    for fold, color in shades.items():
        sub = pixels[pixels["fold"] == fold]
        fig.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                marker={"size": 4, "color": color},
                name=fold,
                hoverinfo="skip",
            )
        )
    if points is not None:
        ok, bad = points[~points["error"]], points[points["error"]]
        fig.add_trace(
            go.Scatter(
                x=ok["x"],
                y=ok["y"],
                mode="markers",
                marker={"size": 3, "color": TEXT, "opacity": 0.35},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bad["x"],
                y=bad["y"],
                mode="markers",
                marker={"size": 6, "color": ERROR},
                hoverinfo="skip",
            )
        )
    else:
        # Too many objects to draw: show test pixels, size by objects, blue by error rate.
        fig.add_trace(
            go.Scatter(
                x=pixel_stats["x"],
                y=pixel_stats["y"],
                mode="markers",
                marker={
                    "size": np.clip(np.sqrt(pixel_stats["n"]) * 1.2, 4, 18),
                    "color": pixel_stats["rate"],
                    "colorscale": [[0, "#2A303B"], [1, ERROR]],
                    "showscale": True,
                    "colorbar": {"title": "erro", "tickformat": ".0%"},
                },
                customdata=np.stack([pixel_stats["n"], pixel_stats["errors"]], axis=-1),
                hovertemplate="%{customdata[0]} objetos, %{customdata[1]} erros<extra></extra>",
            )
        )
    return fig


def rate_bars(table: pd.DataFrame) -> go.Figure:
    """Error rate per bin with Wilson intervals (ice-blue: these are errors)."""
    fig = _base(260, yaxis={"tickformat": ".0%"})
    fig.add_trace(
        go.Bar(
            x=table["label"],
            y=table["rate"],
            marker={"color": ERROR},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (table["ci_hi"] - table["rate"]).fillna(0),
                "arrayminus": (table["rate"] - table["ci_lo"]).fillna(0),
                "color": TEXT_MUTED,
                "thickness": 1,
            },
            customdata=np.stack([table["errors"], table["n"]], axis=-1),
            hovertemplate="%{x}<br>%{customdata[0]} erros em %{customdata[1]} objetos"
            "<br>taxa %{y:.1%}<extra></extra>",
        )
    )
    return fig


def galaxy_scatter(df: pd.DataFrame) -> go.Figure:
    """True galaxies in the (z, G) plane; errors on top in ice-blue."""
    fig = _base(380, xaxis={"type": "log"}, yaxis={"autorange": "reversed"})
    ok, bad = df[~df["error"]], df[df["error"]]
    fig.add_trace(
        go.Scatter(
            x=ok["z"].clip(lower=1e-3),
            y=ok["G"],
            mode="markers",
            marker={"size": 4, "color": TEXT, "opacity": 0.3},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bad["z"].clip(lower=1e-3),
            y=bad["G"],
            mode="markers",
            marker={"size": 8, "color": ERROR},
            customdata=np.stack([bad["pred"], bad["confidence"]], axis=-1),
            hovertemplate="z=%{x:.3f}, G=%{y:.1f}<br>classificada como %{customdata[0]} "
            "(confiança %{customdata[1]:.2f})<extra></extra>",
        )
    )
    fig.update_xaxes(
        title={"text": "redshift z (escala log)", "font": {"color": TEXT_MUTED}}
    )
    fig.update_yaxes(
        title={
            "text": "magnitude G (mais fraca embaixo)",
            "font": {"color": TEXT_MUTED},
        }
    )
    return fig


def layer2_f1(systems: pd.DataFrame) -> go.Figure:
    """Macro-F1 per Layer 2 system with intervals. ``systems``: unit, macro_f1(_lo, _hi, _mark).

    Systems below the reporting floor have no point to draw and are left off the chart.
    """
    shown = systems[systems["macro_f1_mark"] == ""]
    fig = _base(40 * max(len(shown), 1) + 50, xaxis={"tickformat": ".2f"})
    for r in shown.itertuples():
        fig.add_trace(
            go.Scatter(
                x=[r.macro_f1_lo, r.macro_f1_hi],
                y=[r.unit] * 2,
                mode="lines",
                line={"color": NEUTRAL, "width": 2},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r.macro_f1],
                y=[r.unit],
                mode="markers",
                marker={"size": 13, "color": TEXT, "symbol": "circle"},
                hovertemplate=f"{r.unit}<br>F1 macro {r.macro_f1:.3f}<br>IC 95%: "
                f"{r.macro_f1_lo:.3f}–{r.macro_f1_hi:.3f}<extra></extra>",
            )
        )
    fig.update_yaxes(autorange="reversed")
    return fig
