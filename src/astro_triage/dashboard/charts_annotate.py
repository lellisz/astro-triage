"""Annotation helpers shared by the dashboard charts: titles, direct labels, reference lines.

``section_title(fig, text)``: serif title at the left edge.
``direct_label(fig, x, y, text, *, xref, yref, xanchor, yanchor, color, size,
xshift, yshift, arrow=(dx_px, dy_px))``: label next to a mark, optional thin arrow.
``ref_line(fig, *, x=None, y=None, text="", side="end", dash="dash", color=NEUTRAL)``: dashed
line across the plot with a label.
``interval_row(fig, lo, hi, y, name="")``: horizontal interval with short end caps.
``value_label(texts)``: trace kwargs for value labels; ``category_axis(categories)``: ordered yaxis.
"""

from __future__ import annotations

import plotly.graph_objects as go

from astro_triage.dashboard.charts_base import (
    FONT_ANNOT,
    FONT_LABEL,
    FONT_TITLE,
    NEUTRAL,
    SANS,
    SHORT_TICK,
    TEXT,
    TEXT_MUTED,
)


def section_title(fig: go.Figure, text: str) -> None:
    """Inter 600 title aligned to the left edge of the chart (the top margin grows to fit)."""
    fig.update_layout(
        title={
            "text": text,
            "x": 0,
            "xref": "container",
            "xanchor": "left",
            "y": 1,
            "yref": "container",
            "yanchor": "top",
            "automargin": True,
            "font": {"family": SANS, "size": FONT_TITLE, "color": TEXT, "weight": 600},
        }
    )


def direct_label(
    fig: go.Figure,
    x: float,
    y: float | str,
    text: str,
    *,
    xref: str = "x",
    yref: str = "y",
    xanchor: str = "left",
    yanchor: str = "middle",
    serif: bool = False,
    italic: bool = False,
    color: str = TEXT_MUTED,
    size: int = FONT_LABEL,
    xshift: int = 0,
    yshift: int = 0,
    arrow: tuple[int, int] | None = None,
) -> None:
    """Direct label at (``x``, ``y``); ``serif``/``italic`` are accepted for old callers and ignored (always Inter, regular).

    ``arrow=(dx_px, dy_px)`` puts the text that many pixels from the point and draws a thin
    arrow to it. ``xref``/``yref`` accept ``"paper"`` for plot-relative placement.
    """
    arrow_style = (
        {
            "showarrow": True,
            "ax": arrow[0],
            "ay": arrow[1],
            "arrowhead": 2,
            "arrowsize": 0.8,
            "arrowwidth": 1,
            "arrowcolor": TEXT_MUTED,
            "standoff": SHORT_TICK + SHORT_TICK,
        }
        if arrow
        else {"showarrow": False}
    )
    fig.add_annotation(
        x=x,
        y=y,
        xref=xref,
        yref=yref,
        xanchor=xanchor,
        yanchor=yanchor,
        xshift=xshift,
        yshift=yshift,
        align="left" if xanchor == "left" else "right",
        text=text,
        font={
            "family": SANS,
            "size": size,
            "color": color,
            "style": "normal",
        },
        **arrow_style,
    )


def ref_line(
    fig: go.Figure,
    *,
    x: float | None = None,
    y: float | None = None,
    text: str = "",
    side: str = "end",
    dash: str = "dash",
    color: str = NEUTRAL,
) -> None:
    """Dashed reference line across the whole plot at ``x`` or ``y`` (exactly one), labelled.

    ``side``: ``"end"`` (right / top) or ``"start"`` (left / bottom) end of the line holds
    the label.
    """
    if (x is None) == (y is None):
        raise ValueError("ref_line needs exactly one of x or y")
    end = side == "end"
    line = {"color": color, "width": 1, "dash": dash}
    if y is not None:
        fig.add_shape(
            type="line", xref="paper", x0=0, x1=1, yref="y", y0=y, y1=y, line=line
        )
        label_at = {"x": 1 if end else 0, "y": y, "xref": "paper", "yref": "y"}
        anchors = {"xanchor": "right" if end else "left", "yanchor": "bottom"}
    else:
        fig.add_shape(
            type="line", yref="paper", y0=0, y1=1, xref="x", x0=x, x1=x, line=line
        )
        label_at = {"x": x, "y": 1 if end else 0, "xref": "x", "yref": "paper"}
        anchors = {"xanchor": "left", "yanchor": "top" if end else "bottom"}
    if text:
        direct_label(
            fig,
            label_at["x"],
            label_at["y"],
            text,
            xref=label_at["xref"],
            yref=label_at["yref"],
            size=FONT_ANNOT,
            xshift=0 if y is not None else SHORT_TICK,
            yshift=2 if y is not None else 0,
            **anchors,
        )


def interval_row(fig: go.Figure, lo: float, hi: float, y: str, name: str = "") -> None:
    """Horizontal interval ``lo``..``hi`` on row ``y``: thin line with short end caps."""
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[y, y],
            mode="lines+markers",
            name=name or f"Intervalo: {y}",
            line={"color": NEUTRAL, "width": 2},
            marker={
                "symbol": "line-ns",
                "size": 9,
                "color": NEUTRAL,
                "line": {"width": 2, "color": NEUTRAL},
            },
            hoverinfo="skip",
        )
    )


def value_label(texts: list[str]) -> dict:
    """Trace kwargs (markers plus a direct value label above each marker)."""
    return {
        "mode": "markers+text",
        "text": texts,
        "textposition": "top center",
        "textfont": {"family": SANS, "size": FONT_LABEL, "color": TEXT},
        "cliponaxis": False,
    }


def category_axis(categories: list[str]) -> dict:
    """``yaxis`` layout that keeps ``categories`` in the given order, first one on top."""
    return {
        "autorange": "reversed",
        "categoryorder": "array",
        "categoryarray": list(categories),
    }
