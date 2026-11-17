"""Deep-learning comparison charts: paired differences with intervals, and risk-coverage.

Pure: take parsed data, return Plotly figures. No Streamlit, no HTML.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt
from astro_triage.dashboard.charts_base import (
    ACCENT,
    FONT_LABEL,
    GRID,
    SANS,
    SOFT,
    TEXT,
    TEXT_MUTED,
    _base,
    label,
    rgba,
)
from astro_triage.dashboard.html_dl import pair_label, system_label

PANEL_CLASSES = ("QSO", "GALAXY")
_ROW_PX = 40
_PANEL_PX = 60
_BAND_OPACITY = 0.18
_DASHES = ("solid", "dash", "dot", "dashdot")
_LEGEND_RAW = "bruto, sem reponderação"


def pair_points(
    results: dict,
    prior: str,
    completeness: str,
    cls: str,
    roles: tuple[str, ...] | None = None,
) -> list[dict]:
    """Rows ``pair, d, lo, hi, delta, verdict`` for one class; pairs without a usable cell skipped.

    ``roles`` keeps only the pairs whose cell has one of those roles (None keeps all).
    """
    rows: list[dict] = []
    for pair, priors in results.get("compare", {}).get("pairs", {}).items():
        full = priors.get(prior, {}).get(completeness, {})
        if roles is not None and full.get("role") not in roles:
            continue
        cell = full.get("classes", {}).get(cls)
        if not cell or cell.get("na") or cell.get("ci") is None:
            continue
        rows.append(
            {
                "pair": pair,
                "d": cell["d"],
                "lo": cell["ci"][0],
                "hi": cell["ci"][1],
                "delta": cell.get("delta"),
                "verdict": cell.get("class_verdict"),
            }
        )
    return rows


def _hover(row: dict) -> str:
    text = (
        f"{pair_label(row['pair'])}<br>diferença: {fmt.pt_num(row['d'], 4)}"
        f"<br>intervalo: {fmt.pt_num(row['lo'], 4)} a {fmt.pt_num(row['hi'], 4)}"
    )
    if row["delta"] is not None:
        text += f"<br>margem δ: {fmt.pt_num(row['delta'], 4)}"
    return text


def pair_interval_figure(
    results: dict,
    prior: str = "P2",
    completeness: str = "0.80",
    roles: tuple[str, ...] | None = None,
) -> go.Figure:
    """One panel per class (independent x axes); point with interval per pair, ±δ band."""
    from plotly.subplots import make_subplots

    panels = {
        c: pair_points(results, prior, completeness, c, roles) for c in PANEL_CLASSES
    }
    n_rows = max((len(r) for r in panels.values()), default=0)
    height = _ROW_PX * max(n_rows, 1) + _PANEL_PX
    fig = make_subplots(
        rows=1,
        cols=len(PANEL_CLASSES),
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.28,
        subplot_titles=[label(c) for c in PANEL_CLASSES],
    )
    base = _base(height, grid="x")
    fig.update_layout(base.layout)
    fig.update_layout(margin={"l": 8, "r": 16, "t": 36, "b": 8, "pad": 4})
    fig.update_annotations(font={"family": SANS, "size": FONT_LABEL, "color": TEXT})
    for col, cls in enumerate(PANEL_CLASSES, start=1):
        rows = panels[cls]
        names = [pair_label(r["pair"]) for r in rows]
        for i, row in enumerate(rows):
            if row["delta"] is not None:
                fig.add_shape(
                    type="rect", xref=f"x{col}", yref=f"y{col}",
                    x0=-row["delta"], x1=row["delta"], y0=i - 0.4, y1=i + 0.4,
                    fillcolor=rgba(SOFT, _BAND_OPACITY), line={"width": 0}, layer="below",
                )  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=[r["d"] for r in rows],
                y=names,
                mode="markers",
                name=label(cls),
                showlegend=False,
                marker={"size": 10, "color": ACCENT},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [r["hi"] - r["d"] for r in rows],
                    "arrayminus": [r["d"] - r["lo"] for r in rows],
                    "color": ACCENT,
                    "thickness": 2,
                    "width": 5,
                },
                hovertext=[_hover(r) for r in rows],
                hoverinfo="text",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(
            zeroline=True, zerolinecolor=TEXT_MUTED, gridcolor=GRID, tickformat=".3f",
            row=1, col=col,
        )  # fmt: skip
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    return fig


def risk_coverage_figure(
    frame: pd.DataFrame, systems: list[str] | None = None
) -> go.Figure:
    """Accuracy against coverage per system (seed ``mean`` rows), raw and unweighted."""
    fig = _base(
        340,
        grid="xy",
        xaxis={"title": {"text": "Cobertura"}, "tickformat": ".0%"},
        yaxis={"title": {"text": f"Acerto ({_LEGEND_RAW})"}, "tickformat": ".1%"},
    )
    data = frame[frame["seed"].astype(str) == "mean"]
    names = systems if systems is not None else list(dict.fromkeys(data["system"]))
    for i, name in enumerate(names):
        sub = data[data["system"] == name].sort_values("coverage")
        shown = system_label(name)
        fig.add_trace(
            go.Scatter(
                x=sub["coverage"],
                y=sub["accuracy"],
                mode="lines",
                name=f"{shown} ({_LEGEND_RAW})",
                line={"width": 2, "color": SOFT, "dash": _DASHES[i % len(_DASHES)]},
                hovertemplate="cobertura %{x:.2%}<br>acerto %{y:.2%}<extra>"
                + shown
                + "</extra>",
            )
        )
    return fig
