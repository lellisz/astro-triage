"""Galaxy error-rate bars by astrometric solution, and the helpers shared by the rate charts.

The (z, G) scatter lives in ``charts_galaxies_scatter`` and the dot chart with Wilson
intervals in ``charts_galaxies_rates``.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt
from astro_triage.dashboard.charts_base import (
    ERROR,
    TEXT_MUTED,
    TEXT_SOFT,
    _base,
)
from astro_triage.dashboard.charts_galaxies_annot import (
    RATE_HOVER,
    RATE_LOG_RANGE,
    RATE_TICKS,  # noqa: F401  (re-export used by charts.py)
    interval_bars,
    rate_axis,
    rate_customdata,
)


def zero_error_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows with objects and no errors: their bar has no height on a log axis."""
    return table[(table["n"] > 0) & (table["errors"] == 0)]


def tiny_rate_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows with errors whose rate falls under the axis lower limit (bar not visible)."""
    return table[(table["errors"] > 0) & (table["rate"] < 10 ** RATE_LOG_RANGE[0])]


def low_n_rows(table: pd.DataFrame, floor: int) -> pd.DataFrame:
    """Rows with errors, drawn on the axis, but with fewer than ``floor`` objects."""
    visible = table["rate"] >= 10 ** RATE_LOG_RANGE[0]
    return table[(table["n"] > 0) & (table["n"] < floor) & visible]


def n_labels(table: pd.DataFrame) -> list[str]:
    """Tick text of each row: label plus its object count."""
    return [
        f"{lab}<br>sem objetos" if n == 0 else f"{lab}<br>n={fmt.pt_int(n)}"
        for lab, n in zip(table["label"], table["n"], strict=True)
    ]


def floor_marker(
    x: list[str] | list[float],
    text: list[str],
    hover: list[str],
    symbol: str = "circle-open",
    name: str | None = None,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=[10 ** RATE_LOG_RANGE[0]] * len(x),
        mode="markers+text",
        name=name,
        marker={"symbol": symbol, "size": 12, "color": ERROR, "line": {"width": 2}},
        text=text,
        textposition="top center",
        textfont={"color": TEXT_MUTED},
        cliponaxis=False,
        hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
    )


def base_marker_traces(
    table: pd.DataFrame, x: list[str] | list[float]
) -> list[go.Scatter]:
    """Baseline markers: ``x`` for zero errors, down triangle for a rate below the axis.

    ``x`` holds one position per row of ``table`` (tick label or numeric slot).
    """
    pos = pd.Series(list(x), index=table.index)
    below = f"< {fmt.pt_pct(10 ** RATE_LOG_RANGE[0])}"
    traces = []
    for rows, symbol, name in (
        (zero_error_rows(table), "x", "zero erros"),
        (tiny_rate_rows(table), "triangle-down", "taxa abaixo do eixo"),
    ):
        if rows.empty:
            continue
        text = [
            f"sem erros (IC até {fmt.pt_pct(hi, 2)})" if k == 0 else below
            for k, hi in zip(rows["errors"], rows["ci_hi"], strict=True)
        ]
        hover = [
            f"{lab}: {fmt.pt_int(k)} erros em {fmt.pt_int(n)} objetos; "
            f"taxa {fmt.pt_pct(rt, 3)} ({fmt.pt_ci_label()}: {fmt.pt_pct(lo, 3)} a "
            f"{fmt.pt_pct(hi, 3)})"
            for lab, k, n, rt, lo, hi in zip(
                rows["label"],
                rows["errors"],
                rows["n"],
                rows["rate"],
                rows["ci_lo"],
                rows["ci_hi"],
                strict=True,
            )
        ]
        traces.append(floor_marker(list(pos[rows.index]), text, hover, symbol, name))
    return traces


def rate_bars(table: pd.DataFrame, floor: int | None = None) -> go.Figure:
    """Error rate per bin with Wilson intervals (``ERROR`` colour: these are errors).

    The y axis is log with fixed limits so every panel is read on the same scale. Bins with
    fewer than ``floor`` objects get an open circle on top of the bar; zero-error and
    below-axis bins get the baseline markers of ``base_marker_traces``.
    """
    fig = _base(380, margin={"l": 8, "r": 8, "t": 28, "b": 8}, yaxis=rate_axis())
    labels = n_labels(table)
    fig.add_trace(
        go.Bar(
            name="taxa de erro",
            x=labels,
            y=table["rate"],
            width=0.5,
            marker={"color": ERROR},
            error_y=interval_bars(table["rate"], table["ci_lo"], table["ci_hi"]),
            customdata=rate_customdata(table),
            hovertemplate=RATE_HOVER,
        )
    )
    shown = table[(table["n"] > 0) & (table["rate"] >= 10 ** RATE_LOG_RANGE[0])]
    if not shown.empty:
        fig.add_trace(
            go.Scatter(
                name="taxa de erro (rótulo)",
                x=[labels[i] for i in table.index.get_indexer(shown.index)],
                y=shown["ci_hi"],
                mode="text",
                text=[fmt.pt_pct(r) for r in shown["rate"]],
                textposition="top center",
                textfont={"color": TEXT_SOFT},
                cliponaxis=False,
                hoverinfo="skip",
            )
        )
    if floor is not None:
        few = low_n_rows(table, floor)
        if not few.empty:
            fig.add_trace(
                go.Scatter(
                    name="poucos objetos",
                    x=[labels[i] for i in table.index.get_indexer(few.index)],
                    y=few["rate"],
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": 14,
                        "color": TEXT_SOFT,
                        "line": {"width": 2},
                    },
                    hovertext=[f"n={fmt.pt_int(n)}, poucos objetos" for n in few["n"]],
                    hovertemplate="%{x}<br>%{hovertext}<extra></extra>",
                )
            )
    for trace in base_marker_traces(table, labels):
        fig.add_trace(trace)
    return fig
