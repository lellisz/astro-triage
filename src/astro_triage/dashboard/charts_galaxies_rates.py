"""One dot chart of galaxy error rates by magnitude G and by redshift z, with Wilson intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import fmt
from astro_triage.dashboard.charts_base import (
    ERROR,
    NEUTRAL,
    TEXT_MUTED,
    TEXT_SOFT,
    _base,
)
from astro_triage.dashboard.charts_galaxies import (
    RATE_LOG_RANGE,
    base_marker_traces,
    n_labels,
)
from astro_triage.dashboard.charts_galaxies_annot import (
    RATE_HOVER,
    band_trace,
    interval_bars,
    overlay_axis,
    rate_axis,
    rate_customdata,
    separator,
)

BLOCK_GAP = 1  # empty x slots between the magnitude block and the redshift block


def _layout_positions(g: pd.DataFrame, z: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Both tables stacked, with a numeric ``x`` slot per row and the first slot of block z."""
    first_z = len(g) + BLOCK_GAP
    g = g.assign(x=np.arange(len(g)), block="G")
    z = z.assign(x=np.arange(len(z)) + first_z, block="z")
    return pd.concat([g, z], ignore_index=True), first_z


def _dots(
    sub: pd.DataFrame, name: str, symbol: str, floor: int, few: bool
) -> go.Scatter | None:
    sub = sub[(sub["errors"] > 0) & (sub["rate"] >= 10 ** RATE_LOG_RANGE[0])]
    sub = sub[(sub["n"] < floor) if few else (sub["n"] >= floor)]
    if sub.empty:
        return None
    return go.Scatter(
        name=name,
        x=sub["x"],
        y=sub["rate"],
        mode="markers",
        marker={
            "symbol": symbol,
            "size": 11,
            "color": ERROR,
            "line": {"width": 2, "color": ERROR},
        },
        error_y=interval_bars(sub["rate"], sub["ci_lo"], sub["ci_hi"]),
        customdata=rate_customdata(sub),
        hovertemplate=RATE_HOVER,
    )


def rate_dots(
    by_g: pd.DataFrame, by_z: pd.DataFrame, mean_rate: float, floor: int
) -> go.Figure:
    """Filled dot: ``n >= floor``. Open circle: fewer than ``floor`` objects (wide interval).

    Zero-error rows are an ``x`` and rows below the axis a down triangle, both on the baseline.
    The dashed line is the overall galaxy error rate; faint bands alternate by bin and a
    dotted rule separates the G block from the z block.
    """
    table, first_z = _layout_positions(by_g, by_z)
    fig = _base(
        470,
        margin={"l": 8, "r": 8, "t": 40, "b": 8},
        yaxis=rate_axis(
            title={"text": "taxa de erro (escala log)", "font": {"color": TEXT_MUTED}}
        ),
        yaxis2=overlay_axis(),
        xaxis={
            "tickvals": list(table["x"]),
            "ticktext": n_labels(table),
            "range": [-0.6, table["x"].max() + 0.6],
            "showgrid": False,
            "zeroline": False,
        },
    )
    fig.add_trace(band_trace([float(x) for x in table["x"][::2]]))
    for sub_name, symbol, few in (
        ("n suficiente", "circle", False),
        ("poucos objetos", "circle-open", True),
    ):
        trace = _dots(table, sub_name, symbol, floor, few)
        if trace is not None:
            fig.add_trace(trace)
    for trace in base_marker_traces(table, list(table["x"])):
        fig.add_trace(trace)
    separator(fig, first_z - (BLOCK_GAP + 1) / 2, "G | z")
    fig.add_hline(
        y=mean_rate,
        line={"dash": "dash", "color": NEUTRAL, "width": 1},
        annotation_text=f"média {fmt.pt_pct(mean_rate)}",
        annotation_position="top right",
        annotation_font={"color": TEXT_MUTED},
    )
    for text, lo, hi in (
        ("magnitude G (mais fraca à direita)", 0, first_z - BLOCK_GAP - 1),
        ("redshift z", first_z, table["x"].max()),
    ):
        fig.add_annotation(
            x=(lo + hi) / 2,
            xref="x",
            y=1,
            yref="paper",
            yshift=8,
            text=text,
            showarrow=False,
            font={"color": TEXT_SOFT},
        )
    return fig
