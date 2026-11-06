"""Galaxy (z, G) scatter with a separate strip for ``z <= Z_LOW``.

The old figure clipped z with ``clip(lower=1e-3)``; that clip, not missing data, created the
pile of points at z = 0.001. Rows with ``z <= Z_LOW`` (negative z included) now sit in their own
strip left of the log axis, spread by a seeded horizontal jitter. Hover shows the real z.
Per-object confidence is deliberately not sent to the browser: the figure takes the sample
built by ``data_galaxies_z.scatter_sample`` and only z, G and the predicted class per error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from astro_triage.dashboard.charts_base import (
    ACCENT,
    CLASS_PT,
    ERROR,
    FONT_LABEL,
    HAIRLINE,
    TEXT,
    TEXT_MUTED,
    _base,
)
from astro_triage.dashboard.charts_galaxies_density import (
    density_contours,
    divider,
    rug_traces,
)
from astro_triage.dashboard.data_galaxies_z import Z_LOW, z_low_label
from astro_triage.dashboard.data_release import SEED

STRIP_HALF_WIDTH = 0.35
_Z_TICKS = (
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
)  # fixed: ".2g" crowded labels below z = 0.1
STRIP_SHARE = 0.16  # width of the strip column relative to the log-axis column

_HOVER_REST = "G = %{y:.1f}<br>classificada como %{customdata[1]}<extra></extra>"
_SCI_DIGITS = 2
_SCI = f".{_SCI_DIGITS}e"
_HOVER = "<b>erro</b><br>z = %{customdata[0]:.4f}<br>" + _HOVER_REST
# real z in scientific notation: the strip hides its spread
_HOVER_LOW = f"<b>erro</b><br>z = %{{customdata[0]:{_SCI}}}<br>" + _HOVER_REST


def _jitter(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).uniform(-STRIP_HALF_WIDTH, STRIP_HALF_WIDTH, n)


def _error_trace(
    sub: pd.DataFrame, x: pd.Series | np.ndarray, hover: str, **style: object
):
    return go.Scatter(
        x=x,
        y=sub["G"],
        mode="markers",
        customdata=np.stack([sub["z"], sub["pred"].map(CLASS_PT).str.lower()], axis=-1),
        hovertemplate=hover,
        **style,
    )


def galaxy_scatter(df: pd.DataFrame) -> go.Figure:
    """Open circle: became a quasar. Diamond: became a star. Small violet: classified right."""
    low = (df["z"] <= Z_LOW).to_numpy()
    jit = pd.Series(_jitter(len(df)), index=df.index)
    base = _base(450)
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[STRIP_SHARE, 1 - STRIP_SHARE],
        horizontal_spacing=0.01,
    )
    fig.update_layout(base.layout)
    fig.update_layout(
        showlegend=True,
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        legend={
            "orientation": "h",
            "x": 1.0,
            "xanchor": "right",
            "y": 1.0,
            "yanchor": "bottom",
            "font": {"color": TEXT, "size": FONT_LABEL},
        },
    )
    ok = ~df["error"].to_numpy()
    sel_ok = df[~low & ok]
    contours = density_contours(sel_ok["z"].to_numpy(), sel_ok["G"].to_numpy())
    if contours is not None:
        fig.add_trace(contours, row=1, col=2)
    for col, mask in ((1, low), (2, ~low)):
        sel = df[mask & ok]
        fig.add_trace(
            go.Scattergl(
                name="classificada certo",
                x=jit[sel.index] if col == 1 else sel["z"],
                y=sel["G"],
                mode="markers",
                marker={"size": 5, "color": ACCENT, "opacity": 0.45},
                hoverinfo="skip",
                showlegend=col == 2,
                legendgroup="ok",
            ),
            row=1,
            col=col,
        )
        for pred, symbol, name, size in (
            ("QSO", "circle-open", f"virou {CLASS_PT['QSO'].lower()}", 11),
            ("STAR", "diamond", f"virou {CLASS_PT['STAR'].lower()}", 11),
        ):
            sub = df[mask & ~ok & (df["pred"] == pred)]
            fig.add_trace(
                _error_trace(
                    sub,
                    jit[sub.index] if col == 1 else sub["z"],
                    _HOVER_LOW if col == 1 else _HOVER,
                    name=name,
                    legendgroup=pred,
                    showlegend=col == 2,
                    marker={
                        "symbol": symbol,
                        "size": size,
                        "color": ERROR,
                        "line": {"width": 2, "color": ERROR},
                    },
                ),
                row=1,
                col=col,
            )
    errors = df[~low & ~ok]
    if not errors.empty:
        z_edge = max(float(df.loc[~low, "z"].min()) * 0.9, Z_LOW * 1.05)
        for rug in rug_traces(errors, float(df["G"].max()) + 0.2, z_edge):
            fig.add_trace(rug, row=1, col=2)
    axis_style = {"gridcolor": HAIRLINE, "linecolor": HAIRLINE, "tickcolor": HAIRLINE}
    fig.update_xaxes(
        range=[-0.5, 0.5],
        tickvals=[0],
        ticktext=[z_low_label()],
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
        **{k: v for k, v in axis_style.items() if k != "gridcolor"},
    )
    fig.update_xaxes(
        type="log",
        row=1,
        col=2,
        title={"text": "redshift z (escala log)", "font": {"color": TEXT_MUTED}},
        tickvals=_Z_TICKS,
        ticktext=[f"{t:g}".replace(".", ",") for t in _Z_TICKS],
        **axis_style,
    )
    fig.update_yaxes(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE, col=2, row=1)
    fig.update_yaxes(
        autorange="reversed",
        gridcolor=HAIRLINE,
        zerolinecolor=HAIRLINE,
        linecolor=HAIRLINE,
        tickcolor=HAIRLINE,
        title={
            "text": "magnitude G (mais fraca embaixo)",
            "font": {"color": TEXT_MUTED},
        },
        row=1,
        col=1,
    )
    divider(fig, f"◂ {z_low_label()}")
    return fig
