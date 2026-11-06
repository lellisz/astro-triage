"""Density contours and axis rugs for the (z, G) scatter.

Contours come from a coarse 2D histogram traced here (marching squares), so the browser gets
a few hundred line vertices, not another copy of the points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from astro_triage.dashboard import theme
from astro_triage.dashboard.charts_base import ERROR, FONT_ANNOT, SANS, SOFT, TEXT

BINS_Z = 24
BINS_G = 18
MIN_POINTS = 30
LEVELS = (0.12, 0.3, 0.55, 0.8)  # fractions of the peak of the smoothed histogram
# cell corner order: 0 (x0,y0), 1 (x1,y0), 2 (x1,y1), 3 (x0,y1); edges 0..3 between 0-1, 1-2, 2-3, 3-0
_SEGMENTS = {
    1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 5: ((3, 0), (1, 2)),
    6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),), 9: ((0, 2),), 10: ((0, 1), (2, 3)),
    11: ((1, 2),), 12: ((1, 3),), 13: ((0, 1),), 14: ((3, 0),),
}  # fmt: skip
_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))


def _smooth(h: np.ndarray) -> np.ndarray:
    p = np.pad(h, 1)
    out = np.zeros_like(p)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out += np.roll(np.roll(p, dx, 0), dy, 1)
    return out / 9.0


def _edge_point(
    xs: np.ndarray,
    ys: np.ndarray,
    v: np.ndarray,
    i: int,
    j: int,
    edge: int,
    level: float,
) -> tuple[float, float]:
    (ax, ay), (bx, by) = _CORNERS[edge], _CORNERS[(edge + 1) % 4]
    va, vb = v[i + ax, j + ay], v[i + bx, j + by]
    t = (level - va) / (vb - va)
    return (
        float(xs[i + ax] + t * (xs[i + bx] - xs[i + ax])),
        float(ys[j + ay] + t * (ys[j + by] - ys[j + ay])),
    )


def density_contours(z: np.ndarray, g: np.ndarray) -> go.Scatter | None:
    """Monochrome contour lines of the (log z, G) density; None when there are too few points."""
    if len(z) < MIN_POINTS:
        return None
    lz = np.log10(z)
    h, ex, ey = np.histogram2d(lz, g, bins=(BINS_Z, BINS_G))
    v = _smooth(h)
    xs = np.concatenate([[ex[0] - (ex[1] - ex[0]) / 2], (ex[:-1] + ex[1:]) / 2])
    xs = np.append(xs, xs[-1] + (ex[1] - ex[0]))
    ys = np.concatenate([[ey[0] - (ey[1] - ey[0]) / 2], (ey[:-1] + ey[1:]) / 2])
    ys = np.append(ys, ys[-1] + (ey[1] - ey[0]))
    px: list[float] = []
    py: list[float] = []
    for frac in LEVELS:
        level = frac * float(v.max())
        for i in range(v.shape[0] - 1):
            for j in range(v.shape[1] - 1):
                idx = sum(
                    1 << k
                    for k, (cx, cy) in enumerate(_CORNERS)
                    if v[i + cx, j + cy] > level
                )
                for e0, e1 in _SEGMENTS.get(idx, ()):
                    a = _edge_point(xs, ys, v, i, j, e0, level)
                    b = _edge_point(xs, ys, v, i, j, e1, level)
                    px += [a[0], b[0], np.nan]
                    py += [a[1], b[1], np.nan]
    if not px:
        return None
    return go.Scatter(
        name="densidade das certas",
        x=10 ** np.asarray(px),
        y=py,
        mode="lines",
        line={"color": SOFT, "width": 1.25},
        hoverinfo="skip",
        showlegend=False,
    )


def rug_traces(errors: pd.DataFrame, g_edge: float, z_edge: float) -> list[go.Scatter]:
    """Tick marks on the bottom (z) and left (G) edges at each error, in the right panel."""
    mark = {"size": 11, "color": ERROR, "line": {"width": 1.5, "color": ERROR}}
    common = {"mode": "markers", "hoverinfo": "skip", "showlegend": False}
    return [
        go.Scatter(
            name="erros em z (marcas no eixo)",
            x=errors["z"],
            y=np.full(len(errors), g_edge),
            marker={"symbol": "line-ns-open", **mark},
            **common,
        ),
        go.Scatter(
            name="erros em G (marcas no eixo)",
            x=np.full(len(errors), z_edge),
            y=errors["G"],
            marker={"symbol": "line-ew-open", **mark},
            **common,
        ),
    ]


def divider(fig: go.Figure, label: str) -> None:
    """Dotted rule at the boundary between the strip column and the log column, labelled."""
    x = (fig.layout.xaxis.domain[1] + fig.layout.xaxis2.domain[0]) / 2
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        line={"color": theme.DASH, "width": 1, "dash": "dot"},
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=x,
        y=1,
        text=label,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=4,
        font={"family": SANS, "color": TEXT, "size": FONT_ANNOT},
    )
