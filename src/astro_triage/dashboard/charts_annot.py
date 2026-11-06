"""Small shared helpers for direct labels and subtle bands on the charts."""

from __future__ import annotations

import plotly.graph_objects as go

from astro_triage.dashboard.charts_base import FONT_ANNOT, SANS, TEXT_MUTED


def rgba(color: str, alpha: float) -> str:
    """``#rrggbb`` plus an opacity as a CSS ``rgba(...)`` string."""
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def note(fig: go.Figure, x: float, y: float, text: str, **kwargs: object) -> None:
    """Direct label in the muted text colour in Inter."""
    kwargs.setdefault("showarrow", False)
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        font={"family": SANS, "color": TEXT_MUTED, "size": FONT_ANNOT},
        **kwargs,
    )
