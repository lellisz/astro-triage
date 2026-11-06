"""Palette, class/model labels and the shared figure layout for the dashboard charts."""

from __future__ import annotations

import plotly.graph_objects as go

from astro_triage.dashboard import theme

BG = theme.BG
HAIRLINE = theme.HAIRLINE
TEXT = theme.TEXT
TEXT_SOFT = theme.TEXT
TEXT_MUTED = theme.TEXT_SECONDARY
DASH = theme.DASH
NEUTRAL = DASH
BELOW_FLOOR = theme.DASH
ACCENT = theme.ACCENT
SOFT = theme.SOFT
# Soft grey is normal data, the blue accent is the error/highlight; shape and label still carry the meaning.
DATA = SOFT
ERROR = ACCENT

# One family for every chart text: Inter, regular. Sizes are shared constants (px).
SANS = theme.FONT_SANS
FONT_TICK = 15
FONT_LABEL = 16
FONT_ANNOT = 16
FONT_TITLE = 20

SYMBOL = {"STAR": "circle", "GALAXY": "square", "QSO": "triangle-up"}
GLYPH = {"STAR": "●", "GALAXY": "■", "QSO": "▲"}
CLASS_PT = {"STAR": "Estrela", "GALAXY": "Galáxia", "QSO": "Quasar"}
MODEL_PT = {
    "lgbm_full": "LightGBM completo",
    "lgbm_noind": "LightGBM sem indicadores de astrometria",
    "dsc_allosmod": "Gaia DSC (allosmod)",
}
MODEL_STYLE = {
    "lgbm_full": {"color": DATA, "symbol": "circle"},
    "lgbm_noind": {"color": TEXT, "symbol": "square"},
    "dsc_allosmod": {"color": NEUTRAL, "symbol": "diamond"},
}


def label(cls: str) -> str:
    return f"{GLYPH[cls]} {CLASS_PT[cls]}"


def rgba(color: str, alpha: float) -> str:
    """``'#rrggbb'`` token to an ``rgba()`` string with the given opacity."""
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


GRID = rgba(HAIRLINE, 0.6)
SHORT_TICK = 4
_STANDOFF = 10


def _axis(grid: bool, line: bool) -> dict:
    return {
        "showgrid": grid,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": True,
        "zerolinecolor": rgba(NEUTRAL, 0.6),
        "zerolinewidth": 1,
        "showline": line,
        "linecolor": NEUTRAL,
        "linewidth": 1,
        "ticks": "outside",
        "ticklen": SHORT_TICK,
        "tickwidth": 1,
        "tickcolor": NEUTRAL,
        "tickfont": {"family": SANS, "size": FONT_TICK, "color": TEXT_MUTED},
        "automargin": True,
        "title": {
            "font": {"family": SANS, "size": FONT_LABEL, "color": TEXT_MUTED},
            "standoff": _STANDOFF,
        },
    }


def _base(
    height: int, *, grid: str = "y", hovermode: str = "closest", **layout
) -> go.Figure:
    """Shared figure: transparent, hairline horizontal grid, short outside ticks.

    ``grid``: which axes draw a grid, ``"y"`` (default), ``"x"``, ``"xy"`` or ``""``.
    ``layout`` merges recursively over these defaults (``xaxis={...}`` keeps the rest).
    """
    fig = go.Figure()
    fig.update_layout(**theme.plotly_layout(height))
    fig.update_layout(
        separators=",.",
        hovermode=hovermode,
        font={"family": SANS, "size": FONT_TICK, "color": TEXT_MUTED},
        margin={"l": 8, "r": 16, "t": 12, "b": 8, "pad": 4},
        uniformtext={"minsize": FONT_TICK, "mode": "show"},
        transition={"duration": 0},
        hoverlabel={
            "bgcolor": theme.SECONDARY_BG,
            "bordercolor": HAIRLINE,
            "font": {"family": SANS, "size": FONT_TICK, "color": TEXT},
            "namelength": -1,
            "align": "left",
        },
        xaxis=_axis("x" in grid, True),
        yaxis=_axis("y" in grid, False),
    )
    fig.update_layout(**layout)
    return fig
