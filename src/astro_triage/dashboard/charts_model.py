"""Overview figure: purity/completeness dumbbell, one row per scenario, one 0-100% axis."""

from __future__ import annotations

import plotly.graph_objects as go

from astro_triage.dashboard import fmt
from astro_triage.dashboard.charts_base import (
    ACCENT,
    FONT_ANNOT,
    FONT_LABEL,
    FONT_TICK,
    GRID,
    NEUTRAL,
    SANS,
    SOFT,
    TEXT,
    TEXT_MUTED,
    _base,
)
from astro_triage.dashboard.charts_model_pairs import error_bars, pair_text  # noqa: F401

_RIGHT_MARGIN_DUMBBELL = 250
_ROW_PX = 80
_TOP_PX = 48
_BOTTOM_PX = 44
_COLUMN_GAP = 16
_TICKS = [0, 0.25, 0.5, 0.75, 1]
_SPAN = [-0.06, 1.06]
_MARKER = {"purity": 20, "completeness": 14}
_BIG = FONT_ANNOT + 6


def _interval_bars(points: list[dict], key: str) -> dict | None:
    """Horizontal 95% interval bars; points without an interval get zero length."""
    if key != "purity" or all(p.get(f"{key}_lo") is None for p in points):
        return None
    up = [0 if p.get(f"{key}_hi") is None else p[f"{key}_hi"] - p[key] for p in points]
    down = [
        0 if p.get(f"{key}_lo") is None else p[key] - p[f"{key}_lo"] for p in points
    ]
    return {
        "type": "data", "symmetric": False, "array": up, "arrayminus": down,
        "color": NEUTRAL, "thickness": 1.5, "width": 5,
    }  # fmt: skip


def per_100(purity: float) -> str:
    return fmt.pt_num(purity * 100, 1)


def _hover(p: dict, key: str, noun: str) -> str:
    value, lo, hi = p[key], p.get(f"{key}_lo"), p.get(f"{key}_hi")
    shown = fmt.pt_pct(value, 1) if lo is None else fmt.pt_pct_interval(value, lo, hi)
    return f"{p['label']}<br>{noun}: {shown}"


def _legend(fig: go.Figure) -> None:
    """One-line key above the plot; the glyphs carry the marker colours."""
    glyph = "<span style='color:{}'>{}</span>"
    fig.add_annotation(
        x=0, y=1, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
        yshift=_COLUMN_GAP, showarrow=False, align="left",
        text=f"{glyph.format(ACCENT, '●')} pureza   {glyph.format(SOFT, '○')} completude"
        "   · traço: validação",
        font={"family": SANS, "size": FONT_LABEL, "color": TEXT_MUTED},
    )  # fmt: skip


def _column(fig: go.Figure, points: list[dict]) -> None:
    """Right-hand column: 'N of every 100' large, interval on a quieter second line."""
    fig.add_annotation(
        x=1, y=1, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
        xshift=_COLUMN_GAP, yshift=_COLUMN_GAP, showarrow=False, align="left",
        text="apontados que são reais",
        font={"family": SANS, "size": FONT_LABEL, "color": TEXT_MUTED},
    )  # fmt: skip
    for p in points:
        lo, hi = p.get("purity_lo"), p.get("purity_hi")
        second = (
            ""
            if lo is None
            else f"<br><span style='font-size:{FONT_TICK}px;color:{TEXT_MUTED}'>"
            f"{fmt.pt_ci_label()}: {fmt.pt_pct(lo)}–{fmt.pt_pct(hi)}</span>"
        )
        fig.add_annotation(
            x=1, y=p["label"], xref="paper", yref="y", xanchor="left", yanchor="middle",
            xshift=_COLUMN_GAP, showarrow=False, align="left",
            text=f"{per_100(p['purity'])} de cada 100{second}",
            font={"family": SANS, "size": _BIG, "color": TEXT, "weight": 600},
        )  # fmt: skip


def dumbbell(points: list[dict]) -> go.Figure:
    """One horizontal row per point: filled circle = purity, open circle = completeness.

    ``points``: dicts from ``data_headline.purity_points``. Dashed rows are validation.
    """
    fig = _base(
        _ROW_PX * len(points) + _TOP_PX + _BOTTOM_PX,
        grid="x",
        margin={"l": 8, "r": _RIGHT_MARGIN_DUMBBELL, "t": _TOP_PX, "b": _BOTTOM_PX},
        xaxis={
            "range": _SPAN, "tickformat": ".0%", "tickmode": "linear", "tick0": 0,
            "dtick": _TICKS[1], "tickangle": 0, "side": "bottom", "showline": False,
            "zeroline": False, "ticks": "",
        },
        yaxis={"autorange": "reversed", "zeroline": False},
    )  # fmt: skip
    for i in range(1, len(points)):
        fig.add_shape(
            type="line", xref="paper", x0=0, x1=1, yref="y", y0=i - 0.5, y1=i - 0.5,
            line={"color": GRID, "width": 1}, layer="below",
        )  # fmt: skip
    for p in points:
        fig.add_trace(
            go.Scatter(
                x=[p["completeness"], p["purity"]],
                y=[p["label"]] * 2,
                mode="lines",
                name=f"Ligação: {p['label']}",
                line={
                    "color": NEUTRAL,
                    "width": 2,
                    "dash": "dash" if p.get("dashed") else "solid",
                },
                hoverinfo="skip",
            )
        )
    for name, key, symbol, noun, pos, color, tcolor in (
        ("Completude", "completeness", "circle-open", "completude", "bottom center", SOFT, TEXT_MUTED),
        ("Pureza", "purity", "circle", "pureza", "top center", ACCENT, TEXT),
    ):  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=[p[key] for p in points],
                y=[p["label"] for p in points],
                mode="markers+text",
                name=name,
                marker={
                    "size": _MARKER[key],
                    "symbol": symbol,
                    "color": color,
                    "line": {"width": 2, "color": color},
                },
                error_x=_interval_bars(points, key),
                text=[fmt.pt_pct(p[key], 1) for p in points],
                textposition=pos,
                textfont={"family": SANS, "color": tcolor, "size": FONT_LABEL},
                cliponaxis=False,
                hovertext=[_hover(p, key, noun) for p in points],
                hoverinfo="text",
            )
        )
    _legend(fig)
    _column(fig, points)
    return fig
