"""Sky map drawn with HTML/CSS (Mollweide 2:1). Pure: no streamlit import.

Every position comes from ``data_sky.mollweide`` and every colour from ``theme.ramp``;
the only constants here are the geometry of the graticule. One single-line string per block.
"""

from __future__ import annotations

import math
from html import escape
from typing import Final

import numpy as np
import pandas as pd

from astro_triage.dashboard import fmt, theme
from astro_triage.dashboard.data_sky import BAND_EDGES_DEG, mollweide

X_MAX: Final = 2 * math.sqrt(2)
Y_MAX: Final = math.sqrt(2)
MERIDIANS_DEG: Final = (30, 60, 90, 120, 150)
LABELLED_LATS_DEG: Final = (-60, -30, 0, 30, 60)
LON_STEP_DEG: Final = 30
HALF_TURN_DEG: Final = 180
CAP_PERCENTILE: Final = 95.0
PERCENT: Final = 100.0
HALF: Final = 50.0

SKY_CSS: Final = (
    "<style>"
    ".at-map{position:relative;width:100%;aspect-ratio:2/1;margin:8px 0 34px}"
    ".at-map>*{position:absolute}"
    f".at-mo{{inset:0;border-radius:50%;border:1px solid {theme.DASH};background:{theme.SECONDARY_BG}}}"
    f".at-me{{top:0;bottom:0;border-radius:50%;border:1px solid {theme.HAIRLINE};box-sizing:border-box}}"
    f".at-m0{{top:0;bottom:0;left:50%;border-left:1px solid {theme.HAIRLINE}}}"
    f".at-pl{{border-top:1px solid {theme.HAIRLINE}}}"
    f".at-pd{{border-top:1px dashed {theme.DASH}}}"
    ".at-pt{width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;box-sizing:border-box}"
    f".at-ring{{border:1.5px solid {theme.RAMP_LOW};background:transparent}}"
    ".at-lat{transform:translate(calc(-100% - 8px),-50%)}"
    ".at-lon{top:50%;transform:translate(-50%,6px)}"
    ".at-lg{display:flex;align-items:center;gap:12px;max-width:420px;margin:0 auto}"
    ".at-lgb{flex:1;height:10px;border-radius:5px}"
    "</style>"
)


def _t(text: object) -> str:
    return escape(str(text), quote=True)


def parallel_geometry(lat_deg: float) -> tuple[float, float, float]:
    """(top %, left %, width %) of the parallel at ``lat_deg``: its width is cos(theta)."""
    _, y = mollweide(np.zeros(1), np.array([float(lat_deg)]))
    s = float(y[0]) / Y_MAX
    width = math.sqrt(max(0.0, 1.0 - s * s)) * PERCENT
    return HALF - s * HALF, (PERCENT - width) / 2, width


def meridian_style(lon_deg: float) -> str:
    """Ellipse for the meridians at +-``lon_deg``: width proportional to lon/180."""
    frac = lon_deg / HALF_TURN_DEG
    return f"left:{HALF - frac * HALF:.2f}%;width:{frac * PERCENT:.2f}%"


def _lat_text(lat: float) -> str:
    sign = "+" if lat > 0 else "−" if lat < 0 else ""
    return f"{sign}{abs(lat):g}°"


def lon_label_positions() -> list[tuple[int, float]]:
    """``(l, left %)`` for the labels on the equator; l grows to the left, 0 and 180 omitted."""
    out = []
    for lon in range(LON_STEP_DEG, 2 * HALF_TURN_DEG, LON_STEP_DEG):
        if lon == HALF_TURN_DEG:
            continue
        signed = lon - 2 * HALF_TURN_DEG if lon > HALF_TURN_DEG else lon
        out.append((lon, HALF - signed / HALF_TURN_DEG * HALF))
    return out


def graticule() -> str:
    """Outline, meridians, parallels (dashed on band limits and equator) and their labels."""
    parts = ["<div class='at-mo'></div>"]
    parts += [
        f"<div class='at-me' style='{meridian_style(m)}'></div>" for m in MERIDIANS_DEG
    ]
    parts.append("<div class='at-m0'></div>")
    edges = {s * e for e in BAND_EDGES_DEG for s in (-1, 1)}
    for lat in sorted(edges | set(LABELLED_LATS_DEG)):
        top, left, width = parallel_geometry(lat)
        dashed = lat in edges or lat == 0
        cls = "at-pd" if dashed else "at-pl"
        parts.append(
            f"<div class='{cls}' style='top:{top:.2f}%;left:{left:.2f}%;width:{width:.2f}%'></div>"
        )
        if lat in LABELLED_LATS_DEG:
            parts.append(
                f"<span class='at-cap at-lat' style='top:{top:.2f}%;left:{left:.2f}%'>{_lat_text(lat)}</span>"
            )
    parts += [
        f"<span class='at-cap at-lon' style='left:{left:.2f}%'>{lon}°</span>"
        for lon, left in lon_label_positions()
    ]
    return "".join(parts)


def color_cap(stats: pd.DataFrame, by_rate: bool, floor: int) -> float:
    """95th percentile of the displayed values (rates at or above ``floor``, or counts)."""
    values = stats.loc[stats["n"] >= floor, "rate"] if by_rate else stats["n"]
    return float(np.percentile(values, CAP_PERCENTILE)) if len(values) else 0.0


def region_points(
    stats: pd.DataFrame, by_rate: bool, floor: int, hide_below: bool, cap: float
) -> str:
    """One dot per region at its projected position; hollow ring below the floor (rate mode)."""
    parts = []
    for r in stats.itertuples(index=False):
        if not (math.isfinite(r.x) and math.isfinite(r.y)):
            continue
        pos = f"left:{HALF + r.x / X_MAX * HALF:.2f}%;top:{HALF - r.y / Y_MAX * HALF:.2f}%"
        tip = f"{fmt.pt_int(r.n)} objetos, {fmt.pt_int(r.errors)} erros"
        if by_rate and r.n < floor:
            if not hide_below:
                parts.append(
                    f"<i class='at-pt at-ring' style='{pos}' title='{_t(tip)}'></i>"
                )
            continue
        value = r.rate if by_rate else r.n
        fill = theme.ramp(value / cap if cap > 0 else 0.0)
        parts.append(
            f"<i class='at-pt' style='{pos};background:{fill}' title='{_t(tip)}'></i>"
        )
    return "".join(parts)


def lead_line(text: str) -> str:
    """Highlighted sentence above the map."""
    return f"<p style='margin:0 0 24px;font-size:19px'>{_t(text)}</p>"


def legend(by_rate: bool, cap: float) -> str:
    """Gradient bar from zero to the colour cap, both ends labelled with computed values."""
    if by_rate:
        left, right = fmt.pt_pct(0, 0), f"{fmt.pt_pct(cap)} ou mais"
    else:
        left, right = "0", f"{fmt.pt_int(round(cap))} objetos ou mais"
    grad = f"linear-gradient(90deg,{theme.ramp(0)},{theme.ramp(1)})"
    return (
        f"<div class='at-lg'><span class='at-cap'>{_t(left)}</span>"
        f"<div class='at-lgb' style='background:{grad}'></div>"
        f"<span class='at-cap'>{_t(right)}</span></div>"
    )


def build_map(
    stats: pd.DataFrame, by_rate: bool, floor: int, hide_below: bool = False
) -> str:
    """CSS plus map plus legend as one single-line HTML string."""
    cap = color_cap(stats, by_rate, floor)
    what = "a taxa de erro" if by_rate else "a quantidade de objetos"
    label = f"Mapa Mollweide com {what} de {fmt.pt_int(len(stats))} regiões do teste"
    caption = "l = 0° (centro da Via Láctea)"
    return (
        f"{SKY_CSS}<div style='max-width:{theme.WIDE_WIDTH_PX}px;margin:0 auto'>"
        f"<div class='at-map' role='img' aria-label='{_t(label)}'>"
        f"{graticule()}{region_points(stats, by_rate, floor, hide_below, cap)}"
        f"<span class='at-cap' style='left:50%;bottom:-28px;transform:translateX(-50%)'>{_t(caption)}</span>"
        f"</div>{legend(by_rate, cap)}</div>"
    )
