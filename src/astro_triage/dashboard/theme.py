"""Design tokens and style builders for the dashboard. Pure: does not import streamlit."""

from __future__ import annotations

import re

BG = "#0a0d14"
SECONDARY_BG = "#11151f"
TEXT = "#eef1f7"
TEXT_SECONDARY = "#a9b2c6"
SOFT = "#cfd6e4"  # normal data marks and emphasised body text
HAIRLINE = "#262d3d"
GRID = "#181d2a"
# Design value #4b546a gives only 2.57:1 on BG; this is the nearest tone that reaches 3:1.
DASH = "#566077"
ACCENT = "#6db3ff"  # blue: the single highlight (focus, errors, selected values)
RAMP_LOW = "#59647c"  # low end of every sequential scale; the high end is ACCENT
# Information is never carried by colour alone: shapes and direct labels stay.
# Contrast (contrast_ratio) is asserted in tests/test_dashboard_phase2_theme.py:
#   TEXT, TEXT_SECONDARY, SOFT, ACCENT on BG >= 4.5 (text allowed).
#   DASH and RAMP_LOW on BG >= 3.0: graphics and rules only, never body text.
#   HAIRLINE, GRID: decorative separators only.

MIN_SECONDARY_PX = 15
BODY_PX = 18
CONTROL_PX = 16
HEADING_PX = 27
READING_WIDTH_PX = 800
WIDE_WIDTH_PX = 1040
MIN_TARGET_PX = 44
MIN_TEXT_CONTRAST = 4.5
MIN_GRAPHIC_CONTRAST = 3.0

SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 24
# The fixed Streamlit header is 60 px tall at every width; keep the first block clear of it.
PAGE_TOP_PX = 76
NARROW_SCREEN_PX = 640
PAGE_BOTTOM_PX = 56

FONT_SERIF = "'Instrument Serif', Georgia, serif"
FONT_SANS = "'Inter', 'Segoe UI', sans-serif"

_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def _luminance(color: str) -> float:
    if not isinstance(color, str) or not _HEX.fullmatch(color):
        raise ValueError(f"expected a '#rrggbb' color, got {color!r}")
    channels = []
    for i in (1, 3, 5):
        c = int(color[i : i + 2], 16) / 255
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ramp(t: float, accent: str = ACCENT) -> str:
    """Colour at position ``t`` (clamped to [0, 1]) from RAMP_LOW to ``accent``, as ``rgb(r,g,b)``."""
    u = min(1.0, max(0.0, t))
    lo = [int(RAMP_LOW[i : i + 2], 16) for i in (1, 3, 5)]
    hi = [int(accent[i : i + 2], 16) for i in (1, 3, 5)]
    r, g, b = (round(a + (z - a) * u) for a, z in zip(lo, hi, strict=True))
    return f"rgb({r},{g},{b})"


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 2.x contrast ratio between two '#rrggbb' colors, in [1, 21]."""
    hi, lo = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


_FONT_DIR = "app/static/fonts"


def _font_face(family: str, file: str, style: str, weight: str) -> str:
    return (
        f"@font-face {{ font-family: '{family}'; src: url('{_FONT_DIR}/{file}') "
        f"format('woff2'); font-style: {style}; font-weight: {weight}; "
        "font-display: swap; }"
    )


# Tabular lining numerals keep chart columns aligned; the modebar is hidden (static charts).
CHART_CSS = (
    ".js-plotly-plot text, .js-plotly-plot .hovertext text { "
    'font-feature-settings: "tnum" 1, "cv11" 1; font-variant-numeric: tabular-nums; }\n'
    ".js-plotly-plot .modebar-container { display: none; }"
)


def css(wide: bool = False) -> str:
    """Page stylesheet; ``wide`` widens the content column for the chart pages."""
    from astro_triage.dashboard import (
        style_kit,  # deferred: style_kit imports this module
    )

    kit = style_kit.kit_css()
    width = WIDE_WIDTH_PX if wide else READING_WIDTH_PX
    faces = "\n".join(
        [
            _font_face(
                "Instrument Serif", "instrument-serif-latin-400.woff2", "normal", "400"
            ),
            _font_face(
                "Instrument Serif",
                "instrument-serif-latin-400-italic.woff2",
                "italic",
                "400",
            ),
            _font_face("Inter", "inter-latin-400-600.woff2", "normal", "400 600"),
        ]
    )
    return f"""
<style>
{faces}
html, body, [class*="css"], .stApp {{ font-family: {FONT_SANS}; }}
.stApp {{ background: {BG}; color: {TEXT}; }}
[data-testid="stHeader"] {{ background: transparent; }}
@media (max-width: {NARROW_SCREEN_PX}px) {{ [data-testid="stHeader"] {{ background: {BG}; }} }}
.block-container {{ max-width: {width}px; padding-top: {PAGE_TOP_PX}px; padding-bottom: {PAGE_BOTTOM_PX}px; }}[data-testid="stVerticalBlock"] {{ gap: {SPACE_MD}px; }}
.stApp, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ font-size: {BODY_PX}px; line-height: 1.55; font-weight: 400; letter-spacing: normal; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ font-size: {MIN_SECONDARY_PX}px; line-height: 1.5; color: {TEXT_SECONDARY}; }}
h1 {{ font-family: {FONT_SERIF} !important; font-weight: 400 !important; color: {TEXT}; letter-spacing: normal; padding: {SPACE_SM}px 0; }}
h2, h3 {{ font-family: {FONT_SANS} !important; font-weight: 600 !important; font-size: {HEADING_PX}px !important; line-height: 1.25; color: {TEXT}; letter-spacing: normal; padding: {SPACE_SM}px 0 {SPACE_XS}px; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] input {{ font-size: {CONTROL_PX}px; }}
[data-testid="stSidebar"] h3 {{ font-size: 20px !important; }}
[data-testid="stButtonGroup"] button p, [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, .stButton button p {{ font-size: {CONTROL_PX}px; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{ font-family: {FONT_SERIF}; font-weight: 400; font-size: 48px; line-height: 1.1; }}
[data-testid="stMarkdownContainer"] p {{ margin: 0 0 {SPACE_SM}px; }}
.small {{ color: {TEXT_SECONDARY}; font-size: {MIN_SECONDARY_PX}px; line-height: 1.5; }}
.big {{ font-family: {FONT_SERIF}; font-weight: 400; font-size: 56px; line-height: 1; color: {TEXT}; }}
.headline {{ font-family: {FONT_SERIF}; font-weight: 400; font-size: 34px; line-height: 1.2; color: {TEXT}; margin: {SPACE_SM}px 0 {SPACE_MD}px; }}
.seal {{ display: inline-block; border: 1px solid {ACCENT}; padding: 0.1rem 0.6rem; color: {TEXT}; font-size: {MIN_SECONDARY_PX}px; letter-spacing: 0.04em; }}
.rule {{ border-top: 1px solid {HAIRLINE}; margin: {SPACE_MD}px 0; }}
.notice {{ border-left: 2px solid {ACCENT}; padding: {SPACE_XS}px 0 {SPACE_XS}px {SPACE_MD}px; margin: {SPACE_XS}px 0; color: {TEXT}; }}
.err {{ color: {ACCENT}; font-weight: 600; }}
[data-testid="stButtonGroup"] button {{ font-size: {CONTROL_PX}px; min-height: {MIN_TARGET_PX}px; }}
[data-testid="stExpander"] {{ border: 1px solid {HAIRLINE}; background: transparent; margin: {SPACE_XS}px 0; }}
{CHART_CSS}
{kit}
</style>
"""


def plotly_layout(height: int, **overrides: object) -> dict[str, object]:
    """Kwargs for ``fig.update_layout``; ``overrides`` replace top-level keys (shallow merge)."""
    layout: dict[str, object] = {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": FONT_SANS,
            "size": MIN_SECONDARY_PX,
            "color": TEXT_SECONDARY,
        },
        "showlegend": False,
        "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
    }
    layout.update(overrides)
    return layout
