"""Palette and class/model labels shared by the dashboard text builders and the web exporter.

The Plotly figure layout lived here while the Streamlit dashboard drew the charts; the web front
end draws its own SVG, so only the constants remain.
"""

from __future__ import annotations

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
