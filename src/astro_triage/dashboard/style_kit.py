"""CSS of the design kit (classes prefixed ``at-``). Pure: does not import streamlit.

``theme.css()`` already includes ``kit_css()``; the markup comes from ``html_kit``.
"""

from __future__ import annotations

from astro_triage.dashboard import theme as T

_ROW = "display: grid; column-gap: 20px; align-items: center;"
_MARK = "position: absolute; top: 50%;"
_DOT = f"{_MARK} width: 14px; height: 14px; margin: -7px 0 0 -7px; border-radius: 50%;"
_BAR = f"{_MARK} height: 2px; margin-top: -1px;"
_GRID_X = f"repeating-linear-gradient(90deg, {T.GRID} 0 1px, transparent 1px 20%)"


def kit_css() -> str:
    """Rules for the ``at-*`` vocabulary: sections, rows, plot tracks, marks, boxes, tables."""
    return f"""
.at-sec {{ padding: 72px 0 8px; border-top: 1px solid {T.HAIRLINE}; margin-top: 72px; }}
.at-h2 {{ font-size: 28px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; line-height: 1.2; color: {T.TEXT}; }}
.at-lead {{ color: {T.TEXT_SECONDARY}; margin: 0 0 8px; max-width: 680px; }}
.at-how {{ color: {T.TEXT_SECONDARY}; font-size: {T.MIN_SECONDARY_PX}px; margin: 0 0 32px; max-width: 680px; }}
.at-how b {{ color: {T.SOFT}; font-weight: 600; }}
.at-cap {{ font-size: 14px; color: {T.TEXT_SECONDARY}; line-height: 1.45; }}
.at-hdr {{ font-size: 13px; color: {T.TEXT_SECONDARY}; letter-spacing: 0.04em; text-transform: uppercase; }}
.at-ov {{ overflow-x: auto; }}
.at-in {{ min-width: 640px; }}
.at-rowp {{ {_ROW} grid-template-columns: 210px minmax(0, 1fr) 92px 92px; min-height: 64px; border-bottom: 1px solid {T.HAIRLINE}; }}
.at-rowe {{ {_ROW} grid-template-columns: 190px minmax(0, 1fr) 72px 72px; min-height: 46px; }}
.at-rowg {{ {_ROW} grid-template-columns: 150px minmax(0, 1fr) 84px 76px; min-height: 52px; border-top: 1px solid {T.HAIRLINE}; }}
.at-plot {{ position: relative; height: 100%; min-height: 52px; border-left: 1px solid {T.HAIRLINE}; border-right: 1px solid {T.HAIRLINE}; background: {_GRID_X}; }}
.at-axis {{ position: relative; height: 20px; }}
.at-tick {{ position: absolute; font-size: 13px; color: {T.TEXT_SECONDARY}; transform: translateX(-50%); white-space: nowrap; }}
.at-num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.at-strong {{ font-weight: 600; }}
.at-dotf {{ {_DOT} }}
.at-doth {{ {_DOT} box-sizing: border-box; border: 3px solid {T.SOFT}; background: {T.BG}; }}
.at-link {{ {_BAR} background: {T.DASH}; }}
.at-ci {{ {_BAR} background: {T.ACCENT}; }}
.at-track {{ height: 22px; background: {T.SECONDARY_BG}; border-radius: 4px; }}
.at-fill {{ height: 100%; border-radius: 4px; }}
.at-big {{ font-family: {T.FONT_SERIF}; font-size: 64px; line-height: 1; font-variant-numeric: tabular-nums; color: {T.TEXT}; }}
.at-big.at-accent {{ color: {T.ACCENT}; }}
.at-stat-cap {{ margin-top: 6px; }}
.at-step {{ border: 1px solid {T.HAIRLINE}; border-radius: 14px; padding: 22px; background: {T.SECONDARY_BG}; }}
.at-chip {{ display: inline-block; font-size: 13px; font-weight: 600; padding: 3px 10px; border-radius: 999px; margin-bottom: 12px; background: {T.SECONDARY_BG}; color: {T.SOFT}; }}
.at-gl {{ display: grid; grid-template-columns: 160px minmax(0, 1fr); column-gap: 24px; padding: 12px 0; border-bottom: 1px solid {T.HAIRLINE}; }}
.at-box {{ border: 1px solid {T.HAIRLINE}; background: {T.SECONDARY_BG}; border-radius: 12px; padding: 18px 22px; margin-top: 28px; }}
.at-box-t {{ margin: 0 0 10px; font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: {T.TEXT_SECONDARY}; font-weight: 600; }}
.at-box p, .at-box li {{ font-size: {T.MIN_SECONDARY_PX}px; margin: 0 0 8px; color: {T.SOFT}; }}
.at-tt {{ width: 100%; border-collapse: collapse; font-size: {T.MIN_SECONDARY_PX}px; font-variant-numeric: tabular-nums; }}
.at-tt th {{ text-align: right; font-weight: 500; color: {T.TEXT_SECONDARY}; font-size: 13px; padding: 6px 8px; border-bottom: 1px solid {T.HAIRLINE}; }}
.at-tt td {{ text-align: right; padding: 8px; border-bottom: 1px solid {T.HAIRLINE}; }}
.at-tt th:first-child, .at-tt td:first-child {{ text-align: left; }}
[data-testid="stToggle"] label p {{ font-size: {T.CONTROL_PX}px; font-weight: 500; }}
"""
