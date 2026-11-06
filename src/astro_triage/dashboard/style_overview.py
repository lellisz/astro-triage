"""CSS of the Visão geral page (classes prefixed ``ao-``). Pure: does not import streamlit."""

from __future__ import annotations

from astro_triage.dashboard import theme as T

_PLOT_H = 300
_DOT = "position: absolute; width: 14px; height: 14px; margin: 0 0 -7px -7px; border-radius: 50%; box-sizing: border-box;"
_GRID = (
    f"repeating-linear-gradient(0deg, {T.GRID} 0 1px, transparent 1px 25%), "
    f"repeating-linear-gradient(90deg, {T.GRID} 0 1px, transparent 1px 25%)"
)


def overview_css() -> str:
    """``<style>`` block for the hero, step cards, number grid, header row and the triage plot."""
    return f"""<style>
.ao-hero {{ padding-top: 40px; }}
.ao-h1 {{ font-family: {T.FONT_SERIF}; font-weight: 400; font-size: 64px; line-height: 1.05; margin: 0 0 16px; max-width: 840px; color: {T.TEXT}; }}
.ao-lead {{ color: {T.TEXT_SECONDARY}; font-size: 19px; margin: 0 0 40px; max-width: 760px; }}
.ao-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; }}
.ao-chip-a {{ background: {T.ACCENT}; color: {T.BG}; }}
.ao-chip-o {{ border: 1px solid {T.DASH}; }}
.ao-step-t {{ font-weight: 600; margin-bottom: 6px; }}
.ao-step-b {{ font-size: {T.MIN_SECONDARY_PX}px; color: {T.TEXT_SECONDARY}; }}
.ao-nums {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 32px; }}
.ao-sub {{ margin-top: 6px; }}
.ao-hdrrow {{ min-height: 32px; border-bottom: none; }}
.ao-tri {{ display: grid; grid-template-columns: 60px minmax(0, 1fr); column-gap: 12px; }}
.ao-triy {{ position: relative; height: {_PLOT_H}px; }}
.ao-yt {{ position: absolute; right: 0; transform: translateY(50%); font-size: 13px; color: {T.TEXT_SECONDARY}; }}
.ao-plot {{ position: relative; height: {_PLOT_H}px; border-left: 1px solid {T.HAIRLINE}; border-bottom: 1px solid {T.HAIRLINE}; background: {_GRID}; }}
.ao-pa {{ {_DOT} background: {T.ACCENT}; }}
.ao-pb {{ {_DOT} border: 3px solid {T.SOFT}; background: {T.BG}; }}
.ao-la {{ position: absolute; margin: 0 0 20px; font-size: 14px; white-space: nowrap; color: {T.TEXT}; }}
.ao-la-l {{ transform: translateX(-7px); }}
.ao-la-r {{ transform: translateX(calc(-100% + 7px)); }}
.ao-lb {{ position: absolute; right: 20px; margin-bottom: 12px; font-size: 14px; white-space: nowrap; color: {T.TEXT}; }}
.ao-xt {{ margin-top: 6px; }}
.ao-xc {{ text-align: center; margin-top: 6px; }}
.ao-note {{ margin: 14px 0 0; }}
</style>"""
