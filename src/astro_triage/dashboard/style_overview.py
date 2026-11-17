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
.ao-r-insc {{ font-family: {T.FONT_SERIF}; font-style: italic; font-weight: 400; font-size: clamp(28px, 4.2vw, 52px); line-height: 1.15; margin: 24px 0 24px; max-width: 900px; color: {T.TEXT}; }}
.ao-r-states {{ display: grid; grid-template-columns: 1fr 1fr; gap: 32px; margin-top: 16px; }}
.ao-r-st {{ border-top: 1px solid {T.HAIRLINE}; padding-top: 16px; min-width: 0; }}
.ao-r-h {{ font-weight: 600; font-size: {T.HEADING_PX}px; color: {T.TEXT}; }}
.ao-r-sub, .ao-r-cap, .ao-r-ciline, .ao-r-prior {{ font-size: {T.MIN_SECONDARY_PX}px; color: {T.TEXT_SECONDARY}; }}
.ao-r-prior {{ margin-top: 4px; }}
.ao-r-warn {{ margin-top: 8px; padding: 6px 10px; border: 1px solid {T.DASH}; font-size: {T.MIN_SECONDARY_PX}px; color: {T.TEXT}; }}
.ao-r-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 16px; }}
.ao-r-m {{ min-width: 0; }}
.ao-r-k {{ font-size: {T.CONTROL_PX}px; color: {T.SOFT}; }}
.ao-r-num, .ao-r-accv {{ font-family: {T.FONT_SERIF}; font-weight: 400; font-size: clamp(44px, 5vw, 72px); line-height: 1; color: {T.ACCENT}; }}
.ao-r-ciline {{ font-family: ui-monospace, Menlo, Consolas, monospace; margin-top: 4px; }}
.ao-r-bar {{ position: relative; height: 12px; background: {T.GRID}; margin: 10px 0 4px; }}
.ao-r-ci {{ position: absolute; top: -4px; bottom: -4px; border: solid {T.TEXT}; border-width: 0 1px; }}
.ao-r-ci b {{ position: absolute; left: 0; right: 0; top: 50%; border-top: 1px solid {T.TEXT}; }}
.ao-r-tick {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: {T.ACCENT}; }}
.ao-r-field {{ margin-top: 20px; }}
.ao-r-field .ao-r-cap {{ margin-top: 8px; }}
.ao-r-acc {{ display: flex; flex-direction: column; gap: 24px; margin-top: 8px; }}
.ao-r-def {{ border-top: 1px solid {T.HAIRLINE}; padding-top: 12px; }}
@media (max-width: 1000px) {{ .ao-r-states {{ grid-template-columns: 1fr; }} }}
@media (max-width: 560px) {{ .ao-r-pair {{ grid-template-columns: 1fr; }} }}
</style>"""
