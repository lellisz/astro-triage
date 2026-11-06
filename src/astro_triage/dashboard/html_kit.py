"""Pure HTML builders for the design kit (classes in ``style_kit``). No streamlit import.

Every text argument is escaped with ``html.escape``. Arguments named ``*_html`` are
trusted markup, built by the other functions here. Callers pass computed values only.
Output is single-line (no blank lines, no indentation) so Markdown keeps it as HTML.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from astro_triage.dashboard import theme

PERCENT = 100.0


def _t(text: object) -> str:
    return escape(str(text), quote=True)


def _pct(value: float) -> str:
    """``value`` as a CSS percentage clamped to [0, 100]."""
    return f"{min(PERCENT, max(0.0, float(value))):.2f}%"


def section(title: str, lead: str, how: str | None = None) -> str:
    """Section header: title (.at-h2), lead sentence and an optional 'Como ler' line."""
    how_html = f"<p class='at-how'><b>Como ler:</b> {_t(how)}</p>" if how else ""
    return (
        "<div class='at-sec'>"
        f"<div class='at-h2' role='heading' aria-level='2'>{_t(title)}</div>"
        f"<p class='at-lead'>{_t(lead)}</p>{how_html}</div>"
    )


def axis(ticks: list[tuple[float, str]]) -> str:
    """Ruler with ``(position in %, label)`` ticks."""
    spans = "".join(
        f"<span class='at-tick' style='left:{_pct(pos)}'>{_t(text)}</span>"
        for pos, text in ticks
    )
    return f"<div class='at-axis'>{spans}</div>"


def dot(pos: float, hollow: bool = False, color: str = theme.ACCENT) -> str:
    """Mark on a plot track at ``pos`` %: filled with ``color`` or hollow (ring)."""
    if hollow:
        return f"<div class='at-doth' style='left:{_pct(pos)}'></div>"
    return (
        f"<div class='at-dotf' style='left:{_pct(pos)};background:{_t(color)}'></div>"
    )


def span(lo: float, hi: float, ci: bool = False, opacity: float = 1.0) -> str:
    """Line between ``lo`` and ``hi`` (%): neutral link, or the accent interval if ``ci``."""
    a, b = sorted((min(PERCENT, max(0.0, lo)), min(PERCENT, max(0.0, hi))))
    cls = "at-ci" if ci else "at-link"
    alpha = min(1.0, max(0.0, opacity))
    return (
        f"<div class='{cls}' style='left:{a:.2f}%;width:{b - a:.2f}%;"
        f"opacity:{alpha:.2f}'></div>"
    )


def dot_row(label: str, sub: str, plot_html: str, col1: str, col2: str) -> str:
    """Row with name and caption, a plot track holding ``plot_html`` and two numeric columns."""
    return (
        f"<div class='at-rowp'><div>{_t(label)}<div class='at-cap'>{_t(sub)}</div></div>"
        f"<div class='at-plot'>{plot_html}</div>"
        f"<div class='at-num at-strong'>{_t(col1)}</div><div class='at-num'>{_t(col2)}</div></div>"
    )


def h_bar(
    label: str, frac: float, value_text: str, color: str, extra_text: str = ""
) -> str:
    """Horizontal bar of length ``frac`` (clamped to [0, 1]) with value and extra columns."""
    width = _pct(frac * PERCENT)
    return (
        f"<div class='at-rowe'><div>{_t(label)}</div>"
        f"<div class='at-track'><div class='at-fill' style='width:{width};background:{_t(color)}'></div></div>"
        f"<div class='at-num at-strong'>{_t(value_text)}</div>"
        f"<div class='at-num at-cap'>{_t(extra_text)}</div></div>"
    )


def box(title: str, inner_html: str) -> str:
    """Framed block with an uppercase title and trusted inner markup."""
    return (
        f"<div class='at-box'><div class='at-box-t'>{_t(title)}</div>{inner_html}</div>"
    )


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Numeric table (.at-tt): first column left-aligned, the rest right-aligned."""
    head = "".join(f"<th>{_t(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_t(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table class='at-tt'><tr>{head}</tr>{body}</table>"


def stat(value: str, caption: str, sub: str | None = None, accent: bool = False) -> str:
    """Large number with a caption and an optional small line; ``accent`` paints it blue."""
    cls = "at-big at-accent" if accent else "at-big"
    sub_html = f"<div class='at-cap'>{_t(sub)}</div>" if sub else ""
    return (
        f"<div class='at-stat'><div class='{cls}'>{_t(value)}</div>"
        f"<div class='at-stat-cap'>{_t(caption)}</div>{sub_html}</div>"
    )
