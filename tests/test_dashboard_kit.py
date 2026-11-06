"""Design kit: theme.ramp, html_kit escaping and clamping, style_kit and contrast."""

from __future__ import annotations

import re

import pytest

from astro_triage.dashboard import html_kit as H
from astro_triage.dashboard import style_kit, theme


def _rgb(text: str) -> tuple[int, ...]:
    return tuple(
        int(n) for n in re.fullmatch(r"rgb\((\d+),(\d+),(\d+)\)", text).groups()
    )


def _hex(color: str) -> tuple[int, ...]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def test_ramp_extremes_match_endpoints():
    assert _rgb(theme.ramp(0.0)) == _hex(theme.RAMP_LOW)
    assert _rgb(theme.ramp(1.0)) == _hex(theme.ACCENT)
    assert _rgb(theme.ramp(1.0, "#ffffff")) == (255, 255, 255)


@pytest.mark.parametrize(("raw", "clamped"), [(-3.0, 0.0), (7.5, 1.0)])
def test_ramp_clamps(raw, clamped):
    assert theme.ramp(raw) == theme.ramp(clamped)


def test_ramp_midpoint_is_between_endpoints():
    mid = _rgb(theme.ramp(0.5))
    for m, lo, hi in zip(mid, _hex(theme.RAMP_LOW), _hex(theme.ACCENT), strict=True):
        assert min(lo, hi) <= m <= max(lo, hi)


EVIL = "<script>alert('x')</script> & \"q\""


@pytest.mark.parametrize(
    "html",
    [
        H.section(EVIL, EVIL, EVIL),
        H.axis([(50, EVIL)]),
        H.dot_row(EVIL, EVIL, "", EVIL, EVIL),
        H.h_bar(EVIL, 0.5, EVIL, EVIL, EVIL),
        H.box(EVIL, ""),
        H.table([EVIL], [[EVIL]]),
        H.stat(EVIL, EVIL, EVIL),
    ],
)
def test_text_arguments_are_escaped(html):
    assert "<script" not in html and "alert(&#x27;x&#x27;)" in html


@pytest.mark.parametrize(
    ("frac", "width"), [(-1.0, "0.00%"), (3.0, "100.00%"), (0.25, "25.00%")]
)
def test_h_bar_clamps_fraction(frac, width):
    assert f"width:{width}" in H.h_bar("a", frac, "v", theme.ACCENT)


def test_section_how_is_optional_and_output_is_single_line():
    assert "Como ler" not in H.section("t", "l")
    assert "Como ler" in H.section("t", "l", "h")
    assert "\n" not in H.section("t", "l", "h") + H.table(["a"], [["b"]])


def test_stat_accent_and_sub():
    assert "at-accent" in H.stat("1", "c", accent=True)
    assert "at-accent" not in H.stat("1", "c")
    assert "'at-cap'" in H.stat("1", "c", "s") and "'at-cap'" not in H.stat("1", "c")


def test_marks_clamp_positions():
    assert "left:100.00%" in H.dot(250)
    assert "left:0.00%;width:100.00%" in H.span(-5, 500)


def test_theme_css_includes_kit_and_uses_constants():
    css = theme.css()
    assert style_kit.kit_css() in css
    for cls in (".at-sec", ".at-rowp", ".at-plot", ".at-dotf", ".at-tt"):
        assert cls in css
    assert f"max-width: {theme.WIDE_WIDTH_PX}px" in theme.css(wide=True)
    assert theme.WIDE_WIDTH_PX == 1040


@pytest.mark.parametrize(
    "fg", [theme.TEXT, theme.TEXT_SECONDARY, theme.SOFT, theme.ACCENT]
)
def test_text_colors_meet_aa_on_both_backgrounds(fg):
    for bg in (theme.BG, theme.SECONDARY_BG):
        assert theme.contrast_ratio(fg, bg) >= theme.MIN_TEXT_CONTRAST


@pytest.mark.parametrize("fg", [theme.DASH, theme.RAMP_LOW])
def test_graphic_colors_meet_3_to_1_on_bg(fg):
    assert theme.contrast_ratio(fg, theme.BG) >= theme.MIN_GRAPHIC_CONTRAST
