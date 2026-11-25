"""Phase 7 app pieces: navigation, cache stamp, notes, errors, theme, constants (synthetic)."""

from __future__ import annotations

import re

from astro_triage.dashboard import constants, theme

M = ("lgbm_full", "lgbm_noind")


# ----------------------------------------------------------------------- (d) cache


# ------------------------------------------------------------ (e) notes and errors


# ----------------------------------------------------------------------- (f) theme
def test_theme_width_and_button_target() -> None:
    narrow, wide = theme.css(wide=False), theme.css(wide=True)
    assert (
        f"max-width: {theme.READING_WIDTH_PX}px" in narrow
        and theme.READING_WIDTH_PX == 800
    )
    assert f"max-width: {theme.WIDE_WIDTH_PX}px" in wide
    assert str(theme.WIDE_WIDTH_PX) not in narrow and "800px" not in wide
    for css in (narrow, wide):
        rule = re.search(r'\[data-testid="stButtonGroup"\] button\s*\{([^}]*)\}', css)
        assert rule and re.search(r"min-height:\s*44px", rule.group(1))
    assert theme.MIN_TARGET_PX == 44


# ---------------------------------------------------------------- (h) constants
def test_constants() -> None:
    assert constants.ADR_NUMBER == 6
