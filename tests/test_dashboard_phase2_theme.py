"""Phase 2 dashboard: theme tokens, contrast, CSS fonts and Streamlit config."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from astro_triage.dashboard import charts_base, theme

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "src" / "astro_triage" / "dashboard" / "static" / "fonts"


def test_contrast_known_values():
    assert theme.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert theme.contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)
    assert theme.contrast_ratio("#777777", "#777777") == pytest.approx(1.0)


@pytest.mark.parametrize("bad", ["fff", "#12345", "red", None, "#gggggg"])
def test_contrast_rejects_malformed(bad):
    with pytest.raises(ValueError):
        theme.contrast_ratio(bad, theme.BG)


@pytest.mark.parametrize(
    "fg,bg",
    [
        (theme.TEXT, theme.BG),
        (theme.TEXT_SECONDARY, theme.BG),
        (theme.TEXT, theme.SECONDARY_BG),
        (theme.TEXT_SECONDARY, theme.SECONDARY_BG),
        (theme.ACCENT, theme.BG),
        (theme.SOFT, theme.BG),
        (theme.ACCENT, theme.SECONDARY_BG),
        (theme.SOFT, theme.SECONDARY_BG),
    ],
)
def test_text_pairs_meet_aa(fg, bg):
    assert theme.contrast_ratio(fg, bg) >= 4.5


def test_dash_graphic_contrast():
    assert theme.contrast_ratio(theme.DASH, theme.BG) >= 3.0


def test_ramp_low_has_graphic_contrast():
    assert theme.contrast_ratio(theme.RAMP_LOW, theme.BG) >= theme.MIN_GRAPHIC_CONTRAST


def test_chart_tokens_follow_theme():
    assert charts_base.ERROR == theme.ACCENT
    assert charts_base.DATA == theme.SOFT


def test_every_text_constant_in_charts_base_meets_aa():
    names = [n for n in dir(charts_base) if n.startswith("TEXT")]
    assert {"TEXT", "TEXT_SOFT", "TEXT_MUTED"} <= set(names)
    for n in names:
        assert theme.contrast_ratio(getattr(charts_base, n), theme.BG) >= 4.5, n


def test_css_font_faces_point_to_existing_files():
    css = theme.css()
    assert css.count("@font-face") == 3
    urls = re.findall(r"url\('([^']+)'\)", css)
    assert len(urls) == 3
    for url in urls:
        assert url.startswith("app/static/fonts/")
        assert (FONTS / url.rsplit("/", 1)[1]).is_file(), url


def test_css_has_no_external_or_retired_tokens():
    css = theme.css()
    for banned in ("#8EC5FF", "Fraunces", "googleapis"):
        assert banned.lower() not in css.lower()
    for cls in (".small", ".big", ".headline", ".seal"):
        assert cls in css


@pytest.fixture(scope="module")
def config():
    path = ROOT / ".streamlit" / "config.toml"
    with path.open("rb") as fh:
        return tomllib.load(fh)


def test_streamlit_config_values(config):
    assert config["client"]["toolbarMode"] == "minimal"
    assert config["server"]["enableStaticServing"] is True
    assert config["theme"]["base"] == "dark"
    assert config["theme"]["backgroundColor"] == theme.BG
    assert config["theme"]["secondaryBackgroundColor"] == theme.SECONDARY_BG
    assert config["theme"]["textColor"] == theme.TEXT
    assert config["theme"]["primaryColor"] == theme.ACCENT


def test_streamlit_config_has_no_headless_or_address(config):
    assert "headless" not in config.get("server", {})
    assert "address" not in config.get("server", {})
