"""Céu and Método layout: calm section pattern, collapsed technical tables, language, escaping."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import method_notes as N
from astro_triage.dashboard import sky_text
from astro_triage.dashboard.html_system import CARD_LEVEL

EVIL = "<b>&"
ESCAPED = "&lt;b&gt;&amp;"


def _bands() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "band": ["A", "B", "C"],
            "n_pixels": [1, 2, 3],
            "n": [10, 20, 30],
            "errors": [1, 2, 3],
            "rate": [0.1, 0.1, 0.1],
            "ci_lo": [0.05, 0.05, 0.05],
            "ci_hi": [0.2, 0.2, 0.2],
        }
    )


def _top() -> pd.DataFrame:
    return pd.DataFrame(
        {"l": [10.0], "b": [-20.0], "n": [40], "errors": [5], "rate": [0.125]}
    )


def test_sky_header_uses_the_section_pattern_and_escapes():
    out = sky_text.header_html(EVIL, EVIL, EVIL)
    assert "class='hs-sec'" in out and "class='hs-sen'" in out
    assert f"role='heading' aria-level='{CARD_LEVEL}'" in out
    assert EVIL not in out and ESCAPED in out
    assert "\n" not in out


def test_neighbour_note_is_small_text():
    from astro_triage.dashboard.html_system import small_note

    out = small_note(N.NEIGHBOUR_NOTE)
    assert out.startswith("<p class='hs-small'>") and "medida, não removida" in out


def test_sky_tables_are_collapsed_details_without_box():
    from astro_triage.dashboard.html_system import details

    bands = details(sky_text.BANDS_TITLE, sky_text.bands_body(_bands()))
    top = details(sky_text.top_title(5), sky_text.top_body(_top(), 5))
    for out in (bands, top):
        assert out.startswith("<details class='hs-det'>") and " open" not in out
        assert "<table" in out and "at-box" not in out
    assert sky_text.bands_box(_bands()).startswith("<div class='at-box'>")


def test_top_body_formats_the_position_from_data():
    assert "l = 10°, b = −20°" in sky_text.top_body(_top(), 5)


def test_method_blocks_are_collapsed_details():
    reading = HM.reading_html([EVIL])
    glossary = HM.glossary_details()
    limits = HM.limitations_html(["No DR14Q crossmatch.", EVIL])
    for out, title in (
        (reading, HM.READING_TITLE),
        (glossary, HM.GLOSSARY_TITLE),
        (limits, HM.LIMITS_TITLE),
    ):
        assert out.startswith("<details class='hs-det'>")
        assert f"<summary>{title}</summary>" in out and " open" not in out
    assert EVIL not in reading + limits and ESCAPED in reading + limits
    assert "Sem crossmatch com o DR14Q." in limits
    assert HM.ORIGINAL_TITLE in limits and limits.count("<details") == 2


def test_method_rule_box_and_sources_stay_visible():
    box = HM.rule_html()
    assert "maior que δ" in box and "<details" not in box
    src = HM.sources_html()
    assert "<details" not in src and "Bailer-Jones" in src
    assert f"aria-level='{CARD_LEVEL}'" in src


def test_method_headings_and_simple_mode_content():
    out = HM.lead_html()
    assert f"role='heading' aria-level='{CARD_LEVEL}'" in out and "hs-sec" in out
    simple = HM.simple_html()
    assert HM.LEAD in simple and "Modo técnico" in simple and "hs-small" in simple
    assert "<details" not in simple
    assert "hs-small" in HM.sources_line()


def test_method_text_and_mix_caveat_are_secondary_markup():
    out = HM.method_html({"STAR": 0.5, "GALAXY": 0.3, "QSO": 0.2})
    assert HM.METHOD_TEXT in out and "hs-small" in out


_REL = SimpleNamespace(
    meta={
        "n_test": 10,
        "n_test_pixels": 3,
        "n_boot": 7,
        "floor": {"min_objects": 5, "min_pixels": 2},
        "pi_src_combmod": {"QSO": 0.001},
    },
    test=pd.DataFrame({"true": ["STAR", "GALAXY", "QSO"]}),
    results={"limitations": ["Some caveat."]},
)


