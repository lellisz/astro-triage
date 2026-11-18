"""Céu and Método layout: calm section pattern, collapsed technical tables, language, escaping."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import method_notes as N
from astro_triage.dashboard import sky_text, views_method, views_sky
from astro_triage.dashboard.html_system import CARD_LEVEL, PAGE_LANG

DASH = Path(views_sky.__file__).parent
EVIL = "<b>&"
ESCAPED = "&lt;b&gt;&amp;"


class _Capture:
    def __init__(self, monkeypatch, *mods):
        self.sent: list[str] = []
        for mod in mods:
            monkeypatch.setattr(
                mod.st, "markdown", lambda html, **_: self.sent.append(html)
            )


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


@pytest.mark.parametrize("tech", [False, True])
def test_method_page_wraps_every_fragment_in_the_language_root(monkeypatch, tech):
    cap = _Capture(monkeypatch, views_method)
    monkeypatch.setattr(views_method, "tech_on", lambda: tech)
    views_method.method_page(_REL)
    css, *frags = cap.sent
    assert css.startswith("<style>") and frags
    assert all(f"lang='{PAGE_LANG}'" in f for f in frags)
    if tech:
        assert "<details" in "".join(frags)
        assert "<h" not in "".join(frags)


def test_sky_module_injects_system_css_and_keeps_the_map_untouched():
    src = (DASH / "views_sky.py").read_text(encoding="utf-8")
    assert "system_css()" in src and "build_map(stats, by_rate, floor, hide)" in src
    assert "at-cap" not in src and 'st.markdown(\n        f"' not in src


def _has_digits(text: str) -> bool:
    text = re.sub(r"\b(19|20)\d{2}\b", "", text)  # citation years
    text = re.sub(r"\.\d[f%]", "", text)  # format specs
    text = re.sub(r"\b[A-Za-z_]+\d*[A-Za-z_]*\b", "", text)  # DR3, P2, identifiers
    return bool(re.search(r"\d", text))


@pytest.mark.parametrize("name", ["method_notes.py", "html_method.py", "adr_board.py"])
def test_no_digit_in_string_literals(name):
    tree = ast.parse((DASH / name).read_text(encoding="utf-8"))
    docs = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
    }
    bad = [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docs
        and _has_digits(n.value)
    ]
    assert not bad, bad
