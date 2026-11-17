"""The neighbour caveat of ADR 0003 reaches the sky and method pages without any digit."""

from __future__ import annotations

import re

from astro_triage.dashboard import method_notes as N
from astro_triage.dashboard import views_method, views_sky


def test_neighbour_note_names_the_adr_and_the_limit() -> None:
    assert "divisão do céu" in N.NEIGHBOUR_NOTE
    assert "não há margem de exclusão" in N.NEIGHBOUR_NOTE
    assert "medida, não removida" in N.NEIGHBOUR_NOTE


def test_neighbour_note_has_no_digit() -> None:
    assert not re.search(r"\d", N.NEIGHBOUR_NOTE)


def test_method_reading_includes_the_neighbour_note() -> None:
    facts = views_method.MethodFacts("1", "2", "3", "4", "5", "6", "7")
    assert any(N.NEIGHBOUR_NOTE in p for p in views_method.reading_paragraphs(facts))


def test_sky_view_imports_the_note_module() -> None:
    assert views_sky.method_notes is N
