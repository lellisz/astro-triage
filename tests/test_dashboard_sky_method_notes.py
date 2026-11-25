"""The neighbour caveat of ADR 0003 reaches the sky and method pages without any digit."""

from __future__ import annotations

import re

from astro_triage.dashboard import method_notes as N


def test_neighbour_note_names_the_adr_and_the_limit() -> None:
    assert "divisão do céu" in N.NEIGHBOUR_NOTE
    assert "não há margem de exclusão" in N.NEIGHBOUR_NOTE
    assert "medida, não removida" in N.NEIGHBOUR_NOTE


def test_neighbour_note_has_no_digit() -> None:
    assert not re.search(r"\d", N.NEIGHBOUR_NOTE)




