"""Comparação page v2: grid, fixed point, power phrase, absolute purity, no typed verdict text."""

from __future__ import annotations

import re

from test_dashboard_comparacao import TIE, _results

from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import compare_text as CT


def _model():
    return CM.build(_results(ftt=(TIE, "win", "tie")))






def test_grid_has_four_cells_qso_galaxy_by_mlp_ftt():
    got = [(g.cls, g.net) for g in _model().grid]
    assert got == [
        ("QSO", "MLP"),
        ("QSO", "FTT"),
        ("GALAXY", "MLP"),
        ("GALAXY", "FTT"),
    ]
    assert [n for n, _ in _model().pairs] == ["MLP", "FTT"]












def test_answer_has_no_banned_words():
    text = CT.answer(_model())
    assert not re.search(r"\b(só|melhor|equivalente)\b", text)
    assert "MLP" in text and "FTT" in text


