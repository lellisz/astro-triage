"""Método page v2: one rule text and three limitations."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import method_notes as notes
from astro_triage.dashboard.html_system import NA

_REL = SimpleNamespace(
    meta={
        "n_test": 10,
        "n_test_pixels": 3,
        "n_boot": 7,
        "floor": {"min_objects": 5, "min_pixels": 2},
        "pi_src_combmod": {"QSO": 0.001},
    },
    test=pd.DataFrame({"true": ["STAR", "GALAXY", "QSO"]}),
    results={"limitations": ["Selection by crossmatch: x."]},
)










def test_three_limitations_present():
    out = HM.three_limits_html(["Selection by crossmatch: the matched x."])
    assert HM.THREE_LIMITS_TITLE in out
    assert out.count("<li>") == 3
    assert notes.NEIGHBOUR_NOTE in out and notes.PRIOR_NOTE in out
    assert "Seleção por crossmatch" in out or "Selection by crossmatch" in out
    assert f"<li>{NA}</li>" in HM.three_limits_html([])




def test_glossary_missing_term_is_na():
    out = HM.glossary_html((("Termo", None), ("Outro", " "), ("Ok", "definição")))
    assert out.count(f"<span>{NA}</span>") == 2 and "definição" in out
    assert "<details" in HM.glossary_details()

