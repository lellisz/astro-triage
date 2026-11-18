"""Método page v2: one rule text and three limitations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from test_dashboard_comparacao import TIE, _results

from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import method_notes as notes
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard import views_dl, views_method
from astro_triage.dashboard.html_system import NA, method_note

DASH = Path(views_method.__file__).parent
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


def _method_html(monkeypatch, tech):
    sent: list[str] = []
    monkeypatch.setattr(views_method.st, "markdown", lambda h, **_: sent.append(h))
    monkeypatch.setattr(views_method, "tech_on", lambda: tech)
    views_method.method_page(_REL)
    return "".join(sent)


def test_rule_text_method_equals_comparison(monkeypatch):
    note = method_note(VR.rule_text())
    comparison = views_dl.page_html(CM.build(_results(ftt=(TIE, "win", "tie"))), "")
    assert note in comparison and note in HM.rule_html()
    for tech in (False, True):
        assert note in _method_html(monkeypatch, tech)


def test_rule_has_a_single_source(monkeypatch):
    monkeypatch.setattr(VR, "rule_lines", lambda: ("REGRA UNICA",))
    comparison = views_dl.page_html(CM.build(_results(ftt=(TIE, "win", "tie"))), "")
    assert "REGRA UNICA" in comparison and "REGRA UNICA" in HM.rule_html()


def test_old_next_step_card_absent():
    old = ("Comparar com " + "redes neurais", "Próximo " + "passo")
    for path in DASH.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(s in text for s in old), path.name


def test_three_limitations_present():
    out = HM.three_limits_html(["Selection by crossmatch: the matched x."])
    assert HM.THREE_LIMITS_TITLE in out
    assert out.count("<li>") == 3
    assert notes.NEIGHBOUR_NOTE in out and notes.PRIOR_NOTE in out
    assert "Seleção por crossmatch" in out or "Selection by crossmatch" in out
    assert f"<li>{NA}</li>" in HM.three_limits_html([])


def test_page_shows_the_three_limitations_and_links_as_na(monkeypatch):
    for tech in (False, True):
        html = _method_html(monkeypatch, tech)
        assert HM.THREE_LIMITS_TITLE in html and notes.PRIOR_NOTE in html
        assert HM.LINKS_TITLE in html and HM.GLOSSARY_TITLE in html


def test_glossary_missing_term_is_na():
    out = HM.glossary_html((("Termo", None), ("Outro", " "), ("Ok", "definição")))
    assert out.count(f"<span>{NA}</span>") == 2 and "definição" in out
    assert "<details" in HM.glossary_details()

