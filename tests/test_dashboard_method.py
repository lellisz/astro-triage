"""Método page: registration, wording and limitation fallback, without loading a real release."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import limits_pt
from astro_triage.dashboard import method_notes as N
from astro_triage.dashboard import views_method as VM

APP = Path(VM.__file__).with_name("app.py")


def _rel():
    meta = {
        "n_test": 1234567,
        "n_test_pixels": 321,
        "n_boot": 777,
        "floor": {"min_objects": 55, "min_pixels": 13},
        "pi_src_combmod": {"QSO": 0.000989},
    }
    return SimpleNamespace(meta=meta, results={"limitations": []})


@pytest.fixture
def facts():
    return VM.method_facts(_rel())


def test_page_is_registered_last():
    src = APP.read_text(encoding="utf-8")
    assert re.search(
        r'PAGES = \("Visão geral", "Mapa do céu", "Galáxias", "Comparação com DL", "Método"\)',
        src,
    )
    assert "lambda: method_page(rel)" in src


def test_facts_come_from_the_artifact(facts):
    assert facts.n_test == "1.234.567" and facts.n_pixels == "321"
    assert facts.prior_qso == "0,099%" and facts.n_boot == "777"
    assert facts.min_objects == "55" and facts.min_pixels == "13"
    assert facts.completeness == "0,80"


def test_reading_paragraphs_use_the_notes(facts):
    text = "\n".join(VM.reading_paragraphs(facts))
    assert N.PRIOR_NOTE in text
    assert N.sample_note(facts.n_test, facts.n_pixels, facts.prior_qso) in text
    assert N.INTERVAL_NOTE in text and N.MAP_NOTE in text
    assert N.bootstrap_note("777") in text
    assert "1.234.567" in text and "0,099%" in text and "0,80" in text


def test_wording_has_no_seal_or_provisional_label(facts):
    parts = [
        HM.LEAD,
        HM.METHOD_TEXT,
        *VM.reading_paragraphs(facts),
        *VM.usage_paragraphs(),
    ]
    parts += [*HM.SOURCES, *(t for pair in HM.GLOSSARY for t in pair)]
    text = "\n".join(parts)
    assert not re.search(r"provis", text, re.IGNORECASE)
    assert "selo" not in text.lower()
    assert "Delchambre" in text and "Bailer-Jones" in text


def test_glossary_has_no_digits():
    assert not re.search(r"\d", "".join(t for pair in HM.GLOSSARY for t in pair))


def test_limitations_translated_and_original_fallback():
    known = "No DR14Q crossmatch."
    unknown = "Some brand new caveat 12."
    lines = limits_pt.limitation_lines([known, unknown])
    assert lines[0] == "- Sem crossmatch com o DR14Q."
    assert lines[1] == f"- {unknown} (sem tradução)"
