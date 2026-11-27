"""Método page: registration, wording and limitation fallback, without loading a real release."""

from __future__ import annotations

import re
from types import SimpleNamespace

from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import limits_pt


def _rel():
    meta = {
        "n_test": 1234567,
        "n_test_pixels": 321,
        "n_boot": 777,
        "floor": {"min_objects": 55, "min_pixels": 13},
        "pi_src_combmod": {"QSO": 0.000989},
    }
    return SimpleNamespace(meta=meta, results={"limitations": []})












def test_glossary_has_no_digits():
    # Só os identificadores dos priors e do catálogo podem ter algarismo; nenhum número de dado.
    text = " ".join(t for pair in HM.GLOSSARY for t in pair if t)
    assert not re.search(r"\d", re.sub(r"\b(?:P[123]|DR3)\b", "", text))


def test_limitations_translated_and_original_fallback():
    known = "No DR14Q crossmatch."
    unknown = "Some brand new caveat 12."
    lines = limits_pt.limitation_lines([known, unknown])
    assert lines[0] == "- Sem crossmatch com o DR14Q."
    assert lines[1] == f"- {unknown} (sem tradução)"
