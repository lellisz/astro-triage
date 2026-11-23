"""Método page blocks as HTML (design kit). Pure: no streamlit; no number is written here.

The decision rule is the one text of ``verdict_rule`` (the same as on the Comparação page).
Deep-learning numbers are not written here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Final

from astro_triage.dashboard import fmt
from astro_triage.dashboard import method_notes as notes
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard.data_headline import PAIR_NAMES
from astro_triage.dashboard.data_release import CLASSES
from astro_triage.dashboard.html_system import (
    NA,
    details,
    method_note,
    section,
    small_note,
)
from astro_triage.dashboard.limits_pt import omit_record_refs, translate

LEAD: Final = (
    "Um modelo de árvores aprende com objetos cujo tipo já é conhecido pelo espectro "
    "e é testado em objetos que nunca viu."
)
HINT: Final = (
    'Ligue o "Modo técnico" no topo para ver os termos, os detalhes do treino '
    "e as métricas completas."
)
METHOD_TEXT: Final = (
    "O modelo é um LightGBM. O classificador do próprio Gaia (DSC) serve de referência, "
    "e a verdade vem dos espectros do SDSS."
)
SOURCES: Final = (
    "Gaia DR3: fotometria, astrometria e as probabilidades do Gaia DSC.",
    "SDSS: espectros que fornecem os rótulos de classe.",
    "Gaia DSC (Delchambre+2023): classificador de referência e origem dos priors P2 e P3.",
    "Bailer-Jones+2019: origem do prior P1.",
)
GLOSSARY: Final[tuple[tuple[str, str | None], ...]] = (
    ("Pureza", "dos que o modelo aponta como de uma classe, quantos são dela."),
    ("Completude", "dos objetos de uma classe que existem, quantos o modelo acha."),
    (
        "Espectro",
        "a medida detalhada da luz de um objeto, que revela o tipo dele com certeza.",
    ),
    ("Prior", "proporção das classes que o classificador assume que existe no céu."),
    (
        "Piso de relato",
        "tamanho mínimo de uma célula para que seu valor seja publicado.",
    ),
    (
        "IC",
        "intervalo de confiança: a faixa em que o valor verdadeiro provavelmente está.",
    ),
)
READING_TITLE: Final = "Como ler os números"
GLOSSARY_TITLE: Final = "Glossário"
LIMITS_TITLE: Final = "Limitações da avaliação"
ORIGINAL_TITLE: Final = "Texto original (inglês, sem as citações internas)"
SOURCES_TITLE: Final = "Fontes"
RULE_TITLE: Final = "Regra de decisão"
THREE_LIMITS_TITLE: Final = "Três limitações"
LINKS_TITLE: Final = "Links"
REFERENCES_PATH: Final = Path(__file__).with_name("references.json")
HTTPS_PREFIX: Final = "https://"
UNTRANSLATED: Final = " (sem tradução)"
CROSSMATCH_PREFIX: Final = "Selection by crossmatch"


def mix_sentence(mix: Mapping[str, float]) -> str:
    parts = ", ".join(
        f"{PAIR_NAMES[c]} {fmt.pt_pct(mix.get(c, 0.0), 0)}" for c in CLASSES
    )
    return (
        "A amostra só tem objetos que o SDSS observou e que casam com o Gaia, então a "
        f"mistura do teste ({parts}) não é a do céu."
    )


def lead_html() -> str:
    return section("Método", LEAD)


def simple_html() -> str:
    return lead_html() + small_note(HINT)


def sources_line() -> str:
    """'Fontes: ...' with the source names (text before the colon of each entry)."""
    names = " · ".join(line.split(":")[0] for line in SOURCES)
    return small_note("Fontes: " + names)


def rule_html() -> str:
    """The decision rule; same renderer and same text as the Comparação page."""
    return section(RULE_TITLE, "", method_note(VR.rule_text()))


def crossmatch_limit(limitations: Sequence[str]) -> str:
    """Translated 'Selection by crossmatch' limitation from the artifact; n/a when absent."""
    for text in limitations:
        if text.startswith(CROSSMATCH_PREFIX):
            line, translated = translate(text)
            return line if translated else omit_record_refs(line) + UNTRANSLATED
    return NA


def three_limits_html(limitations: Sequence[str]) -> str:
    """Catalogue intersection, neighbouring regions and the assumed prior."""
    items = [crossmatch_limit(limitations), notes.NEIGHBOUR_NOTE, notes.PRIOR_NOTE]
    return section(THREE_LIMITS_TITLE, "", _list(items))


def glossary_html(entries: Sequence[tuple[str, str | None]] = GLOSSARY) -> str:
    """Term and definition; a term without a definition shows n/a (none is invented)."""
    return "".join(
        f"<div class='at-gl'><b>{escape(term)}</b><span>{escape((text or '').strip() or NA)}</span></div>"
        for term, text in entries
    )


def _list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{escape(t)}</li>" for t in items) + "</ul>"


def method_html(mix: Mapping[str, float]) -> str:
    """How the model was built and tested: the method sentence and the test-mix caveat."""
    return f"<p class='hs-sen'>{escape(METHOD_TEXT)}</p>" + small_note(
        mix_sentence(mix)
    )


def usage_html(title: str, paragraphs: list[str]) -> str:
    return details(title, "".join(small_note(p) for p in paragraphs))


def reading_html(paragraphs: list[str]) -> str:
    return usage_html(READING_TITLE, paragraphs)


def glossary_details() -> str:
    return details(GLOSSARY_TITLE, glossary_html())


def _references() -> list[tuple[str, str, str]]:
    """(label, note, url) read from the data file; invalid entries are dropped, never raised."""
    try:
        data = json.loads(REFERENCES_PATH.read_bytes())
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    refs = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        label, note, url = (entry.get(k) for k in ("label", "note", "url"))
        if not (isinstance(label, str) and label.strip()):
            continue
        if not (isinstance(url, str) and url.startswith(HTTPS_PREFIX)):
            continue
        refs.append((label.strip(), note.strip() if isinstance(note, str) else "", url))
    return refs


def _link_item(label: str, note: str, url: str) -> str:
    anchor = (
        f'<a href="{escape(url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{escape(label)}</a>'
    )
    return f"<li>{anchor}{' — ' + escape(note) if note else ''}</li>"


def links_html() -> str:
    """References and links from ``references.json``; n/a when there is no valid entry."""
    refs = _references()
    if not refs:
        return section(LINKS_TITLE, "", small_note(NA))
    return section(
        LINKS_TITLE, "", "<ul>" + "".join(_link_item(*r) for r in refs) + "</ul>"
    )


def limitations_html(originals: list[str]) -> str:
    """Translated limitations; the original (English) text is a nested collapsed block."""
    shown = []
    for text in originals:
        line, translated = translate(text)
        shown.append(line if translated else omit_record_refs(line) + UNTRANSLATED)
    nested = details(ORIGINAL_TITLE, _list([omit_record_refs(t) for t in originals]))
    return details(LIMITS_TITLE, _list(shown) + nested)


def sources_html() -> str:
    return section(SOURCES_TITLE, body_html=_list(list(SOURCES)))
