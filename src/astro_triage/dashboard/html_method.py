"""Método page blocks as HTML (design kit). Pure: no streamlit; no number is written here.

The deep-learning box states the pre-registered rule and how it was run. The seed count and
the margin come from ``delta.json`` when it exists; otherwise they are left out. DL numbers
are not written here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from typing import Final

from astro_triage.dashboard import fmt
from astro_triage.dashboard.constants import ADR_NUMBER, FIXED_COMPLETENESS
from astro_triage.dashboard.data_headline import PAIR_NAMES
from astro_triage.dashboard.data_release import CLASSES
from astro_triage.dashboard.html_kit import box, section

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
GLOSSARY: Final = (
    ("Pureza", "dos que o modelo aponta como de uma classe, quantos são dela."),
    ("Completude", "dos objetos de uma classe que existem, quantos o modelo acha."),
    (
        "Espectro",
        "a medida detalhada da luz de um objeto, que revela o tipo dele com certeza.",
    ),
    ("Prior", "proporção das classes que o classificador assume que existe no céu."),
    ("Piso de relato", "tamanho mínimo de uma célula para que seu valor seja publicado."),
    ("IC", "intervalo de confiança: a faixa em que o valor verdadeiro provavelmente está."),
)
_DELTA_CLASSES: Final = ("GALAXY", "QSO")
_FEATURE_SET: Final = "full"


@dataclass(frozen=True)
class DeltaFacts:
    n_seeds: int | None
    multiplier: float | None
    completeness: float
    margins: tuple[tuple[str, float], ...]


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _dig(node: object, *keys: str) -> object:
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def delta_facts(delta: Mapping[str, object] | None) -> DeltaFacts:
    """Seed count, multiplier, C* and per-class margins from delta.json; absent -> None/default."""
    doc = delta or {}
    comp = _finite(doc.get("primary_completeness"))
    n_seeds = _finite(doc.get("n_seeds"))
    margins: list[tuple[str, float]] = []
    prior = doc.get("primary_prior")
    if comp is not None and isinstance(prior, str):
        for cls in _DELTA_CLASSES:
            keys = ("delta", _FEATURE_SET, cls, f"{comp:.2f}", prior, "delta")
            value = _finite(_dig(doc, *keys))
            if value is not None:
                margins.append((PAIR_NAMES[cls], value))
    return DeltaFacts(
        None if n_seeds is None else int(n_seeds),
        _finite(doc.get("delta_sd_multiplier")),
        FIXED_COMPLETENESS if comp is None else comp,
        tuple(margins),
    )


def rule_lines(f: DeltaFacts) -> list[str]:
    mult = "um múltiplo do" if f.multiplier is None else f"{fmt.pt_g(f.multiplier)} × o"
    seeds = "das sementes" if f.n_seeds is None else f"de {fmt.pt_int(f.n_seeds)} sementes"
    lines = [
        (
            "Métrica principal: pureza de quasar e de galáxia com completude fixa em "
            f"{fmt.pt_num(f.completeness, 2)}."
        ),
        f"Margem δ = {mult} desvio-padrão {seeds} do LightGBM, medido na validação.",
    ]
    if f.margins:
        values = "; ".join(f"{name} {fmt.pt_num(v, 3)}" for name, v in f.margins)
        lines.append(f"δ medido na validação: {values}.")
    lines += [
        (
            "Vitória na classe: a diferença de pureza D é maior que δ e o limite inferior "
            "do intervalo de confiança (bootstrap em blocos do céu) fica acima de zero. "
            "O par vence só com vitória nas duas classes. Em caso de empate, fica o LightGBM."
        ),
        (
            "O teste do aprendizado profundo foi aberto uma vez, depois de "
            "congelar a validação. O resultado, sempre provisório, está na aba Comparação."
        ),
        f"O protocolo da ADR {ADR_NUMBER:04d} foi executado como registrado; a ADR segue em rascunho.",
    ]
    return lines


def dl_box(delta: Mapping[str, object] | None) -> str:
    items = "".join(f"<li>{escape(t)}</li>" for t in rule_lines(delta_facts(delta)))
    title = f"Comparação com aprendizado profundo (ADR {ADR_NUMBER:04d}, rascunho)"
    return box(title, f"<ul>{items}</ul>")


def mix_sentence(mix: Mapping[str, float]) -> str:
    parts = ", ".join(f"{PAIR_NAMES[c]} {fmt.pt_pct(mix.get(c, 0.0), 0)}" for c in CLASSES)
    return (
        "A amostra só tem objetos que o SDSS observou e que casam com o Gaia, então a "
        f"mistura do teste ({parts}) não é a do céu."
    )


def lead_html() -> str:
    return section("Método", LEAD)


def simple_html() -> str:
    return lead_html() + f"<p class='at-cap'>{escape(HINT)}</p>"


def sources_line() -> str:
    """'Fontes: ...' with the source names (text before the colon of each entry)."""
    names = " · ".join(line.split(":")[0] for line in SOURCES)
    return f"<p class='at-cap'>{escape('Fontes: ' + names)}</p>"


def glossary_html() -> str:
    return "".join(
        f"<div class='at-gl'><b>{escape(term)}</b><span>{escape(text)}</span></div>"
        for term, text in GLOSSARY
    )
