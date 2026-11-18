"""HTML of the Visão geral top: hero with step cards, the three numbers and the training box.

Pure builders (no streamlit): each takes a ``PageData`` and returns one single-line HTML string.
"""

from __future__ import annotations

from html import escape

from astro_triage.dashboard import fmt, html_kit
from astro_triage.dashboard.data_headline import (
    PAIR_NAMES,
)
from astro_triage.dashboard.data_overview_page import PageData


def _e(text: object) -> str:
    return escape(str(text), quote=True)


def data_box_html(d: PageData) -> str:
    """Technical box 'Dados e treino': every item comes from the artifacts."""
    items: list[str] = []
    if d.n_pairs is not None:
        items.append(f"{fmt.pt_int(d.n_pairs)} pares Gaia–SDSS casados.")
    items.append(
        f"Teste: {fmt.pt_int(d.n_test)} objetos em {fmt.pt_int(d.n_pixels)} regiões do céu. "
        "A divisão entre treino, validação e teste é por região, para que vizinhos "
        "não vazem de um lado ao outro."
    )
    if d.class_mix:
        mix = ", ".join(
            f"{PAIR_NAMES[c]} {fmt.pt_pct(s, 0)}" for c, s in d.class_mix.items()
        )
        items.append(f"Mistura do teste: {mix}. Não há corte de magnitude.")
    items.append(
        "Modelo: LightGBM com calibração. Referência: o classificador do próprio Gaia (DSC)."
    )
    inner = "<ul>" + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>"
    return html_kit.box("Dados e treino", inner)


_ADR_SPLIT = 3
_ADR_WIDTH = 4


def adr_split_ref() -> str:
    return f"ADR {str(_ADR_SPLIT).zfill(_ADR_WIDTH)}"


def partial_note(prior: str) -> str:
    return f"{prior} vale só para parte do céu (bandas B e C; {adr_split_ref()})"
