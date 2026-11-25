"""Colour-neutral variant of the sky 'Como ler' sentence for the monochrome web front.

The Streamlit text (``sky_text.how``) names a colour; the web map is greyscale, so the
export words the same sentence by grey level instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pandas as pd

from astro_triage.dashboard import fmt
from astro_triage.dashboard.kit_types import NA_WORD

KEY_TIE: Final = "no_detectable_difference"
KEY_WIN: Final = "network_wins"
KEY_NA: Final = "unavailable"
QUESTION: Final = (
    "As redes neurais superam o LightGBM ao classificar estrela, galáxia e quasar "
    "nos dados do Gaia DR3?"
)
TITLES: Final[Mapping[str, str]] = {
    KEY_TIE: "Sem diferença detectável",
    KEY_WIN: "Vitória sobre o LightGBM",
    KEY_NA: "Comparação indisponível",
}
SUMMARY_EN: Final = (
    "Star, galaxy and quasar classifier on Gaia DR3 with SDSS spectroscopic labels, "
    "comparing LightGBM with neural networks; every result is provisional."
)
ERROR_ROWS_NOTE: Final = (
    "As faixas listadas são pequenas (veja o n de cada uma); a taxa de todas as "
    "galáxias é calculada sobre o conjunto inteiro, por isso as duas não se contradizem."
)


def how(by_rate: bool, n_hidden: int, floor: int, hidden_on_map: bool) -> str:
    """Same structure as ``sky_text.how``, with the colour reference replaced by grey level."""
    if not by_rate:
        return (
            "quanto mais claro o ponto, mais objetos de teste a região tem. "
            "Anel vazio não aparece neste modo."
        )
    few = f"{fmt.pt_int(n_hidden)} regiões com menos de {fmt.pt_int(floor)} objetos"
    tail = (
        f"As {few} não aparecem no mapa."
        if hidden_on_map
        else f"Anel vazio = {few}, sem taxa confiável."
    )
    return f"quanto mais claro o ponto, maior a taxa de erro da região. {tail}"


def missing_note(by_g: pd.DataFrame) -> str:
    """Same structure as ``missing_note_text``, without the colour key (no web chart highlights a rate)."""
    gone = by_g[by_g["lo"].isna()]
    n = int(gone["n"].sum()) if len(gone) else 0
    if not n:
        return ""
    return (
        f"Faltam {fmt.pt_int(n)} galáxias sem magnitude medida, "
        "que não entram na parte de brilho."
    )


PAIR_NO_DIFF = "sem diferença detectável; fica o LightGBM"


def headline_lead(key: str, nets: list[str], winners: list[str]) -> str:
    """One sentence naming the networks of the pairs; no figure is written here."""
    if key == KEY_WIN:
        return f"Superou o LightGBM nas duas classes avaliadas: {', '.join(winners)}."
    return f"Nenhuma rede ({', '.join(nets)}) superou o LightGBM com esta amostra."


def headline_caveat(power_phrase: str) -> str:
    """Why a missing difference is not a tie; the phrase itself comes from compare_text."""
    return (
        "Com esta amostra não dá para afirmar que as redes igualam o LightGBM: "
        f"o que há é {power_phrase}."
    )


def contrast_text(accuracy: str, purity: str, *, purity_lower: bool = True) -> str:
    """Overall accuracy against QSO purity of the shipped rule; n/a if either is missing.

    ``purity_lower`` (decided by the caller from the two values) picks the connective, so the
    text never says "only" about a purity that is not below the accuracy.
    """
    if NA_WORD in (accuracy, purity):
        return NA_WORD
    tail = "dos objetos que o modelo chama de quasar são quasares na regra como entregue."
    if purity_lower:
        return f"Acerto geral {accuracy}, mas só {purity} {tail}"
    return f"Acerto geral {accuracy}; {purity} {tail}"
