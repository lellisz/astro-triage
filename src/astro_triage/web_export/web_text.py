"""Colour-neutral variant of the sky 'Como ler' sentence for the monochrome web front.

The Streamlit text (``sky_text.how``) names a colour; the web map is greyscale, so the
export words the same sentence by grey level instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pandas as pd

from astro_triage.dashboard import fmt
from astro_triage.dashboard.html_dl import class_label
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
NETWORKS_ABSENT: Final = (
    "As redes neurais (MLP e FT-Transformer) não aparecem nesta página: desenhar a curva "
    "de calibração e a matriz delas exigiria reler as previsões do conjunto de teste, "
    "que já foi aberto uma vez. Aqui só há o LightGBM."
)
STAR_Z_NOTE: Final = (
    "Estrelas ficam fora do gráfico por redshift: o z espectroscópico de uma estrela é uma "
    "velocidade radial perto de zero, não uma distância."
)


def charts_scope(n_test: int, model_label: str) -> str:
    """Model, fold and rule of every chart; says why these shares differ from Visão geral."""
    return (
        f"Todos os gráficos usam o modelo {model_label} no conjunto de teste "
        f"({fmt.pt_int(n_test)} objetos). A classe prevista é a de maior probabilidade "
        "calibrada; a regra de decisão e o prior das outras páginas não valem aqui. "
        "Por isso estes acertos e erros não são a pureza nem a completude da Visão geral."
    )


def charts_floor_note(floor: int) -> str:
    """Floor rule: 1D charts keep the count, the confusion cells hide it too."""
    return (
        f"Faixas e células com menos de {fmt.pt_int(floor)} objetos não têm valor "
        "confiável e aparecem como n/a. Nos gráficos por faixa o n continua visível; "
        "nas células da matriz de confusão, não."
    )


def missing_g_note(counts: Mapping[str, int]) -> str:
    """Objects without a measured G that the by-magnitude charts leave out; "" if none."""
    shown = [
        f"{class_label(cls).lower()}: {fmt.pt_int(n)}" for cls, n in counts.items() if n
    ]
    if not shown:
        return ""
    return (
        "Objetos sem magnitude medida não entram nos gráficos por brilho ("
        + "; ".join(shown)
        + ")."
    )


def cmd_note(n_missing: int, n_outside: int, floor: int) -> str:
    """Floor rule of the colour-magnitude map and the two totals that are not on it."""
    return (
        f"Células com menos de {fmt.pt_int(floor)} objetos, vazias incluídas, não têm taxa "
        "confiável e aparecem hachuradas, sem valor. Ficam fora do mapa "
        f"{fmt.pt_int(n_missing)} objetos sem magnitude G ou sem cor BP-RP e "
        f"{fmt.pt_int(n_outside)} fora da área desenhada."
    )


def how(by_rate: bool, n_hidden: int, floor: int, hidden_on_map: bool) -> str:
    """Same structure as ``sky_text.how``, with the colour reference replaced by grey level."""
    if not by_rate:
        base = "quanto mais claro o ponto, mais objetos de teste a região tem. "
        if hidden_on_map:
            return base + "Anel vazio não aparece neste modo."
        return (
            base
            + f"Regiões com menos de {fmt.pt_int(floor)} objetos não têm contagem exibida "
            "e aparecem no tom mais fraco. Anel vazio não aparece neste modo."
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
