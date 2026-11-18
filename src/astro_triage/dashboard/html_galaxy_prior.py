"""Galáxias page, 'Censo Silencioso': one verdict line and five quiet blocks, no dot field.

Pure: no Streamlit. Every number comes from ``data_galaxy_prior`` (None shows as n/a); every
text that comes from data goes through ``escape``. The sentences below carry no digits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Final

from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import compare_text as CT
from astro_triage.dashboard import fmt
from astro_triage.dashboard import html_dl as HD
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard.data_dl import ClassCell
from astro_triage.dashboard.data_galaxy_prior import (
    NETWORKS,
    ErrorRate,
    VerdictKey,
)
from astro_triage.dashboard.kit_types import NA_WORD, BarRow, Value, Verdict
from astro_triage.dashboard.kit_types import Interval as KitInterval
from astro_triage.dashboard.limits_pt import translate
from astro_triage.dashboard.ui_state import RULE_LABEL, Rule, UiState

CROSSMATCH_ADR: Final = 4
VERDICT_SCOPE: Final = "Vale para a regra"
VERDICT_TEXT: Final[Mapping[VerdictKey, str]] = {
    VerdictKey.DEPENDS_WIDE: (
        "A pureza de galáxia depende do prior assumido; "
        "a comparação com as redes tem intervalos largos."
    ),
    VerdictKey.DEPENDS: "A pureza de galáxia depende do prior assumido.",
    VerdictKey.WIDE: (
        "Sem diferença detectável entre P1 e P2; "
        "a comparação com as redes tem intervalos largos."
    ),
    VerdictKey.NO_DIFFERENCE: "Sem diferença detectável entre P1 e P2.",
    VerdictKey.UNAVAILABLE: hs.NA,
}
NETWORK_NAME: Final[Mapping[str, str]] = {
    "mlp_full": "MLP",
    "ftt_full": "FT-Transformer",
}
PURITY: Final = "Pureza"
COMPLETENESS: Final = "Completude"
OBJECTS: Final = "galáxias"
PURITY_DEF: Final = (
    "Entre os objetos que o modelo chama de galáxia, a fração que é galáxia."
)
COMPLETENESS_DEF: Final = (
    "Entre as galáxias do teste, a fração que o modelo chama de galáxia."
)
HERO_TITLE: Final = "Galáxias no prior de quasar assumido"
HERO_SENTENCE: Final = (
    "Das galáxias apontadas, a pureza é a fração certa; das galáxias reais, "
    "a completude é a fração encontrada."
)
DEPEND_TITLE: Final = "Quanto depende do prior"
DEPEND_SENTENCE: Final = (
    "A mesma escala nas duas colunas; o prior parcial fica fora da regra do veredito."
)
DELTA_TITLE: Final = "Redes contra o LightGBM"
DELTA_SENTENCE: Final = (
    "Veredito na classe galáxia, pela mesma regra da aba de comparação; a faixa é a "
    "margem δ: duas vezes o desvio entre sementes do LightGBM, na validação."
)
ERRORS_TITLE: Final = "Onde o modelo erra"
ERRORS_SENTENCE: Final = "Taxa de erro entre as galáxias, com intervalo de Wilson; faixa abaixo do piso é n/a."
CROSSMATCH_TITLE: Final = "Nota sobre o cruzamento"
CROSSMATCH_ORIGIN: Final = "Origem:"
CROSSMATCH_UNMEASURED: Final = "n/a"
NOT_RECOMPUTED: Final = ("não recalculad", "not recomputed")  # translated and original
TECH_DETAILS: Final = "Tabelas por brilho e por distância"


def _t(text: object) -> str:
    return escape(str(text), quote=True)


def verdict_text(key: VerdictKey) -> str:
    return VERDICT_TEXT.get(key, hs.NA)


def section(title: str, sentence: str, body_html: str) -> str:
    """Titled block with a single sentence; ``title`` and ``sentence`` are escaped."""
    return (
        f"<section class='hs-sec'><div class='hs-pt' role='heading' "
        f"aria-level='{hs.CARD_LEVEL}'>"
        f"{_t(title)}</div><p class='hs-sen'>{_t(sentence)}</p>{body_html}</section>"
    )


def _value(r: ErrorRate | None) -> Value:
    if r is None:
        return Value(None, None)
    # Wilson at the edges can differ from the point by float noise; the kit wants point inside.
    lo, hi = min(r.ci_lo, r.rate), max(r.ci_hi, r.rate)
    return Value(r.rate, KitInterval(lo, hi))


def purity_card(rows: Sequence[BarRow]) -> str:
    return hs.number_card(PURITY, _selected(rows), PURITY_DEF)


def completeness_card(rows: Sequence[BarRow]) -> str:
    return hs.number_card(COMPLETENESS, _selected(rows), COMPLETENESS_DEF)


def _selected(rows: Sequence[BarRow]) -> Value:
    return next((r.value for r in rows if r.selected), Value(None, None))


def cards_html(purity: Sequence[BarRow], completeness: Sequence[BarRow]) -> str:
    cards = hs.cards([purity_card(purity), completeness_card(completeness)])
    return section(HERO_TITLE, HERO_SENTENCE, cards)


def dependence_html(purity: Sequence[BarRow], completeness: Sequence[BarRow]) -> str:
    body = hs.pair(
        hs.hbar_group(PURITY, purity), hs.hbar_group(COMPLETENESS, completeness)
    )
    return section(DEPEND_TITLE, DEPEND_SENTENCE, body)


def verdict_cards_html(
    cells: Mapping[str, ClassCell | None],
    error: str | None = None,
    fixed_point: str | None = None,
) -> str:
    """Two GALAXY verdict cards, one per network, built by the Comparação rule (one source).

    ``fixed_point`` labels the block with the protocol point (same text as the Comparação).
    """
    items = []
    for net in NETWORKS:
        title = f"{NETWORK_NAME.get(net, net)} × {CT.REFERENCE_NAME}"
        cell = cells.get(net)
        if cell is None:
            card = hs.verdict_card(title, Verdict(NA_WORD, None, None, None))
        else:
            grid = CM.grid_cell(NETWORK_NAME.get(net, net), cell)
            warn = hs.method_note(CT.POWER_PHRASE) if grid.underpowered else ""
            card = hs.verdict_card(title, grid.verdict) + warn
        items.append(f"<div>{card}</div>")
    body = (hs.method_note(fixed_point) if fixed_point else "") + hs.pair(*items)
    if error:
        body += hs.method_note(error)
    elif all(cells.get(n) is None for n in NETWORKS):
        body += hs.method_note(HD.UNAVAILABLE_NOTE)
    return section(DELTA_TITLE, DELTA_SENTENCE, body)


def errors_html(rows: Sequence[tuple[str, ErrorRate | None]]) -> str:
    bars = [
        BarRow(label, _value(r), note=f"{fmt.pt_int(r.n)} {OBJECTS}" if r else None)
        for label, r in rows
    ]
    return section(ERRORS_TITLE, ERRORS_SENTENCE, hs.hbar_group(ERRORS_TITLE, bars))


def crossmatch_html(note: str | None, release: str) -> str:
    """Small note with the origin of the rates; the text is the artifact's own.

    The "n/a" line for ``release`` appears only when the text itself says the rates were
    not recomputed for it; otherwise the translated text stands alone.
    """
    text = translate(note)[0] if note else hs.NA
    origin = f"{CROSSMATCH_ORIGIN} ADR {CROSSMATCH_ADR:04d}"
    unmeasured = any(m in text for m in NOT_RECOMPUTED)
    other = (
        f"<p class='hs-small'>{_t(f'{release}: {CROSSMATCH_UNMEASURED}')}</p>"
        if unmeasured
        else ""
    )
    return (
        f"<section class='hs-sec'><div class='hs-pt' role='heading' "
        f"aria-level='{hs.CARD_LEVEL}'>{_t(CROSSMATCH_TITLE)}</div>"
        f"{hs.method_note(f'{origin}. {text}')}{other}</section>"
    )


def page_html(
    release: str,
    state: UiState,
    verdict: VerdictKey,
    rows: tuple[Sequence[BarRow], Sequence[BarRow]],
    cells: Mapping[str, ClassCell | None],
    error_rows: Sequence[tuple[str, ErrorRate | None]],
    note: str | None,
    dl_error: str | None = None,
    fixed_point: str | None = None,
) -> str:
    purity, completeness = rows
    sentence = verdict_text(verdict)
    if verdict is not VerdictKey.UNAVAILABLE:
        sentence = f"{sentence} {VERDICT_SCOPE} {RULE_LABEL[Rule.MATCHED]}."
    body = (
        hs.verdict_oneline(sentence)
        + cards_html(purity, completeness)
        + dependence_html(purity, completeness)
        + verdict_cards_html(cells, dl_error, fixed_point)
        + errors_html(error_rows)
        + crossmatch_html(note, release)
    )
    return hs.lang_root(body)
