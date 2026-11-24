"""Comparação page blocks as HTML (design kit, 'prior ruler' layout). Pure: no streamlit.

Every figure is formatted at run time from ``results_dl.json`` (through ``data_dl``) or from the
constants of ``dl_compare``; no number is written here. Text is escaped by the kit. The texts
of the verdicts live in the state tables below, never in loose strings at the call sites.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Final

from astro_triage.dashboard import fmt
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard.data_dl import (
    ClassVerdict,
    PairCell,
    parse_class_verdict,
)
from astro_triage.dashboard.data_headline import PAIR_NAMES
from astro_triage.dashboard.html_kit import box, section, table
from astro_triage.dashboard.limits_pt import omit_record_refs

TITLE: Final = "Comparação: LightGBM × deep learning"
UNAVAILABLE: Final = "não disponível"
UNAVAILABLE_NOTE: Final = (
    "Esta versão dos dados não tem a avaliação do aprendizado profundo. Escolha outra "
    "versão em Ajustes."
)
NA: Final = hs.NA
DESCRIPTIVE_TITLE: Final = "Descritivo: conjuntos de sementes e frentes de indicadores"
LIMITS_TITLE: Final = (
    "Limitações registradas no results_dl (texto original em inglês, sem as citações internas)"
)
PROTOCOL: Final = "protocolo da comparação"
_STATUS_PT: Final = {"Proposed": "em proposta", "Accepted": "aceito"}
_ACCEPTED: Final = "Accepted"
_NET: Final = {"mlp": "MLP", "ftt": "FTT"}
_FAMILY: Final = {"lgbm": "LightGBM", "mlp": "MLP", "ftt": "FT-Transformer"}
_FEATURES: Final = {"full": "completo", "noind": "sem indicadores"}
_ENSEMBLE: Final = "ens_"
_ROLE: Final = {
    "headline": "destaque",
    "sensitivity": "sensibilidade",
    "descriptive": "descritivo",
}
_FRONT: Final = "front"
_CLASS_STATE: Final = {
    ClassVerdict.WIN: "vitória na classe",
    ClassVerdict.TIE: "empate na classe",
    ClassVerdict.WORSE: "pior na classe (descritivo; não decide o par)",
    ClassVerdict.NA: "n/a: célula não relatável; conta como empate",
}


def system_label(name: str) -> str:
    """``ens_mlp_full`` -> 'conjunto MLP completo'; unknown parts fall back to 'n/d'."""
    ensemble = name.startswith(_ENSEMBLE)
    family, _, feats = name.removeprefix(_ENSEMBLE).partition("_")
    text = f"{_FAMILY.get(family, 'n/d')} {_FEATURES.get(feats, 'n/d')}"
    return f"conjunto {text}" if ensemble else text


def pair_label(pair: str) -> str:
    first, _, second = pair.partition("-")
    return f"{system_label(first)} × {system_label(second)}"


def net_name(pair: str) -> str:
    """Short name of the first system of a pair ('MLP', 'FTT')."""
    family = pair.partition("-")[0].partition("_")[0]
    return _NET.get(family, family.upper())


def role_label(role: str) -> str:
    if role.startswith(_FRONT):
        return f"frente {role.removeprefix(_FRONT)}"
    return _ROLE.get(role, "n/d")


def class_label(cls: str) -> str:
    return PAIR_NAMES.get(cls, cls).capitalize()


def class_verdict(key: str | None) -> str:
    """Text of a class verdict token; ``None`` is a cell without verdict, an unknown token raises."""
    if key is None:
        return "descritivo"
    return _CLASS_STATE[parse_class_verdict(key, "veredito de classe")]


def signed(x: float | None) -> str:
    """Four decimals with an explicit sign (U+2212 for negatives); NA for a missing value."""
    if x is None:
        return NA
    if round(x, 4) == 0:
        return fmt.pt_num(0.0, 4)
    return fmt.pt_num(x, 4, True).replace("-", "−")


def plain(x: float | None) -> str:
    return NA if x is None else fmt.pt_num(x, 4)


def interval(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return NA
    return f"[{signed(lo)}; {signed(hi)}]"


def c_star(level: str) -> str:
    try:
        return fmt.pt_num(float(level), 2)
    except ValueError:
        return NA


def level_text(prior: str, level: str) -> str:
    return f"{prior}, C* = {c_star(level)}"


def _p(text: str, cls: str = "") -> str:
    attr = f" class='{cls}'" if cls else ""
    return f"<p{attr}>{escape(text)}</p>"


def _scroll(table_html: str) -> str:
    return f"<div class='at-ov'>{table_html}</div>"


def unavailable_html() -> str:
    return section(TITLE, f"{UNAVAILABLE_NOTE} ({UNAVAILABLE})")


def error_html(message: str) -> str:
    body = _p(message, "err") + _p("A aba não mostra números que não consegue ler.")
    return hs.lang_root(box("Dados da comparação ilegíveis", body))


def reasons_text(status: str | None, freeze: bool, below: int, has_limits: bool) -> str:
    """Reason sentence derived from the protocol record header and the artifacts read."""
    word = _STATUS_PT.get(status or "")
    if word is None:
        protocol = f"{PROTOCOL}: {NA}"
    else:
        extra = [] if status == _ACCEPTED else ["aceitação aberta"]
        extra.append("o freeze existe" if freeze else f"freeze {NA}")
        protocol = f"{PROTOCOL} {word} ({'; '.join(extra)})"
    reasons = [protocol]
    if below > 0:
        reasons.append("células do destaque abaixo do piso de relato")
    reasons.append(
        "limitações do results_dl listadas abaixo"
        if has_limits
        else f"limitações do results_dl: {NA}"
    )
    return "Razões: " + "; ".join(reasons) + "."


def reasons_html(
    status: str | None,
    freeze: bool,
    floor_cells: int,
    limitations: bool,
) -> str:
    """Reason line as HTML. The PROVISIONAL seal itself is printed once per tab by the shell."""
    return _p(reasons_text(status, freeze, floor_cells, limitations), "at-cap")


def _cell_rows(cells: Sequence[PairCell]) -> list[list[str]]:
    return [
        [
            pair_label(cell.pair),
            role_label(cell.role),
            class_label(cls),
            signed(c.d),
            interval(c.lo, c.hi),
            plain(c.delta),
        ]
        for cell in cells
        for cls, c in cell.classes.items()
    ]


def descriptive_html(cells: Sequence[PairCell], prior: str, level: str) -> str:
    title = f"Descritivo: {level_text(prior, level)}"
    note = (
        "D é a diferença de pureza, primeiro sistema menos segundo. Conjuntos e frentes não "
        "têm veredito e não entram na decisão."
    )
    if not cells:
        return hs.lang_root(section(title, UNAVAILABLE))
    headers = ("Par", "Papel", "Classe", "D", fmt.pt_ci_label(), "δ")
    return hs.lang_root(
        section(title, note) + _scroll(table(headers, _cell_rows(cells)))
    )


def limitations_html(items: Sequence[str] | None) -> str:
    if not items:
        return hs.lang_root(_p(NA))
    return hs.lang_root(
        "<ul>" + "".join(f"<li>{escape(omit_record_refs(t))}</li>" for t in items)
        + "</ul>"
    )
