"""Comparação page blocks as HTML (design kit, 'prior ruler' layout). Pure: no streamlit.

Every figure is formatted at run time from ``results_dl.json`` (through ``data_dl``) or from the
constants of ``dl_compare``; no number is written here. Text is escaped by the kit. The texts
of the verdicts live in the state tables below, never in loose strings at the call sites.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import TYPE_CHECKING, Final

from astro_triage.dashboard import fmt
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard.constants import ADR_NUMBER
from astro_triage.dashboard.data_dl import (
    ClassCell,
    ClassVerdict,
    DlDataError,
    Headline,
    PairCell,
    PairVerdict,
    parse_class_verdict,
)
from astro_triage.dashboard.data_headline import PAIR_NAMES
from astro_triage.dashboard.html_kit import box, section, table

if TYPE_CHECKING:
    from astro_triage.dashboard.views_dl import RuleFacts

TITLE: Final = "Comparação: LightGBM × deep learning"
UNAVAILABLE: Final = "não disponível"
UNAVAILABLE_NOTE: Final = (
    "Esta versão dos dados não tem a avaliação do aprendizado profundo. Escolha outra "
    "versão na barra lateral."
)
NA: Final = hs.NA
LEAD: Final = (
    "Duas redes neurais (MLP e FT-Transformer, FTT) foram postas contra o LightGBM, no "
    "teste, com uma regra escrita antes de olhar o resultado."
)
LIGHTGBM: Final = "LightGBM"
DESCRIPTIVE_TITLE: Final = "Descritivo: conjuntos de sementes e frentes de indicadores"
LIMITS_TITLE: Final = "Limitações registradas no results_dl (texto original, inglês)"
INCONSISTENT: Final = (
    "inconsistência no artefato: o veredito do par não segue os vereditos das classes."
)
NO_POWER: Final = "sem poder para decidir; isso é falta de poder, não indica que os sistemas sejam iguais"
WITHIN_DELTA: Final = "diferença dentro de ±δ"
ONE_CLASS_NOT_ENOUGH: Final = (
    "Vencer só numa classe não basta: o par exige as duas classes."
)
RULER_LEGEND: Final = (
    "Cada régua tem a própria escala, simétrica em torno de zero. A faixa clara é a margem "
    "±δ, a barra é o intervalo de confiança e o ponto é D, a diferença de pureza (rede "
    "menos LightGBM). O eixo marca só −δ, zero e +δ."
)
ABS_NOTE: Final = (
    "Barras a partir de zero: a mediana das sementes e, na linha fina, o mínimo e o máximo. "
    "A pureza de quasar é baixa em todos os sistemas porque o prior de quasar do ponto de "
    "operação é pequeno; a comparação entre eles é a diferença D, não o nível."
)
SENSITIVITY_LEAD: Final = (
    "Só os vereditos, em outros priors e outra completude. Nunca substituem o destaque."
)
_STATUS_PT: Final = {"Proposed": "Proposta", "Accepted": "Aceita"}
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
# State tables: (long text for cards and reading, short text for the sensitivity table).
_CLASS_STATE: Final = {
    ClassVerdict.WIN: ("vitória na classe", "vitória"),
    ClassVerdict.TIE: ("empate na classe", "empate"),
    ClassVerdict.WORSE: (
        "pior na classe (descritivo; não decide o par)",
        "pior (descritivo)",
    ),
    ClassVerdict.NA: ("n/a: célula não relatável; conta como empate", NA),
}
_PAIR_STATE: Final = {
    PairVerdict.TIE: ("Empate: fica o LightGBM.", "empate"),
    PairVerdict.WIN: ("{net} vence o par: vitória nas duas classes.", "vitória do par"),
}
_PRIMARY_PURITY: Final = "lgbm_full"
L5_LEVEL: Final = (
    5  # sky-block level of the alternative interval ('ci_l5' in the artifact)
)


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
    return _CLASS_STATE[parse_class_verdict(key, "veredito de classe")][0]


def class_text(verdict: ClassVerdict | None) -> str:
    return "descritivo" if verdict is None else _CLASS_STATE[verdict][0]


def pair_text(cell: PairCell) -> str:
    if cell.verdict is None:
        return UNAVAILABLE
    return _PAIR_STATE[cell.verdict][0].format(net=net_name(cell.pair))


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


def numbers_text(c: ClassCell) -> str:
    """D, interval and delta spelled out beside a mark."""
    return (
        f"D = {signed(c.d)} · {fmt.pt_ci_label()}: {interval(c.lo, c.hi)} · "
        f"δ = {plain(c.delta)}"
    )


def honesty(c: ClassCell) -> str:
    """Note under a tied class: no power (interval wider than the band) or inside the band."""
    if c.verdict is not ClassVerdict.TIE or None in (c.lo, c.hi, c.delta):
        return ""
    if c.hi - c.lo > 2 * c.delta:
        return NO_POWER
    if c.lo >= -c.delta and c.hi <= c.delta:
        return WITHIN_DELTA
    return ""


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


def seal_html(
    release: str,
    status: str | None,
    freeze: bool,
    floor_cells: int,
    limitations: bool,
) -> str:
    """Seal plus a reason line derived from the ADR header and the artifacts read."""
    word = _STATUS_PT.get(status or "")
    if word is None:
        adr = f"ADR {ADR_NUMBER:04d}: {NA}"
    else:
        extra = [] if status == _ACCEPTED else ["aceitação aberta"]
        extra.append("o freeze existe" if freeze else f"freeze {NA}")
        adr = f"ADR {ADR_NUMBER:04d} {word} ({'; '.join(extra)})"
    reasons = [adr]
    if floor_cells > 0:
        reasons.append("células do destaque abaixo do piso de relato")
    reasons.append(
        "limitações do results_dl listadas abaixo"
        if limitations
        else f"limitações do results_dl: {NA}"
    )
    return hs.seal(release) + _p("Razões: " + "; ".join(reasons) + ".", "at-cap")


def verdict_text(cells: Sequence[PairCell]) -> str:
    """One line derived from the pair and class verdicts, in the kit vocabulary."""
    items = [
        {
            "family": net_name(c.pair),
            "pair_verdict": c.verdict.value if c.verdict else None,
            "classes": {
                class_label(k).lower(): v.verdict.value if v.verdict else None
                for k, v in c.classes.items()
            },
        }
        for c in cells
    ]
    return hs.verdict_line(items)


def card_html(cell: PairCell) -> str:
    rows = [
        {
            "name": class_label(cls),
            "verdict": class_text(c.verdict),
            "numbers": numbers_text(c),
            "note": honesty(c),
        }
        for cls, c in cell.classes.items()
    ]
    warn = INCONSISTENT if cell.inconsistent else ""
    return hs.pair_card(
        f"{net_name(cell.pair)} × {LIGHTGBM}", pair_text(cell), rows, warn
    )


def ruler_panels(head: Headline) -> list[dict]:
    """One panel per class with both networks on the same ruler; delta must agree."""
    panels = []
    classes = next(iter(head.cells)).classes
    for cls in classes:
        rows = [
            {
                "name": net_name(cell.pair),
                "d": cell.classes[cls].d,
                "lo": cell.classes[cls].lo,
                "hi": cell.classes[cls].hi,
                "detail": numbers_text(cell.classes[cls]),
            }
            for cell in head.cells
        ]
        deltas = [
            cell.classes[cls].delta
            for cell in head.cells
            if cell.classes[cls].delta is not None
        ]
        if deltas and any(abs(x - deltas[0]) > abs(deltas[0]) * 1e-9 for x in deltas):
            raise DlDataError(f"δ difere entre os pares na classe {cls}")
        panels.append(
            {
                "title": class_label(cls),
                "delta": deltas[0] if deltas else None,
                "rows": rows,
            }
        )
    return panels


def reading_html(head: Headline) -> str:
    out = []
    for cls in next(iter(head.cells)).classes:
        clauses = []
        for cell in head.cells:
            c = cell.classes[cls]
            extra = honesty(c)
            text = f"{net_name(cell.pair)}: {class_text(c.verdict)}"
            clauses.append(f"{text} ({extra})" if extra else text)
        out.append(_p(f"{class_label(cls)}. " + "; ".join(clauses) + "."))
    if any(
        cell.verdict is PairVerdict.TIE
        and any(c.verdict is ClassVerdict.WIN for c in cell.classes.values())
        for cell in head.cells
    ):
        out.append(_p(ONE_CLASS_NOT_ENOUGH))
    return "".join(out)


def abs_rows(head: Headline, cls: str) -> list[dict]:
    """Seed range of the absolute purity of every system in the headline pairs, LightGBM first."""
    seen: dict[str, tuple[float, float, float]] = {}
    for cell in head.cells:
        for system, stats in cell.classes[cls].purity.items():
            seen.setdefault(system, stats)
    names = sorted(seen, key=lambda s: s != _PRIMARY_PURITY)
    return [
        {
            "name": LIGHTGBM if s == _PRIMARY_PURITY else net_name(s),
            "low": seen[s][0],
            "median": seen[s][1],
            "high": seen[s][2],
            "detail": (
                f"mediana {fmt.pt_pct(seen[s][1], 2)} · mínimo a máximo das sementes: "
                f"{fmt.pt_pct(seen[s][0], 2)} a {fmt.pt_pct(seen[s][2], 2)}"
            ),
        }
        for s in names
    ]


def _short(c: ClassCell) -> str:
    return NA if c.verdict is None else _CLASS_STATE[c.verdict][1]


def sensitivity_html(cells: Sequence[PairCell]) -> str:
    if not cells:
        return section("Sensibilidade", f"{SENSITIVITY_LEAD} {UNAVAILABLE}.")
    rows = []
    for cell in cells:
        l5 = all(c.ci_l5 is not None for c in cell.classes.values())
        rows.append(
            [
                level_text(cell.prior, cell.level),
                f"{net_name(cell.pair)} × {LIGHTGBM}",
                *(_short(c) for c in cell.classes.values()),
                _PAIR_STATE[cell.verdict][1] if cell.verdict else NA,
                "disponível" if l5 else NA,
            ]
        )
    classes = [class_label(k) for k in cells[0].classes]
    headers = ("Prior e C*", "Par", *classes, "Par vence?", f"IC nível {L5_LEVEL}")
    return section("Sensibilidade", SENSITIVITY_LEAD) + _scroll(table(headers, rows))


def page_html(head: Headline, sensitivity: Sequence[PairCell], seal: str) -> str:
    """Seal, verdict line, pair cards, rulers, reading, absolute purity and sensitivity."""
    level = level_text(head.prior, head.level)
    abs_title = f"Pureza absoluta de {PAIR_NAMES['QSO']}"
    qso = next((c for c in head.cells if "QSO" in c.classes), None)
    abs_block = (
        hs.abs_bars(abs_rows(head, "QSO")) if qso is not None else _p(UNAVAILABLE)
    )
    parts = [
        seal,
        hs.verdict_oneline(verdict_text(head.cells)),
        section("Os dois pares", f"Ponto de destaque: {level}.")
        + hs.cards([card_html(c) for c in head.cells]),
        section("Diferença de pureza por classe", RULER_LEGEND)
        + hs.delta_rulers(ruler_panels(head)),
        section("Leitura por classe", f"Ponto de destaque: {level}.")
        + reading_html(head),
        section(abs_title, ABS_NOTE) + abs_block,
        sensitivity_html(sensitivity),
    ]
    return hs.lang_root("".join(parts))


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
        "<ul>" + "".join(f"<li>{escape(t)}</li>" for t in items) + "</ul>"
    )


def method_lines(rule: RuleFacts, n_boot: int | None) -> list[str]:
    boot = NA if n_boot is None else fmt.pt_int(n_boot)
    return [
        (
            "Vitória na classe: célula relatável, D maior que δ e limite inferior do "
            "intervalo maior que zero. Pior na classe: D menor que −δ e limite superior "
            "menor que zero; é descritivo e não decide o par."
        ),
        (
            "O par vence só com vitória nas duas classes. Célula abaixo do piso de relato "
            "conta como empate. Em empate, fica o LightGBM."
        ),
        "Sem correção de multiplicidade entre os dois pares.",
        (
            f"δ = {fmt.pt_g(rule.multiplier)} desvios-padrão (ddof = {rule.ddof}) das "
            f"{fmt.pt_int(rule.n_seeds)} sementes do LightGBM, medido na validação."
        ),
        (
            f"{fmt.pt_ci_label()} por bootstrap de blocos do céu (n_boot {boot}; bloco = "
            f"pixel nível {rule.block_level} da divisão por "
            f"{fmt.pt_int(rule.block_divisor)}, ao menos {fmt.pt_int(rule.min_blocks)} blocos)."
        ),
        (
            f"Métrica: pureza à completude fixa C* = {fmt.pt_num(rule.primary_completeness, 2)}, "
            f"prior {rule.primary_prior}; sensibilidade em {', '.join(rule.sensitivity_priors)}."
        ),
    ]


def method_html(rule: RuleFacts, n_boot: int | None) -> str:
    items = "".join(f"<li>{escape(t)}</li>" for t in method_lines(rule, n_boot))
    return hs.lang_root(
        box(f"Nota de método (ADR {ADR_NUMBER:04d})", f"<ul>{items}</ul>")
    )
