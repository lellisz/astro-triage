"""HTML of the Visão geral top: hero with step cards, the three numbers and the training box.

Pure builders (no streamlit): each takes a ``PageData`` and returns one single-line HTML string.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from html import escape

from astro_triage.dashboard import fmt, html_kit
from astro_triage.dashboard.data_headline import (
    PAIR_NAMES,
    PARTIAL_PRIORS,
    SHIPPED_PRIOR,
    SKY_PRIORS,
)
from astro_triage.dashboard.data_overview_page import PageData
from astro_triage.dashboard.html_system import FIELD_COLS, FIELD_N, NA, ONE, PERCENT
from astro_triage.dashboard.html_system import dot_field as _dot_field
from astro_triage.dashboard.html_system import prior_ruler as _prior_ruler

TITLE = "Estrela, galáxia ou quasar? Ensinamos um modelo a decidir."
_CHIP_ACCENT = "ao-chip-a"
_STEPS = (
    (
        "O problema",
        "",
        "Confirmar o tipo de cada objeto é caro",
        (
            "Só dá para ter certeza com um espectro (a luz do objeto separada por cor). "
            "O modelo tenta adivinhar o tipo só com as medidas que o Gaia já tem."
        ),
    ),
    (
        "Já medido",
        _CHIP_ACCENT,
        "Um modelo de árvores (LightGBM)",
        "Treinado e avaliado em objetos que ele nunca viu. Os resultados estão abaixo.",
    ),
)


def _e(text: object) -> str:
    return escape(str(text), quote=True)


def _step(index: int, chip: str, style: str, title: str, body: str) -> str:
    return (
        "<div class='at-step'>"
        f"<span class='at-chip {style}'>{index} · {_e(chip)}</span>"
        f"<div class='ao-step-t'>{_e(title)}</div><div class='ao-step-b'>{_e(body)}</div></div>"
    )


def hero_html(d: PageData) -> str:
    """Title, supporting sentence (only when the pair total exists) and the step cards."""
    lead = ""
    if d.n_pairs is not None:
        lead = (
            f"<p class='ao-lead'>Usamos {_e(fmt.pt_int(d.n_pairs))} objetos do telescópio Gaia, "
            "conferidos contra os espectros do SDSS, que dizem o tipo verdadeiro de cada um.</p>"
        )
    steps = "".join(_step(i, *s) for i, s in enumerate(_STEPS, start=1))
    return (
        f"<div class='ao-hero'><h1 class='ao-h1'>{_e(TITLE)}</h1>{lead}"
        f"<div class='ao-steps'>{steps}</div></div>"
    )


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


def _num(value: str, caption: str, sub: str, tech_line: str, accent: bool) -> str:
    cls = "at-big at-accent" if accent else "at-big"
    sub_html = f"<div class='at-cap'>{_e(sub)}</div>" if sub else ""
    tech_html = f"<div class='at-cap ao-sub'>{_e(tech_line)}</div>" if tech_line else ""
    return (
        f"<div><div class='{cls}'>{_e(value)}</div>"
        f"<div class='at-stat-cap'>{_e(caption)}</div>{sub_html}{tech_html}</div>"
    )


_TITLES = ("O resultado", "O resultado em dois números", "O resultado em três números")


def results_title(n_cells: int) -> str:
    """Section title naming only the count of cells actually shown (neutral beyond the list)."""
    return _TITLES[n_cells - 1] if 0 < n_cells <= len(_TITLES) else _TITLES[0]


def numbers_html(d: PageData, tech: bool) -> str:
    """Section 'O resultado'; ``tech`` adds the technical line under each cell."""
    cells = [
        _num(
            fmt.pt_pct(d.accuracy, 2),
            "dos objetos classificados corretamente",
            "",
            f"{fmt.pt_int(d.n_errors)} erros em {fmt.pt_int(d.n_test)} objetos de teste"
            if tech
            else "",
            False,
        )
    ]
    s, m = d.shipped, d.matched
    if s:
        line = (
            f"pureza {fmt.pt_pct_interval(s['purity'], s['purity_lo'], s['purity_hi'])} · "
            f"completude {fmt.pt_pct(s['completeness'])} "
            f"({fmt.pt_int(d.qso_hits)} de {fmt.pt_int(d.qso_total)} quasares)"
        )
        cells.append(
            _num(
                fmt.pt_pct(s["purity"]),
                f"Quando diz “quasar”, {fmt.pt_pct(s['purity'])} são quasares de verdade",
                f"mas acha {fmt.pt_pct(s['completeness'])} dos quasares que existem",
                line if tech else "",
                True,
            )
        )
    if m:
        line = (
            f"pureza {fmt.pt_pct_interval(m['purity'], m['purity_lo'], m['purity_hi'])} "
            f"com o prior {d.prior} · completude ≈{fmt.pt_pct(m['completeness'], 0)}"
        )
        cells.append(
            _num(
                fmt.pt_pct(m["purity"]),
                f"Se quasares forem tratados como raros, a pureza passa a {fmt.pt_pct(m['purity'])}",
                f"mas acha só ≈{fmt.pt_pct(m['completeness'], 0)} dos quasares",
                line if tech else "",
                False,
            )
        )
    head = html_kit.section(
        results_title(len(cells)),
        "O mesmo modelo, avaliado de formas diferentes.",
        "cada número responde a uma pergunta diferente: acertou o tipo, acertou quando "
        "apontou quasar, e quantos quasares achou.",
    )
    return f"{head}<div class='ao-nums'>{''.join(cells)}</div>"


# --- "Régua do prior" screen -------------------------------------------------------------
PRIOR_DIGITS = 3
ACCURACY_DIGITS = 2
P2_CITATION = "Delchambre+2023"
_ADR_SPLIT = 3
_ADR_WIDTH = 4


def _ok(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _pct(x: object, digits: int = 1) -> str:
    return fmt.pt_pct(float(x), digits) if _ok(x) else NA  # type: ignore[arg-type]


def _ci_text(lo: object, hi: object) -> str:
    if not (_ok(lo) and _ok(hi)):
        return f"{fmt.pt_ci_label()}: {NA}"
    return f"{fmt.pt_ci_label()}: {_pct(lo)} a {_pct(hi)}"


def _place(x: float) -> float:
    return max(0.0, min(PERCENT, x * PERCENT))


def _bar(point: object, lo: object, hi: object) -> str:
    if not (_ok(point) and _ok(lo) and _ok(hi)):
        return ""
    left = _place(float(lo))  # type: ignore[arg-type]
    width = _place(float(hi)) - left  # type: ignore[arg-type]
    return (
        "<div class='ao-r-bar'>"
        f"<div class='ao-r-ci' style='left:{left:.2f}%;width:{width:.2f}%'><b></b></div>"
        f"<div class='ao-r-tick' style='left:{_place(float(point)):.2f}%'></div></div>"  # type: ignore[arg-type]
    )


def _metric(label: str, caption: str, point: object, lo: object, hi: object) -> str:
    return (
        f"<div class='ao-r-m'><div class='ao-r-k'>{_e(label)}</div>"
        f"<div class='ao-r-num'>{_e(_pct(point))}</div>"
        f"<div class='ao-r-cap'>{_e(caption)}</div>"
        f"<div class='ao-r-ciline'>{_e(_ci_text(lo, hi))}</div>{_bar(point, lo, hi)}</div>"
    )


def adr_split_ref() -> str:
    return f"ADR {str(_ADR_SPLIT).zfill(_ADR_WIDTH)}"


def partial_note(prior: str) -> str:
    return f"{prior} vale só para parte do céu (bandas B e C; {adr_split_ref()})"


def inscription_html() -> str:
    return (
        f"<p class='ao-r-insc'>De cada {_e(fmt.pt_int(FIELD_N))} objetos que o modelo "
        "chama de quasar, quantos são?</p>"
    )


def state_html(title: str, sub: str, prior: str, state: Mapping | None) -> str:
    """One state: purity and completeness with the same weight, their CIs and the dot field."""
    s = state or {}
    legend = (
        f"{fmt.pt_int(ONE)} ponto ≈ {fmt.pt_pct(ONE / FIELD_N, 1)} da seleção · "
        "composição esperada sob o prior, não contagem de objetos"
    )
    warn = (
        f"<div class='ao-r-warn' role='note'>{_e(partial_note(prior))}</div>"
        if prior in PARTIAL_PRIORS
        else ""
    )
    return (
        f"<section class='ao-r-st'><div class='ao-r-h'>{_e(title)}</div>"
        f"<div class='ao-r-sub'>{_e(sub)}</div>"
        f"<div class='ao-r-prior'>prior de quasar assumido: "
        f"{_e(_pct(s.get('qso_prior'), PRIOR_DIGITS))}</div>{warn}"
        "<div class='ao-r-pair'>"
        + _metric(
            "Pureza de quasar",
            "dos objetos chamados de quasar, quantos são quasares",
            s.get("purity"),
            s.get("purity_lo"),
            s.get("purity_hi"),
        )
        + _metric(
            "Completude de quasar",
            "dos quasares de verdade, quantos o modelo acha",
            s.get("completeness"),
            s.get("completeness_lo"),
            s.get("completeness_hi"),
        )
        + "</div>"
        f"<div class='ao-r-field'>{_dot_field(s.get('purity'), FIELD_N, FIELD_COLS)}"
        f"<div class='ao-r-cap'>{_e(legend)}</div></div></section>"
    )


def states_html(
    shipped: Mapping | None, states: Mapping[str, Mapping], selected: str
) -> str:
    """The two states side by side: as shipped (fixed prior) and adjusted to ``selected``."""
    left = state_html(
        "Como entregue",
        f"prior {SHIPPED_PRIOR} fixo · probabilidades do treino",
        SHIPPED_PRIOR,
        shipped,
    )
    right = state_html(
        "Ajustada ao prior",
        f"prior {selected} selecionado · ajuste da {adr_split_ref()}",
        selected,
        states.get(selected),
    )
    return f"<div class='ao-r-states'>{left}{right}</div>"


def ruler_html(states: Mapping[str, Mapping], selected: str) -> str:
    """Prior ruler (P1, P2, P3 on a log scale) with the explanation and the partial-prior notes."""
    priors = {p: (states.get(p) or {}).get("qso_prior") for p in SKY_PRIORS}
    callouts = {}
    shipped_q = priors.get(SHIPPED_PRIOR)
    if _ok(shipped_q):
        callouts[SHIPPED_PRIOR] = f"{P2_CITATION} · {_pct(shipped_q, PRIOR_DIGITS)}"
    notes = "".join(
        f"<p class='ao-r-cap'>{_e(partial_note(p))}</p>" for p in SKY_PRIORS if p in PARTIAL_PRIORS
    )
    head = html_kit.section(
        "Régua do prior",
        "O prior é a fração de quasares assumida (escala logarítmica). "
        f"“Como entregue” fica fixo em {SHIPPED_PRIOR}.",
    )
    return f"{head}{_prior_ruler(priors, selected, None, None, callouts)}{notes}"


def accuracy_html(defs: Mapping | None) -> str:
    """Overall accuracy: two definitions stacked, each labelled with its own source."""
    a = defs or {}
    n_seeds = a.get("n_seeds")
    seeds_n = fmt.pt_int(n_seeds) if _ok(n_seeds) else NA
    regime = "regime balanceado" if a.get("balanced") else f"regime: {NA}"
    span = (
        f"mínimo a máximo: {_pct(a.get('seeds_min'), ACCURACY_DIGITS)} "
        f"a {_pct(a.get('seeds_max'), ACCURACY_DIGITS)}"
    )
    head = html_kit.section(
        "Acerto geral",
        "Duas definições, lidas de arquivos diferentes; não se comparam entre si.",
    )
    return (
        f"{head}<div class='ao-r-acc'>"
        f"<div class='ao-r-def'><div class='ao-r-accv'>"
        f"{_e(_pct(a.get('shipped'), ACCURACY_DIGITS))}</div>"
        "<div class='ao-r-cap'>acurácia como entregue: soma da diagonal da matriz de "
        "confusão, um modelo, sem pesos</div></div>"
        f"<div class='ao-r-def'><div class='ao-r-accv'>"
        f"{_e(_pct(a.get('seeds_mean'), ACCURACY_DIGITS))}</div>"
        f"<div class='ao-r-cap'>média de {_e(seeds_n)} sementes, {_e(regime)}</div>"
        f"<div class='ao-r-cap'>{_e(span)}</div></div>"
        "<p class='ao-r-cap'>acerto alto não garante pureza alta de quasar</p></div>"
    )
