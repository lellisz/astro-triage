"""HTML of the Visão geral top: hero with step cards, the three numbers and the training box.

Pure builders (no streamlit): each takes a ``PageData`` and returns one single-line HTML string.
"""

from __future__ import annotations

from html import escape

from astro_triage.dashboard import fmt, html_kit
from astro_triage.dashboard.data_headline import PAIR_NAMES
from astro_triage.dashboard.data_overview_page import PageData

TITLE = "Estrela, galáxia ou quasar? Ensinamos um modelo a decidir."
_CHIP_ACCENT = "ao-chip-a"
_CHIP_OUTLINE = "ao-chip-o"
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
    (
        "Próximo passo",
        _CHIP_OUTLINE,
        "Comparar com redes neurais",
        (
            "A mesma prova, com as mesmas regras, para o aprendizado profundo. "
            "Teste aberto uma vez; resultado provisório."
        ),
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
    """Title, supporting sentence (only when the pair total exists) and the three step cards."""
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
