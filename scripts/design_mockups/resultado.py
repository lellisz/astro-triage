"""Screen 1, 'Resultado': purity field, prior ruler, overall accuracy and test census."""

from __future__ import annotations

import math

from astro_triage.dashboard import fmt, theme

from . import loaders
from .kit import MONO, NA, Ledger, e
from .loaders import Cell

SCREEN = "resultado"
TITLE = "Resultado"
FIELD_N = 1000
FIELD_COLS = (40, 20)  # desktop 40x25, mobile 20x50
P2_CITATION = "Delchambre+2023"
P3_NOTE = loaders.Cell(
    "P3 vale só para parte do céu (bandas B e C; ADR 0003)",
    "docs/adr/0003-split-and-priors.md",
    "linha 88 · P3 is applied only to test objects in bands B and C",
)
STATES = (
    ("A", "as_shipped", "Como entregue", "prior do céu P2 · probabilidades do treino"),
    (
        "B",
        "prior_matched",
        "Ajustada ao prior do céu",
        "prior do céu P2 · ajuste da ADR 0003",
    ),
)

CSS = f"""
.insc{{font:italic 400 clamp(28px,4.2vw,52px)/1.15 {theme.FONT_SERIF};margin:0 0 32px;max-width:900px}}
.tgi{{position:absolute;opacity:0}}
.tgl{{display:none;gap:8px;margin:16px 0}}
.tgl label{{border:1px solid {theme.HAIRLINE};padding:10px 14px;min-height:{theme.MIN_TARGET_PX}px;font-size:{theme.CONTROL_PX}px;color:{theme.SOFT}}}
.states{{display:grid;grid-template-columns:1fr 1fr;gap:32px}}
.st{{border-top:1px solid {theme.HAIRLINE};padding-top:16px;min-width:0}}
.st h2{{margin:0}}
.st .body{{display:grid;grid-template-columns:minmax(150px,auto) 1fr;gap:24px;margin-top:16px}}
.num{{font:400 clamp(64px,8vw,112px)/1 {theme.FONT_SERIF};color:{theme.ACCENT}}}
.field{{display:grid;grid-template-columns:repeat(var(--cols,{FIELD_COLS[0]}),1fr);gap:3px;min-width:0}}
.field i{{aspect-ratio:1;border-radius:50%;box-shadow:inset 0 0 0 1px {theme.DASH}}}
.field i.q{{background:{theme.ACCENT};box-shadow:none}}
.bar{{position:relative;height:12px;background:{theme.GRID};margin:8px 0 4px}}
.bar .fill{{position:absolute;inset:0 auto 0 0;background:{theme.ACCENT}}}
.bar .ci{{position:absolute;top:-4px;bottom:-4px;border:solid {theme.TEXT};border-width:0 1px}}
.bar .ci b{{position:absolute;left:0;right:0;top:50%;border-top:1px solid {theme.TEXT}}}
.bar .tick{{position:absolute;top:-4px;bottom:-4px;width:2px;background:{theme.ACCENT}}}
.lg{{margin-top:8px}}
.rl-scroll{{overflow-x:auto;margin:16px 0}}
.rl{{position:relative;min-width:640px;height:190px;margin:0 44px}}
.rl .axis{{position:absolute;left:0;right:0;top:80px;border-top:1px solid {theme.SOFT}}}
.rl .t{{position:absolute;top:92px;transform:translateX(-50%);font:{theme.MIN_SECONDARY_PX}px {MONO};color:{theme.TEXT_SECONDARY}}}
.rl .t::before{{content:"";position:absolute;left:50%;top:-12px;height:8px;border-left:1px solid {theme.SOFT}}}
.rl .mk{{position:absolute;top:73px;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:{theme.ACCENT}}}
.rl .mkl{{position:absolute;top:46px;transform:translateX(-50%);font:{theme.MIN_SECONDARY_PX}px {MONO};color:{theme.TEXT}}}
.rl .cal{{position:absolute;top:0;transform:translateX(-50%);white-space:nowrap;font:{theme.MIN_SECONDARY_PX}px {MONO};color:{theme.ACCENT}}}
.rl .cal::after{{content:"";position:absolute;left:50%;top:100%;height:24px;border-left:1px solid {theme.ACCENT}}}
.rl .s{{position:absolute;top:130px;width:10px;height:10px;margin-left:-5px;border-radius:50%;background:{theme.ACCENT}}}
.rl .s.off{{background:transparent;box-shadow:inset 0 0 0 1px {theme.DASH}}}
.rl .scap{{position:absolute;top:150px;left:0;white-space:nowrap}}
.tw{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;min-width:640px;font-size:{theme.MIN_SECONDARY_PX}px}}
th,td{{text-align:left;padding:8px 12px 8px 0;border-bottom:1px solid {theme.HAIRLINE};vertical-align:top}}
th{{color:{theme.TEXT_SECONDARY};font-weight:400}}
.sec{{margin-top:56px}}
.acc{{display:grid;grid-template-columns:1fr 1fr;gap:32px}}
.acc .v{{font:400 56px/1 {theme.FONT_SERIF};color:{theme.TEXT}}}
.census{{margin-top:56px;padding-top:16px;border-top:1px solid {theme.HAIRLINE}}}
.census .mono{{font-size:20px;color:{theme.TEXT}}}
@media(max-width:1000px){{.states,.acc{{grid-template-columns:1fr}}}}
@media(max-width:720px){{
.sw{{display:flex;flex-direction:column}}.states{{order:1}}.tgl{{order:2;display:flex}}
.st .body{{grid-template-columns:1fr}}.field{{--cols:{FIELD_COLS[1]}}}
#sta:checked ~ .states .stB,#stb:checked ~ .states .stA{{display:none}}
#sta:checked ~ .tgl label[for=sta],#stb:checked ~ .tgl label[for=stb]{{border-color:{theme.ACCENT};color:{theme.ACCENT}}}
#sta:focus-visible ~ .tgl label[for=sta],#stb:focus-visible ~ .tgl label[for=stb]{{outline:2px solid {theme.ACCENT}}}}}
"""


def _pos(value: float) -> float:
    return max(0.0, min(100.0, value * 100))


def _bar(point: Cell, lo: Cell, hi: Cell, fill: bool) -> str:
    if not (point.ok and lo.ok and hi.ok):
        return ""
    parts = (
        [f"<div class='fill' style='width:{_pos(point.value):.2f}%'></div>"]
        if fill
        else []
    )
    width = _pos(hi.value) - _pos(lo.value)
    parts.append(
        f"<div class='ci' style='left:{_pos(lo.value):.2f}%;width:{width:.2f}%'><b></b></div>"
    )
    if not fill:
        parts.append(f"<div class='tick' style='left:{_pos(point.value):.2f}%'></div>")
    return f"<div class='bar'>{''.join(parts)}</div>"


def _field(L: Ledger, where: str, purity: Cell) -> tuple[str, int | None]:
    if not purity.ok:
        return f"<div class='sm'>{NA}</div>", None
    n_blue = round(FIELD_N * purity.value)
    L.derived(
        fmt.pt_int(n_blue),
        f"{where} · pontos azuis do campo",
        purity,
        f"round({FIELD_N} × pureza)",
    )
    dots = "<i class='q'></i>" * n_blue + "<i></i>" * (FIELD_N - n_blue)
    label = f"{n_blue} de {FIELD_N} pontos são quasares"
    return f"<div class='field' role='img' aria-label='{e(label)}'>{dots}</div>", n_blue


def _state(
    L: Ledger, tag: str, variant: str, title: str, sub: str, rel
) -> tuple[str, int | None]:
    c = loaders.qso_state(rel, loaders.STATE_PRIOR, variant)
    where = f"estado {tag}"
    pur = L.pct(c["purity_point"], f"{where} · pureza de quasar")
    pur_ic = L.interval(c["purity_lo"], c["purity_hi"], f"{where} · pureza")
    comp = L.pct(c["completeness_point"], f"{where} · completude de quasar")
    comp_ic = L.interval(
        c["completeness_lo"], c["completeness_hi"], f"{where} · completude"
    )
    field, n_blue = _field(L, where, c["purity_point"])
    per_dot = L.derived(
        fmt.pt_pct(1 / FIELD_N, 1),
        f"{where} · fração por ponto",
        c["purity_point"],
        f"1 / {FIELD_N}",
    )
    html = (
        f"<section class='st st{tag}'><h2>{title}</h2><div class='sm'>{sub}</div>"
        f"<div class='body'><div><div class='num'>{e(pur)}</div>"
        "<div class='sm'>dos chamados de quasar são quasares</div>"
        f"<div class='sm mono'>{e(pur_ic)}</div>"
        f"{_bar(c['purity_point'], c['purity_lo'], c['purity_hi'], False)}</div>"
        f"<div>{field}<div class='sm lg'>1 ponto ≈ {e(per_dot)} da seleção · composição esperada "
        "sob o prior, não contagem de objetos</div></div></div>"
        f"<div style='margin-top:20px'>Quasares encontrados: <b>{e(comp)}</b>"
        f"<div class='sm mono'>{e(comp_ic)}</div>"
        f"{_bar(c['completeness_point'], c['completeness_lo'], c['completeness_hi'], True)}</div>"
        "</section>"
    )
    return html, n_blue


def _log_pos(q: float, lo: float, hi: float) -> float:
    return (math.log10(q) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * 100


def _ruler(L: Ledger, rel) -> str:
    lo_c, hi_c = loaders.sweep_range(rel)
    if not (lo_c.ok and hi_c.ok):
        return f"<div class='sm'>{NA}</div>"
    lo, hi = lo_c.value, hi_c.value
    priors = loaders.sky_priors(rel)
    ticks = []
    for k in range(math.ceil(math.log10(lo)), math.floor(math.log10(hi)) + 1):
        label = L.derived(
            fmt.pt_g(10.0**k * 100) + "%",
            f"régua · escala 10^{k}",
            lo_c,
            f"10^{k} em %",
        )
        ticks.append(
            f"<span class='t' style='left:{_log_pos(10.0**k, lo, hi):.2f}%'>{e(label)}</span>"
        )
    marks = []
    for name, cell in priors.items():
        if not cell.ok:
            continue
        x = _log_pos(cell.value, lo, hi)
        marks.append(
            f"<div class='mkl' style='left:{x:.2f}%'>{e(name)}{'*' if name == 'P3' else ''}</div><div class='mk' style='left:{x:.2f}%'></div>"
        )
        if name == loaders.STATE_PRIOR:
            shown = L.pct(cell, f"régua · prior {name} do quasar", 3)
            marks.append(
                f"<div class='cal' style='left:{x:.2f}%'>{P2_CITATION} · {e(shown)}</div>"
            )
    dots = "".join(
        f"<div class='s{'' if ok else ' off'}' style='left:{_log_pos(c.value, lo, hi):.2f}%'"
        f"{'' if ok else ' title="sem dado"'}></div>"
        for c, ok in loaders.sweep_marks(rel)
    )
    lo_txt = L.derived(
        fmt.pt_g(lo * 100) + "%", "régua · varredura S, mínimo", lo_c, "min(qso_prior)"
    )
    hi_txt = L.derived(
        fmt.pt_g(hi * 100) + "%", "régua · varredura S, máximo", hi_c, "max(qso_prior)"
    )
    caption = (
        f"varredura S: quasar de {e(lo_txt)} a {e(hi_txt)} · marca vazada: sem dado (n/a) · "
        "P0 (uniforme) fica fora: é o conjunto balanceado, não o céu"
    )
    return (
        "<div class='rl-scroll'><div class='rl'><div class='axis'></div>"
        f"{''.join(ticks)}{''.join(marks)}{dots}<div class='sm scap'>{caption}</div></div></div>"
    )


def _prior_table(L: Ledger, rel, p3_note: str) -> str:
    rows = []
    for name, cell in loaders.sky_priors(rel).items():
        c = loaders.qso_state(rel, name, "prior_matched")
        w = f"tabela {name}"
        pur_ic = L.interval(c["purity_lo"], c["purity_hi"], f"{w} · pureza ajustada")
        comp_ic = L.interval(
            c["completeness_lo"], c["completeness_hi"], f"{w} · completude ajustada"
        )
        note = f"<div class='sm'>{e(p3_note)}</div>" if name == "P3" else ""
        rows.append(
            f"<tr><td class='mono'>{e(name)}{note}</td><td>{e(L.pct(cell, f'{w} · prior do quasar', 3))}</td>"
            f"<td>{e(L.pct(c['purity_point'], f'{w} · pureza ajustada'))}<div class='mono'>{e(pur_ic)}</div></td>"
            f"<td>{e(L.pct(c['completeness_point'], f'{w} · completude ajustada'))}"
            f"<div class='mono'>{e(comp_ic)}</div></td></tr>"
        )
    return (
        "<div class='tw'><table><tr><th>Prior</th><th>Prior de quasar assumido</th>"
        "<th>Pureza, ajustada ao prior</th><th>Completude, ajustada ao prior</th></tr>"
        f"{''.join(rows)}</table></div>"
    )


def _accuracy(L: Ledger, rel) -> str:
    a = loaders.accuracy(rel)
    regime = str(a["regime"].value or "")
    balanced = "BALANCED" in regime and "unweighted" in regime
    n_seeds = L.integer(a["n_seeds"], "acerto geral · número de sementes")
    lo = L.num(a["seeds_min"], "acerto geral · mínimo das sementes")
    hi = L.num(a["seeds_max"], "acerto geral · máximo das sementes")
    regime_txt = (
        "regime balanceado (prior uniforme), sem pesos" if balanced else f"regime: {NA}"
    )
    return (
        "<section class='sec'><h2>Acerto geral</h2>"
        "<p class='sm'>Duas definições, lidas de arquivos diferentes. Não são o mesmo número "
        "e não se comparam entre si.</p><div class='acc'>"
        f"<div><div class='v'>{e(L.pct(a['shipped'], 'acerto geral · como entregue', 2))}</div>"
        "<div><b>Como entregue</b>: 1 modelo, probabilidades do treino, mistura do conjunto "
        "de teste, sem pesos.</div></div>"
        f"<div><div class='v'>{e(L.pct(a['seeds_mean'], 'acerto geral · média das sementes', 2))}</div>"
        f"<div><b>Média de {e(n_seeds)} sementes</b>: {regime_txt}.</div>"
        f"<div class='sm mono'>mínimo a máximo (fração): {e(lo)} a {e(hi)}</div></div>"
        "</div></section>"
    )


def render(rel, L: Ledger) -> tuple[str, str, dict]:
    """Returns (body, footer, facts); the kit wraps them with navigation and seal."""
    states, blue = [], {}
    for tag, variant, title, sub in STATES:
        html, n = _state(L, tag, variant, title, sub, rel)
        states.append(html)
        blue[tag] = n
    cen = loaders.census(rel)
    p3_note = L.text(P3_NOTE, "régua e tabela · aviso de P3 parcial")
    body = (
        f"<p class='insc'>De cada {fmt.pt_int(FIELD_N)} objetos que o modelo chama de quasar, quantos são?</p>"
        "<div class='sw'><input class='tgi' type='radio' name='st' id='sta' checked aria-label='Como entregue'>"
        "<input class='tgi' type='radio' name='st' id='stb' aria-label='Ajustada ao prior'>"
        "<div class='tgl'><label for='sta'>Como entregue</label><label for='stb'>Ajustada ao prior</label></div>"
        f"<div class='states'>{''.join(states)}</div></div>"
        "<section class='sec'><h2>Régua do prior</h2>"
        "<p class='sm'>O prior é a fração de quasares assumida (escala logarítmica). “Como entregue” "
        "fica fixo em P2; a régua muda o prior e o ajuste do modelo (ADR 0003). "
        f"* {e(p3_note)}.</p>"
        f"{_ruler(L, rel)}{_prior_table(L, rel, p3_note)}</section>"
        f"{_accuracy(L, rel)}"
        "<section class='census'><div class='mono'>"
        f"{e(L.integer(cen['regions'], 'faixa-censo · regiões de teste'))} regiões de teste · "
        f"{e(L.integer(cen['objects'], 'faixa-censo · objetos de teste'))} objetos</div>"
        "<div class='sm'>Regiões vizinhas de treino não são removidas (ADR 0003).</div></section>"
    )
    footer = (
        f"Release {e(rel.name)} · {e(fmt.pt_ci_label())} por bootstrap de regiões "
        f"(n_boot {e(L.integer(cen['n_boot'], 'rodapé · n_boot'))}, semente "
        f"{e(L.text(cen['seed'], 'rodapé · semente'))})"
    )
    return body, footer, {"blue_dots": blue}
