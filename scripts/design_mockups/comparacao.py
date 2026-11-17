"""Screen 3, 'DL × LightGBM': one derived verdict line, per-class difference rulers, absolute band."""

from __future__ import annotations

from itertools import pairwise

from astro_triage.dashboard import fmt, theme
from astro_triage.dashboard.data_dl import load_results_dl
from astro_triage.dashboard.data_headline import PAIR_NAMES
from astro_triage.dashboard.html_dl import class_verdict, pair_label, role_label

from . import loaders
from .kit import MONO, NA, TABLE_CSS, Ledger, e
from .loaders import Cell

SCREEN = "comparacao"
TITLE = "DL × LightGBM"
FAMILY = {"lgbm": "LightGBM", "mlp": "MLP", "ftt": "FTT"}
PAIR_VERDICT = {"tie": "empate", "win": "vitória do par"}
CLASS_ORDER = ("QSO", "GALAXY")
SCALE_PAD = 1.15  # axis half-range = largest drawn value x SCALE_PAD
ABS_MAX = 0.10  # absolute-purity axis of the spec: 0 to 10%
PARAM_FILE = "scripts/design_mockups/comparacao.py"
MIN_TICK_GAP_PX = 280  # narrowest track the three tick labels must fit in (390 viewport)
CHAR_PX = 9.0  # width of one mono character at the secondary size, for the overlap check

CSS = (
    TABLE_CSS
    + f"""
.panels{{display:grid;grid-template-columns:1fr 1fr;gap:32px}}
.panel{{border-top:1px solid {theme.HAIRLINE};padding-top:16px;min-width:0}}
.panel h2{{margin:0}}
.dr{{display:grid;grid-template-columns:72px minmax(0,1fr);gap:4px 12px;margin-top:20px;align-items:center}}
.dr .dn{{font-size:{theme.CONTROL_PX}px}}
.dr .dv{{grid-column:2;font:{theme.MIN_SECONDARY_PX}px/1.5 {MONO};color:{theme.TEXT_SECONDARY}}}
.trk{{position:relative;height:36px}}
.trk .zero{{position:absolute;left:50%;top:0;bottom:0;border-left:1px solid {theme.TEXT_SECONDARY}}}
.trk .band{{position:absolute;top:0;bottom:0;background:{theme.ACCENT};opacity:.16}}
.trk .ci{{position:absolute;top:13px;height:10px;background:{theme.ACCENT};opacity:.55;border:solid {theme.ACCENT};border-width:0 1px}}
.trk .pt{{position:absolute;top:9px;width:18px;height:18px;margin-left:-9px;border-radius:50%;background:{theme.ACCENT};box-shadow:0 0 0 2px {theme.BG}}}
.trk .lo{{position:absolute;left:0;right:0;top:50%;border-top:1px solid {theme.HAIRLINE}}}
.ax{{position:relative;height:24px;font:{theme.MIN_SECONDARY_PX}px {MONO};color:{theme.TEXT_SECONDARY}}}
.ax span{{position:absolute;top:2px;white-space:nowrap;transform:translateX(-50%)}}
.ax span.l{{transform:none}}.ax span.r{{transform:translateX(-100%)}}
.abs{{max-width:760px}}
details{{margin-top:16px;border-top:1px solid {theme.HAIRLINE};padding-top:12px}}
summary{{cursor:pointer;min-height:{theme.MIN_TARGET_PX}px;color:{theme.SOFT};font-size:{theme.CONTROL_PX}px}}
summary:focus-visible{{outline:2px solid {theme.ACCENT}}}
@media(max-width:1000px){{.panels{{grid-template-columns:1fr}}}}
"""
)


def _family(system: str) -> str:
    key = system.partition("_")[0]
    return FAMILY.get(key, key.upper())


def _pos(value: float, half: float) -> float:
    return max(0.0, min(100.0, 50 + value / half * 50))


def _ic(L: Ledger, c: dict[str, Cell], what: str) -> str:
    lo, hi = L.num(c["lo"], f"{what} (IC, inferior)", 4, True), L.num(
        c["hi"], f"{what} (IC, superior)", 4, True
    )
    if NA in (lo, hi):
        return f"{fmt.pt_ci_label()}: {NA}"
    return f"{fmt.pt_ci_label()}: {lo} a {hi}"


def _verdict_text(L: Ledger, dl: dict, headline: list[str]) -> str:
    """One line derived from the verdicts read; the second sentence only when the data say so."""
    heads, wins = [], []
    for pair in headline:
        fam = _family(pair)
        raw = loaders.dl_pair_verdict(dl, pair)
        L.text(raw, f"veredito do par lido · {fam}")
        key = str(raw.value or "").partition(":")[0].strip()
        pv = PAIR_VERDICT.get(key, NA)
        heads.append(f"{fam}: {pv}")
        won, behind = [], []
        for cls in CLASS_ORDER:
            cv = loaders.dl_class(dl, pair, cls)["verdict"]
            L.text(cv, f"veredito da classe lido · {fam} · {PAIR_NAMES[cls]}")
            if cv.value == "win":
                won.append(PAIR_NAMES[cls])
            elif cv.value == "worse":
                behind.append(PAIR_NAMES[cls])
        if won and key != "win":
            wins.append(f"{fam} vence só em {' e '.join(won)} (classe)")
        if behind:
            wins.append(f"{fam} fica atrás só em {' e '.join(behind)} (classe)")
    line = " · ".join(heads) + "."
    if wins:
        line += f" {'; '.join(wins)}; o par exige as duas classes."
    first = loaders.dl_pair_verdict(dl, headline[0])
    return L.derived(line, "veredito de uma linha", first, "derivado dos vereditos lidos")


def _tick_row(labels: list[str]) -> str:
    cls = ("l", "", "r")
    pos = (0, 50, 100)
    spans = "".join(
        f"<span{' class=' + repr(c) if c else ''} style='left:{p}%'>{e(t)}</span>"
        for t, c, p in zip(labels, cls, pos, strict=True)
    )
    return f"<div class='dr'><div></div><div class='ax' aria-hidden='true'>{spans}</div></div>"


def _check_ticks(labels: list[str]) -> None:
    """The three labels sit at 0, 50 and 100% of the narrowest track; their boxes must not touch."""
    w = [len(t) * CHAR_PX for t in labels]
    edges = (
        (0, w[0]),
        (MIN_TICK_GAP_PX / 2 - w[1] / 2, MIN_TICK_GAP_PX / 2 + w[1] / 2),
        (MIN_TICK_GAP_PX - w[2], MIN_TICK_GAP_PX),
    )
    if any(a[1] >= b[0] for a, b in pairwise(edges)):
        raise ValueError(f"comparação: rótulos do eixo se tocam: {labels}")


def _diff_panel(L: Ledger, dl: dict, headline: list[str], cls: str) -> tuple[str, dict]:
    name = PAIR_NAMES[cls]
    cells = {p: loaders.dl_class(dl, p, cls) for p in headline}
    vals = [
        abs(c[k].value)
        for c in cells.values()
        for k in ("d", "lo", "hi", "delta")
        if c[k].ok
    ]
    if not vals:
        return f"<section class='panel'><h2>{e(name.capitalize())}</h2><div>{NA}</div></section>", {}
    half = max(vals) * SCALE_PAD
    anchor = next(iter(cells.values()))["delta"]
    half_txt = L.derived(
        fmt.pt_num(half, 4),
        f"{name} · meia-escala do eixo",
        anchor,
        f"{SCALE_PAD} × maior valor desenhado",
    )
    ticks = [f"−{half_txt}", "0", f"+{half_txt}"]
    _check_ticks(ticks)
    rows, halfw = [], []
    for pair, c in cells.items():
        fam = _family(pair)
        w = f"{name} · {fam}"
        d, delta = L.num(c["d"], f"{w} · D", 4, True), L.num(c["delta"], f"{w} · δ", 4)
        verdict = class_verdict(c["verdict"].value)
        marks = ""
        if c["delta"].ok:
            dl_, dh_ = _pos(-c["delta"].value, half), _pos(c["delta"].value, half)
            marks += f"<div class='band' style='left:{dl_:.2f}%;width:{dh_ - dl_:.2f}%'></div>"
        marks += "<div class='lo'></div><div class='zero'></div>"
        if c["lo"].ok and c["hi"].ok:
            a, b = _pos(c["lo"].value, half), _pos(c["hi"].value, half)
            marks += f"<div class='ci' style='left:{a:.2f}%;width:{b - a:.2f}%'></div>"
            hw = L.derived(
                fmt.pt_num((c["hi"].value - c["lo"].value) / 2, 2),
                f"{w} · meia-largura do IC",
                c["hi"],
                "(limite superior − limite inferior) / 2",
            )
            halfw.append((hw, c))
        if c["d"].ok:
            marks += f"<div class='pt' style='left:{_pos(c['d'].value, half):.2f}%'></div>"
        rows.append(
            f"<div class='dr'><div class='dn'>{e(fam)}</div><div class='trk'>{marks}</div>"
            f"<div class='dv'>D = {e(d)} · {e(_ic(L, c, w))} · δ = {e(delta)}<br>{e(verdict)}</div></div>"
        )
    note = ""
    wide = [
        hw
        for hw, c in halfw
        if c["verdict"].value == "tie"
        and c["lo"].value < 0 < c["hi"].value
        and c["hi"].value - c["lo"].value > 2 * c["delta"].value
    ]
    if wide and len(wide) == len(halfw):
        span = f"±{wide[0]}" if len(set(wide)) == 1 else f"±{min(wide)} a ±{max(wide)}"
        note = (
            f"<p class='sm'>IC ≈ {e(span)}: sem poder para distinguir. Isso é falta de poder "
            "para decidir, não indica que os dois sejam iguais.</p>"
        )
    html = (
        f"<section class='panel'><h2>{e(name.capitalize())}</h2>"
        "<div class='sm'>escala própria; faixa clara = ±δ; barra = intervalo de confiança; "
        "ponto = D (rede menos LightGBM)</div>"
        f"{''.join(rows)}{_tick_row(ticks)}{note}</section>"
    )
    return html, {"half": round(half, 6)}


def _abs_band(L: Ledger, dl: dict, headline: list[str], rel) -> str:
    systems = [("lgbm_full", headline[0])] + [(p.partition("-")[0], p) for p in headline]
    axis = [
        L.derived(
            fmt.pt_pct(v, 0),
            f"faixa absoluta · eixo {fmt.pt_pct(v, 0)}",
            Cell(ABS_MAX, PARAM_FILE, "ABS_MAX"),
            "0, metade e fim do eixo de 0 a 10%",
        )
        for v in (0.0, ABS_MAX / 2, ABS_MAX)
    ]
    _check_ticks(axis)
    rows = []
    for system, pair in systems:
        c = loaders.dl_purity(dl, pair, system, "QSO")
        fam = _family(system)
        w = f"faixa absoluta · {fam}"
        med, lo, hi = (
            L.pct(c["median"], f"{w} · mediana", 2),
            L.pct(c["min"], f"{w} · mínimo", 2),
            L.pct(c["max"], f"{w} · máximo", 2),
        )
        marks = "<div class='lo'></div>"
        if all(c[k].ok for k in c):
            if c["max"].value > ABS_MAX:
                raise ValueError(f"comparação: {fam} acima do eixo de 0 a 10%")
            a, b = c["min"].value / ABS_MAX * 100, c["max"].value / ABS_MAX * 100
            marks += f"<div class='ci' style='left:{a:.2f}%;width:{max(b - a, 0.4):.2f}%'></div>"
            marks += f"<div class='pt' style='left:{c['median'].value / ABS_MAX * 100:.2f}%'></div>"
        rows.append(
            f"<div class='dr'><div class='dn'>{e(fam)}</div><div class='trk'>{marks}</div>"
            f"<div class='dv'>mediana {e(med)} · mínimo a máximo das sementes: {e(lo)} a {e(hi)}</div></div>"
        )
    prior = loaders.sky_priors(rel)[loaders.STATE_PRIOR]
    qso_prior = L.pct(prior, "faixa absoluta · prior de quasar P2", 3)
    return (
        "<section class='sec abs'><h2>Pureza absoluta de quasar</h2>"
        f"<p class='sm'>Eixo de 0 a 10%. O prior de quasar é {e(qso_prior)}, por isso o nível "
        "é baixo em todos; a comparação entre os sistemas é a diferença D acima, não o nível.</p>"
        f"{''.join(rows)}{_tick_row(axis)}</section>"
    )


def _rows(L: Ledger, dl: dict, pairs: list[str]) -> str:
    out = []
    for pair in pairs:
        role = loaders.dl_pairs(dl)[pair]["role"]
        for cls in CLASS_ORDER:
            c = loaders.dl_class(dl, pair, cls)
            w = f"{pair} · {PAIR_NAMES[cls]}"
            out.append(
                f"<tr><td>{e(pair_label(pair))}</td><td>{e(role_label(role))}</td>"
                f"<td>{e(PAIR_NAMES[cls])}</td><td class='mono'>{e(L.num(c['d'], f'{w} · D', 4, True))}</td>"
                f"<td class='mono'>{e(_ic(L, c, w))}</td>"
                f"<td class='mono'>{e(L.num(c['delta'], f'{w} · δ', 4))}</td>"
                f"<td>{e(class_verdict(c['verdict'].value))}</td></tr>"
            )
    return (
        "<div class='tw'><table><tr><th>Par</th><th>Papel</th><th>Classe</th><th>D</th>"
        f"<th>{e(fmt.pt_ci_label())}</th><th>δ</th><th>Veredito da classe</th></tr>"
        f"{''.join(out)}</table></div>"
    )


def _indicator_line(L: Ledger, dl: dict) -> str:
    pairs = [
        p
        for p, b in loaders.dl_pairs(dl).items()
        if str(b.get("role", "")).startswith(("front3", "front4"))
    ]
    cells = [loaders.dl_class(dl, p, "QSO") for p in pairs]
    if not pairs or not all(c["d"].ok for c in cells):
        return f"<p>Efeito dos indicadores na pureza de quasar: {NA}.</p>"
    ds = [c["d"] for c in cells]
    lo = L.num(min(ds, key=lambda c: c.value), "indicadores · menor D", 3)
    hi = L.num(max(ds, key=lambda c: c.value), "indicadores · maior D", 3)
    above = all(c["lo"].ok and c["lo"].value > 0 for c in cells)
    tail = " O intervalo fica acima de zero nas três." if above and len(cells) == 3 else ""
    n = L.derived(
        str(len(pairs)), "indicadores · número de famílias", ds[0], "contagem das frentes 3 e 4"
    )
    return (
        f"<p>Efeito dos indicadores na pureza de quasar (D, com indicadores menos sem): "
        f"{e(lo)} a {e(hi)} nas {e(n)} famílias.{tail}</p>"
    )


def render(rel, L: Ledger) -> tuple[str, str, dict]:
    dl = load_results_dl(rel)
    if dl is None:
        body = f"<p class='lead'>Comparação: {NA}</p><p class='sm'>Esta versão não tem a avaliação do aprendizado profundo.</p>"
        return body, f"Release {e(rel.name)}", {"available": False}
    prior, cstar = loaders.dl_level(dl)
    pairs = loaders.dl_pairs(dl)
    headline = [p for p, b in pairs.items() if b.get("role") == "headline"]
    level_cell = Cell(float(cstar), loaders.DL_FILE, f"compare.pairs.*.{prior}.{cstar}")
    level = f"{prior}, C* = {L.num(level_cell, 'nível de destaque · C*', 2)}"
    line = _verdict_text(L, dl, headline)
    panels, facts = [], {}
    for cls in CLASS_ORDER:
        html, f = _diff_panel(L, dl, headline, cls)
        panels.append(html)
        facts[cls] = f
    descriptive = [p for p, b in pairs.items() if b.get("role") == "descriptive"]
    fronts = [p for p, b in pairs.items() if str(b.get("role", "")).startswith("front")]
    body = (
        f"<p class='lead'>{e(line)}</p>"
        f"<p class='sm'>Ponto de operação do destaque: {e(level)}. D é a diferença de pureza "
        "(rede menos LightGBM); a regra pré-registrada decide por classe e o par exige as duas.</p>"
        f"<div class='panels sec' style='margin-top:32px'>{''.join(panels)}</div>"
        f"{_abs_band(L, dl, headline, rel)}"
        "<section class='sec'><h2>Efeito dos indicadores</h2>"
        f"{_indicator_line(L, dl)}</section>"
        "<section class='sec'><h2>Descritivo</h2>"
        "<details><summary>Conjuntos de sementes (descritivo)</summary>"
        f"{_rows(L, dl, descriptive)}</details>"
        "<details><summary>Frentes 2 a 4: indicadores (descritivo)</summary>"
        f"{_rows(L, dl, fronts)}</details></section>"
    )
    n_boot = L.integer(
        Cell(dl["compare"]["n_boot"], loaders.DL_FILE, "compare.n_boot"), "rodapé · n_boot"
    )
    footer = (
        f"Release {e(rel.name)} · {e(fmt.pt_ci_label())} por bootstrap em blocos do céu "
        f"(n_boot {e(n_boot)})"
    )
    return body, footer, {"verdict": line, "level": level, "axes": facts}
