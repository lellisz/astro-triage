"""One function per site figure. Every number comes from the exported page JSON."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from astro_triage.web_export.figures.canvas import (
    ACCENT,
    LINE,
    SERIES,
    TEXT2,
    TEXT3,
    Canvas,
    M,
    W,
    decimals_for,
    nice_ticks,
    place_labels,
    pt_dec,
    pt_int,
    pt_pct,
    tw,
    wrap,
)

Pages = Mapping[str, Any]
NICE_TOPS: Final = (0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0)
PCT_TICKS: Final = (0, 0.25, 0.5, 0.75, 1)


@dataclass(frozen=True)
class Figure:
    id: str
    title: str
    caption: str
    aria_label: str
    svg: str


def _done(cv: Canvas, subtitle: str, desc: str, release: str) -> Figure:
    caption = f"{cv.title}. {subtitle}. PROVISÓRIO · {release}."
    return Figure(cv.fid, cv.title, caption, f"{cv.title}: {desc}", cv.render(desc))


def _ci(t: str) -> str:
    return t.replace("IC 95%: ", "")


def _rep(est: Mapping[str, Any], below_floor: bool = False) -> bool:
    return not below_floor and est["point"] is not None


def _top(values: list[float]) -> float:
    peak = max(values, default=0.0)
    return next((t for t in NICE_TOPS if t >= peak), 1.0)


# 1a ------------------------------------------------------------- calibration
def calibration(p: Pages) -> Figure:
    g, rel = p["graficos"], p["meta"]["release"]
    cal = g["calibration"]
    sub = f"{g['model_label']} · conjunto de teste"
    cv = Canvas("calibracao", "Diagrama de confiabilidade", sub)
    px, ps = 72, 336
    py = cv.y + 26

    def x_(v: float) -> float:
        return px + v * ps

    def y_(v: float) -> float:
        return py + (1 - v) * ps

    for t in PCT_TICKS:
        cv.line(x_(t), py, x_(t), py + ps, "grid")
        cv.line(px, y_(t), px + ps, y_(t), "grid")
        cv.text(x_(t), py + ps + 22, pt_pct(t), "tk", "middle")
        cv.text(px - 8, y_(t) + 5, pt_pct(t), "tk", "end")
    cv.line(px, py + ps, px + ps, py + ps, "axis")
    cv.line(px, py, px, py + ps, "axis")
    cv.line(x_(0), y_(0), x_(1), y_(1), "ref")
    cv.text(px + ps - 5, py + ps - 10, "Calibração perfeita", "n", "end")
    best: tuple[float, tuple[float, float] | None] = (0.0, None)
    for si, s in reversed(list(enumerate(cal["series"]))):
        k = si + 1
        ok = [b for b in s["bins"] if b["p_mean"]["value"] is not None and b["freq"]["point"] is not None]
        for b in ok:
            if b["freq"]["lo"] is not None:
                x = x_(b["p_mean"]["value"])
                cv.line(x, y_(b["freq"]["lo"]), x, y_(b["freq"]["hi"]), f"ci{k}")
        raw = [(x_(b["raw_p_mean"]), y_(b["raw_freq"])) for b in ok
               if b["raw_p_mean"] is not None and b["raw_freq"] is not None]
        if len(raw) > 1:
            cv.poly(raw, f"raw{k}")
        cv.poly([(x_(b["p_mean"]["value"]), y_(b["freq"]["point"])) for b in ok], f"ln{k}")
        for b in ok:
            if b["raw_p_mean"] is None or b["raw_freq"] is None:
                continue
            d = abs(x_(b["raw_p_mean"]) - x_(b["p_mean"]["value"])) + abs(y_(b["raw_freq"]) - y_(b["freq"]["point"]))
            if d > best[0]:
                best = (d, (x_(b["raw_p_mean"]), y_(b["raw_freq"])))
    marks = []
    for s in cal["series"]:
        ok = [b for b in s["bins"] if b["p_mean"]["value"] is not None and b["freq"]["point"] is not None]
        b = min(ok, key=lambda b: abs(b["freq"]["point"] - 0.5))
        x, y = x_(b["p_mean"]["value"]), y_(b["freq"]["point"])
        cv.line(x - 2, y - 2, x - 9, y - 9, "lead")
        cv.text(x - 12, y - 29, s["label"], "nm", "end")
        pair = f"{b['p_mean']['text']} / {b['freq']['text']}"
        cv.text(x - 12, y - 12, pair, "val", "end")
        marks.append(f"{s['label']}: {pair} ({_ci(b['freq']['ci_text'])})")
    if best[1]:
        rx, ry = best[1]
        cv.line(rx + 3, ry - 3, rx + 40, ry - 26, "lead")
        cv.text(rx + 44, ry - 24, "antes do ajuste", "n")
    cv.text(px + ps / 2, py + ps + 48, "Probabilidade média prevista (%)", "s", "middle")
    cv.vtext(20, py + ps / 2, "Frequência observada (%)", "s")
    cv.y = py + ps + 48
    ns = [b["n"]["value"] for b in cal["series"][0]["bins"] if b["n"]["value"] is not None]
    cv.notes([
        f"n = {pt_int(min(ns))} a {pt_int(max(ns))} objetos por faixa",
        f"{cal['n_bins']} faixas · total {pt_int(sum(ns))} objetos no teste",
        "Barras: IC 95% · valor: prevista / observada",
        "Tracejado fino: pontos antes do ajuste de temperatura",
    ])
    cv.provisional(g["model_label"], rel, f"Piso de {pt_int(g['floor'])} objetos por faixa")
    return _done(cv, sub, "; ".join(marks), rel)


# 2 ---------------------------------------------------------------- rate by bin
def _rate(p: Pages, fid: str, title: str, sub: str, series: list[dict[str, Any]], xlabel: str,
          extra: list[str]) -> Figure:
    g, rel = p["graficos"], p["meta"]["release"]
    cv = Canvas(fid, title, sub)
    px, pw, ph = 64, 262, 240
    py = cv.y + 30
    base = py + ph
    rows0 = series[0]["rows"]
    band = pw / len(rows0)

    def rep(r: Mapping[str, Any]) -> bool:
        return _rep(r["rate"], r["below_floor"])

    top = _top([r["rate"]["hi"] if r["rate"]["hi"] is not None else r["rate"]["point"]
                for s in series for r in s["rows"] if rep(r)])
    ticks, step = nice_ticks(0, top, 4)
    top = ticks[-1]
    dec = decimals_for(step * 100)

    def y_(v: float) -> float:
        return base - v / top * ph

    def xc(i: int) -> float:
        return px + band * (i + 0.5)

    for v in ticks:
        cv.line(px, y_(v), px + pw, y_(v), "grid")
        cv.text(px - 8, y_(v) + 5, pt_pct(v, dec), "tk", "end")
    cv.line(px, base, px + pw, base, "axis")
    cv.line(px, py, px, base, "axis")
    for i, r in enumerate(rows0):
        cv.line(xc(i), base, xc(i), base + 4, "axis")
        cv.vtext(xc(i), base + 10, r["label"], "tk", "end")
        if not any(rep(s["rows"][i]) for s in series):
            cv.vtext(xc(i), base - 6, "n/a", "tk", "start")
    wanted, ends = [], {}
    dodge = (-3, 0, 3) if len(series) > 1 else (0,)
    for si, s in enumerate(series):
        k = si + 1
        runs: list[list[tuple[int, Mapping[str, Any]]]] = []
        cur: list[tuple[int, Mapping[str, Any]]] = []
        for i, r in enumerate(s["rows"]):
            if rep(r):
                cur.append((i, r))
                if r["rate"]["lo"] is not None:
                    x = xc(i) + dodge[si]
                    cv.line(x, y_(r["rate"]["lo"]), x, y_(r["rate"]["hi"]), f"ci{k}")
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        for run in runs:
            pts = [(xc(i), y_(r["rate"]["point"])) for i, r in run]
            if len(pts) == 1:
                pts = [(pts[0][0] - 5, pts[0][1]), (pts[0][0] + 5, pts[0][1])]
            cv.poly(pts, f"ln{k}")
        good = [(i, r) for i, r in enumerate(s["rows"]) if rep(r)]
        if not good:
            continue
        li, lr = good[-1]
        mi, mr = max(good, key=lambda t: t[1]["rate"]["point"])
        if mi != li:
            cv.text(xc(mi), y_(mr["rate"]["point"]) - 9, mr["rate"]["text"], "val", "middle")
        ends[si] = (li, lr)
        wanted.append((si, y_(lr["rate"]["point"])))
    pos = place_labels(wanted, 38, py + 4, base - 18)
    lx = px + pw + 10
    for si, (li, lr) in ends.items():
        y = pos[si]
        cv.line(xc(li) + 3, y_(lr["rate"]["point"]), lx - 3, y - 5, "lead")
        cv.text(lx, y, series[si]["label"], "nm")
        cv.text(lx, y + 18, lr["rate"]["text"], "val")
    lab_h = max(tw(r["label"]) for r in rows0) + 12
    cv.text(px + pw / 2, base + lab_h + 24, xlabel, "s", "middle")
    cv.vtext(20, py + ph / 2, "Taxa de erro (%)", "s")
    cv.y = base + lab_h + 24
    ns = [r["n"]["value"] for s in series for r in s["rows"] if rep(r)]
    cv.notes([
        f"n por faixa com valor: {pt_int(min(ns))} a {pt_int(max(ns))} objetos" if ns
        else "Nenhuma faixa acima do piso: sem taxa nesta figura",
        "Barras verticais: IC 95% · linha interrompida: faixa abaixo do piso (n/a)",
        *extra,
    ])
    cv.provisional(g["model_label"], rel, f"Piso de {pt_int(g['floor'])} objetos por faixa")
    desc = "; ".join(
        f"{s['label']}: " + ", ".join(f"{r['label']} {r['rate']['text']} ({_ci(r['rate']['ci_text'])})"
                                       for r in s["rows"] if rep(r))
        for s in series
    )
    return _done(cv, sub, desc, rel)


def rate_by_g(p: Pages) -> Figure:
    g = p["graficos"]
    eg = g["error_by_g"]
    sub = f"{g['model_label']} · conjunto de teste · por classe"
    return _rate(p, "taxa-por-g", "Taxa de erro por faixa de magnitude G", sub, eg["series"],
                 "Magnitude G (mag)", [eg["missing_note"]])


def _rate_by_z(cls: str, fid: str) -> Callable[[Pages], Figure]:
    def build(p: Pages) -> Figure:
        g = p["graficos"]
        s = next(s for s in g["error_by_z"]["series"] if s["cls"] == cls)
        sub = f"{g['model_label']} · conjunto de teste · {s['label']}"
        return _rate(p, fid, f"Taxa de erro por faixa de redshift: {s['label']}", sub, [s],
                     "Redshift espectroscópico z (adimensional)", [g["error_by_z"]["star_note"]])

    return build


# 3 ---------------------------------------------------------------- confusion
def confusion(p: Pages) -> Figure:
    g, rel = p["graficos"], p["meta"]["release"]
    conf = g["confusion"]
    classes = conf["classes"]
    lab = {c["cls"]: c["label"] for c in classes}
    sub = f"{g['model_label']} · conjunto de teste · fração de cada linha"
    cv = Canvas("confusao", "Matriz de confusão", sub)
    cw = (W - 2 * M) / len(classes)
    cv.y += 30
    cv.text(M + cw * len(classes) / 2, cv.y, "Classe prevista", "s", "middle")
    cv.y += 22
    for j, c in enumerate(classes):
        cv.text(M + cw * (j + 0.5), cv.y, c["label"], "nm", "middle")
    ch = 84
    desc = []
    y = cv.y + 6
    for r in conf["rows"]:
        y += 28
        cv.text(M, y, lab[r["true"]], "nm")
        cv.text(M + tw(lab[r["true"]], bold=True) + 8, y, f"verdadeira · n = {r['n']['text']}", "n")
        y += 8
        for j, c in enumerate(r["cells"]):
            x = M + j * cw
            ok = _rep(c["share"], c["below_floor"])
            alpha = min(max(c["share"]["point"] or 0, 0), 1) * 0.32 if ok else 0
            diag = c["diagonal"]
            cv.rect(x + 1, y + 1, cw - 2, ch - 2, fill=SERIES[0], alpha=alpha,
                    edge=TEXT3 if diag else None, lw=1.5 if diag else 0)
            cx = x + cw / 2
            if ok:
                cv.text(cx, y + 28, c["share"]["text"], "big", "middle")
                cv.text(cx, y + 49, f"n = {c['n']['text']}", "s", "middle")
                cv.text(cx, y + 70, _ci(c["share"]["ci_text"]), "n", "middle")
            else:
                cv.text(cx, y + 28, c["share"]["text"], "n", "middle")
            desc.append(f"{lab[r['true']]} prevista como {lab[c['pred']]}: {c['share']['text']}, "
                        f"n = {c['n']['text']}, {c['share']['ci_text']}")
        y += ch
    cv.y = y
    total = sum(r["n"]["value"] for r in conf["rows"])
    cv.notes([
        "Linha: classe verdadeira, soma 100% · célula: fração, n e IC 95%",
        "Contorno: diagonal (acertos) · tom de cinza proporcional à fração",
        f"Total: {pt_int(total)} objetos no teste",
    ])
    cv.provisional(g["model_label"], rel, f"Célula com menos de {pt_int(g['floor'])} objetos: n/a")
    return _done(cv, sub, "; ".join(desc), rel)


# 4 ---------------------------------------------------------------- colour-magnitude map
def cmd_map(p: Pages) -> Figure:
    g, rel = p["graficos"], p["meta"]["release"]
    cmd = g["cmd"]
    n_test = p["overview"]["technical"]["data_box"]["n_test"]["value"]
    rows, cols = len(cmd["g_edges"]) - 1, len(cmd["c_edges"]) - 1
    sub = f"{g['model_label']} · célula de magnitude G por cor BP-RP"
    cv = Canvas("cmd-taxa", "Taxa de erro no diagrama cor-magnitude", sub)
    px, cwc, chc = 64, 23, 16
    pw, ph = cols * cwc, rows * chc
    py = cv.y + 24
    base = py + ph
    g0, gn = cmd["g_edges"][0], cmd["g_edges"][-1]
    c0, cn = cmd["c_edges"][0], cmd["c_edges"][-1]
    cap = cmd["cap"]["rate"]
    rep = [c for c in cmd["cells"] if _rep(c["rate"], c["below_floor"])]
    for c in cmd["cells"]:
        x, y = px + c["ci"] * cwc, py + c["gi"] * chc
        if c in rep:
            t = min(max(c["rate"]["point"] / cap, 0), 1) if cap else 0
            cv.rect(x, y, cwc, chc, fill=SERIES[0], alpha=0.15 + 0.85 * t)
        else:
            cv.rect(x, y, cwc, chc, hatch=True)
    cv.line(px, base, px + pw, base, "axis")
    cv.line(px, py, px, base, "axis")
    for t in cmd["c_ticks"]:
        x = px + (t["value"] - c0) / (cn - c0) * pw
        cv.line(x, base, x, base + 5, "axis")
        cv.text(x, base + 22, t["text"], "tk", "middle")
    for t in cmd["g_ticks"]:
        y = py + (t["value"] - g0) / (gn - g0) * ph
        cv.line(px - 5, y, px, y, "axis")
        cv.text(px - 9, y + 5, t["text"], "tk", "end")
    cv.text(px + pw / 2, base + 46, "Cor BP-RP (mag)", "s", "middle")
    cv.vtext(20, py + ph / 2, "Magnitude G (mag)", "s")
    out = cmd["n_outside"]
    if rep:
        mx = max(rep, key=lambda c: c["rate"]["point"])
        x, y = px + mx["ci"] * cwc, py + mx["gi"] * chc
        cv.rect(x, y, cwc, chc, edge=ACCENT, lw=2)
        lx = x + cwc / 2
        cv.line(lx, y + chc, lx, base + 62, "lead")
        head = f"Maior taxa: {mx['rate']['text']} · n = {mx['n']['text']}"
        tx = min(lx + 6, W - M - tw(head, bold=True))
        cv.text(tx, base + 72, head, "val")
        cv.text(tx, base + 92, mx["rate"]["ci_text"], "n")
        cv.text(tx, base + 112, f"G {cmd['g_labels'][mx['gi']]} · BP-RP {cmd['c_labels'][mx['ci']]}", "n")
        desc = (f"maior taxa {mx['rate']['text']} ({mx['rate']['ci_text']}, n = {mx['n']['text']}) em G "
                f"{cmd['g_labels'][mx['gi']]}, BP-RP {cmd['c_labels'][mx['ci']]}; {out['text']} objetos fora da área")
    else:
        cv.text(M, base + 72, "Nenhuma célula acima do piso", "n")
        desc = f"nenhuma célula acima do piso; {out['text']} objetos fora da área"
    yb = base + 144
    cv.text(M, yb, "Taxa de erro (%)", "s")
    yb += 22
    cv.text(px - 8, yb + 5, cmd["cap"]["rate_zero_text"], "tk", "end")
    steps = 40
    for i in range(steps):
        cv.rect(px + 200 * i / steps, yb - 5, 200 / steps + 0.5, 10, fill=SERIES[0],
                alpha=0.15 + 0.85 * i / (steps - 1))
    cv.text(px + 208, yb + 5, (cmd["cap"]["rate_text"] or "n/a") + " ou mais", "tk")
    cv.y = yb
    ns = [c["n"]["value"] for c in rep]
    cv.notes([
        f"n por célula com valor: {pt_int(min(ns))} a {pt_int(max(ns))} objetos ({len(rep)} células)" if ns
        else "Nenhuma célula com valor",
        (f"Fora da área do diagrama: {out['text']} objetos ({pt_pct(out['value'] / n_test, 1)} do teste), "
        "sem taxa nesta figura"),
        cmd["note"],
    ])
    cv.provisional(g["model_label"], rel, f"Piso de {pt_int(g['floor'])} objetos por célula")
    return _done(cv, sub, desc, rel)


# 5 ---------------------------------------------------------------- horizontal bars with CI
@dataclass(frozen=True)
class _Bar:
    name: str
    est: Mapping[str, Any]
    bold: bool = False
    partial: bool = False
    extra: str = ""


def _hbars(cv: Canvas, bars: list[_Bar], top: float, axis: str) -> None:
    pw, rh = 300, 56

    def x_(v: float) -> float:
        return M + min(max(v, 0), top) / top * pw

    y0 = cv.y + 26
    bottom = y0 + rh * len(bars)
    ticks, step = nice_ticks(0, top, 4)
    dec = decimals_for(step * 100)
    for t in ticks:
        cv.line(x_(t), y0, x_(t), bottom, "grid")
        cv.text(x_(t), bottom + 22, pt_pct(t, dec), "tk", "middle")
    cv.line(M, bottom, M + pw, bottom, "axis")
    for i, b in enumerate(bars):
        y = y0 + i * rh + 18
        cv.text(M, y, b.name + (f" · {b.extra}" if b.extra else ""), "nm" if b.bold else "s")
        cv.text(W - M, y, _ci(b.est["ci_text"]), "n", "end")
        e = b.est
        if e["point"] is None:
            cv.text(M, y + 22, e["text"], "n")
            continue
        if b.bold:
            cv.rect(M - 4, y - 16, W - 2 * M + 8, rh - 4, edge=TEXT2, lw=1)
        if b.partial:
            cv.rect(M, y + 9, x_(e["point"]) - M, 14, edge=SERIES[1], lw=1.5, dashes=(4, 3))
        else:
            cv.rect(M, y + 9, x_(e["point"]) - M, 14, fill=SERIES[2])
        if e["lo"] is not None:
            cv.line(x_(e["lo"]), y + 16, x_(e["hi"]), y + 16, "cil")
        vx = max(x_(e["point"]), x_(e["hi"] if e["hi"] is not None else 0)) + 8
        cv.text(vx, y + 21, e["text"], "val")
    cv.text(M + pw / 2, bottom + 46, axis, "s", "middle")
    cv.y = bottom + 46


def _quasar(pick: str, fid: str, title: str, axis: str) -> Callable[[Pages], Figure]:
    def build(p: Pages) -> Figure:
        o, meta, g = p["overview"], p["meta"], p["graficos"]
        rel = meta["release"]
        st = {(s["rule"], s["prior"]): s["label"] for s in meta["states"]}
        ds = meta["default_state"]
        sub = f"{g['model_label']} · conjunto de teste · por regra e prior"
        cv = Canvas(fid, title, sub)
        bars = [
            _Bar(st[(r["rule"], r["prior"])] + (" (parcial)" if r["partial"] else ""), r[pick],
                 bold=(r["rule"], r["prior"]) == (ds["rule"], ds["prior"]), partial=r["partial"])
            for r in o["rows"]
        ]
        _hbars(cv, bars, 1.0, axis)
        cv.notes([
            f"n = {o['technical']['data_box']['n_test']['text']} objetos no teste",
            f"Traço sobre a barra: IC 95% (bootstrap, {pt_int(meta['n_boot'])} reamostragens)",
            f"Contorno: estado padrão ({st[(ds['rule'], ds['prior'])]}) · barra tracejada: parcial",
            *[r["partial_notice"] for r in o["rows"] if r["partial_notice"]],
        ])
        cv.provisional(g["model_label"], rel, None)
        desc = "; ".join(f"{b.name}: {b.est['text']}, {b.est['ci_text']}" for b in bars)
        return _done(cv, sub, desc, rel)

    return build


def galaxy_errors(p: Pages) -> Figure:
    ga, meta, g = p["galaxies"], p["meta"], p["graficos"]
    rel = meta["release"]
    rows = ga["error_rows"]
    sub = (f"{g['model_label']} · classe de maior probabilidade "
           f"(regra {meta['rule_labels']['as_shipped']})")
    cv = Canvas("galaxia-erros", "Taxa de erro por grupo de galáxias", sub)
    bars = [_Bar(r["label"], r["rate"], extra=f"n = {r['n']['text']}") for r in rows]
    _hbars(cv, bars, _top([r["rate"]["hi"] or 0 for r in rows]), "Taxa de erro (%)")
    cv.notes([
        "Traço sobre a barra: IC 95% · n: galáxias no grupo",
        ga["error_rows_note"],
        ga["floor_note"],
    ])
    cv.provisional(g["model_label"], rel, None)
    desc = "; ".join(f"{r['label']}: {r['rate']['text']}, {r['rate']['ci_text']}, n = {r['n']['text']}"
                     for r in rows)
    return _done(cv, sub, desc, rel)


# 6, 10 ----------------------------------------------------------- interval rows
@dataclass(frozen=True)
class _Row:
    name: str
    lo: float
    hi: float
    point: float
    value: str
    detail: str
    band: float | None = None
    bold: bool = False


def _intervals(cv: Canvas, rows: list[_Row], fmt: Callable[[float, int], str], axis: str,
               zero: bool, pad_frac: float = 0.08) -> None:
    pxl, pw = 24, 266
    bounds = [v for r in rows for v in (r.lo, r.hi)]
    bounds += [v for r in rows if r.band is not None for v in (-r.band, r.band)]
    if zero:
        bounds.append(0.0)
    lo, hi = min(bounds), max(bounds)
    pad = (hi - lo) * pad_frac
    ticks, step = nice_ticks(lo - pad, hi + pad, 4)

    def x_(v: float) -> float:
        return pxl + (v - ticks[0]) / (ticks[-1] - ticks[0]) * pw

    dec = decimals_for(step)
    top = cv.y + 16
    y = top
    placed = []
    for r in rows:
        y += 22
        for k, ln in enumerate(wrap(r.name, W - 2 * M, bold=r.bold)):
            if k:
                y += LINE
            cv.text(M, y, ln, "nm" if r.bold else "s")
        placed.append((r, y + 22))
        y += 48
    bottom = y + 8
    for t in ticks:
        cv.line(x_(t), top, x_(t), bottom, "grid")
        cv.text(x_(t), bottom + 22, fmt(t, dec), "tk", "middle")
    for r, iy in placed:
        if r.band is not None:
            cv.rect(x_(-r.band), iy - 12, x_(r.band) - x_(-r.band), 24, fill=SERIES[2], alpha=0.35)
    if zero:
        cv.line(x_(0), top, x_(0), bottom, "ref")
    cv.line(pxl, bottom, pxl + pw, bottom, "axis")
    for r, iy in placed:
        cv.line(x_(r.lo), iy, x_(r.hi), iy, "rng")
        cv.line(x_(r.point), iy - 8, x_(r.point), iy + 8, "est")
        cv.text(pxl + pw + 10, iy + 5, r.value, "val")
        cv.text(pxl + pw + 10, iy + 25, r.detail, "n")
    cv.text(pxl + pw / 2, bottom + 46, axis, "s", "middle")
    cv.y = bottom + 46


def dsc_diffs(p: Pages) -> Figure:
    o, rel = p["overview"], p["meta"]["release"]
    dsc = o["technical"]["dsc"]
    pop = dsc.get("population_text") or f"população {dsc['population']}"
    prior = dsc.get("prior_text") or f"prior {dsc['prior']}"
    sub = f"{pop} · {prior}"
    cv = Canvas("dsc-diferencas", "Diferenças pareadas da medida F macro", sub)
    rows = [
        _Row(f"{d['a']} − {d['b']}", d["est"]["lo"], d["est"]["hi"], d["est"]["point"],
             d["est"]["text"], _ci(d["est"]["ci_text"]))
        for d in dsc["paired_diffs"] if d["est"]["point"] is not None and d["est"]["lo"] is not None
    ]
    _intervals(cv, rows, lambda v, d: pt_dec(v, d, sign=True), "Diferença de F macro (escala 0 a 1)", zero=True)
    cv.notes([
        f"n = {dsc['n_rows']['text']} objetos pareados",
        "Traço horizontal: IC 95% pareado · traço vertical: estimativa",
        "Tracejado vertical: diferença zero",
    ])
    cv.provisional("LightGBM e Gaia DSC", rel, None)
    desc = "; ".join(f"{d['a']} − {d['b']}: {d['est']['text']}, {d['est']['ci_text']}" for d in dsc["paired_diffs"])
    return _done(cv, sub, desc, rel)


def _pp(v: float, d: int = 2) -> str:
    return pt_dec(v * 100, d, sign=True)


def decision_diffs(p: Pages) -> Figure:
    c, rel = p["compare"], p["meta"]["release"]
    sub = c["fixed_point"]["text"]
    cv = Canvas("decisao-diferencas", "Diferenças nas células da decisão", sub)
    desc = []
    for cls_label in dict.fromkeys(gr["cls_label"] for gr in c["grid"]):
        cells = [gr for gr in c["grid"] if gr["cls_label"] == cls_label]
        cv.y += 30
        cv.text(M, cv.y, f"{cls_label} · escala própria", "nm")
        rows = [
            _Row(f"{gr['net']} − LightGBM · {gr['word']}", gr["lo"] * 100, gr["hi"] * 100, gr["d"] * 100,
                 f"{_pp(gr['d'])} p.p.", f"{_pp(gr['lo'])} a {_pp(gr['hi'])}", band=gr["delta"] * 100)
            for gr in cells
        ]
        _intervals(cv, rows, lambda v, d: pt_dec(v, d, sign=True), "Diferença de pureza (p.p.)", zero=True)
        cv.y += 4
        cv.text(M, cv.y + 22, f"Faixa cinza: ±δ = {pt_dec(cells[0]['delta'] * 100, 2)} p.p.", "n")
        cv.y += 22
        desc += [f"{gr['net']} − LightGBM, {cls_label}: {_pp(gr['d'])} p.p. (IC 95% {_pp(gr['lo'])} a "
                 f"{_pp(gr['hi'])}), δ {pt_dec(gr['delta'] * 100, 2)} p.p., {gr['word']}" for gr in cells]
    metric = next((ln for ln in c["rule_lines"] if ln.startswith("Métrica")), None)
    under = any(gr["underpowered"] for gr in c["grid"])
    cv.notes([
        "Traço horizontal: IC 95% · traço vertical: diferença D · tracejado: zero",
        *([metric] if metric else []),
        *([f"Galáxia: {c['power_note']}"] if under else []),
    ])
    cv.provisional("LightGBM, MLP e FT-Transformer", rel, None)
    return _done(cv, sub, "; ".join(desc), rel)


# 7 ---------------------------------------------------------------- feature gain
def _importance(index: int) -> Callable[[Pages], Figure]:
    def build(p: Pages) -> Figure:
        rel = p["meta"]["release"]
        m = p["overview"]["technical"]["models"][index]
        imp = m["importance"]
        sub = "Participação de cada atributo no ganho total do modelo"
        cv = Canvas(f"ganho-{m['model'].replace('_', '-')}", f"Ganho por atributo: {m['label']}", sub)
        pw, rh = 320, 42
        ticks, step = nice_ticks(0, max(i["gain"]["value"] for i in imp), 3)

        def x_(v: float) -> float:
            return M + v / ticks[-1] * pw

        top = cv.y + 20
        bottom = top + rh * len(imp)
        for t in ticks:
            cv.line(x_(t), top, x_(t), bottom, "grid")
            cv.text(x_(t), bottom + 22, pt_pct(t, decimals_for(step * 100)), "tk", "middle")
        cv.line(M, bottom, M + pw, bottom, "axis")
        for i, r in enumerate(imp):
            y = top + i * rh + 16
            cv.text(M, y, r["feature"], "s")
            cv.rect(M, y + 7, x_(r["gain"]["value"]) - M, 12, fill=SERIES[0])
            cv.text(x_(r["gain"]["value"]) + 8, y + 18, r["gain"]["text"], "val")
        cv.text(M + pw / 2, bottom + 46, "Ganho (% do ganho total)", "s", "middle")
        cv.y = bottom + 46
        cv.notes([
            f"{len(imp)} maiores de {m['n_features']} atributos · melhor iteração {pt_int(m['best_iteration'])}",
            "Sem IC: ganho do modelo treinado, não é estimativa amostral",
            f"Temperatura de calibração: {m['temperature']['text']}",
        ])
        cv.provisional(m["label"], rel, None)
        return _done(cv, sub, "; ".join(f"{r['feature']}: {r['gain']['text']}" for r in imp), rel)

    return build


# 10 --------------------------------------------------------------- absolute purity
def purity_range(p: Pages) -> Figure:
    c, rel = p["compare"], p["meta"]["release"]
    pur = c["purity"]
    sub = c["fixed_point"]["text"]
    cv = Canvas("pureza-sistemas", "Pureza absoluta por sistema, no ponto fixo", sub)
    rows = [
        _Row(r["label"] + (" · referência" if r["reference"] else ""), r["low"], r["high"], r["median"],
             pt_pct(r["median"], 2), f"{pt_pct(r['low'], 2)} a {pt_pct(r['high'], 2)}", bold=r["reference"])
        for r in pur
    ]
    _intervals(cv, rows, lambda v, d: pt_pct(v, max(d - 2, 0)),
               "Pureza absoluta (%)", zero=False, pad_frac=0.15)
    cv.notes([
        "Atenção: o eixo não começa em zero",
        "Traço horizontal: mínimo a máximo entre sementes (não é IC) · traço vertical: mediana",
        *[f"{r['label']}: {r['detail_text']}" for r in pur],
    ])
    cv.provisional("LightGBM, MLP e FT-Transformer", rel, None)
    return _done(cv, sub, "; ".join(f"{r['label']}: {r['detail_text']}" for r in pur), rel)


def _models(p: Pages) -> list[Callable[[Pages], Figure]]:
    return [_importance(i) for i in range(len(p["overview"]["technical"]["models"]))]


def builders(p: Pages) -> list[Callable[[Pages], Figure]]:
    """Builders of every MUDAR figure (02-graficos-veredito.md), in page order."""
    zs = [_rate_by_z(s["cls"], f"taxa-por-z-{s['cls'].lower()}") for s in p["graficos"]["error_by_z"]["series"]]
    return [
        calibration, rate_by_g, *zs, confusion, cmd_map,
        _quasar("purity", "quasar-pureza", "Pureza de quasar por regra e prior", "Pureza de quasar (%)"),
        _quasar("completeness", "quasar-completude", "Completude de quasar por regra e prior",
                "Completude de quasar (%)"),
        galaxy_errors, dsc_diffs, decision_diffs, *_models(p), purity_range,
    ]


__all__ = ["Figure", "Pages", "builders"]
