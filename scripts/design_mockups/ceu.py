"""Screen 2, 'Céu': Mollweide census of the regions, built from per-region counts only."""

from __future__ import annotations

import math

import numpy as np

from astro_triage.dashboard import fmt, theme
from astro_triage.dashboard.data_sky import HEALPIX_NSIDE, mollweide

from . import loaders
from .kit import MONO, TABLE_CSS, Ledger, e
from .loaders import Cell

SCREEN = "ceu"
TITLE = "Céu"
PER_DOT = 25  # design parameter of the spec: objects per mark
JITTER_SEED = 20261025
SUB_LEVEL = 4  # each dot picks one of 4**SUB_LEVEL sub-pixels of its own pixel, then an offset inside it
W, CY, SC = 1000.0, 255.0, 500.0 / (2 * math.sqrt(2))
HIST_BASE, HIST_MAX, HEIGHT = 612.0, 72.0, 618.0
BIN_DEG = 2
RING_R = 2.6
ERR_CAP_Q = 0.95  # as the dashboard map: colour cap at the 95th percentile of rates
PARAM_FILE = "scripts/design_mockups/ceu.py"

CSS = (
    TABLE_CSS
    + f"""
.tgi{{position:absolute;opacity:0}}
.tgl{{display:flex;gap:8px;margin:0 0 16px}}
.tgl label{{border:1px solid {theme.HAIRLINE};padding:10px 14px;min-height:{theme.MIN_TARGET_PX}px;font-size:{theme.CONTROL_PX}px;color:{theme.SOFT};cursor:pointer}}
#lyc:checked ~ .tgl label[for=lyc],#lye:checked ~ .tgl label[for=lye]{{border-color:{theme.ACCENT};color:{theme.ACCENT}}}
#lyc:focus-visible ~ .tgl label[for=lyc],#lye:focus-visible ~ .tgl label[for=lye]{{outline:2px solid {theme.ACCENT}}}
.L-e{{display:none}}
#lye:checked ~ .sky-scroll .L-e,#lye:checked ~ .legend .L-e{{display:inline}}
#lye:checked ~ .sky-scroll .L-c,#lye:checked ~ .legend .L-c{{display:none}}
#lye:checked ~ .sky-scroll .L-dim{{opacity:.35}}
.sky-scroll{{overflow-x:auto}}
.sky{{min-width:640px;position:relative}}
.sky svg{{display:block;width:100%;height:auto}}
.ticks{{position:relative;height:28px;font:{theme.MIN_SECONDARY_PX}px {MONO};color:{theme.TEXT_SECONDARY}}}
.ticks span{{position:absolute;top:4px;transform:translateX(-50%);white-space:nowrap}}
.ticks span.l{{transform:none}}.ticks span.r{{transform:translateX(-100%)}}
.legend{{margin-top:12px;display:flex;flex-wrap:wrap;gap:8px 24px;align-items:center}}
.legend .sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:baseline}}
.legend .ring{{background:transparent;box-shadow:inset 0 0 0 1px {theme.SOFT}}}
.nums{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px 32px;margin-top:32px}}
.nums .v{{font:400 40px/1.1 {theme.FONT_SERIF}}}
.nums .v.acc{{color:{theme.ACCENT}}}
"""
)


def _xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = mollweide(lon, lat)
    return W / 2 + x * SC, CY - y * SC


def _path(xs: np.ndarray, ys: np.ndarray) -> str:
    return "".join(f"M{x:.1f} {y:.1f}h0" for x, y in zip(xs, ys, strict=True))


def _line(lon: np.ndarray, lat: np.ndarray) -> str:
    xs, ys = _xy(lon, lat)
    return "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in zip(xs, ys, strict=True))


def _graticule() -> str:
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    parts = []
    for deg in range(-150, 151, 30):
        d = _line(np.full_like(lat, deg), lat)
        parts.append(f"<path d='{d}' stroke='{theme.DASH if deg == 0 else theme.HAIRLINE}'/>")
    for deg in (-60, -30, 0, 30, 60):
        d = _line(lon, np.full_like(lon, deg))
        parts.append(f"<path d='{d}' stroke='{theme.DASH if deg == 0 else theme.HAIRLINE}'/>")
    right = _line(np.full_like(lat, 179.999), lat)
    left = _line(np.full_like(lat, -179.999), lat[::-1])
    parts.append(f"<path d='{right}{left[left.index('L'):]}Z' stroke='{theme.DASH}'/>")
    return f"<g fill='none' stroke-width='.8'>{''.join(parts)}</g>"


def _cross() -> str:
    cx, cy = _xy(np.array([0.0]), np.array([0.0]))
    cx, cy = float(cx[0]), float(cy[0])
    arms = "".join(
        f"<path d='M{cx + a:.1f} {cy + b:.1f}L{cx + c:.1f} {cy + d:.1f}'/>"
        for a, b, c, d in ((-15, 0, -10, 0), (10, 0, 15, 0), (0, -15, 0, -10), (0, 10, 0, 15))
    )
    return (
        f"<g fill='none' stroke='{theme.TEXT}' stroke-width='.7'>"
        f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='8'/><circle cx='{cx:.1f}' cy='{cy:.1f}' r='3.5'/>"
        f"{arms}</g>"
    )


def _scatter(frame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One mark per PER_DOT objects (at least one per region), placed uniformly inside the real pixel."""
    from astropy.coordinates import SkyCoord
    from astropy_healpix import HEALPix

    n_dots = np.maximum(1, np.rint(frame["n"].to_numpy() / PER_DOT).astype(int))
    idx = np.repeat(np.arange(len(frame)), n_dots)
    parent = frame["healpix_l5"].to_numpy().astype(np.int64)[idx]
    child = parent * 4**SUB_LEVEL + rng.integers(0, 4**SUB_LEVEL, len(idx))
    dx, dy = rng.uniform(size=(2, len(idx)))
    grid = HEALPix(nside=HEALPIX_NSIDE * 2**SUB_LEVEL, order="nested")
    lon, lat = grid.healpix_to_lonlat(child, dx, dy)
    gal = SkyCoord(lon, lat, frame="icrs").galactic
    return _xy(((gal.l.deg + 180) % 360) - 180, gal.b.deg)


def _dots(xs: np.ndarray, ys: np.ndarray, rng, colour: str, opacities, width: float) -> str:
    bucket = rng.integers(0, len(opacities), len(xs))
    return "".join(
        f"<path d='{_path(xs[bucket == k], ys[bucket == k])}' stroke='{colour}' "
        f"stroke-width='{width}' stroke-linecap='round' opacity='{op}'/>"
        for k, op in enumerate(opacities)
    )


def _error_layer(test, floor: int, cap: float) -> str:
    xs, ys = _xy(
        ((test["l_centre"].to_numpy() + 180) % 360) - 180, test["b_centre"].to_numpy()
    )
    out = []
    for x, y, n, rate in zip(xs, ys, test["n"], test["rate"], strict=True):
        if n < floor:
            out.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{RING_R}' fill='none' "
                f"stroke='{theme.SOFT}' stroke-width='.7'/>"
            )
        else:
            op = 0.15 + 0.85 * min(rate / cap, 1.0)
            out.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{RING_R}' fill='{theme.ACCENT}' "
                f"opacity='{op:.2f}'/>"
            )
    return "".join(out)


def _histogram(df) -> tuple[str, float]:
    lw = ((df["l_centre"].to_numpy() + 180) % 360) - 180
    bins = np.floor((lw + 180) / BIN_DEG).astype(int).clip(0, 360 // BIN_DEG - 1)
    n_bins = 360 // BIN_DEG
    tot, tst = np.zeros(n_bins), np.zeros(n_bins)
    is_test = (df["fold"] == "test").to_numpy()
    np.add.at(tot, bins, df["n"].to_numpy())
    np.add.at(tst, bins[is_test], df["n"].to_numpy()[is_test])
    unit = HIST_MAX / tot.max()
    bar_w = W * BIN_DEG / 360
    rects = []
    for i in range(n_bins):
        l_hi = -180 + (i + 1) * BIN_DEG
        x0 = (180 - l_hi) / 360 * W + 0.5
        h_t, h_o = tst[i] * unit, (tot[i] - tst[i]) * unit
        if h_o > 0:
            rects.append(
                f"<rect x='{x0:.1f}' y='{HIST_BASE - h_t - h_o:.1f}' width='{bar_w - 1:.1f}' "
                f"height='{h_o:.1f}' fill='{theme.TEXT_SECONDARY}' opacity='.55'/>"
            )
        if h_t > 0:
            rects.append(
                f"<rect x='{x0:.1f}' y='{HIST_BASE - h_t:.1f}' width='{bar_w - 1:.1f}' "
                f"height='{h_t:.1f}' fill='{theme.ACCENT}'/>"
            )
    base = f"<path d='M0 {HIST_BASE}H{W}' stroke='{theme.DASH}' stroke-width='.8'/>"
    return base + "".join(rects), float(tot.max())


def _ticks() -> str:
    spans = []
    for deg in range(180, -181, -30):
        pos = (180 - deg) / 360 * 100
        cls = " class='l'" if deg == 180 else (" class='r'" if deg == -180 else "")
        spans.append(f"<span{cls} style='left:{pos:.2f}%'>{deg % 360}°</span>")
    return f"<div class='ticks' aria-hidden='true'>{''.join(spans)}</div>"


def _check(name: str, got: int, expected: Cell) -> None:
    if not expected.ok or got != expected.value:
        raise ValueError(f"céu: {name} = {got}, esperado {expected.value} ({expected.file})")


def render(rel, L: Ledger) -> tuple[str, str, dict]:
    df = loaders.sky_regions(rel)
    cells = loaders.dataset_cells(rel)
    cen = loaders.census(rel)
    floor = int(rel.meta["floor"]["min_objects"])
    test = df[df["fold"] == "test"].copy()
    _check("soma dos objetos", int(df["n"].sum()), cells["final_rows"])
    _check("regiões de teste", len(test), cen["regions"])
    _check("objetos de teste", int(test["n"].sum()), cen["objects"])
    _check("regiões de teste", len(test), cells["test_pixels_total"])
    if not np.array_equal(test["n"].to_numpy(), test["n_scored"].to_numpy()):
        raise ValueError("céu: objetos de teste sem escore calibrado")
    test["rate"] = test["errors"] / test["n"]
    ok = test["n"] >= floor
    _check(
        f"regiões de teste com pelo menos {floor} objetos",
        int(ok.sum()),
        cells["test_pixels_with_at_least_100_objects"],
    )
    cap = float(np.quantile(test.loc[ok, "rate"], ERR_CAP_Q))

    rng = np.random.default_rng(JITTER_SEED)
    other = df[df["fold"] != "test"]
    gx, gy = _scatter(other, rng)
    bx, by = _scatter(test, rng)
    hist, hist_max = _histogram(df)
    svg = (
        f"<svg viewBox='0 0 {W:.0f} {HEIGHT:.0f}' role='img' aria-label='Mapa Mollweide em "
        "coordenadas galácticas: cada ponto representa um grupo de objetos; teste em azul, "
        "treino e validação em cinza' xmlns='http://www.w3.org/2000/svg'>"
        f"{_graticule()}<g class='L-dim'>{_dots(gx, gy, rng, theme.TEXT_SECONDARY, (0.35, 0.6, 0.85), 1.5)}</g>"
        f"<g class='L-c'>{_dots(bx, by, rng, theme.ACCENT, (0.85, 1), 1.7)}</g>"
        f"<g class='L-e'>{_error_layer(test, floor, cap)}</g>"
        f"{_cross()}{hist}</svg>"
    )

    n_test = L.integer(cen["objects"], "céu · objetos de teste")
    n_reg = L.integer(cen["regions"], "céu · regiões de teste")
    per_fold = {
        f: L.integer(
            Cell(int(df.loc[df["fold"] == f, "n"].sum()), loaders.LABELS_FILE, f"count(*) where fold = {f}"),
            f"céu · objetos de {f}",
        )
        for f in ("train", "val")
    }
    n_other_reg = L.integer(
        Cell(len(other), loaders.LABELS_FILE, "count(distinct healpix_l5) where fold != test"),
        "céu · regiões de treino e validação",
    )
    pdot = L.integer(Cell(PER_DOT, PARAM_FILE, "PER_DOT"), "céu · objetos por ponto")
    dots = {
        "grey": len(gx),
        "blue": len(bx),
    }
    n_grey = L.derived(
        fmt.pt_int(dots["grey"]),
        "céu · pontos cinza",
        Cell(len(gx), loaders.LABELS_FILE, "sum(max(1, round(n / PER_DOT))) por região"),
        "um ponto por PER_DOT objetos, ao menos um por região",
    )
    n_blue = L.derived(
        fmt.pt_int(dots["blue"]),
        "céu · pontos azuis",
        Cell(len(bx), loaders.LABELS_FILE, "sum(max(1, round(n / PER_DOT))) por região de teste"),
        "um ponto por PER_DOT objetos, ao menos um por região",
    )
    cap_txt = L.derived(
        fmt.pt_pct(cap, 1),
        "céu · teto da escala de erro",
        Cell(cap, loaders.LABELS_FILE, "taxa de erro por região de teste, percentil 95"),
        f"percentil {int(ERR_CAP_Q * 100)} das regiões com pelo menos {floor} objetos",
    )
    n_ring = L.derived(
        fmt.pt_int(int((~ok).sum())),
        "céu · regiões de teste abaixo do piso (anel n/a)",
        cells["test_pixels_total"],
        "regiões de teste menos regiões com o mínimo de objetos",
    )
    floor_txt = L.integer(
        Cell(floor, loaders.RESULTS_FILE, "meta.floor.min_objects"),
        "céu · piso de objetos por região",
    )
    hist_txt = L.integer(
        Cell(int(hist_max), loaders.LABELS_FILE, "max por faixa de longitude de 2 graus"),
        "céu · faixa de longitude mais cheia (objetos)",
    )
    reg_total = L.integer(
        Cell(len(df), loaders.LABELS_FILE, "count(distinct healpix_l5)"),
        "céu · regiões com objetos",
    )

    body = (
        f"<p class='lead'>O céu que o modelo nunca viu: {e(n_reg)} regiões reservadas para teste.</p>"
        "<input class='tgi' type='radio' name='ly' id='lyc' checked aria-label='Contagem'>"
        "<input class='tgi' type='radio' name='ly' id='lye' aria-label='Erro'>"
        "<div class='tgl'><label for='lyc'>Contagem</label><label for='lye'>Erro</label></div>"
        f"<div class='sky-scroll'><div class='sky'>{svg}{_ticks()}</div></div>"
        "<div class='legend sm'>"
        "<span class='L-c'><span class='sw' style='background:" + theme.ACCENT + "'></span>teste</span>"
        "<span class='L-c'><span class='sw' style='background:" + theme.TEXT_SECONDARY + ";opacity:.7'></span>treino e validação</span>"
        f"<span class='L-c mono'>1 ponto ≈ {e(pdot)} objetos</span>"
        "<span class='L-e'><span class='sw' style='background:" + theme.ACCENT + "'></span>"
        f"teste: intensidade do azul = taxa de erro, até {e(cap_txt)}</span>"
        f"<span class='L-e'><span class='sw ring'></span>n/a: menos de {e(floor_txt)} objetos na região</span>"
        "</div>"
        "<p class='sm'>Histograma abaixo do mapa: objetos por faixa de longitude de "
        f"{BIN_DEG}°, azul = teste, cinza = treino e validação; a faixa mais cheia tem {e(hist_txt)} "
        "objetos. Cruz de registro em l = 0°, b = 0°. Longitude cresce para a esquerda.</p>"
        "<div class='nums'>"
        f"<div><div class='v acc'>{e(n_reg)}</div><div>regiões de teste</div>"
        f"<div class='sm mono'>{e(n_test)} objetos</div></div>"
        f"<div><div class='v'>{e(n_other_reg)}</div><div>regiões de treino e validação</div>"
        f"<div class='sm mono'>{e(per_fold['train'])} de treino · {e(per_fold['val'])} de validação</div></div>"
        f"<div><div class='v'>{e(n_ring)}</div><div>regiões de teste abaixo do piso</div>"
        "<div class='sm mono'>mostradas como anel n/a na camada Erro</div></div>"
        "</div>"
    )
    footer = (
        "Posição dentro do pixel é sintética: cada ponto é espalhado de forma determinística "
        f"(semente {e(L.text(Cell(JITTER_SEED, PARAM_FILE, 'JITTER_SEED'), 'céu · semente do jitter'))}) dentro da região de nível 5 "
        f"({e(reg_total)} regiões com objetos); só as contagens por região são dado. "
        f"Pontos: {e(n_blue)} azuis e {e(n_grey)} cinza. A taxa de erro é a do LightGBM como entregue. "
        "Nota: regiões vizinhas de treino não são removidas (ADR 0003)."
    )
    facts = {
        "regions": len(df),
        "objects": int(df["n"].sum()),
        "test_regions": len(test),
        "test_objects": int(test["n"].sum()),
        "test_regions_at_floor": int(ok.sum()),
        "dots": dots,
        "error_cap": round(cap, 5),
    }
    return body, footer, facts
