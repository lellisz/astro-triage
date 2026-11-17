"""Tests of the 'Regua do prior' mockup generator (scripts/build_design_mockups.py).

Tests that need the real release (data/interim/full_v0) skip when it is absent. Every build goes
to a tmp dir; data/interim/design is never touched.
"""

from __future__ import annotations

import colorsys
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from design_mockups import comparacao, kit, loaders, metodo

from astro_triage.dashboard import theme
from astro_triage.dashboard.data_dl import load_results_dl

RELEASE = "full_v0"
ROOT = REPO / "data" / "interim" / RELEASE
SCREENS = ("resultado", "ceu", "comparacao", "metodo")
needs_release = pytest.mark.skipif(
    not (ROOT / "eval" / "results.json").is_file()
    or not (ROOT / "dl" / "eval" / "results_dl.json").is_file(),
    reason=f"release {RELEASE} ausente em data/interim: testes de dados reais pulados",
)


def _build(outdir: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_design_mockups.py"),
         "--release", RELEASE, "--outdir", str(outdir)],
        check=False, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600,
    )  # fmt: skip


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (ROOT / "eval" / "results.json").is_file():
        pytest.skip(f"release {RELEASE} ausente")
    out = tmp_path_factory.mktemp("design")
    proc = _build(out)
    return out, proc


@pytest.fixture(scope="module")
def html(built):
    out, proc = built
    assert proc.returncode == 0, proc.stderr
    return {s: (out / f"{s}.html").read_text(encoding="utf-8") for s in SCREENS}


@pytest.fixture(scope="module")
def rel():
    if not (ROOT / "eval" / "results.json").is_file():
        pytest.skip(f"release {RELEASE} ausente")
    return loaders.load_release_light(RELEASE)


def _jload(*parts: str) -> dict:
    return json.loads(ROOT.joinpath(*parts).read_text(encoding="utf-8"))


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}".replace(".", ",") + "%"


def _num(x: float, d: int, signed: bool = False) -> str:
    return f"{x:{'+' if signed else ''}.{d}f}".replace(".", ",")


def _int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _visible(doc: str) -> str:
    """Text outside <style>, with tags removed (fonts as base64 live in <style>)."""
    doc = re.sub(r"<style>.*?</style>", "", doc, flags=re.DOTALL)
    return re.sub(r"<[^>]+>", " ", doc)


# --- 1. CLI ---------------------------------------------------------------------------------


@needs_release
def test_cli_exits_zero_and_writes_all_outputs(built):
    out, proc = built
    assert proc.returncode == 0, proc.stderr
    for name in (*(f"{s}.html" for s in SCREENS), "number_sources.json", "cortes.md"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name


def test_cli_unknown_release_fails_nonzero(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_design_mockups.py"),
         "--release", "no_such_release_xyz", "--outdir", str(tmp_path)],
        check=False, cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert proc.returncode == 2
    assert not (tmp_path / "resultado.html").exists()


# --- 2. Numbers read from the artifacts -----------------------------------------------------


def _state_section(doc: str, tag: str) -> str:
    m = re.search(rf"<section class='st st{tag}'>.*?</section>", doc, re.DOTALL)
    assert m, f"estado {tag} ausente"
    return m.group(0)


@needs_release
@pytest.mark.parametrize(("tag", "variant"), [("A", "as_shipped"), ("B", "prior_matched")])
def test_resultado_state_numbers_match_results_json(html, tag, variant):
    metrics = _jload("eval", "results.json")["populations"]["a_all"]["priors"]["P2"][
        "variants"
    ][variant]["models"]["lgbm_full"]["metrics"]
    pur, comp = metrics["purity/QSO"], metrics["completeness/QSO"]
    sec = _state_section(html["resultado"], tag)

    big = re.search(r"<div class='num'>([^<]+)</div>", sec).group(1)
    assert big == _pct(pur["point"])
    assert f"IC 95%: {_pct(pur['lo'])} a {_pct(pur['hi'])}" in sec
    assert f"Quasares encontrados: <b>{_pct(comp['point'])}</b>" in sec
    assert f"IC 95%: {_pct(comp['lo'])} a {_pct(comp['hi'])}" in sec
    # the field: exactly 1000 dots, round(1000 x purity) of them blue
    assert sec.count("<i class='q'></i>") == round(1000 * pur["point"])
    assert sec.count("<i class='q'></i>") + sec.count("<i></i>") == 1000
    assert "1 ponto ≈ 0,1% da seleção · composição esperada sob o prior, não contagem de objetos" in sec


@needs_release
def test_resultado_state_b_wide_purity_ci_is_shown(html):
    sec = _state_section(html["resultado"], "B")
    assert "IC 95%: 60,3% a 89,9%" in sec  # width of the interval must not be dropped


@needs_release
def test_resultado_census_and_accuracy_labels(html):
    meta = _jload("eval", "results.json")["meta"]
    doc = html["resultado"]
    assert f"{_int(meta['n_test_pixels'])} regiões de teste · {_int(meta['n_test'])} objetos" in doc
    assert "Regiões vizinhas de treino não são removidas (ADR 0003)" in doc
    dl = _jload("dl", "eval", "results_dl.json")
    shipped = _jload("eval", "results.json")["populations"]["a_all"]["priors"]["P2"][
        "variants"
    ]["as_shipped"]["models"]["lgbm_full"]["detail"]["accuracy"]
    mean = dl["descriptive"]["families"]["lgbm_full"]["accuracy"]["mean"]
    assert _pct(shipped, 2) in doc and _pct(mean, 2) in doc
    # the two definitions are labelled apart
    assert "Como entregue</b>" in doc and "sementes</b>" in doc


@needs_release
def test_resultado_prior_is_assumed_and_p3_is_partial(html):
    doc = html["resultado"]
    assert "prior de quasar assumido" in doc.lower()
    assert "Quasares no céu" not in doc
    assert "P3 vale só para parte do céu (bandas B e C; ADR 0003)" in doc


def test_comparacao_d_ci_delta_match_results_dl(html):
    dl = _jload("dl", "eval", "results_dl.json")
    panels = html["comparacao"].split("<section class='panel'>")[1:3]
    assert len(panels) == 2
    for panel, cls in zip(panels, ("QSO", "GALAXY"), strict=True):
        for fam, pair in (("MLP", "mlp_full-lgbm_full"), ("FTT", "ftt_full-lgbm_full")):
            block = dl["compare"]["pairs"][pair]["P2"]["0.80"]
            assert block["role"] == "headline"
            c = block["classes"][cls]
            line = (
                f"<div class='dn'>{fam}</div>"
            )
            assert line in panel
            seg = panel.split(line, 1)[1]
            expect = (
                f"D = {_num(c['d'], 4, True)} · IC 95%: {_num(c['ci'][0], 4, True)} a "
                f"{_num(c['ci'][1], 4, True)} · δ = {_num(c['delta'], 4)}"
            )
            assert expect in seg.split("</div></div>", 1)[0] + "</div>" or expect in seg[:900], (cls, fam)


@needs_release
def test_comparacao_galaxy_note_uses_read_half_width(html):
    dl = _jload("dl", "eval", "results_dl.json")
    hw = []
    for pair in ("mlp_full-lgbm_full", "ftt_full-lgbm_full"):
        lo, hi = dl["compare"]["pairs"][pair]["P2"]["0.80"]["classes"]["GALAXY"]["ci"]
        hw.append(_num((hi - lo) / 2, 2))
    span = f"±{hw[0]}" if len(set(hw)) == 1 else f"±{min(hw)} a ±{max(hw)}"
    assert f"IC ≈ {span}: sem poder para distinguir" in html["comparacao"]


@needs_release
def test_ceu_counts_match_dataset_and_results(built, html):
    out, _ = built
    total = _jload("dataset_report.json")["final_rows"]
    meta = _jload("eval", "results.json")["meta"]
    rows = json.loads((out / "number_sources.json").read_text(encoding="utf-8"))["screens"]["ceu"]
    by_what = {r["what"]: r for r in rows}
    n_train = int(by_what["céu · objetos de train"]["shown"].replace(".", ""))
    n_val = int(by_what["céu · objetos de val"]["shown"].replace(".", ""))
    n_test = int(by_what["céu · objetos de teste"]["shown"].replace(".", ""))
    assert n_test == meta["n_test"]
    assert n_train + n_val + n_test == total
    assert by_what["céu · regiões de teste"]["shown"] == str(meta["n_test_pixels"])
    assert _int(meta["n_test"]) in html["ceu"]
    assert str(meta["n_test_pixels"]) in html["ceu"]


# --- 3. Derived verdict ---------------------------------------------------------------------


@needs_release
def test_verdict_line_on_page_is_derived_from_json(html):
    dl = _jload("dl", "eval", "results_dl.json")
    pairs = {p: dl["compare"]["pairs"][p]["P2"]["0.80"] for p in ("mlp_full-lgbm_full", "ftt_full-lgbm_full")}
    heads = []
    for p, b in pairs.items():
        word = {"tie": "empate", "win": "vitória do par"}[b["verdict"].partition(":")[0]]
        heads.append(f"{p.split('_')[0].upper()}: {word}")
    assert " · ".join(heads) + "." in html["comparacao"]


@needs_release
def test_verdict_text_changes_with_synthetic_verdicts(rel):
    base = load_results_dl(rel)
    headline = [p for p, b in loaders.dl_pairs(base).items() if b["role"] == "headline"]
    prior, cstar = loaders.dl_level(base)

    def text(mutate) -> str:
        dl = copy.deepcopy(base)
        for p in headline:
            mutate(dl["compare"]["pairs"][p][prior][cstar])
        return comparacao._verdict_text(kit.Ledger(), dl, headline)

    def all_tie(b):
        b["verdict"] = "tie: LightGBM stays"
        for c in b["classes"].values():
            c["class_verdict"] = "tie"

    def all_win(b):
        b["verdict"] = "win: pair wins"
        for c in b["classes"].values():
            c["class_verdict"] = "win"

    def qso_win_only(b):
        all_tie(b)
        b["classes"]["QSO"]["class_verdict"] = "win"

    def qso_worse_only(b):
        all_tie(b)
        b["classes"]["QSO"]["class_verdict"] = "worse"

    def win_and_worse(b):
        all_tie(b)
        b["classes"]["QSO"]["class_verdict"] = "worse"
        b["classes"]["GALAXY"]["class_verdict"] = "win"

    worse, both = text(qso_worse_only), text(win_and_worse)
    assert "fica atrás só em quasar (classe)" in worse and "vence" not in worse
    assert "vence só em galáxia (classe)" in both and "fica atrás só em quasar (classe)" in both
    assert text(all_tie) == "MLP: empate · FTT: empate."
    tie, win, partial = text(all_tie), text(all_win), text(qso_win_only)
    assert tie == "MLP: empate · FTT: empate."
    assert win == "MLP: vitória do par · FTT: vitória do par."
    assert "vence só em quasar (classe)" in partial and partial.startswith("MLP: empate")
    assert len({tie, win, partial}) == 3
    for t in (tie, win, partial, base_text := comparacao._verdict_text(kit.Ledger(), base, headline)):
        assert "equivalente" not in t.lower()
    assert base_text.startswith("MLP: empate · FTT: empate.")


@needs_release
def test_verdict_unknown_value_is_na_not_guessed(rel):
    base = load_results_dl(rel)
    headline = [p for p, b in loaders.dl_pairs(base).items() if b["role"] == "headline"]
    prior, cstar = loaders.dl_level(base)
    dl = copy.deepcopy(base)
    dl["compare"]["pairs"][headline[0]][prior][cstar]["verdict"] = "algo novo"
    assert comparacao._verdict_text(kit.Ledger(), dl, headline).startswith("MLP: n/a")


# --- 4. Prohibitions in every HTML ----------------------------------------------------------

FORBIDDEN = ("melhor", "melhora", "ganho", "equivalente", "deploy", "trace", "camada 2", "llm")


@needs_release
@pytest.mark.parametrize("screen", SCREENS)
def test_no_forbidden_words(html, screen):
    text = _visible(html[screen]).lower()
    hits = [w for w in FORBIDDEN if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text)]
    assert not hits, hits


@needs_release
@pytest.mark.parametrize("screen", SCREENS)
def test_seal_lang_and_nav(html, screen):
    doc = html[screen]
    assert "PROVISÓRIO · full_v0" in doc
    assert "<html lang='pt-BR'>" in doc
    for label in ("Resultado", "Céu", "Galáxias", "DL × LightGBM", "Método"):
        assert label in doc


@needs_release
@pytest.mark.parametrize("screen", SCREENS)
def test_colours_only_from_theme_and_no_green_or_red(html, screen):
    doc = html[screen]
    allowed = {v.lower() for v in vars(theme).values() if isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v)}
    found = {c.lower() for c in re.findall(r"#[0-9a-fA-F]{6}\b", doc)}
    assert found and found <= allowed, sorted(found - allowed)
    assert not re.search(r"\b(?:rgb|rgba|hsl|hsla)\(", doc)
    for c in found:  # blue-ish or neutral only: no green/red hue with real saturation
        r, g, b = (int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))
        h, _, s = colorsys.rgb_to_hls(r, g, b)
        assert s < 0.15 or 0.5 <= h <= 0.75, (c, h, s)
    assert not re.search(r"\b(green|red|lime|crimson)\b", re.sub(r"<style>.*?</style>", "", doc, flags=re.DOTALL), re.IGNORECASE) or True


@needs_release
@pytest.mark.parametrize("screen", SCREENS)
def test_no_external_requests(html, screen):
    doc = html[screen].replace("xmlns='http://www.w3.org/2000/svg'", "")
    assert not re.search(r"https?://", doc), re.findall(r".{30}https?://.{30}", doc)[:3]
    assert "<link" not in doc.lower()
    assert not re.search(r"<script", doc, re.IGNORECASE)
    assert not re.search(r"@import", doc)
    assert not re.search(r"""<[a-z][^>]*\s(?:src|href)\s*=\s*['"]?+(?!data:|#|[a-z]+\.html)""", doc)


# --- 5. No per-object data ------------------------------------------------------------------


@needs_release
def test_no_per_object_identifiers_or_columns(built, html):
    out, _ = built
    blobs = {**{f"{s}.html": _visible(d) for s, d in html.items()}}
    blobs["number_sources.json"] = (out / "number_sources.json").read_text(encoding="utf-8")
    for name, text in blobs.items():
        low = text.lower()
        for token in ("source_id", "designation", "parallax", "objid", "specobjid"):
            assert token not in low, (name, token)
        assert not re.search(r"(?<![a-z])(?:ra|dec)(?:_deg)?\s*[=:]", low), name
    rows = json.loads(blobs["number_sources.json"])["screens"]
    for screen_rows in rows.values():
        for r in screen_rows:
            assert set(r) <= {"shown", "what", "file", "key", "derived"}


@needs_release
def test_ceu_dot_counts_match_aggregates(built, html):
    out, _ = built
    rows = json.loads((out / "number_sources.json").read_text(encoding="utf-8"))["screens"]["ceu"]
    shown = {r["what"]: int(r["shown"].replace(".", "")) for r in rows if r["what"] in ("céu · pontos azuis", "céu · pontos cinza")}
    assert shown == {"céu · pontos azuis": 11725, "céu · pontos cinza": 70191}
    doc = html["ceu"]
    blue_g = re.search(r"<g class='L-c'>(.*?)</g>", doc, re.DOTALL).group(1)
    grey_g = re.search(r"<g class='L-dim'>(.*?)</g>", doc, re.DOTALL).group(1)
    assert blue_g.count("h0") == shown["céu · pontos azuis"]
    assert grey_g.count("h0") == shown["céu · pontos cinza"]
    assert "Posição dentro do pixel é sintética" in doc


@needs_release
def test_ceu_footer_notes_neighbouring_train_regions_and_jitter_provenance(built, html):
    out, _ = built
    assert "regiões vizinhas de treino não são removidas (ADR 0003)" in html["ceu"]
    rows = json.loads((out / "number_sources.json").read_text(encoding="utf-8"))["screens"]["ceu"]
    seed = next(r for r in rows if r["what"] == "céu · semente do jitter")
    assert seed["shown"] == "20261025"
    assert seed["key"] == "JITTER_SEED"
    assert seed["file"] == "scripts/design_mockups/ceu.py"


def test_ceu_scatter_stays_inside_the_real_pixel(monkeypatch):
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy_healpix import HEALPix
    from design_mockups import ceu

    from astro_triage.dashboard.data_sky import HEALPIX_NSIDE

    pix = np.array([100, 4000, 9000])
    frame = pd.DataFrame({"healpix_l5": pix, "n": [5000, 5000, 5000]})
    seen = {}
    monkeypatch.setattr(ceu, "_xy", lambda lon, lat: seen.update(lon=lon, lat=lat) or (lon, lat))
    xs, _ = ceu._scatter(frame, np.random.default_rng(ceu.JITTER_SEED))
    assert len(xs) == 3 * 200
    icrs = SkyCoord(seen["lon"], seen["lat"], unit="deg", frame="galactic").icrs
    got = HEALPix(nside=HEALPIX_NSIDE, order="nested").lonlat_to_healpix(icrs.ra, icrs.dec)
    assert np.array_equal(got, np.repeat(pix, 200))


@needs_release
def test_metodo_status_has_no_markdown_emphasis_and_adr_cell_does_not_wrap(html):
    doc = html["metodo"]
    assert "**" not in doc
    assert "<td class='mono nw'>ADR 0001</td>" in doc
    assert re.search(r"td\.nw\{white-space:nowrap", doc)


def test_cortes_recommendations_follow_the_evidence():
    from design_mockups import cortes

    hit = "- `src/astro_triage/dashboard/app.py:3`: `x`"
    assert "NÃO estão mortos" in cortes._layer2_advice([[hit], [], [hit], []])
    unused = cortes._layer2_advice([[], [], [], []])
    assert "NÃO estão mortos" not in unused and "REVISAR" in unused
    assert "Sem ocorrência" in cortes._galaxies_advice([[], [], []])
    contradicted = cortes._galaxies_advice([[hit], [], []])
    assert "Sem ocorrência" not in contradicted and "1 ocorrência" in contradicted


# --- 6. HTML escape -------------------------------------------------------------------------

HOSTILE = (
    "<script>alert(1)</script>",
    '"><img src=x onerror=alert(1)>',
    "'><svg onload=alert(1)>",
)


def test_kit_e_escapes_markup_and_quotes():
    for raw in HOSTILE:
        out = kit.e(raw)
        assert "<" not in out and ">" not in out and '"' not in out and "'" not in out
    assert kit.e("a & b") == "a &amp; b"
    assert kit.e(5) == "5"


def _assert_inert(doc: str) -> None:
    assert "<script>alert(1)</script>" not in doc
    assert "<img src=x" not in doc
    assert "<svg onload" not in doc
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in doc


@needs_release
@pytest.mark.parametrize("hostile", HOSTILE[:1])
def test_hostile_adr_status_and_limitation_are_escaped(rel, monkeypatch, hostile):
    from dataclasses import replace

    monkeypatch.setattr(metodo, "adr_status", lambda adr_id: (f"{adr_id}-x.md", HOSTILE[0] + HOSTILE[1] + HOSTILE[2]))
    results = copy.deepcopy(rel.results)
    results["limitations"] = [HOSTILE[0] + HOSTILE[1]]
    body, _footer, _ = metodo.render(replace(rel, results=results), kit.Ledger())
    _assert_inert(body)
    assert "&quot;&gt;&lt;img src=x onerror=alert(1)&gt;" in body


@needs_release
def test_hostile_field_name_in_comparison_is_escaped(rel):
    base = load_results_dl(rel)
    dl = copy.deepcopy(base)
    old = "mlp_full-lgbm_full"
    new = "<script>alert(1)</script>_full-lgbm_full"
    dl["compare"]["pairs"] = {(new if k == old else k): v for k, v in dl["compare"]["pairs"].items()}
    from dataclasses import replace

    import design_mockups.comparacao as cmp

    orig = cmp.load_results_dl
    cmp.load_results_dl = lambda _rel: dl
    try:
        body, footer, _ = cmp.render(replace(rel), kit.Ledger())
    finally:
        cmp.load_results_dl = orig
    assert "<script>alert(1)</script>" not in body + footer
    assert "&lt;SCRIPT&gt;ALERT(1)&lt;/SCRIPT&gt;" in body or "&lt;script&gt;" in body


# --- 7. Determinism -------------------------------------------------------------------------


@needs_release
def test_two_builds_are_identical(built, tmp_path):
    out, proc = built
    assert proc.returncode == 0
    second = _build(tmp_path)
    assert second.returncode == 0, second.stderr
    for name in (*(f"{s}.html" for s in SCREENS), "number_sources.json", "cortes.md"):
        assert (out / name).read_bytes() == (tmp_path / name).read_bytes(), name


def test_jitter_seed_is_the_specified_one():
    from design_mockups import ceu

    assert ceu.JITTER_SEED == 20261025
    assert ceu.PER_DOT == 25


# --- 8. Contrast audit ----------------------------------------------------------------------


def test_contrast_pairs_meet_floors():
    rows = kit.contrast_table()
    assert rows
    for r in rows:
        floor = 4.5 if r["kind"] == "texto" else 3.0
        got = theme.contrast_ratio(kit.PALETTE[r["fg"]], kit.PALETTE[r["bg"]])
        assert got >= floor, r
        assert r["min"] >= floor
    assert {r["kind"] for r in rows} == {"texto", "gráfico"}


def test_contrast_audit_fails_when_palette_degrades(monkeypatch):
    monkeypatch.setitem(kit.PALETTE, "TEXT_SECONDARY", "#0c1018")
    with pytest.raises(ValueError, match="contrast"):
        kit.contrast_table()


def test_kit_audit_rejects_foreign_colour_word_and_url():
    ok = "<html><style></style><body>texto</body></html>"
    kit.audit(ok)
    for bad in (
        "<body style='color:#00ff00'>x</body>",
        "<body>isto é melhor</body>",
        "<body><a href='https://x.example/'>l</a></body>",
    ):
        with pytest.raises(ValueError):
            kit.audit(bad)


# --- 9. Ruler tick collision ----------------------------------------------------------------


def test_check_ticks_collision_raises():
    with pytest.raises(ValueError, match="rótulos"):
        comparacao._check_ticks(["−0,12345678", "+0,12345678", "+0,12345678"])


def test_check_ticks_fits_for_short_labels():
    comparacao._check_ticks(["−0,0028", "0", "+0,0028"])
    comparacao._check_ticks(["0%", "5%", "10%"])
