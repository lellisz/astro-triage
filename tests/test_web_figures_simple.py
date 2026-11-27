"""Independent tests of the simple-mode site figures (lay reader), from the acceptance criteria.

Simple figures: every ``*-simples`` id and every ``quasar-resumo-*`` id. They must carry no
statistical jargon, no interval marks, no notes and no source/release line, and every number
they show must come from the exported JSON. Checks run on the committed fixture ``fixture_v0``.
"""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path

import pytest

# Helpers of the figure suite (sibling module, same rootdir import style as other tests).
from test_web_figures import (
    FIX,
    PAGES,
    REL,
    _copy_fixture,
    _derived,
    _edit,
    _forms,
    _generate,
    _texts,
    _ticks,
    _tokens,
    _walk,
)

from astro_triage.dashboard.html_galaxies import FARTHEST, NEAREST
from astro_triage.web_export.figures import charts
from astro_triage.web_export.figures.canvas import TEXT2, Canvas


def _is_simple(fid: str) -> bool:
    return fid.endswith("-simples") or fid.startswith("quasar-resumo-")


@pytest.fixture(scope="module")
def gen(tmp_path_factory) -> Path:
    return _generate(tmp_path_factory.mktemp("figs-simple"))


@pytest.fixture(scope="module")
def svgs(gen) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(gen.glob("*.svg"))}


@pytest.fixture(scope="module")
def simple(svgs) -> dict[str, str]:
    out = {k: v for k, v in svgs.items() if _is_simple(k)}
    assert len(out) == 8, sorted(out)  # 3 *-simples + galaxia + pureza + 4 quasar-resumo (one per state)
    return out


@pytest.fixture(scope="module")
def pages() -> dict[str, dict]:
    return {n: json.loads((FIX / f"{n}.json").read_text(encoding="utf-8")) for n in PAGES}


def _all_text(svg: str) -> list[str]:
    """Every <text>, <title> and <desc> body of an SVG."""
    out = [t for _, t in _texts(svg)]
    out += [unescape(m) for m in re.findall(r"<title\b[^>]*>([^<]*)</title>", svg)]
    out += [unescape(m) for m in re.findall(r"<desc\b[^>]*>([^<]*)</desc>", svg)]
    return out


# ---------------------------------------------------------------- 1. jargon
JARGON = [
    r"\bIC\b", r"intervalo", r"bootstrap", r"p\.p\.", r"prior", r"regra", r"F macro", r"piso",
    r"calibra", r"temperatura", r"redshift", r"δ", r"\bn\s*=", r"semente",
]
JARGON_RE = [re.compile(p, re.IGNORECASE) for p in JARGON]


def _null_estimates(fid: str, p: dict[str, dict]) -> int:
    """How many of the JSON values a simple figure shows are null (each may print as n/a)."""
    g, o, ga, c = p["graficos"], p["overview"], p["galaxies"], p["compare"]
    if fid == "taxa-por-g-simples":
        ests = [r["rate"] for s in g["error_by_g"]["series"] for r in s["rows"]]
    elif fid == "confusao-simples":
        ests = [cell["share"] for r in g["confusion"]["rows"] for cell in r["cells"]]
    elif fid == "galaxia-erros-simples":
        ests = [r["rate"] for r in ga["error_rows"] if r["label"] not in (NEAREST, FARTHEST)]
    elif fid == "pureza-sistemas-simples":
        ests = [{"point": r["median"]} for r in c["purity"]]
    elif fid.startswith("quasar-resumo-"):
        row = next(r for r in o["rows"]
                   if f"quasar-resumo-{r['rule']}-{r['prior']}".replace("_", "-").lower() == fid)
        ests = [row["purity"], row["completeness"]]
    else:
        raise AssertionError(f"unknown simple figure {fid}")
    return sum(1 for e in ests if e.get("point") is None)


def _jargon_violations(fid: str, svg: str, nulls: int) -> list[str]:
    texts = _all_text(svg)
    bad = [f"{rx.pattern}: {t!r}" for t in texts for rx in JARGON_RE if rx.search(t)]
    if fid != "pureza-sistemas-simples":
        bad += [f"LightGBM: {t!r}" for t in texts if "lightgbm" in t.lower()]
    # n/a only where the JSON value is null: never more n/a texts than null values shown.
    na = sum(len(re.findall(r"\bn/a\b", t, re.IGNORECASE)) for _, t in _texts(svg))
    if na > nulls:
        bad.append(f"n/a x{na} with only {nulls} null values in the JSON")
    return bad


def test_1_simple_figures_have_no_jargon(simple, pages):
    found = {fid: v for fid, svg in simple.items() if (v := _jargon_violations(fid, svg, _null_estimates(fid, pages)))}
    assert not found, found


def test_1_lightgbm_appears_where_allowed(simple):
    # counter-proof: the one figure allowed to name LightGBM does name it (the rule is not vacuous)
    assert any("LightGBM" in t for t in _all_text(simple["pureza-sistemas-simples"]))


def test_1_technical_counterparts_do_carry_jargon(svgs):
    # counter-proof: the scan finds jargon in the technical figures, so it can find it at all
    for fid in ("galaxia-erros", "taxa-por-g", "quasar-pureza"):
        assert _jargon_violations(fid, svgs[fid], 99), fid


@pytest.mark.parametrize(
    "fid,old,new",
    [
        ("confusao-simples", ">Tipo real<", ">Tipo real (IC 95%)<"),
        ("taxa-por-g-simples", "</title>", " por redshift</title>"),
        ("galaxia-erros-simples", "</desc>", "; semente 3</desc>"),
        ("quasar-resumo-as-shipped-p2", ">Porcentagem (%)<", ">Porcentagem, n = 15<"),
        ("quasar-resumo-prior-matched-p1", ">Porcentagem (%)<", ">LightGBM<"),
        ("quasar-resumo-prior-matched-p2", ">Porcentagem (%)<", ">n/a<"),
        ("pureza-sistemas-simples", "</title>", " após calibração</title>"),
    ],
)
def test_1_mutation_planted_term_is_flagged(simple, pages, fid, old, new):
    clean = simple[fid]
    nulls = _null_estimates(fid, pages)
    assert not _jargon_violations(fid, clean, nulls)
    planted = clean.replace(old, new, 1)
    assert planted != clean, f"anchor {old!r} not found in {fid}"
    assert _jargon_violations(fid, planted, nulls)


def test_1_na_is_allowed_when_the_json_value_is_null(tmp_path):
    # boundary: a quasar row with a null purity prints n/a in the summary and that is not a violation
    data = _copy_fixture(tmp_path)

    def null_purity(doc: dict) -> None:
        doc["rows"][1]["purity"] = {"point": None, "lo": None, "hi": None, "text": "n/a", "ci_text": "n/a"}

    _edit(data, "overview", null_purity)
    from astro_triage.web_export.figures import write_figures

    write_figures(data, tmp_path / "out", REL)
    fid = "quasar-resumo-prior-matched-p1"
    svg = (tmp_path / "out" / REL / "figures" / f"{fid}.svg").read_text(encoding="utf-8")
    p = {n: json.loads((data / REL / f"{n}.json").read_text(encoding="utf-8")) for n in PAGES}
    assert any(t == "n/a" for _, t in _texts(svg))
    assert not _jargon_violations(fid, svg, _null_estimates(fid, p))


# ---------------------------------------------------------------- 2. no technical annotation
@pytest.fixture(scope="module")
def ops(pages) -> dict[str, list[tuple[str, str, str]]]:
    """Drawing calls per figure: (kind, class, text). Recorded at the Canvas API, before matplotlib."""
    rec: dict[str, list[tuple[str, str, str]]] = {}
    text0, vtext0, poly0 = Canvas.text, Canvas.vtext, Canvas.poly

    def text(self, x, y, s, cls, anchor="start"):
        rec.setdefault(self.fid, []).append(("text", cls, s))
        return text0(self, x, y, s, cls, anchor)

    def vtext(self, x, y, s, cls, anchor="middle"):
        rec.setdefault(self.fid, []).append(("text", cls, s))
        return vtext0(self, x, y, s, cls, anchor)

    def poly(self, pts, cls):
        rec.setdefault(self.fid, []).append(("line", cls, ""))
        return poly0(self, pts, cls)

    mp = pytest.MonkeyPatch()
    mp.setattr(Canvas, "text", text)
    mp.setattr(Canvas, "vtext", vtext)
    mp.setattr(Canvas, "poly", poly)
    mp.setattr(Canvas, "render", lambda self, desc: "<svg/>")  # only the calls matter here
    try:
        for build in charts.builders(pages):
            build(pages)
    finally:
        mp.undo()
    return rec


def _annotation_violations(calls: list[tuple[str, str, str]]) -> list[str]:
    bad = [f"interval line {c}" for k, c, _ in calls if k == "line" and (c == "cil" or re.fullmatch(r"ci\d+", c))]
    # Class "n" is the note/source style; in a simple figure it may only print a null value (n/a).
    bad += [f"note text {s!r}" for k, c, s in calls if k == "text" and c == "n" and s != "n/a"]
    return bad


def test_2_simple_figures_draw_no_interval_and_no_note(ops):
    simple_ids = [f for f in ops if _is_simple(f)]
    assert len(simple_ids) == 8, simple_ids
    found = {f: v for f in simple_ids if (v := _annotation_violations(ops[f]))}
    assert not found, found


@pytest.mark.parametrize("fid", ["taxa-por-g", "galaxia-erros", "confusao", "pureza-sistemas", "quasar-pureza"])
def test_2_technical_counterpart_does_draw_them(ops, fid):
    # counter-proof: the detector sees the interval marks and notes of the technical figures
    assert _annotation_violations(ops[fid]), fid


def test_2_simple_svg_has_no_source_line_nor_interval_stroke(simple):
    for fid, svg in simple.items():
        texts = [t for _, t in _texts(svg)]
        assert not any(REL in t for t in texts), (fid, "release in a text")
        assert not any("·" in t for t in texts), (fid, "source line 'fonte · release'")
        # ci{k} and cil are the only 1.5 px strokes of the canvas
        assert "stroke-width: 1.5" not in svg, (fid, "interval stroke")


def test_2_technical_svg_does_have_them(svgs):
    # counter-proof for the SVG-level scan
    for fid in ("galaxia-erros", "taxa-por-g"):
        texts = [t for _, t in _texts(svgs[fid])]
        assert any(REL in t for t in texts), fid
        assert "stroke-width: 1.5" in svgs[fid], fid


# ---------------------------------------------------------------- 3. numbers from the JSON
@pytest.fixture(scope="module")
def allowed(pages) -> set[str]:
    # Same mechanism as test_web_figures.test_5: every pt-BR form of every JSON number.
    nums: list[float] = []
    strs: list[str] = []
    lens: list[int] = []
    _walk(pages, nums, strs, lens)
    ok: set[str] = set()
    for v in nums + [float(n) for n in lens] + list(_derived(pages)):
        ok |= _forms(v)
    for s in strs:
        ok |= set(_tokens(s))
    return ok


def _invented(svg: str, allowed: set[str]) -> list[str]:
    texts = _texts(svg)
    ticks = _ticks(texts)
    bad = [tok for i, (_, t) in enumerate(texts) if i not in ticks for tok in _tokens(t) if tok not in allowed]
    desc = unescape(re.search(r"<desc[^>]*>(.*?)</desc>", svg, flags=re.DOTALL).group(1))
    title = unescape(re.search(r"<title[^>]*>(.*?)</title>", svg, flags=re.DOTALL).group(1))
    bad += [tok for tok in _tokens(desc) + _tokens(title) if tok not in allowed]
    return bad


def test_3_every_number_in_a_simple_figure_comes_from_the_json(simple, allowed):
    found = {fid: v for fid, svg in simple.items() if (v := _invented(svg, allowed))}
    assert not found, found


def test_3_each_simple_figure_shows_some_json_value(simple, pages):
    o, ga, c = pages["overview"], pages["galaxies"], pages["compare"]
    for r in o["rows"]:
        fid = f"quasar-resumo-{r['rule']}-{r['prior']}".replace("_", "-").lower()
        texts = [t for _, t in _texts(simple[fid])]
        # the summary of a state shows that state's values, not another row's
        assert r["purity"]["text"] in texts and r["completeness"]["text"] in texts, fid
    gtexts = [t for _, t in _texts(simple["galaxia-erros-simples"])]
    assert ga["error_rows"][0]["rate"]["text"] in gtexts
    ptexts = [t for _, t in _texts(simple["pureza-sistemas-simples"])]
    assert all(f"{r['median'] * 100:.2f}".replace(".", ",") + "%" in ptexts for r in c["purity"])


def test_3_mutation_planted_number_is_flagged(simple, allowed):
    svg = simple["galaxia-erros-simples"]
    planted = svg.replace(">mais brilhantes<", ">mais brilhantes 63,7%<", 1)
    assert planted != svg
    assert "63,7%" not in allowed and "63,7%" in _invented(planted, allowed)


# ---------------------------------------------------------------- 4. galaxy groups
def _bar_labels(svg: str) -> list[str]:
    # bar names: secondary text colour, left-anchored at the margin
    return [t for a, t in _texts(svg)
            if f"fill: {TEXT2}" in a.get("style", "") and "text-anchor: start" in a.get("style", "")
            and a.get("x") == "16"]


def test_4_galaxy_simple_has_exactly_the_non_redshift_groups(simple, pages):
    labels = [r["label"] for r in pages["galaxies"]["error_rows"]]
    assert len(labels) == 4 and NEAREST in labels and FARTHEST in labels, labels  # the filter is not vacuous
    want = [lab for lab in labels if lab not in (NEAREST, FARTHEST)]
    assert want == ["Todas as galáxias", "mais brilhantes"]
    got = _bar_labels(simple["galaxia-erros-simples"])
    assert got == want, got
    every = " ".join(_all_text(simple["galaxia-erros-simples"]))
    assert NEAREST not in every and FARTHEST not in every


def test_4_galaxy_technical_keeps_the_four_groups(svgs, pages):
    labels = [r["label"] for r in pages["galaxies"]["error_rows"]]
    got = _bar_labels(svgs["galaxia-erros"])
    for lab in labels:
        assert sum(1 for t in got if t == lab or t.startswith(f"{lab} ·")) == 1, (lab, got)


def _real_band_rows(doc: dict) -> None:
    # REV-01: with real data the band labels carry the bin name, not the bare NEAREST/FARTHEST
    rate = {"point": 0.25, "lo": 0.1, "hi": 0.45, "text": "25,0%", "ci_text": "IC 95%: 10,0% a 45,0%"}
    for row, label in zip(doc["error_rows"][2:], ("z < 0,1 (redshift quase nulo)", "z 1,0–1,5"), strict=True):
        row.update(label=label, rate=dict(rate), n={"text": "40", "value": 40},
                   errors={"text": "10", "value": 10})


def test_4_galaxy_simple_filters_by_position_with_real_band_labels(tmp_path):
    from astro_triage.web_export.figures import write_figures

    data = _copy_fixture(tmp_path)
    _edit(data, "galaxies", _real_band_rows)
    write_figures(data, tmp_path / "out", REL)
    svg = (tmp_path / "out" / REL / "figures" / "galaxia-erros-simples.svg").read_text(encoding="utf-8")
    assert _bar_labels(svg) == ["Todas as galáxias", "mais brilhantes"]
    every = " ".join(_all_text(svg))
    assert "redshift" not in every and "z " not in every, every


def test_4_galaxy_simple_refuses_a_wrong_row_count(tmp_path):
    from astro_triage.web_export.figures import FigureError, write_figures

    data = _copy_fixture(tmp_path)
    _edit(data, "galaxies", lambda doc: doc["error_rows"].pop())
    with pytest.raises(FigureError, match="error_rows"):
        write_figures(data, tmp_path / "out", REL)
