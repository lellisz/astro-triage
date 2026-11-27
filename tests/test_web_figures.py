"""Independent tests of the matplotlib site figures (astro_triage.web_export.figures).

Written from the acceptance criteria, not from the implementation: numbers are formatted here
with an own pt-BR formatter, and every check runs on the committed fixture ``fixture_v0``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from html import unescape
from itertools import combinations, pairwise
from pathlib import Path

import pytest

from astro_triage.web_export.figures import FigureError, write_figures

ROOT = Path(__file__).resolve().parents[1]
FIX_ROOT = ROOT / "web" / "test" / "fixtures"
REL = "fixture_v0"
FIX = FIX_ROOT / REL
COMMITTED = FIX / "figures"
PAGES = ("meta", "overview", "galaxies", "compare", "graficos")

EXPECTED_IDS = {
    "calibracao",
    "taxa-por-g",
    "taxa-por-z-galaxy",
    "taxa-por-z-qso",
    "confusao",
    "cmd-taxa",
    "quasar-pureza",
    "quasar-completude",
    "galaxia-erros",
    "dsc-diferencas",
    "decisao-diferencas",
    "ganho-lgbm-full",
    "ganho-lgbm-noind",
    "pureza-sistemas",
}


def _generate(out: Path) -> Path:
    write_figures(FIX_ROOT, out, REL)
    return out / REL / "figures"


@pytest.fixture(scope="module")
def gen(tmp_path_factory) -> Path:
    return _generate(tmp_path_factory.mktemp("figs"))


@pytest.fixture(scope="module")
def svgs(gen) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(gen.glob("*.svg"))}


@pytest.fixture(scope="module")
def pages() -> dict[str, dict]:
    return {n: json.loads((FIX / f"{n}.json").read_text(encoding="utf-8")) for n in PAGES}


def _texts(svg: str) -> list[tuple[dict[str, str], str]]:
    out = []
    for attrs, body in re.findall(r"<text\b([^>]*)>([^<]*)</text>", svg):
        out.append((dict(re.findall(r'([a-z-]+)="([^"]*)"', attrs)), unescape(body)))
    return out


# ---------------------------------------------------------------- failure paths first
def _copy_fixture(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / REL).mkdir(parents=True)
    for n in PAGES:
        shutil.copy(FIX / f"{n}.json", data / REL / f"{n}.json")
    return data


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "x y"])
def test_invalid_release_name_is_refused_before_any_write(tmp_path, bad):
    with pytest.raises(FigureError):
        write_figures(FIX_ROOT, tmp_path / "out", bad)
    assert not (tmp_path / "out").exists()


def test_missing_page_json_is_refused_without_writing(tmp_path):
    data = _copy_fixture(tmp_path)
    (data / REL / "compare.json").unlink()
    with pytest.raises(FigureError, match="compare.json"):
        write_figures(data, tmp_path / "out", REL)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "field,value",
    [("provisional", False), ("release", "outra"), ("schema_version", -1)],
)
def test_page_header_mismatch_is_refused(tmp_path, field, value):
    data = _copy_fixture(tmp_path)
    f = data / REL / "graficos.json"
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc[field] = value
    f.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(FigureError):
        write_figures(data, tmp_path / "out", REL)
    assert not (tmp_path / "out").exists()


def test_malformed_chart_json_leaves_no_half_set(tmp_path):
    data = _copy_fixture(tmp_path)
    f = data / REL / "compare.json"
    doc = json.loads(f.read_text(encoding="utf-8"))
    del doc["purity"]  # last figure built: the earlier ones were already rendered in memory
    f.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(FigureError):
        write_figures(data, tmp_path / "out", REL)
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- 1. ids
def test_1_generates_exactly_the_14_ids_plus_manifest(gen):
    names = {p.name for p in gen.iterdir()}
    assert names == {f"{i}.svg" for i in EXPECTED_IDS} | {"figures.json"}
    manifest = json.loads((gen / "figures.json").read_text(encoding="utf-8"))
    assert set(manifest["figures"]) == EXPECTED_IDS
    assert manifest["release"] == REL and manifest["provisional"] is True
    for fid, entry in manifest["figures"].items():
        assert all(isinstance(entry[k], str) and entry[k].strip() for k in ("title", "caption", "aria_label")), fid


# ---------------------------------------------------------------- 2. no markers or glyphs
FORBIDDEN_SHAPES = ["<circle", "<polygon", "<ellipse", "<use ", "<use>", "<marker"]
FORBIDDEN_GLYPHS = ["●", "■", "▲", "◆", "▸", "→"]


def _marker_violations(svg: str) -> list[str]:
    bad = [b for b in FORBIDDEN_SHAPES + FORBIDDEN_GLYPHS if b in svg]
    # matplotlib defines markers as <path id=...> inside <defs> and reuses them with <use>;
    # a hatch <pattern> also holds a <path>, so patterns are removed before the scan.
    for defs in re.findall(r"<defs>(.*?)</defs>", svg, flags=re.DOTALL):
        rest = re.sub(r"<pattern\b.*?</pattern>", "", defs, flags=re.DOTALL)
        if "<path" in rest:
            bad.append("marker <path> in <defs>")
    return bad


def test_2_no_marker_shapes_or_glyphs(svgs):
    found = {fid: v for fid, svg in svgs.items() if (v := _marker_violations(svg))}
    assert not found, found


@pytest.mark.parametrize(
    "planted",
    [
        '<svg><defs><path id="m1" d="M 0 0"/></defs></svg>',
        '<svg><use xlink:href="#m1" x="1"/></svg>',
        '<svg><text>● Estrela</text></svg>',
        '<svg><text>A → B</text></svg>',
        '<svg><circle r="2"/></svg>',
    ],
)
def test_2_scan_detects_a_planted_marker(planted):
    assert _marker_violations(planted)


# ---------------------------------------------------------------- 3. determinism
def test_3_two_generations_are_byte_identical(tmp_path):
    a, b = _generate(tmp_path / "a"), _generate(tmp_path / "b")
    files = sorted(p.name for p in a.iterdir())
    assert files == sorted(p.name for p in b.iterdir())
    for name in files:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_3_generation_is_independent_of_hash_seed(tmp_path):
    env_path = os.pathsep.join(p for p in sys.path if p)
    outs = []
    for seed in ("1", "2"):
        out = tmp_path / f"seed{seed}"
        code = (
            "import sys; from pathlib import Path;"
            "from astro_triage.web_export.figures import write_figures;"
            f"write_figures(Path(sys.argv[1]), Path(sys.argv[2]), {REL!r})"
        )
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": env_path}
        res = subprocess.run([sys.executable, "-c", code, str(FIX_ROOT), str(out)], env=env,
                             capture_output=True, text=True, timeout=300, check=False)
        assert res.returncode == 0, res.stderr[-2000:]
        outs.append(out / REL / "figures")
    for p in sorted(outs[0].iterdir()):
        assert p.read_bytes() == (outs[1] / p.name).read_bytes(), p.name


# ---------------------------------------------------------------- 4. committed fixture is current
def test_4_committed_fixture_matches_current_generation(gen):
    committed = sorted(p.name for p in COMMITTED.iterdir())
    assert committed == sorted(p.name for p in gen.iterdir())
    stale = [n for n in committed if (COMMITTED / n).read_bytes() != (gen / n).read_bytes()]
    assert not stale, f"fixture desatualizada: {stale}"


# ---------------------------------------------------------------- 5. numbers come from the JSON
def _pt(x: float, d: int, sign: bool = False) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("+-")
    return s.replace(".", ",")


def _pt_int(n: float) -> str:
    return f"{round(n):,}".replace(",", ".")


NUM = re.compile(r"(?<![A-Za-z_\d.,])[+\-−]?\d+(?:[.,]\d+)*%?")


def _tokens(s: str) -> list[str]:
    return [t.replace("−", "-") for t in NUM.findall(s)]


def _walk(node, nums: list[float], strs: list[str], lens: list[int]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _walk(v, nums, strs, lens)
    elif isinstance(node, list):
        lens.append(len(node))
        for v in node:
            _walk(v, nums, strs, lens)
    elif isinstance(node, bool) or node is None:
        return
    elif isinstance(node, (int, float)):
        nums.append(float(node))
    elif isinstance(node, str):
        strs.append(node)


def _forms(v: float) -> set[str]:
    out = {_pt_int(v)}
    for d in range(4):
        for x in (v, v * 100):
            for sign in (False, True):
                s = _pt(x, d, sign)
                out |= {s, s + "%"}
    return {o.replace("−", "-") for o in out}


def _derived(p: dict[str, dict]) -> set[float]:
    """Values a figure computes from the JSON (sums and ratios), recomputed here."""
    g = p["graficos"]
    out: set[float] = set()
    for s in g["calibration"]["series"]:
        out.add(sum(b["n"]["value"] for b in s["bins"] if b["n"]["value"] is not None))
    out.add(sum(r["n"]["value"] for r in g["confusion"]["rows"]))
    n_test = p["overview"]["technical"]["data_box"]["n_test"]["value"]
    out.add(round(g["cmd"]["n_outside"]["value"] / n_test * 100, 1))
    return out


@pytest.fixture(scope="module")
def allowed(pages) -> set[str]:
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


def _ticks(texts: list[tuple[dict[str, str], str]]) -> set[int]:
    """Indices of axis tick labels: >= 3 bare numbers on one row (or column) in arithmetic step."""
    bare = re.compile(r"^[+\-−]?\d+(?:,\d+)?%?$")
    idx = [i for i, (_, t) in enumerate(texts) if bare.match(t.strip())]
    out: set[int] = set()
    for key in ("y", "x"):
        groups: dict[str, list[int]] = {}
        for i in idx:
            groups.setdefault(texts[i][0].get(key, ""), []).append(i)
        for members in groups.values():
            if len(members) < 3:
                continue
            vals = sorted(float(texts[i][1].strip().rstrip("%").replace("−", "-").replace(",", "."))
                          for i in members)
            steps = {round(b - a, 6) for a, b in pairwise(vals)}
            if len(steps) == 1 and steps != {0}:
                out |= set(members)
    return out


def test_5_no_invented_number_in_any_text(svgs, allowed):
    bad = []
    for fid, svg in svgs.items():
        texts = _texts(svg)
        ticks = _ticks(texts)
        for i, (_, t) in enumerate(texts):
            if i in ticks:
                continue
            bad += [(fid, t, tok) for tok in _tokens(t) if tok not in allowed]
        desc = unescape(re.search(r"<desc[^>]*>(.*?)</desc>", svg, flags=re.DOTALL).group(1))
        bad += [(fid, "<desc>", tok) for tok in _tokens(desc) if tok not in allowed]
    assert not bad, bad[:20]


def test_5_scan_flags_a_planted_number(svgs, allowed):
    planted = svgs["confusao"].replace("PROVISÓRIO", "PROVISÓRIO 97,3%", 1)
    texts = _texts(planted)
    ticks = _ticks(texts)
    toks = [tok for i, (_, t) in enumerate(texts) if i not in ticks for tok in _tokens(t)]
    assert "97,3%" in toks and "97,3%" not in allowed


def _rep(e: dict) -> bool:
    return e.get("point") is not None and e.get("text") not in (None, "n/a")


def _source_values(p: dict[str, dict]) -> dict[str, list[str]]:
    """One or more texts read straight from the JSON that each figure must show."""
    g, o, ga, c = p["graficos"], p["overview"], p["galaxies"], p["compare"]
    zs = {s["cls"]: s for s in g["error_by_z"]["series"]}

    def rates(rows):
        # fixture_v0 has series with every bin below the floor; then the bin labels (z or G
        # ranges read from the JSON) are the numbers the figure must still show.
        got = [r["rate"]["text"] for r in rows if not r["below_floor"] and _rep(r["rate"])]
        return got or [r["label"] for r in rows if re.search(r"\d", r["label"])]

    models = {m["model"].replace("_", "-"): m for m in o["technical"]["models"]}
    return {
        "calibracao": [b["n"]["text"] for s in g["calibration"]["series"] for b in s["bins"]
                       if b["n"].get("text")][:1] + [f"{_pt_int(g['floor'])}"],
        "taxa-por-g": rates([r for s in g["error_by_g"]["series"] for r in s["rows"]]),
        "taxa-por-z-galaxy": rates(zs["GALAXY"]["rows"]),
        "taxa-por-z-qso": rates(zs["QSO"]["rows"]),
        "confusao": [cell["share"]["text"] for r in g["confusion"]["rows"] for cell in r["cells"]
                     if not cell["below_floor"] and _rep(cell["share"])],
        "cmd-taxa": [g["cmd"]["n_outside"]["text"]],
        "quasar-pureza": [r["purity"]["text"] for r in o["rows"] if _rep(r["purity"])],
        "quasar-completude": [r["completeness"]["text"] for r in o["rows"] if _rep(r["completeness"])],
        "galaxia-erros": [r["n"]["text"] for r in ga["error_rows"]],
        "dsc-diferencas": [o["technical"]["dsc"]["n_rows"]["text"]]
        + [d["est"]["text"] for d in o["technical"]["dsc"]["paired_diffs"] if _rep(d["est"])],
        "decisao-diferencas": [_pt(gr["d"] * 100, 2, sign=True) + " p.p." for gr in c["grid"]],
        "ganho-lgbm-full": [i["gain"]["text"] for i in models["lgbm-full"]["importance"]],
        "ganho-lgbm-noind": [i["gain"]["text"] for i in models["lgbm-noind"]["importance"]],
        "pureza-sistemas": [_pt(r["median"] * 100, 2) + "%" for r in c["purity"]],
    }


def test_5_every_figure_shows_numbers_read_from_the_json(svgs, pages):
    src = _source_values(pages)
    assert set(src) == EXPECTED_IDS
    missing = []
    for fid, wanted in src.items():
        assert wanted, f"{fid}: o JSON da fixture não tem valor reportável para conferir"
        texts = [t for _, t in _texts(svgs[fid])]
        assert any(re.search(r"\d", t) for t in texts), fid
        hit = [w for w in wanted if any(w in t for t in texts)]
        if not hit:
            missing.append((fid, wanted[:4]))
    assert not missing, missing


# ---------------------------------------------------------------- 6. seal and accessibility
def test_6_seal_role_title_and_no_external_content(svgs):
    for fid, svg in svgs.items():
        assert svg.startswith("<svg"), fid
        root = svg[: svg.index(">") + 1]
        assert 'role="img"' in root, fid
        title = re.search(r"<title\b[^>]*>([^<]+)</title>", svg)
        assert title and title.group(1).strip(), fid
        assert any(t.strip() == "PROVISÓRIO" for _, t in _texts(svg)), fid
        for bad in ("<image", "data:", "<script"):
            assert bad not in svg.lower(), (fid, bad)


# ---------------------------------------------------------------- 7. ids prefixed, no collision
def _ids(svg: str) -> list[str]:
    return re.findall(r'\bid="([^"]+)"', svg)


def test_7_ids_are_prefixed_unique_and_resolved(svgs):
    for fid, svg in svgs.items():
        ids = _ids(svg)
        assert ids, fid
        assert len(ids) == len(set(ids)), (fid, "id repetido no mesmo SVG")
        assert all(i.startswith(f"{fid}-") for i in ids), (fid, [i for i in ids if not i.startswith(f"{fid}-")][:5])
        refs = re.findall(r"url\(#([^)]+)\)", svg) + re.findall(r'href="#([^"]+)"', svg)
        refs += [r for attr in re.findall(r'aria-(?:labelledby|describedby)="([^"]+)"', svg) for r in attr.split()]
        assert set(refs) <= set(ids), (fid, sorted(set(refs) - set(ids)))


def test_7_no_id_collision_between_figures(svgs):
    for (a, sa), (b, sb) in combinations(svgs.items(), 2):
        clash = set(_ids(sa)) & set(_ids(sb))
        assert not clash, (a, b, sorted(clash)[:5])


# ---------------------------------------------------------------- REV-06 / REV-01: axis ticks
AXIS_FILL = "fill: #8c8c8c"
GRID_STROKE = "stroke: #2a2a2a"
TICK_STROKE = "stroke: #8a8a8a"
TICK_TOL = 6.0  # px: matplotlib puts the text baseline ~5 px below a horizontal grid line
BARE_NUM = re.compile(r"^[+\-−]?\d+(?:,\d+)?%?$")
SEG = re.compile(r'<path d="M ([\d.]+) ([\d.]+) \s*L ([\d.]+) ([\d.]+) \s*" style="([^"]*)"')

# Figures that must have a numeric axis (lines, calibration, heat map and every bar chart).
AXIS_FIGS = EXPECTED_IDS - {"confusao"}


def _ptbr(t: str) -> float:
    return float(t.strip().rstrip("%").replace("−", "-").replace(".", "").replace(",", "."))


def _axes(svg: str) -> list[tuple[str, list[tuple[float, str]]]]:
    """Tick groups: ('v', [(y, label)]) for a vertical axis, ('h', [(x, label)]) for a horizontal one.

    A tick label is a bare number in the axis grey; a vertical axis is >= 3 of them right-aligned on
    the same x, a horizontal axis is >= 3 of them centred on the same y.
    """
    groups: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for attrs, body in _texts(svg):
        style = attrs.get("style", "")
        if AXIS_FILL not in style or not BARE_NUM.match(body.strip()):
            continue
        if "text-anchor: end" in style:
            groups.setdefault(("v", attrs["x"]), []).append((float(attrs["y"]), body.strip()))
        elif "text-anchor: middle" in style:
            groups.setdefault(("h", attrs["y"]), []).append((float(attrs["x"]), body.strip()))
    return [(k[0], sorted(v)) for k, v in sorted(groups.items()) if len(v) >= 3]


def _reference_lines(svg: str) -> tuple[list[float], list[float]]:
    """(y of horizontal lines, x of vertical lines): grid lines, or tick marks if there is no grid."""
    segs = SEG.findall(svg)
    grid = [s for s in segs if GRID_STROKE in s[4]]
    use = grid or [s for s in segs if TICK_STROKE in s[4]]
    hy = [float(s[1]) for s in use if s[1] == s[3]]
    vx = [float(s[0]) for s in use if s[0] == s[2]]
    return hy, vx


def _axis_problems(svg: str) -> list[str]:
    probs: list[str] = []
    hy, vx = _reference_lines(svg)
    for kind, ticks in _axes(svg):
        labels = [t for _, t in ticks]
        if len(set(labels)) != len(labels):
            probs.append(f"{kind}: rótulos repetidos {labels}")
        vals = [_ptbr(t) for t in labels]
        steps = {round(b - a, 6) for a, b in pairwise(vals)}
        if len(steps) != 1 or 0 in steps:
            probs.append(f"{kind}: fora de progressão aritmética {labels}")
        lines = hy if kind == "v" else vx
        for pos, lab in ticks:
            if not any(abs(pos - ln) <= TICK_TOL for ln in lines):
                probs.append(f"{kind}: rótulo {lab} em {pos} sem linha de grade")
    return probs


@pytest.fixture(scope="module", params=["committed", "generated"])
def any_svgs(request, svgs) -> dict[str, str]:
    if request.param == "generated":
        return svgs
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(COMMITTED.glob("*.svg"))}


def test_rev06_every_numeric_figure_has_an_axis(any_svgs):
    without = sorted(fid for fid in AXIS_FIGS if not _axes(any_svgs[fid]))
    assert not without, without


def test_rev06_ticks_distinct_arithmetic_and_on_grid(any_svgs):
    bad = {fid: p for fid in sorted(AXIS_FIGS) if (p := _axis_problems(any_svgs[fid]))}
    assert not bad, bad


def _fake_axis(labels: list[str], grid: list[float], offset: float = 5.0) -> str:
    st = f"text-anchor: end; {AXIS_FILL}"
    texts = "".join(f'<text style="{st}" x="56" y="{y + offset}">{t}</text>' for t, y in zip(labels, grid, strict=True))
    paths = "".join(f'<path d="M 64 {y} \nL 326 {y} \n" style="fill: none; {GRID_STROKE}"/>' for y in grid)
    return f"<svg>{paths}{texts}</svg>"


def test_rev06_control_old_defect_is_rejected():
    grid = [312.0, 252.0, 192.0, 132.0, 72.0]
    ok = _fake_axis(["0%", "25%", "50%", "75%", "100%"], grid)
    assert _axes(ok) and not _axis_problems(ok)
    old = _fake_axis(["0%", "0%", "1%", "2%", "2%"], grid)
    probs = _axis_problems(old)
    assert any("repetidos" in p for p in probs) and any("progressão" in p for p in probs), probs


def test_rev06_control_label_off_the_grid_is_rejected():
    grid = [312.0, 252.0, 192.0, 132.0, 72.0]
    off = _fake_axis(["0%", "25%", "50%", "75%", "100%"], grid, offset=20.0)
    assert any("sem linha de grade" in p for p in _axis_problems(off))


# ---------------------------------------------------------------- REV-03: no <style>, root strokes
def test_rev03_no_style_element_and_root_stroke_attributes(any_svgs):
    for fid, svg in any_svgs.items():
        assert "<style" not in svg.lower(), fid
        root = svg[: svg.index(">") + 1]
        assert 'stroke-linejoin="round"' in root, fid
        assert 'stroke-linecap="butt"' in root, fid


# ---------------------------------------------------------------- REV-04: floor seal uses pt_int
def test_rev04_calibration_seal_formats_the_floor_with_pt_int(tmp_path):
    data = _copy_fixture(tmp_path)
    f = data / REL / "graficos.json"
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc["floor"] = 1000
    f.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    write_figures(data, tmp_path / "out", REL)
    svg = (tmp_path / "out" / REL / "figures" / "calibracao.svg").read_text(encoding="utf-8")
    texts = " ".join(t for _, t in _texts(svg))
    assert "Piso de 1.000 objetos por faixa" in texts
    assert "menos de 1" not in texts
