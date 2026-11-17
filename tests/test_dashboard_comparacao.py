"""Comparação tab, 'prior ruler' layout: data model, texts, DOM symmetry, ruler units, seal, guards."""

from __future__ import annotations

import copy
import json
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from astro_triage.dashboard import data_dl as DD
from astro_triage.dashboard import html_dl as HD
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard import views_dl
from astro_triage.evaluation import dl_compare

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "src" / "astro_triage" / "dashboard"
REAL = ROOT / "data/interim/full_v0"
REAL_JSON = REAL / "dl/eval/results_dl.json"
needs_real = pytest.mark.skipif(
    not REAL_JSON.is_file(), reason="data/ is not versioned"
)
BANNED = re.compile(
    r"\b(melhor|melhora|ganho|equivalente|verde|vermelho)\b", re.IGNORECASE
)
TIE = "tie: LightGBM stays"


def _cls(d, lo, hi, delta, verdict, **extra):
    return {"d": d, "ci": [lo, hi], "na": False, "delta": delta, "class_verdict": verdict,
            "reportable": True, "purity_seed": {"lgbm_full": {"min": 0.04, "median": 0.05, "max": 0.06}},
            **extra}  # fmt: skip


def _results(
    ftt=("X wins", "win", "win"),
    gal_ci=(-0.0280, 0.0582),
    qso_ci=(0.0006, 0.0022),
):
    def cell(role, token, q, g):
        return {"role": role, "verdict": token, "classes": {"QSO": q, "GALAXY": g}}

    mlp = cell("headline", TIE, _cls(-0.0014, -0.0024, -0.0005, 0.0005, "worse"),
               _cls(-0.0006, -0.0444, 0.0342, 0.0244, "tie"))  # fmt: skip
    f = cell("headline", ftt[0], _cls(0.0014, *qso_ci, 0.0005, ftt[1]),
             _cls(0.0111, *gal_ci, 0.0244, ftt[2]))  # fmt: skip
    sens = cell("sensitivity", TIE, _cls(0.0007, 0.0004, 0.001, 0.0003, "win", ci_l5=[0.0004, 0.001]),
                _cls(0.0126, -0.02, 0.05, 0.0134, "tie"))  # fmt: skip
    return {"compare": {"n_boot": 1000, "pairs": {
        "mlp_full-lgbm_full": {"P2": {"0.80": mlp}},
        "ftt_full-lgbm_full": {"P2": {"0.80": f, "0.90": sens}, "P1": {"0.80": sens}},
    }}}  # fmt: skip


def _page(results):
    head = DD.headline(results)
    seal = HD.seal_html("full_v0", "Proposed", True, DD.below_floor(head), True)
    return HD.page_html(head, DD.sensitivity(results, head), seal)


def _text(html):
    return re.sub(r"<[^>]+>", " ", html)


def _skeleton(html):
    return re.sub(r">[^<]*<", "><", html)


def _real():
    return json.loads(REAL_JSON.read_text(encoding="utf-8"))


# A1 -- real data
@needs_real
def test_a1_real_headline_values():
    head = DD.headline(DD.load_results_dl(SimpleNamespace(root=REAL)))
    got = {}
    for cell in head.cells:
        assert HD.pair_text(cell) == "Empate: fica o LightGBM."
        for cls, c in cell.classes.items():
            got[(HD.net_name(cell.pair), cls)] = (
                HD.signed(c.d), HD.interval(c.lo, c.hi), c.verdict, HD.plain(c.delta))  # fmt: skip
    V = DD.ClassVerdict
    assert got[("MLP", "QSO")] == ("−0,0014", "[−0,0024; −0,0005]", V.WORSE, "0,0005")
    assert got[("MLP", "GALAXY")] == ("−0,0006", "[−0,0444; +0,0342]", V.TIE, "0,0244")
    assert got[("FTT", "QSO")] == ("+0,0014", "[+0,0006; +0,0022]", V.WIN, "0,0005")
    assert got[("FTT", "GALAXY")] == ("+0,0111", "[−0,0280; +0,0582]", V.TIE, "0,0244")
    assert HD.class_text(V.WORSE) != HD.class_text(V.TIE)


@needs_real
def test_a1_headline_selected_by_role_not_position():
    raw = _real()
    pairs = raw["compare"]["pairs"]
    raw["compare"]["pairs"] = dict(reversed(list(pairs.items())))
    head = DD.headline(raw)
    assert {c.pair for c in head.cells} == {"mlp_full-lgbm_full", "ftt_full-lgbm_full"}
    assert (head.prior, head.level) == ("P2", "0.80")


def test_headline_missing_is_explicit_error():
    raw = _results()
    for p in raw["compare"]["pairs"].values():
        for lv in p.values():
            for c in lv.values():
                if c["role"] == "headline":
                    c["role"] = "sensitivity"
    with pytest.raises(DD.DlDataError, match="headline"):
        DD.headline(raw)


def test_producer_grammar_is_parsed():
    win = dl_compare.pair_verdict({"QSO": "win", "GALAXY": "win"})
    tie = dl_compare.pair_verdict({"QSO": "win", "GALAXY": "tie"})
    assert DD.parse_pair_verdict(win, "x") is DD.PairVerdict.WIN
    assert DD.parse_pair_verdict(tie, "x") is DD.PairVerdict.TIE


# A2 -- regression of 503b0fd and strict tokens
def test_a2_system_wins_token_renders_pair_win():
    html = _page(_results())
    assert "FTT vence o par: vitória nas duas classes." in html
    assert "inconsistência" not in html


def test_a2_class_win_with_pair_tie_renders_tie():
    html = _page(_results(ftt=(TIE, "win", "tie")))
    assert html.count("Empate: fica o LightGBM.") == 2
    assert "vence o par" not in html


@pytest.mark.parametrize(
    "token",
    [
        "ftt_full beats", "wins", "win", "xwins", "tie", "",
        "LightGBM wins", "lgbm_full wins", "ftt_full wins", "x wins", "X  wins",
        " X wins", "X wins ", "TIE: LightGBM stays", "tie:LightGBM stays", None,
    ],
)  # fmt: skip
def test_a2_unknown_pair_token_fails(token):
    with pytest.raises(DD.DlDataError):
        DD.headline(_results(ftt=(token, "win", "win")))


def test_a2_x_wins_names_the_net_from_the_pair_not_the_token():
    html = _page(_results())
    assert "FTT vence o par: vitória nas duas classes." in html
    assert "MLP vence o par" not in html


def test_a2_x_wins_with_mlp_pair_renders_mlp_win():
    raw = _results()
    pairs = raw["compare"]["pairs"]
    pairs["mlp_full-lgbm_full"]["P2"]["0.80"] = pairs["ftt_full-lgbm_full"]["P2"][
        "0.80"
    ]
    pairs["ftt_full-lgbm_full"]["P2"]["0.80"] = _results(ftt=(TIE, "win", "tie"))[
        "compare"
    ]["pairs"]["ftt_full-lgbm_full"]["P2"]["0.80"]
    html = _page(raw)
    assert "MLP vence o par: vitória nas duas classes." in html
    assert "FTT vence o par" not in html


@pytest.mark.parametrize(
    "token", ["draw", "WIN", "wins", None, "n/d", "n/a", "N/A", " win"]
)
def test_a2_unknown_class_token_fails(token):
    with pytest.raises(DD.DlDataError):
        DD.headline(_results(ftt=(TIE, token, "tie")))


def test_a2_unknown_token_is_visible_in_the_page(monkeypatch):
    chunks = []
    fake = SimpleNamespace(
        markdown=lambda t, **_: chunks.append(t), expander=lambda *_: nullcontext()
    )
    monkeypatch.setattr(views_dl, "st", fake)
    monkeypatch.setattr(
        views_dl.DC, "load_results_dl", lambda r: _results(ftt=("zzz", "win", "win"))
    )
    views_dl.dl_page(SimpleNamespace(name="full_v0", root=Path("nowhere")))
    out = " ".join(chunks)
    assert "ilegíveis" in out and "zzz" in out and "vence o par" not in out


def test_a2_not_reportable_is_na_and_counts_as_tie():
    raw = _results(ftt=(TIE, "win", "tie"))
    raw["compare"]["pairs"]["ftt_full-lgbm_full"]["P2"]["0.80"]["classes"]["QSO"][
        "na"
    ] = True
    head = DD.headline(raw)
    ftt = next(c for c in head.cells if c.pair.startswith("ftt"))
    assert ftt.classes["QSO"].verdict is DD.ClassVerdict.NA
    assert not ftt.inconsistent and DD.below_floor(head) == 1
    assert "n/a" in HD.card_html(ftt)


def test_a2_inconsistent_artifact_warns():
    html = _page(_results(ftt=("X wins", "win", "tie")))
    assert "inconsistência no artefato" in html


# A3 -- same DOM in both cards
def _cards(raw):
    head = DD.headline(raw)
    return [HD.card_html(c) for c in head.cells]


def _synthetic_variants():
    na = _results(ftt=(TIE, "win", "tie"))
    na["compare"]["pairs"]["ftt_full-lgbm_full"]["P2"]["0.80"]["classes"]["QSO"][
        "na"
    ] = True
    return [
        _results(),
        _results(ftt=(TIE, "win", "tie")),
        na,
        _results(ftt=("X wins", "win", "tie")),
    ]


def test_a3_cards_have_the_same_structure():
    for raw in _synthetic_variants():
        a, b = _cards(raw)
        assert _skeleton(a) == _skeleton(b)


@needs_real
def test_a3_real_cards_have_the_same_structure():
    a, b = _cards(_real())
    assert _skeleton(a) == _skeleton(b)


# A4 -- honesty and banned words
def test_a4_wide_interval_states_lack_of_power_only_when_wide():
    wide = _page(_results(ftt=(TIE, "win", "tie")))
    assert HD.NO_POWER in wide
    inside = _results(ftt=(TIE, "win", "tie"), gal_ci=(-0.0003, 0.0003))
    inside["compare"]["pairs"]["mlp_full-lgbm_full"]["P2"]["0.80"]["classes"]["GALAXY"][
        "ci"
    ] = [-0.0003, 0.0003]
    html = _page(inside)
    assert HD.WITHIN_DELTA in html and HD.NO_POWER not in html


def test_a4_mlp_quasar_worse_is_never_a_tie():
    head = DD.headline(_results())
    mlp = next(c for c in head.cells if c.pair.startswith("mlp"))
    assert (
        HD.class_text(mlp.classes["QSO"].verdict)
        == "pior na classe (descritivo; não decide o par)"
    )
    card = HD.card_html(mlp)
    qso_block = card.split("hs-card-r")[1]
    assert "pior na classe" in qso_block and "empate na classe" not in qso_block


def test_a4_no_banned_terms():
    pages = [_page(r) for r in _synthetic_variants()]
    rule = views_dl.rule_facts()
    pages += [
        HD.method_html(rule, 1000),
        HD.descriptive_html([], "P2", "0.80"),
        HD.error_html("x"),
    ]
    if REAL_JSON.is_file():
        pages.append(_page(_real()))
        pages.append(
            HD.limitations_html(DD.load_limitations(SimpleNamespace(root=REAL)))
        )
    for html in pages:
        assert not BANNED.search(_text(html)), BANNED.search(_text(html))


# A5 -- numbers and axis labels
def _numbers(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(f"{abs(obj):.4f}".replace(".", ","))
        out.add(f"{abs(obj) * 100:.2f}".replace(".", ","))


@needs_real
def test_a5_every_rendered_number_is_in_the_artifact():
    raw = _real()
    allowed = {"95", "0,80", "0,90", "0", "5", "0,00", "0006"}
    _numbers(raw, allowed)
    html = _page(DD.load_results_dl(SimpleNamespace(root=REAL)))
    html = re.sub(r"<[^>]*>", " ", re.sub(r"aria-label='[^']*'", "", html))
    for tok in re.findall(r"(?<![\w])\d+(?:,\d+)?", html):
        assert tok in allowed, tok


def test_a5_axis_labels_only_minus_delta_zero_plus_delta():
    html = _page(_results())
    axes = re.findall(r"<div class='hs-ax[^']*' aria-hidden='true'>(.*?)</div>", html)
    ruler_axes = [a for a in axes if "δ" in a]
    assert len(ruler_axes) == 2
    for a in ruler_axes:
        assert re.findall(r">([^<]*)</span>", a) == ["−δ", "0", "+δ"]


# A6 -- ruler unit at the smallest delta and interval
def _px(html, cls, track):
    out = []
    for m in re.finditer(
        rf"<div class='{cls}' style='left:([\d.]+)%;width:([\d.]+)%'>", html
    ):
        out.append(float(m.group(2)) / 100 * track)
    return out


def _panels_from(cells):
    by_cls = {}
    for cell in cells:
        for cls, c in cell.classes.items():
            by_cls.setdefault((cell.prior, cell.level, cls), []).append(
                {"name": HD.net_name(cell.pair), "d": c.d, "lo": c.lo, "hi": c.hi, "detail": "x", "delta": c.delta})  # fmt: skip
    return [
        {"title": k, "delta": rows[0]["delta"], "rows": rows}
        for k, rows in by_cls.items()
    ]


def _check_units(panels):
    for panel in panels:
        html = hs.delta_ruler(panel)
        for viewport in (hs.NARROW_VIEWPORT_PX, hs.WIDE_VIEWPORT_PX):
            track = hs.track_px(viewport)
            bands = _px(html, "band", track)
            if bands:
                assert min(bands) >= hs.BAND_MIN_PX - 1e-9, (panel["title"], viewport)
            else:
                assert html.count("class='dm'") == 2
            assert all(w >= hs.CI_MIN_PX for w in _px(html, "ci", track)), (
                panel["title"],
                viewport,
            )
    css = hs.system_css()
    assert f"width:{hs.POINT_PX}px;height:{hs.POINT_PX}px" in css
    assert (
        f"min-width:{hs.BAND_MIN_PX}px" in css and f"min-width:{hs.CI_MIN_PX}px" in css
    )


def test_a6_units_on_synthetic_levels():
    raw = _results()
    head = DD.headline(raw)
    cells = [*head.cells, *DD.sensitivity(raw, head)]
    _check_units(_panels_from(cells))


@needs_real
def test_a6_units_on_real_data_including_c90():
    raw = DD.load_results_dl(SimpleNamespace(root=REAL))
    head = DD.headline(raw)
    cells = [*head.cells, *DD.sensitivity(raw, head)]
    assert any(c.level == "0.90" for c in cells)
    _check_units(_panels_from(cells))


def test_a6_tiny_delta_draws_two_labelled_marks():
    panel = {
        "title": "t",
        "delta": 1e-7,
        "rows": [{"name": "a", "d": 0.01, "lo": -0.02, "hi": 0.03, "detail": "x"}],
    }
    html = hs.delta_ruler(panel)
    assert "class='band'" not in html and html.count("class='dm'") == 2
    assert "−δ" in html and "+δ" in html


# A7 -- seal
def test_a7_seal_reason_is_derived():
    s = HD.seal_html("full_v0", "Proposed", True, 0, True)
    assert "PROVISÓRIO · full_v0" in s
    assert "ADR 0006 Proposta (aceitação aberta; o freeze existe)" in s
    assert "piso" not in s
    assert "abaixo do piso de relato" in HD.seal_html(
        "full_v0", "Proposed", True, 2, True
    )
    assert "ADR 0006: n/a" in HD.seal_html("full_v0", None, True, 0, False)
    assert "limitações do results_dl: n/a" in HD.seal_html(
        "full_v0", "Proposed", True, 0, False
    )


def test_a7_reads_fail_to_na(tmp_path):
    assert DD.adr_status(tmp_path) is None
    assert DD.load_limitations(SimpleNamespace(root=tmp_path)) is None
    (tmp_path / "0006-x.md").write_text(
        "# t\n\n- Status: **Accepted** (x)\n", encoding="utf-8"
    )
    assert DD.adr_status(tmp_path) == "Accepted"
    assert "aceitação aberta" not in HD.seal_html("r", "Accepted", True, 0, True)


def test_a7_real_adr_status_and_lang():
    assert DD.adr_status() == "Proposed"
    assert "lang='pt-BR'" in _page(_results())


# A9 -- guards
@pytest.mark.parametrize(
    "name", ["data_dl.py", "html_dl.py", "views_dl.py", "html_system.py"]
)
def test_a9_tab_never_touches_the_test_exposure(name):
    text = (DASH / name).read_text(encoding="utf-8")
    assert "evaluate_dl" not in text and "test_opened" not in text
    assert "subprocess" not in text


def test_a9_page_does_not_write(tmp_path, monkeypatch):
    before = {p for p in tmp_path.rglob("*")}
    _page(_results())
    assert {p for p in tmp_path.rglob("*")} == before
    assert copy.deepcopy(_results()) == _results()
