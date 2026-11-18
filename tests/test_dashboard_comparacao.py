"""Comparação tab, 'prior ruler' layout: data model, texts, DOM symmetry, ruler units, seal, guards."""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from astro_triage.dashboard import compare_model, verdict_rule, views_dl
from astro_triage.dashboard import data_dl as DD
from astro_triage.dashboard import html_dl as HD
from astro_triage.dashboard import html_system as hs
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
    return views_dl.page_html(compare_model.build(results), "")


def _text(html):
    return re.sub(r"<[^>]+>", " ", html)


def _real():
    return json.loads(REAL_JSON.read_text(encoding="utf-8"))


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


def test_a4_no_banned_terms():
    pages = [
        HD.hs.method_note(verdict_rule.rule_text()),
        views_dl.page_html(compare_model.build(_results(ftt=(TIE, "win", "tie"))), "x"),
        HD.descriptive_html([], "P2", "0.80"),
        HD.error_html("x"),
    ]
    if REAL_JSON.is_file():
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
        out.add(f"{abs(obj):.3f}".replace(".", ","))


def _assert_numbers_in_artifact(raw, html):
    allowed = {"95", "0,80", "0,90", "0", "5", "0,00", "0006"}
    # method-note constants come from the producer, not from the artifact
    allowed |= {
        str(int(x))
        for x in (
            dl_compare.DELTA_SD_MULTIPLIER,
            dl_compare.DELTA_DDOF,
            dl_compare.N_SEEDS,
            dl_compare.BLOCK_LEVEL,
            dl_compare.BLOCK_DIVISOR,
            dl_compare.MIN_BLOCKS,
        )
    }
    _numbers(raw, allowed)
    html = re.sub(r"<[^>]*>", " ", re.sub(r"aria-label='[^']*'", "", html))
    for tok in re.findall(r"(?<![\w])\d+(?:,\d+)?", html):
        assert tok in allowed, tok


def test_a5_every_rendered_number_is_in_the_artifact():
    raw = _results(ftt=(TIE, "win", "tie"))
    _assert_numbers_in_artifact(raw, _page(raw))


@needs_real
def test_a5_every_rendered_number_is_in_the_real_artifact():
    raw = DD.load_results_dl(SimpleNamespace(root=REAL))
    _assert_numbers_in_artifact(_real(), _page(raw))


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
    s = HD.reasons_html("Proposed", True, 0, True)
    assert "PROVISÓRIO" not in s  # the seal is printed once, by the shell header
    assert "protocolo da comparação em proposta (aceitação aberta; o freeze existe)" in s
    assert "piso" not in s
    assert "abaixo do piso de relato" in HD.reasons_html("Proposed", True, 2, True)
    assert "protocolo da comparação: n/a" in HD.reasons_html(None, True, 0, False)
    assert "limitações do results_dl: n/a" in HD.reasons_html(
        "Proposed", True, 0, False
    )


def test_comparison_page_does_not_repeat_the_shell_seal():
    reasons = HD.reasons_html("Proposed", True, 0, True)
    page = views_dl.page_html(
        compare_model.build(_results(ftt=(TIE, "win", "tie"))), reasons
    )
    assert hs.seal("full_v0") not in page and "PROVISÓRIO" not in page


def test_a7_reads_fail_to_na(tmp_path):
    assert DD.adr_status(tmp_path) is None
    assert DD.load_limitations(SimpleNamespace(root=tmp_path)) is None
    (tmp_path / "0006-x.md").write_text(
        "# t\n\n- Status: **Accepted** (x)\n", encoding="utf-8"
    )
    assert DD.adr_status(tmp_path) == "Accepted"
    assert "aceitação aberta" not in HD.reasons_html("Accepted", True, 0, True)


def test_a7_adr_status_first_word_and_file_choice(tmp_path):
    (tmp_path / "0006-a.md").write_text(
        "# t\n- Status: **Closed without x**\n", encoding="utf-8"
    )
    assert DD.adr_status(tmp_path) == "Closed"
    (tmp_path / "0006-a.md").write_text("- Status: Accepted\n", encoding="utf-8")
    (tmp_path / "0007-b.md").write_text("- Status: Proposed\n", encoding="utf-8")
    assert DD.adr_status(tmp_path) == "Accepted"


def test_a7_real_adr_status_and_lang():
    assert DD.adr_status() == "Proposed"


# A9 -- guards
@pytest.mark.parametrize(
    "name", ["data_dl.py", "html_dl.py", "views_dl.py", "html_system.py"]
)
def test_a9_tab_never_touches_the_test_exposure(name):
    text = (DASH / name).read_text(encoding="utf-8")
    assert "evaluate_dl" not in text and "test_opened" not in text
    assert "subprocess" not in text
