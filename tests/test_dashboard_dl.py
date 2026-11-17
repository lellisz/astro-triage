"""Comparação tab (LightGBM x deep learning): numbers, rule, priors, escape, wording, contrast."""

from __future__ import annotations

import ast
import json
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from astro_triage.dashboard import (
    charts_dl,
    data_dl,
    html_dl,
    style_kit,
    theme,
    views_dl,
)
from astro_triage.evaluation import dl_compare
from astro_triage.evaluation.runner import GRID

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"
TESTS = Path(__file__).resolve().parent
TAB_FILES = ("views_dl.py", "html_dl.py")
PURE = {"min": 0.0448, "median": 0.0455, "max": 0.0461}
EVIL = "<script>alert(1)</script>"


def _cls(d, lo, hi, delta, verdict):
    return {
        "d": d,
        "ci": [lo, hi],
        "na": False,
        "delta": delta,
        "class_verdict": verdict,
    }


def _cell(role, verdict, qso, gal):
    return {
        "role": role,
        "verdict": verdict,
        "classes": {
            "QSO": {**qso, "purity_seed": {"ftt_full": PURE, "lgbm_full": PURE}},
            "GALAXY": gal,
        },
    }


RESULTS = {
    "compare": {
        "pairs": {
            "ftt_full-lgbm_full": {
                "P2": {
                    "0.80": _cell(
                        "headline",
                        "tie: LightGBM stays",
                        _cls(0.0014, 0.0006, 0.0022, 0.0005, "win"),
                        _cls(0.0111, -0.0280, 0.0582, 0.0047, "tie"),
                    )
                }
            },
            "mlp_full-lgbm_full": {
                "P2": {
                    "0.80": _cell(
                        "headline",
                        "tie: LightGBM stays",
                        _cls(-0.0014, -0.0024, -0.0005, 0.0005, "worse"),
                        _cls(-0.0100, -0.0500, 0.0300, 0.0047, "tie"),
                    )
                }
            },
            "ftt_full-ftt_noind": {
                "P2": {
                    "0.80": _cell(
                        "front3",
                        None,
                        _cls(0.0203, 0.0101, 0.0305, 0.0005, None),
                        _cls(0.0, -0.1, 0.1, 0.0047, None),
                    )
                }
            },
        }
    }
}


@pytest.fixture
def rel(tmp_path):
    out = tmp_path / "dl" / "eval"
    out.mkdir(parents=True)
    (out / "results_dl.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    return SimpleNamespace(root=tmp_path)


class FakeSt:
    def __init__(self):
        self.chunks: list[str] = []

    def markdown(self, text, **_):
        self.chunks.append(text)

    def plotly_chart(self, *_, **__):
        pass

    def expander(self, *_, **__):
        return nullcontext()


def _page(rel, tech, monkeypatch, results=None):
    fake = FakeSt()
    monkeypatch.setattr(views_dl, "st", fake)
    monkeypatch.setattr(views_dl, "tech_on", lambda: tech)
    monkeypatch.setattr(
        views_dl.DC, "load_results_dl", lambda r: results or data_dl.load_results_dl(r)
    )
    monkeypatch.setattr(
        views_dl.DC, "load_risk_coverage_dl", data_dl.load_risk_coverage_dl
    )
    views_dl.dl_page(rel)
    return "\n".join(fake.chunks)


def test_a_numbers_and_verdicts_match_json(rel, monkeypatch):
    html = _page(rel, True, monkeypatch)
    for shown in (
        "+0,0014",
        "[0,0006; 0,0022]",
        "0,0005",
        "vitória na classe",
        "+0,0111",
        "[-0,0280; 0,0582]",
        "pior na classe",
        "-0,0014",
    ):
        assert shown in html
    assert html.count("empate: fica o LightGBM") == 2  # FTT wins QSO but the pair ties
    assert "empate" in html and "0.0014" not in html and "+0.0014" not in html


def test_a_pair_verdict_not_taken_from_class(rel):
    res = data_dl.load_results_dl(rel)
    verdicts = views_dl.pair_verdicts(res, "P2", 0.8)
    assert (
        html_dl.pair_verdict(verdicts["ftt_full-lgbm_full"])
        == "empate: fica o LightGBM"
    )
    ftt_qso = next(
        c
        for c in views_dl.cells(res, "P2", 0.8)
        if c.pair == "ftt_full-lgbm_full" and c.cls == "QSO"
    )
    assert html_dl.class_verdict(ftt_qso.verdict) == "vitória na classe"
    assert html_dl.class_verdict(ftt_qso.verdict) != "vitória do par"


def test_pair_verdict_uses_the_producer_vocabulary():
    # dl_compare.pair_verdict writes "X wins" / "tie: LightGBM stays"; a real win must not
    # fall into "não disponível".
    assert html_dl.pair_verdict("X wins") == "vitória do par"
    assert html_dl.pair_verdict("tie: LightGBM stays") == "empate: fica o LightGBM"
    assert html_dl.pair_verdict("win") == "vitória do par"
    assert html_dl.pair_verdict("sem sentido") == html_dl.UNAVAILABLE
    assert html_dl.pair_verdict(None) == html_dl.UNAVAILABLE


def test_rev01_headline_qso_row_has_class_label_not_pair_label(rel, monkeypatch):
    html = _page(rel, True, monkeypatch)
    row = next(
        r
        for r in re.findall(r"<tr>.*?</tr>", html, re.DOTALL)
        if "FT-Transformer completo × LightGBM completo" in r and "Quasar" in r
    )
    assert "vitória na classe" in row and "vitória do par" not in row


def test_rev02_rule_text_states_real_class_criterion(rel, monkeypatch):
    html = _page(rel, True, monkeypatch)
    assert "limite inferior" in html and "intervalo inteiro" not in html


def test_rev03_main_chart_only_headline_pairs_with_readable_labels(rel, monkeypatch):
    figs: list = []
    fake = FakeSt()
    fake.plotly_chart = lambda fig, *_, **__: figs.append(fig)
    monkeypatch.setattr(views_dl, "st", fake)
    monkeypatch.setattr(views_dl, "tech_on", lambda: False)
    monkeypatch.setattr(views_dl.DC, "load_results_dl", data_dl.load_results_dl)
    views_dl.dl_page(rel)
    fig = figs[0]
    ys = [y for t in fig.data for y in t.y]
    assert len(ys) == 2 * 2  # two headline pairs x two classes, not the front pair
    assert not any("_full" in y or "_noind" in y for y in ys)
    hover = [h for t in fig.data for h in t.hovertext]
    assert not any("_full" in h for h in hover)


def test_rev03_risk_legend_uses_readable_system_names():
    import pandas as pd

    frame = pd.DataFrame(
        {"seed": ["mean"] * 2, "system": ["ftt_full"] * 2,
         "coverage": [0.5, 1.0], "accuracy": [0.9, 0.8]}
    )  # fmt: skip
    fig = charts_dl.risk_coverage_figure(frame, ["ftt_full"])
    assert "_full" not in fig.data[0].name and "FT-Transformer" in fig.data[0].name


def test_b_obsolete_strings_banned():
    banned = [
        "Ainda sem resultado",
        "(proposta, ADR",
        "só foi medida em validação",
        "No DL result appears",
        "não está congelado",
    ]
    # Exception: html_overview_plots.py says LightGBM Phase-1 purity was only measured
    # on validation; that is true and unrelated to the DL tab.
    allowed = {("html_overview_plots.py", "só foi medida em validação")}
    for path in DASH.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for s in banned:
            if s in text:
                assert (path.name, s) in allowed, (path.name, s)
    for name in TAB_FILES:
        assert "equivalente" not in (DASH / name).read_text(encoding="utf-8").lower(), (
            name
        )


@pytest.mark.parametrize("tech", [False, True])
def test_c_unavailable_without_dl_eval(tmp_path, monkeypatch, tech):
    empty = SimpleNamespace(root=tmp_path)
    assert data_dl.load_results_dl(empty) is None
    html = _page(empty, tech, monkeypatch)
    assert "não disponível" in html and "0,0" not in html


def _digit_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and id(n) not in doc
        ):
            text = re.sub(r"\b(19|20)\d{2}\b", "", n.value)  # citation years
            text = re.sub(r"\b\d+d\b|[.:]\d+[fg%]", "", text)  # format specs: 04d, .2f
            text = re.sub(
                r"\b[A-Za-z_]+\d*[A-Za-z_]*\b", "", text
            )  # ids: P0, front3, GCUT-RC3
            if re.search(r"\d", text):
                yield n.lineno, n.value[:60]


def test_d_no_deploy_trace_and_no_digit_literals(rel, monkeypatch):
    html = _page(rel, True, monkeypatch)
    assert not re.search(r"deploy|trace", html, re.IGNORECASE)
    for name in TAB_FILES:
        assert not list(_digit_literals(DASH / name)), name


def test_d_existing_digit_test_does_not_cover_the_tab():
    text = (TESTS / "test_dashboard_phase2_charts.py").read_text(encoding="utf-8")
    assert all(f'"{name}"' not in text for name in TAB_FILES)


def test_d_digit_scan_detects_a_hand_written_number(tmp_path):
    bad = tmp_path / "x.py"
    bad.write_text('T = "O par ganha em 42 casos"\n', encoding="utf-8")
    assert list(_digit_literals(bad))


def test_e_json_text_is_escaped():
    row = views_dl.PriorRow("P0", {"STAR": 0.5})
    cell = views_dl.Cell(
        f"{EVIL}-lgbm_full", EVIL, "QSO", 0.1, 0.0, 0.2, 0.01, EVIL, False
    )
    qso = [views_dl.QsoPurity(EVIL, 0.1, 0.2, 0.3)]
    for html in (
        html_dl.headline_table_html([cell], "P2", 0.8),
        html_dl.verdict_html({EVIL: EVIL}, "P2", 0.8),
        html_dl.qso_purity_html(qso, "P2", 0.8, 0.001),
        html_dl.fronts_html([cell], "P2", 0.8),
        html_dl.sensitivity_html([(EVIL, "0.80", [cell])]),
        html_dl.priors_html([row], [EVIL]),
    ):
        assert "<script>" not in html


def test_e_escaped_form_is_visible():
    cell = views_dl.Cell("a-b", "headline", EVIL, 0.1, 0.0, 0.2, 0.01, "win", False)
    assert "&lt;script&gt;" in html_dl.headline_table_html([cell], "P2", 0.8)


def test_f_rule_text_follows_dl_compare_constants(monkeypatch):
    before = html_dl.rule_html(views_dl.rule_facts())
    assert "2 ×" in before and "ddof = 1" in before and "5 sementes" in before
    monkeypatch.setattr(dl_compare, "DELTA_SD_MULTIPLIER", 3.5)
    monkeypatch.setattr(dl_compare, "DELTA_DDOF", 0)
    monkeypatch.setattr(dl_compare, "N_SEEDS", 7)
    monkeypatch.setattr(dl_compare, "BLOCK_DIVISOR", 32)
    after = html_dl.rule_html(views_dl.rule_facts())
    assert "3,5 ×" in after and "ddof = 0" in after and "7 sementes" in after
    assert "por 32" in after and after != before


def test_g_priors_legend_and_values_from_grid(monkeypatch):
    html = html_dl.priors_html(views_dl.prior_rows(), ["B", "C"])
    assert "Priors de classificador, não frações medidas do céu" in html
    for pid in ("P0", "P1", "P2", "P3"):
        assert f"<td>{pid}</td>" in html
    assert "99,7871%" in html  # P1 STAR from runner.GRID
    patched = {**GRID, "P1": {"STAR": 0.5, "QSO": 0.25, "GALAXY": 0.25}}
    monkeypatch.setattr(views_dl, "GRID", patched)
    assert "50,0000%" in html_dl.priors_html(views_dl.prior_rows(), [])


def test_h_contrast_of_tab_palette():
    css = style_kit.kit_css()
    for sel in (".at-lead", ".at-cap", ".at-big.at-accent", ".at-tt th"):
        color = re.search(re.escape(sel) + r" \{[^}]*?\bcolor: (#[0-9a-fA-F]{6})", css)
        assert color, sel
        for bg in (theme.BG, theme.SECONDARY_BG):
            assert (
                theme.contrast_ratio(color.group(1), bg) >= theme.MIN_TEXT_CONTRAST
            ), sel
    assert theme.contrast_ratio(charts_dl.TEXT, theme.BG) >= theme.MIN_TEXT_CONTRAST
    for mark in (charts_dl.ACCENT, charts_dl.SOFT):
        assert theme.contrast_ratio(mark, theme.BG) >= theme.MIN_GRAPHIC_CONTRAST


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1]
        / "data/interim/full_v0/dl/eval/results_dl.json"
    ).is_file(),
    reason="data/ is not versioned; integration needs the local full_v0 DL evaluation",
)
def test_real_release_headline_matches_json():
    root = Path(__file__).resolve().parents[1] / "data/interim/full_v0"
    res = data_dl.load_results_dl(SimpleNamespace(root=root))
    raw = json.loads((root / "dl/eval/results_dl.json").read_text(encoding="utf-8"))
    ftt = next(
        c
        for c in views_dl.cells(res, "P2", 0.8)
        if c.pair == "ftt_full-lgbm_full" and c.cls == "QSO"
    )
    want = raw["compare"]["pairs"]["ftt_full-lgbm_full"]["P2"]["0.80"]["classes"]["QSO"]
    assert ftt.d == want["d"] and html_dl._num(ftt.d, True) == "+0,0014"
