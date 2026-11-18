"""Comparação tab: wording bans, digit-free sources, escape, rule constants, contrast.

The data model, the texts and the rulers are covered by ``test_dashboard_comparacao.py``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from astro_triage.dashboard import (
    charts_base,
    data_dl,
    html_dl,
    style_kit,
    theme,
    verdict_rule,
)
from astro_triage.evaluation import dl_compare

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"
TESTS = Path(__file__).resolve().parent
TAB_FILES = (
    "views_dl.py",
    "html_dl.py",
    "data_dl.py",
    "verdict_rule.py",
    "compare_model.py",
    "compare_text.py",
)
EVIL = "<script>alert(1)</script>"


def test_b_obsolete_strings_banned():
    banned = [
        "Ainda sem resultado",
        "(proposta, ADR",
        "só foi medida em validação",
        "No DL result appears",
        "não está congelado",
    ]
    allowed = {("html_overview_plots.py", "só foi medida em validação")}
    for path in DASH.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for s in banned:
            if s in text:
                assert (path.name, s) in allowed, (path.name, s)
    for name in TAB_FILES:
        assert "equivalente" not in (DASH / name).read_text(encoding="utf-8").lower()


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
            text = re.sub(r"\butf-8\b", "", n.value)  # codec name, not a result
            text = re.sub(r"\b(19|20)\d{2}\b", "", text)
            text = re.sub(r"\b\d+d\b|[.:]\d+[fg%]", "", text)
            text = re.sub(r"\b[A-Za-z_]+\d*[A-Za-z_]*\b", "", text)
            if re.search(r"\d", text):
                yield n.lineno, n.value[:60]


def test_d_no_digit_literals_in_the_tab_sources():
    for name in TAB_FILES:
        assert not list(_digit_literals(DASH / name)), name


def test_d_digit_scan_detects_a_hand_written_number(tmp_path):
    bad = tmp_path / "x.py"
    bad.write_text('T = "O par ganha em 42 casos"\n', encoding="utf-8")
    assert list(_digit_literals(bad))


def test_e_text_is_escaped():
    assert "<script>" not in html_dl.error_html(EVIL).replace("&lt;script&gt;", "")
    assert "<script>" not in html_dl.limitations_html([EVIL])


def test_f_rule_text_follows_dl_compare_constants(monkeypatch):
    before = verdict_rule.rule_text()
    assert "2 desvios-padrão (ddof = 1)" in before and "5 sementes" in before
    monkeypatch.setattr(dl_compare, "DELTA_SD_MULTIPLIER", 3.5)
    monkeypatch.setattr(dl_compare, "N_SEEDS", 7)
    after = verdict_rule.rule_text()
    assert "3,5 desvios-padrão" in after and "7 sementes" in after and after != before


def test_h_contrast_of_tab_palette():
    css = style_kit.kit_css()
    for sel in (".at-lead", ".at-cap", ".at-big.at-accent", ".at-tt th"):
        color = re.search(re.escape(sel) + r" \{[^}]*?\bcolor: (#[0-9a-fA-F]{6})", css)
        assert color, sel
        for bg in (theme.BG, theme.SECONDARY_BG):
            assert theme.contrast_ratio(color.group(1), bg) >= theme.MIN_TEXT_CONTRAST
    assert theme.contrast_ratio(charts_base.TEXT, theme.BG) >= theme.MIN_TEXT_CONTRAST


def test_loader_returns_none_without_artifact(tmp_path):
    from types import SimpleNamespace

    assert data_dl.load_results_dl(SimpleNamespace(root=tmp_path)) is None
