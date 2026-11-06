"""Phase 3 'nothing hand-written' sweep over the new triage/limits modules.

Criterion (AST, same walk as test_dashboard_phase2_charts.py; docstrings skipped):
  1. No numeric literal equal to a known metric value (96, 1.6, 72.5, 95.8, 88.9, 99.3,
     32461, 8626, ...). Control parameters (threshold 0.90, completeness 0.80, ADR 6,
     axis floor 0.96, layout sizes) are not metrics and are not in the list.
  2. No string literal contains a digit, after removing citation years, format specs
     (.2f, .1%, :04d), identifiers (DR3, P2, a_all), rgba(...) colours and the unit "100".
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"
FILES = [
    "data_triage.py",
    "charts_triage.py",
    "views_method.py",
    "sky_text.py",
    "sky_map_html.py",
    "html_galaxies.py",
    "html_galaxies_tech.py",
    "html_method.py",
    "method_notes.py",
    "html_kit.py",
]
KNOWN_METRICS = {
    96,
    1.6,
    72.5,
    95.8,
    88.9,
    99.3,
    32461,
    8626,
    4.9,
    8677,
    6399,
    2278,
    82,
    0.725,
    0.958,
    0.889,
    0.993,
    0.016,
}
# CSS/markup modules: px sizes and aria levels are layout, not metrics (metric scan still applies).
MARKUP_DIGITS = {"sky_map_html.py", "html_galaxies_tech.py", "html_kit.py"}


def _literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and id(n) not in docstrings
    ]


@pytest.mark.parametrize("name", FILES)
def test_no_hand_written_metric_numbers(name):
    bad = [
        (n.lineno, n.value)
        for n in _literals(DASH / name)
        if isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
        and n.value in KNOWN_METRICS
    ]
    assert not bad, bad


@pytest.mark.parametrize("name", [f for f in FILES if f not in MARKUP_DIGITS])
def test_no_digits_in_string_literals(name):
    bad = []
    for n in _literals(DASH / name):
        if isinstance(n.value, str):
            text = re.sub(r"rgba?\([^)]*\)", "", n.value)
            text = re.sub(r"\b(19|20)\d{2}\b", "", text)
            text = re.sub(r"\.\d[f%]|:\d+[df]", "", text)
            text = re.sub(r"^0?\d[df]$", "", text)  # bare f-string spec, e.g. "04d"
            text = re.sub(r"\b[A-Za-z_]+\d*[A-Za-z_]*\b", "", text)
            text = re.sub(r"\b100\b", "", text)
            if re.search(r"\d", text):
                bad.append((n.lineno, n.value[:70]))
    assert not bad, bad


def test_scan_detects_a_planted_metric(tmp_path):
    """Falsifiability: the detector must flag a hand-written metric."""
    f = tmp_path / "x.py"
    f.write_text('A = 96\nB = "acerta 95,8%"\n', encoding="utf-8")
    nums = [n.value for n in _literals(f) if isinstance(n.value, int)]
    assert 96 in nums and 96 in KNOWN_METRICS
    assert re.search(r"\d", re.sub(r"\b[A-Za-z_]+\d*[A-Za-z_]*\b", "", "acerta 95,8%"))
