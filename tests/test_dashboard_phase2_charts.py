"""Phase 2 dashboard: dumbbell, error bars, text helpers, no hand-written metrics."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"


def _pt(key, pur, com, dashed=False):
    return {
        "key": key,
        "label": f"L-{key}",
        "purity": pur,
        "completeness": com,
        **({"dashed": True} if dashed else {}),
    }


# --- "nothing hand-written" scan ------------------------------------------------
# Criterion: layout constants (pixels, margins, axis ranges, small ints 0-3, 100 as
# percent scale) are legitimate. A metric is a result of the evaluation, so we flag
# (a) any number equal to a known metric value from the brief, and (b) any string
# literal containing digits, except 4-digit years (citations such as "Delchambre+2023").
# Widget parameters of st.slider are controls, not metrics, and are skipped.
KNOWN_METRICS = {96, 1.6, 72.5, 95.8, 4.9, 8677, 8.677, 6399, 6.399, 82, 2278, 0.82}
FILES = [
    "data_overview_page.py",
    "html_overview.py",
    "sky_text.py",
    "sky_map_html.py",
    "html_galaxies.py",
    "html_method.py",
    "method_notes.py",
    "html_kit.py",
]
# CSS/markup modules: px sizes and aria levels are layout, not metrics (metric scan still applies).
MARKUP_DIGITS = {"sky_map_html.py", "html_method.py", "html_kit.py"}


def _literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "slider":
            skip |= {id(n) for n in ast.walk(node)}
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) not in skip | docstrings:
            yield node


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
            text = re.sub(r"\b(19|20)\d{2}\b", "", n.value)  # citation years
            text = re.sub(r"\.\d[f%]", "", text)  # format specs (.1f, .0%)
            text = re.sub(r"\b[A-Za-z_]+\d*[A-Za-z_]*\b", "", text)  # identifiers/keys
            text = re.sub(r"\b100\b", "", text)  # "a cada 100": the unit, not a metric
            if re.search(r"\d", text):
                bad.append((n.lineno, n.value[:70]))
    assert not bad, bad


def test_charts_do_not_hardcode_retired_blue():
    for path in DASH.glob("charts*.py"):
        assert "8ec5ff" not in path.read_text(encoding="utf-8").lower(), path.name
