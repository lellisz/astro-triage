"""Phase 2 dashboard: dumbbell, error bars, text helpers, no hand-written metrics."""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from astro_triage.dashboard import charts_model as CM

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"


def _pt(key, pur, com, dashed=False):
    return {
        "key": key,
        "label": f"L-{key}",
        "purity": pur,
        "completeness": com,
        **({"dashed": True} if dashed else {}),
    }


THREE = [_pt("a", 0.7, 0.9), _pt("v", 0.5, 0.8, True), _pt("m", 0.6, 0.8)]


@pytest.mark.parametrize("points", [THREE, [THREE[0], THREE[2]]])
def test_dumbbell_series_names_and_markers(points):
    fig = CM.dumbbell(points)
    names = [t.name for t in fig.data]
    assert all(n and "trace" not in n.lower() for n in names)
    by_name = {t.name: t for t in fig.data}
    assert by_name["Pureza"].marker.symbol == "circle"
    assert by_name["Completude"].marker.symbol == "circle-open"
    assert len(by_name["Pureza"].x) == len(points)
    assert len(fig.data) == len(points) + 2  # links + 2 markers
    assert fig.layout.xaxis.tickformat == ".0%"
    assert not fig.layout.showlegend
    assert fig.layout.xaxis.tickangle == 0 and fig.layout.xaxis.side != "top"
    assert fig.layout.height >= 70 * len(points)


def test_dumbbell_dashed_only_for_validation():
    fig = CM.dumbbell(THREE)
    links = [t for t in fig.data if t.name.startswith("Ligação")]
    assert [t.line.dash for t in links] == ["solid", "dash", "solid"]


def test_error_bars_three_pairs_first_highlighted():
    pairs = pd.DataFrame(
        {
            "pair": ["x", "y", "z"],
            "n": [10, 5, 1],
            "n_ab": [8, 3, 1],
            "n_ba": [2, 2, 0],
            "share": [0.6, 0.3, 0.1],
        }
    )
    fig = CM.error_bars(pairs)
    assert len(fig.data) == 1 and len(fig.data[0].x) == 3
    colors = list(fig.data[0].marker.color)
    assert colors[0] != colors[1] and colors[1] == colors[2]


def test_pair_text_pt_format_larger_first():
    row = pd.Series({"n_ab": 2278, "n_ba": 6399, "n": 8677, "share": 0.82})
    assert CM.pair_text(row) == "6.399 + 2.278 = 8.677 · 82%"


def test_pair_text_nan_share():
    row = pd.Series({"n_ab": 0, "n_ba": 0, "n": 0, "share": math.nan})
    assert CM.pair_text(row) == "0 + 0 = 0 · sem dado"


def test_per_100_decimal_comma():
    assert CM.per_100(0.9583) == "95,8"
    assert CM.per_100(1.0) == "100,0"
    assert CM.per_100(0.0) == "0,0"


# --- "nothing hand-written" scan ------------------------------------------------
# Criterion: layout constants (pixels, margins, axis ranges, small ints 0-3, 100 as
# percent scale) are legitimate. A metric is a result of the evaluation, so we flag
# (a) any number equal to a known metric value from the brief, and (b) any string
# literal containing digits, except 4-digit years (citations such as "Delchambre+2023").
# Widget parameters of st.slider are controls, not metrics, and are skipped.
KNOWN_METRICS = {96, 1.6, 72.5, 95.8, 4.9, 8677, 8.677, 6399, 6.399, 82, 2278, 0.82}
FILES = [
    "views_overview.py",
    "charts_model.py",
    "data_overview_page.py",
    "html_overview.py",
    "html_overview_plots.py",
    "html_overview_triage.py",
    "sky_text.py",
    "sky_map_html.py",
    "html_galaxies.py",
    "html_galaxies_tech.py",
    "html_method.py",
    "method_notes.py",
    "html_kit.py",
]
# CSS/markup modules: px sizes and aria levels are layout, not metrics (metric scan still applies).
MARKUP_DIGITS = {"sky_map_html.py", "html_galaxies_tech.py", "html_method.py", "html_kit.py"}


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
