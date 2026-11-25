"""Phase 4 sky map: shared digit detector and the below-floor contrast check."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from astro_triage.dashboard import theme
from astro_triage.dashboard.charts_base import BELOW_FLOOR

DASH_DIR = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"
KNOWN_METRICS = {96, 1.6, 72.5, 95.8, 88.9, 99.3, 32461, 8626, 4.9, 8677, 6399,
                 2278, 82, 0.725, 0.958, 0.889, 0.993, 0.016}  # fmt: skip


def _literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    return [
        n for n in ast.walk(tree) if isinstance(n, ast.Constant) and id(n) not in doc
    ]


_FORMAT_SPEC = re.compile(r"^(0?\d{1,2}[bdeEfFgGnosxX]|,?\.\d[bdeEfFgGnosx%])$")


def has_hand_written_digits(value: str) -> bool:
    """True if a string literal carries digits that could be a hand-written number.

    Ignored on purpose (false positives, not numbers): rgb/rgba colours, years, identifiers
    with digits, bare format specs such as ``02x``, ``04d``, ``.1f``, ``.0%`` (a literal that
    is only width/precision plus type, never a percent like ``95%`` or ``95,8%``), and array
    indices ``customdata[N]`` inside hover templates.
    """
    if _FORMAT_SPEC.match(value):
        return False
    t = re.sub(r"rgba?\([^)]*\)", "", value)
    t = re.sub(r"customdata\[\d+\]", "", t)
    t = re.sub(r"\b(19|20)\d{2}\b", "", t)
    t = re.sub(r"\.\d[f%]|:\d+[df]", "", t)
    t = re.sub(r"\b[A-Za-z_]+\d*[A-Za-z_]*\b", "", t)
    t = re.sub(r"\b100\b", "", t)
    return bool(re.search(r"\d", t))


@pytest.mark.parametrize(
    "planted", ["96", "95,8%", "IC 95%", "taxa 72,5 por cento", "95%"]
)
def test_digit_detector_still_flags_planted_numbers(planted):
    assert has_hand_written_digits(planted)


@pytest.mark.parametrize(
    "ok",
    ["02x", "04d", ".1f", ".0%", "%{customdata[0]} objetos, %{customdata[1]} erros"],
)
def test_digit_detector_ignores_format_specs_and_indices(ok):
    assert not has_hand_written_digits(ok)


# (8) label colours: the page now draws with theme.ramp (tests/test_dashboard_sky_v3.py)
def test_below_floor_glyph_contrast_as_graphic():
    assert theme.contrast_ratio(BELOW_FLOOR, theme.BG) >= theme.MIN_GRAPHIC_CONTRAST
