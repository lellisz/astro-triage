"""Kit v2 components (html_system + kit_types): inputs, n/a, geometry, contrast, escape."""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

import pytest
from test_dashboard_phase4_sky2 import _literals, has_hand_written_digits

from astro_triage.dashboard import fmt, theme
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard import kit_types as kt

MODULES = [Path(hs.__file__), Path(kt.__file__)]
XSS = ["<script>alert(1)</script>", '"><img onerror=x>']
CI = kt.Interval(0.2, 0.6)
GOOD = kt.Value(0.4, CI)
VERDICT_WORDS = r"\b(empate|vence|vitória|fica atrás|pior|melhor)\b"
GROUPS = ["hs2-trk", "hs2-fill", "hs2-ci", "hs2-mk", "hs2-strip"]


def _all_new() -> list[str]:
    row = kt.BarRow("rótulo", GOOD, selected=True, note="nota")
    verdict = kt.Verdict("palavra", 0.01, kt.Interval(-0.02, 0.03), 0.05)
    return [
        hs.between_text(CI),
        hs.number_card("t", GOOD, "def"),
        hs.hbar(row),
        hs.hbar_group("g", [row, kt.BarRow("x", kt.Value(None, None))]),
        hs.verdict_card("v", verdict),
        hs.rule_prior_line("regra"),
        hs.method_note("nota"),
        hs.status_chip("Proposta", "proposed"),
    ]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_kit_has_no_hand_written_digits(path):
    bad = [
        (n.lineno, n.value[:70])
        for n in _literals(path)
        if isinstance(n.value, str) and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


def test_hbar_without_ci_raises():
    with pytest.raises(kt.KitInputError):
        hs.hbar(kt.BarRow("a", kt.Value(0.4, None)))


def test_na_renders_without_bar_or_ci():
    out = hs.hbar(kt.BarRow("a", kt.Value(None, None)))
    assert hs.NA in out
    assert not any(g in out for g in GROUPS)
    assert fmt.pt_ci_label() not in out
    card = hs.number_card("t", kt.Value(None, None), "def")
    assert hs.NA in card and fmt.pt_ci_label() not in card


def test_na_value_cannot_carry_an_interval():
    with pytest.raises(kt.KitInputError):
        kt.Value(None, CI)


@pytest.mark.parametrize("bad", [(0.6, 0.2), (float("nan"), 0.1)])
def test_interval_validates(bad):
    with pytest.raises(kt.KitInputError):
        kt.Interval(*bad)


def test_between_text_comes_from_values():
    out = hs.between_text(kt.Interval(0.1234, 0.5678), digits=2)
    assert fmt.pt_pct(0.1234, 2) in out and fmt.pt_pct(0.5678, 2) in out
    assert out.startswith("entre ") and fmt.pt_ci_label() in out
    assert hs.between_text(None) == hs.NA


def test_scale_is_linear_with_labelled_ends():
    out = hs.hbar_group("g", [kt.BarRow("a", kt.Value(0.5, kt.Interval(0.25, 0.75)))])
    assert fmt.pt_pct(0, 0) in out and fmt.pt_pct(1, 0) in out
    assert f"width:{hs.PERCENT * 0.5:.2f}%" in out
    assert f"left:{hs.PERCENT * 0.25:.2f}%" in out


@pytest.mark.parametrize("point", [-0.1, 1.1])
def test_out_of_scale_value_raises(point):
    with pytest.raises(kt.KitInputError):
        hs.hbar(kt.BarRow("a", kt.Value(point, kt.Interval(0.0, 1.0))))


def test_no_log_scale_in_new_components():
    text = "".join(_all_new()) + hs._kit_rules("x").__repr__()
    assert not re.search(r"\blog\b|log10|logarithm", text, re.IGNORECASE)


def test_no_round_marker_or_dot_class():
    blob = "".join(_all_new()) + "".join(hs._kit_rules("x"))
    assert not re.search(r"border-radius\s*:\s*50%", blob)
    for cls in re.findall(r"hs2-[\w-]+", blob):
        assert not re.search(r"dot|ponto|point", cls), cls


def _overlaps(a, b) -> bool:
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


@pytest.mark.parametrize("viewport", [hs.NARROW_VIEWPORT_PX, hs.WIDE_VIEWPORT_PX])
@pytest.mark.parametrize("label", ["Regra ajustada · prior de teste", "x" * 60])
def test_geometry_no_overlap_390_1440(viewport, label):
    boxes = hs.hbar_boxes(viewport, label, fmt.pt_pct(1, hs.ONE))
    assert boxes["bar"][2] > 0
    assert not _overlaps(boxes["bar"], boxes["value"])
    assert not _overlaps(boxes["label"], boxes["bar"])
    assert not _overlaps(boxes["label"], boxes["value"])
    assert boxes["value"][2] <= boxes["page"][2]
    assert boxes["value"][1] - boxes["bar"][2] >= hs.HB_GAP_PX


@pytest.mark.parametrize("viewport", [hs.NARROW_VIEWPORT_PX, hs.WIDE_VIEWPORT_PX])
def test_axis_ends_do_not_overlap(viewport):
    track = hs.hbar_boxes(viewport, "a", "b")["bar"][2]
    ends = [fmt.pt_pct(0, 0), fmt.pt_pct(1, 0)]
    assert not any(a[1] >= b[0] for a, b in pairwise(hs.tick_boxes(ends, track)))


@pytest.mark.parametrize("fg", ["TEXT", "TEXT_SECONDARY", "SOFT", "ACCENT"])
@pytest.mark.parametrize("bg", ["BG", "SECONDARY_BG"])
def test_text_contrast_tokens(fg, bg):
    ratio = theme.contrast_ratio(getattr(theme, fg), getattr(theme, bg))
    assert ratio >= theme.MIN_TEXT_CONTRAST, (fg, bg, ratio)
    assert theme.MIN_SECONDARY_PX >= 15


def test_graphic_tokens_contrast():
    for name in ["TEXT", "TEXT_SECONDARY", "DASH"]:
        ratio = theme.contrast_ratio(getattr(theme, name), theme.BG)
        assert ratio >= theme.MIN_GRAPHIC_CONTRAST, name


def test_interval_contrast_against_adjacent_surfaces():
    """The interval stroke sits on its own strip of the track, never over the fill."""
    pairs = {
        "ci on track": (theme.TEXT, theme.GRID),
        "ci on band": (theme.TEXT, theme.DASH),
        "mark halo on ci": (theme.TEXT, theme.BG),
        "mark halo on fill": (theme.BG, theme.TEXT_SECONDARY),
        "mark halo on selected fill": (theme.BG, theme.ACCENT),
        "mark on halo": (theme.TEXT, theme.BG),
    }
    for name, (fg, bg) in pairs.items():
        assert theme.contrast_ratio(fg, bg) >= theme.MIN_GRAPHIC_CONTRAST, name
    css = "".join(hs._kit_rules("x"))
    assert f".hs2-trk .hs2-fill{{bottom:{hs.HB_CI_PX + 1}px}}" in css
    assert f".hs2-trk .hs2-ci{{top:{hs.HB_TRACK_PX - hs.HB_CI_PX}px;bottom:0px}}" in css
    assert f"box-shadow:0px 0px 0px 1px {theme.BG}" in css


def test_used_text_colours_are_tokens_and_in_css():
    css = "".join(hs._kit_rules("x"))
    for name in ["TEXT", "TEXT_SECONDARY", "SOFT", "ACCENT"]:
        assert getattr(theme, name) in css


def test_selected_row_not_colour_only():
    on = hs.hbar(kt.BarRow("a", GOOD, selected=True))
    off = hs.hbar(kt.BarRow("a", GOOD))
    assert hs.SEL_WORD in on and hs.SEL_MARK in on and hs.SEL_WORD not in off
    assert "font-weight" in "".join(hs._kit_rules("x"))


@pytest.mark.parametrize("payload", XSS)
def test_outputs_escaped(payload):
    row = kt.BarRow(payload, GOOD, selected=True, note=payload)
    outs = [
        hs.number_card(payload, GOOD, payload),
        hs.hbar(row),
        hs.hbar_group(payload, [row]),
        hs.verdict_card(
            payload, kt.Verdict(payload, 0.0, kt.Interval(-0.1, 0.1), 0.05)
        ),
        hs.rule_prior_line(payload),
        hs.method_note(payload),
        hs.status_chip(payload, "closed"),
    ]
    for out in outs:
        assert "<script" not in out.lower() and "<img" not in out.lower()


def test_container_lang_pt_br():
    for out in _all_new()[1:]:
        assert f"lang='{hs.PAGE_LANG}'" in out
    assert hs.PAGE_LANG == "pt-BR"


def test_verdict_card_word_from_caller():
    out = hs.verdict_card("t", kt.Verdict("PALAVRA", 0.0, kt.Interval(-0.1, 0.1), 0.05))
    assert "PALAVRA" in out
    assert not re.search(VERDICT_WORDS, out.replace("PALAVRA", ""), re.IGNORECASE)
    assert "diferença entre" in out and hs.DELTA in out
    assert "hs2-band" in out and "hs2-mk" in out


def test_verdict_card_without_interval_is_na_without_strip():
    out = hs.verdict_card("t", kt.Verdict(kt.NA_WORD, None, None, None))
    assert hs.NA in out and "hs2-strip" not in out


@pytest.mark.parametrize("kind", ["accepted", "proposed", "closed", "na"])
def test_chip_always_has_text_label(kind):
    assert ">Rótulo<" in hs.status_chip("Rótulo", kind)


def test_chip_rejects_unknown_kind_and_blank_label():
    with pytest.raises(kt.KitInputError):
        hs.status_chip("a", "other")  # type: ignore[arg-type]
    with pytest.raises(kt.KitInputError):
        hs.status_chip("  ", "na")


def test_kit_css_is_in_system_css_without_streamlit_selectors():
    css = hs.system_css()
    assert ".hs2-card" in css and ".hs2-chip" in css
    assert "stMarkdown" not in "".join(hs._kit_rules("x"))
