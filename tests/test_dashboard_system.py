"""Prior-ruler kit (html_system): escape, contrast, counts, label collision, verdict, rules."""

from __future__ import annotations

import html
import math
import re
from itertools import pairwise
from pathlib import Path

import pytest

# Same detector and literal walker as the phase 4 sky sweep (imported, not copied).
from test_dashboard_phase4_sky2 import _literals, has_hand_written_digits

from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard import theme

MODULE = Path(hs.__file__)
XSS = ["<script>alert(1)</script>", '"><img onerror=x>']
P_VALUES = [0, 0.0158, 0.5, 0.7246, 1]
HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
TOKENS = {
    v.lower()
    for k, v in vars(theme).items()
    if isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v)
}


def _no_raw(out: str) -> None:
    low = out.lower()
    assert "<script" not in low
    assert "<img" not in low


def _all_outputs() -> list[str]:
    row = {"name": "n", "d": 0.01, "lo": -0.02, "hi": 0.03, "delta": 0.005, "detail": "x"}
    return [
        hs.system_css(),
        hs.seal("r"),
        hs.dot_field(0.3),
        hs.prior_ruler({"A": 0.01, "B": None}, selected="A", callouts={"A": "c"}),
        hs.diff_ruler([{"title": "t", "rows": [row]}]),
    ]


# (1) escape
@pytest.mark.parametrize("payload", XSS)
def test_escape_everywhere(payload):
    _no_raw(hs.seal(payload))
    _no_raw(hs.dot_field(0.5, label=payload))
    _no_raw(hs.prior_ruler({payload: 0.01, "ok": 0.02}, callouts={payload: payload}))
    _no_raw(hs.prior_ruler({payload: None, "ok": 0.02}))
    _no_raw(hs.prior_ruler({"ok": 0.02}, selected=payload, callouts={"ok": payload}))
    row = {"name": payload, "detail": payload, "d": 0.01, "lo": 0.0, "hi": 0.02, "delta": 0.001}
    _no_raw(hs.diff_ruler([{"title": payload, "rows": [row]}]))
    _no_raw(hs.diff_ruler([{"title": payload, "rows": []}]))


def test_escape_is_not_a_noop_on_benign_text():
    assert "Rel 1" in hs.seal("Rel 1")


# (2) contrast and palette
@pytest.mark.parametrize("name", ["TEXT", "TEXT_SECONDARY", "SOFT", "ACCENT"])
def test_text_colours_contrast(name):
    color = getattr(theme, name)
    assert color in hs.system_css()
    assert theme.contrast_ratio(color, theme.BG) >= 4.5


def test_graphic_colour_contrast():
    assert theme.contrast_ratio(theme.DASH, theme.BG) >= 3.0


def test_only_theme_hex_colours():
    for out in _all_outputs():
        found = {h.lower() for h in HEX.findall(out)}
        assert found <= TOKENS, found - TOKENS


def test_no_red_green_semantic_colours():
    css = hs.system_css().lower()
    assert not re.search(r"\b(red|green|lime|crimson)\b", css)


# (3) counts
@pytest.mark.parametrize("p", P_VALUES)
def test_dot_count_matches_rounded_purity(p):
    out = hs.dot_field(p)
    assert out.count("hs-q") == round(1000 * p)
    assert out.count("hs-q") + out.count("hs-n") == 1000
    assert hs.blue_count(p) == round(1000 * p)


@pytest.mark.parametrize(("p", "expect"), [(1.5, 1000), (7, 1000), (-0.2, 0), (-3, 0)])
def test_dot_count_is_clamped(p, expect):
    out = hs.dot_field(p)
    assert out.count("hs-q") == expect
    assert out.count("hs-q") + out.count("hs-n") == 1000
    assert hs.blue_count(p) == expect


@pytest.mark.parametrize("p", [None, math.nan, math.inf, -math.inf])
def test_dot_field_invalid_is_na(p):
    out = hs.dot_field(p)
    assert "n/a" in out
    assert "hs-q" not in out
    assert "hs-n'" not in out
    assert hs.blue_count(p) is None


# (4) label collision
def _labels(out: str) -> list[str]:
    spans = re.findall(r"<span class='[lcr]' style='[^']*'>(.*?)</span>", out)
    return [html.unescape(s) for s in spans]


def _row(d, lo, hi, delta):
    return {"name": "x", "d": d, "lo": lo, "hi": hi, "delta": delta, "detail": "y"}


SCALES = [
    (0.0005, [_row(0.0001, -0.0003, 0.0004, 0.0005)]),
    (0.0244, [_row(0.0100, -0.0200, 0.0300, 0.0244)]),
    (1e-7, [_row(1e-8, -1e-7, 2e-7, 1e-7)]),
    (123456.0, [_row(1e5, -2e5, 3e5, 123456.0)]),
    (0.0, [_row(0.0, 0.0, 0.0, 0.0)]),
    (-0.5, [_row(-0.4, -0.9, 0.1, 0.5)]),
]


@pytest.mark.parametrize(("_", "rows"), SCALES)
def test_diff_labels_do_not_overlap(_, rows):
    out = hs.diff_ruler([{"title": "t", "rows": rows}])
    labels = _labels(out)
    boxes = hs.tick_boxes(labels, hs.MIN_TRACK_PX)
    for a, b in pairwise(boxes):
        assert a[1] < b[0], (labels, boxes)
    for start, end in boxes:
        assert start >= 0
        assert end <= hs.MIN_TRACK_PX


def test_diff_extremes_do_not_overlap():
    rows = [_row(1e-12, -1e-12, 1e-12, 1e-12), _row(9e9, -9e9, 9e9, 9e9)]
    out = hs.diff_ruler([{"title": "t", "rows": rows}])
    labels = _labels(out)
    boxes = hs.tick_boxes(labels, hs.MIN_TRACK_PX)
    assert all(a[1] < b[0] for a, b in pairwise(boxes))


def test_tick_boxes_detects_a_real_overlap():
    boxes = hs.tick_boxes(["x" * 20, "y" * 20, "z" * 20], hs.MIN_TRACK_PX)
    assert any(a[1] >= b[0] for a, b in pairwise(boxes))


NAN = math.nan
PANELS = [
    [],
    [{}],
    [{"title": None, "rows": None}],
    [{"title": "t", "rows": [{}]}],
    [{"title": "t", "rows": [_row(None, None, None, None)]}],
    [{"title": "t", "rows": [_row(NAN, NAN, NAN, NAN)]}],
    [{"title": "t", "rows": [_row(math.inf, -math.inf, math.inf, math.inf)]}],
    [{"title": "t", "rows": [_row(None, None, None, None), _row(0.2, None, None, None)]}],
]


@pytest.mark.parametrize("panels", PANELS)
def test_diff_ruler_never_raises(panels):
    out = hs.diff_ruler(panels)
    assert isinstance(out, str)
    assert "\n" not in out


def test_diff_ruler_all_none_panel_is_na():
    out = hs.diff_ruler([{"title": "t", "rows": [_row(None, None, None, None)]}])
    assert "n/a" in out
    assert "hs-ax" not in out


# (5) verdict symmetry
def _pair(fam, key, c1, c2):
    return {"family": fam, "pair_verdict": key, "classes": {"quasar": c1, "galáxia": c2}}


MLP_TIE = _pair("MLP", "tie", "tie", "tie")


def test_verdict_ftt_wins_only_in_quasar():
    out = hs.verdict_line([MLP_TIE, _pair("FTT", "tie", "win", "tie")])
    assert out == (
        "MLP: empate · FTT: empate. "
        "FTT vence só em quasar (classe); o par exige as duas classes."
    )


def test_verdict_ftt_worse_only_in_quasar():
    out = hs.verdict_line([MLP_TIE, _pair("FTT", "tie", "worse", "tie")])
    assert out == (
        "MLP: empate · FTT: empate. "
        "FTT fica atrás só em quasar (classe, descritivo)."
    )
    assert "o par exige" not in out


def test_verdict_win_and_worse_cite_both():
    out = hs.verdict_line([MLP_TIE, _pair("FTT", "tie", "win", "worse")])
    assert "FTT vence só em quasar (classe)" in out
    assert "FTT fica atrás só em galáxia (classe, descritivo)" in out
    assert out.endswith("o par exige as duas classes.")


# REV-01: the real producer vocabulary (dl_compare.pair_verdict)
REAL_TIE = "tie: LightGBM stays"


def test_verdict_real_tie_string_original_acceptance():
    out = hs.verdict_line(
        [_pair("MLP", REAL_TIE, "tie", "tie"), _pair("FTT", REAL_TIE, "win", "tie")]
    )
    assert out == (
        "MLP: empate · FTT: empate. "
        "FTT vence só em quasar (classe); o par exige as duas classes."
    )


def test_verdict_real_x_wins_is_pair_win_without_class_win_phrase():
    out = hs.verdict_line([_pair("FTT", "X wins", "win", "win")])
    assert out.startswith("FTT: vitória do par")
    assert "FTT: n/a" not in out
    assert "vence só em" not in out


def test_verdict_real_x_wins_with_one_class_win_has_no_class_win_phrase():
    out = hs.verdict_line([_pair("FTT", "X wins", "win", "tie")])
    assert out.startswith("FTT: vitória do par")
    assert "vence só em" not in out


def test_verdict_real_tie_maps_to_empate():
    assert hs.verdict_line([_pair("MLP", REAL_TIE, "tie", "tie")]) == "MLP: empate."


@pytest.mark.parametrize("key", ["X wins", REAL_TIE, "tie", "win", "??", None, ""])
def test_verdict_never_says_pair_defeat(key):
    out = hs.verdict_line([_pair("FTT", key, "worse", "worse")])
    assert "derrota do par" not in out.lower()


def test_verdict_unknown_key_is_na_with_real_looking_string():
    assert hs.verdict_line([_pair("FTT", "Y loses", "tie", "tie")]) == "FTT: n/a."


# REV-02: "só" only when a strict subset of the classes
def test_verdict_tie_all_worse_has_no_only_and_no_pair_needs():
    out = hs.verdict_line([_pair("FTT", REAL_TIE, "worse", "worse")])
    assert " só" not in out
    assert "o par exige" not in out
    assert "FTT fica atrás em quasar e galáxia (classe, descritivo)" in out


def test_verdict_tie_all_win_has_no_only():
    out = hs.verdict_line([_pair("FTT", REAL_TIE, "win", "win")])
    assert " só" not in out
    assert "FTT vence em quasar e galáxia (classe)" in out


# REV-03: verdict_html escapes, verdict_line stays plain text
def test_verdict_html_escapes_family_and_class_names():
    pair = {
        "family": "<script>x</script>",
        "pair_verdict": "tie",
        "classes": {"qu'asar": "win", "galáxia": "tie"},
    }
    out = hs.verdict_html([pair])
    _no_raw(out)
    assert "<script>" not in out
    assert "'asar" not in out.replace("&#x27;", "")
    assert "qu&#x27;asar" in out
    assert out.startswith("<div class='hs-verdict'>")
    assert out.endswith("</div>")


def test_verdict_line_stays_plain_text():
    pair = {"family": "<script>x</script>", "pair_verdict": "tie", "classes": {"q'": "win"}}
    out = hs.verdict_line([pair])
    assert "<script>x</script>" in out
    assert "q'" in out
    assert "&lt;" not in out and "&#x27;" not in out


def test_verdict_all_tie_has_no_extra_phrase():
    out = hs.verdict_line([MLP_TIE, _pair("FTT", "tie", "tie", "tie")])
    assert out == "MLP: empate · FTT: empate."


@pytest.mark.parametrize("key", ["??", None, "", "unknown: x"])
def test_verdict_unknown_pair_is_na(key):
    out = hs.verdict_line([_pair("FTT", key, "tie", "tie")])
    assert out == "FTT: n/a."


def test_verdict_empty_is_na():
    assert hs.verdict_line([]) == "n/a"


def test_verdict_pair_with_suffix_still_parsed():
    out = hs.verdict_line([_pair("MLP", "win: detalhe", "tie", "tie")])
    assert out.startswith("MLP: vitória do par")


def test_verdict_missing_classes_does_not_raise():
    out = hs.verdict_line([{"family": "MLP", "pair_verdict": "tie"}])
    assert out == "MLP: empate."


# (6) prior_ruler
def test_p0_is_ignored():
    out = hs.prior_ruler({"P0": 0.5, "P1": 0.01, "P2": 0.02})
    assert "P0" not in out
    assert "P1" in out
    assert "P2" in out


def test_p0_alone_is_na():
    assert "n/a" in hs.prior_ruler({"P0": 0.5})


@pytest.mark.parametrize("bad", [None, math.nan])
def test_none_mark_is_na_and_off(bad):
    out = hs.prior_ruler({"P1": 0.01, "P2": bad})
    assert "n/a" in out
    assert "off" in out
    assert "P2" in out


def test_off_mark_has_no_position():
    out = hs.prior_ruler({"P1": 0.01, "P2": None})
    assert out.count("class='mkl'") == 1


def test_selected_mark_flagged():
    out = hs.prior_ruler({"P1": 0.01, "P2": 0.02}, selected="P2")
    assert out.count("aria-current='true'") == 1


def test_all_none_priors_is_na():
    out = hs.prior_ruler({"P1": None})
    assert "n/a" in out


def test_log_position_monotonic_and_bounded():
    qs = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 1e6]
    pos = [hs.log_position(q, 1e-4, 1.0) for q in qs]
    assert all(p is not None and 0.0 <= p <= 100.0 for p in pos)
    assert pos == sorted(pos)
    assert pos[0] == 0.0
    assert pos[-1] == 100.0


@pytest.mark.parametrize(
    ("q", "lo", "hi"),
    [(None, 1e-4, 1), (math.nan, 1e-4, 1), (0, 1e-4, 1), (-1, 1e-4, 1), (0.1, 0, 1), (0.1, 1, 1)],
)
def test_log_position_invalid_is_none(q, lo, hi):
    assert hs.log_position(q, lo, hi) is None


# (7) repo rules
def _strings() -> list[str]:
    return [n.value for n in _literals(MODULE) if isinstance(n.value, str)]


def test_module_has_no_hand_written_digits():
    bad = [
        (n.lineno, n.value[:70])
        for n in _literals(MODULE)
        if isinstance(n.value, str) and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


def test_module_has_no_banned_words():
    pattern = r"\b(melhor|melhora|ganho|equivalente|deploy|trace)\b"
    bad = [s for s in _strings() if re.search(pattern, s, re.IGNORECASE)]
    assert not bad, bad
    bad_all = [s for s in _all_outputs() if re.search(pattern, s, re.IGNORECASE)]
    assert not bad_all


def test_module_has_no_ui_or_network_tokens():
    src = MODULE.read_text(encoding="utf-8").lower()
    for word in ("streamlit", "plotly", "http"):
        assert word not in src, word


def test_outputs_are_single_line():
    for out in _all_outputs():
        assert "\n" not in out
    assert "\n" not in hs.verdict_line([MLP_TIE])
    assert "\n" not in hs.dot_field(None)


def test_detector_flags_planted_digits():
    assert has_hand_written_digits("o modelo errou 93,9% das galáxias")
