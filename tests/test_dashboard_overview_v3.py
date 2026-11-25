"""Visão geral v3: model, answer template, rendered HTML against the artifact, kit guards."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_dashboard_phase4_sky2 import _literals, has_hand_written_digits
from test_dashboard_resultado import _results

from astro_triage.dashboard import data_headline as H
from astro_triage.dashboard import fmt, ui_state
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard import kit_types as kt
from astro_triage.dashboard import overview_model as OM
from astro_triage.dashboard import overview_text as OT
from astro_triage.dashboard.ui_state import Rule, UiState

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"
STATES = [UiState(Rule.SHIPPED, "P2", False)] + [
    UiState(Rule.MATCHED, p, False) for p in ui_state.PRIORS
]
VALUE_WORDS = re.compile(r"\b(só|melhor|ótimo|excelente|ruim)\b", re.IGNORECASE)


def _metrics(res, prior, variant):
    cell = res["populations"]["a_all"]["priors"][prior]["variants"][variant]
    return cell["models"]["lgbm_full"]["metrics"]


def _model(state, results=None, tmp_path=None):
    rel = SimpleNamespace(root=tmp_path or Path("."), results=results or _results())
    return OM.build(
        H.shipped_state(rel), H.prior_states(rel), H.accuracy_defs(rel), state
    )


def test_accuracy_is_na_under_matched_rule(tmp_path):
    assert _model(STATES[0], tmp_path=tmp_path).accuracy.point == pytest.approx(0.7251)
    for state in STATES[1:]:
        assert _model(state, tmp_path=tmp_path).accuracy == kt.Value(None, None)


def test_accuracy_ci_is_na(tmp_path):
    model = _model(STATES[0], tmp_path=tmp_path)
    assert model.accuracy.ci is None
    card = hs.number_card("t", model.accuracy, "d")
    assert f"{hs.NA} · {fmt.pt_ci_label()}" in card


def test_answer_changes_with_rule(tmp_path):
    a = OT.answer(_model(STATES[0], tmp_path=tmp_path), STATES[0])
    b = OT.answer(_model(STATES[2], tmp_path=tmp_path), STATES[2])
    assert a != b
    assert ui_state.RULE_LABEL[Rule.SHIPPED] in a
    assert ui_state.RULE_LABEL[Rule.MATCHED] in b


def test_answer_has_no_value_adjective_nor_so(tmp_path):
    for state in STATES:
        text = OT.answer(_model(state, tmp_path=tmp_path), state)
        assert not VALUE_WORDS.search(text), text
        assert "pureza" in text and "completude" in text


def test_answer_says_na_for_na(tmp_path):
    res = _results()
    _metrics(res, "P2", "as_shipped")["completeness/QSO"]["status"] = "low_count"
    text = OT.answer(_model(STATES[0], res, tmp_path), STATES[0])
    assert text.count(hs.NA) == 2 and "%" not in text


def _leaves(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("point", "lo", "hi", "accuracy") and isinstance(v, float):
                yield v
            else:
                yield from _leaves(v)


@pytest.mark.parametrize(
    "name", ["overview_model.py", "overview_text.py"]
)
def test_overview_modules_have_no_hand_written_digits(name):
    bad = [
        (n.lineno, n.value[:70])
        for n in _literals(DASH / name)
        if isinstance(n.value, str) and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


def test_verdict_requires_interval_and_delta_unless_na():
    ci = kt.Interval(-0.1, 0.1)
    for args in ((0.0, None, 0.05), (0.0, ci, None), (None, None, None)):
        with pytest.raises(kt.KitInputError):
            kt.Verdict("palavra", *args)
    assert kt.Verdict(kt.NA_WORD, None, None, None).word == kt.NA_WORD
    assert kt.Verdict("palavra", 0.0, ci, 0.05).delta == 0.05


def test_missing_ci_card_keeps_the_ci_label():
    card = hs.number_card("t", kt.Value(0.5, None), "d")
    assert f"{hs.NA} · {fmt.pt_ci_label()}" in card


@pytest.mark.parametrize("point", [0.05, 0.95])
def test_value_point_must_lie_inside_its_interval(point):
    with pytest.raises(kt.KitInputError):
        kt.Value(point, kt.Interval(0.2, 0.8))
    assert kt.Value(0.5, kt.Interval(0.2, 0.8)).point == 0.5
