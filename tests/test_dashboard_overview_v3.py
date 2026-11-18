"""Visão geral v3: model, answer template, rendered HTML against the artifact, kit guards."""

from __future__ import annotations

import json
import os
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
from astro_triage.dashboard import views_overview as VO
from astro_triage.dashboard.ui_state import Rule, UiState

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"
STATES = [UiState(Rule.SHIPPED, "P2", False)] + [
    UiState(Rule.MATCHED, p, False) for p in ui_state.PRIORS
]
VALUE_WORDS = re.compile(r"\b(só|melhor|ótimo|excelente|ruim)\b", re.IGNORECASE)
PERCENT_TOKEN = re.compile(r"\d+(?:,\d+)?%")


def _metrics(res, prior, variant):
    cell = res["populations"]["a_all"]["priors"][prior]["variants"][variant]
    return cell["models"]["lgbm_full"]["metrics"]


def _model(state, results=None, tmp_path=None):
    rel = SimpleNamespace(root=tmp_path or Path("."), results=results or _results())
    return OM.build(
        H.shipped_state(rel), H.prior_states(rel), H.accuracy_defs(rel), state
    )


def test_every_purity_has_its_completeness(tmp_path):
    res = _results()
    _metrics(res, "P1", "prior_matched")["purity/QSO"]["status"] = "low_count"
    for state in STATES:
        model = _model(state, res, tmp_path)
        for row in model.rows:
            assert (row.purity.point is None) == (row.completeness.point is None)
        html = VO.page_html(model, state)
        for title in (VO.PURITY_TITLE, VO.COMPLETENESS_TITLE):
            assert html.count(title) == 2  # group title and aria-label
        assert html.count("hs2-row") == 8
    na_row = _model(STATES[2], res, tmp_path).rows[1]
    assert na_row.purity.point is None and na_row.completeness.point is None


def test_accuracy_is_na_under_matched_rule(tmp_path):
    assert _model(STATES[0], tmp_path=tmp_path).accuracy.point == pytest.approx(0.7251)
    for state in STATES[1:]:
        assert _model(state, tmp_path=tmp_path).accuracy == kt.Value(None, None)


def test_accuracy_ci_is_na(tmp_path):
    model = _model(STATES[0], tmp_path=tmp_path)
    assert model.accuracy.ci is None
    card = hs.number_card("t", model.accuracy, "d")
    assert f"{hs.NA} · {fmt.pt_ci_label()}" in card


def test_p3_row_and_notice_are_partial(tmp_path):
    model = _model(STATES[3], tmp_path=tmp_path)
    assert [r.prior for r in model.rows if r.partial] == ["P3"]
    assert model.partial_notice and "P3" in model.partial_notice
    html = VO.page_html(model, STATES[3])
    assert OT.PARTIAL_TAG in html and model.partial_notice in html
    assert _model(STATES[1], tmp_path=tmp_path).partial_notice is None


@pytest.mark.parametrize("state", STATES)
def test_four_rows_always_present_and_one_selected(state, tmp_path):
    model = _model(state, tmp_path=tmp_path)
    assert [(r.rule, r.prior) for r in model.rows] == list(OM.ROW_ORDER)
    assert sum(r.selected for r in model.rows) == 1
    assert (model.selected.rule, model.selected.prior) == (state.rule, state.prior)
    assert VO.page_html(model, state).count("hs2-on") == 2


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


def _check_against_artifact(results, tmp_path):
    known = {fmt.pt_pct(v) for v in _leaves(results)}
    allowed = (fmt.pt_ci_label(), fmt.pt_pct(0, 0), fmt.pt_pct(1, 0))
    for state in STATES:
        html = VO.page_html(_model(state, results, tmp_path), state)
        text = re.sub(r"<[^>]+>", " ", html)
        for token in allowed:
            text = re.sub(rf"(?<![\d,]){re.escape(token)}", " ", text)
        text = re.sub(r"\bP\d\b|\bADR \d+\b", " ", text)
        shown = PERCENT_TOKEN.findall(text)
        assert shown, "uma tela sem números não prova nada"
        assert [n for n in shown if n not in known] == [], state
        assert not re.search(r"\d", PERCENT_TOKEN.sub(" ", text)), state


def test_rendered_numbers_exist_in_artifacts(tmp_path):
    _check_against_artifact(_results(), tmp_path)


def test_rendered_numbers_exist_in_real_artifact():
    root = os.environ.get("ASTRO_DATA_ROOT")
    path = Path(root or "") / "interim" / "full_v0" / "eval" / "results.json"
    if not root or not path.is_file():
        pytest.skip("ASTRO_DATA_ROOT sem interim/full_v0/eval/results.json")
    _check_against_artifact(json.loads(path.read_text(encoding="utf-8")), Path("."))


@pytest.mark.parametrize(
    "name", ["overview_model.py", "overview_text.py", "views_overview.py"]
)
def test_overview_modules_have_no_hand_written_digits(name):
    bad = [
        (n.lineno, n.value[:70])
        for n in _literals(DASH / name)
        if isinstance(n.value, str) and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


def test_shortcuts_have_no_numbers():
    assert len(VO.SHORTCUT_TITLES) == 4
    assert not any(re.search(r"\d", t) for t in VO.SHORTCUT_TITLES)


def test_overview_body_has_no_seal_nor_local_prior_picker(tmp_path):
    html = VO.page_html(_model(STATES[0], tmp_path=tmp_path), STATES[0])
    assert hs.SEAL_PREFIX not in html
    assert not hasattr(VO, "PRIOR_KEY") and not hasattr(VO, "_pick_prior")


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
