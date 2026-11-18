"""Shared verdict rule (ADR 0006): recomputed from the artifact numbers, one text for both pages."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_dashboard_comparacao import TIE, _results
from test_dashboard_dl import _digit_literals

from astro_triage.dashboard import data_dl as DD
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard.data_dl import (
    ClassCell,
    ClassVerdict,
    PairCell,
    PairVerdict,
)
from astro_triage.evaluation import dl_compare

SRC = Path(VR.__file__)
WIN = ClassVerdict.WIN
TIE_V = ClassVerdict.TIE
WORSE = ClassVerdict.WORSE
NA = ClassVerdict.NA


def cell(cls, d, lo, hi, delta, verdict):
    return ClassCell(cls, d, lo, hi, delta, verdict, None, {})


def win(cls="QSO"):
    return cell(cls, 0.02, 0.01, 0.03, 0.005, WIN)


def worse(cls="QSO"):
    return cell(cls, -0.02, -0.03, -0.01, 0.005, WORSE)


def wide_tie(cls="GALAXY"):
    return cell(cls, 0.0, -0.05, 0.05, 0.01, TIE_V)


def narrow_tie(cls="GALAXY"):
    return cell(cls, 0.0, -0.004, 0.004, 0.01, TIE_V)


def below_floor(cls="GALAXY"):
    return cell(cls, None, None, None, 0.01, NA)


def test_current_fixture_cells_match_artifact_tokens():
    head = DD.headline(_results(ftt=(TIE, "win", "tie")))
    for pair in head.cells:
        VR.check(pair)
        assert VR.pair_state(pair.classes) is pair.verdict
        for c in pair.classes.values():
            assert VR.class_state(c) is c.verdict


def test_narrow_ci_inside_delta_has_no_power_phrase():
    assert VR.class_state(narrow_tie()) is TIE_V
    assert VR.underpowered(narrow_tie()) is False


def test_wide_ci_tie_has_power_phrase():
    assert VR.underpowered(wide_tie()) is True


def test_below_floor_cell_is_tie_and_pair_tie():
    classes = {"QSO": win(), "GALAXY": below_floor()}
    assert VR.class_state(classes["GALAXY"]) is NA
    assert VR.underpowered(classes["GALAXY"]) is False
    assert VR.pair_state(classes) is PairVerdict.TIE


def test_worse_does_not_change_pair():
    assert VR.pair_state({"QSO": worse(), "GALAXY": win("GALAXY")}) is PairVerdict.TIE
    both = {"QSO": worse(), "GALAXY": worse("GALAXY")}
    assert VR.pair_state(both) is PairVerdict.TIE


def test_win_needs_both_classes():
    assert VR.pair_state({"QSO": win(), "GALAXY": win("GALAXY")}) is PairVerdict.WIN
    assert VR.pair_state({"QSO": win(), "GALAXY": wide_tie()}) is PairVerdict.TIE
    assert VR.pair_state({"QSO": win(), "GALAXY": narrow_tie()}) is PairVerdict.TIE


def test_rule_text_follows_dl_compare_constants(monkeypatch):
    before = VR.rule_text()
    monkeypatch.setattr(dl_compare, "DELTA_SD_MULTIPLIER", 3.5)
    monkeypatch.setattr(dl_compare, "MIN_BLOCKS", 77)
    after = VR.rule_text()
    assert after != before and "3,5 desvios-padrão" in after and "77 blocos" in after
    assert after == " ".join(VR.rule_lines())


def test_mismatch_with_artifact_raises():
    liar = cell("QSO", 0.0, -0.05, 0.05, 0.01, WIN)
    classes = {"QSO": liar, "GALAXY": win("GALAXY")}
    bad = PairCell("p", "P2", "0.80", "headline", classes, PairVerdict.WIN)
    with pytest.raises(DD.DlDataError):
        VR.check(bad)
    good = {"QSO": win(), "GALAXY": win("GALAXY")}
    tie = PairCell("p", "P2", "0.80", "headline", good, PairVerdict.TIE)
    with pytest.raises(DD.DlDataError, match="par"):
        VR.check(tie)


def test_descriptive_cell_has_no_state():
    with pytest.raises(DD.DlDataError):
        VR.class_state(cell("QSO", 0.0, -1.0, 1.0, 0.1, None))


def test_verdict_rule_has_no_hand_written_digits():
    assert not list(_digit_literals(SRC))
