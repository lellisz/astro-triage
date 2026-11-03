"""Proofs for ``abstention.policy`` (ADR 0005 D2/D4): threshold, calibration, combination."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astro_triage.abstention import policy as P
from astro_triage.abstention.policy import ABSTAIN, choose_threshold, combine

CONF = np.array([0.55, 0.60, 0.65, 0.70, 0.95, 0.96, 0.97, 0.98])
CORRECT = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)


def test_choose_threshold_tie_with_grid_value_keeps_the_object() -> None:
    # conf >= tau keeps: tau=0.70 would keep the two wrong 0.70 objects, so 0.71 is the first.
    conf = np.array([0.7, 0.7, 0.9])
    out = choose_threshold(conf, np.array([False, False, True]), 1.0, 1.0)
    assert out["tau"] == 0.71
    assert out["conflict"] is False
    assert out["routed_fraction"] == pytest.approx(2 / 3)
    assert out["accuracy_kept"] == 1.0
    # and a correct tie at the grid value is kept at tau == that value
    ok = choose_threshold(np.array([0.7, 0.7]), np.array([True, True]), 1.0, 1.0)
    assert ok["tau"] == 0.5


def test_choose_threshold_no_conflict() -> None:
    out = choose_threshold(CONF, CORRECT, 1.0, 0.5)
    assert out["tau"] == out["tau_from_a_keep"] == 0.71
    assert out["conflict"] is False
    assert out["routed_fraction"] == 0.5
    assert out["n"] == 8.0


def test_choose_threshold_conflict_with_r_max_lowers_tau() -> None:
    out = choose_threshold(CONF, CORRECT, 1.0, 0.25)
    assert out["conflict"] is True
    assert out["tau_from_a_keep"] == 0.71
    assert out["tau"] == 0.65  # largest grid value routing <= 25 %
    assert out["routed_fraction"] == 0.25
    assert out["accuracy_kept"] == pytest.approx(
        4 / 6
    )  # below a_keep, flagged by conflict
    assert out["accuracy_kept"] < 1.0


def test_choose_threshold_no_tau_reaches_a_keep() -> None:
    with pytest.raises(ValueError, match="kept accuracy"):
        choose_threshold(np.array([0.6, 0.7]), np.array([True, False]), 1.0, 1.0)


def test_choose_threshold_impossible_r_max() -> None:
    conf = np.array([0.1, 0.9, 0.9, 0.9])
    with pytest.raises(ValueError, match="routes"):
        choose_threshold(conf, np.array([False, True, True, True]), 1.0, 0.0)


@pytest.mark.parametrize(
    ("conf", "correct"),
    [
        (np.array([]), np.array([], dtype=bool)),
        (np.array([0.6, np.nan]), np.array([True, True])),
        (np.array([0.6, 0.7]), np.array([True])),
        (np.array([[0.6, 0.7]]), np.array([[True, True]])),
    ],
)
def test_choose_threshold_rejects_bad_input(
    conf: np.ndarray, correct: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        choose_threshold(conf, correct, 0.9, 1.0)


def test_fit_verbal_calibration_none_below_100_answers() -> None:
    raw = np.arange(99, dtype=float)
    assert P.fit_verbal_calibration(raw, raw >= 50) is None
    assert P.fit_verbal_calibration(np.arange(100.0), np.arange(100) >= 50) is not None


def test_fit_verbal_calibration_hand_computed_bins() -> None:
    raw = np.arange(100, dtype=float)
    calib = P.fit_verbal_calibration(raw, raw >= 50)
    assert calib is not None
    assert calib["edges"] == pytest.approx([19.8, 39.6, 59.4, 79.2])
    assert calib["values"] == [0.0, 0.0, 0.5, 1.0, 1.0]
    s = P.apply_verbal_calibration(np.array([0.0, 45.0, 99.0, np.nan]), calib)
    assert s[:3].tolist() == [0.0, 0.5, 1.0]
    assert np.isnan(s[3])


def test_fit_verbal_calibration_constant_scores_make_no_duplicate_bins() -> None:
    calib = P.fit_verbal_calibration(np.full(120, 90.0), np.arange(120) % 2 == 0)
    assert calib is not None
    assert len(calib["edges"]) == 1
    assert calib["values"] == [None, 0.5]  # empty bin below the single edge


def test_fit_verbal_calibration_nan_or_shape_problems() -> None:
    raw = np.arange(100, dtype=float)
    raw[3] = np.nan
    assert P.fit_verbal_calibration(raw, np.ones(100, dtype=bool)) is None
    with pytest.raises(ValueError):
        P.fit_verbal_calibration(np.arange(100.0), np.ones(99, dtype=bool))


def test_uncalibrated_verbal_scores_are_all_nan() -> None:
    assert np.isnan(P.apply_verbal_calibration(np.array([1.0, 2.0]), None)).all()


# combine: 6 objects. conf < tau (0.8) routes the first five; the last is confident.
L1_LABEL = np.array(["STAR"] * 6, dtype=object)
L1_CONF = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.9])  # 0.8 == tau is NOT routed
STATUS = np.array(["OK", "OK", "TIMEOUT", "OK", "OK", "OK"], dtype=object)
L2_LABEL = np.array(["GALAXY", "UNKNOWN", "QSO", "QSO", "QSO", "QSO"], dtype=object)
S_L2 = np.array([0.9, 0.9, 0.9, 0.3, 0.9, 0.9])


def test_combine_abstain_on_fail_all_branches() -> None:
    out = combine(
        L1_LABEL, L1_CONF, STATUS, L2_LABEL, S_L2, 0.8, None, "abstain_on_fail"
    )
    assert out.tolist() == [
        "GALAXY",  # routed, trusted: L2 label
        ABSTAIN,  # routed, L2 answered UNKNOWN (not a class)
        ABSTAIN,  # routed, L2 call failed
        "QSO",  # routed, tau2 None: s_l2 ignored
        "STAR",  # conf == tau stays with L1
        "STAR",  # confident: L2 never consulted
    ]


def test_combine_fallback_l1_never_abstains() -> None:
    out = combine(L1_LABEL, L1_CONF, STATUS, L2_LABEL, S_L2, 0.8, None, "fallback_l1")
    assert out.tolist() == ["GALAXY", "STAR", "STAR", "QSO", "STAR", "STAR"]
    assert ABSTAIN not in out
    bad = np.array(["EMPTY_CONTENT"] * 6, dtype=object)
    every = combine(
        L1_LABEL, np.zeros(6), bad, L2_LABEL, S_L2, 0.8, 0.99, "fallback_l1"
    )
    assert ABSTAIN not in every.tolist()
    assert every.tolist() == ["STAR"] * 6


def test_combine_tau2_gates_on_calibrated_confidence() -> None:
    s = np.array([0.9, 0.9, 0.9, 0.3, np.nan, 0.9])
    out = combine(L1_LABEL, L1_CONF, STATUS, L2_LABEL, s, 0.8, 0.5, "abstain_on_fail")
    assert out[3] == ABSTAIN  # 0.3 < tau2
    assert np.isnan(s[4]) and out[4] == "STAR"  # not routed, NaN irrelevant
    nan_routed = combine(
        np.array(["STAR"], dtype=object),
        np.array([0.1]),
        np.array(["OK"], dtype=object),
        np.array(["QSO"], dtype=object),
        np.array([np.nan]),
        0.8,
        0.5,
        "abstain_on_fail",
    )
    assert nan_routed.tolist() == [ABSTAIN]  # unscored answer is not trusted under tau2
    equal = combine(
        np.array(["STAR"], dtype=object),
        np.array([0.1]),
        np.array(["OK"], dtype=object),
        np.array(["QSO"], dtype=object),
        np.array([0.5]),
        0.8,
        0.5,
        "abstain_on_fail",
    )
    assert equal.tolist() == ["QSO"]  # s_l2 == tau2 is trusted


def test_combine_unknown_policy() -> None:
    with pytest.raises(ValueError, match="policy"):
        combine(L1_LABEL, L1_CONF, STATUS, L2_LABEL, S_L2, 0.8, None, "bogus")  # type: ignore[arg-type]


def test_l1_confidence_reads_columns_by_name_not_position() -> None:
    # Parquet order is star, qso, galaxy; a decoy column sits first and is huge.
    df = pd.DataFrame(
        {
            "decoy": [9.0, 9.0],
            "p_cal_star": [0.1, 0.2],
            "p_cal_qso": [0.3, 0.7],
            "p_cal_galaxy": [0.6, 0.1],
        }
    )
    assert P.l1_confidence(df).tolist() == [0.6, 0.7]
    assert P.score_columns() == ["p_cal_star", "p_cal_galaxy", "p_cal_qso"]
    with pytest.raises(ValueError, match="missing"):
        P.l1_confidence(df.drop(columns="p_cal_qso"))


def test_select_val_keeps_only_val_rows() -> None:
    df = pd.DataFrame({"fold": ["train", "val", "test", "val"], "x": [1, 2, 3, 4]})
    assert P.select_val(df)["x"].tolist() == [2, 4]


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"x": [1]}),  # no fold column
        pd.DataFrame({"fold": ["test", "test"]}),  # only test rows
        pd.DataFrame({"fold": ["train"]}),
        pd.DataFrame({"fold": ["VAL"]}),  # exact match only
        pd.DataFrame({"fold": []}),
    ],
)
def test_select_val_refuses_when_there_is_no_val(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        P.select_val(df)


def test_write_routing_manifest_leaves_no_tmp_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "routing.json"
    P.write_routing_manifest(path, {"tau": 0.71, "b": [1, 2], "a": None})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "tau": 0.71,
        "b": [1, 2],
        "a": None,
    }
    assert [p.name for p in path.parent.iterdir()] == ["routing.json"]
    P.write_routing_manifest(path, {"tau": 0.8})  # overwrite
    assert json.loads(path.read_text(encoding="utf-8")) == {"tau": 0.8}
    assert [p.name for p in path.parent.iterdir()] == ["routing.json"]


def test_write_routing_manifest_failure_keeps_old_file_and_no_tmp(
    tmp_path: Path,
) -> None:
    path = tmp_path / "routing.json"
    P.write_routing_manifest(path, {"tau": 0.7})
    with pytest.raises(ValueError):
        P.write_routing_manifest(path, {"tau": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"tau": 0.7}
    assert [p.name for p in tmp_path.iterdir()] == ["routing.json"]


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_bytes(b"abc")
    assert P.sha256_file(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
