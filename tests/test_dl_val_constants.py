"""val_purity_summary on hand-built data, and constants completeness of the freeze document."""

from __future__ import annotations

import numpy as np
import pytest
from test_dl_val_freeze import (
    APPROVAL,
    N_TEST,
    N_VAL,
    make_env,
    write_labels,
    write_pred,
)

from astro_triage.evaluation import dl_val as V
from astro_triage.evaluation.runner import GRID

# val rows: 0-9 STAR, 10-19 GALAXY, 20-29 QSO. Columns of p: star, galaxy, qso (only the scored one matters).
QSO_TRUE = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]


def hand_scores(cls: str) -> np.ndarray:
    s = np.full(N_VAL, 0.1)
    if cls == "QSO":
        s[20:30] = QSO_TRUE
        s[0], s[1] = 0.7, 0.58  # STAR contaminants
        s[10], s[11] = 0.62, 0.56  # GALAXY contaminants
    else:
        s[10:20] = QSO_TRUE
        s[0], s[1] = 0.9, 0.56  # STAR contaminants
        s[20] = 0.61  # QSO contaminant
    return s


def purity(cls: str, counts: dict[str, int]) -> float:
    """Independent P2 purity: every class has 10 val rows, so each row weighs pi_c / 10."""
    pi = GRID["P2"]
    num = pi[cls] * counts[cls] / 10
    return num / sum(pi[k] * n / 10 for k, n in counts.items())


@pytest.mark.parametrize(
    ("cls", "col", "c_star", "thr", "counts"),
    [
        ("QSO", 2, "0.80", 0.6, {"QSO": 8, "STAR": 1, "GALAXY": 1}),
        ("QSO", 2, "0.90", 0.55, {"QSO": 9, "STAR": 2, "GALAXY": 2}),
        ("GALAXY", 1, "0.80", 0.6, {"GALAXY": 8, "STAR": 1, "QSO": 1}),
        ("GALAXY", 1, "0.90", 0.55, {"GALAXY": 9, "STAR": 2, "QSO": 1}),
    ],
)
def test_val_purity_summary_matches_hand_computation(tmp_path, cls, col, c_star, thr, counts):
    val_ids, test_ids = write_labels(tmp_path / "rel")
    p = np.full((N_VAL + N_TEST, 3), 0.1)
    p[:N_VAL, col] = hand_scores(cls)
    p[N_VAL:, col] = 0.99  # test rows would be selected at every threshold if they leaked in
    path = tmp_path / "predictions_x_s0_v0.parquet"
    write_pred(path, np.concatenate([val_ids, test_ids]), p)
    cell = V.val_purity_summary(tmp_path / "rel", path)[cls][c_star]
    assert cell["threshold"] == thr
    assert cell["purity_p2"] == pytest.approx(purity(cls, counts), rel=1e-12)
    assert set(V.val_purity_summary(tmp_path / "rel", path)) == {"QSO", "GALAXY"}


def test_constants_complete_against_adr_0006_section_15_header(tmp_path):
    """ADR section 15 header lists BLOCK_LEVEL = 3; the other header constants are checked by value."""
    c = V.build_freeze(*make_env(tmp_path), APPROVAL)["constants"]
    assert c["headline_metric"] == "purity_at_completeness"
    assert c["target_classes"] == ["QSO", "GALAXY"] and c["completeness_levels"] == [0.8, 0.9]
    assert c["primary_completeness"] == 0.8 and c["sensitivity_priors"] == ["P1", "P3"]
    assert c["delta_sd_multiplier"] == 2.0 and c["delta_ddof"] == 1
    assert c["block_divisor"] == 16 and c["min_blocks"] == 20 and c["low_count_rule"] == "total"
    assert c["headline_pairs"] == [["mlp_full", "lgbm_full"], ["ftt_full", "lgbm_full"]]
    assert c["descriptive_pairs"] == [["ens_mlp_full", "lgbm_full"], ["ens_ftt_full", "lgbm_full"]]
    assert c["block_level"] == 3  # ADR 0006 section 15: BLOCK_LEVEL = 3
    assert c["delta_fold"] == "val" and c["p3_bands"] == ["B", "C"]
    assert c["prediction_version"] == "v0" and c["max_invalid_fraction"] == 0.01
