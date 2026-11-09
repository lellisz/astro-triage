"""dl_desc (S13): e_sq, risk-coverage, costs copied, seed mean of the METRIC (never of probabilities)."""

from __future__ import annotations

import json
import math
import types

import numpy as np
import pytest

from astro_triage.evaluation import dl_desc, dl_val
from astro_triage.evaluation.runner import EvalError, ModelData
from eval import metrics as M

CL = list(M.CLASSES)


def _ts(y):
    y = np.asarray(y)
    return types.SimpleNamespace(y=y, pixel=np.arange(len(y)) % 5)


def test_e_sq_known_case_and_absent_class():
    y = np.array(["STAR", "STAR", "QSO", "QSO"])
    pred = np.array(["QSO", "STAR", "STAR", "QSO"])
    assert dl_desc.e_sq(y, pred, np.arange(4)) == pytest.approx(0.5)
    pred2 = np.array(["STAR", "STAR", "QSO", "STAR"])  # S->Q 0, Q->S 0.5
    assert dl_desc.e_sq(y, pred2, np.arange(4)) == pytest.approx(0.25)
    assert math.isnan(dl_desc.e_sq(y, pred, np.array([0, 1])))  # QSO absent


def test_risk_coverage_small_case():
    y = np.array(["STAR", "QSO", "GALAXY"])
    p = np.zeros((3, 3))
    p[0, CL.index("STAR")], p[0, CL.index("QSO")] = 0.9, 0.1
    p[1, CL.index("STAR")], p[1, CL.index("QSO")] = 0.6, 0.4  # wrong (conf 0.6)
    p[2, CL.index("GALAXY")], p[2, CL.index("STAR")] = 0.5, 0.3
    p[2, CL.index("QSO")] = 0.2
    rc = {round(r["tau"], 2): r for r in dl_desc.risk_coverage_curve(y, p)}
    assert len(rc) == 50 and min(rc) == 0.5 and max(rc) == 0.99
    assert rc[0.5]["coverage"] == 1.0 and rc[0.5]["accuracy"] == pytest.approx(2 / 3)
    assert rc[0.6]["coverage"] == pytest.approx(2 / 3) and rc[0.6][
        "accuracy"
    ] == pytest.approx(0.5)
    assert rc[0.9]["coverage"] == pytest.approx(1 / 3) and rc[0.9]["accuracy"] == 1.0
    assert rc[0.95]["coverage"] == 0.0 and math.isnan(rc[0.95]["accuracy"])


def _world(wrong_seeds: set[int]):
    y = np.repeat(np.asarray(CL), 4)
    systems = {}
    for name in dl_val.planned_systems():
        fam, k = name.rsplit("_s", 1)
        p = np.zeros((12, 3))
        for r, c in enumerate(y):
            i = CL.index(c)
            bad = fam == "mlp_full" and int(k) in wrong_seeds
            p[r, i], p[r, (i + 1) % 3] = (0.4, 0.6) if bad else (0.9, 0.1)
        systems[name] = ModelData(name, p, np.ones(12, bool), None)
    return _ts(y), systems


def test_family_mean_is_mean_of_metric_not_of_probabilities():
    ts, systems = _world({0, 2, 4})  # 3 of 5 seeds always wrong, 2 always right
    desc, _ = dl_desc.describe(ts, systems, {}, workers=2)
    acc = desc["families"]["mlp_full"]["accuracy"]
    assert acc == {"mean": pytest.approx(0.4), "min": 0.0, "max": 1.0}
    # averaging probabilities would make every row correct (accuracy 1.0): the two differ here
    avg = np.mean([systems[f"mlp_full_s{k}"].proba for k in range(5)], axis=0)
    assert (np.asarray(CL)[avg.argmax(1)] == ts.y).mean() == 1.0
    assert desc["families"]["lgbm_full"]["accuracy"]["mean"] == 1.0


def test_rows_and_costs_copied(tmp_path):
    ts, systems = _world(set())
    costs = {n: {"epoch_seconds_mean": 1.5, "parameter_count": 7} for n in systems}
    desc, rows = dl_desc.describe(ts, systems, costs)
    assert desc["costs"] == costs
    assert all(len(r) == 5 for r in rows)
    assert len(rows) == 50 * (len(systems) + 8)  # 40 systems + 8 family means
    assert sum(r[1] == "mean" for r in rows) == 50 * 8


def test_load_costs_verbatim_and_refusals(tmp_path):
    fits = {
        "mlp_full_s0": {"costs": {"parameter_count": 12, "x": [1, 2]}},
        "bad": 3,
        "n": {"a": 1},
    }
    (tmp_path / "train_report_dl.json").write_text(json.dumps({"fits": fits}))
    assert dl_desc.load_costs(tmp_path) == {
        "mlp_full_s0": {"parameter_count": 12, "x": [1, 2]}
    }
    with pytest.raises(EvalError, match="cannot read costs"):
        dl_desc.load_costs(tmp_path / "missing")
    (tmp_path / "train_report_dl.json").write_text(json.dumps({"fits": [1]}))
    with pytest.raises(EvalError, match="not an object"):
        dl_desc.load_costs(tmp_path)
