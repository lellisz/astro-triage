"""Phase 3: validation-point gating, dumbbell error bars, interval text."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from astro_triage.dashboard import charts_model as CM
from astro_triage.dashboard import data_headline as DH
from astro_triage.dashboard import fmt


# ---- validation point gating ----
def _cell(**over):
    cell = {
        "v_mean": 0.55,
        "val_checks_pass": True,
        "low_count": [False, False],
        "floor": {"passes": True},
    }
    return cell | over


def _delta(cell=None, fold="val"):
    return {
        "fold": fold,
        "primary_completeness": 0.8,
        "delta": {"full": {"QSO": {"0.80": {"P2": cell or _cell()}}}},
    }


def _metric(p, lo, hi):
    return {"status": "ok", "point": p, "lo": lo, "hi": hi}


def _rel(tmp_path, delta):
    if delta is not None:
        d = tmp_path / "lgbm_seeds"
        d.mkdir()
        (d / "delta.json").write_text(json.dumps(delta), encoding="utf-8")
    models = {
        "lgbm_full": {
            "metrics": {
                "purity/QSO": _metric(0.7, 0.6, 0.8),
                "completeness/QSO": _metric(0.9, 0.85, 0.95),
            }
        }
    }
    variants = {v: {"models": models} for v in ("as_shipped", "prior_matched")}
    results = {"populations": {"a_all": {"priors": {"P2": {"variants": variants}}}}}
    return SimpleNamespace(root=tmp_path, results=results)


def test_validation_point_all_ok(tmp_path):
    rel = _rel(tmp_path, _delta())
    pt = DH._validation_point(rel, "lgbm_full", "P2")
    assert pt["purity"] == 0.55 and pt["completeness"] == 0.8
    assert pt["purity_lo"] is None and pt["fold"] == "validação"
    keys = [p["key"] for p in DH.purity_points(rel)]
    assert keys == ["as_shipped", "validation", "prior_matched"]


@pytest.mark.parametrize(
    "delta",
    [
        _delta(fold="test"),
        _delta(_cell(val_checks_pass=False)),
        _delta(_cell(low_count=[False, True])),
        _delta(_cell(floor={"passes": False})),
    ],
    ids=["fold", "checks", "low_count", "floor"],
)
def test_validation_point_failed_gate_is_dropped(tmp_path, delta):
    rel = _rel(tmp_path, copy.deepcopy(delta))
    assert DH._validation_point(rel, "lgbm_full", "P2") is None
    assert [p["key"] for p in DH.purity_points(rel)] == ["as_shipped", "prior_matched"]


def test_validation_point_without_delta_file(tmp_path):
    assert DH._validation_point(_rel(tmp_path, None), "lgbm_full", "P2") is None


def test_delta_sha256_keys_stripped(tmp_path):
    d = _delta()
    d["input_sha256"] = "abc"
    d["delta"]["full"]["QSO"]["0.80"]["P2"]["x_sha256"] = "z"
    loaded = DH.load_delta(_rel(tmp_path, d))
    assert "sha256" not in json.dumps(loaded)


# ---- dumbbell error bars ----
def _p(key, pur, com, lo=None, hi=None, dashed=False):
    return {
        "key": key,
        "label": f"L-{key}",
        "purity": pur,
        "completeness": com,
        "purity_lo": lo,
        "purity_hi": hi,
        "completeness_lo": lo,
        "completeness_hi": hi,
        **({"dashed": True} if dashed else {}),
    }


def test_dumbbell_error_x_asymmetric_and_zero_for_validation():
    pts = [_p("a", 0.7, 0.9, 0.6, 0.85), _p("v", 0.5, 0.8, dashed=True)]
    by = {t.name: t for t in CM.dumbbell(pts).data}
    ex = by["Pureza"].error_x
    assert ex.symmetric is False
    assert list(ex.array) == pytest.approx([0.15, 0.0])
    assert list(ex.arrayminus) == pytest.approx([0.1, 0.0])
    assert by["Completude"].error_x.array is None


def test_dumbbell_no_intervals_no_bars():
    by = {t.name: t for t in CM.dumbbell([_p("v", 0.5, 0.8)]).data}
    assert by["Pureza"].error_x.array is None


# ---- pt-BR interval text ----
def test_pt_pct_interval_format():
    assert fmt.pt_pct_interval(0.753, 0.603, 0.899) == "75,3% (IC 95%: 60,3% a 89,9%)"
