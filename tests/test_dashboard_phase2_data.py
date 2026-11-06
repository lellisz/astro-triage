"""Phase 2 dashboard: data_headline (delta loading, purity points, error pairs)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from astro_triage.dashboard import data_headline as H

CLASSES = ("STAR", "GALAXY", "QSO")


def _metric(point, status="ok"):
    return {"status": status, "point": point, "lo": point - 0.01, "hi": point + 0.01}


def _variant(pur, com, status="ok"):
    return {
        "models": {
            "lgbm_full": {
                "metrics": {
                    "purity/QSO": _metric(pur, status),
                    "completeness/QSO": _metric(com),
                }
            }
        }
    }


def _results(shipped=True, matched=True, matched_status="ok"):
    variants = {}
    if shipped:
        variants["as_shipped"] = _variant(0.70, 0.90)
    if matched:
        variants["prior_matched"] = _variant(0.60, 0.80, matched_status)
    return {"populations": {"a_all": {"priors": {"P2": {"variants": variants}}}}}


def _delta(comp=0.8, v_mean=0.55):
    return {
        "fold": "val",
        "primary_completeness": comp,
        "delta": {"full": {"QSO": {f"{comp:.2f}": {"P2": {"v_mean": v_mean}}}}},
        "inputs_sha256": "abc",
    }


def _rel(tmp_path, results=None, delta=None, test=None):
    if delta is not None:
        (tmp_path / "lgbm_seeds").mkdir()
        (tmp_path / "lgbm_seeds" / "delta.json").write_text(json.dumps(delta))
    return SimpleNamespace(root=tmp_path, results=results or _results(), test=test)


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_load_delta_absent(tmp_path):
    assert H.load_delta(_rel(tmp_path)) is None


def test_load_delta_strips_sha256_at_any_level(tmp_path):
    raw = _delta()
    raw["nested"] = {"deep": {"file_sha256": "x", "ok": 1}, "list": [{"sha256": "y"}]}
    out = H.load_delta(_rel(tmp_path, delta=raw))
    assert out["primary_completeness"] == 0.8
    assert not [k for k in _keys(out) if "sha256" in k]


def test_purity_points_order_and_validation_values(tmp_path):
    pts = H.purity_points(_rel(tmp_path, delta=_delta()))
    assert [p["key"] for p in pts] == ["as_shipped", "validation", "prior_matched"]
    val = pts[1]
    assert val["purity"] == pytest.approx(0.55)
    assert val["completeness"] == pytest.approx(0.8)
    assert val["dashed"] is True
    assert "0,80" in val["label"]


def test_purity_points_without_delta_has_no_validation(tmp_path):
    pts = H.purity_points(_rel(tmp_path))
    assert [p["key"] for p in pts] == ["as_shipped", "prior_matched"]


def test_purity_points_omits_non_ok_status(tmp_path):
    rel = _rel(tmp_path, results=_results(matched_status="below_floor"))
    assert [p["key"] for p in H.purity_points(rel)] == ["as_shipped"]


def test_purity_points_missing_variant_and_malformed_delta(tmp_path):
    rel = _rel(tmp_path, results=_results(shipped=False), delta={"delta": {}})
    assert [p["key"] for p in H.purity_points(rel)] == ["prior_matched"]


def test_purity_points_lgbm_full_maps_to_delta_full(tmp_path):
    rel = _rel(tmp_path, delta=_delta())
    assert any(p["key"] == "validation" for p in H.purity_points(rel, "lgbm_full"))


def _errors(pairs):
    rows = []
    for (t, p), n in pairs.items():
        rows += [{"true": t, "pred_lgbm_full": p, "err_lgbm_full": True}] * n
    rows.append({"true": "STAR", "pred_lgbm_full": "STAR", "err_lgbm_full": False})
    return pd.DataFrame(rows)


def test_error_directions_six_rows_with_zeros(tmp_path):
    d = H.error_directions(_rel(tmp_path, test=_errors({("STAR", "QSO"): 3})))
    expected = {(a, b) for a in CLASSES for b in CLASSES if a != b}
    assert set(zip(d["true"], d["pred"], strict=True)) == expected
    assert len(d) == 6 and d["n"].sum() == 3 and d["n"].iloc[0] == 3
    assert d["n"].is_monotonic_decreasing


def test_error_pairs_sum_share_and_order(tmp_path):
    test = _errors(
        {
            ("STAR", "QSO"): 6,
            ("QSO", "STAR"): 2,
            ("GALAXY", "QSO"): 1,
            ("GALAXY", "STAR"): 1,
        }
    )
    p = H.error_pairs(_rel(tmp_path, test=test))
    assert len(p) == 3
    assert list(p["n"]) == [8, 1, 1]
    assert p["share"].sum() == pytest.approx(1.0)
    top = p.iloc[0]
    assert (top["n_ab"], top["n_ba"]) == (6, 2)
    assert top["pair"] == "estrela ↔ quasar"


def test_error_pairs_no_errors_is_nan_safe(tmp_path):
    test = pd.DataFrame(
        {"true": ["STAR"], "pred_lgbm_full": ["STAR"], "err_lgbm_full": [False]}
    )
    rel = _rel(tmp_path, test=test)
    d, p = H.error_directions(rel), H.error_pairs(rel)
    assert len(d) == 6 and len(p) == 3
    assert (d["n"] == 0).all() and (p["n"] == 0).all()
    assert all(math.isnan(s) for s in p["share"])
