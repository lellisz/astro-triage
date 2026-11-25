"""Overview 'Régua do prior' screen: data_headline states, HTML builders, rules, real release."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# Same detector and literal walker as the phase 4 sky sweep (imported, not copied).
from test_dashboard_phase4_sky2 import _literals, has_hand_written_digits

from astro_triage.dashboard import data_headline as H

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "src" / "astro_triage" / "dashboard"
FULL_V0 = ROOT / "data" / "interim" / "full_v0"
XSS = "<script>x</script>"
PURITY = {"P1": 0.40, "P2": 0.70, "P3": 0.55}
QSO = {"P1": 0.005, "P2": 0.0158, "P3": 0.05}


def _m(point, status="ok"):
    return {"status": status, "point": point, "lo": point - 0.02, "hi": point + 0.02}


def _variant(pur, com, status="ok"):
    return {
        "models": {
            "lgbm_full": {
                "metrics": {"purity/QSO": _m(pur, status), "completeness/QSO": _m(com)},
                "detail": {"accuracy": 0.7251},
            }
        }
    }


def _results(drop=()):
    priors = {}
    for p in ("P1", "P2", "P3"):
        if p in drop:
            continue
        priors[p] = {
            "prior": {"QSO": QSO[p]},
            "variants": {
                "as_shipped": _variant(0.30, 0.90),
                "prior_matched": _variant(PURITY[p], 0.80),
            },
        }
    return {"populations": {"a_all": {"priors": priors}}}


def _rel(tmp_path, results=None, dl=None):
    if dl is not None:
        d = tmp_path / "dl" / "eval"
        d.mkdir(parents=True)
        (d / "results_dl.json").write_text(json.dumps(dl), encoding="utf-8")
    return SimpleNamespace(root=tmp_path, results=results or _results())


# (1) prior_states
def test_prior_states_keys_partial_and_no_p0(tmp_path):
    st = H.prior_states(_rel(tmp_path))
    assert list(st) == ["P1", "P2", "P3"]
    assert [p for p, s in st.items() if s["partial"]] == ["P3"]
    assert "P0" not in json.dumps(st)
    assert st["P2"]["purity"] == pytest.approx(0.70)


def test_low_count_cell_nulls_only_that_prior(tmp_path):
    res = _results()
    res["populations"]["a_all"]["priors"]["P1"]["variants"]["prior_matched"]["models"][
        "lgbm_full"
    ]["metrics"]["purity/QSO"]["status"] = "low_count"
    st = H.prior_states(_rel(tmp_path, res))
    assert all(st["P1"][k] is None for k in H._METRIC_FIELDS)
    assert st["P1"]["qso_prior"] == QSO["P1"]
    assert st["P2"]["purity"] == pytest.approx(0.70)
    assert st["P2"]["completeness_hi"] == pytest.approx(0.82)


def test_missing_p3_is_all_none_without_exception(tmp_path):
    st = H.prior_states(_rel(tmp_path, _results(drop=("P3",))))
    assert list(st) == ["P1", "P2", "P3"]
    assert st["P3"]["qso_prior"] is None
    assert all(st["P3"][k] is None for k in H._METRIC_FIELDS)
    assert st["P2"]["purity"] == pytest.approx(0.70)


# (2) accuracy_defs
def test_accuracy_defs_without_results_dl(tmp_path):
    a = H.accuracy_defs(_rel(tmp_path))
    assert a["shipped"] == pytest.approx(0.7251)
    for k in ("seeds_mean", "seeds_min", "seeds_max", "n_seeds", "regime"):
        assert a[k] is None
    assert a["balanced"] is False


def test_accuracy_defs_reads_seeds(tmp_path):
    dl = {
        "descriptive": {
            "regime": "BALANCED unweighted macro",
            "families": {
                "lgbm_full": {"accuracy": {"mean": 0.9, "min": 0.88, "max": 0.92}}
            },
            "systems": {f"lgbm_full_s{i}": {} for i in range(5)} | {"dl_x_s0": {}},
        }
    }
    a = H.accuracy_defs(_rel(tmp_path, dl=dl))
    assert (a["seeds_mean"], a["seeds_min"], a["seeds_max"]) == (0.9, 0.88, 0.92)
    assert a["n_seeds"] == 5
    assert a["regime"] == "BALANCED unweighted macro"
    assert a["balanced"] is True


# (3) builders


# (6) rules
@pytest.mark.parametrize("name", ["html_overview.py"])
def test_no_hand_written_digits(name):
    bad = [
        (n.lineno, n.value[:70])
        for n in _literals(DASH / name)
        if isinstance(n.value, str) and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


# (7) real release


