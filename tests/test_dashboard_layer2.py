"""Proofs for the Layer 2 tab of the dashboard: data layer, chart and the Streamlit app.

The ``eval_l2.json`` files come from the REAL ``layer2_report.write_outputs`` over the synthetic
release / toy arrays of ``test_layer2_eval`` (no hand-written format). The app tests copy the
small real ``dataset_100k_v0`` release (read only) into ``tmp_path`` and point ``D.INTERIM_DIR``
there; they skip when that release is absent. No network, no key, nothing written under ``data/``.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest
from test_layer2_eval import MODEL, build_release
from test_layer2_report import dead_l2, with_eval

from astro_triage.dashboard import data as D
from astro_triage.evaluation import layer2_data as L2D
from astro_triage.evaluation import layer2_report as R

APP = Path(D.__file__).resolve().parent / "app.py"
HOSTILE = (
    "<script>alert(1)</script>",
    "**negrito**",
    "[x](javascript:alert(1))",
)


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def base_res(tmp_path_factory: pytest.TempPathFactory) -> dict:
    rel = build_release(tmp_path_factory.mktemp("dashl2"))
    return L2D.run_evaluation(rel.rel, MODEL, "blind", enable_thinking=False, n_boot=20)


@pytest.fixture(scope="module")
def results(base_res: dict) -> dict[str, dict]:
    """gain: both pairs; one: only S2-S0 (S1* abstains on everything); none: all below floor;
    flat: evaluable but the CI touches zero."""
    one = with_eval(base_res, 30)
    from test_layer2_eval import toy

    from astro_triage.evaluation import layer2 as L2

    one["eval"] = L2.evaluate(toy(30, tau_star=0.95), n_boot=50)
    return {
        "gain": with_eval(base_res, 30),
        "one": one,
        "none": with_eval(base_res, 1),
        "flat": with_eval(base_res, 30, **dead_l2()),
    }


def write_release(
    base: Path, res: dict | None, name: str = "rel", raw_text: str | None = None
) -> Path:
    """Create ``base/name/layer2/eval_l2.json`` (only the layer2 dir); return the release dir."""
    rel = base / name
    rel.mkdir(parents=True)
    if res is not None:
        R.write_outputs(res, rel / "layer2")
    if raw_text is not None:
        (rel / "layer2").mkdir(exist_ok=True)
        (rel / "layer2" / "eval_l2.json").write_text(raw_text, encoding="utf-8")
    return rel


def load(base: Path, res: dict, name: str = "rel") -> dict:
    write_release(base, res, name)
    out = D.load_layer2(name, base)
    assert out is not None
    return out


# ----------------------------------------------------------------------------- load_layer2
def test_load_layer2_is_none_without_the_file(tmp_path: Path) -> None:
    (tmp_path / "rel").mkdir()
    assert D.load_layer2("rel", tmp_path) is None
    (tmp_path / "rel" / "layer2").mkdir()  # directory without eval_l2.json
    assert D.load_layer2("rel", tmp_path) is None


def test_load_layer2_maps_frames_by_name(results: dict, tmp_path: Path) -> None:
    out = load(tmp_path, results["gain"])
    assert set(out) == {"raw", "systems", "routed", "diffs"}
    assert out["raw"] == json.loads(
        (tmp_path / "rel" / "layer2" / "eval_l2.json").read_text(encoding="utf-8")
    )
    sy, ro, di = out["systems"], out["routed"], out["diffs"]
    assert list(sy["unit"]) == ["S0", "S1", "S1*", "S2", "S3"]
    assert list(ro["unit"]) == list(D.LAYER2_ROUTED)
    units = out["raw"]["eval"]["units"]
    for r in sy.itertuples():  # values follow the unit NAME, not the position
        u = units[r.unit]
        assert r.n == u["n"] and r.n_answered == u["n_answered"]
        assert r.macro_f1 == pytest.approx(u["metrics"]["macro_f1"]["point"])
        assert r.accuracy_lo == pytest.approx(u["metrics"]["accuracy"]["lo"])
        assert r.coverage_hi == pytest.approx(u["metrics"]["coverage"]["hi"])
        assert r.macro_f1_mark == ""
    assert set(di["pair"]) == set(out["raw"]["eval"]["diffs"])
    assert set(di["metric"]) == set(D.LAYER2_SHOWN_METRICS)
    main = di[di["pair"].isin(["S2 - S0", "S3 - S1*", "S3 - S1"])]
    assert main["usable"].all() and (main["mark"] == "").all()
    below = di[
        ~di["pair"].isin(main["pair"])
    ]  # routed pairs: 30 objects, under the floor
    assert len(below) and (~below["usable"]).all() and (below["mark"] == "n/a").all()
    assert below[["point", "lo", "hi"]].isna().all().all()
    assert di["excludes_zero"].dtype == bool


def test_load_layer2_below_floor_is_nan_and_marked(
    results: dict, tmp_path: Path
) -> None:
    out = load(tmp_path, results["none"])
    sy = out["systems"]
    assert (sy["macro_f1_mark"] == "n/a").all()
    for key in D.LAYER2_SHOWN_METRICS:  # no raw value leaks into the frame
        for col in (key, f"{key}_lo", f"{key}_hi"):
            assert sy[col].isna().all(), col
    di = out["diffs"]
    assert not di["usable"].any() and (di["mark"] == "n/a").all()
    assert di[["point", "lo", "hi"]].isna().all().all()
    assert not di["excludes_zero"].any()


def test_load_layer2_corrupt_json_raises_valueerror_the_app_catches(
    tmp_path: Path,
) -> None:
    write_release(tmp_path, None, raw_text="{not json")
    with pytest.raises(
        ValueError
    ):  # JSONDecodeError; main() catches (OSError, ValueError, KeyError)
        D.load_layer2("rel", tmp_path)
    shutil.rmtree(tmp_path / "rel")
    write_release(tmp_path, None, raw_text="{}")
    with pytest.raises(KeyError):
        D.load_layer2("rel", tmp_path)


def test_load_layer2_accepts_a_release_object(results: dict, tmp_path: Path) -> None:
    rel_dir = write_release(tmp_path, results["gain"])
    rel = D.Release("rel", rel_dir, {"meta": {}}, {}, pd.DataFrame(), pd.DataFrame(), 0)
    assert D.load_layer2(rel) is not None


# --------------------------------------------------------------------------- layer2_summary
def test_summary_gain_vs_no_gain_follows_the_preregistered_rule(
    results: dict, tmp_path: Path
) -> None:
    for i, (key, outcome) in enumerate(
        [("gain", "gain"), ("one", "gain"), ("none", "no_gain"), ("flat", "no_gain")]
    ):
        s = D.layer2_summary(load(tmp_path, results[key], f"r{i}"))
        assert s["outcome"] == outcome == results[key]["eval"]["verdict"]["outcome"], (
            key
        )
        assert s["metric"] == "macro_f1"


def test_summary_pairs_one_of_two_gain(results: dict, tmp_path: Path) -> None:
    s = D.layer2_summary(load(tmp_path, results["one"]))
    assert s["n_pairs"] == 2
    p = s["pairs"].set_index("pair")
    assert list(p["gain"]) == [True, False] and int(p["gain"].sum()) == 1
    assert p.loc["S2 - S0", "evaluable"] and not p.loc["S3 - S1*", "evaluable"]
    assert p.loc["S2 - S0", "mark"] == "" and p.loc["S2 - S0", "lo"] > 0
    assert p.loc["S3 - S1*", "mark"] == "n/a"
    assert math.isnan(p.loc["S3 - S1*", "point"])
    assert s["n_evaluable"] == 1


def test_summary_both_pairs_gain(results: dict, tmp_path: Path) -> None:
    s = D.layer2_summary(load(tmp_path, results["gain"]))
    assert s["n_pairs"] == 2 and int(s["pairs"]["gain"].sum()) == 2
    assert s["n_evaluable"] == 2 and (s["pairs"]["lo"] > 0).all()


def test_summary_below_floor_pairs_are_na_without_raw_values(
    results: dict, tmp_path: Path
) -> None:
    s = D.layer2_summary(load(tmp_path, results["none"]))
    p = s["pairs"]
    assert (
        (p["mark"] == "n/a").all() and not p["gain"].any() and not p["evaluable"].any()
    )
    assert p[["point", "lo", "hi"]].isna().all().all()
    assert s["n_evaluable"] == 0 and s["n_pairs"] == 2


def test_summary_flat_pairs_are_evaluable_but_not_gain(
    results: dict, tmp_path: Path
) -> None:
    s = D.layer2_summary(load(tmp_path, results["flat"]))
    p = s["pairs"]
    assert p["evaluable"].all() and not p["gain"].any() and (p["mark"] == "").all()
    assert (p["lo"] <= 0).all()


def test_summary_none_percentiles_become_nan(results: dict, tmp_path: Path) -> None:
    res = copy.deepcopy(results["gain"])
    res["operations"]["latency_ms"].update(mean=None, p50=None, p95=None)
    s = D.layer2_summary(load(tmp_path, res))
    row = s["cost"].set_index("item").loc["latência (ms)"]
    assert row[["mean", "p50", "p95"]].isna().all()


# ------------------------------------------------------------------- old functions intact
def test_old_public_signatures_unchanged() -> None:
    sig = {
        n: str(inspect.signature(getattr(D, n)))
        for n in (
            "load_release",
            "headline",
            "wilson",
            "confusion",
            "errors_by_type",
            "abstention_curve",
            "confidence_histogram",
            "mollweide",
        )
    }
    assert (
        sig["load_release"]
        == "(name: 'str' = 'dataset_100k_v0', base: 'Path | None' = None) -> 'Release'"
    )
    assert (
        sig["headline"] == "(rel: 'Release', model: 'str' = 'lgbm_full') -> 'Headline'"
    )
    assert (
        sig["wilson"]
        == "(k: 'int', n: 'int', z: 'float' = 1.96) -> 'tuple[float, float]'"
    )
    assert (
        sig["confusion"]
        == "(rel: 'Release', model: 'str' = 'lgbm_full') -> 'pd.DataFrame'"
    )
    assert (
        sig["errors_by_type"]
        == "(rel: 'Release', model: 'str' = 'lgbm_full') -> 'pd.DataFrame'"
    )
    assert (
        list(inspect.signature(D.abstention_curve).parameters)[:2]
        == ["rel", "thresholds"]
        or "rel" in sig["abstention_curve"]
    )
    assert D.DEFAULT_RELEASE == "dataset_100k_v0" and D.CLASSES == (
        "STAR",
        "GALAXY",
        "QSO",
    )
    assert D.wilson(0, 10)[0] == 0.0


# -------------------------------------------------------------------------------- charts


# ------------------------------------------------------------------------------------ app


