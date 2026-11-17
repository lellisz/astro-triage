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

from astro_triage.dashboard import charts as C
from astro_triage.dashboard import data as D
from astro_triage.evaluation import layer2_data as L2D
from astro_triage.evaluation import layer2_report as R

APP = Path(D.__file__).resolve().parent / "app.py"
REAL_RELEASE = D.REPO_ROOT / "data" / "interim" / D.DEFAULT_RELEASE
HOSTILE = (
    "<script>alert(1)</script>",
    "**negrito**",
    "[x](javascript:alert(1))",
)
NO_GAIN_TITLE = "não agregou valor mensurável"
PAGES_BASE = [
    "Visão geral",
    "Mapa do céu",
    "Galáxias",
    "Método",
    "Comparação: LightGBM × deep learning",
]


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


def test_summary_failure_unknown_status_cost_and_meta(
    results: dict, tmp_path: Path
) -> None:
    res = results["gain"]
    ops = res["operations"]
    s = D.layer2_summary(load(tmp_path, res))
    assert s["unknown_rate"] == pytest.approx(ops["unknown_rate"])
    assert s["failure_rate"] == pytest.approx(ops["failure_rate"])
    assert s["unknown_rate"] > 0 and s["failure_rate"] > 0  # fixture has both kinds
    st = s["status"].set_index("status")
    assert st.loc["OK", "kind"] == "ok" and st.loc["UNKNOWN", "kind"] == "unknown"
    assert st.loc["TIMEOUT", "kind"] == "falha"
    for key, n in ops["status_counts"].items():
        assert st.loc[key, "n"] == n
        assert st.loc[key, "rate"] == pytest.approx(n / ops["n_routed"])
    assert st["rate"].sum() == pytest.approx(1.0)
    assert st.loc["UNKNOWN", "rate"] == pytest.approx(s["unknown_rate"])
    assert st.loc[st["kind"] == "falha", "rate"].sum() == pytest.approx(
        s["failure_rate"]
    )
    cost = s["cost"].set_index("item")
    assert list(cost.index) == list(D.LAYER2_COST_ROWS.values())
    for key, label in D.LAYER2_COST_ROWS.items():
        for stat in ("mean", "p50", "p95"):
            assert cost.loc[label, stat] == pytest.approx(ops[key][stat])
        assert cost.loc[label, "n"] == ops[key]["n"]
    assert cost.loc["latência (ms)", "mean"] == pytest.approx(1004.0)
    assert cost.loc["tokens no total", "p50"] == pytest.approx(314.0)
    assert s["model_requested"] == res["meta"]["model_requested"] == MODEL
    assert s["models_reported"] == {MODEL: ops["n_routed"]}
    assert s["prompt_version"] == res["meta"]["prompt_version"]
    assert (
        s["n_routed"] == ops["n_routed"] and s["n_boot"] == res["eval"]["n_boot"] == 30
    )
    assert s["class_mix"] == ops["class_mix"]
    assert s["floor"] == res["meta"]["floor"]


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
def test_layer2_f1_valid_figure_and_omits_na(results: dict, tmp_path: Path) -> None:
    sy = load(tmp_path, results["gain"])["systems"].copy()
    fig = C.layer2_f1(sy)
    fig.to_dict()  # validates against the plotly schema
    assert len(fig.data) == 2 * 5
    ys = {y for t in fig.data for y in t.y}
    assert ys == {"S0", "S1", "S1*", "S2", "S3"}
    sy.loc[
        sy["unit"] == "S1*", ["macro_f1_mark", "macro_f1", "macro_f1_lo", "macro_f1_hi"]
    ] = [
        "n/a",
        math.nan,
        math.nan,
        math.nan,
    ]
    fig2 = C.layer2_f1(sy)
    assert len(fig2.data) == 2 * 4
    assert "S1*" not in {y for t in fig2.data for y in t.y}
    assert not any(math.isnan(v) for t in fig2.data for v in t.x)


def test_layer2_f1_with_everything_na_or_empty_does_not_raise(
    results: dict, tmp_path: Path
) -> None:
    sy = load(tmp_path, results["none"])["systems"]
    assert len(C.layer2_f1(sy).data) == 0
    empty = sy.iloc[0:0]
    assert len(C.layer2_f1(empty).data) == 0
    C.layer2_f1(empty).to_dict()


# ------------------------------------------------------------------------------------ app
@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Factory: copy the real small release into tmp, optionally add layer2/eval_l2.json."""
    if not (REAL_RELEASE / "eval" / "results.json").is_file():
        pytest.skip("real dataset_100k_v0 release not present")
    import streamlit as st

    counter = [0]

    def make(res: dict | None, mutate=None, raw_text: str | None = None):
        counter[0] += 1
        name = f"app_rel_{counter[0]}_{tmp_path.name[-6:]}"
        dst = tmp_path / "interim" / name
        shutil.copytree(REAL_RELEASE, dst, ignore=shutil.ignore_patterns("models"))
        if res is not None:
            R.write_outputs(res, dst / "layer2")
            if mutate is not None:
                p = dst / "layer2" / "eval_l2.json"
                p.write_text(mutate(p.read_text(encoding="utf-8")), encoding="utf-8")
        if raw_text is not None:
            (dst / "layer2").mkdir(exist_ok=True)
            (dst / "layer2" / "eval_l2.json").write_text(raw_text, encoding="utf-8")
        return name

    monkeypatch.setattr(D, "INTERIM_DIR", tmp_path / "interim")
    (tmp_path / "interim").mkdir()
    st.cache_resource.clear()
    st.cache_data.clear()
    yield make
    st.cache_resource.clear()
    st.cache_data.clear()


def run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    return at.run()


def nav(at):
    """The page selector: ``st.segmented_control(key="page")`` is a ButtonGroup in AppTest."""
    found = [w for w in at.button_group if w.key == "page"]
    assert len(found) == 1, [w.key for w in at.button_group]
    return found[0]


def pages(at) -> list[str]:
    return list(nav(at).options)


def open_page(at, page: str):
    at.session_state["page"] = page
    return at.run()


def md_values(at) -> list[str]:
    return [m.value for m in at.markdown]


def test_app_without_layer2_has_three_pages_in_order(app_env) -> None:
    app_env(None)
    at = run_app()
    assert not at.exception
    assert pages(at) == PAGES_BASE  # order and set; no Camada 2 entry
    assert at.session_state["page"] == "Visão geral" and not at.tabs
    assert not at.error


def test_app_with_layer2_has_four_pages_and_both_verdict_titles(
    app_env, results: dict
) -> None:
    app_env(results["one"])
    at = run_app()
    assert not at.exception
    assert pages(at) == [*PAGES_BASE, "Camada 2"]
    assert not any("comparações" in s.value for s in at.subheader)  # not rendered yet
    at = open_page(at, "Camada 2")
    assert not at.exception and at.session_state["page"] == "Camada 2"
    titles = [s.value for s in at.subheader]
    assert any("em 1 de 2 comparações" in t for t in titles), titles
    assert not any(NO_GAIN_TITLE in t for t in titles)
    assert len(at.dataframe) >= 3 and not at.error


def test_app_with_both_pairs_gaining_says_two_of_two(app_env, results: dict) -> None:
    app_env(results["gain"])
    at = open_page(run_app(), "Camada 2")
    assert not at.exception
    assert any("em 2 de 2 comparações" in s.value for s in at.subheader)


@pytest.mark.parametrize("key", ["none", "flat"])
def test_app_no_gain_title(app_env, results: dict, key: str) -> None:
    app_env(results[key])
    at = run_app()
    assert not at.exception and len(pages(at)) == len(PAGES_BASE) + 1
    at = open_page(at, "Camada 2")
    assert not at.exception
    titles = [s.value for s in at.subheader]
    assert any(NO_GAIN_TITLE in t for t in titles), titles
    assert not any("comparações" in t for t in titles)


def test_app_hostile_server_strings_never_reach_markdown_or_html(
    app_env, results: dict
) -> None:
    def hostile(raw: str) -> str:
        doc = json.loads(raw)
        doc["meta"]["model_requested"] = HOSTILE[0]
        doc["meta"]["prompt_version"] = HOSTILE[1]
        ops = doc["operations"]
        ops["models_reported"] = {HOSTILE[0]: 3, HOSTILE[1]: 2, HOSTILE[2]: 1}
        ops["status_counts"]["x" + HOSTILE[2]] = 1
        ops["status_rates"]["x" + HOSTILE[2]] = 0.0
        return json.dumps(doc)

    app_env(results["gain"], mutate=hostile)
    at = run_app()
    assert not at.exception
    assert len(pages(at)) == len(PAGES_BASE) + 1
    at = open_page(at, "Camada 2")
    assert not at.exception
    everything_md = "\n".join(md_values(at))
    for bad in HOSTILE:
        assert bad not in everything_md, bad
    assert not any("javascript:" in v for v in md_values(at))
    assert not any(el.type in {"html"} for el in at.main)  # no raw html element at all
    texts = "\n".join(t.value for t in at.text)
    for bad in HOSTILE:
        assert bad in texts  # shown literally, as plain text
    frames = "\n".join(df.value.to_csv() for df in at.dataframe)
    assert "x" + HOSTILE[2] in frames  # status key only in the dataframe


def test_app_corrupt_json_shows_visible_error_and_keeps_three_pages(
    app_env,
) -> None:
    app_env(None, raw_text="{not json")
    at = run_app()
    assert not at.exception
    assert pages(at) == PAGES_BASE
    assert any("ilegível" in e.value for e in at.sidebar.error)
    assert at.session_state["page"] == "Visão geral"


def test_app_missing_key_json_shows_visible_error(app_env) -> None:
    app_env(None, raw_text="{}")
    at = run_app()
    assert not at.exception and pages(at) == PAGES_BASE
    assert any("ilegível" in e.value for e in at.sidebar.error)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="KNOWN DEFECT (MINOR): valid JSON that is not an object raises TypeError, which "
    "main() does not catch (it catches OSError, ValueError, KeyError). Flips when fixed.",
)
@pytest.mark.parametrize("raw_text", ["[]", "null", "42"])
def test_app_non_object_json_does_not_crash(app_env, raw_text: str) -> None:
    app_env(None, raw_text=raw_text)
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
