"""Behaviour of ADR 0005 Amendment 1 (section 17) on the evaluation side: tau2 infeasible and
uncalibrated (17.1), thinking-aware parquet names (17.2), val-only audited fit and no
``--a-keep2`` (17.6), forced verdict. Synthetic releases in ``tmp_path``; no network, no key."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_layer2_eval import (
    MODEL,
    Rel,
    build_release,
    cli,
    diffs_with,
    load_cli,
    nothing_written,
    toy,
)

from astro_triage.evaluation import layer2 as L2
from astro_triage.evaluation import layer2_data as D
from astro_triage.evaluation.layer2_report import render_markdown, to_jsonable
from astro_triage.llm.consolidate import output_path
from astro_triage.llm.prompt import PROMPT_VERSION
from astro_triage.llm.routing import model_slug


def ev(r: Rel, thinking: bool = False) -> dict:
    return D.run_evaluation(r.rel, MODEL, "blind", enable_thinking=thinking, n_boot=10)


def ptd(res: dict, unit: str) -> dict[str, float]:
    return {k: c["point"] for k, c in res["eval"]["units"][unit]["metrics"].items()}


@pytest.fixture(scope="module")
def infeasible(tmp_path_factory: pytest.TempPathFactory) -> Rel:
    return build_release(tmp_path_factory.mktemp("inf"), val_acc=0.0)


@pytest.fixture(scope="module")
def uncal(tmp_path_factory: pytest.TempPathFactory) -> Rel:
    return build_release(tmp_path_factory.mktemp("unc"), n_val=150)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Rel:
    return build_release(tmp_path_factory.mktemp("fit"))


# ----------------------------------------------------------------- 17.1 infeasible
def test_infeasible_tau2_is_not_refused_and_forces_no_gain(infeasible: Rel) -> None:
    code, _out, err = cli(infeasible)
    assert code == 0, err
    res = json.loads((infeasible.rel / "layer2" / "eval_l2.json").read_text("utf-8"))
    fit = res["fit"]
    assert fit["n_val_ok"] >= 100  # the verbal calibration exists, tau2 is what fails
    assert fit["tau2"] is None  # inf is not serialisable: null
    assert fit["tau2_status"].startswith("infeasible")
    assert res["eval"]["verdict"]["outcome"] == "no_gain"
    assert "tau2 infeasible on val" in res["eval"]["verdict"]["forced_reason"]
    md = (infeasible.rel / "layer2" / "eval_l2.md").read_text("utf-8")
    assert "adds no measurable value" in md
    assert fit["tau2_status"] in md  # the status is printed, not just a null


def test_infeasible_tau2_is_inf_in_memory_and_s3_equals_s1_s2_equals_s0(
    infeasible: Rel,
) -> None:
    res = ev(infeasible)
    assert res["fit"]["tau2"] == float("inf")
    assert (
        res["fit"]["tau_star"] is not None
    )  # tau* is still fitted from S3's val coverage
    # No Layer 2 answer is trusted: every metric point of S2 equals S0 and S3 equals S1
    assert ptd(res, "S2") == ptd(res, "S0")
    assert ptd(res, "S3") == ptd(res, "S1")
    # ... although the Layer 2 answers exist (they are wrong here), so this is not vacuous
    assert res["eval"]["units"]["L2_forced"]["n"] > 0
    assert res["eval"]["verdict"]["forced_reason"] == D.FORCED_REASON_INFEASIBLE
    # bootstrap and tables are still computed
    assert res["eval"]["diffs"]["S2 - S0"]["macro_f1"]["point"] == 0.0


def test_forced_reason_never_turns_no_gain_into_gain() -> None:
    d = diffs_with(0.5, 0.5)  # both headline cells positive and evaluable
    assert L2.headline_verdict(d)["outcome"] == "gain"
    v = L2.headline_verdict(d, forced_reason="because")
    assert v["outcome"] == "no_gain" and v["forced_reason"] == "because"
    out = L2.evaluate(toy(30), n_boot=20, forced_reason="because")
    assert out["verdict"]["outcome"] == "no_gain"
    assert L2.evaluate(toy(30), n_boot=20)["verdict"]["outcome"] == "gain"


# ------------------------------------------------------------- 17.1 uncalibrated
def test_uncalibrated_not_applied_and_s3_trusts_every_ok_answer(uncal: Rel) -> None:
    code, _, err = cli(uncal)
    assert code == 0, err
    res = ev(uncal)
    fit = res["fit"]
    assert fit["n_val_ok"] < 100 and fit["tau2"] is None
    assert fit["tau2_status"].startswith("not_applied")
    assert fit["tau2_status"] != D.TAU2_INFEASIBLE
    # S3 gives a Layer 2 label to every routed test answer with status OK and a class label
    tbl = pq.read_table(uncal.out("test")).to_pandas()
    ok = tbl[(tbl["status"].astype(str) == "OK") & tbl["label_l2"].isin(D.CLASSES)]
    n_test = res["meta"]["n_test"]
    n_routed = len(tbl)
    s3 = res["eval"]["units"]["S3"]
    assert s3["n_answered"] == (n_test - n_routed) + len(ok)
    assert res["eval"]["verdict"]["outcome"] in ("gain", "no_gain")
    assert "forced_reason" not in res["eval"]["verdict"]


def test_null_tau2_alone_is_not_interpreted_the_status_decides(
    infeasible: Rel, uncal: Rel
) -> None:
    a, b = ev(infeasible), ev(uncal)
    ja, jb = to_jsonable(a["fit"]), to_jsonable(b["fit"])
    assert ja["tau2"] is None and jb["tau2"] is None  # indistinguishable by tau2
    assert ja["tau2_status"] != jb["tau2_status"]
    # opposite semantics behind the same null: trust none vs trust all
    sa, sb = a["eval"]["units"], b["eval"]["units"]
    assert sa["S3"]["n_answered"] == sa["S1"]["n_answered"]
    assert sb["S3"]["n_answered"] > sb["S1"]["n_answered"]
    ma, mb = render_markdown(a), render_markdown(b)
    assert D.TAU2_INFEASIBLE in ma and D.TAU2_UNCALIBRATED in mb
    assert "Uncalibrated" in mb and "Uncalibrated" not in ma


def test_markdown_renders_for_the_three_statuses(
    infeasible: Rel, uncal: Rel, fitted: Rel
) -> None:
    for r, status in (
        (infeasible, D.TAU2_INFEASIBLE),
        (uncal, D.TAU2_UNCALIBRATED),
        (fitted, D.TAU2_FITTED),
    ):
        res = ev(r)
        assert res["fit"]["tau2_status"] == status
        assert status in render_markdown(res)
    assert ev(fitted)["fit"]["tau2"] is not None


# ---------------------------------------------------------------- 17.2 parquet name
def test_parquet_name_carries_the_reasoning_setting(tmp_path: Path) -> None:
    off = output_path(tmp_path, "val", MODEL, "blind", enable_thinking=False)
    on = output_path(tmp_path, "val", MODEL, "blind", enable_thinking=True)
    assert off != on
    base = f"predictions_l2_{model_slug(MODEL)}_{PROMPT_VERSION}_blind"
    assert off.name == f"{base}_think-off_val.parquet"
    assert on.name == f"{base}_think-on_val.parquet"
    with pytest.raises(TypeError):  # keyword required: a stale caller fails loudly
        output_path(tmp_path, "val", MODEL, "blind")  # type: ignore[call-arg]


def _make_think_on(r: Rel) -> None:
    for fold in ("val", "test"):
        shutil.copy(r.out(fold), r.out(fold, thinking=True))
        tbl = pq.read_table(r.out(fold, thinking=True))
        md = {k.decode(): v.decode() for k, v in (tbl.schema.metadata or {}).items()}
        md["enable_thinking"] = "true"
        pq.write_table(tbl.replace_schema_metadata(md), r.out(fold, thinking=True))


def test_evaluator_thinking_reads_only_the_think_on_file(tmp_path: Path) -> None:
    r = build_release(tmp_path)
    code, out, err = cli(r, "--thinking")  # only think-off exists
    assert code == 1 and "not found" in err and "think-on" in err
    assert out == "" and nothing_written(r)
    _make_think_on(r)
    assert r.out("val").exists() and r.out("val", thinking=True).exists()
    code, _, err = cli(r, "--thinking")
    assert code == 0, err
    res = json.loads((r.rel / "layer2" / "eval_l2.json").read_text("utf-8"))
    assert res["meta"]["enable_thinking"] == "true"
    # and without the flag the think-off file is still what is read
    code, _, err = cli(r)
    assert code == 0, err
    res = json.loads((r.rel / "layer2" / "eval_l2.json").read_text("utf-8"))
    assert res["meta"]["enable_thinking"] == "false"


def test_think_on_file_whose_metadata_says_off_is_refused(tmp_path: Path) -> None:
    r = build_release(tmp_path)
    for fold in ("val", "test"):  # same bytes, name says on, metadata says off
        shutil.copy(r.out(fold), r.out(fold, thinking=True))
    code, _, err = cli(r, "--thinking")
    assert code == 1 and "enable_thinking" in err


def test_legacy_named_parquet_is_never_read(tmp_path: Path) -> None:
    r = build_release(tmp_path)
    for fold in ("val", "test"):
        legacy = r.out(fold).with_name(
            f"predictions_l2_{model_slug(MODEL)}_{PROMPT_VERSION}_blind_{fold}.parquet"
        )
        r.out(fold).rename(legacy)  # the only file left has the pre-amendment name
    code, out, err = cli(r)
    assert code == 1 and "not found" in err and out == ""
    assert nothing_written(r)


# ---------------------------------------------------------- 17.6 audited fit, no override
def test_a_keep2_flag_is_removed(fitted: Rel) -> None:
    with pytest.raises(SystemExit) as exc:
        load_cli().main(
            ["--release", "rel", "--model", MODEL, "--base", str(fitted.base)]
            + ["--a-keep2", "0.8"]
        )
    assert exc.value.code == 2
    import inspect

    assert "a_keep2" not in inspect.signature(D.run_evaluation).parameters


def test_fit_block_is_deterministic_val_only_and_audited(tmp_path: Path) -> None:
    a = build_release(tmp_path / "a", test_acc=0.8)
    b = build_release(tmp_path / "b", test_acc=0.2)  # only the test parquet differs
    fa, fb = ev(a)["fit"], ev(b)["fit"]
    dump = lambda f: json.dumps(to_jsonable(f), sort_keys=True)
    assert dump(fa) == dump(ev(a)["fit"]) == dump(fb)  # twice, and test-independent
    assert (
        fa["val_parquet_sha256"]
        == hashlib.sha256(a.out("val").read_bytes()).hexdigest()
    )
    assert fa["routing_manifest_sha256"] == a.sha()
    # a_keep2 comes from the frozen manifest only
    assert fa["a_keep2"] == 0.8 and fa["r_max2"] == D.R_MAX_TAU2 == 1.0
    assert ev(a)["eval"]["units"]["S3"] != ev(b)["eval"]["units"]["S3"]
    assert np.isfinite(fa["tau_star"])
