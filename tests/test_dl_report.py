"""dl_report (S13, ADR 0006 4.4, 15, 17): negative sentence, no verdict outside H1/H2, JSON strictness."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from dl_run_kit import WIN, env, run, world

from astro_triage.evaluation import dl_desc, dl_eval, dl_report

TIE_ROW = "tie: LightGBM stays"
ENS = "ens_mlp_full-lgbm_full"


def _res(tmp, shifts=None):
    ts, noise = world()
    systems, _, _ = env(tmp, ts, noise, shifts=shifts)
    cmp = run(tmp, ts, noise, shifts=shifts)
    desc, rows = dl_desc.describe(
        ts, systems, {n: {"parameter_count": 3} for n in systems}
    )
    (tmp / dl_eval.FREEZE_NAME).write_text("{}")
    return dl_desc.assemble(ts, cmp, desc, "0" * 64, "ok go"), rows


@pytest.fixture(scope="module")
def ties(tmp_path_factory):
    # ensembles win clearly, headline pairs are LightGBM copies (tie)
    return _res(
        tmp_path_factory.mktemp("t"), {"ens_mlp_full": {"QSO": WIN, "GALAXY": WIN}}
    )[0]


@pytest.fixture(scope="module")
def wins(tmp_path_factory):
    return _res(
        tmp_path_factory.mktemp("w"), {"mlp_full": {"QSO": WIN, "GALAXY": WIN}}
    )[0]


def test_negative_sentence_verbatim_only_on_double_tie(ties, wins):
    md = dl_report.render_markdown(ties)
    assert dl_report.NEGATIVE_SENTENCE in md
    assert md.count(dl_report.NEGATIVE_SENTENCE) == 1
    assert (
        "On the same 21 Gaia DR3 features, MLP and FT-Transformer (5 seeds, untuned defaults) do not "
        "raise the purity of QSO and GALAXY at completeness 0.80 under the Delchambre+2023 prior "
        "beyond 2 seed-SDs of LightGBM with a block-bootstrap CI above zero."
    ) in md
    mdw = dl_report.render_markdown(wins)
    assert "X wins" in mdw and dl_report.NEGATIVE_SENTENCE not in mdw


def test_only_one_h_wins_drops_sentence(ties):
    res = copy.deepcopy(ties)
    res["compare"]["pairs"]["ftt_full-lgbm_full"]["P2"]["0.80"]["verdict"] = "X wins"
    assert dl_report.NEGATIVE_SENTENCE not in dl_report.render_markdown(res)


def test_no_verdict_on_ensemble_rows(ties):
    md = dl_report.render_markdown(ties)
    assert (
        ties["compare"]["pairs"][ENS]["P2"]["0.80"]["classes"]["QSO"]["d"] > 0
    )  # real ensemble gain
    lines = [
        ln
        for ln in md.splitlines()
        if ln.startswith((f"| {ENS} ", "| ens_ftt"))
    ]
    assert lines
    for ln in lines:
        assert (
            "win" not in ln.lower()
            and "tie" not in ln.lower()
            and "worse" not in ln.lower()
        )
    head = md.split("## Descriptive pairs")[0]
    assert ENS not in head


def test_unreportable_cell_named_with_reasons(ties):
    res = copy.deepcopy(ties)
    c = res["compare"]["pairs"]["mlp_full-lgbm_full"]["P2"]["0.80"]["classes"]["QSO"]
    c.update(
        reportable=False,
        floor_ok=False,
        blocks=5,
        low_count=True,
        na=True,
        invalid_fraction=0.25,
        delta=None,
        val_checks_pass=False,
    )
    md = dl_report.render_markdown(res)
    sec = md.split("### Cells not reportable")[1].split("##")[0]
    line = next(
        ln for ln in sec.splitlines() if "mlp_full-lgbm_full P2 C*=0.80 QSO" in ln
    )
    for why in (
        "reporting floor",
        "only 5 blocks",
        "low-count",
        "25.0% invalid",
        "delta not finite",
        "val cells",
    ):
        assert why in line


def test_header_limitations_and_provisional(ties):
    md = dl_report.render_markdown(ties)
    assert f"**PROVISIONAL: {ties['meta']['n_test']} test rows" in md
    assert "B0 (ADR 0006 section 6)" in md and "R17" in md and "## Limitations" in md
    assert "ok go" in md and ties["meta"]["freeze_sha256"] in md


def test_json_strict_and_no_test_key(ties):
    res = copy.deepcopy(ties)
    res["compare"]["pairs"]["mlp_full-lgbm_full"]["P2"]["0.80"]["classes"]["QSO"][
        "d"
    ] = float("nan")
    res["descriptive"]["families"]["mlp_full"]["e_sq"]["mean"] = np.float64("inf")
    doc = json.loads(
        dl_report.to_json(res)
    )  # nan/inf become null, never the invalid NaN token
    assert (
        doc["compare"]["pairs"]["mlp_full-lgbm_full"]["P2"]["0.80"]["classes"]["QSO"][
            "d"
        ]
        is None
    )
    assert doc["descriptive"]["families"]["mlp_full"]["e_sq"]["mean"] is None
    assert "NaN" not in dl_report.to_json(res) and "Infinity" not in dl_report.to_json(
        res
    )

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)

    assert "test" not in set(keys(doc))


def test_csv_header_and_rows(ties):
    out = dl_report.risk_coverage_csv([["lgbm_full", "mean", 0.5, 1.0, float("nan")]])
    assert out.splitlines()[0] == "system,seed,tau,coverage,accuracy"
    assert out.splitlines()[1] == "lgbm_full,mean,0.5,1.0,nan"
