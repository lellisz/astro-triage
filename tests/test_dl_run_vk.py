"""F2: the headline cell carries v_k_spread (min, median, max of delta's LightGBM val purities) and the report shows it."""

from __future__ import annotations

import json
import math

import pytest
from dl_run_kit import cell, env, world

from astro_triage.evaluation import dl_report, dl_run, dl_val

PAIR = "mlp_full-lgbm_full"
NAN = float("nan")
VK = {
    ("QSO", "0.80"): [0.5, 0.9, 0.7, 0.1, 0.3],
    ("GALAXY", "0.80"): [0.2, 0.4, None, NAN, 0.6],  # non-finite entries are dropped
    ("QSO", "0.90"): [None, NAN, None, NAN, None],  # nothing finite -> null
    # ("GALAXY", "0.90"): key absent -> null
}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vk")
    ts, noise = world()
    systems, freeze, d = env(tmp, ts, noise)
    doc = json.loads((d / "delta.json").read_text())
    for (cls, cs), vk in VK.items():
        doc["delta"]["full"][cls][cs]["P2"] = {**doc["delta"]["full"][cls][cs]["P2"], "v_k": vk}
    (d / "delta.json").write_text(json.dumps(doc))  # json.dumps writes NaN literals; loads reads them back
    freeze["delta_sha256"] = dl_val.sha256_file(d / "delta.json")
    return dl_run.run_compare(ts, systems, freeze, lgbm_dir=d, n_boot=20)


def test_spread_min_median_max(res):
    q = cell(res, PAIR)["classes"]["QSO"]["v_k_spread"]
    assert q == {"min": 0.1, "median": 0.5, "max": 0.9}
    g = cell(res, PAIR, cs="0.80")["classes"]["GALAXY"]["v_k_spread"]
    assert g == {"min": 0.2, "median": 0.4, "max": 0.6}


def test_absent_or_nonfinite_v_k_is_null_without_breaking(res):
    assert cell(res, PAIR, cs="0.90")["classes"]["QSO"]["v_k_spread"] is None
    assert cell(res, PAIR, cs="0.90")["classes"]["GALAXY"]["v_k_spread"] is None
    # the cell is otherwise complete and the JSON stays strict
    k = cell(res, PAIR, cs="0.90")["classes"]["QSO"]
    assert math.isfinite(k["d"]) and "class_verdict" in k
    json.dumps(res, allow_nan=False)


def test_spread_only_on_headline_pairs(res):
    assert "v_k_spread" not in cell(res, "ens_mlp_full-lgbm_full")["classes"]["QSO"]


def _row(md, cls, cs):
    pre = f"| {PAIR} | P2 | {cs} |"
    rows = [r for r in md.splitlines() if r.startswith(pre) and f"| {cls} |" in r]
    assert len(rows) == 1, rows
    return rows[0]


def test_markdown_column_and_values(res):
    md = "\n".join(dl_report._headline(res["pairs"]))
    assert "v_k min / med / max (val, LightGBM seeds)" in md
    assert "| 0.1 / 0.5 / 0.9 |" in _row(md, "QSO", "0.80")
    assert "| 0.2 / 0.4 / 0.6 |" in _row(md, "GALAXY", "0.80")
    header = next(r for r in md.splitlines() if r.startswith("| pair |")).split(" | ")
    col = next(i for i, h in enumerate(header) if h.startswith("v_k"))
    assert _row(md, "QSO", "0.90").split(" | ")[col] == "n/a"
    assert _row(md, "GALAXY", "0.90").split(" | ")[col] == "n/a"
    assert _row(md, "QSO", "0.80").split(" | ")[col] == "0.1 / 0.5 / 0.9"


def test_report_tolerates_cell_without_v_k_spread(res):
    import copy

    r = copy.deepcopy(res["pairs"])
    del r[PAIR]["P2"]["0.80"]["classes"]["QSO"]["v_k_spread"]
    assert "n/a" in _row("\n".join(dl_report._headline(r)), "QSO", "0.80")
