"""run_compare (S12, ADR 0006 sections 4.2-4.4, 15): verdicts, roles, determinism, freeze guards."""

from __future__ import annotations

import json

import pytest
from dl_run_kit import STRONG, WIN, cell, env, run, world

from astro_triage.evaluation import dl_compare as C
from astro_triage.evaluation import dl_run
from astro_triage.evaluation.runner import EvalError
from eval import metrics as M

PAIR = "mlp_full-lgbm_full"
BOTH = {"QSO": WIN, "GALAXY": WIN}
PERFECT = {"QSO": STRONG, "GALAXY": STRONG}


@pytest.fixture(scope="module")
def w():
    return world()


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    store: dict = {}

    def get(name, ts, noise, **kw):
        if name not in store:
            store[name] = run(tmp_path_factory.mktemp(name), ts, noise, **kw)
        return store[name]

    return get


def test_copies_of_lgbm_give_zero_and_tie(cache, w):
    r = cache("copies", *w)
    for pair in (PAIR, "ftt_full-lgbm_full"):
        e = cell(r, pair)
        assert all(e["classes"][c]["d"] == 0.0 for c in C.TARGET_CLASSES)
        assert e["verdict"] == "tie: LightGBM stays"
        assert {e["classes"][c]["class_verdict"] for c in C.TARGET_CLASSES} == {"tie"}


def test_qso_only_win_is_a_tie(cache, w):
    r = cache("qso", *w, shifts={"mlp_full": {"QSO": WIN}})
    e = cell(r, PAIR)["classes"]
    assert e["QSO"]["class_verdict"] == "win" and e["QSO"]["d"] > e["QSO"]["delta"]
    assert e["GALAXY"]["class_verdict"] == "tie"
    assert cell(r, PAIR)["verdict"] == "tie: LightGBM stays"


def test_win_on_both_classes(cache, w):
    r = cache("both", *w, shifts={"mlp_full": BOTH})
    assert cell(r, PAIR)["verdict"] == "X wins"
    assert cell(r, "ftt_full-lgbm_full")["verdict"] == "tie: LightGBM stays"


def test_within_delta_is_a_tie(cache, w):
    r = cache("small", *w, shifts={"mlp_full": BOTH}, delta=0.5)
    for c in C.TARGET_CLASSES:
        k = cell(r, PAIR)["classes"][c]
        assert 0 < k["d"] <= k["delta"] and k["ci"][0] > 0
        assert k["class_verdict"] == "tie"
    assert cell(r, PAIR)["verdict"] == "tie: LightGBM stays"


def test_failed_val_checks_is_a_tie(cache, w):
    r = cache("nochk", *w, shifts={"mlp_full": BOTH}, checks=False)
    k = cell(r, PAIR)["classes"]["QSO"]
    assert k["d"] > k["delta"] and not k["reportable"] and k["class_verdict"] == "tie"


def test_galaxy_below_floor_is_a_tie_with_reason():
    ts, noise = world(counts=(1500, 40, 1500))
    import tempfile
    from pathlib import Path

    r = run(Path(tempfile.mkdtemp()), ts, noise, shifts={"mlp_full": BOTH})
    g = cell(r, PAIR)["classes"]["GALAXY"]
    assert g["d"] > g["delta"] and g["floor_ok"] is False
    assert not g["reportable"] and g["class_verdict"] == "tie"
    assert cell(r, PAIR)["classes"]["QSO"]["class_verdict"] == "win"
    assert cell(r, PAIR)["verdict"] == "tie: LightGBM stays"


def test_low_count_is_a_tie_with_reason(cache, w):
    # near-perfect X selects (almost) no contaminants: total < 10 -> not reportable
    r = cache("low", *w, shifts={"mlp_full": PERFECT})
    k = cell(r, PAIR)["classes"]["QSO"]
    assert k["low_count"] is True and k["reportable"] is False
    assert k["d"] > k["delta"] and k["class_verdict"] == "tie"


def test_ensemble_wins_without_verdict_and_descriptive_has_none(cache, w):
    r = cache("ens", *w, shifts={"ens_mlp_full": PERFECT, "ens_ftt_full": PERFECT})
    for pair, per_p in r["pairs"].items():
        is_head = pair in ("mlp_full-lgbm_full", "ftt_full-lgbm_full")
        for p, by_cs in per_p.items():
            for cs, e in by_cs.items():
                assert ("verdict" in e) is is_head, (pair, p, cs)
                for k in e["classes"].values():
                    assert ("class_verdict" in k) is is_head
    e = cell(r, "ens_mlp_full-lgbm_full")
    assert e["classes"]["QSO"]["d"] > 0.05 and e["role"] == "descriptive"


def test_roles_and_level5_only_on_p2(cache, w):
    r = cache("copies", *w)
    for pair, per_p in r["pairs"].items():
        for p, by_cs in per_p.items():
            for cs, e in by_cs.items():
                head = pair in (PAIR, "ftt_full-lgbm_full")
                if head:
                    want = "headline" if (p, cs) == ("P2", "0.80") else "sensitivity"
                    assert e["role"] == want
                for k in e["classes"].values():
                    assert ("ci_l5" in k) is (p == "P2"), (pair, p, cs)
    assert {"P1", "P2", "P3"} == set(r["pairs"][PAIR])
    assert cell(r, "mlp_noind-lgbm_noind")["role"] == "front2"


def test_same_seed_identical_twice(cache, w, tmp_path):
    a = cache("both", *w, shifts={"mlp_full": BOTH})
    b = run(tmp_path, *w, shifts={"mlp_full": BOTH})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["seed"] == M.SEED


def test_na_cell_when_over_one_percent_invalid(tmp_path):
    ts, noise = world(
        counts=(1500, 1500, 1)
    )  # one QSO: its pixel is missed in ~37 % of draws
    k = cell(run(tmp_path, ts, noise), PAIR)["classes"]["QSO"]
    assert k["na"] is True and k["ci"] == [None, None]
    assert k["invalid_fraction"] > M.MAX_INVALID_FRACTION
    assert k["reportable"] is False and k["class_verdict"] == "tie"


def test_replicate_stream_shared_and_thresholds_never_refit(monkeypatch, tmp_path, w):
    seen = []
    orig = M.bootstrap_pixel_draw

    def spy(pix, band, rng):
        seen.append(rng.bit_generator.state["state"]["state"])
        return orig(pix, band, rng)

    def boom(*a, **k):
        raise AssertionError("threshold refitted on test")

    monkeypatch.setattr(M, "bootstrap_pixel_draw", spy)
    monkeypatch.setattr(C, "threshold_at_completeness", boom)
    run(tmp_path, *w, n_boot=7)
    assert len(seen) == 3 * 7  # one draw per replicate and group, not per system
    assert seen[0] == seen[7] == seen[14]  # every group starts at default_rng(SEED)


def test_thresholds_come_from_the_freeze(tmp_path, w):
    ts, noise = w
    systems, freeze, d = env(tmp_path, ts, noise)
    freeze["thresholds"]["mlp_full_s0"]["QSO"]["0.80"] = 2.0  # nothing selected
    r = dl_run.run_compare(ts, systems, freeze, lgbm_dir=d, n_boot=5)
    rc = cell(r, PAIR)["classes"]["QSO"]["realised_completeness"]
    assert rc["mlp_full_s0"] == 0.0 and rc["mlp_full_s1"] > 0.5


def test_delta_json_must_match_freeze_sha(tmp_path, w):
    ts, noise = w
    systems, freeze, d = env(tmp_path, ts, noise)
    (d / "delta.json").write_text(
        (d / "delta.json").read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(EvalError, match="delta.json differs"):
        dl_run.run_compare(ts, systems, freeze, lgbm_dir=d, n_boot=2)
    (d / "delta.json").unlink()
    with pytest.raises(EvalError, match="delta.json differs"):
        dl_run.run_compare(ts, systems, freeze, lgbm_dir=d, n_boot=2)
