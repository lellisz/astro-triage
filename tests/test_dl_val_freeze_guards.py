"""build_freeze guards (ADR 0006 section 5): stale components, ordering, val-only thresholds, tolerances, constants."""

from __future__ import annotations

import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_dl_val_freeze as F
from test_dl_val_freeze import APPROVAL, N_TEST, make_env, val_p

from astro_triage.evaluation import dl_val as V
from astro_triage.evaluation.runner import GRID, EvalError
from astro_triage.models import dl_session


def tamper(path):
    table = pq.read_table(path)
    meta = table.schema.metadata
    col = table.column("p_cal_star").to_numpy()[::-1].copy() * 0.5 + 0.01
    table = table.set_column(table.schema.get_field_index("p_cal_star"), "p_cal_star", pa.array(col))
    pq.write_table(table.replace_schema_metadata(meta), path)


def refresh_delta(lg):
    shas = {fs: {f"lgbm_{fs}_s{k}": V.sha256_file(lg / f"predictions_lgbm_{fs}_s{k}_v0.parquet") for k in range(5)}
            for fs in ("full", "noind")}
    (lg / "delta.json").write_text(json.dumps({"prediction_sha256": shas}), encoding="utf-8")


def set_meta(path, raw):
    table = pq.read_table(path)
    meta = dict(table.schema.metadata)
    meta.pop(b"component_sha256")
    if raw is not None:
        meta[b"component_sha256"] = raw.encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


@pytest.mark.parametrize(("net", "k"), [("mlp", 1), ("ftt", 4), ("mlp", 4), ("ftt", 0)])
def test_net_component_changed_after_ensemble_refused(tmp_path, net, k):
    out, dl, lg = make_env(tmp_path)
    tamper(dl / f"predictions_{net}_full_s{k}_v0.parquet")
    with pytest.raises(EvalError) as ei:
        V.build_freeze(out, dl, lg, APPROVAL)
    msg = str(ei.value)
    assert f"predictions_ens_{net}_full_s{k}_v0.parquet" in msg and "other component files" in msg
    assert f"predictions_{net}_full_s{k}_v0.parquet" in msg and f"predictions_lgbm_full_s{k}_v0.parquet" in msg


@pytest.mark.parametrize("k", [0, 3, 4])
def test_lgbm_component_changed_after_ensemble_refused_even_with_fresh_delta(tmp_path, k):
    out, dl, lg = make_env(tmp_path)
    tamper(lg / f"predictions_lgbm_full_s{k}_v0.parquet")
    refresh_delta(lg)  # delta is consistent, so only the ensemble provenance check can catch it
    with pytest.raises(EvalError) as ei:
        V.build_freeze(out, dl, lg, APPROVAL)
    assert f"predictions_ens_mlp_full_s{k}_v0.parquet" in str(ei.value)
    assert f"predictions_lgbm_full_s{k}_v0.parquet" in str(ei.value)


@pytest.mark.parametrize(("raw", "text"), [(None, "no readable component_sha256"), ("{", "no readable"),
                                           ("[1]", "is not an object"), ('"x"', "is not an object")])
def test_component_sha256_absent_or_not_object_refused(tmp_path, raw, text):
    out, dl, lg = make_env(tmp_path)
    set_meta(dl / "predictions_ens_mlp_full_s2_v0.parquet", raw)
    with pytest.raises(EvalError, match=text) as ei:
        V.build_freeze(out, dl, lg, APPROVAL)
    assert "ens_mlp_full_s2" in str(ei.value)


def test_empty_approval_fails_before_file_checks(tmp_path):
    out, dl, lg = make_env(tmp_path)
    (dl / "predictions_ftt_noind_s2_v0.parquet").unlink()
    (lg / "delta.json").unlink()
    with pytest.raises(EvalError, match="approval text is empty"):
        V.build_freeze(out, dl, lg, "  \n")


@pytest.mark.parametrize(("fs", "k"), [("noind", 3), ("noind", 0), ("noind", 4), ("full", 4)])
def test_stale_sha_for_each_feature_set(tmp_path, fs, k):
    out, dl, lg = make_env(tmp_path)
    tamper(lg / f"predictions_lgbm_{fs}_s{k}_v0.parquet")
    with pytest.raises(EvalError, match="sha mismatch") as ei:
        V.build_freeze(out, dl, lg, APPROVAL)
    assert f"lgbm_{fs}_s{k}" in str(ei.value)


def test_delta_without_noind_entries_is_stale(tmp_path):
    out, dl, lg = make_env(tmp_path)
    doc = json.loads((lg / "delta.json").read_text())
    del doc["prediction_sha256"]["noind"]
    (lg / "delta.json").write_text(json.dumps(doc))
    with pytest.raises(EvalError, match="sha mismatch") as ei:
        V.build_freeze(out, dl, lg, APPROVAL)
    assert all(f"lgbm_noind_s{k}" in str(ei.value) for k in range(5))


@pytest.mark.parametrize("body", ['["a"]', "{", "{}", '{"prediction_sha256": ["a"]}', '{"prediction_sha256": "abc"}',
                                  '{"prediction_sha256": {"full": "x"}}', '{"prediction_sha256": null}'])
def test_malformed_delta_json_is_one_line_eval_error(tmp_path, body):
    out, dl, lg = make_env(tmp_path)
    (lg / "delta.json").write_text(body)
    with pytest.raises(EvalError) as ei:
        V.build_freeze(out, dl, lg, APPROVAL)
    assert "\n" not in str(ei.value) and "delta.json" in str(ei.value)


@pytest.fixture
def extreme_target_rows(monkeypatch):
    orig = F.write_labels

    def labels(root):
        ids = orig(root)
        df = pq.read_table(root / "labels.parquet").to_pandas()
        df.loc[df.fold == "test", "sdss_class"] = ["QSO", "GALAXY"] * (N_TEST // 2)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), root / "labels.parquet")
        return ids

    monkeypatch.setattr(F, "write_labels", labels)
    monkeypatch.setattr(F, "row_p", lambda i, ts: np.full((N_TEST, 3), float(ts)))


def test_thresholds_ignore_target_class_test_rows(tmp_path, extreme_target_rows):
    hi = V.build_freeze(*make_env(tmp_path / "hi", test_seed=1.0), APPROVAL)
    lo = V.build_freeze(*make_env(tmp_path / "lo", test_seed=0.0), APPROVAL)
    assert hi["thresholds"] == lo["thresholds"]
    for i, s in enumerate(V.planned_systems()):
        p = val_p(i, 1)
        for cls, col, a in (("QSO", 2, 20), ("GALAXY", 1, 10)):
            for c_star, k in (("0.80", 8), ("0.90", 9)):
                assert hi["thresholds"][s][cls][c_star] == F.kth(p[a : a + 10, col], k)
SHIFT = {"STAR": 0.5, "GALAXY": 0.3 + 1e-3, "QSO": 0.2 - 1e-3}


def frac(d):
    return {"STAR": 0.5, "GALAXY": 0.3 + d, "QSO": 0.2}


@pytest.mark.parametrize(("fracs", "ok"), [
    ({"ens_mlp_full_s4": frac(5e-10)}, True), ({"mlp_noind_s2": frac(5e-10)}, True),
    ({"ftt_noind_s1": frac(-5e-10)}, True), ({"ftt_noind_s1": frac(2e-9)}, False),
    ({"mlp_full_s0": frac(-2e-9)}, False), ({"ens_ftt_full_s3": frac(2e-9)}, False),
    ({"mlp_noind_s4": frac(2e-9)}, False), ({"lgbm_noind_s4": frac(2e-9)}, False),
    ({s: SHIFT for s in ("lgbm_noind_s0", "mlp_noind_s0", "ftt_noind_s0")}, True),  # noind compares to lgbm_noind
    ({s: SHIFT for s in ("lgbm_noind_s0", "mlp_noind_s0")}, False),
])
def test_fraction_tolerance_boundaries(tmp_path, fracs, ok):
    out, dl, lg = make_env(tmp_path, fracs=fracs)
    if ok:
        assert V.build_freeze(out, dl, lg, APPROVAL)["schema_version"] == 1
    else:
        with pytest.raises(EvalError, match="train_class_fractions"):
            V.build_freeze(out, dl, lg, APPROVAL)


def test_constants_exact(tmp_path):
    c = V.build_freeze(*make_env(tmp_path), APPROVAL)["constants"]
    assert c == {
        "headline_metric": "purity_at_completeness", "target_classes": ["QSO", "GALAXY"],
        "completeness_levels": [0.8, 0.9], "primary_completeness": 0.8, "primary_prior": "P2",
        "sensitivity_priors": ["P1", "P3"], "priors": {p: GRID[p] for p in ("P2", "P1", "P3")},
        "delta_sd_multiplier": 2.0, "delta_ddof": 1, "n_seeds": 5,
        "headline_pairs": [["mlp_full", "lgbm_full"], ["ftt_full", "lgbm_full"]],
        "descriptive_pairs": [["ens_mlp_full", "lgbm_full"], ["ens_ftt_full", "lgbm_full"]],
        "block_level": 3, "block_divisor": 16, "delta_fold": "val", "p3_bands": ["B", "C"],
        "prediction_version": "v0", "max_invalid_fraction": 0.01, "min_blocks": 20,
        "low_count_rule": "total", "low_count": 10, "n_boot": 1000, "rng_seed": 20261025,
    }


def test_has_test_key_any_level_but_not_values():
    assert V._has_test_key({"a": [{"b": {"x_TeSt": 1}}]}) and V._has_test_key({"a": [[{"Test": 0}]]})
    assert not V._has_test_key({"a": "test", "b": ["test", {"c": "latest"}], "d": 3})


def test_build_freeze_rejects_test_key_at_depth_and_inside_list(tmp_path, monkeypatch):
    out, dl, lg = make_env(tmp_path)
    assert V.build_freeze(out, dl, lg, "approved the latest test run")["approval"]  # a value, not a key
    monkeypatch.setattr(V, "_fractions", lambda p: {**F.FRAC, "x_test": 0.1})
    with pytest.raises(EvalError, match="'test'"):
        V.build_freeze(out, dl, lg, APPROVAL)
    monkeypatch.undo()
    monkeypatch.setattr(V, "HEADLINE_PAIRS", (("mlp_full", {"my_Test": 1}),))
    with pytest.raises(EvalError, match="'test'"):
        V.build_freeze(out, dl, lg, APPROVAL)


def test_two_identical_builds_write_identical_bytes(tmp_path):
    a = dl_session.run_freeze(*make_env(tmp_path / "a"), APPROVAL)
    time.sleep(1.1)  # a wall-clock timestamp in the document (even at second resolution) would now differ
    b = dl_session.run_freeze(*make_env(tmp_path / "b"), APPROVAL)
    assert a.read_bytes() == b.read_bytes()
    doc = V.build_freeze(*make_env(tmp_path / "c"), APPROVAL)
    dl_session.write_once(tmp_path / "x.json", doc)
    dl_session.write_once(tmp_path / "y.json", doc)
    assert (tmp_path / "x.json").read_bytes() == (tmp_path / "y.json").read_bytes() == a.read_bytes()
