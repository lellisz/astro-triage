"""load_dl_systems (spec S11): 40 aligned systems, id coverage, p_cal sanity, pi_src vs lgbm reference."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from dl_eval_kit import edit_freeze, frozen_env
from test_dl_val_freeze import row_p, val_p, write_pred

from astro_triage.evaluation import dl_eval as E
from astro_triage.evaluation import dl_val as V
from astro_triage.evaluation.runner import SUM_TOLERANCE, EvalError


def norm(p):
    return p / p.sum(axis=1, keepdims=True)


def setup(tmp_path):
    """make_env writes unnormalised p_cal: rewrite all 40 files with rows summing to 1, refresh hashes."""
    out, dl, lg, _doc = frozen_env(tmp_path)
    lab = pq.read_table(out / "labels.parquet").to_pandas()
    val_ids = lab[lab.fold == "val"].source_id.to_numpy(dtype=np.int64)
    test_ids = lab[lab.fold == "test"].source_id.to_numpy(dtype=np.int64)
    ids = np.concatenate([val_ids, test_ids])
    for i, name in enumerate(V.planned_systems()):
        p = norm(np.vstack([val_p(i, 1), row_p(i, 2)]))
        rewrite_pred(dl, lg, name, ids, p)
    return out, dl, lg, SimpleNamespace(source_id=test_ids), val_ids


def rewrite_pred(dl, lg, name, ids, p, frac=None):
    """Replace one prediction file and refresh its freeze hash (keeps verify_freeze green)."""
    path = E._paths(dl, lg)[name]
    kw = {} if frac is None else {"frac": frac}
    write_pred(path, ids, p, **kw)
    edit_freeze(
        dl, lambda d: d["prediction_sha256"].__setitem__(name, V.sha256_file(path))
    )


def test_happy_path_40_systems_aligned_on_test_ids(tmp_path):
    out, dl, lg, ts, _ = setup(tmp_path)
    systems = E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)
    assert list(systems) == V.planned_systems() and len(systems) == 40
    for i, (name, md) in enumerate(systems.items()):
        assert md.name == name and md.proba.shape == (len(ts.source_id), 3)
        assert md.covered.all() and md.pi_src == {
            "STAR": 0.5,
            "GALAXY": 0.3,
            "QSO": 0.2,
        }
        assert np.array_equal(
            md.proba, norm(row_p(i, 2))
        )  # ts order == labels test-row order


def test_alignment_follows_ts_order_not_file_order(tmp_path):
    out, dl, lg, ts, _ = setup(tmp_path)
    rev = SimpleNamespace(source_id=ts.source_id[::-1].copy())
    a = E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)["mlp_full_s0"].proba
    b = E.load_dl_systems(out, rev, dl_dir=dl, lgbm_dir=lg)["mlp_full_s0"].proba
    assert np.array_equal(a[::-1], b)


def test_freeze_verified_before_any_prediction_read(tmp_path, monkeypatch):
    out, dl, lg, ts, _ = setup(tmp_path)
    (dl / E.FREEZE_NAME).unlink()
    monkeypatch.setattr(
        E, "_read_proba", lambda *a: pytest.fail("prediction read before freeze check")
    )
    with pytest.raises(EvalError, match="val_freeze.json missing"):
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)


def test_tampered_prediction_refused_before_any_read(tmp_path):
    out, dl, lg, ts, _ = setup(tmp_path)
    path = E._paths(dl, lg)["mlp_full_s0"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(EvalError, match="sha256 differs from the freeze"):
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)


def test_val_and_test_ids_overlap_refused(tmp_path):
    out, dl, lg, ts, val_ids = setup(tmp_path)
    clash = SimpleNamespace(source_id=np.concatenate([ts.source_id[:-1], val_ids[:1]]))
    with pytest.raises(EvalError, match="val and test ids overlap"):
        E.load_dl_systems(out, clash, dl_dir=dl, lgbm_dir=lg)


@pytest.mark.parametrize("mode", ["missing_row", "extra_row", "duplicate_row"])
def test_coverage_must_equal_val_union_test(tmp_path, mode):
    out, dl, lg, ts, val_ids = setup(tmp_path)
    ids = np.concatenate([val_ids, ts.source_id])
    p = np.random.default_rng(0).dirichlet([1, 1, 1], size=ids.size)
    if mode == "missing_row":
        ids, p = ids[1:], p[1:]
    elif mode == "extra_row":
        ids, p = np.append(ids, ids.max() + 1), np.vstack([p, p[:1]])
    else:
        ids = np.append(ids[:-1], ids[0])
    rewrite_pred(dl, lg, "ens_mlp_full_s3", ids, p)
    with pytest.raises(
        EvalError, match="ens_mlp_full_s3.*exactly the val \\+ test ids"
    ):
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_p_cal_refused(tmp_path, bad):
    out, dl, lg, ts, val_ids = setup(tmp_path)
    ids = np.concatenate([val_ids, ts.source_id])
    p = np.full((ids.size, 3), 1 / 3)
    p[-1, 1] = bad  # last row is a test row (original sums are then not 1 either)
    rewrite_pred(dl, lg, "ftt_noind_s1", ids, p)
    with pytest.raises(
        EvalError, match="ftt_noind_s1.*not finite or does not sum to 1"
    ):
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)


def test_nonfinite_in_val_rows_is_ignored(tmp_path):
    out, dl, lg, ts, val_ids = setup(tmp_path)
    ids = np.concatenate([val_ids, ts.source_id])
    p = np.full((ids.size, 3), 1 / 3)
    p[0, 0] = np.nan  # val row: only test rows are aligned and checked
    rewrite_pred(dl, lg, "ftt_noind_s1", ids, p)
    assert np.isfinite(
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)["ftt_noind_s1"].proba
    ).all()


def _sum_case(tmp_path, delta):
    out, dl, lg, ts, val_ids = setup(tmp_path)
    ids = np.concatenate([val_ids, ts.source_id])
    p = np.full((ids.size, 3), 1 / 3)
    p[-1, 0] += delta
    rewrite_pred(dl, lg, "mlp_full_s4", ids, p)
    return E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)


def test_sum_outside_tolerance_refused(tmp_path):
    with pytest.raises(EvalError, match="mlp_full_s4.*does not sum to 1"):
        _sum_case(tmp_path, 2 * SUM_TOLERANCE)


def test_sum_inside_tolerance_accepted(tmp_path):
    assert "mlp_full_s4" in _sum_case(tmp_path, SUM_TOLERANCE / 2)


def test_negative_deviation_also_refused(tmp_path):
    with pytest.raises(EvalError, match="does not sum to 1"):
        _sum_case(tmp_path, -2 * SUM_TOLERANCE)


@pytest.mark.parametrize("name", ["mlp_full_s0", "ftt_noind_s4", "ens_ftt_full_s2"])
def test_fractions_beyond_tolerance_refused(tmp_path, name):
    out, dl, lg, ts, _ = setup(tmp_path)
    edit_freeze(dl, lambda d: d["train_class_fractions"][name].update(STAR=0.5 + 1e-8))
    ref = f"lgbm_{'noind' if '_noind_' in name else 'full'}_s{name[-1]}"
    with pytest.raises(
        EvalError, match=f"train_class_fractions of {name} differ from {ref}"
    ):
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)


def test_lgbm_reference_mutated_breaks_its_dl_dependents(tmp_path):
    out, dl, lg, ts, _ = setup(tmp_path)
    edit_freeze(
        dl, lambda d: d["train_class_fractions"]["lgbm_full_s2"].update(QSO=0.25)
    )
    with pytest.raises(
        EvalError, match="train_class_fractions of .*s2 differ from lgbm_full_s2"
    ):
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)


def test_fraction_exactly_at_limit_accepted_and_just_above_refused(
    tmp_path, monkeypatch
):
    out, dl, lg, ts, _ = setup(tmp_path)
    monkeypatch.setattr(
        E, "FRACTION_TOL", 2.0**-30
    )  # exactly representable differences
    edit_freeze(
        dl,
        lambda d: d["train_class_fractions"]["mlp_full_s0"].update(STAR=0.5 + 2.0**-30),
    )
    assert (
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)["mlp_full_s0"].pi_src["STAR"]
        == 0.5 + 2.0**-30
    )
    edit_freeze(
        dl,
        lambda d: d["train_class_fractions"]["mlp_full_s0"].update(STAR=0.5 + 2.0**-29),
    )
    with pytest.raises(EvalError, match="train_class_fractions of mlp_full_s0"):
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)


@pytest.mark.parametrize("edit", ["no_system", "no_class", "not_number", "no_ref"])
def test_malformed_freeze_fractions_refused(tmp_path, edit):
    out, dl, lg, ts, _ = setup(tmp_path)
    f = "train_class_fractions"
    fn = {
        "no_system": lambda d: d[f].pop("mlp_full_s0"),
        "no_class": lambda d: d[f]["mlp_full_s0"].pop("QSO"),
        "not_number": lambda d: d[f]["mlp_full_s0"].update(QSO="abc"),
        "no_ref": lambda d: d[f].pop("lgbm_full_s0"),
    }[edit]
    edit_freeze(dl, fn)
    with pytest.raises(EvalError, match="no train_class_fractions|differ from"):
        E.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)
