"""Tests for build_ensemble in models/ensemble.py (ADR 0006 section 3)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astro_triage.dataset.build import CLASSES
from astro_triage.evaluation.dl_val import sha256_file
from astro_triage.models import ensemble as E
from astro_triage.models import lgbm
from astro_triage.models.lgbm import TrainError
from astro_triage.split import build_pixel_map

SHIFT = 2**49  # source_id // 2**35 // 4**7 == pixel
N_VAL, N_TEST = 30, 12
Y_VAL = np.random.default_rng(5).integers(0, 3, N_VAL)  # val labels written by ids_and_labels
FRAC = {"STAR": 0.5, "GALAXY": 0.3, "QSO": 0.2}


def dirichlet(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet([1.0, 1.0, 1.0], size=n)


def write_comp(path: Path, ids: np.ndarray, p: np.ndarray, frac: dict | None = None) -> None:
    cols = {"source_id": ids.astype(np.int64)}
    for i, c in enumerate(CLASSES):
        cols[f"p_cal_{c.lower()}"] = p[:, i]
    table = pa.Table.from_pandas(pd.DataFrame(cols).iloc[::-1], preserve_index=False)
    meta = {"train_class_fractions": json.dumps(frac or FRAC)}
    pq.write_table(table.replace_schema_metadata(meta), path)


def ids_and_labels(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pm = build_pixel_map()
    vp = pm[pm.fold == "val"].healpix_l5.to_numpy(dtype=np.int64)[:N_VAL]
    tp = pm[pm.fold == "test"].healpix_l5.to_numpy(dtype=np.int64)[:N_TEST]
    val_ids, test_ids = vp * SHIFT + np.arange(1, N_VAL + 1), tp * SHIFT + 7
    y = np.random.default_rng(5).integers(0, 3, N_VAL)
    labels = pd.DataFrame(
        {
            "source_id": np.concatenate([val_ids, test_ids]),
            "sdss_class": [CLASSES[i] for i in y] + ["STAR"] * N_TEST,
            "healpix_l5": np.concatenate([vp, tp]),
            "fold": ["val"] * N_VAL + ["test"] * N_TEST,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(labels, preserve_index=False), root / "labels.parquet")
    return val_ids, test_ids, y


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(lgbm, "REPO_ROOT", tmp_path)
    return tmp_path


def setup(env: Path, pl_val, pn_val, tpl=None, tpn=None, k=0, net="mlp", frac_net=None):
    """Write labels plus both component files; returns (outdir, dl_dir, lgbm_dir, ids)."""
    val_ids, test_ids, _ = ids_and_labels(env / "rel")
    tpl = dirichlet(11, N_TEST) if tpl is None else tpl
    tpn = dirichlet(12, N_TEST) if tpn is None else tpn
    ids = np.concatenate([val_ids, test_ids])
    dl, lg = env / "dl", env / "lg"
    dl.mkdir(exist_ok=True), lg.mkdir(exist_ok=True)
    write_comp(lg / f"predictions_lgbm_full_s{k}_v0.parquet", ids, np.vstack([pl_val, tpl]))
    write_comp(dl / f"predictions_{net}_full_s{k}_v0.parquet", ids, np.vstack([pn_val, tpn]), frac_net)
    return env / "rel", dl, lg, ids


def run(env, pl, pn, **kw):
    out, dl, lg, ids = setup(env, pl, pn, **kw)
    res = E.build_ensemble(out, dl, lg, kw.get("net", "mlp"), kw.get("k", 0))
    return res, dl, ids


def calibrated(pl, pn, w, t):
    """Independent softmax(log(clip(mix))/T); does not call the module under test."""
    m = w * pl + (1.0 - w) * pn
    z = np.log(np.clip(m / m.sum(axis=1, keepdims=True), 1e-12, None)) / t
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def val_loss(pl, pn, w, t):
    return -np.log(calibrated(pl, pn, w, t)[np.arange(N_VAL), Y_VAL]).mean()


def test_build_p_cal_matches_independent_recomputation(env):
    """Val and test rows: stored p_cal == softmax(log(mix)/T) from components, weight and temperature."""
    pl, pn, tpl, tpn = dirichlet(21, N_VAL), dirichlet(22, N_VAL), dirichlet(23, N_TEST), dirichlet(24, N_TEST)
    pn = 0.8 * pn + 0.2 * np.eye(3)[Y_VAL]  # informative net: the optimum weight is asymmetric (w != 1 - w)
    res, dl, ids = run(env, pl, pn, tpl=tpl, tpn=tpn)
    w, t = res["weight"], res["temperature"]
    df = pq.read_table(dl / "predictions_ens_mlp_full_s0_v0.parquet").to_pandas()
    order = np.argsort(ids)
    assert np.array_equal(df.source_id.to_numpy(), ids[order])
    expected = calibrated(np.vstack([pl, tpl])[order], np.vstack([pn, tpn])[order], w, t)
    assert np.allclose(df[list(E.P_COLUMNS)].to_numpy(), expected, rtol=0, atol=1e-6)
    # the reported weight and temperature are the val optima, so fixed/inverted/unit choices fail
    assert w == E.W_GRID[int(np.argmin([val_loss(pl, pn, g, 1.0) for g in E.W_GRID]))] and w != 0.5
    assert abs(t - 1.0) > 0.05
    assert val_loss(pl, pn, w, t) < min(val_loss(pl, pn, w, t * 0.95), val_loss(pl, pn, w, t * 1.05))


def test_build_metadata_sum_and_no_test_key(env):
    pl, pn = dirichlet(21, N_VAL), dirichlet(22, N_VAL)
    res, dl, ids = run(env, pl, pn, k=2, net="ftt")
    target = dl / "predictions_ens_ftt_full_s2_v0.parquet"
    t = pq.read_table(target)
    meta = {k.decode(): v.decode() for k, v in t.schema.metadata.items()}
    assert meta["family"] == "ensemble"
    assert float(meta["weight"]) == res["weight"] and float(meta["temperature"]) == res["temperature"]
    assert res["weight"] in E.W_GRID and E.TIE_ABS_TOL == 1e-12
    shas = json.loads(meta["component_sha256"])
    assert shas == res["component_sha256"]
    assert shas["predictions_ftt_full_s2_v0.parquet"] == sha256_file(
        dl / "predictions_ftt_full_s2_v0.parquet"
    )
    assert json.loads(meta["train_class_fractions"]) == FRAC
    df = t.to_pandas()
    assert np.array_equal(df.source_id.to_numpy(), np.sort(ids))
    assert np.allclose(df[list(E.P_COLUMNS)].sum(axis=1), 1.0, atol=1e-6)
    assert not any("test" in key.lower() for key in res)
    assert not any("test" in key.lower() for key in meta)


def test_test_rows_do_not_move_weight_or_temperature_but_change_output(env, tmp_path_factory):
    pl, pn = dirichlet(31, N_VAL), dirichlet(32, N_VAL)
    r1, dl1, ids = run(env, pl, pn, tpl=dirichlet(41, N_TEST), tpn=dirichlet(42, N_TEST))
    env2 = tmp_path_factory.mktemp("second")
    r2, dl2, _ = run(env2, pl, pn, tpl=dirichlet(43, N_TEST), tpn=dirichlet(44, N_TEST))
    assert r1["weight"] == r2["weight"] and r1["temperature"] == r2["temperature"]
    a = pq.read_table(dl1 / "predictions_ens_mlp_full_s0_v0.parquet").to_pandas()
    b = pq.read_table(dl2 / "predictions_ens_mlp_full_s0_v0.parquet").to_pandas()
    isval = a.source_id.isin(set(ids[:N_VAL].tolist())).to_numpy()
    cols = list(E.P_COLUMNS)
    assert np.array_equal(a[cols].to_numpy()[isval], b[cols].to_numpy()[isval])
    assert not np.allclose(a[cols].to_numpy()[~isval], b[cols].to_numpy()[~isval])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1])
@pytest.mark.parametrize("where", ["val", "test"])
def test_bad_p_cal_raises(env, bad, where):
    pl, pn = dirichlet(51, N_VAL), dirichlet(52, N_VAL)
    tpl = dirichlet(53, N_TEST)
    if where == "val":
        pl[3, 1] = bad
    else:
        tpl[2, 0] = bad
    out, dl, lg, _ = setup(env, pl, pn, tpl=tpl)
    with pytest.raises(TrainError, match="non-finite or negative"):
        E.build_ensemble(out, dl, lg, "mlp", 0)
    assert not (dl / "predictions_ens_mlp_full_s0_v0.parquet").exists()


def test_different_or_non_covering_ids_raise(env):
    pl, pn = dirichlet(61, N_VAL), dirichlet(62, N_VAL)
    out, dl, lg, ids = setup(env, pl, pn)
    p = dirichlet(63, len(ids))
    write_comp(dl / "predictions_mlp_full_s0_v0.parquet", ids + 1, p)  # same size, other ids
    with pytest.raises(TrainError, match="different source_id"):
        E.build_ensemble(out, dl, lg, "mlp", 0)
    # both drop the same val id: equal sets but val is not covered
    write_comp(lg / "predictions_lgbm_full_s0_v0.parquet", ids[1:], p[1:])
    write_comp(dl / "predictions_mlp_full_s0_v0.parquet", ids[1:], p[1:])
    pattern = rf"1 val rows missing from .*lgbm.*mlp.*{re.escape(str(int(ids[0])))}"
    with pytest.raises(TrainError, match=pattern):
        E.build_ensemble(out, dl, lg, "mlp", 0)


def test_rows_not_summing_to_one_raise(env):
    pl = dirichlet(54, N_VAL)
    pl[4] *= 1.5
    out, dl, lg, _ = setup(env, pl, dirichlet(55, N_VAL))
    with pytest.raises(TrainError, match="not summing to 1"):
        E.build_ensemble(out, dl, lg, "mlp", 0)


def test_duplicate_source_id_raises(env):
    out, dl, lg, ids = setup(env, dirichlet(56, N_VAL), dirichlet(57, N_VAL))
    write_comp(lg / "predictions_lgbm_full_s0_v0.parquet", np.r_[ids[1:], ids[1]], dirichlet(58, len(ids)))
    with pytest.raises(TrainError, match="duplicate source_id"):
        E.build_ensemble(out, dl, lg, "mlp", 0)


def test_missing_val_labels_is_train_error(env):
    out, dl, lg, _ = setup(env, dirichlet(59, N_VAL), dirichlet(60, N_VAL))
    (out / "labels.parquet").unlink()
    with pytest.raises(TrainError, match="cannot load val labels"):
        E.build_ensemble(out, dl, lg, "mlp", 0)


def test_fraction_mismatch_raises(env):
    out, dl, lg, _ = setup(
        env, dirichlet(71, N_VAL), dirichlet(72, N_VAL), frac_net={**FRAC, "STAR": 0.5 + 1e-6, "QSO": 0.2 - 1e-6}
    )
    with pytest.raises(TrainError, match="train_class_fractions"):
        E.build_ensemble(out, dl, lg, "mlp", 0)


def test_second_call_refuses_overwrite(env):
    out, dl, lg, _ = setup(env, dirichlet(81, N_VAL), dirichlet(82, N_VAL))
    E.build_ensemble(out, dl, lg, "mlp", 0)
    target = dl / "predictions_ens_mlp_full_s0_v0.parquet"
    before = sha256_file(target)
    with pytest.raises(TrainError, match="refusing to overwrite"):
        E.build_ensemble(out, dl, lg, "mlp", 0)
    assert sha256_file(target) == before


def test_invalid_net_or_seed_raises(env):
    out, dl, lg, _ = setup(env, dirichlet(91, N_VAL), dirichlet(92, N_VAL))
    with pytest.raises(TrainError):
        E.build_ensemble(out, dl, lg, "rf", 0)  # type: ignore[arg-type]
    with pytest.raises(TrainError):
        E.build_ensemble(out, dl, lg, "mlp", -1)
