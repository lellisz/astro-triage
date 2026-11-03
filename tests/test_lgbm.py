"""Tests for astro_triage.models.lgbm (ADR 0003, ADR 0004 D7.4, schema section 6).

All tests are offline and use small synthetic datasets written to ``tmp_path``. LightGBM is
shrunk (one thread, few small trees) through ``run_train``; the production constants are
never touched outside a ``MonkeyPatch.context``.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.special import log_softmax, softmax

from astro_triage.models import lgbm
from astro_triage.models.lgbm import TrainError

CLASSES = ("STAR", "GALAXY", "QSO")
FOLDS = ("train", "val", "test")
FEATURES = lgbm.FEATURE_COLUMNS
ASTROMETRIC_9 = (
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "pm",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "ruwe",
)
NOIND_EXPECTED = (
    "astrometric_excess_noise",
    "astrometric_excess_noise_sig",
    "ipd_gof_harmonic_amplitude",
    "ipd_frac_multi_peak",
    "ipd_frac_odd_win",
    "phot_g_mean_mag",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "phot_bp_rp_excess_factor",
)
FAST_PARAMS = {
    **lgbm.PARAMS,
    "num_threads": 1,
    "num_leaves": 7,
    "min_child_samples": 10,
    "learning_rate": 0.2,
}


# --------------------------------------------------------------------------- helpers


def _write(outdir: Path, labels: pd.DataFrame, feats: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    lab = pa.table(
        {
            "source_id": pa.array(labels["source_id"].to_numpy(), type=pa.int64()),
            "sdss_class": pa.array(labels["sdss_class"].tolist()),
            "fold": pa.array(labels["fold"].tolist()),
        }
    )
    cols: dict[str, pa.Array] = {
        "source_id": pa.array(feats["source_id"].to_numpy(), type=pa.int64())
    }
    for c in feats.columns:
        if c != "source_id":
            cols[c] = pa.array(
                feats[c].to_numpy(dtype=np.float32), type=pa.float32(), from_pandas=True
            )
    pq.write_table(lab, outdir / "labels.parquet")
    pq.write_table(pa.table(cols), outdir / "features.parquet")


def _read(outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pq.read_table(outdir / "labels.parquet").to_pandas()
    feats = pq.read_table(outdir / "features.parquet").to_pandas()
    return labels, feats


def make_frames(
    seed: int = 7,
    sizes: tuple[int, int, int] = (600, 300, 300),
    spacing: float = 0.5,
    noise: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic labels/features. Class k has per-column mean ``mu[col, k]``."""
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    ids = (rng.permutation(n).astype(np.int64) + 1) * 7_919_000_003 + 4 * 10**18
    fold = np.concatenate([[f] * s for f, s in zip(FOLDS, sizes, strict=True)])
    cls = np.concatenate([rng.permutation(np.arange(s) % 3) for s in sizes])
    mu = rng.normal(0.0, spacing, (len(FEATURES), 3))
    x = mu[:, cls].T + noise * rng.normal(size=(n, len(FEATURES)))
    solved = rng.choice([3, 31, 95], n, p=[0.3, 0.4, 0.3])
    x[:, FEATURES.index("astrometric_params_solved")] = solved
    for c in ASTROMETRIC_9:
        x[solved == 3, FEATURES.index(c)] = np.nan
    labels = pd.DataFrame(
        {"source_id": ids, "sdss_class": np.asarray(CLASSES)[cls], "fold": fold}
    )
    feats = pd.DataFrame(x.astype(np.float32), columns=list(FEATURES))
    feats.insert(0, "source_id", ids)
    # Feature rows in a different order than label rows.
    feats = feats.iloc[rng.permutation(n)].reset_index(drop=True)
    return labels, feats


def make_dataset(outdir: Path, **kwargs: object) -> Path:
    labels, feats = make_frames(**kwargs)  # type: ignore[arg-type]
    _write(outdir, labels, feats)
    return outdir


def run_train(outdir: Path) -> dict[str, object]:
    """``train_all`` with a tiny LightGBM; REPO_ROOT patched so tmp paths are relative."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(lgbm, "PARAMS", FAST_PARAMS)
        mp.setattr(lgbm, "MAX_TREES", 150)
        mp.setattr(lgbm, "PATIENCE", 10)
        mp.setattr(lgbm, "REPO_ROOT", outdir.parent)
        return lgbm.train_all(outdir)


def copy_inputs(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("labels.parquet", "features.parquet"):
        shutil.copy(src / name, dst / name)
    return dst


def pred_path(outdir: Path, model_id: str) -> Path:
    return outdir / f"predictions_{model_id}_v0.parquet"


def model_of(report: dict[str, object], model_id: str) -> dict[str, object]:
    models = report["models"]
    assert isinstance(models, list)
    return next(m for m in models if m["model_id"] == model_id)


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    outdir = make_dataset(tmp_path_factory.mktemp("base") / "ds")
    return outdir, run_train(outdir)


# ---------------------------------------------------------------- 1. alignment


def test_class_order_is_star_galaxy_qso() -> None:
    assert lgbm.CLASSES == CLASSES


def test_load_folds_aligns_by_source_id_not_by_position(tmp_path: Path) -> None:
    labels, feats = make_frames()
    # Encode the id into a feature: any positional join between the two files breaks it.
    feats["ruwe"] = (feats["source_id"] % 997).astype(np.float32)
    _write(tmp_path / "ds", labels, feats)
    assert not np.array_equal(
        labels["source_id"].to_numpy(), feats["source_id"].to_numpy()
    )
    folds = lgbm.load_folds(tmp_path / "ds")
    expected = dict(zip(labels["source_id"], labels["sdss_class"], strict=True))
    seen = 0
    for name, fold in folds.items():
        assert np.all(np.diff(fold.source_id) > 0)
        np.testing.assert_array_equal(
            fold.frame["ruwe"].to_numpy(), (fold.source_id % 997).astype(np.float32)
        )
        for sid, lab in zip(fold.source_id, fold.labels, strict=True):
            assert CLASSES[lab] == expected[int(sid)]
        assert (fold.frame["fold"] == name).all()
        seen += len(fold.labels)
    assert seen == len(labels)


@pytest.mark.parametrize(
    "case, pattern",
    [
        ("features_missing_row", "same source_id set"),
        ("features_duplicate_id", "features.source_id has duplicates"),
        ("labels_duplicate_id", "labels.source_id has duplicates"),
        ("features_foreign_id", "same source_id set"),
        ("features_missing_column", "lacks columns"),
        ("fold_missing_class", "lacks classes"),
        ("bad_class", "sdss_class outside"),
        ("bad_fold", "fold outside"),
        ("inf_feature", "inf"),
    ],
)
def test_misaligned_inputs_raise(tmp_path: Path, case: str, pattern: str) -> None:
    labels, feats = make_frames()
    if case == "features_missing_row":
        feats = feats.iloc[1:]
    elif case == "features_duplicate_id":
        feats.loc[0, "source_id"] = feats.loc[1, "source_id"]
    elif case == "labels_duplicate_id":
        labels.loc[0, "source_id"] = labels.loc[1, "source_id"]
    elif case == "features_foreign_id":
        feats.loc[0, "source_id"] = 5  # same length, one id replaced
    elif case == "features_missing_column":
        feats = feats.drop(columns=["bp_rp"])
    elif case == "fold_missing_class":
        labels.loc[
            (labels["fold"] == "val") & (labels["sdss_class"] == "QSO"), "sdss_class"
        ] = "STAR"
    elif case == "bad_class":
        labels.loc[0, "sdss_class"] = "UNKNOWN"
    elif case == "bad_fold":
        labels.loc[0, "fold"] = "holdout"
    elif case == "inf_feature":
        feats.loc[0, "bp_rp"] = np.inf
    _write(tmp_path / "ds", labels, feats)
    with pytest.raises(TrainError, match=pattern):
        lgbm.load_folds(tmp_path / "ds")
    with pytest.raises(TrainError):
        run_train(tmp_path / "ds")
    assert not list((tmp_path / "ds").glob("predictions_*"))


def test_load_folds_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(TrainError, match="build_dataset"):
        lgbm.load_folds(tmp_path)


def test_label_indices_follow_star_galaxy_qso(tmp_path: Path) -> None:
    labels, feats = make_frames()
    _write(tmp_path / "ds", labels, feats)
    folds = lgbm.load_folds(tmp_path / "ds")
    code = {"STAR": 0, "GALAXY": 1, "QSO": 2}
    for name, fold in folds.items():
        part = labels[labels["fold"] == name].sort_values("source_id")
        np.testing.assert_array_equal(
            fold.labels, part["sdss_class"].map(code).to_numpy()
        )


def test_predictions_table_maps_columns_by_name_hand_computed() -> None:
    # Internal order is (STAR, GALAXY, QSO). The parquet names are star, qso, galaxy.
    z = np.array([[2.0, 1.0, 0.0]])
    table, _ = lgbm.predictions_table(np.array([1], dtype=np.int64), z, 2.0)
    s_raw = math.exp(2) + math.exp(1) + 1.0
    s_cal = math.exp(1) + math.exp(0.5) + 1.0
    got = {
        n: float(table[n][0].as_py()) for n in table.column_names if n.startswith("p_")
    }
    assert got["p_raw_star"] == pytest.approx(math.exp(2) / s_raw, abs=1e-6)
    assert got["p_raw_galaxy"] == pytest.approx(math.exp(1) / s_raw, abs=1e-6)
    assert got["p_raw_qso"] == pytest.approx(1.0 / s_raw, abs=1e-6)
    assert got["p_cal_star"] == pytest.approx(math.exp(1) / s_cal, abs=1e-6)
    assert got["p_cal_galaxy"] == pytest.approx(math.exp(0.5) / s_cal, abs=1e-6)
    assert got["p_cal_qso"] == pytest.approx(1.0 / s_cal, abs=1e-6)
    assert got["p_raw_galaxy"] != pytest.approx(got["p_raw_qso"], abs=1e-3)


def test_end_to_end_argmax_by_name_recovers_true_class(tmp_path: Path) -> None:
    """Near-separable data: a positional (star, qso, galaxy) mapping swaps two classes."""
    outdir = make_dataset(tmp_path / "ds", spacing=4.0, noise=0.2)
    run_train(outdir)
    labels, _ = _read(outdir)
    for model_id in ("lgbm_full", "lgbm_noind"):
        pred = pq.read_table(pred_path(outdir, model_id)).to_pandas()
        merged = pred.merge(labels, on="source_id", validate="one_to_one")
        for prefix in ("p_raw", "p_cal"):
            by_name = merged[[f"{prefix}_star", f"{prefix}_galaxy", f"{prefix}_qso"]]
            guess = np.asarray(["STAR", "GALAXY", "QSO"])[
                np.argmax(by_name.to_numpy(), axis=1)
            ]
            acc = float((guess == merged["sdss_class"].to_numpy()).mean())
            assert acc > 0.95, (model_id, prefix, acc)


# ---------------------------------------------------------------- 2. temperature


def _calibrated_sample(t0: float, n: int = 20000, seed: int = 11):
    """Logits z = T0 * log p; labels ~ Categorical(p), so softmax(z / T0) = p is perfect."""
    rng = np.random.default_rng(seed)
    logp = log_softmax(rng.normal(0.0, 1.5, (n, 3)), axis=1)
    cum = np.cumsum(np.exp(logp), axis=1)
    y = (rng.random(n)[:, None] > cum).sum(axis=1).clip(0, 2).astype(np.int64)
    return logp * t0, y


@pytest.mark.parametrize("t0", [0.5, 1.0, 2.0, 3.0])
def test_fit_temperature_recovers_known_T(t0: float) -> None:
    # The expected logloss is minimised at T0 (proper scoring rule); with n = 20000 the
    # sampling error of the minimiser is about 1-2 percent, so 6 percent is a loose-but-
    # meaningful band (the wrong answers 1.0 vs 2.0 or 0.5 differ by 100 percent).
    z, y = _calibrated_sample(t0)
    t = lgbm.fit_temperature(z, y)
    assert t == pytest.approx(t0, rel=0.06)
    lo, hi = lgbm.TEMPERATURE_BOUNDS
    assert lo * 2 < t < hi / 2
    assert lgbm.log_loss(z, y, t) <= lgbm.log_loss(z, y, 1.0) + 1e-12


def test_fit_temperature_hits_upper_bound_on_uninformative_logits() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 5.0, (3000, 3))
    y = rng.integers(0, 3, 3000).astype(np.int64)
    t = lgbm.fit_temperature(z, y)
    assert t >= lgbm.TEMPERATURE_BOUNDS[1] * 0.999  # the bound check can be triggered


def test_log_loss_hand_values() -> None:
    z = np.array([[0.0, 0.0, 0.0]])
    y = np.array([0])
    for t in (0.3, 1.0, 7.0):
        assert lgbm.log_loss(z, y, t) == pytest.approx(math.log(3.0), abs=1e-12)
    z2 = np.array([[2.0, 0.0, 0.0]])
    assert lgbm.log_loss(z2, y) == pytest.approx(
        -math.log(math.exp(2) / (math.exp(2) + 2))
    )
    assert lgbm.log_loss(z2, y, 2.0) == pytest.approx(-math.log(math.e / (math.e + 2)))


def test_temperature_one_leaves_probabilities_unchanged() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 2.0, (200, 3))
    ids = np.arange(200, dtype=np.int64)
    table, cols = lgbm.predictions_table(ids, z, 1.0)
    for c in ("star", "galaxy", "qso"):
        np.testing.assert_array_equal(cols[f"p_raw_{c}"], cols[f"p_cal_{c}"])
    y = rng.integers(0, 3, 200)
    assert lgbm.log_loss(z, y) == lgbm.log_loss(z, y, 1.0)
    assert table.num_rows == 200


def test_argmax_is_invariant_to_temperature() -> None:
    rng = np.random.default_rng(5)
    z = rng.normal(0.0, 3.0, (500, 3))
    ref = np.argmax(z, axis=1)
    for t in (0.05, 0.5, 1.0, 3.0, 20.0):
        np.testing.assert_array_equal(np.argmax(softmax(z / t, axis=1), axis=1), ref)
    ids = np.arange(500, dtype=np.int64)
    _, cols = lgbm.predictions_table(ids, z, 2.5)
    raw = np.stack([cols[f"p_raw_{c}"] for c in ("star", "galaxy", "qso")], axis=1)
    cal = np.stack([cols[f"p_cal_{c}"] for c in ("star", "galaxy", "qso")], axis=1)
    np.testing.assert_array_equal(np.argmax(raw, axis=1), np.argmax(cal, axis=1))
    np.testing.assert_array_equal(np.argmax(raw, axis=1), ref)


def test_pipeline_temperature_is_inside_bounds(
    base: tuple[Path, dict[str, object]],
) -> None:
    _, report = base
    lo, hi = lgbm.TEMPERATURE_BOUNDS
    for m in report["models"]:  # type: ignore[union-attr]
        assert lo * 1.001 < m["temperature"] < hi * 0.999
        assert m["temperature_at_bound"] is False


# ---------------------------------------------------------------- 3. no test usage


def _shuffle_fold_labels(src: Path, dst: Path, fold: str, seed: int) -> None:
    labels, feats = _read(src)
    mask = (labels["fold"] == fold).to_numpy()
    original = labels.loc[mask, "sdss_class"].to_numpy().copy()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(original)
    assert (shuffled != original).mean() > 0.4  # the shuffle really changes the labels
    labels.loc[mask, "sdss_class"] = shuffled
    _write(dst, labels, feats)


def test_test_labels_do_not_influence_booster_temperature_or_predictions(
    base: tuple[Path, dict[str, object]], tmp_path: Path
) -> None:
    src, report = base
    dst = tmp_path / "shuffled"
    _shuffle_fold_labels(src, dst, "test", seed=99)
    shuf = run_train(dst)
    for model_id in ("lgbm_full", "lgbm_noind"):
        a, b = model_of(report, model_id), model_of(shuf, model_id)
        assert a["best_iteration"] == b["best_iteration"]
        assert a["temperature"] == b["temperature"]
        assert a["train_class_fractions"] == b["train_class_fractions"]
        assert a["gain_top10"] == b["gain_top10"]
        assert a["val_calibration_fitted_here"] == b["val_calibration_fitted_here"]
        ma = (src / "models" / f"{model_id}.txt").read_bytes()
        mb = (dst / "models" / f"{model_id}.txt").read_bytes()
        assert ma == mb
        ta = pq.read_table(pred_path(src, model_id))
        tb = pq.read_table(pred_path(dst, model_id))
        assert ta.equals(tb)  # val AND test probabilities: they depend on test X only
        # Only the test metrics may move, and they must (otherwise the control is vacuous).
        assert a["test_not_fitted"] != b["test_not_fitted"]
        acc_a = float(a["test_not_fitted"]["accuracy"])  # type: ignore[index]
        acc_b = float(b["test_not_fitted"]["accuracy"])  # type: ignore[index]
        assert acc_a > acc_b + 0.15


def test_control_val_labels_do_influence_fit(
    base: tuple[Path, dict[str, object]], tmp_path: Path
) -> None:
    """Sensitivity control: the comparison above can fail when a fitted fold changes."""
    src, report = base
    dst = tmp_path / "val_shuffled"
    _shuffle_fold_labels(src, dst, "val", seed=99)
    shuf = run_train(dst)
    a, b = model_of(report, "lgbm_full"), model_of(shuf, "lgbm_full")
    assert (
        a["temperature"] != b["temperature"]
        or a["best_iteration"] != b["best_iteration"]
    )


# ---------------------------------------------------------------- 4. NULL handling


def test_nulls_pass_through_and_predictions_are_valid(tmp_path: Path) -> None:
    labels, feats = make_frames(seed=21)
    feats["phot_bp_mean_mag"] = np.nan  # an entirely NULL column in every fold
    test_ids = set(labels.loc[labels["fold"] == "test", "source_id"].iloc[:5])
    feats.loc[feats["source_id"].isin(test_ids), list(FEATURES)] = (
        np.nan
    )  # all-NULL rows
    outdir = tmp_path / "ds"
    _write(outdir, labels, feats)
    assert pq.read_table(outdir / "features.parquet")[
        "phot_bp_mean_mag"
    ].null_count == len(feats)
    written_nulls = int(
        sum(pq.read_table(outdir / "features.parquet")[c].null_count for c in FEATURES)
    )
    assert written_nulls > 0

    folds = lgbm.load_folds(outdir)
    loaded_nan = sum(
        int(f.frame[list(FEATURES)].isna().sum().sum()) for f in folds.values()
    )
    assert loaded_nan == written_nulls  # no imputation anywhere
    assert np.isnan(folds["test"].matrix(FEATURES)).any()

    report = run_train(outdir)
    n_eval = len(folds["val"].labels) + len(folds["test"].labels)
    for model_id in ("lgbm_full", "lgbm_noind"):
        table = pq.read_table(pred_path(outdir, model_id))
        assert table.num_rows == n_eval
        assert sum(table[c].null_count for c in table.column_names) == 0
        for prefix in ("p_raw", "p_cal"):
            p = np.stack(
                [
                    table[f"{prefix}_{c}"].to_numpy().astype(np.float64)
                    for c in ("star", "qso", "galaxy")
                ],
                axis=1,
            )
            assert np.isfinite(p).all()
            assert p.min() >= 0.0 and p.max() <= 1.0
            assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-6
        assert math.isfinite(float(model_of(report, model_id)["temperature"]))  # type: ignore[arg-type]


def test_nan_probabilities_are_refused() -> None:
    z = np.array([[0.0, np.nan, 1.0]])
    with pytest.raises(TrainError):
        lgbm.predictions_table(np.array([1], dtype=np.int64), z, 1.0)


# ---------------------------------------------------------------- 5. ablation


def test_feature_sets_match_adr_0004_d7_4() -> None:
    assert len(lgbm.MODEL_FULL.features) == 21
    assert lgbm.MODEL_FULL.features == FEATURES
    assert lgbm.MODEL_NOIND.features == NOIND_EXPECTED
    assert len(lgbm.MODEL_NOIND.features) == 11
    removed = set(FEATURES) - set(lgbm.MODEL_NOIND.features)
    assert removed == {"astrometric_params_solved", *ASTROMETRIC_9}
    assert lgbm.MODEL_FULL.model_id == "lgbm_full"
    assert lgbm.MODEL_NOIND.model_id == "lgbm_noind"


def test_trained_boosters_use_exactly_those_features(
    base: tuple[Path, dict[str, object]],
) -> None:
    outdir, report = base
    full = lgb.Booster(model_file=str(outdir / "models" / "lgbm_full.txt"))
    noind = lgb.Booster(model_file=str(outdir / "models" / "lgbm_noind.txt"))
    assert tuple(full.feature_name()) == FEATURES
    assert tuple(noind.feature_name()) == NOIND_EXPECTED
    assert model_of(report, "lgbm_full")["n_features"] == 21
    assert model_of(report, "lgbm_noind")["n_features"] == 11


@pytest.mark.parametrize("column", ["ruwe", "astrometric_params_solved", "parallax"])
def test_changing_a_removed_column_leaves_noind_unchanged(
    base: tuple[Path, dict[str, object]], tmp_path: Path, column: str
) -> None:
    src, _ = base
    labels, feats = _read(src)
    if column == "astrometric_params_solved":
        feats[column] = np.roll(feats[column].to_numpy(), 17)
    else:
        feats[column] = feats[column] + np.float32(3.0)
    dst = tmp_path / "perturbed"
    _write(dst, labels, feats)
    run_train(dst)
    a = pq.read_table(pred_path(src, "lgbm_noind"))
    b = pq.read_table(pred_path(dst, "lgbm_noind"))
    assert a.equals(b)
    assert (src / "models" / "lgbm_noind.txt").read_bytes() == (
        dst / "models" / "lgbm_noind.txt"
    ).read_bytes()


def test_changing_removed_columns_changes_full(
    base: tuple[Path, dict[str, object]], tmp_path: Path
) -> None:
    src, _ = base
    labels, feats = _read(src)
    for c in ASTROMETRIC_9:
        feats[c] = feats[c] + np.float32(3.0)
    feats["astrometric_params_solved"] = np.roll(
        feats["astrometric_params_solved"].to_numpy(), 17
    )
    dst = tmp_path / "perturbed"
    _write(dst, labels, feats)
    run_train(dst)
    full_a = pq.read_table(pred_path(src, "lgbm_full"))
    full_b = pq.read_table(pred_path(dst, "lgbm_full"))
    assert not full_a.equals(full_b)
    assert pq.read_table(pred_path(src, "lgbm_noind")).equals(
        pq.read_table(pred_path(dst, "lgbm_noind"))
    )


# ---------------------------------------------------------------- 6. determinism


def test_same_seed_gives_identical_predictions_and_models(
    base: tuple[Path, dict[str, object]], tmp_path: Path
) -> None:
    src, report = base
    dst = copy_inputs(src, tmp_path / "rerun")
    again = run_train(dst)
    for model_id in ("lgbm_full", "lgbm_noind"):
        ta, tb = (
            pq.read_table(pred_path(src, model_id)),
            pq.read_table(pred_path(dst, model_id)),
        )
        assert ta.equals(tb)
        for name in ta.column_names:
            if name.startswith("p_"):
                np.testing.assert_allclose(
                    ta[name].to_numpy(), tb[name].to_numpy(), rtol=0, atol=1e-12
                )
        a, b = model_of(report, model_id), model_of(again, model_id)
        assert a["temperature"] == b["temperature"]
        assert a["best_iteration"] == b["best_iteration"]
        assert (src / "models" / f"{model_id}.txt").read_bytes() == (
            dst / "models" / f"{model_id}.txt"
        ).read_bytes()


# ---------------------------------------------------------------- 7. writing


def test_predictions_contain_only_val_and_test_one_row_per_source_sorted(
    base: tuple[Path, dict[str, object]],
) -> None:
    outdir, report = base
    labels, _ = _read(outdir)
    train_ids = set(labels.loc[labels["fold"] == "train", "source_id"])
    eval_ids = np.sort(labels.loc[labels["fold"] != "train", "source_id"].to_numpy())
    for model_id in ("lgbm_full", "lgbm_noind"):
        ids = pq.read_table(pred_path(outdir, model_id))["source_id"].to_numpy()
        assert not (set(ids.tolist()) & train_ids)
        assert len(ids) == len(set(ids.tolist())) == len(eval_ids)
        assert np.all(
            np.diff(ids) > 0
        )  # strictly increasing: sorted and unique (schema G4)
        np.testing.assert_array_equal(ids, eval_ids)
        assert model_of(report, model_id)["prediction_rows"] == len(eval_ids)
    # The id order in the files interleaves folds, so sorting was actually exercised.
    assert not np.all(np.diff(labels["source_id"].to_numpy()) > 0)


def test_predictions_schema_and_metadata_follow_schema_section_6(
    base: tuple[Path, dict[str, object]],
) -> None:
    outdir, _ = base
    labels, _ = _read(outdir)
    train = labels[labels["fold"] == "train"]["sdss_class"].value_counts()
    fractions = {
        c: int(train.get(c, 0)) / len(labels[labels["fold"] == "train"])
        for c in CLASSES
    }
    for model_id in ("lgbm_full", "lgbm_noind"):
        path = pred_path(outdir, model_id)
        schema = pq.read_schema(path)
        assert schema.names == [
            "source_id", "model_version",
            "p_raw_star", "p_raw_qso", "p_raw_galaxy",
            "p_cal_star", "p_cal_qso", "p_cal_galaxy",
        ]  # fmt: skip
        assert schema.field("source_id").type == pa.int64()
        assert schema.field("model_version").type == pa.dictionary(
            pa.int8(), pa.string()
        )
        for name in schema.names:
            nullable = name.startswith("p_cal_")
            assert schema.field(name).nullable is nullable, name
            if name.startswith("p_"):
                assert schema.field(name).type == pa.float32()
        meta = {k.decode(): v.decode() for k, v in schema.metadata.items()}
        assert meta["model_id"] == model_id
        assert meta["model_version"] == "v0"
        assert meta["calibration_fit_fold"] == "val"
        assert meta["calibration"] != "none"
        for key in (
            "features_file",
            "labels_file",
            "split_file",
            "train_class_fractions",
        ):
            assert key in meta
        got = json.loads(meta["train_class_fractions"])
        assert got == pytest.approx(
            fractions, abs=1e-12
        )  # pi_src from train labels only
        assert sum(got.values()) == pytest.approx(1.0, abs=1e-12)
        table = pq.read_table(path)
        assert set(table["model_version"].to_pylist()) == {"v0"}
        assert table["p_cal_star"].null_count == 0  # calibrated: never NULL
        md = pq.ParquetFile(path).metadata
        for rg in range(md.num_row_groups):
            for ci in range(md.num_columns):
                assert md.row_group(rg).column(ci).compression == "ZSTD"


def test_no_tmp_files_left_after_success(base: tuple[Path, dict[str, object]]) -> None:
    outdir, _ = base
    assert not list(outdir.rglob("*.tmp"))
    assert (outdir / "train_report.json").is_file()


def test_atomic_helper_keeps_previous_file_when_writer_fails(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old")

    def boom(tmp: str) -> None:
        Path(tmp).write_text("partial")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        lgbm._atomic(target, boom)
    assert target.read_text() == "old"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_failed_replace_leaves_no_tmp_and_no_corruption(
    base: tuple[Path, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_on_call: int,
) -> None:
    """Call 1 is the predictions file, call 2 is the booster file (run_model order)."""
    src, _ = base
    outdir = copy_inputs(src, tmp_path / "ds")
    (outdir / "models").mkdir()
    pred_old = pred_path(outdir, "lgbm_full")
    model_old = outdir / "models" / "lgbm_full.txt"
    pred_old.write_bytes(b"PREVIOUS-PREDICTIONS")
    model_old.write_bytes(b"PREVIOUS-MODEL")
    real_replace = lgbm.os.replace
    calls = {"n": 0}

    def flaky(a: str, b: object) -> None:
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            raise OSError("simulated replace failure")
        real_replace(a, b)  # type: ignore[arg-type]

    monkeypatch.setattr(lgbm.os, "replace", flaky)
    with pytest.raises(OSError, match="simulated"):
        run_train(outdir)
    monkeypatch.undo()
    assert not list(outdir.rglob("*.tmp"))
    if fail_on_call == 1:
        assert pred_old.read_bytes() == b"PREVIOUS-PREDICTIONS"
        assert model_old.read_bytes() == b"PREVIOUS-MODEL"
    else:
        assert pq.read_table(pred_old).num_rows > 0  # new file replaced whole, readable
        assert model_old.read_bytes() == b"PREVIOUS-MODEL"
    assert not (outdir / "train_report.json").exists()
