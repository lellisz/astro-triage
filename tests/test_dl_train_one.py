"""dl_train part 2 (spec S6): `train_one` end to end on a tiny synthetic release, CPU, 2 epochs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from astro_triage.dataset.build import CLASSES, FEATURE_COLUMNS
from astro_triage.models import dl_train, lgbm
from astro_triage.models.dl_data import subsample_index
from astro_triage.models.dl_nets import MLP_CONFIG
from astro_triage.models.lgbm import (
    INDICATOR_COLUMN,
    Fold,
    TrainError,
    predictions_table,
)

SPEC = dl_train.DLSpec("mlp", "full", 0)
SIZES = {"train": 150, "val": 60, "test": 60}
KEYS = (  # noqa: SIM905
    "schema_name schema_version created_utc git_sha inputs model_id model_version features_file "
    "labels_file split_file train_class_fractions calibration calibration_fit_fold family arch "
    "feature_set arch_config seed best_epoch epochs_run temperature preprocessor_sha256 "
    "torch_version device train_subset"
).split()
PRED = "predictions_mlp_full_s0_v0.parquet"


def make_folds(labelled_test: bool = True) -> dict[str, Fold]:
    rng = np.random.default_rng(0)
    folds: dict[str, Fold] = {}
    for k, (name, n) in enumerate(SIZES.items()):
        y = (np.arange(n) % 3).astype(np.int64)
        x = rng.normal(size=(n, len(FEATURE_COLUMNS)))
        x[:, 3:6] += y[:, None] * 1.5
        frame = pd.DataFrame(x, columns=list(FEATURE_COLUMNS))
        frame[INDICATOR_COLUMN] = rng.choice([31.0, 95.0], n)
        frame.loc[rng.random(n) < 0.1, "parallax"] = np.nan
        if name == "test" and not labelled_test:
            y = np.full(n, -1, dtype=np.int64)  # no labels, no sdss_class
        else:
            frame["sdss_class"] = np.asarray(CLASSES)[y]
        folds[name] = Fold(np.arange(n) * 7 + 10_000 * (k + 1), y, frame)
    return folds


def run(root: Path, folds: dict[str, Fold], spec: dl_train.DLSpec = SPEC, **kw):
    kw = {"max_epochs": 2, "patience": 5, **kw}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(lgbm, "REPO_ROOT", root)
        return dl_train.train_one(
            spec, folds, root / "out", root / "ckpt", device="cpu", **kw
        )


def read(root: Path, name: str = PRED) -> pq.ParquetFile:
    return pq.ParquetFile(root / "out" / name)


def fractions(y: np.ndarray) -> dict[str, float]:
    counts = np.bincount(y, minlength=3)
    return dict(zip(CLASSES, (counts / len(y)).tolist(), strict=True))


def val_side(out: dict) -> tuple:
    return out["val_calibration_fitted_here"], out["temperature"], out["best_epoch"]


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("base")
    return root, make_folds(), run(root, make_folds())


def test_parquet_schema_ids_and_probabilities(base) -> None:
    root, folds, _ = base
    table = read(root).read()
    ids = np.concatenate([folds["val"].source_id, folds["test"].source_id])
    want, _ = predictions_table(ids, np.zeros((len(ids), 3)), 1.0)
    assert table.schema.remove_metadata().equals(want.schema.remove_metadata())
    assert table["source_id"].to_numpy().tolist() == sorted(ids.tolist())
    p = np.column_stack([table[f"p_cal_{c.lower()}"].to_numpy() for c in CLASSES])
    assert np.isfinite(p).all() and np.abs(p.sum(axis=1) - 1).max() < 1e-6


def test_metadata_has_adr_keys_as_strings(base) -> None:
    root, _, report = base
    meta = {k.decode(): v.decode() for k, v in read(root).schema_arrow.metadata.items()}
    assert set(KEYS) <= set(meta)
    assert meta["family"] == "deep_learning" and meta["train_subset"] == "none"
    assert meta["model_id"] == "mlp_full_s0" and meta["seed"] == str(SPEC.seed)
    assert json.loads(meta["arch_config"]) == MLP_CONFIG and meta["device"] == "cpu"
    assert float(meta["temperature"]) == report["temperature"]
    assert json.loads(meta["train_class_fractions"]) == report["train_class_fractions"]
    assert int(meta["best_epoch"]) == report["best_epoch"] != 0


def test_report_has_no_test_key_and_no_array(base) -> None:
    _, folds, report = base
    assert "test" not in json.dumps(report).lower()  # also proves plain JSON types
    costs = report["costs"]
    assert "checkpoint_bytes" in costs and "peak_rss_bytes" in costs
    assert costs["predict_seconds_per_1m_rows"] == pytest.approx(
        costs["predict_seconds"] / 120 * 1e6
    )
    assert costs["peak_gpu_bytes"] == "not measured" and costs["device_name"] == "cpu"
    assert costs["epochs_run"] == 2 and costs["parameter_count"] > 0
    assert report["train_class_fractions"] == fractions(folds["train"].labels)
    root = base[0]
    assert report["prediction_file"] == f"out/{PRED}"
    assert report["model_file"] == "out/models/mlp_full_s0.pt"
    assert (root / "ckpt" / "mlp_full_s0" / "last.pt").is_file()
    assert (root / "out" / "models" / "mlp_full_s0.preproc.pkl").is_file()
    assert len(list(root.rglob("*manifest*.json"))) >= 2


@pytest.mark.parametrize(
    "rel", [PRED, "models/mlp_full_s0.pt", "models/mlp_full_s0.preproc.pkl"]
)
def test_any_single_existing_target_refuses(tmp_path: Path, rel: str) -> None:
    target = tmp_path / "out" / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    with pytest.raises(TrainError, match=target.name.replace(".", r"\.")):
        run(tmp_path, make_folds())
    assert target.read_bytes() == b"x" and not (tmp_path / "ckpt").exists()


def test_label_stripped_test_fold_same_val_side(base, tmp_path: Path) -> None:
    _, _, ref = base
    stripped = make_folds(labelled_test=False)
    assert "sdss_class" not in stripped["test"].frame
    assert val_side(run(tmp_path, stripped)) == val_side(ref)


def test_test_feature_values_never_influence_fitting(base, tmp_path: Path) -> None:
    root, folds, ref = base
    other = make_folds()
    cols = [c for c in FEATURE_COLUMNS if c != INDICATOR_COLUMN]
    other["test"].frame[cols] = (
        np.random.default_rng(99).normal(size=(60, len(cols))) * 50
    )
    assert val_side(run(tmp_path, other)) == val_side(ref)
    a, b = read(root).read(), read(tmp_path).read()
    is_val = np.isin(a["source_id"].to_numpy(), folds["val"].source_id)
    for c in ("p_cal_star", "p_cal_galaxy", "p_cal_qso"):
        assert np.array_equal(a[c].to_numpy()[is_val], b[c].to_numpy()[is_val])
        assert not np.array_equal(a[c].to_numpy()[~is_val], b[c].to_numpy()[~is_val])


def test_control_val_labels_do_influence_fit(base, tmp_path: Path) -> None:
    _, _, ref = base
    f = make_folds()
    f["val"] = Fold(f["val"].source_id, np.roll(f["val"].labels, 1), f["val"].frame)
    assert val_side(run(tmp_path, f)) != val_side(ref)


def test_preprocessor_fit_sees_only_train_rows(tmp_path: Path, monkeypatch) -> None:
    seen: list[pd.DataFrame] = []
    orig = dl_train.Preprocessor.fit

    def spy(self, frame):
        seen.append(frame.copy())
        return orig(self, frame)

    monkeypatch.setattr(dl_train.Preprocessor, "fit", spy)
    folds = make_folds()
    run(tmp_path, folds)
    cols = list(FEATURE_COLUMNS)
    assert len(seen) == 1
    assert np.array_equal(
        seen[0][cols].to_numpy(), folds["train"].frame[cols].to_numpy(), equal_nan=True
    )


def test_train_subset_drives_pi_src_and_metadata(tmp_path: Path) -> None:
    folds = make_folds()
    keep = subsample_index(SIZES["train"], 90)
    out = run(tmp_path, folds, train_subset=90)
    assert out["train_class_fractions"] == fractions(folds["train"].labels[keep])
    assert read(tmp_path).schema_arrow.metadata[b"train_subset"] == b"90"


def test_class_absent_from_subsample_raises(tmp_path: Path) -> None:
    folds = make_folds()
    keep = subsample_index(SIZES["train"], 30)
    y = folds["train"].labels.copy()
    y[keep] = 0
    y[np.setdiff1d(np.arange(150), keep)[:6]] = [1, 2, 1, 2, 1, 2]
    folds["train"] = Fold(folds["train"].source_id, y, folds["train"].frame)
    with pytest.raises(TrainError, match="lack a class"):
        run(tmp_path, folds, train_subset=30)
