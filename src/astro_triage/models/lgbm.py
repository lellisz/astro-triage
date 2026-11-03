"""Layer 1 LightGBM multiclass classifier with temperature scaling (ADR 0003, ADR 0004 D7.4).

Protocol (the test fold is read for final evaluation only):
* fit: ``train`` rows; early stopping on ``val`` multiclass logloss;
* calibration: one temperature fitted on ``val`` logits only;
* ``test`` enters ``evaluate_split`` and the predictions table after the booster and T are
  frozen. No function that fits anything receives test rows.
* ADR 0003 prescribes no training sample weights, so none are used (every row weight 1)
  and ``pi_src`` for the prior shift is the plain train class fractions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import accuracy_score, f1_score

from astro_triage.dataset.build import (
    CLASSES,
    DEFAULT_OUTDIR,
    FEATURE_COLUMNS,
    FOLDS,
    REPO_ROOT,
)
from astro_triage.ingest.manifest import (
    GAIA_ACKNOWLEDGEMENT,
    GAIA_TERMS,
    SDSS_ACKNOWLEDGEMENT,
    SDSS_CITATIONS,
    SDSS_TERMS,
    XMATCH_ACKNOWLEDGEMENT,
    XMATCH_CITATIONS,
    XMATCH_TERMS,
    Provenance,
    build_manifest,
    file_sha256,
    write_manifest,
)
from eval.metrics import confusion_matrix, ece, regime_weights

__all__ = [
    "MODEL_VERSION",
    "ModelSpec",
    "TrainError",
    "train_all",
]

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]

SEED: Final = 20261025
MODEL_VERSION: Final = "v0"
MAX_TREES: Final = 2000
PATIENCE: Final = 50
N_THREADS: Final = 12
TEMPERATURE_BOUNDS: Final = (0.05, 20.0)
PROB_SUM_TOL: Final = 1e-6
TOP_GAIN: Final = 10
GAIN_SHARE_ALERT: Final = 0.25
ABLATION_GAP_ALERT: Final = 0.10
CALIBRATION_NAME: Final = "temperature_scaling_multinomial"

PARAMS: Final[dict[str, object]] = {
    "objective": "multiclass",
    "num_class": len(CLASSES),
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": N_THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "seed": SEED,
    "verbosity": -1,
}

# ADR 0004 D7.4: the solution type and the nine columns that are NULL on 2-parameter solutions.
NOIND_DROPPED: Final[tuple[str, ...]] = (
    "astrometric_params_solved",
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
INDICATOR_COLUMN: Final = "astrometric_params_solved"


class TrainError(Exception):
    """An input violates a documented precondition; the message names the fix."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    features: tuple[str, ...]


MODEL_FULL: Final = ModelSpec("lgbm_full", FEATURE_COLUMNS)
MODEL_NOIND: Final = ModelSpec(
    "lgbm_noind", tuple(c for c in FEATURE_COLUMNS if c not in NOIND_DROPPED)
)


@dataclass(frozen=True)
class Fold:
    source_id: Ints
    labels: Ints
    frame: pd.DataFrame

    def matrix(self, features: tuple[str, ...]) -> npt.NDArray[np.float32]:
        return self.frame[list(features)].to_numpy(dtype=np.float32)

    @property
    def names(self) -> npt.NDArray[np.str_]:
        return np.asarray(CLASSES)[self.labels]


def load_folds(outdir: Path) -> dict[str, Fold]:
    """Read labels and features, check they align by ``source_id``, split by ``labels.fold``."""
    try:
        labels = pq.read_table(
            outdir / "labels.parquet", columns=["source_id", "sdss_class", "fold"]
        ).to_pandas()
        feats = pq.read_table(outdir / "features.parquet").to_pandas()
    except (OSError, KeyError, pa.ArrowException) as exc:
        raise TrainError(
            f"cannot read labels/features from {outdir.name}: {exc}; "
            "run scripts/build_dataset.py first"
        ) from exc
    for name, frame in (("labels", labels), ("features", feats)):
        if frame["source_id"].duplicated().any():
            raise TrainError(f"{name}.source_id has duplicates")
    if len(labels) != len(feats) or set(labels["source_id"]) != set(feats["source_id"]):
        raise TrainError("labels and features do not cover the same source_id set")
    missing = [c for c in FEATURE_COLUMNS if c not in feats.columns]
    if missing:
        raise TrainError(f"features.parquet lacks columns {missing}")
    labels["sdss_class"] = labels["sdss_class"].astype(str)
    labels["fold"] = labels["fold"].astype(str)
    frame = labels.merge(feats, on="source_id", how="inner", validate="one_to_one")
    if len(frame) != len(labels):
        raise TrainError("merge on source_id changed the row count")
    frame = frame.sort_values("source_id").reset_index(drop=True)
    if not frame["sdss_class"].isin(CLASSES).all():
        raise TrainError(f"sdss_class outside {CLASSES}")
    if not frame["fold"].isin(FOLDS).all():
        raise TrainError(f"fold outside {FOLDS}")
    if np.isinf(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)).any():
        raise TrainError("features contain inf")
    class_index = {c: i for i, c in enumerate(CLASSES)}
    folds: dict[str, Fold] = {}
    for name in FOLDS:
        part = frame[frame["fold"] == name].reset_index(drop=True)
        present = set(part["sdss_class"])
        if not present >= set(CLASSES):
            raise TrainError(
                f"fold {name} lacks classes {sorted(set(CLASSES) - present)}"
            )
        folds[name] = Fold(
            source_id=part["source_id"].to_numpy(dtype=np.int64),
            labels=part["sdss_class"].map(class_index).to_numpy(dtype=np.int64),
            frame=part,
        )
    return folds


def fit_booster(
    train: Fold,
    val: Fold,
    features: tuple[str, ...],
    params: dict[str, object] | None = None,
) -> tuple[lgb.Booster, int]:
    """Fit on train; early stopping on val logloss. Takes no test rows."""
    dtrain = lgb.Dataset(
        train.matrix(features),
        label=train.labels,
        feature_name=list(features),
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        val.matrix(features),
        label=val.labels,
        feature_name=list(features),
        reference=dtrain,
        free_raw_data=False,
    )
    booster = lgb.train(
        PARAMS if params is None else params,
        dtrain,
        num_boost_round=MAX_TREES,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False)],
    )
    if booster.best_iteration <= 0:
        raise TrainError("early stopping returned no best iteration")
    return booster, int(booster.best_iteration)


def logits(booster: lgb.Booster, fold: Fold, features: tuple[str, ...]) -> Floats:
    raw = booster.predict(
        fold.matrix(features), raw_score=True, num_iteration=booster.best_iteration
    )
    out = np.asarray(raw, dtype=np.float64)
    if out.shape != (len(fold.labels), len(CLASSES)) or not np.isfinite(out).all():
        raise TrainError("booster produced non-finite or mis-shaped logits")
    return out


def log_loss(z: Floats, y: Ints, temperature: float = 1.0) -> float:
    """Multiclass logloss of ``softmax(z / temperature)``."""
    logp = log_softmax(z / temperature, axis=1)
    return float(-np.mean(logp[np.arange(len(y)), y]))


def fit_temperature(val_logits: Floats, val_labels: Ints) -> float:
    """Temperature that minimises val logloss (search over log T, bounded)."""
    lo, hi = (math.log(b) for b in TEMPERATURE_BOUNDS)
    res = minimize_scalar(
        lambda log_t: log_loss(val_logits, val_labels, math.exp(log_t)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-9},
    )
    if not res.success:
        raise TrainError(f"temperature search failed: {res.message}")
    return float(math.exp(res.x))


def shift_prior(proba: Floats, pi_src: Floats) -> Floats:
    """ADR 0003 prior shift to the uniform prior: ``p * (1/K) / pi_src``, renormalised."""
    shifted = proba / pi_src / len(CLASSES)
    return np.asarray(shifted / shifted.sum(axis=1, keepdims=True), dtype=np.float64)


def _ece_summary(
    names: npt.NDArray[np.str_],
    proba: Floats,
    weights: Floats | None,
) -> dict[str, float | None]:
    per_class = ece(names, proba, sample_weight=weights)
    out: dict[str, float | None] = {c: _finite(per_class[c][0].value) for c in CLASSES}
    vals = [v for v in out.values() if v is not None]
    out["mean"] = float(np.mean(vals)) if vals else None
    return out


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def evaluate_split(
    fold: Fold, z: Floats, temperature: float, pi_src: Floats, with_confusion: bool
) -> dict[str, object]:
    """Metrics before (T = 1) and after temperature scaling on one fold."""
    y, names = fold.labels, fold.names
    p_raw, p_cal = softmax(z, axis=1), softmax(z / temperature, axis=1)
    pred = np.argmax(p_raw, axis=1)
    if not np.array_equal(pred, np.argmax(p_cal, axis=1)):
        raise TrainError("temperature scaling changed an argmax decision")
    balanced = regime_weights(names)
    out: dict[str, object] = {
        "n": len(y),
        "logloss_before": log_loss(z, y),
        "logloss_after": log_loss(z, y, temperature),
        "ece_before": _ece_summary(names, p_raw, None),
        "ece_after": _ece_summary(names, p_cal, None),
        "ece_balanced_before": _ece_summary(
            names, shift_prior(p_raw, pi_src), balanced
        ),
        "ece_balanced_after": _ece_summary(names, shift_prior(p_cal, pi_src), balanced),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=list(range(len(CLASSES))), average="macro")
        ),
    }
    if with_confusion:
        cm = confusion_matrix(names, np.asarray(CLASSES)[pred])
        out["confusion_rows_true_cols_pred"] = {
            "classes": list(CLASSES),
            "counts": cm.astype(int).tolist(),
        }
    return out


def gain_importance(booster: lgb.Booster) -> list[dict[str, object]]:
    gain = np.asarray(booster.feature_importance("gain"), dtype=np.float64)
    total = float(gain.sum())
    order = np.argsort(-gain, kind="stable")
    names = booster.feature_name()
    return [
        {
            "rank": rank + 1,
            "feature": names[i],
            "gain": float(gain[i]),
            "share": float(gain[i] / total) if total > 0 else None,
        }
        for rank, i in enumerate(order[:TOP_GAIN])
    ]


def indicator_gain(booster: lgb.Booster) -> dict[str, object] | None:
    names = booster.feature_name()
    if INDICATOR_COLUMN not in names:
        return None
    gain = np.asarray(booster.feature_importance("gain"), dtype=np.float64)
    i = names.index(INDICATOR_COLUMN)
    return {
        "gain": float(gain[i]),
        "share": float(gain[i] / gain.sum()),
        "rank": int((gain > gain[i]).sum()) + 1,
    }


def _git_sha() -> str:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return done.stdout.strip() if done.returncode == 0 else "unavailable"


def _atomic(path: Path, write: Callable[[str], object]) -> None:
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        write(tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _manifest(path: Path, name: str, rows: int, params: str, inputs: list[str]) -> Path:
    provenance = Provenance(
        table_name=name,
        source="derived",
        release="dataset_100k_v0",
        obtained_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        query_text=params,
        query_sha256=hashlib.sha256(params.encode()).hexdigest(),
        row_count=rows,
        file_sha256=file_sha256(path),
        file_bytes=path.stat().st_size,
        terms=f"{XMATCH_TERMS} | {GAIA_TERMS} | {SDSS_TERMS}",
        citations=[*XMATCH_CITATIONS, *SDSS_CITATIONS],
        acknowledgement=(
            f"{XMATCH_ACKNOWLEDGEMENT} {GAIA_ACKNOWLEDGEMENT} {SDSS_ACKNOWLEDGEMENT}"
        ),
        notes=[f"input: {item}" for item in inputs],
    )
    return write_manifest(build_manifest(path, provenance), path.parent)


def _check_probabilities(columns: dict[str, npt.NDArray[np.float32]]) -> None:
    for prefix in ("p_raw", "p_cal"):
        stack = np.stack([columns[f"{prefix}_{c.lower()}"] for c in CLASSES], axis=1)
        if not np.isfinite(stack).all() or stack.min() < 0 or stack.max() > 1:
            raise TrainError(f"{prefix}_* has NaN, inf or values outside [0, 1]")
        worst = float(np.abs(stack.astype(np.float64).sum(axis=1) - 1.0).max())
        if worst > PROB_SUM_TOL:
            raise TrainError(f"{prefix}_* row sums deviate from 1 by {worst:.3g}")


def predictions_table(
    source_id: Ints, z: Floats, temperature: float
) -> tuple[pa.Table, dict[str, npt.NDArray[np.float32]]]:
    raw, cal = softmax(z, axis=1), softmax(z / temperature, axis=1)
    columns: dict[str, npt.NDArray[np.float32]] = {}
    for i, c in enumerate(CLASSES):
        columns[f"p_raw_{c.lower()}"] = raw[:, i].astype(np.float32)
        columns[f"p_cal_{c.lower()}"] = cal[:, i].astype(np.float32)
    _check_probabilities(columns)
    n = len(source_id)
    version = pa.DictionaryArray.from_arrays(
        pa.array(np.zeros(n, dtype=np.int8)), pa.array([MODEL_VERSION])
    )
    fields = [
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("model_version", version.type, nullable=False),
    ]
    arrays: list[pa.Array] = [pa.array(source_id, type=pa.int64()), version]
    for prefix, nullable in (("p_raw", False), ("p_cal", True)):
        for cls in ("star", "qso", "galaxy"):
            fields.append(pa.field(f"{prefix}_{cls}", pa.float32(), nullable=nullable))
            arrays.append(pa.array(columns[f"{prefix}_{cls}"], type=pa.float32()))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields)), columns


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def run_model(
    spec: ModelSpec,
    folds: dict[str, Fold],
    outdir: Path,
    *,
    seed: int = SEED,
    model_id: str | None = None,
) -> dict[str, object]:
    model_id = spec.model_id if model_id is None else model_id
    fit_params = {**PARAMS, "seed": seed}
    train, val, test = folds["train"], folds["val"], folds["test"]
    booster, best_iter = fit_booster(train, val, spec.features, fit_params)
    z_val = logits(booster, val, spec.features)
    temperature = fit_temperature(z_val, val.labels)
    counts = np.bincount(train.labels, minlength=len(CLASSES)).astype(np.float64)
    pi_src = counts / counts.sum()

    z_test = logits(booster, test, spec.features)
    val_metrics = evaluate_split(val, z_val, temperature, pi_src, False)
    test_metrics = evaluate_split(test, z_test, temperature, pi_src, True)

    z_all = np.concatenate([z_val, z_test])
    ids = np.concatenate([val.source_id, test.source_id])
    table, _ = predictions_table(ids, z_all, temperature)
    fractions = {c: float(pi_src[i]) for i, c in enumerate(CLASSES)}
    inputs = ["labels.parquet", "features.parquet"]
    meta = {
        "schema_name": "predictions",
        "schema_version": "1",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "inputs": json.dumps(inputs),
        "model_id": model_id,
        "model_version": MODEL_VERSION,
        "features_file": "features.parquet",
        "labels_file": "labels.parquet",
        "split_file": "labels.parquet (column fold, astro_triage.split)",
        "train_class_fractions": json.dumps(fractions),
        "calibration": CALIBRATION_NAME,
        "calibration_fit_fold": "val",
    }
    pred_path = outdir / f"predictions_{model_id}_{MODEL_VERSION}.parquet"
    model_path = outdir / "models" / f"{model_id}.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    out_table = table.sort_by("source_id").replace_schema_metadata(meta)
    _atomic(pred_path, lambda tmp: pq.write_table(out_table, tmp, compression="zstd"))
    _atomic(
        model_path,
        lambda tmp: booster.save_model(tmp, num_iteration=best_iter),
    )
    params = (
        f"lightgbm {lgb.__version__}; params={json.dumps(fit_params, sort_keys=True)}; "
        f"max_trees={MAX_TREES}; patience={PATIENCE}; best_iteration={best_iter}; "
        f"seed={seed}; calibration={CALIBRATION_NAME} fit on val"
    )
    _manifest(pred_path, pred_path.stem, len(out_table), params, inputs)
    _manifest(model_path, f"model_{model_id}", 1, params, inputs)

    gain = gain_importance(booster)
    t_lo, t_hi = TEMPERATURE_BOUNDS
    return {
        "model_id": model_id,
        "n_features": len(spec.features),
        "features": list(spec.features),
        "best_iteration": best_iter,
        "temperature": temperature,
        "temperature_at_bound": bool(
            temperature <= t_lo * 1.001 or temperature >= t_hi * 0.999
        ),
        "train_class_fractions": fractions,
        "val_calibration_fitted_here": val_metrics,
        "test_not_fitted": test_metrics,
        "gain_top10": gain,
        "astrometric_params_solved_gain": indicator_gain(booster),
        "prediction_rows": len(out_table),
        "prediction_file": pred_path.relative_to(REPO_ROOT).as_posix(),
        "model_file": model_path.relative_to(REPO_ROOT).as_posix(),
    }


def _metric(model: dict[str, object], split: str, key: str) -> float:
    block = model[split]
    if not isinstance(block, dict):
        raise TrainError("malformed model report")
    return float(block[key])


def _alerts(full: dict[str, object], noind: dict[str, object]) -> list[str]:
    alerts: list[str] = []
    for split in ("val_calibration_fitted_here", "test_not_fitted"):
        gap = _metric(full, split, "macro_f1") - _metric(noind, split, "macro_f1")
        if gap > ABLATION_GAP_ALERT:
            alerts.append(
                f"M-noind macro-F1 is {gap:.4f} below M-full on {split} "
                f"(> {ABLATION_GAP_ALERT}): the missingness pattern may carry the "
                "performance (ADR 0004 section 8 item 8; no bootstrap CI computed here)."
            )
    row = full["astrometric_params_solved_gain"]
    if isinstance(row, dict) and (
        row["rank"] == 1 or float(row["share"]) >= GAIN_SHARE_ALERT
    ):
        alerts.append(
            f"{INDICATOR_COLUMN} has gain rank {row['rank']} and share "
            f"{float(row['share']):.4f} in M-full (alert if rank 1 or share >= "
            f"{GAIN_SHARE_ALERT})."
        )
    return alerts


def train_all(outdir: Path = DEFAULT_OUTDIR) -> dict[str, object]:
    """Train M-full and M-noind, write predictions, models, manifests and the report."""
    folds = load_folds(outdir)
    by_class_fold = {
        f: {c: int((folds[f].labels == i).sum()) for i, c in enumerate(CLASSES)}
        for f in FOLDS
    }
    models = [run_model(spec, folds, outdir) for spec in (MODEL_FULL, MODEL_NOIND)]
    report: dict[str, object] = {
        "lightgbm": lgb.__version__,
        "seed": SEED,
        "params": PARAMS,
        "max_trees": MAX_TREES,
        "early_stopping_patience": PATIENCE,
        "n_per_fold": {f: len(folds[f].labels) for f in FOLDS},
        "n_per_class_and_fold": by_class_fold,
        "training_weights": "none (ADR 0003 prescribes none; pi_src = train class fractions)",
        "calibration": f"{CALIBRATION_NAME}, T fitted on val logits only, applied to val and test",
        "ece": "15 equal-mass bins (eval.metrics.ece), one-vs-rest per class, unweighted; "
        "ece_balanced: outputs shifted to the uniform prior with BALANCED weights",
        "test_usage": (
            "test rows reach only evaluate_split and the predictions table, after the "
            "booster, best iteration and T are frozen; fit_booster and fit_temperature "
            "receive train/val only"
        ),
        "models": models,
        "alerts": _alerts(models[0], models[1]),
    }
    path = outdir / "train_report.json"
    text = json.dumps(report, indent=2, default=_json_default) + "\n"
    _atomic(path, lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))
    return report
