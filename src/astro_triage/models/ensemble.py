"""Convex ensemble of LightGBM and a network on calibrated probabilities (ADR 0006 section 3).

The weight and the temperature are fitted on val rows only; holdout rows are mixed with the
frozen weight and pass through the frozen temperature, and their labels are never read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from astro_triage.dataset.build import CLASSES, REPO_ROOT
from astro_triage.evaluation.dl_val import load_val_labels, sha256_file
from astro_triage.evaluation.runner import EvalError
from astro_triage.models.lgbm import (
    CALIBRATION_NAME,
    MODEL_VERSION,
    PROB_SUM_TOL,
    TrainError,
    _atomic,
    _git_sha,
    _manifest,
    fit_temperature,
    log_loss,
    predictions_table,
)

__all__ = ["TIE_ABS_TOL", "W_GRID", "build_ensemble", "fit_weight", "mix"]

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]

W_GRID: Final = tuple(round(i / 10, 1) for i in range(11))
TIE_ABS_TOL: Final = 1e-12
PROB_FLOOR: Final = 1e-12
FRACTION_TOL: Final = 1e-9
P_COLUMNS: Final = tuple(f"p_cal_{c.lower()}" for c in CLASSES)


def mix(p_lgbm: Floats, p_net: Floats, w: float) -> Floats:
    """``w * p_lgbm + (1 - w) * p_net`` with rows renormalised."""
    mixed = w * p_lgbm + (1.0 - w) * p_net
    return np.asarray(mixed / mixed.sum(axis=1, keepdims=True), dtype=np.float64)


def _log_mix(p_lgbm: Floats, p_net: Floats, w: float) -> Floats:
    return np.log(np.clip(mix(p_lgbm, p_net, w), PROB_FLOOR, None))


def fit_weight(p_lgbm_val: Floats, p_net_val: Floats, y_val: Ints) -> float:
    """Val-logloss-minimising LightGBM weight on ``W_GRID``; ties go to the larger weight."""
    best_w, best_loss = 1.0, float("inf")
    for w in sorted(W_GRID, reverse=True):
        loss = log_loss(_log_mix(p_lgbm_val, p_net_val, w), y_val)
        if loss < best_loss - TIE_ABS_TOL:
            best_w, best_loss = w, loss
    return best_w


def _read_component(path: Path) -> tuple[Ints, Floats, dict[str, float]]:
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        raise TrainError(f"cannot read component predictions {path}: {exc}") from exc
    meta = table.schema.metadata or {}
    try:
        fractions = json.loads(meta[b"train_class_fractions"])
        ids = table.column("source_id").to_numpy().astype(np.int64)
        p = np.stack(
            [table.column(c).to_numpy().astype(np.float64) for c in P_COLUMNS], axis=1
        )
    except (KeyError, ValueError) as exc:
        raise TrainError(f"{path.name} lacks source_id/p_cal_*/train_class_fractions") from exc
    order = np.argsort(ids, kind="stable")
    ids, p = ids[order], p[order]
    if len(np.unique(ids)) != len(ids):
        raise TrainError(f"{path.name} has duplicate source_id")
    if not np.isfinite(p).all() or p.min() < 0:
        raise TrainError(f"{path.name} has non-finite or negative p_cal")
    if p.max() > 1.0 or np.abs(p.sum(axis=1) - 1.0).max() > PROB_SUM_TOL:
        raise TrainError(f"{path.name} has p_cal above 1 or rows not summing to 1")
    return ids, p, {c: float(fractions[c]) for c in CLASSES}


def build_ensemble(
    outdir: Path, dl_dir: Path, lgbm_dir: Path, net: Literal["mlp", "ftt"], k: int
) -> dict[str, object]:
    """Mix ``lgbm_full_s<k>`` with ``<net>_full_s<k>``; write ``predictions_ens_<net>_full_s<k>``."""
    if net not in ("mlp", "ftt") or k < 0:
        raise TrainError(f"invalid ensemble request net={net!r} k={k}")
    model_id = f"ens_{net}_full_s{k}"
    target = dl_dir / f"predictions_{model_id}_{MODEL_VERSION}.parquet"
    if target.exists():
        raise TrainError(f"refusing to overwrite {target}; delete it to rebuild")
    lgbm_path = lgbm_dir / f"predictions_lgbm_full_s{k}_{MODEL_VERSION}.parquet"
    net_path = dl_dir / f"predictions_{net}_full_s{k}_{MODEL_VERSION}.parquet"
    ids, p_lgbm, frac_lgbm = _read_component(lgbm_path)
    ids_net, p_net, frac_net = _read_component(net_path)
    if not np.array_equal(ids, ids_net):
        raise TrainError(f"{lgbm_path.name} and {net_path.name} cover different source_id sets")
    if any(abs(frac_lgbm[c] - frac_net[c]) > FRACTION_TOL for c in CLASSES):
        raise TrainError(
            f"train_class_fractions differ between {lgbm_path.name} and {net_path.name}"
        )

    try:
        labels = load_val_labels(outdir)
    except EvalError as exc:
        raise TrainError(f"cannot load val labels: {exc}") from exc
    val_ids = labels["source_id"].to_numpy(dtype=np.int64)
    in_val = np.isin(ids, val_ids)
    if not np.isin(val_ids, ids).all():
        missing = val_ids[~np.isin(val_ids, ids)]
        raise TrainError(
            f"{len(missing)} val rows missing from {lgbm_path.name} / {net_path.name}; "
            f"examples source_id {missing[:5].tolist()}"
        )
    class_index = {c: i for i, c in enumerate(CLASSES)}
    y_by_id = dict(zip(val_ids.tolist(), labels["sdss_class"].map(class_index).tolist(), strict=True))
    y_val = np.asarray([y_by_id[i] for i in ids[in_val].tolist()], dtype=np.int64)

    w = fit_weight(p_lgbm[in_val], p_net[in_val], y_val)
    temperature = fit_temperature(_log_mix(p_lgbm[in_val], p_net[in_val], w), y_val)
    # weight and temperature are frozen from here on; holdout rows only go forward
    table, _ = predictions_table(ids, _log_mix(p_lgbm, p_net, w), temperature)
    inputs = [lgbm_path.name, net_path.name]
    shas = {p.name: sha256_file(p) for p in (lgbm_path, net_path)}
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
        "train_class_fractions": json.dumps(frac_lgbm),
        "calibration": CALIBRATION_NAME,
        "calibration_fit_fold": "val",
        "family": "ensemble",
        "weight": repr(w),
        "temperature": repr(temperature),
        "component_sha256": json.dumps(shas, sort_keys=True),
    }
    out_table = table.sort_by("source_id").replace_schema_metadata(meta)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic(target, lambda tmp: pq.write_table(out_table, tmp, compression="zstd"))
    params = (
        f"ensemble; w_lgbm={w}; grid={list(W_GRID)}; seed_index={k}; "
        f"calibration={CALIBRATION_NAME} fit on val"
    )
    _manifest(target, target.stem, len(out_table), params, inputs)
    return {
        "model_id": model_id,
        "weight": w,
        "temperature": temperature,
        "component_sha256": shas,
        "prediction_rows": len(out_table),
        "prediction_file": _rel(target),
    }


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
