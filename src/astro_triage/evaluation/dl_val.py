"""Val-fold IO for the LightGBM seed margin ``delta`` (ADR 0006 sections 4 and 15).

Every quantity is computed on the VAL fold: test rows are never loaded (labels are filtered by
``fold == "val"`` at read time and predictions are read with a ``source_id in <val ids>`` filter
pushed down to parquet, so rows of other folds are not materialised). The pure functions live in
``dl_compare``, which must never import this module.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from astro_triage.evaluation.dl_compare import (
    COMPLETENESS_LEVELS,
    DELTA_DDOF,
    DELTA_FOLD,
    DELTA_PRIORS,
    DELTA_SD_MULTIPLIER,
    FEATURE_SETS,
    MIN_BLOCKS,
    N_SEEDS,
    P3_BANDS,
    PREDICTION_VERSION,
    PRIMARY_COMPLETENESS,
    PRIMARY_PRIOR,
    TARGET_CLASSES,
    block_ids,
    contaminant_counts,
    is_low_count,
    purity_from_weights,
    threshold_at_completeness,
)
from astro_triage.evaluation.runner import GRID, EvalError, FloatArr, StrArr
from astro_triage.split import build_pixel_map, pixel_l5
from eval import metrics as M
from eval.reweight import reweight

LABEL_COLUMNS: Final = ["source_id", "sdss_class", "healpix_l5"]
DELTA_FORMULA: Final = (
    "v_k = P_{L_s<k>,c}(val; pi, C*), k = 0..4, threshold t_{L_s<k>,c}(C*) fitted on val (F1); "
    "delta_f,c(pi, C*) = 2 * sd(v_0..v_4, ddof=1) = 2 * sqrt(sum_k (v_k - mean(v))^2 / (5 - 1)) "
    "(numpy.std(v, ddof=1))"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_float(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def load_val_labels(outdir: Path) -> pd.DataFrame:
    """Val label rows only: the fold filter is applied at read time, test rows are never loaded."""
    path = outdir / "labels.parquet"
    try:
        table = pq.read_table(
            path, columns=LABEL_COLUMNS, filters=[("fold", "==", DELTA_FOLD)]
        )
    except (OSError, KeyError, ValueError, pa.ArrowException) as exc:
        raise EvalError(f"cannot read val labels from {path}: {exc}") from exc
    val = table.to_pandas()
    val["sdss_class"] = val["sdss_class"].astype(str)
    if val.empty or not val["source_id"].is_unique:
        raise EvalError("val labels are empty or source_id is not unique")
    if not set(val["sdss_class"]) <= set(M.CLASSES):
        raise EvalError(f"unexpected classes in val: {sorted(set(val['sdss_class']))}")
    # canonical row order: delta must not depend on the parquet row order (bit-for-bit)
    val = val.sort_values("source_id", kind="stable", ignore_index=True)
    sid = val["source_id"].to_numpy(dtype=np.int64)
    pixel = val["healpix_l5"].to_numpy(dtype=np.int64)
    if not np.array_equal(pixel_l5(sid), pixel):
        raise EvalError(
            "healpix_l5 in labels disagrees with source_id // 2**35 // 4**7"
        )
    pmap = build_pixel_map()
    if (pmap["fold"].to_numpy()[pixel] != DELTA_FOLD).any():
        raise EvalError(
            "val rows lie in pixels that the committed map assigns elsewhere"
        )
    val["band"] = pmap["lat_band"].to_numpy()[pixel].astype(str)
    return val


def val_scores(path: Path, val_ids: npt.NDArray[np.int64]) -> dict[str, FloatArr]:
    """p_cal_<class> of the val ids, selected by id; rows of other folds are dropped unread."""
    cols = ["source_id", *(f"p_cal_{c.lower()}" for c in M.CLASSES)]
    try:
        pred = (
            pq.read_table(
                path, columns=cols, filters=[("source_id", "in", val_ids.tolist())]
            )
            .to_pandas()
            .set_index("source_id")
        )
    except (OSError, KeyError, ValueError, pa.ArrowException) as exc:
        raise EvalError(f"cannot read predictions {path}: {exc}") from exc
    if not pred.index.is_unique:
        raise EvalError(f"{path.name}: source_id is not unique")
    missing = np.setdiff1d(val_ids, pred.index.to_numpy(), assume_unique=True)
    if missing.size:
        raise EvalError(f"{path.name}: {missing.size} val ids have no prediction")
    val_pred = pred.loc[val_ids]
    out = {
        c: val_pred[f"p_cal_{c.lower()}"].to_numpy(dtype=np.float64) for c in M.CLASSES
    }
    for c, s in out.items():
        if not np.all(np.isfinite(s)):
            raise EvalError(f"{path.name}: non-finite val p_cal for {c}")
    return out


def _cell(
    y: StrArr,
    pixel: npt.NDArray[np.int64],
    band: StrArr,
    scores: list[FloatArr],
    thresholds: list[float],
    cls: str,
    w: FloatArr,
    pop: npt.NDArray[np.bool_],
) -> dict[str, object]:
    """One (feature set, class, prior, C*) cell of delta.json on the population mask ``pop``."""
    yp, pp, bp = y[pop], pixel[pop], band[pop]
    wp = w[pop]
    v = [
        purity_from_weights(yp, s[pop], cls, t, wp)
        for s, t in zip(scores, thresholds, strict=True)
    ]
    counts = [
        contaminant_counts(yp, s[pop], cls, t)
        for s, t in zip(scores, thresholds, strict=True)
    ]
    n, npix, floor_ok = M.floor_status(yp, pp)[cls]
    in_cls = yp == cls
    blocks = int(
        np.unique(
            block_ids(pp[in_cls]) * 4 + np.searchsorted(["A", "B", "C"], bp[in_cls])
        ).size
    )
    arr = np.asarray(v, dtype=np.float64)
    finite = bool(np.all(np.isfinite(arr)))
    delta = (
        DELTA_SD_MULTIPLIER * float(np.std(arr, ddof=DELTA_DDOF))
        if finite
        else math.nan
    )
    low = [is_low_count(c) for c in counts]
    return {
        "delta": json_float(delta),
        "v_k": [json_float(x) for x in v],
        "v_mean": json_float(float(arr.mean())) if finite else None,
        "thresholds": thresholds,
        "contaminant_counts": counts,
        "low_count": low,
        "floor": {"n": n, "pixels": npix, "passes": bool(floor_ok)},
        "blocks": blocks,
        "val_checks_pass": bool(
            finite and floor_ok and blocks >= MIN_BLOCKS and not any(low)
        ),
    }


def compute_delta(outdir: Path, lgbm_dir: Path) -> dict[str, object]:
    """F5: delta of the single-seed LightGBM val purities, every feature set/class/prior/C*.

    Reads val rows only. Returns the delta.json document; the caller writes it.
    """
    val = load_val_labels(outdir)
    y = val["sdss_class"].to_numpy(dtype=str)
    pixel = val["healpix_l5"].to_numpy(dtype=np.int64)
    band = val["band"].to_numpy(dtype=str)
    val_ids = val["source_id"].to_numpy(dtype=np.int64)
    pops = {
        p: (np.isin(band, P3_BANDS) if p == "P3" else np.ones(y.size, dtype=bool))
        for p in DELTA_PRIORS
    }
    weights = {p: reweight(y[pops[p]], GRID[p]) for p in DELTA_PRIORS}
    full_w = {}
    for p in DELTA_PRIORS:
        full_w[p] = np.zeros(y.size, dtype=np.float64)
        full_w[p][pops[p]] = weights[p]

    shas: dict[str, dict[str, str]] = {}
    cells: dict[str, object] = {}
    for fs in FEATURE_SETS:
        scores_by_seed: list[dict[str, FloatArr]] = []
        shas[fs] = {}
        for k in range(N_SEEDS):
            path = lgbm_dir / f"predictions_lgbm_{fs}_s{k}_{PREDICTION_VERSION}.parquet"
            if not path.is_file():
                raise EvalError(f"missing seed predictions {path}")
            shas[fs][f"lgbm_{fs}_s{k}"] = sha256_file(path)
            scores_by_seed.append(val_scores(path, val_ids))
        per_class: dict[str, object] = {}
        for cls in TARGET_CLASSES:
            per_c: dict[str, object] = {}
            for c_star in COMPLETENESS_LEVELS:
                scores = [s[cls] for s in scores_by_seed]
                thr = [threshold_at_completeness(y, s, cls, c_star) for s in scores]
                per_p: dict[str, object] = {}
                for p in DELTA_PRIORS:
                    cell = _cell(y, pixel, band, scores, thr, cls, full_w[p], pops[p])
                    in_pop = pops[p] & (y == cls)
                    cell["val_completeness"] = [
                        float((s[in_pop] >= t).sum()) / int(in_pop.sum())
                        for s, t in zip(scores, thr, strict=True)
                    ]
                    per_p[p] = cell
                per_c[f"{c_star:.2f}"] = per_p
            per_class[cls] = per_c
        cells[fs] = per_class

    n_val_by_class = {c: int((y == c).sum()) for c in M.CLASSES}
    assert pixel.size == y.size == val_ids.size  # val rows only, by construction
    return {
        "schema_name": "astro_triage.lgbm_seeds_delta",
        "schema_version": 1,
        "fold": DELTA_FOLD,
        "formula": DELTA_FORMULA,
        "n_seeds": N_SEEDS,
        "delta_sd_multiplier": DELTA_SD_MULTIPLIER,
        "delta_ddof": DELTA_DDOF,
        "primary_prior": PRIMARY_PRIOR,
        "primary_completeness": PRIMARY_COMPLETENESS,
        "priors": {p: GRID[p] for p in DELTA_PRIORS},
        "p3_population": f"val rows in bands {', '.join(P3_BANDS)} only (threshold fitted on all val)",
        "min_blocks": MIN_BLOCKS,
        "n_val_rows": int(y.size),
        "n_val_by_class": n_val_by_class,
        "labels_sha256": sha256_file(outdir / "labels.parquet"),
        "prediction_sha256": shas,
        "delta": cells,
    }
