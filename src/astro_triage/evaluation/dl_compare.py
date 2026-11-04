"""Purity at fixed completeness and the LightGBM seed margin ``delta`` (ADR 0006 sections 4 and 15).

Only the part needed to compute ``delta`` lives here for now (F1, F2, F5). Every quantity is
computed on the VAL fold: test rows are never loaded (labels are filtered by ``fold == "val"``
at read time and predictions are read with a ``source_id in <val ids>`` filter pushed down to
parquet, so rows of other folds are not materialised).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from astro_triage.evaluation.runner import GRID, EvalError, FloatArr, StrArr
from astro_triage.split import build_pixel_map, pixel_l5
from eval import metrics as M
from eval.reweight import reweight

# ADR 0006 section 15 (header): code constants.
HEADLINE_METRIC: Final = "purity_at_completeness"
TARGET_CLASSES: Final = ("QSO", "GALAXY")
PRIMARY_COMPLETENESS: Final = 0.80  # ADR 0006 section 4.2: C* = 0.80
COMPLETENESS_LEVELS: Final = (0.80, 0.90)  # ADR 0006 section 4.2 (secondary 0.90)
# ADR 0006 section 4.1: P2 (Delchambre+2023 all-sky, ADR 0003 D4 table) is the primary prior;
# values come from runner.GRID, which holds the ADR 0003 D4 grid (P2 = 0.998813/0.000989/0.000198).
PRIMARY_PRIOR: Final = "P2"
SENSITIVITY_PRIORS: Final = ("P1", "P3")
DELTA_SD_MULTIPLIER: Final = 2.0  # ADR 0006 section 4.2, F5: delta = 2 * sd
DELTA_DDOF: Final = 1  # ADR 0006 section 4.2, F5: numpy.std(v, ddof=1)
DELTA_FOLD: Final = "val"  # ADR 0006 section 4.2, "SD fold ... val"
BLOCK_LEVEL: Final = 3
BLOCK_DIVISOR: Final = 16  # ADR 0006 section 4.4: block_i = pixel5 // 16
MIN_BLOCKS: Final = 20  # ADR 0006 section 4.4
N_SEEDS: Final = 5  # ADR 0006 section 4.2, F3
LOW_COUNT_RULE: Final = "total"  # ADR 0006 section 4.4 (iii)
HEADLINE_PAIRS: Final = (("mlp_full", "lgbm_full"), ("ftt_full", "lgbm_full"))
DESCRIPTIVE_PAIRS: Final = (
    ("ens_mlp_full", "lgbm_full"),
    ("ens_ftt_full", "lgbm_full"),
)

FEATURE_SETS: Final = ("full", "noind")  # ADR 0006 section 4.2, F5
DELTA_PRIORS: Final = (PRIMARY_PRIOR, *SENSITIVITY_PRIORS)  # F5: pi in {P1, P2, P3}
# ADR 0003 D4: "P3 is applied only to test objects in bands B and C".
P3_BANDS: Final = ("B", "C")
# Version suffix of the LightGBM seed files (lgbm.MODEL_VERSION); kept local, ADR 0006 section 15
# lists no lgbm import for this module.
PREDICTION_VERSION: Final = "v0"
LABEL_COLUMNS: Final = ["source_id", "sdss_class", "healpix_l5"]
DELTA_FORMULA: Final = (
    "v_k = P_{L_s<k>,c}(val; pi, C*), k = 0..4, threshold t_{L_s<k>,c}(C*) fitted on val (F1); "
    "delta_f,c(pi, C*) = 2 * sd(v_0..v_4, ddof=1) = 2 * sqrt(sum_k (v_k - mean(v))^2 / (5 - 1)) "
    "(numpy.std(v, ddof=1))"
)


def threshold_at_completeness(
    y: StrArr, score: FloatArr, cls: str, completeness: float
) -> float:
    """F1: the k-th largest score of true ``cls`` rows, k = ceil(C* n) (1-based)."""
    s = np.sort(score[y == cls])[::-1]
    n = s.size
    if n == 0:
        raise EvalError(f"no {cls} rows to fit a threshold on")
    if not np.all(np.isfinite(s)):
        raise EvalError(f"non-finite {cls} scores while fitting a threshold")
    # round() guards against a float product a few ulps above an integer making ceil jump by one;
    # verified equal to plain ceil for C* in {0.80, 0.90}, n up to 2e6.
    k = math.ceil(round(completeness * n, 9))
    if not 1 <= k <= n:
        raise EvalError(f"completeness {completeness} gives k={k} outside 1..{n}")
    return float(s[k - 1])


def weighted_purity(
    y: StrArr, score: FloatArr, cls: str, t: float, prior: Mapping[str, float]
) -> float:
    """F2: prior-weighted purity of ``score >= t`` for ``cls``; nan when nothing is selected."""
    w = reweight(y, prior)
    return _purity_from_weights(y, score, cls, t, w)


def _purity_from_weights(
    y: StrArr, score: FloatArr, cls: str, t: float, w: FloatArr
) -> float:
    selected = score >= t
    den = float(w[selected].sum())
    if den == 0.0:
        return math.nan
    return float(w[selected & (y == cls)].sum()) / den


def contaminant_counts(
    y: StrArr, score: FloatArr, cls: str, t: float
) -> dict[str, int]:
    """Raw count of selected rows per other true class."""
    selected = score >= t
    return {k: int((selected & (y == k)).sum()) for k in M.CLASSES if k != cls}


def is_low_count(counts: Mapping[str, int]) -> bool:
    """ADR 0006 section 4.4 (iii), total reading: the sum over other classes is < 10."""
    return sum(counts.values()) < M.LOW_COUNT


def block_ids(pixel5: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return pixel5 // BLOCK_DIVISOR


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_float(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _load_val_labels(outdir: Path) -> pd.DataFrame:
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


def _val_scores(path: Path, val_ids: npt.NDArray[np.int64]) -> dict[str, FloatArr]:
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
        _purity_from_weights(yp, s[pop], cls, t, wp)
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
        "delta": _json_float(delta),
        "v_k": [_json_float(x) for x in v],
        "v_mean": _json_float(float(arr.mean())) if finite else None,
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
    val = _load_val_labels(outdir)
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
            shas[fs][f"lgbm_{fs}_s{k}"] = _sha256(path)
            scores_by_seed.append(_val_scores(path, val_ids))
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
        "labels_sha256": _sha256(outdir / "labels.parquet"),
        "prediction_sha256": shas,
        "delta": cells,
    }
