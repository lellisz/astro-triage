"""Purity at fixed completeness and the LightGBM seed margin ``delta`` (ADR 0006 sections 4 and 15).

Constants and pure functions only, no IO. The val-fold readers and ``compute_delta`` live in
``dl_val``, which imports this module (never the reverse: no re-export, it would be a cycle).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt

from astro_triage.evaluation.runner import EvalError, FloatArr, StrArr
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
    return purity_from_weights(y, score, cls, t, w)


def purity_from_weights(
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
