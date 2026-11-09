"""Block-bootstrap core of the DL comparison (ADR 0006 sections 4.2 and 4.4, spec S12).

No IO. A replicate is a multiplicity vector ``m`` (bincount of the drawn indices); the purity of a
frozen selection is computed from per-class sums of ``m``, so no B x n matrix is ever built and
every system and seed sees the same replicate stream (one ``default_rng(seed)`` per call).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from astro_triage.evaluation.dl_compare import block_ids
from astro_triage.evaluation.runner import BoolArr, EvalError, FloatArr, StrArr
from eval import metrics as M

BANDS = ("A", "B", "C")
N_CLASSES = len(M.CLASSES)
CI_PERCENTILES = (2.5, 97.5)  # ADR 0006 section 4.4: 95 % percentile interval
IntArr = npt.NDArray[np.int64]


class Selection(NamedTuple):
    """Frozen selection kept sparse: indices of selected rows and their true-class codes."""

    ids: IntArr
    code: IntArr


def class_codes(y: StrArr) -> IntArr:
    code = np.full(y.size, -1, dtype=np.int64)
    for i, c in enumerate(M.CLASSES):
        code[y == c] = i
    if (code < 0).any():
        raise EvalError("test labels outside the known classes")
    return code


def prior_vector(prior: Mapping[str, float]) -> FloatArr:
    return np.array([prior[c] for c in M.CLASSES], dtype=np.float64)


def block_labels(pixel5: IntArr, band: StrArr) -> IntArr:
    """ADR 0006 4.4: level-3 parent of the level-5 pixel, split per latitude band."""
    return block_ids(pixel5) * 4 + np.searchsorted(np.asarray(BANDS), band)


def make_selection(code: IntArr, selected: BoolArr) -> Selection:
    ids = np.flatnonzero(selected)
    return Selection(ids, code[ids])


def class_counts(
    code: IntArr, sel: Selection, m: FloatArr | None = None
) -> tuple[FloatArr, FloatArr]:
    """(n_k, cnt_k) under multiplicity ``m`` (None = the observed sample)."""
    if m is None:
        return (
            np.bincount(code, minlength=N_CLASSES).astype(np.float64),
            np.bincount(sel.code, minlength=N_CLASSES).astype(np.float64),
        )
    return (
        np.bincount(code, weights=m, minlength=N_CLASSES),
        np.bincount(sel.code, weights=m[sel.ids], minlength=N_CLASSES),
    )


def purity(n: FloatArr, cnt: FloatArr, pi: FloatArr, c: int) -> float:
    """F2 from class sums; nan when class c is absent or the weighted denominator is 0 (G6)."""
    if n[c] <= 0:
        return math.nan
    present = n > 0
    w = np.zeros(N_CLASSES)
    w[present] = pi[present] / n[present]
    den = float(w @ cnt)
    return math.nan if den == 0.0 else float(w[c] * cnt[c] / den)


def replicate_purities(
    code: IntArr,
    block: IntArr,
    band: StrArr,
    selections: Mapping[Hashable, tuple[Selection, int]],
    priors: Mapping[str, FloatArr],
    n_boot: int,
    seed: int,
) -> dict[tuple[Hashable, str], FloatArr]:
    """Purity of every selection under every prior on each replicate (nan = invalid)."""
    rng = np.random.default_rng(seed)
    out = {(k, p): np.full(n_boot, np.nan) for k in selections for p in priors}
    for b in range(n_boot):
        idx = M.bootstrap_pixel_draw(block, band, rng)
        m = np.bincount(idx, minlength=code.size).astype(np.float64)
        for k, (sel, c) in selections.items():
            n, cnt = class_counts(code, sel, m)
            for p, pi in priors.items():
                out[(k, p)][b] = purity(n, cnt, pi, c)
    return out


def seed_mean_diff(x: list[FloatArr], ref: list[FloatArr]) -> FloatArr:
    """F4 per replicate: nan propagates, so a replicate invalid for any system is invalid."""
    return np.mean(np.stack(x), axis=0) - np.mean(np.stack(ref), axis=0)


def interval(d: FloatArr) -> tuple[float, float, float]:
    """(lo, hi, invalid fraction); the interval is nan when the cell is n/a (> 1 % invalid)."""
    ok = np.isfinite(d)
    frac = int((~ok).sum()) / ok.size
    if frac > M.MAX_INVALID_FRACTION or not ok.any():
        return math.nan, math.nan, frac
    lo, hi = np.percentile(d[ok], CI_PERCENTILES)
    return float(lo), float(hi), frac
