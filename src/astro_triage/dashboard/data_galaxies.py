"""Error rates by magnitude, redshift and astrometric solution; galaxy scatter and gallery."""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Final

import numpy as np
import pandas as pd

from astro_triage.dashboard import fmt
from astro_triage.dashboard.data_release import (
    SEED,
    Release,
    wilson,
)
from astro_triage.dashboard.fmt import CI_LEVEL

SOLVED_NAMES: Final = {3: "2 parâmetros", 31: "5 parâmetros", 95: "6 parâmetros"}
Z_CI: Final = NormalDist().inv_cdf(0.5 + CI_LEVEL / 2)


def wilson_interval(k: int, n: int, z: float = Z_CI) -> tuple[float, float]:
    """Wilson interval at ``CI_LEVEL`` (z derived from the level, not typed by hand)."""
    return wilson(k, n, z)


def error_rate_table(
    rel: Release,
    model: str,
    true_class: str,
    by: str,
    edges: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Error rate of one true class by magnitude G, redshift or astrometric solution.

    ``by`` is ``"G"``, ``"z"`` or ``"solved"``; ``edges`` are the bin edges for G and z.
    """
    t = rel.test[rel.test["true"] == true_class]
    column = {
        "G": "phot_g_mean_mag",
        "z": "sdss_z",
        "solved": "astrometric_params_solved",
    }[by]
    rows = []
    if by == "solved":
        for key, label in SOLVED_NAMES.items():
            sub = t[t[column] == key]
            rows.append(rate_row(label, key, key, sub[f"err_{model}"]))
    else:
        assert edges is not None
        bounds = (-math.inf, *edges, math.inf)
        for lo, hi in itertools.pairwise(bounds):
            sub = t[(t[column] >= lo) & (t[column] < hi)]
            rows.append(rate_row(bin_label(lo, hi), lo, hi, sub[f"err_{model}"]))
        missing = t[t[column].isna()]
        if len(missing):
            rows.append(
                rate_row("sem medida", math.nan, math.nan, missing[f"err_{model}"])
            )
    return pd.DataFrame(rows)


def bin_label(lo: float, hi: float, exclusive_lo: bool = False) -> str:
    """pt-BR band label; ``exclusive_lo`` marks a band that does not include its lower edge."""
    if math.isinf(lo):
        return f"< {fmt.pt_g(hi)}"
    if math.isinf(hi):
        return f"≥ {fmt.pt_g(lo)}"
    return f"{'> ' if exclusive_lo else ''}{fmt.pt_g(lo)}–{fmt.pt_g(hi)}"


def rate_row(label: str, lo: float, hi: float, errors: pd.Series) -> dict:
    n, k = len(errors), int(errors.sum())
    ci_lo, ci_hi = wilson_interval(k, n)
    return {
        "label": label,
        "lo": lo,
        "hi": hi,
        "n": n,
        "errors": k,
        "rate": k / n if n else math.nan,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


GALAXY_G_EDGES: Final = (18.0, 19.0, 19.5, 20.5)
GALAXY_Z_EDGES: Final = (0.05, 0.1, 0.2, 0.3, 0.45)


def galaxy_scatter(rel: Release, model: str = "lgbm_full") -> pd.DataFrame:
    """Every true-galaxy test row with G and z: (z, G, error, confidence, predicted class)."""
    t = rel.test[rel.test["true"] == "GALAXY"]
    out = t[
        [
            "source_id",
            "sdss_z",
            "phot_g_mean_mag",
            f"err_{model}",
            f"conf_{model}",
            f"pred_{model}",
        ]
    ]
    out = out.rename(
        columns={
            "sdss_z": "z",
            "phot_g_mean_mag": "G",
            f"err_{model}": "error",
            f"conf_{model}": "confidence",
            f"pred_{model}": "pred",
        }
    )
    return out.dropna(subset=["z", "G"]).reset_index(drop=True)


def sdss_cutout_url(
    ra: float, dec: float, width_px: int = 160, scale: float = 0.4
) -> str:
    """SDSS DR17 SkyServer cutout (64 arcsec wide with the defaults)."""
    return (
        "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra:.6f}&dec={dec:.6f}&scale={scale}&width={width_px}&height={width_px}"
    )


def gallery_sample(
    rel: Release,
    model: str = "lgbm_full",
    true_class: str = "GALAXY",
    n_correct: int = 4,
    n_wrong: int = 4,
    seed: int = SEED,
) -> pd.DataFrame:
    """Seeded draw of correct and wrong objects of one class, with a cutout URL each."""
    t = rel.test[rel.test["true"] == true_class]
    rng = np.random.default_rng(seed)
    parts = []
    for wrong, k in ((False, n_correct), (True, n_wrong)):
        pool = t[t[f"err_{model}"] == wrong]
        pick = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        parts.append(pool.iloc[np.sort(pick)])
    out = pd.concat(parts, ignore_index=True)
    out = out.assign(
        correct=~out[f"err_{model}"],
        pred=out[f"pred_{model}"],
        confidence=out[f"conf_{model}"],
        url=[sdss_cutout_url(r, d) for r, d in zip(out["ra"], out["dec"], strict=True)],
    )
    cols = [
        "source_id",
        "correct",
        "pred",
        "confidence",
        "sdss_z",
        "phot_g_mean_mag",
        "astrometric_params_solved",
        "ra",
        "dec",
        "url",
    ]
    return out[cols]
