"""Reads the parquet files of one release into the ``Release`` container, keeping memory low.

The object count per (pixel, fold) comes from a two-column read of ``labels.parquet``; the
heavy columns are read for the test fold only (pyarrow filter), features/predictions only for
the test ``source_id``s, and features/predictions are aligned on ``source_id``
with a single ``concat`` instead of one ``merge`` (full copy of the test frame) per file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from astro_triage.dashboard.data_release import (
    CLASSES,
    MODELS,
    SCORE_COLUMN,
    Release,
    classes_upper,
    single_file,
)

_LABEL_COLS = (
    "source_id",
    "sdss_class",
    "sdss_z",
    "healpix_l5",
    "fold",
    "ra",
    "dec",
    "l",
    "b",
)
_DSC_PREFIX = "classprob_dsc_allosmod_"


def _aligned(
    test_ids: np.ndarray, right: pd.DataFrame, renames: dict[str, str]
) -> pd.DataFrame:
    """Columns of ``right`` (renamed) for each id of ``test_ids``, NaN where absent (left join)."""
    right = right.assign(source_id=right["source_id"].astype("int64"))
    if not right["source_id"].is_unique:
        raise ValueError("source_id repetido em um arquivo da versão dos dados")
    out = right.set_index("source_id")[list(renames)].reindex(test_ids)
    out.columns = list(renames.values())
    return out.reset_index(drop=True)


def read_release(root: Path, name: str, results: dict, report: dict) -> Release:
    """Build the ``Release`` of ``root`` from its parquet files (see module docstring)."""
    dsc = [
        c
        for c in pq.read_schema(root / "labels.parquet").names
        if c.startswith(_DSC_PREFIX)
    ]
    path = root / "labels.parquet"
    counted = pd.read_parquet(path, columns=["healpix_l5", "fold"])
    counted["fold"] = counted["fold"].astype("category").cat.rename_categories(str)
    pixel_counts = (
        counted.groupby(["healpix_l5", "fold"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    pixel_counts["fold"] = pixel_counts["fold"].astype(str)
    del counted
    test = pd.read_parquet(
        path, columns=[*_LABEL_COLS, *dsc], filters=[("fold", "==", "test")]
    ).reset_index(drop=True)
    test["source_id"] = test["source_id"].astype("int64")
    test["fold"] = test["fold"].astype(str)
    test["true"] = classes_upper(test["sdss_class"])
    ids = test["source_id"].to_numpy()
    has_dsc = (
        test[dsc].notna().all(axis=1).to_numpy() if dsc else np.zeros(len(test), bool)
    )

    feats = pd.read_parquet(
        root / "features.parquet",
        columns=["source_id", "phot_g_mean_mag", "astrometric_params_solved", "bp_rp"],
        filters=[("source_id", "in", ids.tolist())],
    )
    parts = [
        test,
        _aligned(ids, feats, {c: c for c in feats.columns if c != "source_id"}),
        pd.DataFrame({"dsc_available": has_dsc}),
    ]
    for model in MODELS:
        keep = {f"p_cal_{SCORE_COLUMN[c]}": f"p_{model}_{c}" for c in CLASSES}
        keep |= {f"p_raw_{SCORE_COLUMN[c]}": f"praw_{model}_{c}" for c in CLASSES}
        pred = pd.read_parquet(
            single_file(root, f"predictions_{model}_*.parquet"),
            columns=["source_id", *keep],
            filters=[("source_id", "in", ids.tolist())],
        )
        parts.append(_aligned(ids, pred, keep))
    test = pd.concat(parts, axis=1)

    score_cols = [f"p_{m}_{c}" for m in MODELS for c in CLASSES]
    scored = test[score_cols].notna().all(axis=1)
    n_unscored = int((~scored).sum())
    test = test[scored].reset_index(drop=True)
    derived: dict[str, np.ndarray] = {}
    for model in MODELS:
        probs = test[[f"p_{model}_{c}" for c in CLASSES]].to_numpy(dtype=float)
        derived[f"pred_{model}"] = np.array(CLASSES)[probs.argmax(axis=1)]
        derived[f"conf_{model}"] = probs.max(axis=1)
        derived[f"err_{model}"] = derived[f"pred_{model}"] != test["true"].to_numpy()
    test = pd.concat([test, pd.DataFrame(derived)], axis=1)
    return Release(name, root, results, report, pixel_counts, test, n_unscored)
