"""Tables read straight from ``results.json``: metrics, paired differences, population sizes."""

from __future__ import annotations

import math

import pandas as pd

from astro_triage.dashboard.data_release import (
    Release,
)


def feature_importance(rel: Release, model: str) -> pd.DataFrame:
    """Top-10 gain shares from the training report (feature, share)."""
    for entry in rel.train_report["models"]:
        if entry["model_id"] == model:
            return pd.DataFrame(entry["gain_top10"])[["feature", "share"]]
    raise KeyError(model)


def model_summary(rel: Release, model: str) -> dict:
    """Training facts for one model, straight from the report."""
    entry = next(m for m in rel.train_report["models"] if m["model_id"] == model)
    cal = entry["val_calibration_fitted_here"]
    return {
        "n_features": entry["n_features"],
        "best_iteration": entry["best_iteration"],
        "temperature": entry["temperature"],
        "val_accuracy": cal.get("accuracy"),
        "val_macro_f1": cal.get("macro_f1"),
    }


def frame_from_cells(cells: dict, extra: dict) -> pd.DataFrame:
    rows = []
    for metric, cell in cells.items():
        usable = cell.get("status") == "ok"
        rows.append(
            {
                **extra,
                "metric": metric,
                "point": cell["point"] if usable else math.nan,
                "lo": cell["lo"] if usable else math.nan,
                "hi": cell["hi"] if usable else math.nan,
                "usable": usable,
                "status": cell.get("status"),
            }
        )
    return pd.DataFrame(rows)


def results_variant(rel: Release, population: str, prior: str, variant: str) -> dict:
    return rel.results["populations"][population]["priors"][prior]["variants"][variant]


def metrics_frame(
    rel: Release,
    population: str = "c_sdsc",
    prior: str = "P0",
    variant: str = "prior_matched",
) -> pd.DataFrame:
    """Long table (model, metric, point, lo, hi, usable) from results.json."""
    models = results_variant(rel, population, prior, variant)["models"]
    return pd.concat(
        [frame_from_cells(m["metrics"], {"model": name}) for name, m in models.items()],
        ignore_index=True,
    )


def paired_diffs_frame(
    rel: Release,
    population: str = "c_sdsc",
    prior: str = "P0",
    variant: str = "prior_matched",
) -> pd.DataFrame:
    """Long table (pair, metric, point, lo, hi, usable) of the paired bootstrap differences."""
    diffs = results_variant(rel, population, prior, variant)["diffs"]
    return pd.concat(
        [frame_from_cells(cells, {"pair": pair}) for pair, cells in diffs.items()],
        ignore_index=True,
    )


def population_rows(rel: Release, population: str) -> int:
    return int(rel.results["populations"][population]["n_rows"])
