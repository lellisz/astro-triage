"""graficos.json: LightGBM reliability, error rates by G and z, and the confusion matrix.

Reshapes ``dashboard.data_charts`` into the JSON of ADR 0008 (A1.8). The floor is applied here:
a bin or cell under it has null values, and in the confusion cells ``n`` is null too.
``ece`` and ``cmd`` stay null until their own tasks exist.
"""

from __future__ import annotations

import math
from typing import Final

import pandas as pd

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_charts
from astro_triage.dashboard.charts_base import MODEL_PT
from astro_triage.dashboard.data_release import CLASSES
from astro_triage.dashboard.html_dl import class_label
from astro_triage.dashboard.html_galaxies import SMALL_RATE
from astro_triage.web_export import web_text
from astro_triage.web_export.types import est, head, num

PAGE: Final = "graficos"
MODEL: Final = "lgbm_full"
Z_CLASSES: Final = ("GALAXY", "QSO")


def _finite(x: object) -> float | None:
    """``x`` as a float, or None when it is missing or not finite."""
    if x is None:
        return None
    value = float(x)  # type: ignore[arg-type]
    return value if math.isfinite(value) else None


def _rate(rate: float, lo: float, hi: float) -> dict:
    """Rate with two digits when it is small, one otherwise (same rule as the Galáxias page)."""
    digits = 2 if rate < SMALL_RATE else 1
    return est(rate, lo, hi, digits=digits)


def _cal_bin(r: pd.Series, floor: int) -> dict:
    below = int(r["n"]) < floor
    if below:
        return {
            "p_mean": num(None),
            "freq": est(None, None, None),
            "n": num(int(r["n"])),
            "below_floor": True,
            "raw_p_mean": None,
            "raw_freq": None,
        }
    return {
        "p_mean": num(r["mean_pred"], kind="pct"),
        "freq": est(r["observed"], r["ci_lo"], r["ci_hi"]),
        "n": num(int(r["n"])),
        "below_floor": False,
        "raw_p_mean": _finite(r.get("raw_mean_pred")),
        "raw_freq": _finite(r.get("raw_observed")),
    }


def _calibration(rel: D.Release, floor: int) -> dict:
    series = [
        {
            "cls": cls,
            "label": class_label(cls),
            "bins": [_cal_bin(r, floor) for _, r in frame.iterrows()],
        }
        for cls, frame in data_charts.calibration_bins(rel, MODEL).items()
    ]
    return {
        "n_bins": data_charts.CALIBRATION_BINS,
        "networks_note": web_text.NETWORKS_ABSENT,
        "series": series,
    }


def _rate_row(r: pd.Series, floor: int) -> dict:
    below = int(r["n"]) < floor
    return {
        "label": str(r["label"]),
        "lo": _finite(r["lo"]),
        "hi": _finite(r["hi"]),
        "n": num(int(r["n"])),
        "errors": num(None if below else int(r["errors"])),
        "rate": est(None, None, None)
        if below
        else _rate(r["rate"], r["ci_lo"], r["ci_hi"]),
        "below_floor": below,
    }


def _rate_series(tables: dict[str, pd.DataFrame], floor: int) -> list[dict]:
    """One series per class: rows with a defined position, plus the "sem medida" count."""
    out = []
    for cls, table in tables.items():
        placed = table["lo"].notna()
        out.append(
            {
                "cls": cls,
                "label": class_label(cls),
                "rows": [_rate_row(r, floor) for _, r in table[placed].iterrows()],
                "missing": num(int(table.loc[~placed, "n"].sum())),
            }
        )
    return out


def _confusion(rel: D.Release, floor: int) -> dict:
    cells = data_charts.confusion_cells(rel, MODEL)
    rows = []
    for true in CLASSES:
        sub = cells[cells["true"] == true]
        row_cells = []
        for r in sub.itertuples(index=False):
            below = int(r.n) < floor
            row_cells.append(
                {
                    "pred": str(r.pred),
                    "n": num(None if below else int(r.n)),
                    "share": est(None, None, None)
                    if below
                    else _rate(r.share, r.ci_lo, r.ci_hi),
                    "diagonal": r.pred == true,
                    "below_floor": below,
                }
            )
        rows.append({"true": true, "n": num(int(sub["row_n"].iloc[0])), "cells": row_cells})
    return {
        "classes": [{"cls": c, "label": class_label(c)} for c in CLASSES],
        "rows": rows,
    }


def graficos(rel: D.Release) -> dict:
    """graficos.json of a release; every number comes from ``data_charts``."""
    floor = int(rel.meta["floor"]["min_objects"])
    by_g = _rate_series(data_charts.error_by(rel, MODEL, "G"), floor)
    by_z = _rate_series(data_charts.error_by(rel, MODEL, "z"), floor)
    model_label = MODEL_PT[MODEL]
    return {
        **head(rel.name, PAGE),
        "floor": floor,
        "model_label": model_label,
        "scope_text": web_text.charts_scope(len(rel.test), model_label),
        "floor_text": web_text.charts_floor_note(floor),
        "calibration": _calibration(rel, floor),
        "error_by_g": {
            "missing_note": web_text.missing_g_note(
                {s["cls"]: s["missing"]["value"] for s in by_g}
            ),
            "series": by_g,
        },
        "error_by_z": {"star_note": web_text.STAR_Z_NOTE, "series": by_z},
        "confusion": _confusion(rel, floor),
        "ece": None,
        "cmd": None,
    }
