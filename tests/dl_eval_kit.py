"""Shared fixtures for the dl_eval tests (S11): a frozen synthetic environment, no real test data."""

from __future__ import annotations

import json
from pathlib import Path

from test_dl_val_freeze import APPROVAL, make_env

from astro_triage.evaluation import dl_eval as E
from astro_triage.evaluation import dl_val as V
from astro_triage.evaluation.dl_compare import (
    COMPLETENESS_LEVELS,
    FEATURE_SETS,
    N_SEEDS,
    PRIMARY_PRIOR,
    TARGET_CLASSES,
)


def write_freeze(dl: Path, doc: dict) -> Path:
    path = dl / E.FREEZE_NAME
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def write_real_delta(out: Path, lg: Path) -> dict:
    """Replace the stub delta.json by one with real cells delta[fs][cls][c][PRIMARY_PRIOR].

    Shape as compute_delta writes it (delta, v_k x5, thresholds x5 per seed, val_checks_pass); the
    thresholds are those V.val_purity_summary fits on the seed files, i.e. what build_freeze freezes.
    """
    path = lg / "delta.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    cells: dict = {}
    for fs in FEATURE_SETS:
        summ = [V.val_purity_summary(out, lg / f"predictions_lgbm_{fs}_s{k}_v0.parquet") for k in range(N_SEEDS)]
        cells[fs] = {
            cls: {
                f"{c:.2f}": {
                    PRIMARY_PRIOR: {
                        "delta": 0.02,
                        "v_k": [0.9 + 0.01 * k for k in range(N_SEEDS)],
                        "val_checks_pass": True,
                        "thresholds": [s[cls][f"{c:.2f}"]["threshold"] for s in summ],
                    }
                }
                for c in COMPLETENESS_LEVELS
            }
            for cls in TARGET_CLASSES
        }
    doc["delta"] = cells
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


def frozen_env(root: Path, real_delta: bool = True):
    """(out, dl, lg, doc): 40 synthetic systems, delta.json and a freeze written by build_freeze.

    ``real_delta`` swaps the sha-only stub delta.json for one with real cells (before the freeze)."""
    out, dl, lg = make_env(root)
    if real_delta:
        write_real_delta(out, lg)
    doc = V.build_freeze(out, dl, lg, APPROVAL)
    write_freeze(dl, doc)
    return out, dl, lg, doc


def read_freeze(dl: Path) -> dict:
    return json.loads((dl / E.FREEZE_NAME).read_text(encoding="utf-8"))


def edit_freeze(dl: Path, fn) -> None:
    doc = read_freeze(dl)
    fn(doc)
    write_freeze(dl, doc)
