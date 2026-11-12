"""Descriptive test metrics of the DL comparison (ADR 0006 4.2, spec S13). No verdict is made here.

Every metric is computed per system and seed on the frozen ``p_cal`` and then averaged over seeds as a
METRIC (never as a probability average). Macro-F1, accuracy, per-class metrics, ECE and ``e_sq`` use the
BALANCED regime with the prior-matched probabilities (``runner.variant_proba``); the risk-coverage curve
uses the shipped ``p_cal`` (confidence = max p_cal).
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from astro_triage.evaluation.dl_compare import N_SEEDS
from astro_triage.evaluation.dl_eval import refuse
from astro_triage.evaluation.runner import (
    GRID,
    FloatArr,
    ModelData,
    StrArr,
    TestSet,
    flat_metrics,
    variant_proba,
)
from astro_triage.models.lgbm import _git_sha
from eval import metrics as M

REPORT_NAME: Final = "train_report_dl.json"
TAUS: Final = tuple(round(0.50 + 0.01 * i, 2) for i in range(50))  # ADR 0006 4.2: 0.50..0.99


def e_sq(y: StrArr, pred: StrArr, idx: npt.NDArray[np.int64]) -> float:
    """(R[STAR->QSO] + R[QSO->STAR]) / 2 with R the row-normalised confusion rate; nan if a class is absent."""
    cm = M.confusion_matrix(y[idx], pred[idx])
    s, q = M.CLASSES.index("STAR"), M.CLASSES.index("QSO")
    n_s, n_q = cm[s].sum(), cm[q].sum()
    if n_s == 0 or n_q == 0:
        return math.nan
    return float((cm[s, q] / n_s + cm[q, s] / n_q) / 2)


def risk_coverage_curve(y: StrArr, proba: FloatArr) -> list[dict[str, float]]:
    """Raw selective accuracy over the tau grid; confidence = max p_cal."""
    conf = proba.max(axis=1)
    hit = M.predict_argmax(proba) == y
    rows = []
    for tau in TAUS:
        kept = conf >= tau
        rows.append(
            {
                "tau": tau,
                "coverage": float(kept.mean()),
                "accuracy": float(hit[kept].mean()) if kept.any() else math.nan,
            }
        )
    return rows


def load_costs(dl_dir: Path) -> dict[str, object]:
    """Costs copied verbatim from ``train_report_dl.json`` (model id -> costs dict)."""
    path = dl_dir / REPORT_NAME
    try:
        fits = json.loads(path.read_text(encoding="utf-8"))["fits"]
    except (OSError, ValueError, KeyError) as exc:
        raise refuse(f"cannot read costs from {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(fits, dict):
        raise refuse(f"{path}: fits is not an object")
    return {
        k: v["costs"] for k, v in fits.items() if isinstance(v, dict) and "costs" in v
    }


def _one(ts: TestSet, md: ModelData) -> tuple[dict[str, float], list[dict[str, float]]]:
    p = variant_proba(md.proba, md.pi_src, GRID["P0"], "prior_matched")
    idx = np.arange(len(ts.y))
    out = flat_metrics(ts.y, p, idx, None, True)
    pred = M.predict_argmax(p)
    out["accuracy"] = float((pred == ts.y).mean())
    out["e_sq"] = e_sq(ts.y, pred, idx)
    return out, risk_coverage_curve(ts.y, md.proba)


def _family(name: str) -> str:
    return name.rsplit("_s", 1)[0]


def describe(
    ts: TestSet,
    systems: dict[str, ModelData],
    costs: dict[str, object],
    *,
    workers: int = 1,
) -> tuple[dict[str, object], list[list[object]]]:
    """(descriptive section, risk-coverage CSV rows ``system, seed, tau, coverage, accuracy``)."""
    names = sorted(systems)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        done = list(pool.map(lambda n: _one(ts, systems[n]), names))
    per = {n: d[0] for n, d in zip(names, done, strict=True)}
    curves = {n: d[1] for n, d in zip(names, done, strict=True)}
    fams: dict[str, dict[str, dict[str, float]]] = {}
    rows: list[list[object]] = []
    for fam in sorted({_family(n) for n in names}):
        seeds = [f"{fam}_s{k}" for k in range(N_SEEDS)]
        fams[fam] = {
            key: {
                "mean": float(np.mean([per[s][key] for s in seeds])),
                "min": float(np.min([per[s][key] for s in seeds])),
                "max": float(np.max([per[s][key] for s in seeds])),
            }
            for key in per[seeds[0]]
        }
        for i, tau in enumerate(TAUS):
            rows.append([fam, "mean", tau, *_mean_point(curves, seeds, i)])
    for n in names:
        rows += [[n, n.rsplit("_s", 1)[1], c["tau"], c["coverage"], c["accuracy"]] for c in curves[n]]
    floor = M.floor_status(ts.y, ts.pixel)
    desc: dict[str, object] = {
        "regime": "BALANCED, prior_matched; accuracy and risk-coverage are raw, unweighted",
        "systems": per,
        "families": fams,
        "floor": {c: {"n": f[0], "pixels": f[1], "ok": f[2]} for c, f in floor.items()},
        "costs": {n: costs[n] for n in names if n in costs},
    }
    return desc, rows


def _mean_point(
    curves: dict[str, list[dict[str, float]]], seeds: list[str], i: int
) -> list[float]:
    """Seed mean of coverage and accuracy at grid point ``i`` (metric mean, not probability mean)."""
    return [float(np.mean([curves[s][i][k] for s in seeds])) for k in ("coverage", "accuracy")]


def assemble(
    ts: TestSet,
    compare: dict[str, object],
    desc: dict[str, object],
    freeze_sha256: str,
    approval: str,
) -> dict[str, object]:
    """The result document: run metadata, the bootstrap comparison and the descriptive section."""
    meta = {
        "n_test": len(ts.y),
        "n_test_pixels": int(np.unique(ts.pixel).size),
        "freeze_sha256": freeze_sha256,
        "git_sha": _git_sha(),
        "approval": approval,
    }
    return {"meta": meta, "compare": compare, "descriptive": desc}
