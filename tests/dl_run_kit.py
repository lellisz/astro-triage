"""Synthetic world for run_compare tests (S12): 40 systems built from shared per-seed noise."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np

from astro_triage.evaluation import dl_run, dl_val
from astro_triage.evaluation.dl_compare import TARGET_CLASSES
from astro_triage.evaluation.runner import ModelData
from eval import metrics as M

WEAK = 0.12
WIN = 1.2  # clear win over WEAK with enough contaminants (not low-count); purities are tiny (priors)
STRONG = 50.0  # near-perfect separation: purity 1, no contaminants


def world(n: int = 4000, counts: tuple[int, int, int] | None = None, seed: int = 0):
    """(ts, noise): ts has y/band/pixel; ``counts`` fixes (STAR, GALAXY, QSO) row counts."""
    rng = np.random.default_rng(seed)
    pix = rng.integers(0, 400, n).astype(np.int64)
    if counts is None:
        y = rng.choice(np.asarray(M.CLASSES), n, p=[0.5, 0.25, 0.25])
    else:
        y = np.repeat(np.asarray(M.CLASSES), counts)
        pix = rng.integers(0, 400, y.size).astype(np.int64)
        n = y.size
    band = np.array(["A", "B", "C"])[pix % 3]
    ts = types.SimpleNamespace(y=y, band=band, pixel=pix)
    noise = [rng.dirichlet([1, 1, 1], n) for _ in range(dl_val.N_SEEDS)]
    return ts, noise


def proba(ts, noise_k, shifts: dict[str, float]) -> np.ndarray:
    p = noise_k.copy()
    for i, c in enumerate(M.CLASSES):
        p[ts.y == c, i] += shifts.get(c, WEAK)
    return p / p.sum(1, keepdims=True)


def env(tmp: Path, ts, noise, shifts=None, delta=3e-4, checks=True):
    """(systems, freeze, lgbm_dir). ``shifts``: family (e.g. 'mlp_full') -> {class: shift}."""
    shifts = shifts or {}
    systems, thr = {}, {}
    for name in dl_val.planned_systems():
        fam, k = name.rsplit("_s", 1)
        pr = proba(ts, noise[int(k)], shifts.get(fam, {}))
        systems[name] = ModelData(name, pr, np.ones(len(ts.y), bool), None)
        thr[name] = {
            c: {
                cs: float(np.quantile(pr[ts.y == c, M.CLASSES.index(c)], 1 - float(cs)))
                for cs in dl_run.CSTARS
            }
            for c in TARGET_CLASSES
        }
    cell = {"delta": delta, "val_checks_pass": checks}
    cells = {
        fs: {
            c: {cs: {p: cell for p in ("P1", "P2", "P3")} for cs in dl_run.CSTARS}
            for c in TARGET_CLASSES
        }
        for fs in ("full", "noind")
    }
    (tmp / "delta.json").write_text(json.dumps({"delta": cells}))
    freeze = {"thresholds": thr, "delta_sha256": dl_val.sha256_file(tmp / "delta.json")}
    return systems, freeze, tmp


def run(tmp, ts, noise, n_boot=20, **kw):
    systems, freeze, d = env(tmp, ts, noise, **kw)
    return dl_run.run_compare(ts, systems, freeze, lgbm_dir=d, n_boot=n_boot)


def cell(res, pair, p="P2", cs="0.80"):
    return res["pairs"][pair][p][cs]
