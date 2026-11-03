"""Phase 1 evaluation runner (ADR 0003 D4, ADR 0002 section 4, ADR 0004 D7.4).

Compares LightGBM (M-full, M-noind) with the Gaia DSC baselines on the test fold, in the
BALANCED regime (P0) and the REALISTIC regimes (P1..P3 plus the S sensitivity sweep).
All metric definitions come from ``eval.metrics`` / ``eval.reweight``; this module only
aligns the inputs, builds populations, runs the paired pixel bootstrap and writes reports.
"""

from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from astro_triage.split import build_pixel_map, pixel_l5
from eval import metrics as M
from eval.reweight import EmptyClassError

FloatArr = NDArray[np.float64]
BoolArr = NDArray[np.bool_]
StrArr = NDArray[np.str_]

CLS = M.CLASSES
BALANCED = "BALANCED"
REALISTIC = "REALISTIC"
VARIANTS = ("as_shipped", "prior_matched")
SOLVED_CODES = (3, 31, 95)
LGBM_MODELS = ("lgbm_full", "lgbm_noind")
DSC_COLUMN = {"STAR": "star", "GALAXY": "galaxy", "QSO": "quasar"}
SUM_TOLERANCE = 1e-3


def make_header(n_test: int, n_test_pixels: int) -> str:
    """Header built from the run itself, so a full run never claims to be the 100k sample."""
    return (
        f"PROVISIONAL: {n_test} test rows in {n_test_pixels} level-5 pixels, "
        "r_match 0.75 arcsec; cells below the reporting floor are marked n/a"
    )


GRID: dict[str, dict[str, float]] = {
    "P0": {c: 1.0 / len(CLS) for c in CLS},
    "P1": {"STAR": 0.997871, "QSO": 0.001996, "GALAXY": 0.000133},
    "P2": {"STAR": 0.998813, "QSO": 0.000989, "GALAXY": 0.000198},
    "P3": {"STAR": 0.996830, "QSO": 0.002640, "GALAXY": 0.000530},
}
GRID_SOURCE = {
    "P0": "uniform (= BALANCED)",
    "P1": "Bailer-Jones+2019 Sec. 4",
    "P2": "Delchambre+2023 Tab. 2, all sky (star derived)",
    "P3": "Delchambre+2023, |b| > 11.54 (star derived); bands B and C only",
}
S_QSO = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
S_GALAXY = (1e-4, 3e-4, 1e-3)
COMBMOD_PI_SRC = GRID["P2"]

G_BINS = np.arange(10.0, 23.0, 1.0)
ABSB_BINS = np.array([0.0, 11.54, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0001])

LIMITATIONS = (
    (
        "Selection by crossmatch: the matched class mix differs from the SDSS mix "
        "(match rate STAR 91.1%, QSO 65.7%, GALAXY 31.1%, measured on the 100k batch "
        "only and not recomputed for other releases); metrics "
        "describe the matched intersection, not the Gaia sky."
    ),
    (
        "DSC was trained partly on SDSS objects (DR14Q quasars, DR15 galaxies): partial "
        "circularity. ADR 0002 cut (b) (outside DSC training): not available: "
        "DR14Q/DR15 not ingested."
    ),
    (
        "Reporting floor (>= 100 objects in >= 20 test pixels per class) is not met by "
        "some cells; those are marked n/a (floor) and any raw value printed is "
        "below-floor, not for claims."
    ),
    (
        "Only Gaia sources with an SDSS spectroscopic match are labelled; r_match 0.75 arcsec "
        "on SDSS plug positions."
    ),
    "No DR14Q crossmatch.",
    (
        "Missing astrometry (astrometric_params_solved = 3) is a pattern linked to the "
        "class, so the missingness itself can carry class information (see M-noind)."
    ),
    (
        "combmod has 3 of its 5 classes available; its star score is 1 - p_quasar - "
        "p_galaxy (absorbs WD and binary), ADR 0003 'Effect of the combmod embedded prior' "
        "item 3. The delivered classprob_dsc_combmod_star column is not used."
    ),
    (
        "Prior shift of combmod uses pi_src = P2 (ADR 0003). allosmod has no pi_src "
        "defined in the ADRs: it is never shifted, so its as_shipped and prior_matched "
        "rows are identical."
    ),
    (
        "REALISTIC tables are class-prior corrected; within-class distribution = SDSS "
        "spectroscopic targets (selection bias not corrected)."
    ),
    "Sensitivity sweep S has point estimates only (no bootstrap), per ADR 0003.",
    (
        "LightGBM scores are the temperature-calibrated probabilities (p_cal_*) fitted on "
        "val; test was not used for any fitting."
    ),
)


class EvalError(RuntimeError):
    """Inputs are missing, inconsistent or violate the evaluation contract."""


# ------------------------------------------------------------------------- data model


@dataclass(frozen=True)
class ModelData:
    name: str
    proba: FloatArr  # (n_test, 3) in CLS order; NaN rows where the model has no output
    covered: BoolArr
    pi_src: dict[str, float] | None


@dataclass(frozen=True)
class TestSet:
    source_id: NDArray[np.int64]
    y: StrArr
    pixel: NDArray[np.int64]
    band: StrArr
    solved: NDArray[np.int64]
    ap_row_present: BoolArr
    g_mag: FloatArr
    abs_b: FloatArr
    models: dict[str, ModelData]
    combmod_star_col_max_abs_diff: float
    n_val_test_predictions: int


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise EvalError(f"{name}: missing columns {missing}")


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise EvalError(
            f"cannot read {path.name}: {exc}. Run scripts/build_dataset.py and "
            "scripts/train_lgbm.py first."
        ) from exc


def _lgbm_proba(pred: pd.DataFrame, name: str) -> FloatArr:
    cols = [f"p_cal_{c.lower()}" for c in CLS]
    _require_columns(pred, cols, name)
    p = np.column_stack([pred[c].to_numpy(dtype=np.float64) for c in cols])
    if not np.isfinite(p).all() or np.abs(p.sum(axis=1) - 1.0).max() > SUM_TOLERANCE:
        raise EvalError(f"{name}: probabilities are not finite or do not sum to 1")
    return p


def _dsc_columns(labels: pd.DataFrame, variant: str) -> FloatArr:
    cols = [f"classprob_dsc_{variant}_{DSC_COLUMN[c]}" for c in CLS]
    _require_columns(labels, cols, "labels")
    return np.column_stack([labels[c].to_numpy(dtype=np.float64) for c in cols])


def load_test_set(outdir: Path) -> TestSet:
    """Load and align every input on ``source_id``; raise ``EvalError`` on any mismatch."""
    labels = _read(outdir / "labels.parquet")
    features = _read(outdir / "features.parquet")
    _require_columns(
        labels, ["source_id", "sdss_class", "fold", "healpix_l5", "b"], "labels"
    )
    _require_columns(
        features,
        ["source_id", "astrometric_params_solved", "phot_g_mean_mag"],
        "features",
    )
    if not labels["source_id"].is_unique or not features["source_id"].is_unique:
        raise EvalError("source_id is not unique in labels or features")
    if not np.array_equal(
        np.sort(labels["source_id"].to_numpy()),
        np.sort(features["source_id"].to_numpy()),
    ):
        raise EvalError("labels and features do not hold the same source_id set")
    try:
        report = json.loads((outdir / "train_report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read train_report.json: {exc}") from exc
    pi_src_by_model = {
        m["model_id"]: m["train_class_fractions"] for m in report["models"]
    }
    if report.get("training_weights", "").split(" ")[0] != "none":
        raise EvalError("train_report.training_weights is not 'none'; pi_src undefined")

    merged = labels.merge(
        features[["source_id", "astrometric_params_solved", "phot_g_mean_mag"]],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    test = merged[merged["fold"] == "test"].reset_index(drop=True)
    vt_ids = set(merged.loc[merged["fold"].isin(["val", "test"]), "source_id"])
    y = test["sdss_class"].astype(str).to_numpy()
    if not set(np.unique(y)) <= set(CLS):
        raise EvalError(f"unexpected classes in test: {sorted(set(y) - set(CLS))}")

    sid = test["source_id"].to_numpy(dtype=np.int64)
    pixel = test["healpix_l5"].to_numpy(dtype=np.int64)
    if not np.array_equal(pixel_l5(sid), pixel):
        raise EvalError(
            "healpix_l5 in labels disagrees with source_id // 2**35 // 4**7"
        )
    pmap = build_pixel_map().set_index("healpix_l5")
    if (pmap["fold"].to_numpy()[pixel] != "test").any():
        raise EvalError(
            "test rows lie in pixels that the committed map assigns elsewhere"
        )
    band = pmap["lat_band"].to_numpy()[pixel].astype(str)

    models: dict[str, ModelData] = {}
    n_pred = 0
    for name in LGBM_MODELS:
        pred = _read(outdir / f"predictions_{name}_v0.parquet")
        _require_columns(pred, ["source_id"], name)
        if not pred["source_id"].is_unique or set(pred["source_id"]) != vt_ids:
            raise EvalError(f"{name}: predictions do not cover exactly val + test ids")
        n_pred = len(pred)
        aligned = pred.set_index("source_id").loc[sid].reset_index()
        models[name] = ModelData(
            name,
            _lgbm_proba(aligned, name),
            np.ones(len(test), dtype=bool),
            pi_src_by_model[name],
        )
        pi = pi_src_by_model[name]
        if set(pi) != set(CLS):
            raise EvalError(f"{name}: train_class_fractions keys {sorted(pi)}")

    allos = _dsc_columns(test, "allosmod")
    allos_cov = ~np.isnan(allos).any(axis=1)
    if np.abs(allos[allos_cov].sum(axis=1) - 1.0).max() > SUM_TOLERANCE:
        raise EvalError("allosmod probabilities do not sum to 1 on covered rows")
    models["dsc_allosmod"] = ModelData("dsc_allosmod", allos, allos_cov, None)

    comb = _dsc_columns(test, "combmod")
    comb_cov = ~np.isnan(comb).any(axis=1)
    star_col = comb[:, CLS.index("STAR")].copy()
    q = comb[:, CLS.index("QSO")]
    g = comb[:, CLS.index("GALAXY")]
    comb3 = comb.copy()
    comb3[:, CLS.index("STAR")] = np.clip(1.0 - q - g, 0.0, 1.0)
    diff = np.abs(comb3[comb_cov, CLS.index("STAR")] - star_col[comb_cov])
    models["dsc_combmod"] = ModelData(
        "dsc_combmod", comb3, comb_cov, dict(COMBMOD_PI_SRC)
    )

    solved = test["astrometric_params_solved"].to_numpy(dtype=np.float64)
    if not np.isin(solved, SOLVED_CODES).all():
        raise EvalError("astrometric_params_solved has values outside {3, 31, 95}")
    return TestSet(
        source_id=sid,
        y=y,
        pixel=pixel,
        band=band,
        solved=solved.astype(np.int64),
        ap_row_present=test["ap_row_present"].to_numpy(dtype=bool),
        g_mag=test["phot_g_mean_mag"].to_numpy(dtype=np.float64),
        abs_b=np.abs(test["b"].to_numpy(dtype=np.float64)),
        models=models,
        combmod_star_col_max_abs_diff=float(diff.max()) if diff.size else 0.0,
        n_val_test_predictions=n_pred,
    )


# ------------------------------------------------------------------ metric computation


def shift_prior(
    proba: FloatArr, pi_src: dict[str, float], pi_tgt: dict[str, float]
) -> FloatArr:
    """ADR 0003: p'(c|x) proportional to p(c|x) * pi'_c / pi_src_c, renormalised."""
    ratio = np.array([pi_tgt[c] / pi_src[c] for c in CLS])
    shifted = proba * ratio
    return shifted / shifted.sum(axis=1, keepdims=True)


def variant_proba(
    md_proba: FloatArr,
    pi_src: dict[str, float] | None,
    prior: dict[str, float],
    variant: str,
) -> FloatArr:
    if variant == "as_shipped" or pi_src is None:
        return md_proba
    return shift_prior(md_proba, pi_src, prior)


def metric_keys(with_ece: bool) -> list[str]:
    keys = [f"{m}/{c}" for m in M.BOOT_METRICS for c in CLS] + ["macro_f1"]
    if with_ece:
        keys += [f"ece/{c}" for c in CLS] + ["ece/mean"]
    return keys


def flat_metrics(
    y: StrArr,
    p: FloatArr,
    idx: NDArray[np.int64],
    prior: dict[str, float] | None,
    with_ece: bool,
) -> dict[str, float]:
    """Metrics of one (re)sample; every value is ``nan`` for an invalid replicate."""
    res = M.replicate_metrics(y, p, idx, prior)
    if res is None:
        return {k: math.nan for k in metric_keys(with_ece)}
    out = {f"{m}/{c}": float(res[m][c]) for m in M.BOOT_METRICS for c in CLS}
    out["macro_f1"] = float(np.mean([res["f1"][c] for c in CLS]))
    if with_ece:
        ys = y[idx]
        e = M.ece(ys, p[idx], M.regime_weights(ys, prior))
        vals = [e[c][0].value for c in CLS]
        out.update({f"ece/{c}": float(v) for c, v in zip(CLS, vals)})
        out["ece/mean"] = float(np.mean(vals))
    return out


@dataclass(frozen=True)
class BootTask:
    label: str
    y: StrArr
    pixel: NDArray[np.int64]
    band: StrArr
    natives: dict[str, FloatArr]
    pi_src: dict[str, dict[str, float] | None]
    prior_ids: tuple[str, ...]
    n_boot: int


def _key(model: str, variant: str, prior_id: str) -> str:
    return f"{model}|{variant}|{prior_id}"


def run_boot(task: BootTask) -> dict[str, dict]:
    """Paired pixel bootstrap: one set of replicates shared by every model/variant/prior."""
    probas: dict[str, FloatArr] = {}
    priors: dict[str, dict[str, float] | None] = {}
    with_ece: dict[str, bool] = {}
    for pid in task.prior_ids:
        prior = GRID[pid]
        for name, native in task.natives.items():
            for variant in VARIANTS:
                k = _key(name, variant, pid)
                probas[k] = variant_proba(native, task.pi_src[name], prior, variant)
                priors[k] = None if pid == "P0" else prior
                with_ece[k] = variant == "prior_matched"
    ident = np.arange(len(task.y))
    point = {
        k: flat_metrics(task.y, probas[k], ident, priors[k], with_ece[k])
        for k in probas
    }
    draws = {k: {m: np.full(task.n_boot, np.nan) for m in point[k]} for k in probas}
    rng = np.random.default_rng(M.SEED)
    for r in range(task.n_boot):
        idx = M.bootstrap_pixel_draw(task.pixel, task.band, rng)
        for k, p in probas.items():
            for m, v in flat_metrics(task.y, p, idx, priors[k], with_ece[k]).items():
                draws[k][m][r] = v
    return {"point": point, "draws": draws}


def make_cell(
    point: float, samples: FloatArr, floor_ok: bool, n_boot: int
) -> dict[str, object]:
    n_invalid = int((~np.isfinite(samples)).sum())
    cell: dict[str, object] = {
        "point": point,
        "lo": None,
        "hi": None,
        "n_invalid": n_invalid,
        "status": "ok",
    }
    if not floor_ok:
        cell["status"] = "n/a (floor)"
        return cell
    finite = samples[np.isfinite(samples)]
    if not math.isfinite(point):
        cell["status"] = "n/a (undefined)"
    elif n_invalid > M.MAX_INVALID_FRACTION * n_boot:
        cell["status"] = "n/a (invalid replicates)"
    else:
        lo, hi = np.percentile(finite, [2.5, 97.5])
        cell["lo"], cell["hi"] = float(lo), float(hi)
    return cell


def _metric_floor_ok(metric: str, floor: dict[str, tuple[int, int, bool]]) -> bool:
    cls = metric.split("/")[-1]
    return floor[cls][2] if cls in CLS else all(v[2] for v in floor.values())


def _detail(
    y: StrArr,
    p: FloatArr,
    pixel: NDArray[np.int64],
    prior: dict[str, float] | None,
    realistic: bool,
) -> dict[str, object]:
    pred = M.predict_argmax(p)
    counts = M.confusion_matrix(y, pred)
    out: dict[str, object] = {
        "accuracy": float(np.trace(counts) / counts.sum()),
        "n_predicted": {c: int((pred == c).sum()) for c in CLS},
        "confusion_counts": counts.astype(int).tolist(),
    }
    try:
        w = M.regime_weights(y, prior)
    except EmptyClassError as exc:
        out["weighted"] = f"n/a ({exc})"
        return out
    weighted = M.confusion_matrix(y, pred, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        row = weighted / weighted.sum(axis=1, keepdims=True)
        col = weighted / weighted.sum(axis=0, keepdims=True)
    cm = M.class_metrics(y, pred, pixel, w, 0, 0, flag_low_count=realistic)
    out["confusion_weighted"] = weighted.tolist()
    out["confusion_row_norm"] = row.tolist()
    out["confusion_col_norm"] = col.tolist()
    out["purity_flag"] = {c: cm[c]["purity_flag"] for c in CLS}
    out["contamination"] = {c: cm[c]["contamination"] for c in CLS}
    return out


def _s_sweep(
    y: StrArr,
    pixel: NDArray[np.int64],
    natives: dict[str, FloatArr],
    pi_src: dict[str, dict[str, float] | None],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, native in natives.items():
        for variant in VARIANTS:
            rows = []
            for qso in S_QSO:
                for gal in S_GALAXY:
                    prior = {"STAR": 1.0 - qso - gal, "QSO": qso, "GALAXY": gal}
                    p = variant_proba(native, pi_src[name], prior, variant)
                    pred = M.predict_argmax(p)
                    entry: dict[str, object] = {"qso_prior": qso, "galaxy_prior": gal}
                    try:
                        w = M.regime_weights(y, prior)
                    except EmptyClassError as exc:
                        entry["status"] = f"n/a ({exc})"
                        rows.append(entry)
                        continue
                    cm = M.class_metrics(y, pred, pixel, w, 0, 0, flag_low_count=True)
                    floor = M.floor_status(y, pixel)
                    entry["purity"] = {c: cm[c]["purity"] for c in CLS}
                    entry["purity_flag"] = {c: cm[c]["purity_flag"] for c in CLS}
                    entry["floor_ok"] = {c: floor[c][2] for c in CLS}
                    rows.append(entry)
            out[f"{name}|{variant}"] = {"points": rows}
    return out


# ----------------------------------------------------------------------- populations


@dataclass(frozen=True)
class Population:
    pop_id: str
    description: str
    rows: BoolArr
    models: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]


def build_populations(ts: TestSet) -> list[Population]:
    allos = ts.models["dsc_allosmod"].covered
    comb = ts.models["dsc_combmod"].covered
    full = np.ones(len(ts.y), dtype=bool)
    return [
        Population(
            "a_all",
            (
                "cut (a), all test rows; LightGBM only (DSC absent rows are outside the DSC "
                "evaluation, see coverage)"
            ),
            full,
            LGBM_MODELS,
            (("lgbm_full", "lgbm_noind"),),
        ),
        Population(
            "c_sdsc",
            (
                "cut (c), S_dsc: test rows with all three allosmod probabilities; every "
                "model on the same rows"
            ),
            allos,
            (*LGBM_MODELS, "dsc_allosmod"),
            (
                ("lgbm_full", "dsc_allosmod"),
                ("lgbm_noind", "dsc_allosmod"),
                ("lgbm_full", "lgbm_noind"),
            ),
        ),
        Population(
            "a_combmod",
            (
                "cut (a), DSC combmod rows: test rows with all three combmod probabilities; "
                "every model on the same rows"
            ),
            comb,
            (*LGBM_MODELS, "dsc_combmod"),
            (
                ("lgbm_full", "dsc_combmod"),
                ("lgbm_noind", "dsc_combmod"),
                ("lgbm_full", "lgbm_noind"),
            ),
        ),
    ]


def _subset_task(
    ts: TestSet,
    pop: Population,
    rows: BoolArr,
    prior_ids: tuple[str, ...],
    n_boot: int,
    label: str,
) -> BootTask:
    return BootTask(
        label=label,
        y=ts.y[rows],
        pixel=ts.pixel[rows],
        band=ts.band[rows],
        natives={m: ts.models[m].proba[rows] for m in pop.models},
        pi_src={m: ts.models[m].pi_src for m in pop.models},
        prior_ids=prior_ids,
        n_boot=n_boot,
    )


def _floor_json(
    floor: dict[str, tuple[int, int, bool]],
) -> dict[str, dict[str, object]]:
    return {c: {"n": n, "pixels": px, "passes": ok} for c, (n, px, ok) in floor.items()}


def assemble_population(
    ts: TestSet, pop: Population, boots: dict[str, dict], n_boot: int
) -> dict:
    """Turn raw bootstrap output into cells, paired differences and confusion details."""
    result: dict = {
        "description": pop.description,
        "n_rows": int(pop.rows.sum()),
        "models": list(pop.models),
        "n_by_class": {c: int(((ts.y == c) & pop.rows).sum()) for c in CLS},
        "priors": {},
    }
    sub_cache: dict[str, BoolArr] = {
        "all": pop.rows,
        "bc": pop.rows & (ts.band != "A"),
    }
    for pid, grid_prior in GRID.items():
        which = "bc" if pid == "P3" else "all"
        rows = sub_cache[which]
        boot = boots[f"{pop.pop_id}|{which}"]
        y, pixel = ts.y[rows], ts.pixel[rows]
        floor = M.floor_status(y, pixel)
        realistic = pid != "P0"
        prior = None if pid == "P0" else grid_prior
        block: dict = {
            "regime": REALISTIC if realistic else BALANCED,
            "prior": grid_prior,
            "prior_source": GRID_SOURCE[pid],
            "n_rows": int(rows.sum()),
            "floor": _floor_json(floor),
            "variants": {},
        }
        for variant in VARIANTS:
            models_out: dict = {}
            for name in pop.models:
                k = _key(name, variant, pid)
                pt, dr = boot["point"][k], boot["draws"][k]
                p = variant_proba(
                    ts.models[name].proba[rows],
                    ts.models[name].pi_src,
                    grid_prior,
                    variant,
                )
                models_out[name] = {
                    "metrics": {
                        m: make_cell(pt[m], dr[m], _metric_floor_ok(m, floor), n_boot)
                        for m in pt
                    },
                    "detail": _detail(y, p, pixel, prior, realistic),
                    "shifted": variant == "prior_matched"
                    and ts.models[name].pi_src is not None,
                }
            diffs: dict = {}
            for a, b in pop.pairs:
                ka, kb = _key(a, variant, pid), _key(b, variant, pid)
                keys = [m for m in boot["point"][ka] if m in boot["point"][kb]]
                diffs[f"{a} - {b}"] = {
                    m: make_cell(
                        boot["point"][ka][m] - boot["point"][kb][m],
                        boot["draws"][ka][m] - boot["draws"][kb][m],
                        _metric_floor_ok(m, floor),
                        n_boot,
                    )
                    for m in keys
                }
            block["variants"][variant] = {"models": models_out, "diffs": diffs}
        result["priors"][pid] = block
    rows = pop.rows
    result["s_sweep"] = _s_sweep(
        ts.y[rows],
        ts.pixel[rows],
        {m: ts.models[m].proba[rows] for m in pop.models},
        {m: ts.models[m].pi_src for m in pop.models},
    )
    return result


# ---------------------------------------------------------------------------- strata


def _acc_macro(y: StrArr, pred: StrArr, idx: NDArray[np.int64]) -> tuple[float, float]:
    yy, pp = y[idx], pred[idx]
    if len(yy) == 0:
        return math.nan, math.nan
    acc = float((yy == pp).mean())
    try:
        w = M.regime_weights(yy)
    except EmptyClassError:
        return acc, math.nan
    cm = M.class_metrics(yy, pp, np.zeros(len(yy), dtype=np.int64), w, 0, 0)
    return acc, M.macro_f1(cm)


@dataclass(frozen=True)
class StrataTask:
    y: StrArr
    pixel: NDArray[np.int64]
    band: StrArr
    preds: dict[str, StrArr]
    n_boot: int


def run_strata_boot(task: StrataTask) -> dict[str, dict]:
    ident = np.arange(len(task.y))
    point = {k: _acc_macro(task.y, v, ident) for k, v in task.preds.items()}
    draws = {k: np.full((task.n_boot, 2), np.nan) for k in task.preds}
    rng = np.random.default_rng(M.SEED)
    for r in range(task.n_boot if len(task.y) else 0):
        idx = M.bootstrap_pixel_draw(task.pixel, task.band, rng)
        for k, v in task.preds.items():
            draws[k][r] = _acc_macro(task.y, v, idx)
    return {"point": point, "draws": draws}


def _pair_cell(
    a: tuple[float, FloatArr], b: tuple[float, FloatArr], ok: bool, n: int
) -> dict:
    return make_cell(a[0] - b[0], a[1] - b[1], ok, n)


def assemble_stratum(
    ts: TestSet, code: int, boot: dict[str, dict], n_boot: int
) -> dict[str, object]:
    rows = ts.solved == code
    y, pixel = ts.y[rows], ts.pixel[rows]
    floor = M.floor_status(y, pixel)
    macro_ok = all(v[2] for v in floor.values())
    degenerate = not rows.any() or any((y == c).sum() == 0 for c in CLS)
    out: dict = {
        "n_rows": int(rows.sum()),
        "n_by_class": {c: int((y == c).sum()) for c in CLS},
        "floor": _floor_json(floor),
        "models": {},
        "diffs": {},
    }
    for variant in VARIANTS:
        for name in LGBM_MODELS:
            k = f"{name}|{variant}"
            pt, dr = boot["point"][k], boot["draws"][k]
            p = variant_proba(
                ts.models[name].proba[rows], ts.models[name].pi_src, GRID["P0"], variant
            )
            pred = M.predict_argmax(p)
            try:
                cm = M.class_metrics(y, pred, pixel, M.regime_weights(y), 0, 0)
            except (EmptyClassError, ValueError):  # missing class / empty stratum
                cm = {c: {"purity": None, "completeness": None} for c in CLS}
            out["models"][k] = {
                "accuracy": make_cell(pt[0], dr[:, 0], not degenerate, n_boot),
                "macro_f1": make_cell(pt[1], dr[:, 1], macro_ok, n_boot),
                "per_class": {
                    c: {
                        "purity": cm[c]["purity"],
                        "completeness": cm[c]["completeness"],
                    }
                    for c in CLS
                },
                "confusion_counts": M.confusion_matrix(y, pred).astype(int).tolist(),
            }
        ka, kb = f"lgbm_full|{variant}", f"lgbm_noind|{variant}"
        out["diffs"][f"lgbm_full - lgbm_noind|{variant}"] = {
            "accuracy": _pair_cell(
                (boot["point"][ka][0], boot["draws"][ka][:, 0]),
                (boot["point"][kb][0], boot["draws"][kb][:, 0]),
                not degenerate,
                n_boot,
            ),
            "macro_f1": _pair_cell(
                (boot["point"][ka][1], boot["draws"][ka][:, 1]),
                (boot["point"][kb][1], boot["draws"][kb][:, 1]),
                macro_ok,
                n_boot,
            ),
        }
    dsc: dict[str, object] = {}
    for name in ("dsc_allosmod", "dsc_combmod"):
        cov = rows & ts.models[name].covered
        yc, pc = ts.y[cov], ts.pixel[cov]
        fl = M.floor_status(yc, pc)
        entry: dict[str, object] = {
            "n_covered": int(cov.sum()),
            "coverage": float(cov.sum() / rows.sum()) if rows.sum() else None,
            "n_by_class": {c: int((yc == c).sum()) for c in CLS},
            "floor": _floor_json(fl),
        }
        if cov.sum():
            pred = M.predict_argmax(ts.models[name].proba[cov])
            entry["accuracy"] = float((pred == yc).mean())
            try:
                cm = M.class_metrics(yc, pred, pc, M.regime_weights(yc), 0, 0)
                entry["macro_f1_raw"] = M.macro_f1(cm)
                entry["floor_ok_macro_f1"] = all(v[2] for v in fl.values())
                entry["per_class"] = {
                    c: {
                        "purity": cm[c]["purity"],
                        "completeness": cm[c]["completeness"],
                    }
                    for c in CLS
                }
            except EmptyClassError as exc:
                entry["macro_f1_raw"] = f"n/a ({exc})"
            entry["confusion_counts"] = (
                M.confusion_matrix(yc, pred).astype(int).tolist()
            )
        dsc[name] = entry
    out["dsc"] = dsc
    return out


# -------------------------------------------------------------- coverage, histograms


def coverage_tables(ts: TestSet) -> dict[str, object]:
    layers = {
        "layer1_ap_row_present": ts.ap_row_present,
        "layer2_allosmod_non_null": ts.models["dsc_allosmod"].covered,
        "combmod_non_null": ts.models["dsc_combmod"].covered,
    }

    def table(rows: BoolArr) -> dict[str, object]:
        out: dict[str, object] = {"n": int(rows.sum())}
        for lname, mask in layers.items():
            out[lname] = {
                c: {
                    "n_covered": int((mask & rows & (ts.y == c)).sum()),
                    "n": int((rows & (ts.y == c)).sum()),
                }
                for c in (*CLS, "ALL")
                if c != "ALL"
            } | {"ALL": {"n_covered": int((mask & rows).sum()), "n": int(rows.sum())}}
        return out

    return {
        "overall": table(np.ones(len(ts.y), dtype=bool)),
        "by_solved": {str(c): table(ts.solved == c) for c in SOLVED_CODES},
    }


def histograms(ts: TestSet) -> dict[str, object]:
    out: dict = {
        "g_mag_edges": G_BINS.tolist(),
        "abs_b_edges": ABSB_BINS.tolist(),
        "g_mag": {},
        "abs_b": {},
    }
    for c in CLS:
        m = ts.y == c
        g = ts.g_mag[m]
        out["g_mag"][c] = {
            "counts": np.histogram(g[np.isfinite(g)], bins=G_BINS)[0].tolist(),
            "n_outside_or_nan": int(
                m.sum() - np.histogram(g[np.isfinite(g)], bins=G_BINS)[0].sum()
            ),
        }
        out["abs_b"][c] = {
            "counts": np.histogram(ts.abs_b[m], bins=ABSB_BINS)[0].tolist()
        }
    return out


# ---------------------------------------------------------------------------- driver


def _clean(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def evaluate(
    outdir: Path, n_boot: int = M.N_BOOT, workers: int = 4
) -> dict[str, object]:
    t0 = time.perf_counter()
    ts = load_test_set(outdir)
    pops = build_populations(ts)

    boot_tasks: list[BootTask] = []
    for pop in pops:
        boot_tasks.append(
            _subset_task(
                ts, pop, pop.rows, ("P0", "P1", "P2"), n_boot, f"{pop.pop_id}|all"
            )
        )
        boot_tasks.append(
            _subset_task(
                ts,
                pop,
                pop.rows & (ts.band != "A"),
                ("P3",),
                n_boot,
                f"{pop.pop_id}|bc",
            )
        )
    strata_tasks: list[tuple[int, StrataTask]] = []
    for code in SOLVED_CODES:
        rows = ts.solved == code
        preds = {}
        for name in LGBM_MODELS:
            for variant in VARIANTS:
                p = variant_proba(
                    ts.models[name].proba[rows],
                    ts.models[name].pi_src,
                    GRID["P0"],
                    variant,
                )
                preds[f"{name}|{variant}"] = M.predict_argmax(p)
        strata_tasks.append(
            (code, StrataTask(ts.y[rows], ts.pixel[rows], ts.band[rows], preds, n_boot))
        )

    t_boot = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            f_boot = [pool.submit(run_boot, t) for t in boot_tasks]
            f_str = [pool.submit(run_strata_boot, t) for _, t in strata_tasks]
            boots = {t.label: f.result() for t, f in zip(boot_tasks, f_boot)}
            strata_boots = [f.result() for f in f_str]
    else:
        boots = {t.label: run_boot(t) for t in boot_tasks}
        strata_boots = [run_strata_boot(t) for _, t in strata_tasks]
    boot_seconds = time.perf_counter() - t_boot

    populations = {p.pop_id: assemble_population(ts, p, boots, n_boot) for p in pops}
    strata = {
        str(code): assemble_stratum(ts, code, sb, n_boot)
        for (code, _), sb in zip(strata_tasks, strata_boots)
    }
    return {
        "meta": {
            "header": make_header(len(ts.y), int(np.unique(ts.pixel).size)),
            "n_test": len(ts.y),
            "n_by_class": {c: int((ts.y == c).sum()) for c in CLS},
            "n_test_pixels": int(np.unique(ts.pixel).size),
            "n_boot": n_boot,
            "seed": M.SEED,
            "floor": {"min_objects": M.MIN_OBJECTS, "min_pixels": M.MIN_PIXELS},
            "scores_used": "p_cal_* (temperature scaled on val) for LightGBM",
            "pi_src_lgbm": ts.models["lgbm_full"].pi_src,
            "pi_src_combmod": COMBMOD_PI_SRC,
            "pi_src_allosmod": None,
            "combmod_star_col_max_abs_diff_vs_1_minus_q_minus_g": ts.combmod_star_col_max_abs_diff,
            "n_val_plus_test_predictions": ts.n_val_test_predictions,
            "cut_b": "not available: DR14Q/DR15 not ingested",
            "bootstrap_workers": workers,
            "bootstrap_seconds": round(boot_seconds, 1),
            "total_seconds": round(time.perf_counter() - t0, 1),
        },
        "limitations": list(LIMITATIONS),
        "populations": populations,
        "strata": strata,
        "coverage": coverage_tables(ts),
        "histograms": histograms(ts),
    }


# ----------------------------------------------------------------------------- output


def _fmt_cell(cell: dict, flag: str | None = None, diff: bool = False) -> str:
    st, pt = cell["status"], cell["point"]
    raw = "nan" if pt is None or not math.isfinite(pt) else f"{pt:.3f}"
    if st == "ok":
        s = f"{raw} [{cell['lo']:.3f}, {cell['hi']:.3f}]"
        if diff and (cell["lo"] > 0 or cell["hi"] < 0):
            s += " CI excl. 0"
    elif st == "n/a (floor)":
        s = f"n/a (floor) raw {raw} below-floor, not for claims"
    else:
        s = f"{st} raw {raw}"
    return f"{s} {flag}" if flag else s


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return [*lines, ""]


def _metric_rows(
    cells: dict[str, dict],
    label: str,
    show_completeness: bool,
    flags: dict[str, str | None] | None = None,
    diff: bool = False,
) -> list[list[str]]:
    rows = []
    for m in ("purity", "completeness", "f1", "pr_auc", "ece"):
        if m == "completeness" and not show_completeness:
            continue
        cols = []
        for c in CLS:
            key = f"{m}/{c}"
            flag = flags.get(c) if flags and m == "purity" else None
            cols.append(_fmt_cell(cells[key], flag, diff) if key in cells else "-")
        macro = {"f1": "macro_f1", "ece": "ece/mean"}.get(m)
        cols.append(
            _fmt_cell(cells[macro], diff=diff) if macro and macro in cells else "-"
        )
        rows.append([f"{label} {m}", *cols])
    return rows


def render_markdown(res: dict) -> str:
    meta = res["meta"]
    L: list[str] = [f"# {meta['header']}", ""]
    L += [
        (
            "Phase 1 evaluation, test fold only. Numbers, intervals and flags; no ranking is "
            "claimed."
        ),
        "",
        "## Setup",
        "",
        (
            f"- test rows: {meta['n_test']} in {meta['n_test_pixels']} level-5 pixels; "
            f"by class {meta['n_by_class']}"
        ),
        (
            f"- bootstrap: pixel cluster within latitude bands, B={meta['n_boot']}, "
            f"seed {meta['seed']}, 95% percentile intervals, replicates shared by all models "
            "(paired)"
        ),
        (
            f"- bootstrap wall time {meta['bootstrap_seconds']} s with "
            f"{meta['bootstrap_workers']} worker process(es); total {meta['total_seconds']} s"
        ),
        (
            f"- floor: >= {meta['floor']['min_objects']} objects in >= "
            f"{meta['floor']['min_pixels']} pixels per class"
        ),
        (
            f"- LightGBM scores: {meta['scores_used']}; pi_src (train fractions) "
            f"{json.dumps(meta['pi_src_lgbm'])}"
        ),
        f"- ADR 0002 cut (b): {meta['cut_b']}",
        (
            "- variants: as_shipped = native argmax of each model; prior_matched = LightGBM and "
            "combmod shifted to the grid prior (ADR 0003), allosmod unshifted (no pi_src)"
        ),
        (
            "- completeness is identical across regimes for a fixed decision rule and is "
            "printed in the BALANCED (P0) tables only"
        ),
        "- ECE: 15 equal-mass bins, computed on prior_matched outputs only",
        "- low-count: REALISTIC purity with fewer than 10 contaminating objects",
        "",
        "## Limitations",
        "",
        *[f"- {x}" for x in res["limitations"]],
        "",
        "## Coverage of DSC outputs (test rows)",
        "",
    ]
    cov = res["coverage"]

    def cov_rows(tab: dict) -> list[list[str]]:
        rows = []
        for lname in (
            "layer1_ap_row_present",
            "layer2_allosmod_non_null",
            "combmod_non_null",
        ):
            cols = []
            for c in (*CLS, "ALL"):
                e = tab[lname][c]
                frac = e["n_covered"] / e["n"] if e["n"] else math.nan
                cols.append(f"{e['n_covered']}/{e['n']} ({frac:.3f})")
            rows.append([lname, *cols])
        return rows

    L += _md_table(["layer", *CLS, "ALL"], cov_rows(cov["overall"]))
    for code, tab in cov["by_solved"].items():
        L += [f"Solved = {code} (n={tab['n']})", ""]
        L += _md_table(["layer", *CLS, "ALL"], cov_rows(tab))

    for pop_id, pop in res["populations"].items():
        L += [f"## Population {pop_id}", "", pop["description"], ""]
        L += [f"n rows {pop['n_rows']}; by class {pop['n_by_class']}", ""]
        for pid, blk in pop["priors"].items():
            L += [
                f"### {pop_id} / {pid} ({blk['regime']}; {blk['prior_source']})",
                "",
                f"prior {json.dumps(blk['prior'])}; evaluated rows {blk['n_rows']}",
                "",
                "Floor status: "
                + "; ".join(
                    (
                        f"{c} n={v['n']} pixels={v['pixels']} "
                        f"{'ok' if v['passes'] else 'n/a (floor)'}"
                        for c, v in blk["floor"].items()
                    )
                ),
                "",
            ]
            if blk["regime"] == REALISTIC:
                L += [
                    (
                        "Class-prior corrected; within-class distribution = SDSS spectroscopic "
                        "targets."
                    ),
                    "",
                ]
            for variant, vb in blk["variants"].items():
                L += [f"#### {pop_id} / {pid} / {variant}", ""]
                rows: list[list[str]] = []
                for name, mo in vb["models"].items():
                    flags = {c: mo["detail"].get("purity_flag", {}).get(c) for c in CLS}
                    rows += _metric_rows(mo["metrics"], name, pid == "P0", flags)
                L += _md_table(["model metric", *CLS, "macro"], rows)
                L += [f"Paired differences ({variant}), first minus second:", ""]
                drows: list[list[str]] = []
                for pair, cells in vb["diffs"].items():
                    drows += _metric_rows(cells, pair, pid == "P0", diff=True)
                L += _md_table(["pair metric", *CLS, "macro"], drows)
                crow = []
                for name, mo in vb["models"].items():
                    d = mo["detail"]
                    cm = d["confusion_counts"]
                    crow.append(
                        [
                            name,
                            f"{d['accuracy']:.4f}",
                            json.dumps(d["n_predicted"]),
                            json.dumps([[int(v) for v in r] for r in cm]),
                        ]
                    )
                L += _md_table(
                    [
                        "model",
                        "raw accuracy",
                        "n predicted",
                        "raw confusion (rows true STAR,GALAXY,QSO)",
                    ],
                    crow,
                )
                if blk["regime"] == REALISTIC:
                    ct = []
                    for name, mo in vb["models"].items():
                        for c, per in mo["detail"].get("contamination", {}).items():
                            txt = "; ".join(
                                f"{k}: {v['count']}"
                                + (
                                    f" (upper bound 3/n_k = {v['rate']:.4f})"
                                    if v["upper_bound"]
                                    else (
                                        f" (rate {v['rate']:.4f})"
                                        if v["rate"] is not None
                                        else ""
                                    )
                                )
                                for k, v in per.items()
                            )
                            ct.append([name, f"predicted {c}", txt])
                    L += ["Contaminating objects (raw counts):", ""]
                    L += _md_table(["model", "column", "true class: count"], ct)
        L += [
            f"### {pop_id} / S sweep (point estimates, purity by QSO and galaxy prior)",
            "",
        ]
        srows = []
        for mv, sw in pop["s_sweep"].items():
            for pt in sw["points"]:
                if "purity" not in pt:
                    continue
                srows.append(
                    [
                        mv,
                        f"{pt['qso_prior']:g}",
                        f"{pt['galaxy_prior']:g}",
                        *[
                            _f3(pt["purity"][c])
                            + (
                                ""
                                if pt["floor_ok"][c]
                                else " below-floor, not for claims"
                            )
                            + (
                                f" {pt['purity_flag'][c]}"
                                if pt["purity_flag"][c]
                                else ""
                            )
                            for c in CLS
                        ],
                    ]
                )
        L += _md_table(["model|variant", "qso prior", "galaxy prior", *CLS], srows)

    L += ["## Stratification by astrometric_params_solved (BALANCED, P0)", ""]
    for code, st in res["strata"].items():
        L += [
            f"### Solved = {code}",
            "",
            f"n {st['n_rows']}; by class {st['n_by_class']}; floor "
            + "; ".join(
                f"{c} n={v['n']} pixels={v['pixels']} {'ok' if v['passes'] else 'n/a (floor)'}"
                for c, v in st["floor"].items()
            ),
            "",
        ]
        rows = []
        for k, mo in st["models"].items():
            rows.append(
                [
                    k,
                    _fmt_cell(mo["accuracy"]),
                    _fmt_cell(mo["macro_f1"]),
                    json.dumps(
                        {
                            c: [
                                _r(mo["per_class"][c]["purity"]),
                                _r(mo["per_class"][c]["completeness"]),
                            ]
                            for c in CLS
                        }
                    ),
                    json.dumps(mo["confusion_counts"]),
                ]
            )
        for name, d in st["dsc"].items():
            if "accuracy" in d:
                rows.append(
                    [
                        f"{name} (covered rows only, n={d['n_covered']})",
                        f"{d['accuracy']:.3f} (no CI)",
                        _raw_macro(d),
                        json.dumps(
                            {
                                c: [
                                    _r(d["per_class"][c]["purity"]),
                                    _r(d["per_class"][c]["completeness"]),
                                ]
                                for c in CLS
                            }
                        )
                        if "per_class" in d
                        else "-",
                        json.dumps(d["confusion_counts"]),
                    ]
                )
        L += _md_table(
            [
                "model|variant",
                "accuracy",
                "macro-F1",
                "[purity, completeness] per class",
                "raw confusion",
            ],
            rows,
        )
        drows = [
            [
                k,
                _fmt_cell(v["accuracy"], diff=True),
                _fmt_cell(v["macro_f1"], diff=True),
            ]
            for k, v in st["diffs"].items()
        ]
        L += ["Paired differences:", ""]
        L += _md_table(["pair|variant", "accuracy", "macro-F1"], drows)
        dc = [
            [
                n,
                str(d["n_covered"]),
                f"{d['coverage']:.3f}" if d["coverage"] is not None else "n/a",
                json.dumps(d["n_by_class"]),
            ]
            for n, d in st["dsc"].items()
        ]
        L += ["DSC coverage in this stratum:", ""]
        L += _md_table(["baseline", "covered", "coverage", "covered n by class"], dc)

    L += ["## Test-set class distributions (selection-bias documentation)", ""]
    h = res["histograms"]
    for key, edges in (("g_mag", h["g_mag_edges"]), ("abs_b", h["abs_b_edges"])):
        names = [f"[{a:g},{b:g})" for a, b in pairwise(edges)]
        L += [f"Histogram of {'G magnitude' if key == 'g_mag' else '|b| (deg)'}:", ""]
        L += _md_table(
            ["class", *names],
            [[c, *[str(v) for v in h[key][c]["counts"]]] for c in CLS],
        )

    L += [
        "## What this does NOT show",
        "",
        (
            "- Selection by crossmatch: matching alters the class mix (match rate STAR 91% vs "
            "GALAXY 31%, measured on the 100k batch), so the test mix is not the SDSS mix and "
            "not the Gaia sky."
        ),
        (
            "- The DSC was trained partly on SDSS objects (partial circularity); the cut outside "
            "DSC training (ADR 0002 cut b) is not available: DR14Q/DR15 not ingested."
        ),
        (
            "- Cells below the reporting floor carry no claim; raw values there are marked "
            "below-floor, not for claims."
        ),
        "- r_match 0.75 arcsec on SDSS plug positions; no DR14Q crossmatch.",
        (
            "- Missing astrometry is a pattern linked to the class, so a model can read class "
            "information from it; the M-full / M-noind rows measure this and nothing more."
        ),
        (
            "- No causal statement: differences are descriptive numbers with pixel-bootstrap "
            "intervals."
        ),
        "",
    ]
    return "\n".join(L)


def _f3(x: float | None) -> str:
    return "undefined" if x is None else f"{x:.3f}"


def _r(x: object) -> object:
    return round(x, 3) if isinstance(x, float) and math.isfinite(x) else None


def _raw_macro(d: dict) -> str:
    v = d.get("macro_f1_raw")
    if not isinstance(v, float):
        return str(v)
    tag = "" if d.get("floor_ok_macro_f1") else " below-floor, not for claims"
    return f"{v:.3f} (no CI){tag}"


def write_outputs(res: dict, outdir: Path) -> list[Path]:
    evaldir = outdir / "eval"
    evaldir.mkdir(parents=True, exist_ok=True)
    paths = [
        evaldir / n
        for n in (
            "results.json",
            "results.md",
            "confusion_overall.csv",
            "confusion_by_solved.csv",
        )
    ]
    paths[0].write_text(json.dumps(_clean(res), indent=1), encoding="utf-8")
    paths[1].write_text(render_markdown(_clean(res)), encoding="utf-8")

    with paths[2].open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "population",
                "prior",
                "variant",
                "model",
                "true",
                "pred",
                "count",
                "weighted",
                "row_norm",
                "col_norm",
            ]
        )
        for pop_id, pop in res["populations"].items():
            for pid, blk in pop["priors"].items():
                for variant, vb in blk["variants"].items():
                    for name, mo in vb["models"].items():
                        d = mo["detail"]
                        if "confusion_weighted" not in d:
                            continue
                        for i, t in enumerate(CLS):
                            for j, pr in enumerate(CLS):
                                w.writerow(
                                    [
                                        pop_id,
                                        pid,
                                        variant,
                                        name,
                                        t,
                                        pr,
                                        int(d["confusion_counts"][i][j]),
                                        d["confusion_weighted"][i][j],
                                        d["confusion_row_norm"][i][j],
                                        d["confusion_col_norm"][i][j],
                                    ]
                                )
    with paths[3].open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["solved", "model_variant", "true", "pred", "count"])
        for code, st in res["strata"].items():
            blocks = dict(st["models"])
            blocks.update(
                {k: v for k, v in st["dsc"].items() if "confusion_counts" in v}
            )
            for k, mo in blocks.items():
                for i, t in enumerate(CLS):
                    for j, pr in enumerate(CLS):
                        w.writerow([code, k, t, pr, int(mo["confusion_counts"][i][j])])
    return paths
