"""Data layer of the dashboard: reads one release and computes every number the screens show.

No Streamlit import here, so everything is testable without a browser. Nothing on the
screens is typed by hand: counts, rates, intervals and sentences all come from these functions.

Conventions that matter:
- Classes are always mapped by NAME (``p_cal_star`` ...), never by position: the prediction
  parquet orders its columns star, qso, galaxy, unlike the display order STAR, GALAXY, QSO.
- ``healpix_l5`` ids are HEALPix level 5 (nside 32), NESTED, in ICRS (checked against ra/dec).
  The latitude band of a pixel uses the Galactic latitude of its centre (schema.md, lat_band).
- Metrics below the reporting floor carry ``status != "ok"``; they are exposed as not usable.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
INTERIM_DIR: Final = REPO_ROOT / "data" / "interim"
DEFAULT_RELEASE: Final = "dataset_100k_v0"

CLASSES: Final[tuple[str, ...]] = ("STAR", "GALAXY", "QSO")
MODELS: Final[tuple[str, ...]] = ("lgbm_full", "lgbm_noind")
SCORE_COLUMN: Final = {"STAR": "star", "GALAXY": "galaxy", "QSO": "qso"}

SEED: Final = 20261025
HEALPIX_NSIDE: Final = 32
BAND_EDGES_DEG: Final = (
    11.54,
    30.0,
)  # A < 11.54 <= B < 30 <= C, on |b| of the pixel centre
MAX_POINTS: Final = 20_000  # above this the sky map shows pixels instead of objects
CONFIDENCE_EDGES: Final = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001)
SOLVED_NAMES: Final = {3: "2 parâmetros", 31: "5 parâmetros", 95: "6 parâmetros"}
PRIOR_REGIMES: Final = ("P0", "P1", "P2", "P3")
POPULATIONS: Final = ("a_all", "c_sdsc", "a_combmod")
VARIANTS: Final = ("prior_matched", "as_shipped")


@dataclass(frozen=True)
class Release:
    """Everything loaded from one ``data/interim/<name>/`` directory."""

    name: str
    root: Path
    results: dict
    train_report: dict
    labels: pd.DataFrame  # all folds: source_id, fold, healpix_l5, l, b
    test: pd.DataFrame  # test fold, joined with features and predictions
    n_unscored: int  # test rows without a calibrated score (dropped)

    @property
    def meta(self) -> dict:
        return self.results["meta"]

    @property
    def is_provisional(self) -> bool:
        return "PROVISIONAL" in str(self.meta.get("header", "")).upper()


def _single(root: Path, pattern: str) -> Path:
    files = sorted(root.glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected exactly one {pattern} in {root}, found {len(files)}"
        )
    return files[0]


def _classes_upper(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper()


def load_release(name: str = DEFAULT_RELEASE, base: Path | None = None) -> Release:
    """Read results, training report, labels, features and both models' predictions."""
    root = (base or INTERIM_DIR) / name
    results = json.loads((root / "eval" / "results.json").read_text(encoding="utf-8"))
    report = json.loads((root / "train_report.json").read_text(encoding="utf-8"))

    label_cols = [
        "source_id",
        "sdss_class",
        "sdss_z",
        "healpix_l5",
        "fold",
        "ra",
        "dec",
        "l",
        "b",
    ]
    labels = pd.read_parquet(root / "labels.parquet")
    dsc = [c for c in labels.columns if c.startswith("classprob_dsc_allosmod_")]
    labels = labels[label_cols + dsc].copy()
    labels["source_id"] = labels["source_id"].astype("int64")
    labels["fold"] = labels["fold"].astype(str)
    labels["true"] = _classes_upper(labels["sdss_class"])

    feats = pd.read_parquet(
        root / "features.parquet",
        columns=["source_id", "phot_g_mean_mag", "astrometric_params_solved"],
    )
    feats["source_id"] = feats["source_id"].astype("int64")

    test = labels[labels["fold"] == "test"].merge(feats, on="source_id", how="left")
    test["dsc_available"] = test[dsc].notna().all(axis=1) if dsc else False

    for model in MODELS:
        pred = pd.read_parquet(_single(root, f"predictions_{model}_*.parquet"))
        pred["source_id"] = pred["source_id"].astype("int64")
        keep = {f"p_cal_{SCORE_COLUMN[c]}": f"p_{model}_{c}" for c in CLASSES}
        test = test.merge(
            pred[["source_id", *keep]].rename(columns=keep), on="source_id", how="left"
        )

    score_cols = [f"p_{m}_{c}" for m in MODELS for c in CLASSES]
    scored = test[score_cols].notna().all(axis=1)
    n_unscored = int((~scored).sum())
    test = test[scored].reset_index(drop=True)
    for model in MODELS:
        probs = test[[f"p_{model}_{c}" for c in CLASSES]].to_numpy(dtype=float)
        idx = probs.argmax(axis=1)
        test[f"pred_{model}"] = np.array(CLASSES)[idx]
        test[f"conf_{model}"] = probs.max(axis=1)
        test[f"err_{model}"] = test[f"pred_{model}"] != test["true"]

    labels = labels[["source_id", "fold", "healpix_l5", "l", "b"]]
    return Release(name, root, results, report, labels, test, n_unscored)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; (nan, nan) when n is 0."""
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------- headline numbers


@dataclass(frozen=True)
class Headline:
    model: str
    n_test: int
    n_errors: int
    accuracy: float
    hits_per_100: int
    errors_per_100: int
    star_qso_error_share: float  # STAR<->QSO errors over all errors
    recall: dict[str, float]  # per true class
    precision: dict[str, float]  # per predicted class
    class_mix_train: dict[str, float]
    dsc_coverage: float
    n_test_pixels: int
    n_pixels_with_error: int


def headline(rel: Release, model: str = "lgbm_full") -> Headline:
    t = rel.test
    err = t[f"err_{model}"]
    n, n_err = len(t), int(err.sum())
    pred = t[f"pred_{model}"]
    recall = {c: float((~err[t["true"] == c]).mean()) for c in CLASSES}
    precision = {
        c: float((~err[pred == c]).mean()) if (pred == c).any() else math.nan
        for c in CLASSES
    }
    pair = ((t["true"] == "STAR") & (pred == "QSO")) | (
        (t["true"] == "QSO") & (pred == "STAR")
    )
    acc = 1 - n_err / n
    px = t.groupby("healpix_l5")[f"err_{model}"].any()
    fractions = rel.train_report["models"][0]["train_class_fractions"]
    return Headline(
        model=model,
        n_test=n,
        n_errors=n_err,
        accuracy=acc,
        hits_per_100=round(acc * 100),
        errors_per_100=100 - round(acc * 100),
        star_qso_error_share=float((err & pair).sum() / n_err) if n_err else math.nan,
        recall=recall,
        precision=precision,
        class_mix_train={c: float(fractions[c]) for c in CLASSES},
        dsc_coverage=float(t["dsc_available"].mean()),
        n_test_pixels=int(px.size),
        n_pixels_with_error=int(px.sum()),
    )


def errors_by_type(rel: Release, model: str = "lgbm_full") -> pd.DataFrame:
    """The off-diagonal cells as rows (true, pred, n, share), largest first."""
    t = rel.test[rel.test[f"err_{model}"]]
    out = t.groupby(["true", f"pred_{model}"]).size().rename("n").reset_index()
    out = out.rename(columns={f"pred_{model}": "pred"}).sort_values(
        "n", ascending=False
    )
    out["share"] = out["n"] / out["n"].sum() if len(out) else math.nan
    return out.reset_index(drop=True)


def confusion(rel: Release, model: str = "lgbm_full") -> pd.DataFrame:
    """Counts with rows = true class and columns = predicted class, in CLASSES order."""
    t = rel.test
    table = pd.crosstab(t["true"], t[f"pred_{model}"]).reindex(
        index=CLASSES, columns=CLASSES
    )
    return table.fillna(0).astype(int)


# ---------------------------------------------------------------- confidence and abstention


def calibration_points(
    rel: Release, model: str = "lgbm_full", n_bins: int = 10
) -> dict[str, pd.DataFrame]:
    """One-vs-rest reliability points per class, equal-mass bins (mean_pred, observed, n)."""
    t = rel.test
    out: dict[str, pd.DataFrame] = {}
    for c in CLASSES:
        p = t[f"p_{model}_{c}"].to_numpy(dtype=float)
        hit = (t["true"] == c).to_numpy()
        order = np.argsort(p, kind="stable")
        rows = [
            {
                "mean_pred": float(p[g].mean()),
                "observed": float(hit[g].mean()),
                "n": len(g),
            }
            for g in np.array_split(order, n_bins)
            if len(g)
        ]
        out[c] = pd.DataFrame(rows)
    return out


def confidence_histogram(
    rel: Release, model: str = "lgbm_full", edges: tuple[float, ...] = CONFIDENCE_EDGES
) -> pd.DataFrame:
    """Per confidence bin: objects, wrong ones, error share and a Wilson interval."""
    t = rel.test
    conf, wrong = t[f"conf_{model}"].to_numpy(), t[f"err_{model}"].to_numpy()
    rows = []
    for lo, hi in itertools.pairwise(edges):
        sel = (conf >= lo) & (conf < hi)
        n, k = int(sel.sum()), int(wrong[sel].sum())
        low, high = wilson(k, n)
        rows.append(
            {
                "lo": lo,
                "hi": min(hi, 1.0),
                "n": n,
                "n_wrong": k,
                "error_share": k / n if n else math.nan,
                "ci_lo": low,
                "ci_hi": high,
            }
        )
    return pd.DataFrame(rows)


def abstention_curve(
    rel: Release, model: str = "lgbm_full", thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """For each confidence threshold: coverage kept, accuracy on the kept rows, errors set aside."""
    grid = np.round(np.arange(0.40, 1.0, 0.01), 2) if thresholds is None else thresholds
    t = rel.test
    conf, wrong = t[f"conf_{model}"].to_numpy(), t[f"err_{model}"].to_numpy()
    total_err = int(wrong.sum())
    rows = []
    for thr in grid:
        keep = conf >= thr
        k = int(keep.sum())
        rows.append(
            {
                "threshold": float(thr),
                "coverage": k / len(conf),
                "accuracy_kept": float(1 - wrong[keep].mean()) if k else math.nan,
                "n_set_aside": int((~keep).sum()),
                "errors_set_aside": int(wrong[~keep].sum()),
                "errors_total": total_err,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- models and metrics


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


def _frame_from_cells(cells: dict, extra: dict) -> pd.DataFrame:
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


def _variant(rel: Release, population: str, prior: str, variant: str) -> dict:
    return rel.results["populations"][population]["priors"][prior]["variants"][variant]


def metrics_frame(
    rel: Release,
    population: str = "c_sdsc",
    prior: str = "P0",
    variant: str = "prior_matched",
) -> pd.DataFrame:
    """Long table (model, metric, point, lo, hi, usable) from results.json."""
    models = _variant(rel, population, prior, variant)["models"]
    return pd.concat(
        [
            _frame_from_cells(m["metrics"], {"model": name})
            for name, m in models.items()
        ],
        ignore_index=True,
    )


def paired_diffs_frame(
    rel: Release,
    population: str = "c_sdsc",
    prior: str = "P0",
    variant: str = "prior_matched",
) -> pd.DataFrame:
    """Long table (pair, metric, point, lo, hi, usable) of the paired bootstrap differences."""
    diffs = _variant(rel, population, prior, variant)["diffs"]
    return pd.concat(
        [_frame_from_cells(cells, {"pair": pair}) for pair, cells in diffs.items()],
        ignore_index=True,
    )


def population_rows(rel: Release, population: str) -> int:
    return int(rel.results["populations"][population]["n_rows"])


# ---------------------------------------------------------------- error rate tables


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
            rows.append(_rate_row(label, key, key, sub[f"err_{model}"]))
    else:
        assert edges is not None
        bounds = (-math.inf, *edges, math.inf)
        for lo, hi in itertools.pairwise(bounds):
            sub = t[(t[column] >= lo) & (t[column] < hi)]
            rows.append(_rate_row(_bin_label(lo, hi), lo, hi, sub[f"err_{model}"]))
        missing = t[t[column].isna()]
        if len(missing):
            rows.append(
                _rate_row("sem medida", math.nan, math.nan, missing[f"err_{model}"])
            )
    return pd.DataFrame(rows)


def _bin_label(lo: float, hi: float) -> str:
    if math.isinf(lo):
        return f"< {hi:g}"
    if math.isinf(hi):
        return f"≥ {lo:g}"
    return f"{lo:g}–{hi:g}"


def _rate_row(label: str, lo: float, hi: float, errors: pd.Series) -> dict:
    n, k = len(errors), int(errors.sum())
    ci_lo, ci_hi = wilson(k, n)
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


# ---------------------------------------------------------------- sky


def mollweide(l_deg: np.ndarray, b_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mollweide x, y in units where the map spans x in +-2*sqrt(2), y in +-sqrt(2).

    Galactic centre in the middle, longitude increasing to the left.
    """
    lon = np.asarray(l_deg, dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lam, phi = -np.radians(lon), np.radians(np.asarray(b_deg, dtype=float))
    theta = phi.copy()
    for _ in range(40):
        f = 2 * theta + np.sin(2 * theta) - np.pi * np.sin(phi)
        g = 2 + 2 * np.cos(2 * theta)
        theta = theta - np.divide(f, g, out=np.zeros_like(f), where=np.abs(g) > 1e-9)
    return 2 * math.sqrt(2) / math.pi * lam * np.cos(theta), math.sqrt(2) * np.sin(
        theta
    )


def pixel_table(rel: Release) -> pd.DataFrame:
    """One row per HEALPix pixel: fold, objects, Galactic centre, map x/y and latitude band."""
    from astropy.coordinates import SkyCoord
    from astropy_healpix import HEALPix

    lab = rel.labels
    px = (
        lab.groupby("healpix_l5")
        .agg(fold=("fold", "first"), n=("source_id", "size"))
        .reset_index()
    )
    lon, lat = HEALPix(nside=HEALPIX_NSIDE, order="nested").healpix_to_lonlat(
        px["healpix_l5"].to_numpy()
    )
    gal = SkyCoord(lon, lat, frame="icrs").galactic
    px["l_centre"], px["b_centre"] = gal.l.deg, gal.b.deg
    px["x"], px["y"] = mollweide(px["l_centre"].to_numpy(), px["b_centre"].to_numpy())
    px["band"] = np.where(
        px["b_centre"].abs() < BAND_EDGES_DEG[0],
        "A",
        np.where(px["b_centre"].abs() < BAND_EDGES_DEG[1], "B", "C"),
    )
    return px


def sky_points(rel: Release, model: str = "lgbm_full") -> pd.DataFrame | None:
    """Test objects on the map (x, y, error); None when there are too many to draw."""
    t = rel.test
    if len(t) > MAX_POINTS:
        return None
    x, y = mollweide(t["l"].to_numpy(), t["b"].to_numpy())
    return pd.DataFrame(
        {"x": x, "y": y, "error": t[f"err_{model}"].to_numpy(), "true": t["true"]}
    )


def test_pixel_stats(rel: Release, model: str, pixels: pd.DataFrame) -> pd.DataFrame:
    """Per test pixel: objects, errors, rate and whether it reaches the reporting floor."""
    floor = int(rel.meta["floor"]["min_objects"])
    g = (
        rel.test.groupby("healpix_l5")[f"err_{model}"]
        .agg(n="size", errors="sum")
        .reset_index()
    )
    out = g.merge(pixels[["healpix_l5", "x", "y", "band"]], on="healpix_l5", how="left")
    out["rate"] = out["errors"] / out["n"]
    out["passes_floor"] = out["n"] >= floor
    return out


def lat_band_table(rel: Release, model: str, pixels: pd.DataFrame) -> pd.DataFrame:
    """Test objects and errors by latitude band A/B/C of the pixel centre, with Wilson intervals."""
    t = rel.test.merge(pixels[["healpix_l5", "band"]], on="healpix_l5", how="left")
    rows = []
    for band in ("A", "B", "C"):
        sub = t[t["band"] == band]
        n, k = len(sub), int(sub[f"err_{model}"].sum())
        lo, hi = wilson(k, n)
        rows.append(
            {
                "band": band,
                "n_pixels": int(sub["healpix_l5"].nunique()),
                "n": n,
                "errors": k,
                "rate": k / n if n else math.nan,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def chance_share_of_pixels_with_error(stats: pd.DataFrame, error_rate: float) -> float:
    """Share of test pixels expected to hold at least one error if errors fell at random."""
    return float((1.0 - (1.0 - error_rate) ** stats["n"].to_numpy()).mean())


# ---------------------------------------------------------------- Layer 2 (optional)

LAYER2_SYSTEMS: Final[tuple[str, ...]] = ("S0", "S1", "S1*", "S2", "S3")
LAYER2_ROUTED: Final[tuple[str, ...]] = (
    "L1_routed",
    "L2_forced",
    "L1_ok",
    "L2_selective",
)
LAYER2_SHOWN_METRICS: Final[tuple[str, ...]] = ("accuracy", "macro_f1", "coverage")
LAYER2_COST_ROWS: Final = {
    "prompt_tokens": "tokens de entrada",
    "completion_tokens": "tokens de saída",
    "total_tokens": "tokens no total",
    "calls_per_object": "chamadas (tentativas)",
    "latency_ms": "latência (ms)",
}


def load_layer2(rel: Release | str, base: Path | None = None) -> dict | None:
    """Read ``layer2/eval_l2.json`` of a release; None when the release has no Layer 2 run.

    Returns the parsed document under ``raw`` plus ready-to-show frames: ``systems`` (S0-S3, S1*),
    ``routed`` (the routed-subset views) and ``diffs`` (paired differences with the ``n/a`` mark).
    """
    root = rel.root if isinstance(rel, Release) else (base or INTERIM_DIR) / rel
    path = root / "layer2" / "eval_l2.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    units = raw["eval"]["units"]
    return {
        "raw": raw,
        "systems": _layer2_units_frame(units, LAYER2_SYSTEMS),
        "routed": _layer2_units_frame(units, LAYER2_ROUTED),
        "diffs": _layer2_diffs_frame(raw["eval"]["diffs"]),
    }


def _layer2_units_frame(units: dict, names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for name in names:
        unit = units[name]
        row: dict = {
            "unit": name,
            "n": int(unit["n"]),
            "n_answered": int(unit["n_answered"]),
        }
        for key in LAYER2_SHOWN_METRICS:
            cell = unit["metrics"][key]
            usable = cell.get("status") == "ok" and cell.get("point") is not None
            row[key] = cell["point"] if usable else math.nan
            row[f"{key}_lo"] = (
                cell["lo"] if usable and cell["lo"] is not None else math.nan
            )
            row[f"{key}_hi"] = (
                cell["hi"] if usable and cell["hi"] is not None else math.nan
            )
            row[f"{key}_mark"] = "" if usable else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def _layer2_diffs_frame(diffs: dict) -> pd.DataFrame:
    frames = []
    for pair, cells in diffs.items():
        shown = {k: cells[k] for k in LAYER2_SHOWN_METRICS}
        frames.append(_frame_from_cells(shown, {"pair": pair}))
    out = pd.concat(frames, ignore_index=True)
    out["mark"] = np.where(out["usable"], "", "n/a")
    out["excludes_zero"] = out["usable"] & ((out["lo"] > 0) | (out["hi"] < 0))
    return out


def layer2_summary(layer2: dict) -> dict:
    """Numbers ready to show: verdict of the pre-registered rule, systems, failures and cost."""
    raw = layer2["raw"]
    meta, ops, verdict = raw["meta"], raw["operations"], raw["eval"]["verdict"]
    pairs = []
    for pair, cell in verdict["pairs"].items():
        frame = _frame_from_cells({verdict["metric"]: cell}, {"pair": pair})
        frame["evaluable"] = bool(cell["evaluable"])
        frame["gain"] = bool(cell["gain"])
        frame["mark"] = "" if cell["status"] == "ok" else "n/a"
        pairs.append(frame)
    status = pd.DataFrame(
        [
            {
                "status": key,
                "kind": "ok"
                if key == "OK"
                else "unknown"
                if key == "UNKNOWN"
                else "falha",
                "n": int(ops["status_counts"][key]),
                "rate": float(ops["status_rates"][key]),
            }
            for key in sorted(ops["status_counts"])
        ]
    )
    cost = pd.DataFrame(
        [
            {
                "item": label,
                "n": int(ops[key]["n"]),
                **{
                    stat: math.nan if ops[key][stat] is None else float(ops[key][stat])
                    for stat in ("mean", "p50", "p95")
                },
            }
            for key, label in LAYER2_COST_ROWS.items()
        ]
    )
    return {
        "outcome": verdict["outcome"],
        "metric": verdict["metric"],
        "n_evaluable": int(verdict["n_evaluable"]),
        "n_pairs": len(verdict["pairs"]),
        "pairs": pd.concat(pairs, ignore_index=True),
        "systems": layer2["systems"],
        "routed": layer2["routed"],
        "diffs": layer2["diffs"],
        "status": status,
        "unknown_rate": float(ops["unknown_rate"]),
        "failure_rate": float(ops["failure_rate"]),
        "n_routed": int(ops["n_routed"]),
        "class_mix": {c: int(ops["class_mix"].get(c, 0)) for c in CLASSES},
        "cost": cost,
        "model_requested": str(meta["model_requested"]),
        "models_reported": {str(k): int(v) for k, v in ops["models_reported"].items()},
        "prompt_version": str(meta["prompt_version"]),
        "floor": {
            **meta["floor"],
            "min_objects": int(meta["floor"]["min_objects"]),
            "min_pixels": int(meta["floor"]["min_pixels"]),
        },
        "n_boot": int(raw["eval"]["n_boot"]),
    }
