"""Aggregate loaders for the mockups: every displayed number arrives as a ``Cell`` with its source.

``results.json``, ``train_report.json``, ``dataset_report.json`` and ``results_dl.json`` are read
whole; the only parquet access is ``sky_regions`` (DuckDB aggregates by region: nothing per object
leaves it), so the ``Release`` is built with empty object frames.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from astro_triage.dashboard.data_dl import load_results_dl
from astro_triage.dashboard.data_headline import purity_points
from astro_triage.dashboard.data_release import INTERIM_DIR, Release, single_file
from astro_triage.dashboard.data_sky import pixel_table

MODEL = "lgbm_full"
POPULATION = "a_all"
STATE_PRIOR = "P2"
RULER_PRIORS = ("P1", "P2", "P3")
SWEEP_KEY = f"{MODEL}|prior_matched"
RESULTS_FILE = "eval/results.json"
DL_FILE = "dl/eval/results_dl.json"
LABELS_FILE = "labels.parquet"
DATASET_FILE = "dataset_report.json"
TRAIN_FILE = "train_report.json"


@dataclass(frozen=True)
class Cell:
    """A value (None renders as n/a) plus the artifact file and key it was read from."""

    value: float | int | str | None
    file: str
    key: str

    @property
    def ok(self) -> bool:
        return self.value is not None and not (
            isinstance(self.value, float) and math.isnan(self.value)
        )


def load_release_light(name: str, base: Path | None = None) -> Release:
    root = (base or INTERIM_DIR) / name
    results = json.loads((root / "eval" / "results.json").read_text(encoding="utf-8"))
    report = json.loads((root / "train_report.json").read_text(encoding="utf-8"))
    empty = pd.DataFrame()
    return Release(name, root, results, report, empty, empty, 0)


def _metric_key(prior: str, variant: str, metric: str, part: str) -> str:
    return (
        f"populations.{POPULATION}.priors.{prior}.variants.{variant}"
        f".models.{MODEL}.metrics.{metric}/QSO.{part}"
    )


def qso_state(rel: Release, prior: str, variant: str) -> dict[str, Cell]:
    """QSO purity and completeness with intervals for one (prior, variant); n/a when unusable."""
    point = next(
        (p for p in purity_points(rel, MODEL, prior) if p["key"] == variant), None
    )
    out = {}
    for metric, field in (("purity", "purity"), ("completeness", "completeness")):
        for part, suffix in (("point", ""), ("lo", "_lo"), ("hi", "_hi")):
            value = point[f"{field}{suffix}"] if point else None
            out[f"{metric}_{part}"] = Cell(
                value, RESULTS_FILE, _metric_key(prior, variant, metric, part)
            )
    return out


def sky_priors(rel: Release) -> dict[str, Cell]:
    priors = rel.results["populations"][POPULATION]["priors"]
    return {
        p: Cell(
            priors[p]["prior"]["QSO"],
            RESULTS_FILE,
            f"populations.{POPULATION}.priors.{p}.prior.QSO",
        )
        for p in RULER_PRIORS
    }


def sweep_marks(rel: Release) -> list[tuple[Cell, bool]]:
    """(QSO prior, usable) of the S sweep; unusable when any point there is low-count or below floor."""
    block = rel.results["populations"][POPULATION].get("s_sweep", {}).get(SWEEP_KEY)
    if not block:
        return []
    usable: dict[float, bool] = {}
    for pt in block["points"]:
        good = pt["purity_flag"]["QSO"] is None and pt["floor_ok"]["QSO"]
        usable[pt["qso_prior"]] = usable.get(pt["qso_prior"], True) and good
    key = f"populations.{POPULATION}.s_sweep.{SWEEP_KEY}.points[*].qso_prior"
    return [(Cell(q, RESULTS_FILE, key), ok) for q, ok in sorted(usable.items())]


def sweep_range(rel: Release) -> tuple[Cell, Cell]:
    values = [c.value for c, _ in sweep_marks(rel)]
    key = f"populations.{POPULATION}.s_sweep.{SWEEP_KEY}.points[*].qso_prior"
    if not values:
        return Cell(None, RESULTS_FILE, key), Cell(None, RESULTS_FILE, key)
    return Cell(min(values), RESULTS_FILE, key), Cell(max(values), RESULTS_FILE, key)


def census(rel: Release) -> dict[str, Cell]:
    return {
        "regions": Cell(rel.meta["n_test_pixels"], RESULTS_FILE, "meta.n_test_pixels"),
        "objects": Cell(rel.meta["n_test"], RESULTS_FILE, "meta.n_test"),
        "n_boot": Cell(rel.meta["n_boot"], RESULTS_FILE, "meta.n_boot"),
        "seed": Cell(rel.meta["seed"], RESULTS_FILE, "meta.seed"),
    }


def accuracy(rel: Release) -> dict[str, Cell]:
    """The two 'overall accuracy' definitions, kept apart (see the envelope of the build)."""
    shipped_key = (
        f"populations.{POPULATION}.priors.{STATE_PRIOR}.variants.as_shipped"
        f".models.{MODEL}.detail.accuracy"
    )
    shipped = rel.results["populations"][POPULATION]["priors"][STATE_PRIOR]["variants"][
        "as_shipped"
    ]["models"][MODEL]["detail"].get("accuracy")
    dl = load_results_dl(rel)
    fam = ((dl or {}).get("descriptive", {}).get("families", {}).get(MODEL, {})).get(
        "accuracy", {}
    )
    base = f"descriptive.families.{MODEL}.accuracy"
    systems = (dl or {}).get("descriptive", {}).get("systems", {})
    n_seeds = sum(1 for k in systems if re.fullmatch(rf"{MODEL}_s\d+", k)) or None
    return {
        "shipped": Cell(shipped, RESULTS_FILE, shipped_key),
        "seeds_mean": Cell(fam.get("mean"), DL_FILE, f"{base}.mean"),
        "seeds_min": Cell(fam.get("min"), DL_FILE, f"{base}.min"),
        "seeds_max": Cell(fam.get("max"), DL_FILE, f"{base}.max"),
        "n_seeds": Cell(n_seeds, DL_FILE, f"descriptive.systems.{MODEL}_s*"),
        "regime": Cell(
            (dl or {}).get("descriptive", {}).get("regime"),
            DL_FILE,
            "descriptive.regime",
        ),
    }


def _json(rel: Release, name: str) -> dict:
    return json.loads((rel.root / name).read_text(encoding="utf-8"))


def dataset_cells(rel: Release) -> dict[str, Cell]:
    """Counts of the pipeline (dataset_report.json, train_report.json)."""
    rep, train = _json(rel, DATASET_FILE), _json(rel, TRAIN_FILE)
    model = next(m for m in train["models"] if m["model_id"] == MODEL)
    out = {
        k: Cell(rep[k], DATASET_FILE, k)
        for k in (
            "input_pairs",
            "pairs_after_r_match",
            "final_rows",
            "test_pixels_total",
            "test_pixels_with_at_least_100_objects",
        )
    }
    out["r_match"] = Cell(rep["r_match_arcsec"], DATASET_FILE, "r_match_arcsec")
    for cls, n in rep["final_by_class"].items():
        out[f"class_{cls}"] = Cell(n, DATASET_FILE, f"final_by_class.{cls}")
    for fold, n in rep["final_by_fold"].items():
        out[f"fold_{fold}"] = Cell(n, DATASET_FILE, f"final_by_fold.{fold}")
    out["n_features"] = Cell(
        model["n_features"], TRAIN_FILE, f"models[{MODEL}].n_features"
    )
    return out


_ERR_SQL = """
with t as (
  select l.healpix_l5 as h, upper(l.sdss_class) as y,
         p.p_cal_star as s, p.p_cal_galaxy as g, p.p_cal_qso as q
  from read_parquet('{labels}') l join read_parquet('{pred}') p using (source_id)
  where l.fold = 'test' and p.p_cal_star is not null and p.p_cal_galaxy is not null
    and p.p_cal_qso is not null)
select h as healpix_l5, count(*) as n_scored,
  sum(case when (case when s >= g and s >= q then 'STAR' when g >= q then 'GALAXY'
       else 'QSO' end) <> y then 1 else 0 end) as errors
from t group by h"""


def sky_regions(rel: Release) -> pd.DataFrame:
    """One row per level-5 region: fold, objects, centre, Mollweide x/y and test errors.

    Aggregated by DuckDB (pixel, fold, class and score columns only); no per-object value is
    returned. ``errors``/``n_scored`` are the as-shipped argmax errors of LightGBM, as in the
    dashboard map; they are NaN outside the test fold.
    """
    import duckdb

    labels = (rel.root / LABELS_FILE).as_posix().replace("'", "''")
    pred = single_file(rel.root, f"predictions_{MODEL}_*.parquet").as_posix()
    con = duckdb.connect()
    counts = con.execute(
        f"select healpix_l5, fold::varchar as fold, count(*)::bigint as n "
        f"from read_parquet('{labels}') group by 1, 2"
    ).df()
    errs = con.execute(
        _ERR_SQL.format(labels=labels, pred=pred.replace("'", "''"))
    ).df()
    px = pixel_table(replace(rel, pixel_counts=counts))
    return px.merge(errs, on="healpix_l5", how="left")


def dl_level(dl: dict) -> tuple[str, str]:
    """(prior, C*) of the headline cell, found by its role rather than typed here."""
    for priors in dl["compare"]["pairs"].values():
        for prior, levels in priors.items():
            for cstar, block in levels.items():
                if block.get("role") == "headline":
                    return prior, cstar
    raise KeyError("results_dl.json has no headline cell")


def dl_pairs(dl: dict) -> dict[str, dict]:
    """Pair -> block at the headline level, in file order."""
    prior, cstar = dl_level(dl)
    return {p: v[prior][cstar] for p, v in dl["compare"]["pairs"].items()}


def _dl_base(dl: dict, pair: str) -> str:
    prior, cstar = dl_level(dl)
    return f"compare.pairs.{pair}.{prior}.{cstar}"


def dl_pair_verdict(dl: dict, pair: str) -> Cell:
    return Cell(
        dl_pairs(dl)[pair].get("verdict"), DL_FILE, f"{_dl_base(dl, pair)}.verdict"
    )


def dl_class(dl: dict, pair: str, cls: str) -> dict[str, Cell]:
    """D, interval, delta and verdict of one class; D and interval are None when n/a."""
    c = dl_pairs(dl)[pair]["classes"][cls]
    base = f"{_dl_base(dl, pair)}.classes.{cls}"
    usable = not c.get("na") and c.get("d") is not None
    ci = c.get("ci") or [None, None]
    return {
        "d": Cell(c["d"] if usable else None, DL_FILE, f"{base}.d"),
        "lo": Cell(ci[0] if usable else None, DL_FILE, f"{base}.ci[0]"),
        "hi": Cell(ci[1] if usable else None, DL_FILE, f"{base}.ci[1]"),
        "delta": Cell(c.get("delta"), DL_FILE, f"{base}.delta"),
        "verdict": Cell(c.get("class_verdict"), DL_FILE, f"{base}.class_verdict"),
    }


def dl_purity(dl: dict, pair: str, system: str, cls: str) -> dict[str, Cell]:
    """Median and min/max over seeds of the absolute purity of ``system`` in ``pair``."""
    seed = dl_pairs(dl)[pair]["classes"][cls].get("purity_seed", {}).get(system, {})
    base = f"{_dl_base(dl, pair)}.classes.{cls}.purity_seed.{system}"
    return {
        k: Cell(seed.get(k), DL_FILE, f"{base}.{k}") for k in ("median", "min", "max")
    }
