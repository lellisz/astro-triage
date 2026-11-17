"""Numbers for the overview header: QSO purity/completeness points and error pairs.

No Streamlit import. Hashes in ``delta.json`` are dropped on load and never exposed.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from astro_triage.dashboard import fmt
from astro_triage.dashboard.data_dl import load_results_dl
from astro_triage.dashboard.data_eval import results_variant
from astro_triage.dashboard.data_overview import errors_by_type
from astro_triage.dashboard.data_release import CLASSES, Release, strip_sha256

PURITY_POPULATION = "a_all"
SKY_PRIORS = ("P1", "P2", "P3")
SHIPPED_PRIOR = "P2"
PARTIAL_PRIORS = frozenset({"P3"})  # ADR 0003: P3 applies only to test bands B and C
_METRIC_FIELDS = (
    "purity",
    "purity_lo",
    "purity_hi",
    "completeness",
    "completeness_lo",
    "completeness_hi",
)
PAIR_NAMES = {"STAR": "estrela", "GALAXY": "galáxia", "QSO": "quasar"}
_PAIR_ORDER = (("STAR", "QSO"), ("GALAXY", "QSO"), ("GALAXY", "STAR"))


def load_delta(rel: Release) -> dict | None:
    """Parsed ``lgbm_seeds/delta.json`` without any ``*sha256`` key; None when absent."""
    path = rel.root / "lgbm_seeds" / "delta.json"
    if not path.is_file():
        return None
    return strip_sha256(json.loads(path.read_text(encoding="utf-8")))


def _test_point(rel: Release, variant: str, model: str, prior: str) -> dict | None:
    """Purity and completeness of QSO with their intervals; None unless both are usable."""
    try:
        cells = results_variant(rel, PURITY_POPULATION, prior, variant)
        metrics = cells["models"][model]["metrics"]
        pur, com = metrics["purity/QSO"], metrics["completeness/QSO"]
    except KeyError:
        return None
    if pur.get("status") != "ok" or com.get("status") != "ok":
        return None
    return {
        "purity": pur["point"],
        "purity_lo": pur["lo"],
        "purity_hi": pur["hi"],
        "completeness": com["point"],
        "completeness_lo": com["lo"],
        "completeness_hi": com["hi"],
        "fold": "teste",
    }


def _prior_state(rel: Release, model: str, prior: str, variant: str) -> dict:
    """QSO prior and the six QSO metric fields of one cell; metrics None unless usable."""
    try:
        qso_prior = rel.results["populations"][PURITY_POPULATION]["priors"][prior][
            "prior"
        ]["QSO"]
    except KeyError:
        qso_prior = None
    point = _test_point(rel, variant, model, prior) or {}
    return {
        "prior": prior,
        "qso_prior": qso_prior,
        **{k: point.get(k) for k in _METRIC_FIELDS},
        "partial": prior in PARTIAL_PRIORS,
    }


def prior_states(rel: Release, model: str = "lgbm_full") -> dict[str, dict]:
    """QSO state adjusted to each sky prior (P1, P2, P3), in that order."""
    return {p: _prior_state(rel, model, p, "prior_matched") for p in SKY_PRIORS}


def shipped_state(rel: Release, model: str = "lgbm_full") -> dict | None:
    """QSO state as shipped, fixed at the shipped prior."""
    return _prior_state(rel, model, SHIPPED_PRIOR, "as_shipped")


def accuracy_defs(rel: Release, model: str = "lgbm_full") -> dict:
    """The two overall-accuracy definitions, kept apart: as shipped and over seeds."""
    try:
        shipped = rel.results["populations"][PURITY_POPULATION]["priors"][
            SHIPPED_PRIOR
        ]["variants"]["as_shipped"]["models"][model]["detail"]["accuracy"]
    except KeyError:
        shipped = None
    descriptive = (load_results_dl(rel) or {}).get("descriptive", {})
    fam = descriptive.get("families", {}).get(model, {}).get("accuracy", {})
    systems = descriptive.get("systems", {})
    n_seeds = sum(1 for k in systems if re.fullmatch(rf"{re.escape(model)}_s\d+", k))
    regime = descriptive.get("regime")
    return {
        "shipped": shipped,
        "seeds_mean": fam.get("mean"),
        "seeds_min": fam.get("min"),
        "seeds_max": fam.get("max"),
        "n_seeds": n_seeds or None,
        "regime": regime,
        "balanced": isinstance(regime, str)
        and "BALANCED" in regime
        and "unweighted" in regime,
    }


def _failed_checks(cell: dict) -> bool:
    """True when the cell carries a failing check (keys present in delta.json)."""
    return (
        cell.get("val_checks_pass") is False
        or any(cell.get("low_count", []))
        or cell.get("floor", {}).get("passes") is False
    )


def _validation_point(rel: Release, model: str, prior: str) -> dict | None:
    delta = load_delta(rel)
    if delta is None or delta.get("fold") != "val":
        return None
    try:
        comp = float(delta["primary_completeness"])
        cell = delta["delta"][model.removeprefix("lgbm_")]["QSO"][f"{comp:.2f}"][prior]
        purity = float(cell["v_mean"])
        if _failed_checks(cell):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "purity": purity,
        "purity_lo": None,
        "purity_hi": None,
        "completeness": comp,
        "completeness_lo": None,
        "completeness_hi": None,
        "fold": "validação",
    }


def purity_points(
    rel: Release, model: str = "lgbm_full", prior: str = "P2"
) -> list[dict]:
    """Up to three QSO points in display order: as shipped, validation, prior-matched."""
    shipped = _test_point(rel, "as_shipped", model, prior)
    matched = _test_point(rel, "prior_matched", model, prior)
    valid = _validation_point(rel, model, prior)
    out: list[dict] = []
    if shipped:
        label = f"Como entregue (teste, prior {prior})"
        out.append({"key": "as_shipped", "label": label, **shipped})
    if valid:
        label = (
            f"Validação, completude {comp_label(valid['completeness'])} (prior {prior})"
        )
        out.append({"key": "validation", "label": label, "dashed": True, **valid})
    if matched:
        label = f"Ajustado ao prior (teste, prior {prior})"
        out.append({"key": "prior_matched", "label": label, **matched})
    return out


def comp_label(completeness: float) -> str:
    return fmt.pt_num(completeness, 2)


def error_directions(rel: Release, model: str = "lgbm_full") -> pd.DataFrame:
    """The six error directions (true, pred, n, share), largest first, zeros included."""
    errs = errors_by_type(rel, model)
    counts = {(r.true, r.pred): int(r.n) for r in errs.itertuples()}
    total = sum(counts.values())
    rows = [
        {
            "true": t,
            "pred": p,
            "n": counts.get((t, p), 0),
            "share": counts.get((t, p), 0) / total if total else float("nan"),
        }
        for t in CLASSES
        for p in CLASSES
        if t != p
    ]
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def error_pairs(rel: Release, model: str = "lgbm_full") -> pd.DataFrame:
    """Three class pairs with errors summed in both directions, largest first.

    Columns: pair (label), a, b, n_ab, n_ba, n, share of all errors.
    """
    directions = error_directions(rel, model)
    counts = {(r.true, r.pred): int(r.n) for r in directions.itertuples()}
    total = int(directions["n"].sum())
    rows = []
    for a, b in _PAIR_ORDER:
        n_ab, n_ba = counts.get((a, b), 0), counts.get((b, a), 0)
        rows.append(
            {
                "pair": f"{PAIR_NAMES[a]} ↔ {PAIR_NAMES[b]}",
                "a": a,
                "b": b,
                "n_ab": n_ab,
                "n_ba": n_ba,
                "n": n_ab + n_ba,
                "share": (n_ab + n_ba) / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
