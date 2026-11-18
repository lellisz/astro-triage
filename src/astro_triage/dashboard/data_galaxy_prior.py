"""Data layer and rules of the Galáxias tab, prior-ruler phase (no HTML, no Streamlit).

Every number is read from the release or the DL artifact; a cell that is missing, not
``ok`` or below the reporting floor becomes ``None`` (shown later as n/a), never a raw value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard.data_dl import ClassCell, DlDataError, headline
from astro_triage.dashboard.data_eval import results_variant
from astro_triage.dashboard.data_galaxies import wilson_interval
from astro_triage.dashboard.data_galaxies_z import GalaxyTotals
from astro_triage.dashboard.data_headline import (
    PARTIAL_PRIORS,
    PURITY_POPULATION,
    SKY_PRIORS,
)
from astro_triage.dashboard.data_release import Release
from astro_triage.dashboard.html_galaxies import extreme_bands
from astro_triage.dashboard.kit_types import BarRow, Value
from astro_triage.dashboard.kit_types import Interval as KitInterval
from astro_triage.dashboard.overview_model import ROW_ORDER
from astro_triage.dashboard.overview_text import PARTIAL_TAG
from astro_triage.dashboard.ui_state import RULE_LABEL, Rule, UiState

GALAXY = "GALAXY"
VERDICT_PRIORS = ("P1", "P2")  # P3 is partial and stays out of the verdict
NETWORKS = ("mlp_full", "ftt_full")
REFERENCE = "lgbm_full"
ALL_LABEL = "Todas as galáxias"
BRIGHTEST_LABEL = "As mais brilhantes"
NEAREST_LABEL = "As mais próximas"
FARTHEST_LABEL = "As mais distantes"
_CROSSMATCH_PREFIX = "Selection by crossmatch"


class VerdictKey(StrEnum):
    """Outcome of the prior rule; the texts are chosen by the caller."""

    DEPENDS_WIDE = "depends_wide"
    DEPENDS = "depends"
    WIDE = "wide"
    NO_DIFFERENCE = "no_difference"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float


@dataclass(frozen=True)
class PriorGalaxy:
    """GALAXY purity and completeness under one sky prior; ``None`` means n/a."""

    prior: str
    prior_source: str | None
    n_rows: int | None
    purity: Interval | None
    completeness: Interval | None
    partial: bool


@dataclass(frozen=True)
class GalaxyPriorBlock:
    variant: str
    priors: tuple[PriorGalaxy, ...]

    def get(self, prior: str) -> PriorGalaxy | None:
        return next((p for p in self.priors if p.prior == prior), None)


@dataclass(frozen=True)
class ErrorRate:
    n: int
    errors: int
    rate: float
    ci_lo: float
    ci_hi: float


def _finite(x: object) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return float(x)


def _interval(cell: object, floor_ok: bool) -> Interval | None:
    if not floor_ok or not isinstance(cell, Mapping) or cell.get("status") != "ok":
        return None
    got = [_finite(cell.get(k)) for k in ("point", "lo", "hi")]
    if None in got:
        return None
    return Interval(got[0], got[1], got[2])  # type: ignore[arg-type]


def _prior_galaxy(rel: Release, prior: str, variant: str) -> PriorGalaxy:
    try:
        node = rel.results["populations"][PURITY_POPULATION]["priors"][prior]
        metrics = results_variant(rel, PURITY_POPULATION, prior, variant)["models"][
            REFERENCE
        ]["metrics"]
    except KeyError:
        return PriorGalaxy(prior, None, None, None, None, prior in PARTIAL_PRIORS)
    floor = node.get("floor", {}).get(GALAXY, {})
    min_objects = rel.meta.get("floor", {}).get("min_objects")
    n = floor.get("n")
    floor_ok = (
        floor.get("passes") is not False
        and isinstance(n, int)
        and isinstance(min_objects, int)
        and n >= min_objects
    )
    return PriorGalaxy(
        prior,
        node.get("prior_source"),
        node.get("n_rows"),
        _interval(metrics.get(f"purity/{GALAXY}"), floor_ok),
        _interval(metrics.get(f"completeness/{GALAXY}"), floor_ok),
        prior in PARTIAL_PRIORS,
    )


def load_galaxy_prior_block(
    rel: Release, variant: str = "as_shipped"
) -> GalaxyPriorBlock:
    """GALAXY purity and completeness of the reference model for P1, P2, P3 in that order.

    ``variant`` is ``as_shipped`` or ``prior_matched``. An unusable cell is ``None``.
    """
    return GalaxyPriorBlock(
        variant, tuple(_prior_galaxy(rel, p, variant) for p in SKY_PRIORS)
    )


def dl_galaxy_cells(results: Mapping) -> dict[str, ClassCell | None]:
    """GALAXY cell of each network against the reference, at the headline prior and C*.

    Raises ``DlDataError`` when the artifact has no headline cell or a stored verdict
    differs from the shared rule; a network without a GALAXY cell maps to ``None``.
    """
    head = headline(results)
    by_pair = {c.pair: c for c in head.cells}
    out: dict[str, ClassCell | None] = {}
    for net in NETWORKS:
        cell = by_pair.get(f"{net}-{REFERENCE}")
        if cell is not None:
            VR.check(cell)
        out[net] = cell.classes.get(GALAXY) if cell else None
    return out


def needs_power_note(cell: ClassCell | None) -> bool:
    """True when the shared verdict rule calls ``cell`` underpowered (a tie wider than the band).

    False when the cell is absent or cannot be judged (no verdict, interval or delta).
    """
    if cell is None:
        return False
    try:
        return VR.underpowered(cell)
    except DlDataError:
        return False


def _disjoint(a: Interval, b: Interval) -> bool:
    """Strictly disjoint; intervals that only touch count as overlapping."""
    return a.hi < b.lo or b.hi < a.lo


def prior_depends(block: GalaxyPriorBlock) -> bool | None:
    """Prior axis: True when the P1 and P2 GALAXY purity intervals are strictly disjoint.

    ``None`` when either purity is unavailable. Independent of the networks.
    """
    purities = [p.purity if p else None for p in map(block.get, VERDICT_PRIORS)]
    if None in purities:
        return None
    return _disjoint(purities[0], purities[1])  # type: ignore[arg-type]


def power_networks(cells: Mapping[str, ClassCell | None]) -> tuple[str, ...]:
    """Networks whose comparison needs the power note (see ``needs_power_note``)."""
    return tuple(net for net, c in cells.items() if needs_power_note(c))


def galaxy_verdict(
    block: GalaxyPriorBlock, cells: Mapping[str, ClassCell | None]
) -> VerdictKey:
    """Verdict from two independent axes: the prior (P1 against P2) and the power.

    The prior axis decides depends or no difference and does not look at the networks.
    The power axis is whether any usable network needs the power note; a ``None`` or n/a
    cell is ignored, and with no usable network the verdict keeps only the prior part.
    ``UNAVAILABLE`` only when the prior axis is. Never produces an "equal" outcome.
    """
    depends = prior_depends(block)
    if depends is None:
        return VerdictKey.UNAVAILABLE
    wide = bool(power_networks(cells))
    if depends:
        return VerdictKey.DEPENDS_WIDE if wide else VerdictKey.DEPENDS
    return VerdictKey.WIDE if wide else VerdictKey.NO_DIFFERENCE


def _rate(n: int, errors: int, floor: int) -> ErrorRate | None:
    if n < floor or n <= 0:
        return None
    lo, hi = wilson_interval(errors, n)
    return ErrorRate(n, errors, errors / n, lo, hi)


def galaxy_error_rows(
    total: GalaxyTotals, by_g: pd.DataFrame, by_z: pd.DataFrame, floor: int
) -> tuple[tuple[str, ErrorRate | None], ...]:
    """Four rows (all, brightest, nearest, farthest); a band below the floor is ``None``.

    Pure: the totals and the band tables come from the caller (cached in the view).
    """
    bands = {what: b for what, b in extreme_bands(by_g, by_z, floor)}
    rows: list[tuple[str, ErrorRate | None]] = [
        (ALL_LABEL, _rate(total.n, total.errors, floor))
    ]
    for label in (BRIGHTEST_LABEL, NEAREST_LABEL, FARTHEST_LABEL):
        band = bands.get(label)
        rows.append((label, _rate(band.n, band.errors, floor) if band else None))
    return tuple(rows)


def crossmatch_note(rel: Release) -> str | None:
    """The crossmatch limitation exactly as written in ``results.json``; None if absent."""
    items = rel.results.get("limitations")
    if not isinstance(items, list):
        return None
    return next(
        (s for s in items if isinstance(s, str) and s.startswith(_CROSSMATCH_PREFIX)),
        None,
    )


def _value(iv: Interval | None) -> Value:
    if iv is None:
        return Value(None, None)
    return Value(iv.point, KitInterval(iv.lo, iv.hi))


def galaxy_rows(
    shipped: GalaxyPriorBlock, matched: GalaxyPriorBlock, state: UiState
) -> tuple[tuple[BarRow, ...], tuple[BarRow, ...]]:
    """Purity and completeness bars, one per row of the overview (same order and labels).

    ``shipped`` and ``matched`` are the blocks of the two rules; the shipped row exists only at the
    shipped prior. A cell that is missing or below the floor is an n/a value.
    """
    purity: list[BarRow] = []
    completeness: list[BarRow] = []
    for rule, prior in ROW_ORDER:
        pg = (shipped if rule is Rule.SHIPPED else matched).get(prior)
        label = f"{RULE_LABEL[rule]} · {prior}"
        chosen = (rule, prior) == (state.rule, state.prior)
        note = PARTIAL_TAG if pg is not None and pg.partial else None
        purity.append(BarRow(label, _value(pg.purity if pg else None), chosen, note))
        completeness.append(
            BarRow(label, _value(pg.completeness if pg else None), chosen, note)
        )
    return tuple(purity), tuple(completeness)
