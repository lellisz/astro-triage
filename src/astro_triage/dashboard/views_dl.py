"""Comparação page (LightGBM x deep learning): data to plain structures, then the Streamlit page.

The rule and the priors are read from the code that ran the evaluation (``dl_compare`` and
``runner.GRID``); the numbers come from ``results_dl.json``. No HTML here (see ``html_dl``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import streamlit as st

from astro_triage.dashboard import charts_dl
from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import html_dl as HD
from astro_triage.dashboard.constants import NO_BAR
from astro_triage.dashboard.views_common import tech_on
from astro_triage.evaluation import dl_compare as RULE
from astro_triage.evaluation.runner import GRID

HEADLINE_ROLE = "headline"
QSO = "QSO"
_PRIOR_IDS = ("P0", "P1", "P2", "P3")
_CHART_SYSTEMS = ("lgbm_full", "mlp_full", "ftt_full")
_FRONT_ROLES = ("front3", "front4")


@dataclass(frozen=True)
class RuleFacts:
    multiplier: float
    ddof: int
    n_seeds: int
    block_level: int
    block_divisor: int
    min_blocks: int
    primary_completeness: float
    levels: tuple[float, ...]
    primary_prior: str
    sensitivity_priors: tuple[str, ...]
    p3_bands: tuple[str, ...]


@dataclass(frozen=True)
class Cell:
    pair: str
    role: str
    cls: str
    d: float | None
    lo: float | None
    hi: float | None
    delta: float | None
    verdict: str | None
    na: bool


@dataclass(frozen=True)
class QsoPurity:
    system: str
    low: float
    median: float
    high: float


@dataclass(frozen=True)
class PriorRow:
    prior: str
    fractions: Mapping[str, float]


def rule_facts() -> RuleFacts:
    return RuleFacts(
        multiplier=RULE.DELTA_SD_MULTIPLIER,
        ddof=RULE.DELTA_DDOF,
        n_seeds=RULE.N_SEEDS,
        block_level=RULE.BLOCK_LEVEL,
        block_divisor=RULE.BLOCK_DIVISOR,
        min_blocks=RULE.MIN_BLOCKS,
        primary_completeness=RULE.PRIMARY_COMPLETENESS,
        levels=tuple(RULE.COMPLETENESS_LEVELS),
        primary_prior=RULE.PRIMARY_PRIOR,
        sensitivity_priors=tuple(RULE.SENSITIVITY_PRIORS),
        p3_bands=tuple(RULE.P3_BANDS),
    )


def level_key(completeness: float) -> str:
    """Key used for a completeness level in ``results_dl.json`` (two decimals)."""
    return f"{completeness:.2f}"


def _cell_of(
    results: Mapping, pair: str, prior: str, completeness: float
) -> dict | None:
    priors = results.get("compare", {}).get("pairs", {}).get(pair, {})
    return priors.get(prior, {}).get(level_key(completeness))


def cells(
    results: Mapping,
    prior: str,
    completeness: float,
    roles: tuple[str, ...] | None = None,
    classes: tuple[str, ...] = tuple(RULE.TARGET_CLASSES),
) -> list[Cell]:
    """One row per pair and class at a prior and completeness; ``roles`` filters the pairs."""
    out: list[Cell] = []
    for pair in results.get("compare", {}).get("pairs", {}):
        cell = _cell_of(results, pair, prior, completeness)
        if not cell or (roles is not None and cell.get("role") not in roles):
            continue
        for cls in classes:
            x = cell.get("classes", {}).get(cls)
            if not x:
                continue
            ci = x.get("ci")
            out.append(
                Cell(
                    pair=pair,
                    role=str(cell.get("role", "")),
                    cls=cls,
                    d=x.get("d"),
                    lo=None if ci is None else ci[0],
                    hi=None if ci is None else ci[1],
                    delta=x.get("delta"),
                    verdict=x.get("class_verdict"),
                    na=bool(x.get("na")),
                )
            )
    return out


def pair_verdicts(
    results: Mapping, prior: str, completeness: float
) -> dict[str, str | None]:
    """Pair-level verdict text of the headline pairs, keyed by pair."""
    out: dict[str, str | None] = {}
    for pair in results.get("compare", {}).get("pairs", {}):
        cell = _cell_of(results, pair, prior, completeness)
        if cell and cell.get("role") == HEADLINE_ROLE:
            out[pair] = cell.get("verdict")
    return out


def qso_purity(results: Mapping, prior: str, completeness: float) -> list[QsoPurity]:
    """Absolute QSO purity across seeds (min, median, max) of each system in the headline pairs."""
    seen: dict[str, QsoPurity] = {}
    for pair in pair_verdicts(results, prior, completeness):
        cell = _cell_of(results, pair, prior, completeness) or {}
        x = cell.get("classes", {}).get(QSO, {})
        for system, stats in (x.get("purity_seed") or {}).items():
            if system not in seen and {"min", "median", "max"} <= set(stats):
                seen[system] = QsoPurity(
                    system, stats["min"], stats["median"], stats["max"]
                )
    return list(seen.values())


def front_cells(results: Mapping, prior: str, completeness: float) -> list[Cell]:
    """QSO differences (with indicators minus without) of the indicator-removal fronts."""
    return cells(results, prior, completeness, roles=_FRONT_ROLES, classes=(QSO,))


def prior_rows() -> list[PriorRow]:
    return [PriorRow(p, dict(GRID[p])) for p in _PRIOR_IDS if p in GRID]


def chart_systems(frame_systems: list[str]) -> list[str]:
    return [s for s in _CHART_SYSTEMS if s in frame_systems]


def dl_page(rel: D.Release) -> None:
    results = DC.load_results_dl(rel)
    if results is None:
        st.markdown(HD.unavailable_html(), unsafe_allow_html=True)
        return
    rule = rule_facts()
    prior, comp = rule.primary_prior, rule.primary_completeness
    st.markdown(HD.lead_html(), unsafe_allow_html=True)
    st.markdown(HD.seal_html(), unsafe_allow_html=True)
    st.markdown(
        HD.verdict_html(pair_verdicts(results, prior, comp), prior, comp),
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        charts_dl.pair_interval_figure(
            results, prior, level_key(comp), roles=(HEADLINE_ROLE,)
        ),
        width="stretch",
        config=NO_BAR,
    )
    st.markdown(HD.chart_caption_html(), unsafe_allow_html=True)
    if not tech_on():
        st.markdown(HD.hint_html(), unsafe_allow_html=True)
        return
    st.markdown(HD.rule_html(rule), unsafe_allow_html=True)
    st.markdown(HD.reading_html(), unsafe_allow_html=True)
    head = cells(results, prior, comp, roles=(HEADLINE_ROLE,))
    st.markdown(HD.headline_table_html(head, prior, comp), unsafe_allow_html=True)
    others = [
        (p, level_key(c), cells(results, p, c))
        for p in (prior, *rule.sensitivity_priors)
        for c in rule.levels
        if (p, c) != (prior, comp)
    ]
    with st.expander(HD.SENSITIVITY_TITLE):
        st.markdown(HD.sensitivity_html(others), unsafe_allow_html=True)
    st.markdown(
        HD.qso_purity_html(
            qso_purity(results, prior, comp), prior, comp, GRID[prior][QSO]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        HD.fronts_html(front_cells(results, prior, comp), prior, comp),
        unsafe_allow_html=True,
    )
    st.markdown(HD.priors_html(prior_rows(), rule.p3_bands), unsafe_allow_html=True)
    frame = DC.load_risk_coverage_dl(rel)
    if frame is not None and not frame.empty:
        st.markdown(HD.risk_title_html(), unsafe_allow_html=True)
        systems = chart_systems(list(frame["system"].unique()))
        st.plotly_chart(
            charts_dl.risk_coverage_figure(frame, systems),
            width="stretch",
            config=NO_BAR,
        )
    st.markdown(HD.limits_html(), unsafe_allow_html=True)
