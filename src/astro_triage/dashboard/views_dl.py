"""Comparação page (LightGBM x deep learning): data to plain structures, then the Streamlit page.

The rule is read from the code that ran the evaluation (``dl_compare``); the numbers come from
``results_dl.json`` through ``data_dl``. No HTML here (see ``html_dl``). The page has no
technical switch: every block is always shown, the descriptive ones inside expanders.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import data_dl as DD
from astro_triage.dashboard import html_dl as HD
from astro_triage.dashboard import html_system as hs
from astro_triage.evaluation import dl_compare as RULE


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


def _md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def dl_page(rel: D.Release) -> None:
    results = DC.load_results_dl(rel)
    if results is None:
        _md(HD.unavailable_html())
        return
    try:
        head = DD.headline(results)
        sensitivity = DD.sensitivity(results, head)
        descriptive = DD.descriptive(results, head)
        limitations = DD.load_limitations(rel)
        seal = HD.seal_html(
            rel.name,
            DD.adr_status(),
            DD.freeze_exists(rel),
            DD.below_floor(head),
            limitations is not None,
        )
        page = HD.page_html(head, sensitivity, seal)
    except DD.DlDataError as exc:
        _md(HD.error_html(str(exc)))
        return
    _md(hs.system_css() + page)
    with st.expander(HD.DESCRIPTIVE_TITLE):
        _md(HD.descriptive_html(descriptive, head.prior, head.level))
    with st.expander(HD.LIMITS_TITLE):
        _md(HD.limitations_html(limitations))
    n_boot = (results.get("compare") or {}).get("n_boot")
    _md(HD.method_html(rule_facts(), n_boot if isinstance(n_boot, int) else None))
