"""Método page: one sentence in simple mode; rule, method, glossary and limits in technical mode."""

from __future__ import annotations

import streamlit as st

from astro_triage.dashboard import data as D
from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard.html_system import lang_root, system_css
from astro_triage.dashboard.method_model import (  # noqa: F401
    MethodFacts,
    class_mix,
    method_facts,
    reading_paragraphs,
    usage_paragraphs,
)
from astro_triage.dashboard.views_common import tech_on

USAGE_TITLE = "Dispersão por objeto"


def _html(block: str) -> None:
    st.markdown(lang_root(block), unsafe_allow_html=True)


def method_page(rel: D.Release) -> None:
    st.markdown(system_css(), unsafe_allow_html=True)
    mix = class_mix(rel)
    limitations = list(rel.results["limitations"])
    common = HM.method_html(mix) + HM.rule_html() + HM.three_limits_html(limitations)
    if not tech_on():
        _html(
            HM.simple_html()
            + common
            + HM.glossary_details()
            + HM.links_html()
            + HM.sources_line()
        )
        return
    _html(
        HM.lead_html()
        + common
        + HM.reading_html(reading_paragraphs(method_facts(rel)))
        + HM.usage_html(USAGE_TITLE, usage_paragraphs())
        + HM.glossary_details()
        + HM.limitations_html(limitations)
        + HM.links_html()
        + HM.sources_html()
    )
