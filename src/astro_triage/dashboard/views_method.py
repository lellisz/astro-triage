"""Método page: one sentence in simple mode; rule, method, glossary and limits in technical mode."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import fmt
from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import method_notes as N
from astro_triage.dashboard.html_system import lang_root, system_css
from astro_triage.dashboard.views_common import tech_on

USAGE_TITLE = "Dispersão por objeto"


@dataclass(frozen=True)
class MethodFacts:
    """Figures from the release artifact, already formatted pt-BR."""

    n_test: str
    n_pixels: str
    prior_qso: str
    min_objects: str
    min_pixels: str
    n_boot: str
    completeness: str


def method_facts(rel: D.Release) -> MethodFacts:
    meta = rel.meta
    return MethodFacts(
        n_test=fmt.pt_int(meta["n_test"]),
        n_pixels=fmt.pt_int(meta["n_test_pixels"]),
        prior_qso=fmt.pt_pct(meta["pi_src_combmod"]["QSO"], 3),
        min_objects=fmt.pt_int(meta["floor"]["min_objects"]),
        min_pixels=fmt.pt_int(meta["floor"]["min_pixels"]),
        n_boot=fmt.pt_int(meta["n_boot"]),
        completeness=fmt.pt_num(HM.delta_facts(DC.load_delta(rel)).completeness, 2),
    )


def reading_paragraphs(f: MethodFacts) -> list[str]:
    return [
        N.sample_note(f.n_test, f.n_pixels, f.prior_qso),
        N.PRIOR_NOTE,
        (
            "A pureza só tem sentido junto com a completude. A pureza à completude fixa de "
            f"{f.completeness} aparece na validação e, na comparação com aprendizado profundo, "
            "no teste."
        ),
        N.floor_note(f.min_objects, f.min_pixels),
        (
            "Os intervalos de acerto por faixa usam o método de Wilson. "
            + N.bootstrap_note(f.n_boot)
            + " "
            + N.INTERVAL_NOTE
        ),
        N.sky_scale_note(f.min_objects, True)
        + " "
        + N.MAP_NOTE
        + " "
        + N.NEIGHBOUR_NOTE,
        N.COUNT_UNDER_BAND_NOTE,
    ]


def usage_paragraphs() -> list[str]:
    return [N.SCATTER_OFF_NOTE]


def _html(block: str) -> None:
    st.markdown(lang_root(block), unsafe_allow_html=True)


def method_page(rel: D.Release) -> None:
    st.markdown(system_css(), unsafe_allow_html=True)
    if not tech_on():
        _html(HM.simple_html())
        _html(HM.sources_line())
        return
    mix = rel.test["true"].value_counts(normalize=True).to_dict()
    _html(
        HM.lead_html()
        + HM.dl_box(DC.load_delta(rel))
        + HM.method_html(mix)
        + HM.reading_html(reading_paragraphs(method_facts(rel)))
        + HM.usage_html(USAGE_TITLE, usage_paragraphs())
        + HM.glossary_details()
        + HM.limitations_html(list(rel.results["limitations"]))
        + HM.sources_html()
    )
