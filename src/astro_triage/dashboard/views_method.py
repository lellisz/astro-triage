"""Método page: one sentence in simple mode; rule, method, glossary and limits in technical mode."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import fmt
from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import method_notes as N
from astro_triage.dashboard.limits_pt import limitation_lines
from astro_triage.dashboard.views_common import tech_on


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
        N.sky_scale_note(f.min_objects, True) + " " + N.MAP_NOTE,
        N.COUNT_UNDER_BAND_NOTE,
    ]


def usage_paragraphs() -> list[str]:
    return [N.SCATTER_OFF_NOTE]


def _html(block: str) -> None:
    st.markdown(block, unsafe_allow_html=True)


def _limitations(rel: D.Release) -> None:
    originals = list(rel.results["limitations"])
    st.subheader("Limitações da avaliação")
    st.markdown("\n".join(limitation_lines(originals)))
    with st.expander("Texto original (inglês)"):
        st.markdown("\n".join(f"- {text}" for text in originals))


def method_page(rel: D.Release) -> None:
    if not tech_on():
        _html(HM.simple_html())
        _html(HM.sources_line())
        return
    _html(HM.lead_html())
    _html(HM.dl_box(DC.load_delta(rel)))
    mix = rel.test["true"].value_counts(normalize=True).to_dict()
    st.markdown(HM.METHOD_TEXT)
    st.markdown(HM.mix_sentence(mix))
    st.subheader("Como ler os números")
    for text in reading_paragraphs(method_facts(rel)):
        st.markdown(text)
    with st.expander("Dispersão por objeto"):
        for text in usage_paragraphs():
            st.markdown(text)
    st.subheader("Glossário")
    _html(HM.glossary_html())
    _limitations(rel)
    st.subheader("Fontes")
    st.markdown("\n".join(f"- {line}" for line in HM.SOURCES))
