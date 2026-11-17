"""Overview tab and its technical details."""

from __future__ import annotations

import streamlit as st

from astro_triage.dashboard import charts as C
from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import fmt
from astro_triage.dashboard.constants import NO_BAR
from astro_triage.dashboard.data_headline import SHIPPED_PRIOR, SKY_PRIORS
from astro_triage.dashboard.data_overview_page import PRIOR, load_page
from astro_triage.dashboard.html_overview import (
    accuracy_html,
    data_box_html,
    inscription_html,
    ruler_html,
    states_html,
)
from astro_triage.dashboard.html_overview_plots import errors_html
from astro_triage.dashboard.html_overview_triage import triage_html
from astro_triage.dashboard.html_system import lang_root, seal, system_css
from astro_triage.dashboard.style_overview import overview_css
from astro_triage.dashboard.views_common import MODEL, tech_on

PRIOR_KEY = "overview_prior"


def _css(css: str) -> None:
    st.markdown(css, unsafe_allow_html=True)


def _md(html: str) -> None:
    st.markdown(lang_root(html), unsafe_allow_html=True)


def _pick_prior() -> str:
    """Native selector of the sky prior; a stale saved value falls back to the shipped prior."""
    if st.session_state.get(PRIOR_KEY) not in SKY_PRIORS:
        st.session_state[PRIOR_KEY] = SHIPPED_PRIOR
    chosen = st.segmented_control(
        "Prior de quasar da coluna “Ajustada ao prior”",
        SKY_PRIORS,
        key=PRIOR_KEY,
        required=True,
    )
    return chosen if chosen in SKY_PRIORS else SHIPPED_PRIOR


def overview(rel: D.Release) -> None:
    """Visão geral: prior-ruler result screen; the technical switch adds details."""
    tech = tech_on()
    d = load_page(
        rel,
        DC.headline(rel, MODEL),
        DC.purity_points(rel, MODEL, PRIOR),
        DC.abstention_curve(rel, MODEL),
        MODEL,
    )
    states = DC.prior_states(rel, MODEL)
    _css(system_css() + overview_css())
    _md(seal(rel.name))
    _md(inscription_html())
    selected = _pick_prior()
    _md(states_html(DC.shipped_state(rel, MODEL), states, selected))
    _md(ruler_html(states, selected))
    _md(accuracy_html(DC.accuracy_defs(rel, MODEL)))
    if tech:
        _md(data_box_html(d))
    _md(errors_html(d, tech))
    _md(triage_html(d, tech))
    if tech:
        st.markdown("<div class='at-sec'></div>", unsafe_allow_html=True)
        technical(rel)


def technical(rel: D.Release) -> None:
    st.markdown("### Comparação com o Gaia DSC")
    pop, prior = "c_sdsc", "P0"
    st.markdown(
        f"<span class='small'>{fmt.pt_int(D.population_rows(rel, pop))} objetos de teste com "
        "as três probabilidades do DSC; regime equilibrado (as três classes com o mesmo peso); "
        f"IC por bootstrap em regiões do céu, {rel.meta['n_boot']} repetições.</span>",
        unsafe_allow_html=True,
    )
    m = D.metrics_frame(rel, pop, prior, "prior_matched")
    f1 = m[m["metric"] == "macro_f1"]
    st.plotly_chart(
        C.f1_with_ci(f1, list(D.MODELS) + ["dsc_allosmod"]),
        width="stretch",
        config=NO_BAR,
    )
    d = D.paired_diffs_frame(rel, pop, prior, "prior_matched")
    d = d[(d["metric"] == "macro_f1") & d["usable"]]
    st.markdown(
        "**Diferença pareada de F1 macro** (intervalo sem o zero: diferença maior que o ruído):"
    )
    st.plotly_chart(C.paired_diffs(d), width="stretch", config=NO_BAR)

    st.markdown("### O que cada modelo usa")
    a, b = st.columns(2, gap="large")
    for col, model in zip((a, b), D.MODELS, strict=True):
        s = D.model_summary(rel, model)
        with col:
            st.markdown(f"**{C.MODEL_PT[model]}**")
            st.markdown(
                f"<span class='small'>{s['n_features']} variáveis · {s['best_iteration']} árvores · "
                f"temperatura {fmt.pt_num(s['temperature'])}</span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                C.importance(D.feature_importance(rel, model)),
                width="stretch",
                config=NO_BAR,
            )

    st.markdown("### Confiança e calibração")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Confiança × erro**")
        st.plotly_chart(
            C.confidence_hist(D.confidence_histogram(rel, MODEL)),
            width="stretch",
            config=NO_BAR,
        )
    with c2:
        st.markdown("**A probabilidade bate com a frequência?**")
        st.plotly_chart(
            C.calibration(DC.calibration_points(rel, MODEL)),
            width="stretch",
            config=NO_BAR,
        )
