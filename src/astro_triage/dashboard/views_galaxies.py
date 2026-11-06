"""Galáxias page: four bars in simple mode; log tables and opt-in extras in technical mode."""

from __future__ import annotations

import streamlit as st

from astro_triage.dashboard import charts as C
from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import data_galaxies_z as GZ
from astro_triage.dashboard import fmt
from astro_triage.dashboard.constants import NO_BAR
from astro_triage.dashboard.html_galaxies import simple_html
from astro_triage.dashboard.html_galaxies_tech import tech_html
from astro_triage.dashboard.views_common import MODEL, tech_on
from astro_triage.dashboard.views_galaxies_gallery import gallery_section


def _scatter_section(rel: D.Release) -> None:
    with st.expander("Redshift × magnitude G, galáxia por galáxia"):
        st.markdown(D.z_low_sentence(DC.z_low_summary(rel, MODEL)))
        on = st.checkbox(
            "Mostrar galáxia por galáxia",
            value=False,
            key="galaxy_scatter_on",
        )
        if not on:
            return
        full = DC.galaxy_scatter(rel, MODEL)
        sample = GZ.scatter_sample(full)
        n_ok_shown = int((~sample["error"]).sum())
        st.markdown(
            f"Todos os {fmt.pt_int(int(full['error'].sum()))} erros e uma amostra de "
            f"{fmt.pt_int(n_ok_shown)} acertos (de {fmt.pt_int(len(full))} galáxias)."
        )
        st.plotly_chart(C.galaxy_scatter(sample), width="stretch", config=NO_BAR)


def galaxies_page(rel: D.Release) -> None:
    tot = DC.galaxy_totals(rel, MODEL)
    by_g = DC.error_rate_table(rel, MODEL, "GALAXY", "G", D.GALAXY_G_EDGES)
    by_z = DC.error_rate_table_z(rel, MODEL)
    floor = int(rel.meta["floor"]["min_objects"])
    if not tech_on():
        st.markdown(simple_html(tot, by_g, by_z, floor), unsafe_allow_html=True)
        return
    st.markdown(tech_html(tot, by_g, by_z, floor), unsafe_allow_html=True)
    _scatter_section(rel)
    gallery_section(rel)
