"""Galáxias page: verdict line and five quiet blocks; technical mode adds the band tables."""

from __future__ import annotations

import streamlit as st

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard.data_dl import ClassCell, DlDataError
from astro_triage.dashboard.data_galaxy_prior import (
    NETWORKS,
    crossmatch_note,
    dl_galaxy_cells,
    galaxy_error_rows,
    galaxy_verdict,
    load_galaxy_prior_block,
)
from astro_triage.dashboard.html_galaxies_tech import band_table, missing_note
from astro_triage.dashboard.html_galaxy_prior import TECH_DETAILS, page_html
from astro_triage.dashboard.html_system import details, system_css
from astro_triage.dashboard.views_common import MODEL, tech_on


def _dl_cells(rel: D.Release) -> dict[str, ClassCell | None]:
    """GALAXY cell of each network; empty when the DL artifact is absent or has no headline."""
    results = DC.load_results_dl(rel)
    if results is None:
        return dict.fromkeys(NETWORKS)
    try:
        return dl_galaxy_cells(results)
    except DlDataError:
        return dict.fromkeys(NETWORKS)


def galaxies_page(rel: D.Release) -> None:
    block = load_galaxy_prior_block(rel)
    cells = _dl_cells(rel)
    verdict = galaxy_verdict(block, cells)
    by_g = DC.error_rate_table(rel, MODEL, "GALAXY", "G", D.GALAXY_G_EDGES)
    by_z = DC.error_rate_table_z(rel, MODEL)
    rows = galaxy_error_rows(
        DC.galaxy_totals(rel, MODEL),
        by_g,
        by_z,
        int(rel.meta["floor"]["min_objects"]),
    )
    st.markdown(system_css(), unsafe_allow_html=True)
    st.markdown(
        page_html(rel.name, verdict, block, cells, rows, crossmatch_note(rel)),
        unsafe_allow_html=True,
    )
    if tech_on():
        tables = (
            band_table("Por brilho (magnitude G; menor = mais brilhante)", by_g)
            + band_table("Por distância (redshift z; maior = mais distante)", by_z)
            + missing_note(by_g)
        )
        st.markdown(details(TECH_DETAILS, tables), unsafe_allow_html=True)
