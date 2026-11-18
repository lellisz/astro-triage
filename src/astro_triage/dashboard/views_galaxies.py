"""Galáxias page: verdict line and five quiet blocks; technical mode adds the band tables."""

from __future__ import annotations

import streamlit as st

from astro_triage.dashboard import compare_text as CT
from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import ui_state
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard.data_dl import ClassCell, DlDataError
from astro_triage.dashboard.data_galaxy_prior import (
    NETWORKS,
    crossmatch_note,
    dl_galaxy_cells,
    galaxy_error_rows,
    galaxy_rows,
    galaxy_verdict,
    load_galaxy_prior_block,
)
from astro_triage.dashboard.html_galaxies_tech import band_table, missing_note
from astro_triage.dashboard.html_galaxy_prior import TECH_DETAILS, page_html
from astro_triage.dashboard.html_system import details, system_css
from astro_triage.dashboard.ui_state import LINE_LABEL
from astro_triage.dashboard.views_common import MODEL

UNREADABLE = "O arquivo da avaliação do aprendizado profundo está ilegível; os vereditos ficam n/a."


def _dl_cells(rel: D.Release) -> tuple[dict[str, ClassCell | None], str | None]:
    """GALAXY cell of each network and a data-error text (no numbers) when the artifact is unreadable."""
    none = dict.fromkeys(NETWORKS)
    try:
        results = DC.load_results_dl(rel)
        return (none if results is None else dl_galaxy_cells(results)), None
    except (DlDataError, ValueError, OSError) as exc:
        text = str(exc) if isinstance(exc, DlDataError) else UNREADABLE
        return none, text


def galaxies_page(rel: D.Release) -> None:
    state = ui_state.normalize(st.session_state)
    shipped = load_galaxy_prior_block(rel, "as_shipped")
    matched = load_galaxy_prior_block(rel, "prior_matched")
    cells, dl_error = _dl_cells(rel)
    # The sentence judges the same P1/P2 intervals the bars show (rule "ajustada"), not as_shipped.
    verdict = galaxy_verdict(matched, cells)
    fixed_point = CT.fixed_point_line(
        LINE_LABEL, VR.RULE_PRIOR, str(VR.RULE_COMPLETENESS)
    )
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
        page_html(
            rel.name,
            state,
            verdict,
            galaxy_rows(shipped, matched, state),
            cells,
            rows,
            crossmatch_note(rel),
            dl_error,
            fixed_point,
        ),
        unsafe_allow_html=True,
    )
    if state.tech:
        tables = (
            band_table("Por brilho (magnitude G; menor = mais brilhante)", by_g)
            + band_table("Por distância (redshift z; maior = mais distante)", by_z)
            + missing_note(by_g)
        )
        st.markdown(details(TECH_DETAILS, tables), unsafe_allow_html=True)
