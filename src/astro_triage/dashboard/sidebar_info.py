"""Sidebar notes about the loaded release: test size, discarded rows and count mismatches."""

from __future__ import annotations

import streamlit as st

from astro_triage.dashboard import fmt
from astro_triage.dashboard.data_release import Release


def sidebar_notes(rel: Release) -> list[str]:
    """Plain-text notes (pt-BR); the n_test mismatch is stated neutrally, not as an alarm."""
    n_test = len(rel.test)
    notes = [f"{fmt.pt_int(n_test)} objetos de teste"]
    if rel.n_unscored > 0:
        notes.append(f"{fmt.pt_int(rel.n_unscored)} objetos sem escore descartados")
    n_meta = rel.meta.get("n_test")
    if n_meta is not None and n_meta != n_test:
        notes.append(
            f"A avaliação registra {fmt.pt_int(n_meta)} objetos de teste; "
            f"esta tela usa {fmt.pt_int(n_test)}"
        )
    return notes


def sidebar_counts(rel: Release) -> None:
    html = "".join(f"<span class='small'>{n}</span><br>" for n in sidebar_notes(rel))
    with st.sidebar:
        st.markdown(html, unsafe_allow_html=True)
