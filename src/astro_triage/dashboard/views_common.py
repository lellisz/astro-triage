"""Pieces shared by the dashboard views."""

from __future__ import annotations

import streamlit as st

from astro_triage.dashboard import ui_state

MODEL = "lgbm_full"


def tech_on() -> bool:
    """Whether the 'Modo técnico' switch of the sidebar is on (details for specialists)."""
    return ui_state.normalize(st.session_state).tech
