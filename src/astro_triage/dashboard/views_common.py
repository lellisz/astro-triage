"""Pieces shared by the dashboard views."""

from __future__ import annotations

import streamlit as st

MODEL = "lgbm_full"


def tech_on() -> bool:
    """Whether the 'Modo técnico' switch is on (details for specialists)."""
    return bool(st.session_state.get("tech", False))


def tech_toggle() -> None:
    """Render the 'Modo técnico' switch; its state lives in ``st.session_state['tech']``."""
    st.toggle("Modo técnico", key="tech")
