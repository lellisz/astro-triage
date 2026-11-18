"""Sidebar controls of the shell: decision rule, prior, technical mode and data version.

Every widget has its own key (``_w_*``); an ``on_change`` copies it to the keys of
``ui_state``, which the views read. The widget keys are re-seeded from the normalized state on
each run, so a rule that disables a prior also corrects the prior widget.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import streamlit as st

from astro_triage.dashboard import html_system, ui_state
from astro_triage.dashboard.html_overview import partial_note
from astro_triage.dashboard.ui_state import Rule, UiState

KEYS: Final = ui_state.KEYS
W_RULE: Final = "_w_rule"
W_PRIOR: Final = "_w_prior"
W_TECH: Final = "_w_tech"


def current() -> UiState:
    return ui_state.normalize(st.session_state)


def partial_notice(prior: str) -> str | None:
    """Warning for a prior that covers only part of the sky (see the ADR on the split); None otherwise."""
    return partial_note(prior) if ui_state.is_partial(prior) else None


def _copy(widget_key: str, state_key: str) -> None:
    st.session_state[state_key] = st.session_state[widget_key]


def _rule_label(value: str) -> str:
    return ui_state.RULE_LABEL[Rule(value)]


def controls(releases: Sequence[str], default_release: str) -> tuple[str, UiState]:
    """Render the controls inside the sidebar; ``releases`` must not be empty."""
    state = current()
    st.session_state[W_RULE] = state.rule.value
    st.session_state[W_PRIOR] = state.prior
    st.session_state[W_TECH] = state.tech
    with st.sidebar:
        st.markdown("### astro-triage")
        st.radio(
            "Regra de decisão",
            [r.value for r in Rule],
            format_func=_rule_label,
            key=W_RULE,
            on_change=_copy,
            args=(W_RULE, ui_state.KEY_RULE),
        )
        restricted = len(ui_state.enabled_priors(state.rule)) < len(ui_state.PRIORS)
        st.segmented_control(
            "Prior de quasar",
            ui_state.PRIORS,
            key=W_PRIOR,
            required=True,
            disabled=restricted,
            on_change=_copy,
            args=(W_PRIOR, ui_state.KEY_PRIOR),
        )
        notes = [
            f"{p}: {note}"
            for p in ui_state.PRIORS
            if (note := ui_state.disabled_note(state.rule, p))
        ]
        if notes:
            st.caption(" · ".join(notes))
        notice = partial_notice(state.prior)
        if notice:
            st.caption(notice)
        st.toggle(
            "Modo técnico",
            key=W_TECH,
            on_change=_copy,
            args=(W_TECH, ui_state.KEY_TECH),
        )
        default = releases.index(default_release) if default_release in releases else 0
        name = st.selectbox("Versão dos dados", releases, index=default)
        if default_release not in releases:
            st.warning(f"A versão {default_release} não foi encontrada; usando {name}.")
        st.markdown(html_system.seal(name), unsafe_allow_html=True)
    return name, state
