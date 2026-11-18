"""Streamlit dashboard. Run with:

    .venv\\Scripts\\python.exe -m streamlit run src/astro_triage/dashboard/app.py

Every number and sentence below is computed from one release in ``data/interim/<name>/``.
Only the page picked in the top navigation is rendered, so the other pages cost nothing.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import paths, sidebar, theme, ui_state
from astro_triage.dashboard.constants import DEFAULT_RELEASE_NAME
from astro_triage.dashboard.html_system import lang_root, seal, system_css
from astro_triage.dashboard.page_errors import describe_error, page_error
from astro_triage.dashboard.sidebar_info import sidebar_counts
from astro_triage.dashboard.views_dl import dl_page
from astro_triage.dashboard.views_galaxies import galaxies_page
from astro_triage.dashboard.views_method import method_page
from astro_triage.dashboard.views_overview import overview
from astro_triage.dashboard.views_sky import sky_page

_LOG = logging.getLogger(__name__)
PAGES = ("Visão geral", "Mapa do céu", "Galáxias", "Comparação com DL", "Método")
PAGE_PATHS = ("", "ceu", "galaxias", "comparacao-dl", "metodo")  # "" = default page
NARROW_PAGES = frozenset({"Método"})
# Widgets on pages that are not rendered lose their state; re-assigning keeps it across pages.
PERSISTED_KEYS = (
    "galaxy_scatter_on",
    "sky_mode",
    "sky_hide_below",
)


@st.cache_resource(show_spinner="Lendo a versão dos dados…", max_entries=1)
def get_release(name: str, stamp: int) -> D.Release:
    """``stamp`` is only a cache key: a rewritten artifact reloads the release."""
    return D.load_release(name)


@st.cache_data(show_spinner=False, max_entries=1)
def get_pixels(name: str, stamp: int) -> pd.DataFrame:
    return D.pixel_table(get_release(name, stamp))


NO_DATA_MESSAGE = (
    "Não encontrei a pasta de dados. Defina a variável de ambiente "
    f"{paths.ENV_VAR} com a pasta que contém interim/ e reinicie o painel."
)


def available_releases() -> list[str]:
    if not D.INTERIM_DIR.is_dir():
        return []
    found = sorted(
        p.parent.parent.name for p in D.INTERIM_DIR.glob("*/eval/results.json")
    )
    return found or [D.DEFAULT_RELEASE]


def shell_header(
    release: str, state: ui_state.UiState, show_partial_notice: bool = True
) -> None:
    """Seal and 'Regra · prior' line above every page; the partial notice is shown once in the content."""
    lines = [escape(ui_state.rule_prior_line(state))]
    notice = sidebar.partial_notice(state.prior) if show_partial_notice else None
    if notice:
        lines.append(escape(notice))
    body = "".join(f"<p class='hs-small'>{line}</p>" for line in lines)
    st.markdown(system_css(), unsafe_allow_html=True)
    st.markdown(lang_root(seal(release) + body), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="astro-triage",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    try:
        paths.require_dir(D.INTERIM_DIR)
    except paths.DataRootMissing:
        st.error(NO_DATA_MESSAGE)
        st.stop()
    for key in PERSISTED_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    releases = available_releases()
    if (
        not releases
    ):  # the guard above makes this unreachable; the selector never gets []
        st.error(NO_DATA_MESSAGE)
        st.stop()
    settings = st.popover("Ajustes")
    name, state = sidebar.controls(releases, DEFAULT_RELEASE_NAME, settings)
    stamp = DC.root_stamp(D.INTERIM_DIR / name)
    loaded = st.session_state.get("_loaded_release")
    if loaded is not None and loaded != name:
        get_release.clear()
        get_pixels.clear()
    st.session_state["_loaded_release"] = name
    try:
        rel = get_release(name, stamp)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Não foi possível ler a versão dos dados: {describe_error(exc)}.")
        st.stop()
    with settings:
        sidebar_counts(rel)
    renderers: tuple[Callable[[], None], ...] = (
        lambda: overview(rel, pages),
        lambda: sky_page(rel, get_pixels(rel.name, stamp)),
        lambda: galaxies_page(rel),
        lambda: dl_page(rel),
        lambda: method_page(rel),
    )
    pages = [
        st.Page(
            render,
            title=title,
            url_path=url_path or None,
            default=not url_path,
        )
        for render, title, url_path in zip(renderers, PAGES, PAGE_PATHS, strict=True)
    ]
    current = st.navigation(pages, position="top")
    st.markdown(
        theme.css(wide=current.title not in NARROW_PAGES), unsafe_allow_html=True
    )
    # The overview body carries the partial notice itself; the other pages rely on the header.
    shell_header(rel.name, state, show_partial_notice=current.title != PAGES[0])
    try:
        current.run()
    except Exception as exc:  # noqa: BLE001 - one broken page must not take the app down
        _LOG.exception("page %r failed", current.title)
        st.error(page_error(exc))


main()
