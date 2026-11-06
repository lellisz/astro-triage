"""Streamlit dashboard. Run with:

    .venv\\Scripts\\python.exe -m streamlit run src/astro_triage/dashboard/app.py

Every number and sentence below is computed from one release in ``data/interim/<name>/``.
Only the page picked in the selector is rendered, so the other pages cost nothing.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import theme
from astro_triage.dashboard.constants import DEFAULT_RELEASE_NAME
from astro_triage.dashboard.page_errors import describe_error, page_error
from astro_triage.dashboard.sidebar_info import sidebar_counts
from astro_triage.dashboard.views_common import tech_toggle
from astro_triage.dashboard.views_galaxies import galaxies_page
from astro_triage.dashboard.views_layer2 import layer2_page
from astro_triage.dashboard.views_method import method_page
from astro_triage.dashboard.views_overview import overview
from astro_triage.dashboard.views_sky import sky_page

_LOG = logging.getLogger(__name__)
PAGES = ("Visão geral", "Mapa do céu", "Galáxias", "Método")
LAYER2_PAGE = "Camada 2"
NARROW_PAGES = frozenset({"Método"})
# Widgets on pages that are not rendered lose their state; re-assigning keeps it across pages.
PERSISTED_KEYS = (
    "galaxy_scatter_on",
    "sky_mode",
    "sky_hide_below",
    "tech",
)


@st.cache_resource(show_spinner="Lendo a versão dos dados…", max_entries=1)
def get_release(name: str, stamp: int) -> D.Release:
    """``stamp`` is only a cache key: a rewritten artifact reloads the release."""
    return D.load_release(name)


@st.cache_data(show_spinner=False, max_entries=1)
def get_pixels(name: str, stamp: int) -> pd.DataFrame:
    return D.pixel_table(get_release(name, stamp))


def available_releases() -> list[str]:
    found = sorted(
        p.parent.parent.name for p in D.INTERIM_DIR.glob("*/eval/results.json")
    )
    return found or [D.DEFAULT_RELEASE]


def main() -> None:
    st.set_page_config(page_title="astro-triage", page_icon="✦", layout="wide")
    for key in PERSISTED_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    wide = st.session_state.get("page") not in NARROW_PAGES
    st.markdown(theme.css(wide=wide), unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### astro-triage")
        releases = available_releases()
        default = (
            releases.index(DEFAULT_RELEASE_NAME)
            if DEFAULT_RELEASE_NAME in releases
            else 0
        )
        name = st.selectbox("Versão dos dados", releases, index=default)
        if DEFAULT_RELEASE_NAME not in releases:
            st.warning(
                f"A versão {DEFAULT_RELEASE_NAME} não foi encontrada; usando {name}."
            )
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
    sidebar_counts(rel)
    try:
        layer2 = DC.load_layer2(rel)
    except (OSError, ValueError, KeyError) as exc:
        layer2 = None
        st.sidebar.error(f"Avaliação da Camada 2 ilegível: {describe_error(exc)}.")
    renderers: dict[str, Callable[[], None]] = {
        PAGES[0]: lambda: overview(rel),
        PAGES[1]: lambda: sky_page(rel, get_pixels(rel.name, stamp)),
        PAGES[2]: lambda: galaxies_page(rel),
        PAGES[3]: lambda: method_page(rel),
    }
    if layer2 is not None:
        renderers[LAYER2_PAGE] = lambda: layer2_page(layer2)
    if st.session_state.get("page") not in renderers:
        st.session_state["page"] = PAGES[0]
    kw = {"key": "page", "required": True, "label_visibility": "collapsed"}
    nav_col, tech_col = st.columns([5, 1.4], vertical_alignment="center")
    with nav_col:
        page = st.segmented_control("Página", list(renderers), **kw)
    with tech_col:
        tech_toggle()
    try:
        renderers[page or PAGES[0]]()
    except Exception as exc:  # noqa: BLE001 - one broken page must not take the app down
        _LOG.exception("page %r failed", page)
        st.error(page_error(exc))


main()
