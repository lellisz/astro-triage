"""Sky map page: simple reading by default, selector and tables in technical mode."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import (
    data_sky,
    fmt,
    method_notes,
    sky_map_html,
    sky_text,
)
from astro_triage.dashboard.html_system import (
    details,
    lang_root,
    small_note,
    system_css,
)
from astro_triage.dashboard.views_common import MODEL, tech_on

MODE_KEY = "sky_mode"
HIDE_KEY = "sky_hide_below"
MODES = (sky_text.MODE_RATE, sky_text.MODE_COUNT)


def _md(html: str) -> None:
    st.markdown(lang_root(html), unsafe_allow_html=True)


def _pick_mode() -> bool:
    """True for the error-rate map. Technical mode only; a stale saved value falls back to rate."""
    if st.session_state.get(MODE_KEY) not in MODES:
        st.session_state[MODE_KEY] = MODES[0]
    chosen = st.segmented_control(
        "O que o mapa mostra", MODES, key=MODE_KEY, required=True
    )
    return chosen != sky_text.MODE_COUNT


def sky_page(rel: D.Release, px: pd.DataFrame) -> None:
    stats = DC.test_pixel_stats(rel, MODEL, px)
    floor = int(rel.meta["floor"]["min_objects"])
    tech = tech_on()
    by_rate, hide = True, False
    if tech:
        by_rate = _pick_mode()
        if by_rate:
            hide = st.checkbox(
                f"Ocultar regiões com menos de {fmt.pt_int(floor)} objetos",
                value=False,
                key=HIDE_KEY,
            )
    n_hidden = int((stats["n"] < floor).sum())
    st.markdown(system_css(), unsafe_allow_html=True)
    _md(
        sky_text.header_html(
            sky_text.TITLE,
            sky_text.lead(len(stats)),
            sky_text.how(by_rate, n_hidden, floor, hide),
        )
    )
    _md(sky_map_html.lead_line(sky_text.map_lead(stats, floor)))
    _md(sky_map_html.build_map(stats, by_rate, floor, hide))
    _md(small_note(method_notes.NEIGHBOUR_NOTE))
    if not tech:
        return
    top = data_sky.top_error_regions(stats, floor)
    _md(
        details(sky_text.BANDS_TITLE, sky_text.bands_body(D.lat_band_table(stats)))
        + details(sky_text.top_title(floor), sky_text.top_body(top, floor))
    )
    chance = D.chance_share_of_pixels_with_error(
        stats, float(stats["errors"].sum()) / float(stats["n"].sum())
    )
    with_error = int((stats["errors"] > 0).sum())
    _md(
        small_note(
            f"{fmt.pt_int(with_error)} regiões têm ao menos um erro "
            f"({fmt.pt_pct(with_error / len(stats), 0)}); ao acaso, "
            f"seriam ≈{fmt.pt_pct(chance, 0)}."
        )
    )
