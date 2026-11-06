"""Opt-in SDSS cutouts: the browser fetches them, nothing is downloaded or stored here."""

from __future__ import annotations

import html
import math

import streamlit as st

from astro_triage.dashboard import charts as C
from astro_triage.dashboard import data as D
from astro_triage.dashboard import fmt
from astro_triage.dashboard.views_common import MODEL

MODES = ("Monocromático", "Cor original")
_GRAYSCALE = "filter: grayscale(1);"


def _img(url: str, mono: bool, alt: str) -> str:
    style = f"width: 100%; {_GRAYSCALE if mono else ''}"
    return (
        f"<img src='{html.escape(url, quote=True)}' alt='{html.escape(alt, quote=True)}' "
        f"style='{style}' referrerpolicy='no-referrer' loading='lazy'>"
    )


def _caption(pred: str, correct: bool, confidence: float, z: float) -> str:
    z_text = "sem z" if math.isnan(z) else f"z={fmt.pt_num(z, 3)}"
    true = html.escape(C.CLASS_PT["GALAXY"])
    shown = html.escape(C.CLASS_PT.get(pred, pred))
    return (
        f"classe verdadeira: {true}; prevista: {shown}<br>"
        f"{'acerto' if correct else 'erro'} · confiança {fmt.pt_num(confidence, 2)} · {z_text}"
    )


def gallery_section(rel: D.Release) -> None:
    st.subheader("Como são, a olho")
    on = st.toggle(
        "Mostrar imagens do SDSS (o navegador consulta o site do SDSS)",
        value=False,
        key="gallery_on",
    )
    if not on:
        return
    mode = st.radio(
        "Cor dos recortes", MODES, index=0, horizontal=True, key="gallery_mode"
    )
    g = D.gallery_sample(rel, MODEL)
    cols = st.columns(4)
    for i, r in enumerate(g.itertuples()):
        with cols[i % 4]:
            st.markdown(
                _img(r.url, mode == MODES[0], f"Recorte do SDSS, objeto {i + 1}"),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='small {'' if r.correct else 'err'}'>{_caption(r.pred, r.correct, r.confidence, r.sdss_z)}</span>",
                unsafe_allow_html=True,
            )
    st.markdown(
        "<span class='small'>Imagens: SDSS SkyServer.</span>", unsafe_allow_html=True
    )
