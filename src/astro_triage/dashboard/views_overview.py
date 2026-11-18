"""Overview tab and its technical details."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from astro_triage.dashboard import charts as C
from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import fmt, overview_model, overview_text, ui_state
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard.constants import NO_BAR
from astro_triage.dashboard.data_overview_page import PRIOR, load_page
from astro_triage.dashboard.html_overview import data_box_html
from astro_triage.dashboard.html_overview_plots import errors_html
from astro_triage.dashboard.html_overview_triage import triage_html
from astro_triage.dashboard.html_system import lang_root, system_css
from astro_triage.dashboard.kit_types import BarRow
from astro_triage.dashboard.overview_model import OverviewModel, QsoRow
from astro_triage.dashboard.style_overview import overview_css
from astro_triage.dashboard.views_common import MODEL

SHORTCUT_TITLES = ("Mapa do céu", "Galáxias", "Comparação com DL", "Método")
PURITY_TITLE = "Pureza de quasar nas quatro linhas"
COMPLETENESS_TITLE = "Completude de quasar nas quatro linhas"


def _css(css: str) -> None:
    st.markdown(css, unsafe_allow_html=True)


def _md(html: str) -> None:
    st.markdown(lang_root(html), unsafe_allow_html=True)


def _label(row: QsoRow) -> str:
    return f"{ui_state.RULE_LABEL[row.rule]} · {row.prior}"


def _bars(title: str, rows: tuple[QsoRow, ...], field: str) -> str:
    bars = [
        BarRow(
            _label(r),
            getattr(r, field),
            r.selected,
            overview_text.PARTIAL_TAG if r.partial else None,
        )
        for r in rows
    ]
    return hs.hbar_group(title, bars)


def page_html(model: OverviewModel, state: ui_state.UiState) -> str:
    """Answer, three cards, two bar groups, partial notice and method note, in reading order."""
    sel, titles, defs = (
        model.selected,
        overview_text.CARD_TITLES,
        overview_text.CARD_DEFS,
    )
    cards = hs.cards(
        [
            hs.number_card(titles["purity"], sel.purity, defs["purity"]),
            hs.number_card(
                titles["completeness"], sel.completeness, defs["completeness"]
            ),
            hs.number_card(titles["accuracy"], model.accuracy, defs["accuracy"]),
        ]
    )
    notice = hs.method_note(model.partial_notice) if model.partial_notice else ""
    return (
        f"{hs.verdict_oneline(overview_text.answer(model, state))}"
        f"{cards}{notice}"
        f"{_bars(PURITY_TITLE, model.rows, 'purity')}"
        f"{_bars(COMPLETENESS_TITLE, model.rows, 'completeness')}"
        f"{hs.method_note(overview_text.ACCURACY_NOTE)}"
    )


def _shortcuts(pages: Sequence[object] | None) -> None:
    by_title = {getattr(p, "title", None): p for p in pages or ()}
    cols = st.columns(len(SHORTCUT_TITLES))
    for col, title in zip(cols, SHORTCUT_TITLES, strict=True):
        if title in by_title:
            with col:
                st.page_link(by_title[title], label=title)


def overview(rel: D.Release, pages: Sequence[object] | None = None) -> None:
    """Visão geral: answer, three cards, bars of the four QSO rows; details under the technical switch."""
    state = ui_state.normalize(st.session_state)
    model = overview_model.build(
        DC.shipped_state(rel, MODEL),
        DC.prior_states(rel, MODEL),
        DC.accuracy_defs(rel, MODEL),
        state,
    )
    _css(system_css() + overview_css())
    _md(page_html(model, state))
    _shortcuts(pages)
    if state.tech:
        d = load_page(
            rel,
            DC.headline(rel, MODEL),
            DC.purity_points(rel, MODEL, PRIOR),
            DC.abstention_curve(rel, MODEL),
            MODEL,
        )
        _md(data_box_html(d))
        _md(errors_html(d, True))
        _md(triage_html(d, True))
        st.markdown("<div class='at-sec'></div>", unsafe_allow_html=True)
        technical(rel)


def technical(rel: D.Release) -> None:
    st.markdown("### Comparação com o Gaia DSC")
    pop, prior = "c_sdsc", "P0"
    st.markdown(
        f"<span class='small'>{fmt.pt_int(D.population_rows(rel, pop))} objetos de teste com "
        "as três probabilidades do DSC; regime equilibrado (as três classes com o mesmo peso); "
        f"IC por bootstrap em regiões do céu, {rel.meta['n_boot']} repetições.</span>",
        unsafe_allow_html=True,
    )
    m = D.metrics_frame(rel, pop, prior, "prior_matched")
    f1 = m[m["metric"] == "macro_f1"]
    st.plotly_chart(
        C.f1_with_ci(f1, list(D.MODELS) + ["dsc_allosmod"]),
        width="stretch",
        config=NO_BAR,
    )
    d = D.paired_diffs_frame(rel, pop, prior, "prior_matched")
    d = d[(d["metric"] == "macro_f1") & d["usable"]]
    st.markdown(
        "**Diferença pareada de F1 macro** (intervalo sem o zero: diferença maior que o ruído):"
    )
    st.plotly_chart(C.paired_diffs(d), width="stretch", config=NO_BAR)

    st.markdown("### O que cada modelo usa")
    a, b = st.columns(2, gap="large")
    for col, model in zip((a, b), D.MODELS, strict=True):
        s = D.model_summary(rel, model)
        with col:
            st.markdown(f"**{C.MODEL_PT[model]}**")
            st.markdown(
                f"<span class='small'>{s['n_features']} variáveis · {s['best_iteration']} árvores · "
                f"temperatura {fmt.pt_num(s['temperature'])}</span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                C.importance(D.feature_importance(rel, model)),
                width="stretch",
                config=NO_BAR,
            )

    st.markdown("### Confiança e calibração")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Confiança × erro**")
        st.plotly_chart(
            C.confidence_hist(D.confidence_histogram(rel, MODEL)),
            width="stretch",
            config=NO_BAR,
        )
    with c2:
        st.markdown("**A probabilidade bate com a frequência?**")
        st.plotly_chart(
            C.calibration(DC.calibration_points(rel, MODEL)),
            width="stretch",
            config=NO_BAR,
        )
