"""Comparação page (LightGBM x deep learning): the Streamlit page over a pure HTML builder.

Verdicts come from ``compare_model`` (the shared rule, ``verdict_rule``) and the numbers from
``results_dl.json`` through ``data_dl``. Verdict texts live in ``compare_text``. The page has no
technical switch and ignores the sidebar rule and prior: it shows the fixed point of the protocol.
"""

from __future__ import annotations

import streamlit as st

from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import compare_text as CT
from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import data_dl as DD
from astro_triage.dashboard import fmt
from astro_triage.dashboard import html_dl as HD
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard.constants import ADR_NUMBER
from astro_triage.dashboard.ui_state import LINE_LABEL
from astro_triage.evaluation import dl_compare

GRID_LEAD = "Quatro células: classe por rede, cada uma contra o LightGBM."
PURITY_LEAD = "Barras a partir de zero: a mediana das sementes e, na linha fina, o mínimo e o máximo."
UNREADABLE = "O arquivo da avaliação do aprendizado profundo está ilegível."
METHOD_TITLE = f"Nota de método (ADR {ADR_NUMBER:04d})"


def _md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def _purity_rows(model: CM.CompareModel) -> list[dict]:
    return [
        {
            "name": CT.REFERENCE_NAME
            if s == CM.REFERENCE_SYSTEM
            else HD.system_label(s),
            "low": lo,
            "median": mid,
            "high": hi,
            "detail": (
                f"mediana de {fmt.pt_int(dl_compare.N_SEEDS)} sementes {fmt.pt_pct(mid, 2)}; "
                f"mínimo a máximo {fmt.pt_pct(lo, 2)} a {fmt.pt_pct(hi, 2)} ({CT.NOT_CI_LABEL})"
            ),
        }
        for s, lo, mid, hi in model.purity
    ]


def _cell_html(g: CM.GridCell) -> str:
    title = f"{HD.class_label(g.cls)} · {g.net} × {CT.REFERENCE_NAME}"
    note = hs.method_note(CT.POWER_PHRASE) if g.underpowered else ""
    return f"<div>{hs.verdict_card(title, g.verdict)}{note}</div>"


def page_html(model: CM.CompareModel, seal: str) -> str:
    """Seal, fixed-point line, answer, 2x2 grid, absolute purity and the method note."""
    per_row = len(model.pairs)
    rows = [model.grid[i : i + per_row] for i in range(0, len(model.grid), per_row)]
    grid = "".join(hs.pair(*(_cell_html(g) for g in row)) for row in rows)
    purity = hs.section(
        f"Pureza absoluta de {HD.class_label('QSO').lower()}",
        PURITY_LEAD,
        hs.abs_bars(_purity_rows(model)),
    )
    method = hs.section(METHOD_TITLE, "", hs.method_note(VR.rule_text()))
    parts = [
        seal,
        hs.rule_prior_line(CT.fixed_point_line(LINE_LABEL, model.prior, model.level)),
        hs.verdict_oneline(CT.answer(model)),
        hs.section("Veredito por classe e rede", GRID_LEAD, grid),
        purity,
        method,
    ]
    return hs.lang_root("".join(parts))


def dl_page(rel: D.Release) -> None:
    try:
        results = DC.load_results_dl(rel)
    except (ValueError, OSError):
        _md(HD.error_html(UNREADABLE))
        return
    if results is None:
        _md(HD.unavailable_html())
        return
    try:
        model = CM.build(results)
        head = DD.headline(results)
        descriptive = DD.descriptive(results, head)
        limitations = DD.load_limitations(rel)
        below = sum(g.state is DD.ClassVerdict.NA for g in model.grid)
        seal = HD.seal_html(
            rel.name,
            DD.adr_status(),
            DD.freeze_exists(rel),
            below,
            limitations is not None,
        )
        page = page_html(model, seal)
    except DD.DlDataError as exc:
        _md(HD.error_html(str(exc)))
        return
    _md(hs.system_css() + page)
    with st.expander(HD.DESCRIPTIVE_TITLE):
        _md(HD.descriptive_html(descriptive, head.prior, head.level))
    with st.expander(HD.LIMITS_TITLE):
        _md(HD.limitations_html(limitations))
