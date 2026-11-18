"""HTML plots of the Visão geral: purity/completeness rows, error bars and the triage scatter.

Pure builders (no streamlit): each takes a ``PageData`` and returns one single-line HTML string.
"""

from __future__ import annotations

from html import escape

from astro_triage.dashboard import fmt, html_kit, theme
from astro_triage.dashboard.data_headline import PAIR_NAMES
from astro_triage.dashboard.data_overview_page import PageData

PERCENT = 100.0
SIMPLE_ERROR_BARS = 3
PLURAL = {"STAR": "Estrelas", "GALAXY": "Galáxias", "QSO": "Quasares"}


def _e(text: object) -> str:
    return escape(str(text), quote=True)


def _ticks(lo: float, hi: float, count: int) -> list[tuple[float, float]]:
    """``count`` evenly spaced ``(position in %, value)`` pairs from ``lo`` to ``hi``."""
    steps = count - 1
    return [(i / steps * PERCENT, lo + (hi - lo) * i / steps) for i in range(count)]


def errors_html(d: PageData, tech: bool) -> str:
    """Section '{pct} dos {n} erros são o mesmo engano': 3 largest bars (6 in technical mode)."""
    directions = sorted(d.directions, key=lambda row: -row[2])
    total = sum(n for _, _, n in directions)
    if total <= 0:
        return html_kit.section(
            "Sem erros no teste", "O modelo não errou nenhum objeto de teste.", None
        )
    top_true, top_pred, top_n = directions[0]
    head = html_kit.section(
        f"{fmt.pt_pct(top_n / total, 0)} dos {fmt.pt_int(total)} erros são o mesmo engano",
        f"{PLURAL[top_true]} que o modelo tomou por {PAIR_NAMES[top_pred]}.",
        "cada barra é um tipo de engano, de verdadeiro para previsto. O comprimento é a quantidade.",
    )
    shown = directions if tech else directions[:SIMPLE_ERROR_BARS]
    bars = "".join(
        html_kit.h_bar(
            f"{PAIR_NAMES[t]} → {PAIR_NAMES[p]}",
            n / top_n,
            fmt.pt_int(n),
            theme.ACCENT if i == 0 else theme.SOFT,
            fmt.pt_pct(n / total),
        )
        for i, (t, p, n) in enumerate(shown)
    )
    return f"{head}<div class='at-ov'><div class='at-in'>{bars}</div></div>"
