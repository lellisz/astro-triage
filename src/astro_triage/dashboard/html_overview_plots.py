"""HTML plots of the Visão geral: purity/completeness rows, error bars and the triage scatter.

Pure builders (no streamlit): each takes a ``PageData`` and returns one single-line HTML string.
"""

from __future__ import annotations

from html import escape

from astro_triage.dashboard import fmt, html_kit, theme
from astro_triage.dashboard.data_headline import PAIR_NAMES
from astro_triage.dashboard.data_overview_page import PageData

PERCENT = 100.0
MIN_CI_WIDTH = 0.4  # keeps a very narrow interval visible, in plot percent
SIMPLE_ERROR_BARS = 3
PLURAL = {"STAR": "Estrelas", "GALAXY": "Galáxias", "QSO": "Quasares"}


def _e(text: object) -> str:
    return escape(str(text), quote=True)


def _ticks(lo: float, hi: float, count: int) -> list[tuple[float, float]]:
    """``count`` evenly spaced ``(position in %, value)`` pairs from ``lo`` to ``hi``."""
    steps = count - 1
    return [(i / steps * PERCENT, lo + (hi - lo) * i / steps) for i in range(count)]


def _purity_rows(d: PageData, tech: bool) -> list[str]:
    rows: list[str] = []
    for point, name, sub in (
        (d.shipped, "Modelo original", "teste"),
        (d.matched, "Quasares tratados como raros", "teste"),
    ):
        if not point:
            continue
        p, c = point["purity"] * PERCENT, point["completeness"] * PERCENT
        plot = html_kit.span(min(p, c), max(p, c))
        if tech:
            lo, hi = point["purity_lo"] * PERCENT, point["purity_hi"] * PERCENT
            plot += html_kit.span(lo, max(hi, lo + MIN_CI_WIDTH), ci=True)
        plot += html_kit.dot(p) + html_kit.dot(c, hollow=True)
        approx = "≈ " if point is d.matched else ""
        rows.append(
            html_kit.dot_row(
                name,
                sub,
                plot,
                fmt.pt_pct(point["purity"]),
                approx + fmt.pt_pct(point["completeness"]),
            )
        )
    v = d.validation
    if tech and v:
        p, c = v["purity"] * PERCENT, v["completeness"] * PERCENT
        plot = (
            html_kit.span(min(p, c), max(p, c))
            + html_kit.dot(p)
            + html_kit.dot(c, hollow=True)
        )
        name = f"Completude fixa em {fmt.pt_pct(v['completeness'], 0)}"
        rows.append(
            html_kit.dot_row(
                name,
                "só validação",
                plot,
                "≈ " + fmt.pt_pct(v["purity"]),
                fmt.pt_pct(v["completeness"]),
            )
        )
    return rows


def purity_html(d: PageData, tech: bool) -> str:
    """Section 'Quando o modelo diz quasar, ele acerta?' with two rows (three in technical mode)."""
    how = (
        "círculo cheio = acerta quando aponta (pureza). Círculo vazio = acha dos que "
        "existem (completude)."
    )
    if tech:
        how += f" A linha azul é a incerteza ({fmt.pt_ci_label()})."
    head = html_kit.section(
        "Quando o modelo diz “quasar”, ele acerta?",
        "Acertar muito e acertar quando aponta são coisas diferentes: quasares são raros.",
        how,
    )
    rows = _purity_rows(d, tech)
    if not rows:
        return (
            head
            + "<p class='at-cap'>Sem pureza e completude de quasar utilizáveis nesta versão.</p>"
        )
    axis = html_kit.axis([(pos, fmt.pt_pct(v, 0)) for pos, v in _ticks(0.0, 1.0, 6)])
    header = (
        f"<div class='at-rowp ao-hdrrow'><div></div>{axis}"
        "<div class='at-hdr at-num'>acerta quando aponta</div>"
        "<div class='at-hdr at-num'>acha dos que existem</div></div>"
    )
    out = f"{head}<div class='at-ov'><div class='at-in'>{header}{''.join(rows)}</div></div>"
    if tech and d.validation:
        c = fmt.pt_num(d.validation["completeness"], 2)
        out += (
            f"<p class='at-cap ao-note'>≈ valor aproximado. A pureza com completude fixa em {_e(c)} "
            "só foi medida em validação. O prior vem do classificador do Gaia (Delchambre+2023): "
            "é uma premissa, não a fração real de quasares no céu.</p>"
        )
    return out


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
