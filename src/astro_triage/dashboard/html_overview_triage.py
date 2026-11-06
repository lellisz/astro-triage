"""HTML of the Visão geral triage section: two points on a scatter, no curve between them."""

from __future__ import annotations

import math

from astro_triage.dashboard import fmt, html_kit
from astro_triage.dashboard.data_overview_page import PageData
from astro_triage.dashboard.html_overview_plots import PERCENT, _e, _ticks

Y_FLOOR = 0.96  # the plot starts here (stated in its note) unless a point is lower
X_FLOOR = 0.80
FLOOR_STEP = 0.01
TICK_COUNT = 5
HOW_GAIN = "quanto mais ele deixa para o espectro, mais acerta no que decide."
LABEL_FLIP = 50.0  # points right of this share of the plot get a label that ends at the point


def label_side(xa: float) -> str:
    """CSS class anchoring the label of the first point: it grows right from a point near the
    left edge and left from a point near the right edge, so it never leaves the plot."""
    return "ao-la-r" if xa > LABEL_FLIP else "ao-la-l"


def _floor(value: float, ceiling: float) -> float:
    """``ceiling`` or, when ``value`` is below it, ``value`` rounded down to a whole percent."""
    return min(ceiling, math.floor(value / FLOOR_STEP) * FLOOR_STEP)


def triage_html(d: PageData, tech: bool) -> str:
    """Section 'Decidir sozinho ou pedir um espectro': two points, no curve between them."""
    t = d.triage
    if t is None:
        return html_kit.section(
            "Decidir sozinho ou pedir um espectro",
            "Sem curva de abstenção nesta versão.",
            None,
        )
    thr, cov, acc = fmt.pt_num(t["threshold"], 2), t["coverage"], t["accuracy_kept"]
    y0 = _floor(min(acc, d.accuracy), Y_FLOOR)
    x0 = _floor(cov, X_FLOOR)
    head = html_kit.section(
        "Decidir sozinho ou pedir um espectro",
        f"Com limiar {thr} o modelo decide {fmt.pt_pct(cov)} dos objetos e acerta "
        f"{fmt.pt_pct(acc)} deles. Os outros {fmt.pt_int(t['n_set_aside'])} vão para um espectro.",
        HOW_GAIN if acc > d.accuracy else None,
    )
    ya = (acc - y0) / (1 - y0) * PERCENT
    yb = (d.accuracy - y0) / (1 - y0) * PERCENT
    xa = (cov - x0) / (1 - x0) * PERCENT
    y_ticks = "".join(
        f"<span class='ao-yt' style='bottom:{pos:.2f}%'>{_e(fmt.pt_pct(v, 0))}</span>"
        for pos, v in _ticks(y0, 1.0, TICK_COUNT)
    )
    x_axis = html_kit.axis(
        [(pos, fmt.pt_pct(v, 0)) for pos, v in _ticks(x0, 1.0, TICK_COUNT)]
    )
    label_a = f"limiar {thr} · decide {fmt.pt_pct(cov)} · acerta {fmt.pt_pct(acc)}"
    plot = (
        f"<div class='ao-pa' style='left:{xa:.2f}%;bottom:{ya:.2f}%'></div>"
        f"<div class='ao-la {label_side(xa)}' style='left:{xa:.2f}%;bottom:{ya:.2f}%'>{_e(label_a)}</div>"
        f"<div class='ao-pb' style='left:{PERCENT:.2f}%;bottom:{yb:.2f}%'></div>"
        f"<div class='ao-lb' style='bottom:{yb:.2f}%'>decidir tudo · acerta {_e(fmt.pt_pct(d.accuracy, 2))}</div>"
    )
    chart = (
        "<div class='at-ov'><div class='at-in ao-tri'>"
        f"<div class='ao-triy'>{y_ticks}</div>"
        f"<div><div class='ao-plot'>{plot}</div><div class='ao-xt'>{x_axis}</div>"
        "<div class='at-cap ao-xc'>parte decidida pelo modelo</div></div></div></div>"
    )
    note = (
        f"<p class='at-cap ao-note'>O eixo vertical começa em {_e(fmt.pt_pct(y0, 0))}. "
        "São dois pontos medidos, sem curva entre eles.</p>"
    )
    box = ""
    if tech:
        decided = d.n_test - int(t["n_set_aside"])
        wrong = int(t["errors_total"]) - int(t["errors_set_aside"])
        box = html_kit.box(
            "Detalhe técnico",
            "<p>"
            + _e(
                f"Limiar de confiança {thr}: decide {fmt.pt_int(decided)} objetos "
                f"({fmt.pt_pct(cov)}) e erra {fmt.pt_int(wrong)} deles. Os "
                f"{fmt.pt_int(t['n_set_aside'])} restantes vão para espectro e concentram "
                f"{fmt.pt_int(t['errors_set_aside'])} dos {fmt.pt_int(t['errors_total'])} erros."
            )
            + "</p>",
        )
    return head + chart + note + box
