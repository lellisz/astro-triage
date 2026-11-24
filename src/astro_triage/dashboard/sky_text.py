"""Texts and tables of the sky page. Pure: every number comes from the data frames."""

from __future__ import annotations

from html import escape
from typing import Final

import pandas as pd

from astro_triage.dashboard import fmt, html_kit
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard.constants import ERROR_HIGH
from astro_triage.dashboard.data_sky import (
    BAND_EDGES_DEG,
    HEALPIX_NSIDE,
    chance_share_of_pixels_with_error,
)

MODE_RATE: Final = "Taxa de erro"
MODE_COUNT: Final = "Quantidade de objetos"
TITLE: Final = "Onde o modelo erra, região por região"
HOW_LABEL: Final = "Como ler:"
BANDS_TITLE: Final = "Por faixa de latitude galáctica"


CARD_REGIONS: Final = "Regiões de teste"
CARD_BELOW: Final = "Regiões abaixo do piso"
CARD_MAX: Final = "Maior erro numa região"
POSITION_NOTE: Final = (
    "A posição de cada ponto é a do centro da região (uma posição sintética), "
    "não a de um objeto."
)


def count_card(title: str, shown: str, definition: str) -> str:
    """Count card in the kit's card markup: a count is not an estimate, so it carries no interval."""
    return (
        f"<div class='hs2-card' lang='{hs.PAGE_LANG}'>"
        f"<div class='hs2-ct'>{escape(title)}</div>"
        f"<div class='hs2-cv'>{escape(shown)}</div>"
        f"<p class='hs2-cd'>{escape(definition)}</p></div>"
    )


def regions_def(n_objects: int) -> str:
    return f"Regiões do céu com objetos de teste; juntas reúnem {fmt.pt_int(n_objects)} objetos."


def below_def(floor: int) -> str:
    return (
        f"Regiões com menos de {fmt.pt_int(floor)} objetos: aparecem só como contorno, "
        "sem taxa de erro (n/a)."
    )


def max_def(floor: int) -> str:
    return (
        f"A maior taxa de uma região com ao menos {fmt.pt_int(floor)} objetos. "
        "É o máximo de uma região, não o erro geral."
    )


def diverge_note(meta_regions: int | None, meta_objects: int | None) -> str:
    """Shown only when the artifact header and the per-region aggregate disagree."""
    na = "n/a"
    shown = [na if v is None else fmt.pt_int(v) for v in (meta_regions, meta_objects)]
    return (
        f"O cabeçalho do artefato indica {shown[0]} regiões e {shown[1]} objetos; "
        "os cartões mostram a soma das regiões."
    )


def lead(n_regions: int) -> str:
    return (
        f"{fmt.pt_int(n_regions)} regiões do céu do teste, em coordenadas galácticas."
    )


def how(by_rate: bool, n_hidden: int, floor: int, hidden_on_map: bool) -> str:
    """The 'Como ler' sentence for the active map mode."""
    if not by_rate:
        return (
            "quanto mais azul, mais objetos de teste a região tem. "
            "Anel vazio não aparece neste modo."
        )
    few = f"{fmt.pt_int(n_hidden)} regiões com menos de {fmt.pt_int(floor)} objetos"
    tail = (
        f"As {few} não aparecem no mapa."
        if hidden_on_map
        else f"Anel vazio = {few}, sem taxa confiável."
    )
    return f"quanto mais azul, maior a taxa de erro da região. {tail}"


def header_html(title: str, lead_text: str, how_text: str) -> str:
    """Title, opening sentence and the 'Como ler' line in the calm section pattern."""
    how_html = f"<p class='hs-small'><b>{escape(HOW_LABEL)}</b> {escape(how_text)}</p>"
    return hs.section(title, lead_text, how_html)


def map_lead(stats: pd.DataFrame, floor: int) -> str:
    """'A taxa geral é X%. Em N das M regiões com dados confiáveis, ela passa de 10%.'"""
    overall = float(stats["errors"].sum()) / float(stats["n"].sum())
    reliable = stats[stats["n"] >= floor]
    over = int((reliable["rate"] > ERROR_HIGH).sum())
    return (
        f"A taxa geral é {fmt.pt_pct(overall)}. Em {fmt.pt_int(over)} das "
        f"{fmt.pt_int(len(reliable))} regiões com dados confiáveis, "
        f"ela passa de {fmt.pt_pct(ERROR_HIGH, 0)}."
    )


def spread_note(stats: pd.DataFrame) -> str:
    """Regions with at least one error, against the share expected if errors were spread at random."""
    chance = chance_share_of_pixels_with_error(
        stats, float(stats["errors"].sum()) / float(stats["n"].sum())
    )
    with_error = int((stats["errors"] > 0).sum())
    return (
        f"{fmt.pt_int(with_error)} regiões têm ao menos um erro "
        f"({fmt.pt_pct(with_error / len(stats), 0)}); ao acaso, "
        f"seriam ≈{fmt.pt_pct(chance, 0)}."
    )


def band_names() -> dict[str, str]:
    lo, hi = (fmt.pt_g(e) for e in BAND_EDGES_DEG)
    return {
        "A": f"A · |b| < {lo}° (plano da Via Láctea)",
        "B": f"B · {lo}° a {hi}°",
        "C": f"C · acima de {hi}°",
    }


def _note(text: str) -> str:
    return f"<p>{escape(text)}</p>"


def bands_body(bands: pd.DataFrame) -> str:
    names = band_names()
    rows = [
        [
            names[r.band],
            fmt.pt_int(r.n_pixels),
            fmt.pt_int(r.n),
            fmt.pt_int(r.errors),
            fmt.pt_pct(r.rate, 2),
            f"{fmt.pt_pct(r.ci_lo, 2)} a {fmt.pt_pct(r.ci_hi, 2)}",
        ]
        for r in bands.itertuples(index=False)
    ]
    head = ["Faixa", "Regiões", "Objetos", "Erros", "Taxa", fmt.pt_ci_label()]
    note = _note(
        "A latitude é a do centro de cada região. "
        "As linhas tracejadas no mapa marcam os limites entre as faixas."
    )
    return html_kit.table(head, rows) + note


def bands_box(bands: pd.DataFrame) -> str:
    return html_kit.box(BANDS_TITLE, bands_body(bands))


def _sign(b: float) -> str:
    return "−" if b < 0 else "+"


def position_text(l_deg: float, b_deg: float) -> str:
    """'l = X°, b = ±Y°' with integer degrees; l wraps so 360 prints as 0."""
    lon = round(l_deg) % 360
    return f"l = {lon}°, b = {_sign(b_deg)}{abs(round(b_deg))}°"


def top_title(floor: int) -> str:
    return f"Regiões com mais erro (mínimo de {fmt.pt_int(floor)} objetos)"


def top_body(top: pd.DataFrame, floor: int) -> str:
    rows = [
        [
            position_text(r.l, r.b),
            fmt.pt_int(r.n),
            fmt.pt_int(r.errors),
            fmt.pt_pct(r.rate),
        ]
        for r in top.itertuples(index=False)
    ]
    note = _note(
        f"Regiões HEALPix (nside {HEALPIX_NSIDE}) com objetos de teste. "
        f"Anel vazio = menos de {fmt.pt_int(floor)} objetos, sem taxa confiável."
    )
    return html_kit.table(["Posição (l, b)", "Objetos", "Erros", "Taxa"], rows) + note


def top_box(top: pd.DataFrame, floor: int) -> str:
    return html_kit.box(top_title(floor), top_body(top, floor))
