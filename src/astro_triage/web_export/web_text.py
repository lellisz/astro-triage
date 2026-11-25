"""Colour-neutral variant of the sky 'Como ler' sentence for the monochrome web front.

The Streamlit text (``sky_text.how``) names a colour; the web map is greyscale, so the
export words the same sentence by grey level instead.
"""

from __future__ import annotations

import pandas as pd

from astro_triage.dashboard import fmt


def how(by_rate: bool, n_hidden: int, floor: int, hidden_on_map: bool) -> str:
    """Same structure as ``sky_text.how``, with the colour reference replaced by grey level."""
    if not by_rate:
        return (
            "quanto mais claro o ponto, mais objetos de teste a região tem. "
            "Anel vazio não aparece neste modo."
        )
    few = f"{fmt.pt_int(n_hidden)} regiões com menos de {fmt.pt_int(floor)} objetos"
    tail = (
        f"As {few} não aparecem no mapa."
        if hidden_on_map
        else f"Anel vazio = {few}, sem taxa confiável."
    )
    return f"quanto mais claro o ponto, maior a taxa de erro da região. {tail}"


def missing_note(by_g: pd.DataFrame) -> str:
    """Same structure as ``missing_note_text``, without the colour key (no web chart highlights a rate)."""
    gone = by_g[by_g["lo"].isna()]
    n = int(gone["n"].sum()) if len(gone) else 0
    if not n:
        return ""
    return (
        f"Faltam {fmt.pt_int(n)} galáxias sem magnitude medida, "
        "que não entram na parte de brilho."
    )
