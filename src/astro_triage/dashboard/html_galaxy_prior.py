"""Galáxias page, 'Censo Silencioso': one verdict line and five quiet blocks, no dot field.

Pure: no Streamlit. Every number comes from ``data_galaxy_prior`` (None shows as n/a); every
text that comes from data goes through ``escape``. The sentences below carry no digits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Final

from astro_triage.dashboard import fmt
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard.data_dl import ClassCell
from astro_triage.dashboard.data_galaxy_prior import (
    NETWORKS,
    ErrorRate,
    GalaxyPriorBlock,
    PriorGalaxy,
    VerdictKey,
    power_networks,
)
from astro_triage.dashboard.data_headline import SHIPPED_PRIOR
from astro_triage.dashboard.limits_pt import translate

CROSSMATCH_ADR: Final = 4
VERDICT_TEXT: Final[Mapping[VerdictKey, str]] = {
    VerdictKey.DEPENDS_WIDE: (
        "A pureza de galáxia depende do prior assumido; "
        "a comparação com as redes tem intervalos largos."
    ),
    VerdictKey.DEPENDS: "A pureza de galáxia depende do prior assumido.",
    VerdictKey.WIDE: (
        "Sem diferença detectável entre P1 e P2; "
        "a comparação com as redes tem intervalos largos."
    ),
    VerdictKey.NO_DIFFERENCE: "Sem diferença detectável entre P1 e P2.",
    VerdictKey.UNAVAILABLE: hs.NA,
}
VARIANT_TEXT: Final[Mapping[str, str]] = {
    "as_shipped": "como entregue",
    "prior_matched": "ajustada ao prior",
}
NETWORK_NAME: Final[Mapping[str, str]] = {
    "mlp_full": "MLP",
    "ftt_full": "FT-Transformer",
}
PURITY: Final = "Pureza"
COMPLETENESS: Final = "Completude"
PARTIAL: Final = "parcial"
GALAXY_ROW: Final = "Galáxia"
OBJECTS: Final = "galáxias"
POWER_NOTE: Final = "Falta de poder na comparação com {}, não igualdade."
JOIN_AND: Final = " e "
HERO_TITLE: Final = "Galáxias no prior assumido"
HERO_SENTENCE: Final = (
    "Das galáxias apontadas, a pureza é a fração certa; das galáxias reais, "
    "a completude é a fração encontrada."
)
DEPEND_TITLE: Final = "Quanto depende do prior"
DEPEND_SENTENCE: Final = (
    "A mesma escala nas duas colunas; o prior parcial fica fora da regra do veredito."
)
DELTA_TITLE: Final = "Redes contra o LightGBM"
DELTA_SENTENCE: Final = (
    "Diferença na linha da galáxia; a faixa é a margem δ: duas vezes o desvio "
    "entre sementes do LightGBM, na validação."
)
ERRORS_TITLE: Final = "Onde o modelo erra"
ERRORS_SENTENCE: Final = "Taxa de erro entre as galáxias, com intervalo de Wilson; faixa abaixo do piso é n/a."
CROSSMATCH_TITLE: Final = "Nota sobre o cruzamento"
CROSSMATCH_ORIGIN: Final = "Origem:"
CROSSMATCH_UNMEASURED: Final = "n/a"
NOT_RECOMPUTED: Final = ("não recalculad", "not recomputed")  # translated and original
TECH_DETAILS: Final = "Tabelas por brilho e por distância"


def _t(text: object) -> str:
    return escape(str(text), quote=True)


def verdict_text(key: VerdictKey) -> str:
    return VERDICT_TEXT.get(key, hs.NA)


def section(title: str, sentence: str, body_html: str) -> str:
    """Titled block with a single sentence; ``title`` and ``sentence`` are escaped."""
    return (
        f"<section class='hs-sec'><div class='hs-pt' role='heading' "
        f"aria-level='{hs.CARD_LEVEL}'>"
        f"{_t(title)}</div><p class='hs-sen'>{_t(sentence)}</p>{body_html}</section>"
    )


def _point(iv: object) -> dict:
    if iv is None:
        return {}
    return {"point": iv.point, "lo": iv.lo, "hi": iv.hi}  # type: ignore[attr-defined]


def hero_html(block: GalaxyPriorBlock, prior: str = SHIPPED_PRIOR) -> str:
    """Purity and completeness side by side, with the assumed prior and the rule named."""
    pg = block.get(prior)
    rule = VARIANT_TEXT.get(block.variant, block.variant)
    tag = f"prior de quasar assumido: {prior} · regra: {rule}"
    pair = hs.big_pair(
        [
            {"name": PURITY, **_point(pg.purity if pg else None)},
            {"name": COMPLETENESS, **_point(pg.completeness if pg else None)},
        ]
    )
    return section(
        HERO_TITLE, HERO_SENTENCE, f"<p class='hs-small'>{_t(tag)}</p>{pair}"
    )


def _prior_label(pg: PriorGalaxy) -> str:
    return f"{pg.prior} ({PARTIAL})" if pg.partial else pg.prior


def _metric_panel(title: str, block: GalaxyPriorBlock, field: str) -> str:
    rows = []
    for pg in block.priors:
        iv = getattr(pg, field)
        args = (iv.point, iv.lo, iv.hi) if iv is not None else (None, None, None)
        rows.append(hs.interval_strip(*args, label=_prior_label(pg)))
    return (
        f"<div class='hs-panel'><div class='hs-pt' role='heading' "
        f"aria-level='{hs.CARD_LEVEL}'>{_t(title)}</div>{''.join(rows)}</div>"
    )


def dependence_html(block: GalaxyPriorBlock) -> str:
    body = (
        "<div class='hs-panels'>"
        f"{_metric_panel(PURITY, block, 'purity')}"
        f"{_metric_panel(COMPLETENESS, block, 'completeness')}</div>"
    )
    return section(DEPEND_TITLE, DEPEND_SENTENCE, body)


def _delta_panel(net: str, cell: ClassCell | None) -> dict:
    name = NETWORK_NAME.get(net, net)
    if cell is None or cell.d is None:
        return {"title": name, "delta": None, "rows": []}
    detail = (
        fmt.pt_interval(cell.d, cell.lo, cell.hi, "", True)
        if cell.lo is not None and cell.hi is not None
        else fmt.pt_num(cell.d, 3, True)
    )
    row = {
        "name": GALAXY_ROW,
        "d": cell.d,
        "lo": cell.lo,
        "hi": cell.hi,
        "detail": detail,
    }
    return {"title": name, "delta": cell.delta, "rows": [row]}


def delta_html(cells: Mapping[str, ClassCell | None]) -> str:
    """Two ±delta rulers (tick for the point) and, only when the data needs it, the power note."""
    panels = "".join(
        hs.delta_ruler(_delta_panel(net, cells.get(net)), tick=True) for net in NETWORKS
    )
    body = f"<div class='hs-panels'>{panels}</div>"
    names = [NETWORK_NAME.get(n, n) for n in power_networks(cells)]
    if names:
        body += f"<p class='hs-sen'>{_t(POWER_NOTE.format(JOIN_AND.join(names)))}</p>"
    return section(DELTA_TITLE, DELTA_SENTENCE, body)


def errors_html(rows: Sequence[tuple[str, ErrorRate | None]]) -> str:
    body = "".join(
        hs.interval_strip(
            *((r.rate, r.ci_lo, r.ci_hi) if r else (None, None, None)),
            label=label,
            extra=f"{fmt.pt_int(r.n)} {OBJECTS}" if r else "",
        )
        for label, r in rows
    )
    return section(ERRORS_TITLE, ERRORS_SENTENCE, body)


def crossmatch_html(note: str | None, release: str) -> str:
    """Small note with the origin of the rates; the text is the artifact's own.

    The "n/a" line for ``release`` appears only when the text itself says the rates were
    not recomputed for it; otherwise the translated text stands alone.
    """
    text = translate(note)[0] if note else hs.NA
    origin = f"{CROSSMATCH_ORIGIN} ADR {CROSSMATCH_ADR:04d}"
    unmeasured = any(m in text for m in NOT_RECOMPUTED)
    other = (
        f"<p class='hs-small'>{_t(f'{release}: {CROSSMATCH_UNMEASURED}')}</p>"
        if unmeasured
        else ""
    )
    return (
        f"<section class='hs-sec'><div class='hs-pt' role='heading' "
        f"aria-level='{hs.CARD_LEVEL}'>{_t(CROSSMATCH_TITLE)}</div>"
        f"<p class='hs-small'>{_t(origin)}. {_t(text)}</p>{other}</section>"
    )


def page_html(
    release: str,
    verdict: VerdictKey,
    block: GalaxyPriorBlock,
    cells: Mapping[str, ClassCell | None],
    error_rows: Sequence[tuple[str, ErrorRate | None]],
    note: str | None,
) -> str:
    body = (
        hs.seal(release)
        + hs.verdict_oneline(verdict_text(verdict))
        + hero_html(block)
        + dependence_html(block)
        + delta_html(cells)
        + errors_html(error_rows)
        + crossmatch_html(note, release)
    )
    return hs.lang_root(body)
